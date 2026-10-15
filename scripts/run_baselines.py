from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import load_favorita, chronological_split
from src.features import make_features, feature_columns
from src.train import train_lightgbm
from src.baselines import EMAAdapter, KalmanAdapter
from src.levy_process import LevyOU
from src.jump_filter import JumpAwareFilter
from src.evaluate import metrics, online_adapter_backtest

def subset_series(frame, max_series):
    if not max_series: return frame
    ids=frame.loc[frame.date <= "2015-12-31",["store_nbr","family"]].drop_duplicates().sort_values(["store_nbr","family"]).head(max_series)
    return frame.merge(ids,on=["store_nbr","family"],how="inner",validate="many_to_one")

def main():
    p=argparse.ArgumentParser(description="Fit fixed LightGBM and run matched one-step online baselines")
    p.add_argument("--raw-dir",default="data/raw"); p.add_argument("--config",default="configs/default.json"); p.add_argument("--max-series",type=int)
    p.add_argument("--seed",type=int); p.add_argument("--out-dir",default="results/baselines"); a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8")); seed=a.seed if a.seed is not None else cfg["seed"]
    raw,aux=load_favorita(a.raw_dir); raw=subset_series(raw,a.max_series or cfg.get("max_series"))
    parts=chronological_split(raw,cfg["train_end"],cfg["validation_end"]); feat=make_features(raw,aux)
    tr=feat.loc[feat.date<=cfg["train_end"]].dropna(subset=["sales"]); va=feat.loc[(feat.date>cfg["train_end"])&(feat.date<=cfg["validation_end"])].dropna(subset=["sales"]); te=feat.loc[feat.date>cfg["validation_end"]].dropna(subset=["sales"])
    cols=feature_columns(feat); tr=tr.copy(); va=va.copy(); te=te.copy()
    # Freeze categorical vocabularies from train; unseen categories map to missing.
    for col in cols:
        if isinstance(feat[col].dtype,pd.CategoricalDtype):
            categories=tr[col].dropna().unique()
            for split in (tr,va,te): split[col]=split[col].cat.set_categories(categories)
    model=train_lightgbm(tr,va,cols,cfg.get("model"),seed,cfg.get("num_threads",4))
    va["base_log"]=model.predict(va[cols]); te["base_log"]=model.predict(te[cols])
    # Select online correction settings on validation only; freeze before test.
    tune_ids=va[["store_nbr","family"]].drop_duplicates().sort_values(["store_nbr","family"]).head(100)
    va_tune=va.merge(tune_ids,on=["store_nbr","family"],how="inner",validate="many_to_one")
    best=None
    for rho in (.9,.98):
        for q in (0.0,.005):
            for R in (.1,.3):
                for jump_p in (0.0,.01):
                    for q_jump in (.1,.5):
                        proc=LevyOU(rho=rho,q=q,p=jump_p,q_jump=q_jump)
                        pred=online_adapter_backtest(va_tune,va_tune.base_log,JumpAwareFilter(proc,R),"candidate")
                        score=metrics(pred.actual,pred.prediction)["rmse_log"]
                        if best is None or score<best[0]: best=(score,proc,R)
    _,proc,R=best
    outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    preds=[]
    # Every adapter receives the same fixed offline base prediction and only online residual updates.
    for name,adapter in [("lgbm_ema",EMAAdapter(.1)),("lgbm_kalman",KalmanAdapter(LevyOU(proc.mu,proc.rho,proc.q,0,0),R)),("lgbm_jump",JumpAwareFilter(proc,R))]:
        pr=online_adapter_backtest(te,te.base_log,adapter,name); preds.append(pr)
    base=np.maximum(0,np.expm1(te.base_log.to_numpy()))
    base_df=te[["date","store_nbr","family","sales"]].copy(); base_df["prediction"]=base; base_df["model"]="lightgbm"; base_df["latent_correction"]=0.; base_df["jump_probability"]=np.nan; base_df["forecast_error"]=base_df.sales-base_df.prediction; preds.append(base_df.rename(columns={"sales":"actual"}))
    # Seasonal naive fallback is the global zero prediction where lag-7 is unavailable.
    naive=te.copy(); naive["prediction"]=np.maximum(0,naive.sales_lag7.fillna(0)); naive["model"]="seasonal_naive"; naive["latent_correction"]=np.nan; naive["jump_probability"]=np.nan; naive["forecast_error"]=naive.sales-naive.prediction; preds.append(naive.rename(columns={"sales":"actual"}))
    allp=pd.concat(preds,ignore_index=True); allp.to_csv(outdir/"predictions.csv.gz",index=False,compression="gzip")
    rows=[]
    for name,grp in allp.groupby("model"):
        res=metrics(grp.actual,grp.prediction,list(zip(grp.store_nbr,grp.family))); rows.append({"model":name,**res})
    pd.DataFrame(rows).to_csv(outdir/"metrics.csv",index=False)
    # Compact test diagnostics. Date aggregates avoid plotting millions of individual points.
    daily=allp.assign(log_sq=(np.log1p(allp.actual)-np.log1p(allp.prediction.clip(lower=0)))**2).groupby(["date","model"]).log_sq.mean().pow(.5).rename("daily_rmse_log").reset_index()
    daily["rolling_28d_rmse_log"]=daily.sort_values("date").groupby("model").daily_rmse_log.transform(lambda x:x.rolling(28,min_periods=5).mean())
    daily.to_csv(outdir/"daily_rolling_error.csv",index=False)
    fig,ax=plt.subplots(figsize=(11,4))
    for name,g in daily.groupby("model"): ax.plot(g.date,g.rolling_28d_rmse_log,label=name,lw=1)
    ax.set(title="28-day rolling log-scale error",ylabel="RMSE(log1p sales)",xlabel="date"); ax.legend(ncol=3,fontsize=8); fig.tight_layout(); fig.savefig(outdir/"rolling_error.png",dpi=140); plt.close(fig)
    jump_daily=daily.loc[daily.model=="lgbm_jump"].sort_values("date")
    peaks=[]
    for row in jump_daily.sort_values("daily_rmse_log",ascending=False).itertuples():
        if all(abs((row.date-d).days)>=28 for d in peaks): peaks.append(row.date)
        if len(peaks)==3: break
    fig,axes=plt.subplots(len(peaks),1,figsize=(10,max(3,2.5*len(peaks))),squeeze=False)
    for ax,peak in zip(axes[:,0],peaks):
        window=daily[(daily.date>=peak-pd.Timedelta(days=14))&(daily.date<=peak+pd.Timedelta(days=14))]
        for name,g in window.groupby("model"): ax.plot(g.date,g.daily_rmse_log,label=name,lw=1)
        ax.axvline(peak,color="black",lw=.7,alpha=.5); ax.set_title(f"Daily error near {peak.date()}"); ax.set_ylabel("log RMSE")
    if peaks: axes[0,0].legend(ncol=3,fontsize=8); fig.tight_layout(); fig.savefig(outdir/"error_around_peaks.png",dpi=140)
    plt.close(fig)
    chosen=allp[allp.model.isin(["lightgbm","lgbm_jump"])]
    sample=pd.concat([g.sample(n=min(5000,len(g)),random_state=seed) for _,g in chosen.groupby("model")],ignore_index=True) if len(chosen) else chosen
    fig,ax=plt.subplots(figsize=(5,5))
    for name,g in sample.groupby("model"): ax.scatter(g.actual,g.prediction,s=5,alpha=.2,label=name)
    ax.set(xscale="symlog",yscale="symlog",title="Forecast versus actual (sample)",xlabel="actual sales",ylabel="predicted sales"); ax.legend(); fig.tight_layout(); fig.savefig(outdir/"forecast_vs_actual.png",dpi=140); plt.close(fig)
    jump=allp.loc[allp.model=="lgbm_jump"].groupby("date").agg(latent_correction=("latent_correction","mean"),jump_probability=("jump_probability","mean"))
    jump.to_csv(outdir/"jump_diagnostics.csv")
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    jump.latent_correction.plot(ax=axes[0]); axes[0].set_ylabel("mean correction")
    jump.jump_probability.plot(ax=axes[1]); axes[1].set_ylabel("mean jump probability"); axes[1].set_xlabel("date")
    fig.suptitle("Jump filter diagnostics"); fig.tight_layout(); fig.savefig(outdir/"jump_diagnostics.png",dpi=140); plt.close(fig)
    (outdir/"run_config.json").write_text(json.dumps({**cfg,"seed":seed,"selected_filter":{"rho":proc.rho,"q":proc.q,"R":R,"p":proc.p,"q_jump":proc.q_jump},"max_series":a.max_series},indent=2),encoding="utf-8")
    logging.info("Saved test metrics, predictions and selected validation parameters under %s",outdir)
if __name__=="__main__": main()
