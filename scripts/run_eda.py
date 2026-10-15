from __future__ import annotations
import argparse, logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.data import load_favorita

def main():
    p=argparse.ArgumentParser(description="Create EDA tables and plots from local Favorita data")
    p.add_argument("--raw-dir",default="data/raw"); p.add_argument("--out-dir",default="results/eda"); a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    df,aux=load_favorita(a.raw_dir); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    summary=pd.DataFrame([{"rows":len(df),"start":df.date.min(),"end":df.date.max(),"stores":df.store_nbr.nunique(),"families":df.family.nunique(),"series":df.groupby(["store_nbr","family"]).ngroups,"missing_sales":df.sales.isna().sum(),"zero_sales":(df.sales==0).sum()}])
    summary.to_csv(out/"summary.csv",index=False)
    daily=df.assign(dow=df.date.dt.dayofweek,month=df.date.dt.month).groupby("dow").sales.agg(["mean","median","count"]); daily.to_csv(out/"sales_by_weekday.csv")
    df.assign(month=df.date.dt.month).groupby("month").sales.agg(["mean","median","count"]).to_csv(out/"sales_by_month.csv")
    series=df.groupby(["store_nbr","family"]).sales.agg(mean="mean",std="std",zero_rate=lambda x:(x==0).mean()).reset_index(); series.to_csv(out/"series_profile.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,4)); ax.hist(df.sales.clip(upper=df.sales.quantile(.99)),bins=60); ax.set(title="Sales distribution (upper 1% clipped for display)",xlabel="sales",ylabel="rows"); fig.tight_layout(); fig.savefig(out/"sales_distribution.png",dpi=140); plt.close(fig)
    daily_all=df.groupby("date").sales.sum(); fig,ax=plt.subplots(figsize=(11,4)); daily_all.plot(ax=ax); ax.set(title="Total sales by date",ylabel="sales",xlabel="date"); fig.tight_layout(); fig.savefig(out/"daily_sales.png",dpi=140); plt.close(fig)
    if "onpromotion" in df:
        promo=df.groupby(df.onpromotion.gt(0)).sales.agg(["mean","median","count"]); promo.to_csv(out/"promotion_summary.csv")
    if "holidays_events" in aux:
        hol=aux["holidays_events"].copy(); hol["date"]=pd.to_datetime(hol.date)
        hol=hol.loc[~hol.transferred.fillna(False).astype(bool)] if "transferred" in hol else hol
        day=df.assign(is_holiday=df.date.isin(hol.date)).groupby("is_holiday").sales.agg(["mean","median","count"]); day.to_csv(out/"holiday_summary.csv")
    logging.info("Wrote EDA artifacts to %s",out)
if __name__=="__main__": main()
