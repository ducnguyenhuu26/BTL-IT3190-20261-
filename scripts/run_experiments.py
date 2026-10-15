from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.levy_process import LevyOU
from src.jump_filter import JumpAwareFilter
from src.evaluate import metrics

def run_case(name, process, seed, n=1200, out=None):
    rng=np.random.default_rng(seed); theta,jumps=process.sample(n,seed)
    if name=="stationary": theta=np.zeros(n); jumps[:]=False
    elif name=="isolated_outliers":
        theta=np.zeros(n); jumps[:]=False
        positions=np.unique(np.clip([n//4,n//2,3*n//4],0,max(n-1,0))) if n else []
        jumps[positions]=True
    noise=rng.normal(0,.2,n); obs=theta+noise
    if name=="isolated_outliers": obs[jumps]=np.resize(np.array([2.,-2.,2.]),int(jumps.sum()))
    filt=JumpAwareFilter(process,observation_variance=.04,prior_variance=1)
    means=[]; probs=[]
    for i,y in enumerate(obs):
        prior=filt.prior("series")[0]; means.append(prior); state=filt.update("series",y); probs.append(state["jump_probability"])
    burn=30; detected=np.array(probs)>0.5; truth=jumps
    matched=set(); delays=[]
    for event in np.flatnonzero(truth & (np.arange(n)>=burn)):
        candidates=np.flatnonzero(detected[event:min(event+15,n)])+event
        candidates=[int(c) for c in candidates if c not in matched]
        if candidates:
            matched.add(candidates[0]); delays.append(candidates[0]-int(event))
    tp=len(delays); fp=int(sum(bool(x) for i,x in enumerate(detected) if i>=burn and i not in matched)); fn=int((truth[burn:]).sum())-tp
    result={"case":name,"tracking_rmse":float(np.sqrt(np.mean((np.array(means)[burn:]-theta[burn:])**2))),"jump_precision":tp/max(tp+fp,1),"jump_recall":tp/max(tp+fn,1),"false_alarms":fp,"mean_detection_delay":float(np.mean(delays)) if delays else np.nan,"n":n}
    if out:
        pd.DataFrame({"step":np.arange(n),"theta_true":theta,"observation":obs,"jump_true":jumps,"prior_mean":means,"jump_probability":probs}).to_csv(out/f"synthetic_{name}.csv",index=False)
    return result

def main():
    p=argparse.ArgumentParser(description="Controlled synthetic filter experiment")
    p.add_argument("--seed",type=int,default=42); p.add_argument("--steps",type=int,default=1200); p.add_argument("--out-dir",default="results/synthetic")
    a=p.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    cases=[("stationary",LevyOU(rho=.9,q=0,p=0,q_jump=0)),("gaussian_drift",LevyOU(rho=.98,q=.01,p=0,q_jump=0)),("jump_only",LevyOU(rho=.98,q=0,p=.02,q_jump=.8)),("mixed",LevyOU(rho=.97,q=.005,p=.01,q_jump=.4)),("isolated_outliers",LevyOU(rho=.9,q=0,p=0,q_jump=0))]
    rows=[run_case(name,proc,a.seed+i,a.steps,out) for i,(name,proc) in enumerate(cases)]
    pd.DataFrame(rows).to_csv(out/"synthetic_metrics.csv",index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
