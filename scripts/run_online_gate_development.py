"""Validation-only screen of causal online residual gates; never reads test."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.online_residual import GateConfig, run_gate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/usgs_ablation_exploratory/validation_predictions.csv.gz"
OUT = ROOT / "results/online_gate_development"


def main() -> None:
    frame = pd.read_csv(SOURCE, usecols=["date", "store_nbr", "family", "actual", "LightGBM"],
                        parse_dates=["date"])
    actual_log = np.log1p(frame.actual.to_numpy(float))
    base_sales = frame.LightGBM.to_numpy(float)
    base_log = np.log1p(base_sales)
    jan_feb = (frame.date < "2016-03-01").to_numpy()
    variants = [GateConfig(alpha=a, method="ogd", eta=eta)
                for a in (.02, .05, .1, .2, .5) for eta in (.03, .1, .3, 1.0)]
    variants += [GateConfig(alpha=a, method="ridge", ridge=lam, discount=discount)
                 for a in (.02, .05, .1, .2, .5)
                 for lam in (.01, .1, 1., 10.) for discount in (1.0, .98)]
    rows = [{"method": "LightGBM", "alpha": np.nan, "eta": np.nan,
             "ridge": np.nan, "discount": np.nan,
             "jan_feb_rmsle": float(np.sqrt(np.mean((actual_log[jan_feb] -
                                                    base_log[jan_feb]) ** 2))),
             "march_rmsle": float(np.sqrt(np.mean((actual_log[~jan_feb] -
                                                  base_log[~jan_feb]) ** 2)))}]
    for cfg in variants:
        pred = run_gate(frame, base_sales, cfg)["prediction_log"]
        rows.append({"method": cfg.method, "alpha": cfg.alpha, "eta": cfg.eta,
                     "ridge": cfg.ridge, "discount": cfg.discount,
                     "jan_feb_rmsle": float(np.sqrt(np.mean((actual_log[jan_feb] -
                                                            pred[jan_feb]) ** 2))),
                     "march_rmsle": float(np.sqrt(np.mean((actual_log[~jan_feb] -
                                                          pred[~jan_feb]) ** 2)))})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "grid.csv", index=False)
    print(pd.DataFrame(rows).nsmallest(6, "jan_feb_rmsle").to_string(index=False))


if __name__ == "__main__":
    main()
