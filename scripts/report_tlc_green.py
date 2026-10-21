"""Audit TLC forecasts, summarize zones, and estimate paired uncertainty."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.collect_tlc_green import PROCESSED, RAW, ROOT, sha256
from scripts.run_tlc_green_forecast import OUT, PANEL, load_frame, splits


def main() -> None:
    lock = json.loads((OUT / "selection_lock.json").read_text(encoding="utf-8"))
    quality = json.loads((PROCESSED / "quality_audit.json").read_text(encoding="utf-8"))
    if sha256(PANEL) != lock["sha256"]["panel"] or sha256(RAW / "manifest.json") != quality["manifest_sha256"]:
        raise ValueError("Locked TLC data changed")
    pred = pd.read_csv(OUT / "test_predictions.csv.gz", parse_dates=["date"])
    frame = load_frame()
    test = splits(frame)["test"]
    if not np.array_equal(pred.date.to_numpy(), test.date.to_numpy()) or not np.array_equal(
            pred.zone.to_numpy(int), test.zone.to_numpy(int)) or not np.array_equal(
                pred.actual.to_numpy(int), test.pickups.to_numpy(int)):
        raise ValueError("Test predictions do not match collected panel")
    y = pred.actual.to_numpy(float)
    z, base, method = np.log1p(y), np.log1p(pred.LightGBM), np.log1p(pred.selected)
    e, d = z-base, method-base
    naive = test.lag7.to_numpy(float)
    if not np.isfinite(naive).all():
        raise ValueError("Seasonal baseline lacks test lag7")
    naive_metrics = {"rmsle": float(np.sqrt(np.mean((z-np.log1p(naive))**2))),
                     "mae": float(np.mean(np.abs(y-naive)))}
    risk = {"correction_energy": float(np.mean(d*d)),
            "twice_alignment": float(2*np.mean(e*d)),
            "delta_mse_log": float(np.mean((e-d)**2)-np.mean(e*e))}
    block = ((pred.date-pred.date.min()).dt.days//28).to_numpy(int)
    n_blocks = int(block.max())+1
    count = np.bincount(block, minlength=n_blocks)
    b_sq = np.bincount(block, weights=e*e, minlength=n_blocks)
    m_sq = np.bincount(block, weights=(e-d)**2, minlength=n_blocks)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, n_blocks, size=(2000,n_blocks))
    denominator = count[draws].sum(axis=1)
    delta = np.sqrt(m_sq[draws].sum(axis=1)/denominator)-np.sqrt(
        b_sq[draws].sum(axis=1)/denominator)
    interval = {"delta_rmsle": float(np.sqrt(np.mean((e-d)**2))-np.sqrt(np.mean(e*e))),
                "ci95_low": float(np.quantile(delta,.025)),
                "ci95_high": float(np.quantile(delta,.975)),
                "blocks": n_blocks, "block_days": 28, "bootstrap_reps": 2000,
                "improved_blocks": int(((m_sq-b_sq)<0).sum())}
    lookup = pd.read_csv(ROOT / "provenance/tlc_taxi_zone_lookup.csv")
    zones = []
    for zone, group in pred.groupby("zone", sort=True):
        a = np.log1p(group.actual.to_numpy(float))
        b = np.log1p(group.LightGBM.to_numpy(float))
        m = np.log1p(group.selected.to_numpy(float))
        meta = lookup.loc[lookup.LocationID == zone].iloc[0]
        zones.append({"zone_id": int(zone), "borough": meta.Borough, "zone": meta.Zone,
                      "test_pickups": int(group.actual.sum()),
                      "base_rmsle": float(np.sqrt(np.mean((a-b)**2))),
                      "selected_rmsle": float(np.sqrt(np.mean((a-m)**2)))})
    zone_table = pd.DataFrame(zones)
    zone_table.to_csv(OUT / "zone_test_metrics.csv", index=False)
    annual = frame.groupby(frame.date.dt.year).pickups.agg(["sum","mean"]).reset_index()
    annual.to_csv(OUT / "annual_selected_zone_demand.csv", index=False)
    summary = {"naive": naive_metrics, "risk_identity": risk,
               "paired_interval": interval,
               "zones_improved": int((zone_table.selected_rmsle < zone_table.base_rmsle).sum()),
               "annual_selected_zone_pickups": {str(r.date): int(r.sum) for r in annual.itertuples(index=False)}}
    (OUT / "report_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(11,7))
    panel = pd.read_csv(PANEL, parse_dates=["date"])
    for zone in zone_table.sort_values("test_pickups", ascending=False).head(3).zone_id:
        curve = panel.loc[panel.zone == zone].set_index("date").pickups.rolling(28).mean()
        axes[0].plot(curve.index, curve, label=f"Zone {zone}")
    axes[0].set(title="TLC green taxi pickups: trailing 28-day means", ylabel="Trips per day")
    axes[0].legend()
    axes[1].scatter(zone_table.base_rmsle, zone_table.selected_rmsle)
    bound = max(zone_table.base_rmsle.max(),zone_table.selected_rmsle.max())
    axes[1].plot([0,bound],[0,bound],color="black",linewidth=1)
    axes[1].set(xlabel="LightGBM RMSLE", ylabel="Selected RMSLE", title="2025 test by pickup zone")
    fig.tight_layout()
    fig.savefig(OUT / "demand_and_test.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
