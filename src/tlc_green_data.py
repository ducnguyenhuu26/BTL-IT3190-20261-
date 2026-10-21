"""Causal daily-zone features for self-collected NYC TLC pickup counts."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "zone", "pickups"}
    if not required.issubset(panel):
        raise ValueError(f"Missing required fields: {required - set(panel)}")
    if panel.duplicated(["date", "zone"]).any() or (panel.pickups < 0).any():
        raise ValueError("Duplicate zone-day or negative pickup count")
    frame = panel.sort_values(["zone", "date"]).copy()
    if not (frame.groupby("zone").date.diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Each zone must have a complete daily calendar")
    frame["dow"] = frame.date.dt.dayofweek.astype(int)
    frame["month"] = frame.date.dt.month.astype(int)
    frame["day_of_year"] = frame.date.dt.dayofyear.astype(int)
    group = frame.groupby("zone", observed=True).pickups
    for lag in (1, 7, 28):
        frame[f"lag{lag}"] = group.shift(lag)
    for window, minimum in ((7, 3), (28, 7)):
        frame[f"roll{window}"] = group.transform(
            lambda series: series.shift(1).rolling(window, min_periods=minimum).mean())
    if not np.isfinite(frame.pickups.to_numpy(float)).all():
        raise ValueError("Missing pickup target")
    return frame
