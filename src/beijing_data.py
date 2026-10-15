"""Causal, calendar-hour features for the UCI Beijing station series."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

START = pd.Timestamp("2013-03-01")
TRAIN_END = pd.Timestamp("2016-01-01")
VAL_END = pd.Timestamp("2016-07-01")
END = pd.Timestamp("2017-03-01")
MEASURES = ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM")
NONNEGATIVE = ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "RAIN", "WSPM")
LAGS = (1, 3, 24, 168)


def load_beijing(root: Path, through: pd.Timestamp = END) -> pd.DataFrame:
    files = sorted(root.glob("PRSA_Data_20130301-20170228/PRSA_Data_*_20130301-20170228.csv"))
    if len(files) != 12:
        raise FileNotFoundError(f"Expected 12 official station CSVs in {root}; found {len(files)}")
    parts = [pd.read_csv(path, na_values=["NA"]) for path in files]
    raw = pd.concat(parts, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw[["year", "month", "day", "hour"]])
    raw = raw.loc[raw.timestamp < through, ["timestamp", "station", *MEASURES, "wd"]].copy()
    if raw.duplicated(["timestamp", "station"]).any():
        raise ValueError("Duplicate station-hour records")
    if raw.timestamp.min() != START or raw.timestamp.max() >= END or raw.station.nunique() != 12:
        raise ValueError("Unexpected Beijing coverage")
    return raw.sort_values(["station", "timestamp"], kind="stable").reset_index(drop=True)


def make_beijing_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Reindex each station to real hours, so all shifts are exact-hour shifts."""
    raw = raw.copy()
    for col in NONNEGATIVE:
        raw.loc[raw[col] < 0, col] = np.nan
    stations = sorted(raw.station.unique())
    hours = pd.date_range(raw.timestamp.min(), raw.timestamp.max(), freq="h")
    index = pd.MultiIndex.from_product([stations, hours], names=["station", "timestamp"])
    full = raw.set_index(["station", "timestamp"]).reindex(index)
    features = ["station", "hour", "dow", "month", "day_of_year"]
    t = full.index.get_level_values("timestamp")
    full["hour"] = t.hour
    full["dow"] = t.dayofweek
    full["month"] = t.month
    full["day_of_year"] = t.dayofyear
    group = full.groupby(level="station", sort=False)
    for lag in LAGS:
        col = f"pm_lag{lag}"
        full[col] = group["PM2.5"].shift(lag)
        full[f"{col}_missing"] = full[col].isna().astype("int8")
        features.extend((col, f"{col}_missing"))
    # shift(1) occurs before rolling; missing target observations are skipped,
    # never replaced with zero or interpolated from future hours.
    prior = group["PM2.5"].shift(1)
    for window in (24, 168):
        by_station = prior.groupby(level="station", sort=False)
        for stat in ("mean", "std"):
            col = f"pm_roll{window}_{stat}"
            full[col] = by_station.transform(lambda x: getattr(x.rolling(window, min_periods=1), stat)())
            features.append(col)
    for col in MEASURES[1:]:
        name = f"{col}_lag1"
        full[name] = group[col].shift(1)
        full[f"{name}_missing"] = full[name].isna().astype("int8")
        features.extend((name, f"{name}_missing"))
    full["wd_lag1"] = group["wd"].shift(1)
    full["wd_lag1_missing"] = full.wd_lag1.isna().astype("int8")
    features.extend(("wd_lag1", "wd_lag1_missing"))
    full = full.reset_index()
    full["station"] = pd.Categorical(full.station, categories=stations)
    full["wd_lag1"] = pd.Categorical(full.wd_lag1)
    return full, features


def split_beijing(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t = frame.timestamp
    result = {
        "train": frame.loc[t < TRAIN_END].copy(),
        "validation": frame.loc[(t >= TRAIN_END) & (t < VAL_END)].copy(),
        "test": frame.loc[(t >= VAL_END) & (t < END)].copy(),
    }
    if any(x.empty for x in result.values()):
        raise ValueError("Empty chronological split")
    return result
