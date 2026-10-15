"""Leakage-safe, one-step-ahead features. Lag values are strictly shifted."""
from __future__ import annotations
import numpy as np
import pandas as pd

KEYS = ["store_nbr", "family"]
TARGET = ["date", *KEYS, "sales"]


def make_features(sales: pd.DataFrame, aux: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    df = sales.copy()
    df["date"] = pd.to_datetime(df.date)
    df = df.sort_values([*KEYS, "date"], kind="mergesort").reset_index(drop=True)
    for col in ("sales_lag1", "sales_lag7", "sales_roll_mean_7", "sales_roll_std_7", "sales_roll_mean_28", "sales_roll_std_28"):
        df[col] = np.nan
    one_day, seven_days = pd.Timedelta(days=1), pd.Timedelta(days=7)
    for _, group in df.groupby(KEYS, sort=False, observed=True):
        dates = pd.DatetimeIndex(group.date)
        sales = pd.Series(group.sales.to_numpy(dtype=float), index=dates)
        df.loc[group.index, "sales_lag1"] = sales.reindex(dates - one_day).to_numpy()
        df.loc[group.index, "sales_lag7"] = sales.reindex(dates - seven_days).to_numpy()
        for window in (7, 28):
            rolling = sales.rolling(f"{window}D", closed="left", min_periods=1)
            df.loc[group.index, f"sales_roll_mean_{window}"] = rolling.mean().to_numpy()
            df.loc[group.index, f"sales_roll_std_{window}"] = sales.rolling(f"{window}D", closed="left", min_periods=2).std().to_numpy()
    d = df.date
    df["day_of_week"] = d.dt.dayofweek.astype("int8")
    df["day_of_month"] = d.dt.day.astype("int8")
    df["month"] = d.dt.month.astype("int8")
    df["day_of_year"] = d.dt.dayofyear.astype("int16")
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype("int8")
    if "onpromotion" not in df:
        df["onpromotion"] = 0.0
    df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0)
    if aux:
        stores = aux.get("stores")
        if stores is not None and "store_nbr" in stores:
            df = df.merge(stores, on="store_nbr", how="left", validate="many_to_one", suffixes=("", "_store"))
        oil = aux.get("oil")
        if oil is not None and {"date", "dcoilwtico"}.issubset(oil.columns):
            oil = oil[["date", "dcoilwtico"]].sort_values("date").copy()
            oil["dcoilwtico"] = pd.to_numeric(oil.dcoilwtico, errors="coerce").ffill()
            oil = oil.dropna(subset=["dcoilwtico"])
            # The backward as-of join excludes same-day values and carries the last published price.
            df = pd.merge_asof(df.sort_values("date"), oil, on="date", direction="backward", allow_exact_matches=False)
            df = df.rename(columns={"dcoilwtico": "oil_price_lag1"})
        holidays = aux.get("holidays_events")
        if holidays is not None and "date" in holidays:
            h = holidays.copy()
            h["date"] = pd.to_datetime(h.date)
            if "transferred" in h:
                h = h.loc[~h["transferred"].fillna(False).astype(bool)]
            h = h.groupby("date").size().rename("holiday_count").reset_index()
            df = df.merge(h, on="date", how="left", validate="many_to_one")
            df["holiday_count"] = df.holiday_count.fillna(0).astype("int16")
    # Categorical identifiers remain categorical for LightGBM.
    df["family"] = df.family.astype("category")
    for c in ("type", "cluster", "city", "state"):
        if c in df:
            df[c] = df[c].astype("category")
    return df


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = set(TARGET) | {"id", "transactions", "dcoilwtico"}
    return [c for c in frame.columns if c not in excluded]
