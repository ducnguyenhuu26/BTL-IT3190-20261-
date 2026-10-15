"""Data loading, validation and chronological split helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class DateSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_favorita(raw_dir: str | Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    raw = Path(raw_dir)
    path = raw / "train.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required labeled data not found: {path}. Place Kaggle train.csv in data/raw/.")
    sales = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "store_nbr", "family", "sales"}
    missing = required.difference(sales.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    if sales[["date", "store_nbr", "family"]].isna().any().any():
        raise ValueError("date/store_nbr/family cannot be null")
    if sales.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError("Duplicate date/store_nbr/family rows found")
    if (sales.sales.dropna() < 0).any():
        raise ValueError("Negative sales found; log1p target requires nonnegative sales")
    aux = {}
    for name in ("stores", "holidays_events", "oil", "transactions"):
        candidate = raw / f"{name}.csv"
        if candidate.exists():
            aux[name] = pd.read_csv(candidate, parse_dates=["date"] if name != "stores" else None)
    return sales, aux


def chronological_split(df: pd.DataFrame, train_end="2015-12-31", validation_end="2016-03-31") -> DateSplits:
    dates = pd.to_datetime(df.date)
    train_end, validation_end = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    if train_end >= validation_end:
        raise ValueError("train_end must be earlier than validation_end")
    result = DateSplits(df.loc[dates <= train_end].copy(),
                        df.loc[(dates > train_end) & (dates <= validation_end)].copy(),
                        df.loc[dates > validation_end].copy())
    if any(part.empty for part in (result.train, result.validation, result.test)):
        raise ValueError("A chronological split is empty; check dataset date range and configured boundaries")
    return result
