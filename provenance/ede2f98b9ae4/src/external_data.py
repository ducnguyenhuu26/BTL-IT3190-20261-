"""Normalize a retrospective USGS catalog and build strictly prior-day features."""
from __future__ import annotations

from datetime import timedelta, timezone
import numpy as np
import pandas as pd

ECUADOR_TZ = timezone(timedelta(hours=-5))
EVENT_COLUMNS = ["event_id", "event_time_utc", "updated_at_utc", "local_date", "magnitude",
                 "magnitude_type", "longitude", "latitude", "depth_km", "place", "status", "source_url"]
MODEL_FEATURES = ["quake_count_lag1", "strong_quake_count_lag1", "quake_max_magnitude_lag1",
                  "quake_count_prev7d"]


def join_quake_features(frame: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Join only the explicitly causal feature allowlist, retaining row order."""
    required = ["date", *MODEL_FEATURES]
    if not set(required).issubset(calendar.columns) or calendar.date.isna().any():
        raise ValueError("External calendar lacks required dates/features")
    if calendar.date.duplicated().any():
        raise ValueError("External calendar must have exactly one row per date")
    if set(MODEL_FEATURES).intersection(frame.columns):
        raise ValueError("External features already present in frame")
    result = frame.merge(calendar[required], on="date", how="left", sort=False,
                         validate="many_to_one", indicator=True)
    if not result._merge.eq("both").all() or len(result) != len(frame):
        raise ValueError("External calendar does not cover every sales row")
    count_features = [name for name in MODEL_FEATURES if "magnitude" not in name]
    values = result[count_features].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("External count features must be finite and nonnegative")
    return result.drop(columns="_merge")


def normalize_catalog(payloads: list[dict], config: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    for payload in payloads:
        if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
            raise ValueError("USGS response is not a GeoJSON FeatureCollection")
        features = payload["features"]
        count = payload.get("metadata", {}).get("count")
        if (count is not None and count != len(features)) or len(features) >= 20000:
            raise ValueError("USGS response count mismatch or possible truncation; narrow the interval")
        for feature in features:
            p, geo = feature.get("properties", {}), feature.get("geometry", {})
            coordinates = geo.get("coordinates", [])
            if not feature.get("id") or p.get("type") != "earthquake" or geo.get("type") != "Point" or len(coordinates) != 3:
                raise ValueError("Malformed USGS event identity/type/coordinates")
            if any(p.get(name) is None for name in ("time", "updated", "mag")):
                raise ValueError("Missing mandatory USGS event time/update/magnitude")
            lon, lat, depth = map(float, coordinates)
            magnitude = float(p["mag"])
            box = config["bbox"]
            if not np.isfinite([lon, lat, depth, magnitude]).all() or not (
                    box["minlongitude"] <= lon <= box["maxlongitude"] and
                    box["minlatitude"] <= lat <= box["maxlatitude"] and magnitude >= config["min_magnitude"]):
                raise ValueError("Event outside the requested coordinate/magnitude constraints")
            when = pd.to_datetime(p["time"], unit="ms", utc=True)
            updated = pd.to_datetime(p["updated"], unit="ms", utc=True)
            if pd.isna(when) or pd.isna(updated) or updated < when:
                raise ValueError("Invalid USGS timestamps")
            rows.append(dict(event_id=feature["id"], event_time_utc=when, updated_at_utc=updated,
                             local_date=when.tz_convert(ECUADOR_TZ).tz_localize(None).normalize(),
                             magnitude=magnitude, magnitude_type=p.get("magType"), longitude=lon,
                             latitude=lat, depth_km=depth, place=p.get("place"), status=p.get("status"),
                             source_url=p.get("url")))
    frame = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if frame.empty:
        raise ValueError("No earthquakes collected; inspect source responses before using zero counts")
    raw_rows = len(frame)
    # Inclusive USGS year boundaries may repeat the same event. Keep the newest
    # catalog revision, with stable source-order tie handling.
    frame = frame.sort_values("updated_at_utc", kind="stable").drop_duplicates("event_id", keep="last")
    duplicate_rows = raw_rows - len(frame)
    start = pd.Timestamp(config["start_date"]) - pd.Timedelta(days=config["lookback_days"])
    end = pd.Timestamp(config["end_date"])
    outside = ~frame.local_date.between(start, end)
    audit = {"raw_response_rows": raw_rows, "duplicate_ids_removed": duplicate_rows,
             "outside_local_window_removed": int(outside.sum())}
    frame = frame.loc[~outside].sort_values(["event_time_utc", "event_id"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No events inside the local study/lookback window")
    return frame, audit


def make_quake_features(events: pd.DataFrame, start: str, end: str,
                        lookback_days: int = 7, strong_magnitude: float = 6.) -> pd.DataFrame:
    if lookback_days < 7 or pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("Require at least seven lookback days and ordered study dates")
    first = pd.Timestamp(start) - pd.Timedelta(days=lookback_days)
    daily = pd.DataFrame(index=pd.date_range(first, end, freq="D", name="date"))
    by_date = events.groupby("local_date")
    daily["quake_count"] = by_date.size().reindex(daily.index, fill_value=0).astype("int32")
    daily["quake_max_magnitude"] = by_date.magnitude.max().reindex(daily.index)
    strong = events.loc[events.magnitude >= strong_magnitude].groupby("local_date").size()
    daily["strong_quake_count"] = strong.reindex(daily.index, fill_value=0).astype("int32")
    daily["quake_count_lag1"] = daily.quake_count.shift(1)
    daily["strong_quake_count_lag1"] = daily.strong_quake_count.shift(1)
    daily["quake_max_magnitude_lag1"] = daily.quake_max_magnitude.shift(1)
    daily["quake_count_prev7d"] = daily.quake_count.shift(1).rolling(7, min_periods=7).sum()
    return daily.loc[start:end].reset_index()


def aggregate_sales(path) -> pd.DataFrame:
    """Read the 3M-row labeled source in bounded chunks; emit only daily totals."""
    parts = []
    for chunk in pd.read_csv(path, usecols=["date", "sales"], parse_dates=["date"], chunksize=200000):
        if chunk.sales.isna().any() or (chunk.sales < 0).any():
            raise ValueError("Daily sales collection join requires valid nonnegative labels")
        chunk["zero_rows"] = chunk.sales.eq(0).astype("int32")
        chunk["log_sales_sum"] = np.log1p(chunk.sales)
        parts.append(chunk.groupby("date").agg(sales_sum=("sales", "sum"), sales_rows=("sales", "size"),
                                               zero_rows=("zero_rows", "sum"), log_sales_sum=("log_sales_sum", "sum")))
    totals = pd.concat(parts).groupby(level="date").sum()
    totals["sales_mean"] = totals.sales_sum / totals.sales_rows
    totals["log_sales_mean"] = totals.log_sales_sum / totals.sales_rows
    totals["zero_rate"] = totals.zero_rows / totals.sales_rows
    return totals.reset_index()
