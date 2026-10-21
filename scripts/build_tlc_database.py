"""Build a provenance-aware SQLite warehouse from collected TLC Parquet files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

import pandas as pd
import pyarrow.parquet as pq

from scripts.collect_tlc_green import CONFIG, RAW, ROOT, sha256

PROCESSED = ROOT / "data/processed/tlc_green/2026-10-16"
LOOKUP = ROOT / "provenance/tlc_taxi_zone_lookup.csv"
DEFAULT_DB = PROCESSED / "green_taxi.sqlite"
BATCH_ROWS = 50_000


def build(database: Path = DEFAULT_DB) -> dict:
    """Create a clean database; replace the destination only after commit succeeds."""
    manifest_path = RAW / "manifest.json"
    crawl_manifest_path = RAW / "catalog_crawl_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    crawl_manifest = json.loads(crawl_manifest_path.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected_months = [str(p) for p in pd.period_range(config["first_month"], config["last_month"], freq="M")]
    if manifest.get("config_sha256") != sha256(CONFIG):
        raise ValueError("Manifest and current collection configuration disagree")
    if [row["month"] for row in manifest.get("files", [])] != expected_months:
        raise ValueError("Raw manifest does not contain every configured month in order")
    if crawl_manifest.get("raw_manifest_sha256") != sha256(manifest_path):
        raise ValueError("Crawl manifest does not match the immutable raw-file manifest")
    if crawl_manifest.get("config_sha256") != sha256(CONFIG):
        raise ValueError("Crawl manifest and current collection configuration disagree")
    if [row["month"] for row in crawl_manifest.get("files", [])] != expected_months:
        raise ValueError("Crawl manifest does not contain every configured month")
    if not crawl_manifest.get("catalog_crawl", {}).get("html_sha256"):
        raise ValueError("Raw manifest lacks the TLC catalog crawl provenance")

    database.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix="tlc_warehouse_", suffix=".sqlite", dir=database.parent, delete=False)
    temp_db = Path(handle.name)
    handle.close()
    try:
        result = _build_database(temp_db, manifest, crawl_manifest, manifest_path, crawl_manifest_path, config)
        result["database"] = str(database)
        (PROCESSED / "database_quality_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        temp_db.replace(database)
        return result
    finally:
        temp_db.unlink(missing_ok=True)


def _build_database(database: Path, manifest: dict, crawl_manifest: dict,
                    manifest_path: Path, crawl_manifest_path: Path, config: dict) -> dict:
    con = sqlite3.connect(database)
    try:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA foreign_keys=ON")
        con.executescript("""
            CREATE TABLE collection_run (
                run_id INTEGER PRIMARY KEY,
                collected_at_utc TEXT NOT NULL,
                source_page TEXT NOT NULL,
                catalog_html_sha256 TEXT NOT NULL,
                config_sha256 TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL,
                crawl_manifest_sha256 TEXT NOT NULL,
                raw_rows INTEGER NOT NULL DEFAULT 0,
                kept_rows INTEGER NOT NULL DEFAULT 0,
                rejected_rows INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE source_file (
                month TEXT PRIMARY KEY,
                filename TEXT NOT NULL UNIQUE,
                source_url TEXT NOT NULL,
                link_source TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                raw_rows INTEGER NOT NULL,
                missing_timestamp INTEGER NOT NULL,
                outside_month INTEGER NOT NULL,
                invalid_zone INTEGER NOT NULL,
                kept_rows INTEGER NOT NULL
            );
            CREATE TABLE taxi_zone (
                zone_id INTEGER PRIMARY KEY,
                borough TEXT,
                zone_name TEXT,
                service_zone TEXT,
                selected_for_model INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE trip_record (
                trip_id INTEGER PRIMARY KEY,
                source_month TEXT NOT NULL REFERENCES source_file(month),
                source_row INTEGER NOT NULL,
                pickup_timestamp TEXT NOT NULL,
                pickup_date TEXT NOT NULL,
                pickup_zone_id INTEGER NOT NULL REFERENCES taxi_zone(zone_id),
                UNIQUE(source_month, source_row)
            );
            CREATE TABLE daily_zone_demand (
                pickup_date TEXT NOT NULL,
                zone_id INTEGER NOT NULL REFERENCES taxi_zone(zone_id),
                pickup_count INTEGER NOT NULL CHECK(pickup_count >= 0),
                selected_for_model INTEGER NOT NULL,
                PRIMARY KEY(pickup_date, zone_id)
            );
            CREATE INDEX ix_trip_date_zone ON trip_record(pickup_date, pickup_zone_id);
            CREATE INDEX ix_daily_zone_date ON daily_zone_demand(zone_id, pickup_date);
        """)
        catalog = crawl_manifest["catalog_crawl"]
        con.execute("""INSERT INTO collection_run
            (collected_at_utc, source_page, catalog_html_sha256, config_sha256,
             manifest_sha256, crawl_manifest_sha256)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (catalog["retrieved_at_utc"], catalog["url"], catalog["html_sha256"],
             manifest["config_sha256"], sha256(manifest_path), sha256(crawl_manifest_path)))

        lookup = pd.read_csv(LOOKUP).rename(columns={"LocationID": "zone_id", "Borough": "borough",
                                                       "Zone": "zone_name", "service_zone": "service_zone"})
        lookup["zone_id"] = pd.to_numeric(lookup.zone_id, errors="raise").astype(int)
        con.executemany("INSERT INTO taxi_zone(zone_id, borough, zone_name, service_zone) VALUES (?, ?, ?, ?)",
                        lookup[["zone_id", "borough", "zone_name", "service_zone"]].itertuples(index=False, name=None))

        total_raw = total_kept = 0
        audits: list[dict] = []
        crawl_by_month = {row["month"]: row for row in crawl_manifest["files"]}
        for entry in manifest["files"]:
            path = RAW / entry["filename"]
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise ValueError(f"Missing or changed raw Parquet source: {path}")
            parquet = pq.ParquetFile(path)
            raw_rows = parquet.metadata.num_rows
            crawl_entry = crawl_by_month[entry["month"]]
            raw_offset = kept = missing_ts = outside_month = invalid_zone = 0
            period = pd.Period(entry["month"], freq="M")
            con.execute("""INSERT INTO source_file
                (month, filename, source_url, link_source, bytes, sha256, raw_rows,
                 missing_timestamp, outside_month, invalid_zone, kept_rows)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0)""",
                (entry["month"], entry["filename"], crawl_entry["url"], crawl_entry["link_source"],
                 entry["bytes"], entry["sha256"], raw_rows))
            batch_inserts: list[tuple] = []
            for batch in parquet.iter_batches(batch_size=BATCH_ROWS,
                                               columns=["lpep_pickup_datetime", "PULocationID"]):
                frame = batch.to_pandas()
                timestamp = pd.to_datetime(frame.lpep_pickup_datetime, errors="coerce")
                zone = pd.to_numeric(frame.PULocationID, errors="coerce")
                valid_month = timestamp.ge(period.start_time) & timestamp.lt((period + 1).start_time)
                valid_zone = zone.between(config["valid_zone_min"], config["valid_zone_max"]) & zone.mod(1).eq(0)
                valid = valid_month & valid_zone
                missing_ts += int(timestamp.isna().sum())
                outside_month += int((~valid_month & timestamp.notna()).sum())
                invalid_zone += int((~valid_zone).sum())
                valid_positions = valid.to_numpy().nonzero()[0]
                if len(valid_positions):
                    valid_ts = timestamp.iloc[valid_positions]
                    valid_zone_ids = zone.iloc[valid_positions].astype(int)
                    batch_inserts.extend((entry["month"], raw_offset + int(i),
                                          valid_ts.iloc[j].isoformat(), valid_ts.iloc[j].date().isoformat(),
                                          int(valid_zone_ids.iloc[j]))
                                         for j, i in enumerate(valid_positions))
                    kept += len(valid_positions)
                raw_offset += len(frame)
                if len(batch_inserts) >= BATCH_ROWS:
                    con.executemany("""INSERT INTO trip_record
                        (source_month, source_row, pickup_timestamp, pickup_date, pickup_zone_id)
                        VALUES (?, ?, ?, ?, ?)""", batch_inserts)
                    batch_inserts.clear()
            if batch_inserts:
                con.executemany("""INSERT INTO trip_record
                    (source_month, source_row, pickup_timestamp, pickup_date, pickup_zone_id)
                    VALUES (?, ?, ?, ?, ?)""", batch_inserts)
            if raw_offset != raw_rows or not kept:
                raise ValueError(f"Unexpected row count after reading {entry['filename']}")
            record = {"month": entry["month"], "filename": entry["filename"],
                      "bytes": entry["bytes"], "sha256": entry["sha256"], "raw_rows": raw_rows,
                      "missing_timestamp": missing_ts, "outside_month": outside_month,
                      "invalid_zone": invalid_zone, "kept_rows": kept}
            con.execute("""UPDATE source_file SET missing_timestamp=:missing_timestamp,
                outside_month=:outside_month, invalid_zone=:invalid_zone, kept_rows=:kept_rows
                WHERE month=:month""", record)
            audits.append(record)
            total_raw += raw_rows
            total_kept += kept
            print(f"Loaded {entry['month']}: {kept:,}/{raw_rows:,} trips", flush=True)

        selected = [row[0] for row in con.execute("""SELECT pickup_zone_id FROM trip_record
            WHERE pickup_date >= '2023-01-01' AND pickup_date < '2024-01-01'
            GROUP BY pickup_zone_id ORDER BY COUNT(*) DESC, pickup_zone_id ASC LIMIT ?""",
            (config["top_zones"],)).fetchall()]
        if len(selected) != config["top_zones"]:
            raise ValueError("Could not select the configured number of model zones")
        con.executemany("UPDATE taxi_zone SET selected_for_model=1 WHERE zone_id=?", [(z,) for z in selected])
        dates = pd.date_range(config["first_month"] + "-01", pd.Period(config["last_month"], freq="M").end_time.normalize())
        selected_set = set(selected)
        daily = {(date, int(zone)): int(count) for date, zone, count in con.execute("""SELECT pickup_date,
            pickup_zone_id, COUNT(*) FROM trip_record GROUP BY pickup_date, pickup_zone_id""")}
        con.executemany("INSERT INTO daily_zone_demand VALUES (?, ?, ?, ?)",
                        ((date.date().isoformat(), zone, daily.get((date.date().isoformat(), zone), 0), int(zone in selected_set))
                         for date in dates for zone in selected))

        total_rejected = total_raw - total_kept
        con.execute("UPDATE collection_run SET raw_rows=?, kept_rows=?, rejected_rows=? WHERE run_id=1",
                    (total_raw, total_kept, total_rejected))
        con.execute("CREATE INDEX ix_daily_selected_date ON daily_zone_demand(selected_for_model, pickup_date)")
        con.commit()
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity}")
        panel_rows = con.execute("SELECT COUNT(*) FROM daily_zone_demand WHERE selected_for_model=1").fetchone()[0]
        result = {"database": str(database), "source_months": len(audits), "raw_trips": total_raw,
                  "kept_trips": total_kept, "rejected_trips": total_rejected,
                  "selected_zone_ids_2023": selected, "model_panel_rows": panel_rows,
                  "integrity_check": integrity, "manifest_sha256": sha256(manifest_path),
                  "crawl_manifest_sha256": sha256(crawl_manifest_path),
                  "catalog_html_sha256": catalog["html_sha256"],
                  "scraped_month_links": catalog["scraped_month_count"],
                  "template_fallback_months": catalog["template_fallback_months"]}
        return result
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    print(json.dumps(build(args.database), indent=2), flush=True)


if __name__ == "__main__":
    main()
