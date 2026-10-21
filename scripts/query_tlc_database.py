"""Query the local Green Taxi SQLite warehouse from the command line."""
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3

import pandas as pd

from scripts.build_tlc_database import DEFAULT_DB


def connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}. Run scripts.build_tlc_database first.")
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def show(frame: pd.DataFrame, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"Saved {len(frame):,} rows to {output}")
    else:
        print(frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("summary", "daily", "monthly"))
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--start-date", help="Required by daily, inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", help="Required by daily, inclusive YYYY-MM-DD")
    parser.add_argument("--zone", type=int, help="Optional pickup zone ID")
    parser.add_argument("--year", type=int, help="Required by monthly")
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    args = parser.parse_args()

    with connect_read_only(args.database) as con:
        if args.command == "summary":
            frame = pd.read_sql_query("""SELECT source_months.months AS source_months,
                source_months.raw_rows, source_months.kept_rows,
                daily.panel_rows, daily.model_zones, daily.first_date, daily.last_date
                FROM (SELECT COUNT(*) AS months, SUM(raw_rows) AS raw_rows,
                      SUM(kept_rows) AS kept_rows FROM source_file) AS source_months
                CROSS JOIN (SELECT COUNT(*) AS panel_rows, COUNT(DISTINCT zone_id) AS model_zones,
                            MIN(pickup_date) AS first_date, MAX(pickup_date) AS last_date
                            FROM daily_zone_demand WHERE selected_for_model=1) AS daily""", con)
        elif args.command == "daily":
            if not args.start_date or not args.end_date:
                parser.error("daily requires --start-date and --end-date")
            if args.end_date < args.start_date:
                parser.error("--end-date must be on or after --start-date")
            sql = """SELECT d.pickup_date AS date, d.zone_id, z.borough, z.zone_name,
                     d.pickup_count AS pickups FROM daily_zone_demand d
                     JOIN taxi_zone z USING(zone_id)
                     WHERE d.selected_for_model=1 AND d.pickup_date BETWEEN ? AND ?"""
            params: list[object] = [args.start_date, args.end_date]
            if args.zone is not None:
                sql += " AND d.zone_id=?"
                params.append(args.zone)
            sql += " ORDER BY d.pickup_date, d.zone_id"
            frame = pd.read_sql_query(sql, con, params=params)
        else:
            if args.year is None:
                parser.error("monthly requires --year")
            start, end = f"{args.year:04d}-01-01", f"{args.year + 1:04d}-01-01"
            sql = """SELECT substr(d.pickup_date,1,7) AS month, d.zone_id,
                     z.borough, z.zone_name, SUM(d.pickup_count) AS pickups
                     FROM daily_zone_demand d JOIN taxi_zone z USING(zone_id)
                     WHERE d.selected_for_model=1 AND d.pickup_date>=? AND d.pickup_date<?"""
            params = [start, end]
            if args.zone is not None:
                sql += " AND d.zone_id=?"
                params.append(args.zone)
            sql += " GROUP BY month,d.zone_id,z.borough,z.zone_name ORDER BY month,d.zone_id"
            frame = pd.read_sql_query(sql, con, params=params)
        show(frame, args.output)


if __name__ == "__main__":
    main()
