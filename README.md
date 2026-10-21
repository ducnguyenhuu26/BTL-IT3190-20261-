# Green Taxi Data Mining and Demand Forecasting

This project builds a reproducible data product from NYC Green Taxi trip records. It crawls the official TLC catalog, collects monthly Parquet files, checks and cleans trip records, loads a SQLite warehouse, explores demand by pickup zone, and forecasts next-day pickups.

The prediction target is the number of completed Green Taxi pickups per day in a pickup zone. It measures recorded trips; it does not measure requests that could not be served. The project uses one dataset and one forecasting question throughout.

## Data workflow

```text
TLC catalog HTML
      │ crawl monthly file links
      ▼
Raw Parquet snapshots + SHA-256 manifest
      │ validate month, timestamp and zone
      ▼
SQLite warehouse ──► EDA and daily zone panel
                           │ prior-day features
                           ▼
               LightGBM + online residual correction
```

The primary source is the [NYC TLC Trip Record Data catalog](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). The collector fetches and parses the catalog HTML. In the current snapshot, it found 27 of the 36 configured monthly links on the page. TLC no longer lists nine older months there, so the collector used the configured official archive URL pattern for those files and records that fallback in the crawl manifest. All 36 local Parquet files were present and checksum-verified.

The collected period is January 2023 through December 2025. The pipeline read 2,038,653 source rows, retained 2,032,098 records, and rejected 6,555 rows that failed the timestamp-month or pickup-zone rules. The reason counts can overlap; the rejected total counts distinct source rows. The model dataset contains 12 zones selected using 2023 trips only, with a complete daily panel of 13,152 zone-days. One zone-day has zero pickups after all monthly files were verified.

The SQLite database is created locally at `data/processed/tlc_green/2026-10-16/green_taxi.sqlite`. It stores retained trip records, monthly source metadata and quality counts, zone lookup information, and the daily panel used by the forecasting model. Raw files, database files, models, and generated results are excluded from Git. Source URLs and hashes are tracked in [the raw-file manifest](provenance/tlc_green_manifest.json) and [the catalog crawl manifest](provenance/tlc_green_crawl_manifest.json).

For the full collection, cleaning and schema notes, see [the data pipeline report](reports/DATA_PIPELINE.md). For the forecasting setup and measured results, see [the project study](reports/TLC_GREEN_DEMAND.md).

## Set up

Python 3.11 or newer is recommended. The SQLite driver is part of Python; PyArrow reads the source Parquet files.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run from collected files

These commands recrawl the source page, reuse local Parquet files when their hashes match, refresh the crawl provenance, and rebuild the SQLite database.

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_tlc_green --stage all
.\.venv\Scripts\python.exe -m scripts.build_tlc_database
.\.venv\Scripts\python.exe -m scripts.query_tlc_database summary
.\.venv\Scripts\python.exe -m scripts.query_tlc_database monthly --year 2025
.\.venv\Scripts\python.exe -m scripts.run_tlc_green_forecast --stage develop
.\.venv\Scripts\python.exe -m scripts.run_tlc_green_forecast --stage test
.\.venv\Scripts\python.exe -m scripts.report_tlc_green
```

To inspect a date interval or save the result to CSV:

```powershell
.\.venv\Scripts\python.exe -m scripts.query_tlc_database daily --start-date 2025-01-01 --end-date 2025-01-07 --output results/sample_week.csv
```

The forecast runner currently reads the checksum-verified daily panel produced from the same collected Parquet snapshots. SQLite is the queryable data warehouse; the compressed CSV panel remains as a simple model interchange file.

## Data mining and forecasting

The collector keeps source files immutable and records the catalog URL, response status, retrieval time, HTML hash, extracted file links, fallback months, and Parquet hashes. Preparation reads only pickup timestamps and pickup zone IDs. A trip is retained when its timestamp falls within the month represented by its source file and its pickup zone ID is an integer from 1 through 263. Pickup zones with the highest trip counts in 2023 are selected before looking at later years.

The forecast is one step ahead for each of the 12 selected zones. LightGBM is fit on 2023 and stays fixed. Features use calendar fields and prior observations only: lags of 1, 7 and 28 days, and shifted 7- and 28-day means. Parameters for the online Gaussian residual correction are selected on January–June 2024. July–December 2024 is a continuation period; 2025 is the locked test period. Actual daily counts update the residual filter only after all zones for that date have been forecast.

On the 2025 test set (4,380 zone-days), seasonal naive, fixed LightGBM, and LightGBM plus the selected Kalman correction had RMSLE 0.321347, 0.302883, and 0.263461. These results apply to this city, service, selected zone set, and time split. They do not establish that the correction will improve every baseline, city, or future period. The [study](reports/TLC_GREEN_DEMAND.md) includes MAE, uncertainty intervals, zone-level scores, and limitations.

## Repository map

- `scripts/collect_tlc_green.py` crawls the TLC catalog, downloads and verifies source files, and prepares the daily panel.
- `scripts/build_tlc_database.py` builds the local SQLite warehouse from raw snapshots.
- `scripts/query_tlc_database.py` provides summary, daily and monthly warehouse queries.
- `src/tlc_green_data.py` creates causal forecasting features.
- `src/online_gaussian_bias.py` implements the online residual correction.
- `scripts/run_tlc_green_forecast.py` trains, selects and evaluates the forecasting models.
- `scripts/report_tlc_green.py` produces the Green Taxi results report and figure.
- `reports/DATA_PIPELINE.md` documents crawl, cleaning, integration and database construction.

Historical research files from earlier directions remain in the checkout. They are not part of the current project question or Green Taxi data workflow.
