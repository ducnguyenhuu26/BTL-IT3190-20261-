# Project instructions — Green Taxi Data Mining and Forecasting

## Project goal

Build one reproducible data product for **next-day Green Taxi pickup counts by NYC taxi zone**. The core coursework must demonstrate data mining and data engineering: crawl a public source, preserve raw snapshots, clean and audit records, integrate them into a local SQLite database, explore demand patterns, then train and evaluate a forecasting model. Do not treat Kaggle downloads as the project's data-collection contribution.

The project uses completed trips as an observable proxy for realized activity. It does not estimate unmet passenger demand. Keep Favorita, USGS, Wikimedia, Beijing, and bike experiments out of the primary narrative and do not combine their data or scores with the Green Taxi study.

## Source and data protocol

- Primary source: NYC TLC Trip Record Data catalog, linked from `configs/tlc_green_taxi.json`.
- Use the official catalog HTML to discover monthly Green Taxi Parquet links. If a historical month is absent from the page, use only the documented official URL template and mark it as a fallback in crawl provenance.
- Save raw files under `data/raw/` with URL, retrieval time, byte count, and SHA-256. Keep raw files, SQLite databases, models, and generated results out of Git.
- Validate downloaded Parquet files before accepting them. Before processing, recheck hashes and required columns.
- Clean using pickup timestamp and pickup zone only. Retain a row only when the pickup timestamp belongs to the source month and the zone is an integer in 1–263. Preserve monthly raw/kept/rejection counts; reason counts may overlap.
- Build a local SQLite warehouse with source-file lineage, taxi-zone dimension, cleaned trip facts, and a complete daily panel for the selected model zones. Add indexes and run SQLite integrity checks.
- Select model zones from 2023 data only and freeze that set across later periods. Do not infer zero demand when a source file or date is missing.

## Forecasting protocol

- Target: completed pickups per `(zone, date)`.
- Forecast one date at a time using only information available before that date. All zone forecasts for a date must be formed before same-date labels update online state.
- Lag and rolling features must exclude the target date. Do not use drop-off fields, same-day totals from other zones, or test labels for model/parameter selection.
- Current split: train on 2023; select online correction settings on 2024-01 through 2024-06; use 2024-07 through 2024-12 as continuation; keep 2025 as the locked test period.
- Report seasonal naive, fixed LightGBM, and online correction on the same rows. Report RMSLE and MAE; state the city/service/zone scope and the monthly-publication delay.
- Describe observed residual spikes as forecast errors; do not call them verified regime shifts or ground-truth jumps.

## Code and delivery

- Use Python 3.11+, CPU, `argparse`, logging/clear CLI output, and small focused modules. No notebooks or unnecessary framework dependencies.
- Keep acquisition, cleaning/features, database, model, evaluation, and reporting responsibilities separate.
- Primary docs: `README.md`, `reports/DATA_PIPELINE.md`, `reports/BTL_MAIN_OUTLINE.md`, and `reports/TLC_GREEN_DEMAND.md`.
- Never commit raw data, `.venv`, secrets, SQLite databases, trained models, or large result artifacts. Do not push unless the user asks.
