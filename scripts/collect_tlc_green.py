"""Crawl the TLC catalog, collect monthly Parquet files, and prepare daily demand."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tlc_green_taxi.json"
RAW = ROOT / "data/raw/tlc_green/2026-10-16"
PROCESSED = ROOT / "data/processed/tlc_green/2026-10-16"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
MONTHLY_FILE = re.compile(r"green_tripdata_(\d{4}-\d{2})\.parquet(?:$|[?#])", re.I)


class TLCLinkParser(HTMLParser):
    """Find monthly Green Taxi archive links in the public TLC catalog HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href and MONTHLY_FILE.search(href):
            self.hrefs.append(href)


def discover_sources(config: dict) -> tuple[dict[str, str], dict]:
    """Crawl the official catalog page and discover its monthly Parquet URLs."""
    request = Request(config["source_page"], headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        body = response.read()
        final_url = response.geturl()
        status = response.status
    parser = TLCLinkParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    base_host = urlparse(config["source_page"]).hostname
    urls: dict[str, str] = {}
    for href in parser.hrefs:
        url = urljoin(final_url, href)
        match = MONTHLY_FILE.search(url)
        if not match:
            continue
        host = urlparse(url).hostname or ""
        if host not in {base_host, "d37ci6vzurychx.cloudfront.net"}:
            continue
        month = match.group(1)
        if month in months(config):
            urls[month] = url
    expected = months(config)
    scraped_months = sorted(set(expected) & urls.keys())
    missing = sorted(set(expected) - urls.keys())
    # TLC's catalog page does not consistently retain every historical month.
    # The official archive URL pattern is documented in the project config;
    # record those months separately instead of silently claiming they were
    # discovered in the HTML.
    template = config.get("source_template")
    if missing and not template:
        raise ValueError(f"TLC catalog omitted months and no official archive template is configured: {missing}")
    for month in missing:
        urls[month] = template.format(month=month)
    catalog = {"url": config["source_page"], "final_url": final_url,
               "http_status": status, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
               "html_bytes": len(body), "html_sha256": hashlib.sha256(body).hexdigest(),
               "scraped_months": scraped_months, "scraped_month_count": len(scraped_months),
               "template_fallback_months": missing, "discovered_months": len(urls)}
    return urls, catalog


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def months(config: dict) -> list[str]:
    first = pd.Period(config["first_month"], freq="M")
    last = pd.Period(config["last_month"], freq="M")
    return [str(month) for month in pd.period_range(first, last, freq="M")]


def validate_parquet(path: Path) -> None:
    """Reject truncated downloads and incompatible monthly files before caching."""
    import pyarrow.parquet as pq

    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise ValueError(f"Invalid or incomplete Parquet file: {path}") from exc
    required = {"lpep_pickup_datetime", "PULocationID"}
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Parquet file lacks required columns {sorted(missing)}: {path}")
    if parquet.metadata.num_rows <= 0:
        raise ValueError(f"Parquet file contains no rows: {path}")


def collect() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_urls, catalog = discover_sources(config)
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "manifest.json"
    crawl_manifest_path = RAW / "catalog_crawl_manifest.json"
    existing = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_sha256"] != sha256(CONFIG):
            raise ValueError("Existing TLC snapshot uses a different config")
        for item in manifest["files"]:
            if sha256(RAW / item["filename"]) != item["sha256"]:
                raise ValueError(f"Cached TLC file changed: {item['filename']}")
            validate_parquet(RAW / item["filename"])
            existing[item["month"]] = item
        if [item["month"] for item in manifest["files"]] != months(config):
            raise ValueError("Incomplete existing TLC manifest")
    records = []
    for month in months(config):
        filename = f"green_tripdata_{month}.parquet"
        url = source_urls[month]
        path = RAW / filename
        if month not in existing:
            temp = path.with_suffix(".download")
            request = Request(url, headers={"User-Agent": USER_AGENT})
            for attempt in range(5):
                try:
                    with urlopen(request, timeout=90) as response, temp.open("wb") as target:
                        while block := response.read(1 << 20):
                            target.write(block)
                    validate_parquet(temp)
                    temp.replace(path)
                    break
                except (HTTPError, URLError) as exc:
                    if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                        temp.unlink(missing_ok=True)
                        raise
                    temp.unlink(missing_ok=True)
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
                except (OSError, ValueError) as exc:
                    temp.unlink(missing_ok=True)
                    if attempt == 4:
                        raise
                    print(f"Invalid download for {month}: {exc}; retrying", flush=True)
                    time.sleep(2**attempt)
            time.sleep(0.3)
        record = existing.get(month, {})
        record.update({"month": month, "filename": filename, "url": url,
                       "link_source": "catalog_html" if month in catalog["scraped_months"] else "configured_archive_template",
                       "bytes": path.stat().st_size, "sha256": sha256(path)})
        record.setdefault("recorded_at_utc", datetime.now(timezone.utc).isoformat())
        records.append(record)
        print(f"{month}: {'verified cached' if month in existing else 'collected'} {path.stat().st_size:,} bytes", flush=True)
    crawl_manifest = {"source_page": config["source_page"], "config_sha256": sha256(CONFIG),
                      "user_agent": USER_AGENT, "catalog_crawl": catalog,
                      "raw_manifest_sha256": sha256(manifest_path), "files": records}
    crawl_manifest_path.write_text(json.dumps(crawl_manifest, indent=2), encoding="utf-8")
    print(f"Saved crawl provenance: {crawl_manifest_path}", flush=True)


def prepare() -> None:
    import pyarrow.parquet as pq

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    if sha256(CONFIG) != manifest["config_sha256"] or [x["month"] for x in manifest["files"]] != months(config):
        raise ValueError("Manifest and config disagree")
    chunks, audits = [], []
    for entry in manifest["files"]:
        path = RAW / entry["filename"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Raw monthly hash mismatch: {path}")
        table = pq.read_table(path, columns=["lpep_pickup_datetime", "PULocationID"])
        data = table.to_pandas()
        date = pd.to_datetime(data.lpep_pickup_datetime, errors="coerce")
        zone = pd.to_numeric(data.PULocationID, errors="coerce")
        month = pd.Period(entry["month"], freq="M")
        in_month = (date >= month.start_time) & (date < (month+1).start_time)
        valid_zone = zone.between(config["valid_zone_min"], config["valid_zone_max"]) & (zone % 1 == 0)
        valid = in_month & valid_zone
        counted = pd.DataFrame({"date": date[valid].dt.floor("D"),
                                "zone": zone[valid].astype(int)})
        chunks.append(counted.groupby(["date", "zone"], sort=True).size().rename("pickups").reset_index())
        audits.append({"month": entry["month"], "raw_rows": len(data),
                       "missing_timestamp": int(date.isna().sum()),
                       "outside_month": int((~in_month & date.notna()).sum()),
                       "invalid_zone": int((~valid_zone).sum()),
                       "kept_rows": int(valid.sum())})
        print(f"Prepared {entry['month']}: kept {int(valid.sum()):,}/{len(data):,}", flush=True)
    daily = pd.concat(chunks).groupby(["date", "zone"], as_index=False).pickups.sum()
    train_totals = daily.loc[daily.date < "2024-01-01"].groupby("zone").pickups.sum()
    top = train_totals.sort_values(ascending=False, kind="stable").head(config["top_zones"]).index.to_list()
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    skeleton = pd.MultiIndex.from_product([dates, top], names=["date", "zone"]).to_frame(index=False)
    panel = skeleton.merge(daily, on=["date", "zone"], how="left", validate="one_to_one")
    panel["pickups"] = panel.pickups.fillna(0).astype(int)
    # A zero here means the complete monthly file contains no valid pickup in
    # this zone on that day. No missing month is silently transformed to zero.
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED / "daily_top_zones.csv.gz", index=False, compression="gzip")
    pd.DataFrame(audits).to_csv(PROCESSED / "monthly_quality.csv", index=False)
    quality = {"months": len(audits), "raw_rows": int(sum(a["raw_rows"] for a in audits)),
               "kept_rows": int(sum(a["kept_rows"] for a in audits)),
               "zero_zone_days": int((panel.pickups == 0).sum()),
               "top_zones_selected_from_2023": top, "panel_rows": len(panel),
               "manifest_sha256": sha256(RAW / "manifest.json"),
               "panel_sha256": sha256(PROCESSED / "daily_top_zones.csv.gz")}
    (PROCESSED / "quality_audit.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(json.dumps(quality, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("collect", "prepare", "all"), required=True)
    stage = parser.parse_args().stage
    if stage in ("collect", "all"):
        collect()
    if stage in ("prepare", "all"):
        prepare()


if __name__ == "__main__":
    main()
