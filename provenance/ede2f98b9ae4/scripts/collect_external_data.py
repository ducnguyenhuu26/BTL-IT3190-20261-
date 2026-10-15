"""Collect USGS GeoJSON, normalize it, and document a Favorita enrichment join."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.external_data import ECUADOR_TZ, MODEL_FEATURES, aggregate_sales, make_quake_features, normalize_catalog

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/collection_usgs.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def year_queries(config: dict) -> list[tuple[int, str]]:
    first = pd.Timestamp(config["start_date"]) - pd.Timedelta(days=config["lookback_days"])
    stop = pd.Timestamp(config["end_date"]) + pd.Timedelta(days=1)
    result = []
    for year in range(first.year, stop.year + 1):
        left = max(first, pd.Timestamp(year=year, month=1, day=1))
        right = min(stop, pd.Timestamp(year=year + 1, month=1, day=1))
        if left >= right:
            continue
        def utc_text(value):
            return value.to_pydatetime().replace(tzinfo=ECUADOR_TZ).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        parameters = dict(format="geojson", starttime=utc_text(left), endtime=utc_text(right),
                          minmagnitude=config["min_magnitude"], eventtype="earthquake",
                          orderby="time-asc", limit=20000, **config["bbox"])
        result.append((year, config["endpoint"] + "?" + urlencode(parameters)))
    return result


def fetch(url: str, config: dict) -> tuple[bytes, dict]:
    for attempt in range(config["attempts"]):
        try:
            request = Request(url, headers={"User-Agent": "MLCourse-FavoritaCollection/1.0 (educational research)",
                                            "Accept": "application/geo+json, application/json"})
            with urlopen(request, timeout=config["timeout_seconds"]) as response:
                if response.status != 200 or "json" not in response.headers.get("Content-Type", ""):
                    raise ValueError("Expected HTTP 200 JSON; refusing to store an error page as data")
                body = response.read()
                metadata = {"http_status": response.status, "content_type": response.headers.get("Content-Type"),
                            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                            "url": url, "final_url": response.url}
            parsed = json.loads(body)
            features = parsed.get("features")
            count = parsed.get("metadata", {}).get("count")
            if (parsed.get("type") != "FeatureCollection" or not isinstance(features, list) or
                    (count is not None and count != len(features))):
                raise ValueError("USGS response schema/count invalid")
            if len(parsed["features"]) >= 20000:
                raise ValueError("Possible response truncation; reduce query interval")
            return body, metadata
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt + 1 == config["attempts"]:
                raise
        except (URLError, TimeoutError):
            if attempt + 1 == config["attempts"]:
                raise
        time.sleep(min(2 ** (attempt + 1), 8))
    raise RuntimeError("USGS download attempts exhausted")


def collect(snapshot: Path, config: dict) -> None:
    snapshot.mkdir(parents=True, exist_ok=True)
    records = []
    for year, url in year_queries(config):
        raw, meta = snapshot / f"events_{year}.geojson", snapshot / f"events_{year}.metadata.json"
        if raw.exists() or meta.exists():
            if not raw.exists() or not meta.exists():
                raise ValueError(f"Incomplete cached response: {year}; preserve it and use a new snapshot directory")
            metadata = json.loads(meta.read_text(encoding="utf-8"))
            if metadata["url"] != url or digest(raw) != metadata["sha256"]:
                raise ValueError(f"Cached response URL/hash mismatch: {year}")
            print(f"Cached {year}: {metadata['response_event_count']} events", flush=True)
        else:
            body, metadata = fetch(url, config)
            raw.write_bytes(body)
            metadata.update(sha256=digest(raw), bytes=len(body),
                            response_event_count=len(json.loads(body)["features"]))
            save_json(meta, metadata)
            print(f"Collected {year}: {metadata['response_event_count']} events, {len(body):,} bytes", flush=True)
            time.sleep(config["request_interval_seconds"])
        records.append({"year": year, "file": raw.name, "metadata_file": meta.name, **metadata})
    manifest = {"source": config["source"], "configuration": config, "configuration_sha256": digest(CONFIG),
                "responses": records,
                "availability": "Retrospective catalog; origin time and last update are not first publication time",
                "scope": "Bounding rectangle around mainland Ecuador, including nearby offshore/neighbor events"}
    save_json(snapshot / "manifest.json", manifest)


def load_snapshot(snapshot: Path, config: dict) -> tuple[list[dict], dict]:
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest["configuration_sha256"] != digest(CONFIG) or manifest["configuration"] != config:
        raise ValueError("Collection configuration differs from archived manifest")
    expected = year_queries(config)
    if [(r["year"], r["url"]) for r in manifest["responses"]] != expected:
        raise ValueError("Snapshot does not cover every required yearly query")
    payloads = []
    for response in manifest["responses"]:
        path = snapshot / response["file"]
        if digest(path) != response["sha256"]:
            raise ValueError(f"Raw response checksum changed: {path}")
        payloads.append(json.loads(path.read_bytes()))
    return payloads, manifest


def prepare(snapshot: Path, config: dict, processed: Path) -> None:
    payloads, manifest = load_snapshot(snapshot, config)
    events, audit = normalize_catalog(payloads, config)
    daily = make_quake_features(events, config["start_date"], config["end_date"],
                                config["lookback_days"], config["strong_magnitude"])
    source = ROOT / "data/raw/train.csv"
    sales_hash = digest(source)
    sales = aggregate_sales(source)
    joined = sales.merge(daily, on="date", how="left", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all() or joined["quake_count_lag1"].isna().any():
        raise ValueError("Missing external calendar coverage in sales join")
    if digest(source) != sales_hash:
        raise ValueError("Sales source changed during read-only aggregation")
    processed.mkdir(parents=True, exist_ok=True)
    events.to_csv(processed / "events.csv", index=False)
    daily.to_csv(processed / "daily_calendar.csv", index=False)
    joined.drop(columns="_merge").to_csv(processed / "daily_sales_external_join.csv", index=False)
    study = events.loc[events.local_date.ge(pd.Timestamp(config["start_date"]))]
    audit.update(events_including_lookback=len(events), study_events=len(study),
                 study_calendar_days=len(daily), sales_days_joined=len(joined),
                 labeled_sales_rows=int(joined.sales_rows.sum()), missing_sales_calendar_days=len(daily)-len(joined),
                 min_magnitude=float(study.magnitude.min()), max_magnitude=float(study.magnitude.max()),
                 study_days_with_event=int(daily.quake_count.gt(0).sum()),
                 source_sales_sha256=sales_hash, source_manifest_sha256=digest(snapshot / "manifest.json"),
                 processed_sha256={name: digest(processed / name) for name in
                                   ("events.csv", "daily_calendar.csv", "daily_sales_external_join.csv")},
                 model_feature_names=MODEL_FEATURES,
                 origin_time_availability="Known only after occurrence; latest catalog revisions are retrospective",
                 historical_first_publication="NOT VERIFIED")
    save_json(processed / "quality_audit.json", audit)
    print(f"Prepared {len(events)} events; {len(daily)} calendar days; joined {int(joined.sales_rows.sum()):,} sales rows via {len(joined)} dates", flush=True)


def table(frame: pd.DataFrame) -> str:
    def cell(value):
        if pd.isna(value):
            return "N/A"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |",
                      *("| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None))])


def report(snapshot: Path, config: dict, processed: Path, output: Path) -> None:
    _, manifest = load_snapshot(snapshot, config)
    audit = json.loads((processed / "quality_audit.json").read_text(encoding="utf-8"))
    if audit["source_manifest_sha256"] != digest(snapshot / "manifest.json"):
        raise ValueError("Prepared data manifest differs from current snapshot")
    if audit["source_sales_sha256"] != digest(ROOT / "data/raw/train.csv"):
        raise ValueError("Sales source differs from prepared collection join")
    for name, expected in audit["processed_sha256"].items():
        if digest(processed / name) != expected:
            raise ValueError(f"Prepared collection artifact changed: {name}")
    events = pd.read_csv(processed / "events.csv", parse_dates=["local_date", "event_time_utc", "updated_at_utc"])
    daily = pd.read_csv(processed / "daily_calendar.csv", parse_dates=["date"])
    joined = pd.read_csv(processed / "daily_sales_external_join.csv", parse_dates=["date"])
    train = joined.loc[joined.date <= "2015-12-31"].copy()
    train["previous_day_event"] = train.quake_count_lag1.gt(0)
    association = train.groupby("previous_day_event").agg(days=("date", "size"),
                 mean_daily_sales=("sales_mean", "mean"), mean_daily_log_sales=("log_sales_mean", "mean"),
                 mean_zero_rate=("zero_rate", "mean")).reset_index()
    annual = daily.assign(year=daily.date.dt.year).groupby("year").agg(events=("quake_count", "sum"),
             event_days=("quake_count", lambda x: int(x.gt(0).sum())), max_magnitude=("quake_max_magnitude", "max"),
             strong_events=("strong_quake_count", "sum")).reset_index()
    top = events.loc[events.local_date >= config["start_date"]].nlargest(10, "magnitude")
    output.mkdir(parents=True, exist_ok=True)
    association.to_csv(output / "train_association.csv", index=False)
    annual.to_csv(output / "annual_collection.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    monthly = daily.set_index("date").quake_count.resample("MS").sum()
    axes[0].plot(monthly.index, monthly, color="#176b87")
    axes[0].set(xlabel="Local mainland Ecuador date (UTC-05)", ylabel="Catalog events / month",
                title="USGS retrospective collection: M >= 4.5, Ecuador-area rectangle")
    axes[1].scatter(events.local_date, events.magnitude, s=12, alpha=.6, color="#ad483c")
    axes[1].set(xlabel="Local event date (UTC-05)", ylabel="Reported magnitude (mixed magnitude types)",
                title="Collected events; event timing is not a sales regime-change label")
    fig.savefig(output / "collection_overview.png", dpi=140)
    plt.close(fig)
    responses = pd.DataFrame([{k:r[k] for k in ("year", "response_event_count", "bytes", "retrieved_at_utc", "sha256")} for r in manifest["responses"]])
    def rel(path):
        return "../" + path.relative_to(ROOT).as_posix()
    text = rf"""# Thu thập dữ liệu web bổ sung cho bài toán Favorita

## Vai trò trong bài tập lớn

Dự án tự viết chương trình thu thập qua HTTP API công khai của USGS, lưu GeoJSON gốc, chuẩn hóa schema/thời gian, kiểm tra dữ liệu và ghép với doanh số. Đây là **thu thập web qua API**, không phải cào DOM HTML. Nhãn doanh số vẫn đến từ dataset Favorita có sẵn; phần tự thu thập là dữ liệu động đất ngoại sinh. Không mô tả toàn bộ nhãn sales là dữ liệu do nhóm tự cào. Bộ dữ liệu mới chưa tham gia các model/số liệu Task 01–05 đã khóa.

## Nguồn và phạm vi đã thực hiện

- Nguồn: [USGS Earthquake Catalog API]({config['documentation']}); [schema GeoJSON](https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php).
- Cửa sổ nghiên cứu: {config['start_date']}–{config['end_date']}; thu thêm {config['lookback_days']} ngày trước đó để tính lag/rolling đầu kỳ.
- Hình chữ nhật: latitude [{config['bbox']['minlatitude']}, {config['bbox']['maxlatitude']}], longitude [{config['bbox']['minlongitude']}, {config['bbox']['maxlongitude']}], magnitude >= {config['min_magnitude']}. Vùng này gồm Ecuador lục địa, vùng biển và một phần nước lân cận; không gán mọi event là nằm trong biên giới Ecuador. Không đại diện Galapagos hay mức rung tại từng cửa hàng.
- {len(manifest['responses'])} truy vấn theo năm, timeout {config['timeout_seconds']} giây, tối đa {config['attempts']} lần khi lỗi mạng/429/5xx, nghỉ {config['request_interval_seconds']} giây giữa truy vấn. Cache kiểm tra checksum; chạy prepare/report không cần tải lại. Từ chối response lỗi schema/count hoặc chạm giới hạn 20.000 events để tránh cắt dữ liệu âm thầm.

{table(responses)}

Manifest lưu đầy đủ URL, HTTP status, thời điểm truy xuất UTC, dung lượng, SHA-256 và cấu hình. [Manifest]({rel(snapshot/'manifest.json')}).

## Kết quả tiền xử lý và kiểm tra

Thu được **{audit['events_including_lookback']} events** kể cả lookback; **{audit['study_events']} events** trong kỳ nghiên cứu; **{audit['study_days_with_event']} ngày có ít nhất một event** trong {audit['study_calendar_days']} ngày lịch. Có {audit['duplicate_ids_removed']} ID lặp được loại theo revision mới nhất và {audit['outside_local_window_removed']} hàng ngoài cửa sổ được loại. Magnitude nghiên cứu từ {audit['min_magnitude']:.1f} đến {audit['max_magnitude']:.1f}.

Thời gian gốc milliseconds UTC được chuyển về UTC-05:00 trước khi lấy ngày. Magnitude/coordinates/time/ID phải hợp lệ; `updated_at_utc` được giữ nguyên và **không coi là thời điểm công bố đầu tiên**. API hiện tại có thể không trả `metadata.count`; khi có trường này phải khớp số features, còn manifest luôn đếm số features thực nhận. Các magnitude types khác nhau được giữ trong bảng, chưa đồng nhất sang một thang duy nhất. Ngày không có event trong kết quả truy vấn nhận count=0; magnitude ngày đó để missing. Count=0 chỉ nói không có bản ghi thỏa truy vấn, không chứng minh không có động đất hoặc catalog đầy đủ vật lý.

[Events chuẩn hóa]({rel(processed/'events.csv')}); [calendar/feature theo ngày]({rel(processed/'daily_calendar.csv')}); [quality audit]({rel(processed/'quality_audit.json')}).

## Ghép dữ liệu và feature

Ghép many-to-one theo ngày nếu dùng từng dòng sales; bản minh họa cộng doanh số theo ngày rồi ghép one-to-one, phủ **{audit['labeled_sales_rows']:,} nhãn doanh số qua {audit['sales_days_joined']} ngày**. Có {audit['missing_sales_calendar_days']} ngày lịch không có bản ghi sales, giữ chúng là thiếu doanh số. [Bảng ghép theo ngày]({rel(processed/'daily_sales_external_join.csv')}). Không nhân bản hay thay đổi các nhãn gốc. SHA-256 train.csv trước/sau: `{audit['source_sales_sha256']}`.

Đặt $C_t$ là số event theo local date, $S_t$ là số event magnitude >= {config['strong_magnitude']}, $M_t$ là magnitude lớn nhất ngày đó. Feature đề xuất:

$$x'_{{g,t}}=[x_{{g,t}}, C_{{t-1}}, S_{{t-1}}, M_{{t-1}}, \sum_{{j=1}}^7 C_{{t-j}}].$$

`MODEL_FEATURES = {MODEL_FEATURES}`. Các cột event cùng ngày chỉ phục vụ EDA, không nằm trong danh sách feature mô hình. Chỉ mục này dùng thông tin sự kiện trước ngày dự báo, nhưng **availability lịch sử vẫn NOT VERIFIED**: catalog lấy hôm nay đã được revision, không có snapshot/bản tin đầu tiên tại từng ngày 2013–2017. Vì thế feature trễ hiện chỉ sẵn sàng cho ablation hồi cứu/exploratory; không đủ chứng minh pipeline vận hành as-of không leakage. Magnitude cùng ngày, thiệt hại tổng kết sau sự kiện và posterior jump không phải feature tiên tri.

## Phân tích mô tả

{table(annual)}

![Số lượng và magnitude đã thu thập]({rel(output/'collection_overview.png')})

Các event magnitude lớn nhất (mô tả nguồn, không chọn model hoặc định nghĩa jump ground truth):

{table(top[['event_id','local_date','magnitude','magnitude_type','depth_km']])}

Liên hệ doanh số chỉ tính trên **train đến 2015-12-31**, chưa điều chỉnh weekday, mùa vụ, promotion hoặc aftershock clustering:

{table(association)}

Bảng này là association thô, không kết luận động đất gây biến động sales. Dữ liệu mới không được dùng để chọn lại model theo test đã xem. Ablation tương lai cần so cùng model/seed/split với và không có các feature này; báo cáo cả RMSLE, MAE và mức coverage/availability của nguồn. Chưa có số liệu chứng minh tăng accuracy từ dữ liệu mới.

## Tái lập

```powershell
./.venv/Scripts/python.exe -m scripts.collect_external_data --stage collect --snapshot {snapshot.relative_to(ROOT).as_posix()}
./.venv/Scripts/python.exe -m scripts.collect_external_data --stage prepare --snapshot {snapshot.relative_to(ROOT).as_posix()}
./.venv/Scripts/python.exe -m scripts.collect_external_data --stage report --snapshot {snapshot.relative_to(ROOT).as_posix()}
```

Mục tiêu học phần được minh họa bằng chuỗi **thu thập web → raw snapshot → tiền xử lý/QA → ghép nguồn → EDA → thiết kế feature và ablation ML**. API collection và HTML scraping là hai cách khác nhau; cần mô tả đúng cách đã thực hiện. Không đảm bảo tiêu chí chấm khi chưa có rubric chính thức yêu cầu tự thu thập toàn bộ target.
"""
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/DATA_COLLECTION.md").write_text(text, encoding="utf-8")
    print(f"Collection report: {ROOT / 'reports/DATA_COLLECTION.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("collect", "prepare", "report", "all"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    processed = ROOT / "data/processed/external/usgs_ecuador" / snapshot.name
    output = ROOT / "results/data_collection/usgs_ecuador" / snapshot.name
    if args.stage in ("collect", "all"):
        collect(snapshot, config)
    if args.stage in ("prepare", "all"):
        prepare(snapshot, config, processed)
    if args.stage in ("report", "all"):
        report(snapshot, config, processed, output)


if __name__ == "__main__":
    main()
