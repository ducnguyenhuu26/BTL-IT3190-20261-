"""Full Favorita 2x2 ID/USGS ablation, validation blends, and one locked test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.collect_external_data import CONFIG as COLLECTION_CONFIG, load_snapshot
from scripts.run_full_favorita import prepare, score, sha256
from src.external_data import MODEL_FEATURES, join_quake_features

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/usgs_ablation.json"
VARIANTS = (("LightGBM", False, False), ("LightGBM + USGS", False, True),
            ("LightGBM + IDs", True, False), ("LightGBM + IDs + USGS", True, True))
BLENDS = (("Ensemble (base, USGS)", "LightGBM", "LightGBM + USGS"),
          ("Ensemble (IDs, IDs+USGS)", "LightGBM + IDs", "LightGBM + IDs + USGS"))
NAMES = ("Seasonal naive", *(name for name, _, _ in VARIANTS), *(name for name, _, _ in BLENDS))
CODE = ("scripts/run_usgs_ablation.py", "scripts/run_full_favorita.py", "scripts/collect_external_data.py", "src/data.py",
        "src/features.py", "src/external_data.py", "configs/usgs_ablation.json")


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def input_hashes(config: dict) -> dict[str, str]:
    snapshot = ROOT / config["snapshot"]
    collection = json.loads(COLLECTION_CONFIG.read_text(encoding="utf-8"))
    _, manifest = load_snapshot(snapshot, collection)
    processed = ROOT / "data/processed/external/usgs_ecuador" / snapshot.name
    audit = json.loads((processed / "quality_audit.json").read_text(encoding="utf-8"))
    if audit["source_manifest_sha256"] != sha256(snapshot / "manifest.json"):
        raise ValueError("USGS preprocessing belongs to a different raw snapshot")
    for name, expected in audit["processed_sha256"].items():
        if sha256(processed / name) != expected:
            raise ValueError(f"USGS processed checksum mismatch: {name}")
    if sha256(ROOT / "data/raw/train.csv") != audit["source_sales_sha256"]:
        raise ValueError("Favorita source differs from the collected-data join")
    paths = [ROOT / "data/raw/train.csv", COLLECTION_CONFIG, snapshot / "manifest.json",
             processed / "quality_audit.json", processed / "daily_calendar.csv"]
    for response in manifest["responses"]:
        paths.extend((snapshot / response["file"], snapshot / response["metadata_file"]))
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def code_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in CODE}


def prepare_joined(config: dict, through=None) -> tuple[pd.DataFrame, list[str]]:
    frame, base_cols = prepare(through)
    processed = ROOT / "data/processed/external/usgs_ecuador" / Path(config["snapshot"]).name
    calendar = pd.read_csv(processed / "daily_calendar.csv", parse_dates=["date"])
    frame = join_quake_features(frame, calendar)
    for col in MODEL_FEATURES:
        frame[col] = frame[col].astype("float32")
    frame["store_nbr"] = pd.Categorical(frame.store_nbr, categories=sorted(frame.store_nbr.unique()))
    # Preserve the earlier full-run training row order for matched histogram
    # construction; only online validation/test predictions need date sorting.
    return frame, base_cols


def best_blend(actual_log: np.ndarray, base_log: np.ndarray, augmented_log: np.ndarray) -> float:
    """Closed-form validation MSLE optimum, constrained to a convex blend."""
    actual_log, base_log, augmented_log = map(lambda a: np.asarray(a, dtype=float),
                                             (actual_log, base_log, augmented_log))
    if not (actual_log.shape == base_log.shape == augmented_log.shape) or actual_log.ndim != 1:
        raise ValueError("Blend arrays must be aligned 1D predictions")
    if not all(np.isfinite(a).all() for a in (actual_log, base_log, augmented_log)):
        raise ValueError("Non-finite blend inputs")
    direction = augmented_log - base_log
    denominator = np.dot(direction, direction)
    return 0. if denominator <= 1e-20 else float(np.clip(np.dot(actual_log-base_log, direction)/denominator, 0, 1))


def prediction_frame(frame: pd.DataFrame, logs: dict[str, np.ndarray], blends: dict) -> pd.DataFrame:
    wide = frame[["date", "store_nbr", "family"]].copy()
    # Preserve the full float64 representation of the pipeline's float32 target
    # when writing CSV, so recomputing metrics from the artifact is exact.
    wide["actual"] = frame.sales.to_numpy(float)
    lag7, lag1 = frame.sales_lag7.to_numpy(float), frame.sales_lag1.to_numpy(float)
    wide["Seasonal naive"] = np.maximum(0, np.where(np.isfinite(lag7), lag7, np.where(np.isfinite(lag1), lag1, 0)))
    for name, log_values in logs.items():
        wide[name] = np.expm1(log_values)
    for name, left, right in BLENDS:
        beta = blends[name]["beta"]
        wide[name] = np.expm1((1-beta)*logs[left] + beta*logs[right])
    if not np.isfinite(wide[list(NAMES)].to_numpy(float)).all():
        raise ValueError("Non-finite forecast in ablation")
    return wide


def metrics_from_predictions(wide: pd.DataFrame) -> pd.DataFrame:
    if wide.duplicated(["date", "store_nbr", "family"]).any() or wide.actual.isna().any():
        raise ValueError("Duplicate prediction keys or missing Favorita labels")
    frame = wide.rename(columns={"actual": "sales"}).copy()
    frame["family"] = frame.family.astype("category")
    return pd.DataFrame([{"model": name, **score(frame, wide[name].to_numpy(float))} for name in NAMES])


def validate(config: dict, output: Path) -> None:
    lock_path = output / "selected_configs.json"
    if lock_path.exists():
        raise FileExistsError("Ablation already locked; preserve this run and its first test")
    initial_inputs, initial_code = input_hashes(config), code_hashes()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame, base_cols = prepare_joined(config, pd.Timestamp(config["validation_end"]))
    train = frame.loc[frame.date <= config["train_end"]].copy()
    val = frame.loc[frame.date > config["train_end"]].sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    if len(train) != 1945944 or len(val) != 162162:
        raise ValueError("Unexpected train/validation full-cohort coverage")
    feature_rows = []
    for split, part in (("train", train), ("validation", val)):
        for col in [*base_cols, "store_nbr", "family", *MODEL_FEATURES]:
            feature_rows.append({"split": split, "feature": col, "dtype": str(part[col].dtype),
                                 "rows": len(part), "missing": int(part[col].isna().sum())})
    pd.DataFrame(feature_rows).to_csv(output / "preprocessing_audit.csv", index=False)
    del frame
    gc.collect()
    models, logs = {}, {}
    for name, identifiers, external in VARIANTS:
        cols = base_cols + (["store_nbr", "family"] if identifiers else []) + (MODEL_FEATURES if external else [])
        categories = {col: train[col].cat.categories.tolist() for col in cols
                      if isinstance(train[col].dtype, pd.CategoricalDtype)}
        model = lgb.LGBMRegressor(objective="regression", **config["model"], random_state=config["seed"],
                                  n_jobs=config["num_threads"], verbosity=-1)
        begin = time.perf_counter()
        model.fit(train[cols], np.log1p(train.sales.to_numpy(float)), categorical_feature=list(categories),
                  eval_set=[(val[cols], np.log1p(val.sales.to_numpy(float)))],
                  callbacks=[lgb.early_stopping(config["early_stopping_rounds"], verbose=False), lgb.log_evaluation(100)])
        # Clip in log space before convex mixing; matches nonnegative output.
        logs[name] = np.maximum(0, model.predict(val[cols], num_threads=config["num_threads"]))
        filename = "lightgbm_" + str(len(models)) + ".txt"
        model.booster_.save_model(str(output / filename))
        models[name] = {"file": filename, "sha256": sha256(output / filename), "features": cols,
                        "categories": categories, "best_iteration": int(model.best_iteration_),
                        "fit_seconds": time.perf_counter()-begin}
        print(f"Validation {name}: features={len(cols)}, trees={model.best_iteration_}, "
              f"RMSLE={score(val, np.expm1(logs[name]))['rmsle']:.6f}", flush=True)
        del model
        gc.collect()
    blends = {name: {"left": left, "right": right,
                     "beta": best_blend(np.log1p(val.sales.to_numpy(float)), logs[left], logs[right])}
              for name, left, right in BLENDS}
    wide = prediction_frame(val, logs, blends)
    metrics = metrics_from_predictions(wide)
    # Source/model validation selection is finalized before any test stage.
    if input_hashes(config) != initial_inputs or code_hashes() != initial_code:
        raise ValueError("Input/code changed during validation")
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    wide.to_csv(output / "validation_predictions.csv.gz", index=False, compression="gzip")
    selected = min(enumerate(metrics.itertuples()), key=lambda pair: (pair[1].rmsle, pair[0]))[1].model
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
            "input_hashes": initial_inputs, "code_hashes": initial_code, "models": models, "blends": blends,
            "selected_on_validation": selected, "validation_seconds": time.perf_counter()-started,
            "validation_predictions_sha256": sha256(output / "validation_predictions.csv.gz"),
            "validation_metrics_sha256": sha256(output / "validation_metrics.csv"),
            "preprocessing_audit_sha256": sha256(output / "preprocessing_audit.csv"),
            "split_rows": {"train": len(train), "validation": len(val), "test": 892782},
            "availability": "USGS retrospective revisions; historical first publication NOT VERIFIED",
            "protocol": "One-step actual-lag feedback; fixed base models; validation-only convex blends; exploratory test"}
    save_json(lock_path, lock)
    print(f"Locked validation winner: {selected}; blend weights={blends}", flush=True)


def read_lock(config: dict, output: Path) -> dict:
    lock = json.loads((output / "selected_configs.json").read_text(encoding="utf-8"))
    if lock["config"] != config or lock["input_hashes"] != input_hashes(config) or lock["code_hashes"] != code_hashes():
        raise ValueError("Ablation config/input/code changed after validation lock")
    for model in lock["models"].values():
        if sha256(output / model["file"]) != model["sha256"]:
            raise ValueError("Locked LightGBM model checksum changed")
    for filename, key in (("validation_metrics.csv", "validation_metrics_sha256"),
                          ("validation_predictions.csv.gz", "validation_predictions_sha256"),
                          ("preprocessing_audit.csv", "preprocessing_audit_sha256")):
        if sha256(output / filename) != lock[key]:
            raise ValueError(f"Locked validation artifact changed: {filename}")
    return lock


def test(config: dict, output: Path) -> None:
    if any((output / name).exists() for name in ("test_started.json", "test_predictions.csv.gz", "test_metrics.csv")):
        raise FileExistsError("Ablation test already started/evaluated; use --stage report on saved predictions")
    lock = read_lock(config, output)
    started = time.perf_counter()
    frame, _ = prepare_joined(config)
    part = frame.loc[frame.date > config["validation_end"]].sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    del frame
    gc.collect()
    if len(part) != 892782 or part[["store_nbr", "family"]].drop_duplicates().shape[0] != 1782:
        raise ValueError("Unexpected full-cohort test coverage")
    marker = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
              "lock_sha256": sha256(output / "selected_configs.json")}
    with (output / "test_started.json").open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2)
    logs = {}
    for name, model in lock["models"].items():
        data = part[model["features"]].copy()
        for col, categories in model["categories"].items():
            data[col] = data[col].cat.set_categories(categories)
        booster = lgb.Booster(model_file=str(output / model["file"]))
        logs[name] = np.maximum(0, booster.predict(data, num_threads=config["num_threads"]))
        print(f"Test predicted {name}: {len(part):,} rows", flush=True)
        del booster, data
        gc.collect()
    wide = prediction_frame(part, logs, lock["blends"])
    path = output / "test_predictions.csv.gz"
    temporary = output / "test_predictions.partial.csv.gz"
    wide.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)
    save_json(output / "test_prediction_manifest.json", {"sha256": sha256(path),
              "lock_sha256": marker["lock_sha256"], "rows": len(wide),
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter()-started})
    del part, logs, wide
    gc.collect()
    build_report(config, output)


def intervals(wide: pd.DataFrame, config: dict) -> pd.DataFrame:
    block = ((wide.date-wide.date.min()).dt.days // config["bootstrap"]["block_days"]).to_numpy(int)
    count = np.bincount(block)
    actual_log = np.log1p(wide.actual.to_numpy(float))
    squares = {name: np.bincount(block, weights=(np.log1p(wide[name])-actual_log)**2) for name in NAMES}
    draws = np.random.default_rng(config["seed"]).integers(0, len(count),
            size=(config["bootstrap"]["repetitions"], len(count)))
    rmsles = {name: np.sqrt(values[draws].sum(axis=1)/count[draws].sum(axis=1)) for name, values in squares.items()}
    pairs = [(name, "LightGBM") for name in NAMES if name != "LightGBM"]
    pairs += [("LightGBM + IDs + USGS", "LightGBM + IDs"),
              ("LightGBM + IDs + USGS", "LightGBM + USGS"),
              ("Ensemble (IDs, IDs+USGS)", "LightGBM + IDs"),
              ("Ensemble (base, USGS)", "LightGBM + USGS")]
    rows = []
    for name, ref in pairs:
        delta = rmsles[name]-rmsles[ref]
        lo, hi = np.quantile(delta, [.025, .975])
        point = np.sqrt(squares[name].sum()/count.sum())-np.sqrt(squares[ref].sum()/count.sum())
        rows.append({"model": name, "reference": ref, "delta_rmsle": point, "ci_low": lo, "ci_high": hi,
                     "blocks": len(count), **config["bootstrap"], "seed": config["seed"]})
    return pd.DataFrame(rows)


def markdown(data: pd.DataFrame, digits=6) -> str:
    def fmt(value):
        if pd.isna(value):
            return "N/A"
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(data.columns) + " |", "| " + " | ".join(["---"]*len(data.columns)) + " |",
                      *("| " + " | ".join(map(fmt, row)) + " |" for row in data.itertuples(index=False, name=None))])


def build_report(config: dict, output: Path) -> None:
    lock = read_lock(config, output)
    prediction_meta = json.loads((output / "test_prediction_manifest.json").read_text(encoding="utf-8"))
    if prediction_meta["lock_sha256"] != sha256(output / "selected_configs.json") or prediction_meta["sha256"] != sha256(output / "test_predictions.csv.gz"):
        raise ValueError("Saved test prediction checksum/lock mismatch")
    wide = pd.read_csv(output / "test_predictions.csv.gz", parse_dates=["date"])
    if len(wide) != 892782 or wide.date.min() != pd.Timestamp("2016-04-01") or wide.date.max() != pd.Timestamp("2017-08-15"):
        raise ValueError("Saved test prediction coverage differs from locked protocol")
    metrics = metrics_from_predictions(wide)
    ci = intervals(wide, config)
    metrics.to_csv(output / "test_metrics.csv", index=False)
    ci.to_csv(output / "paired_bootstrap.csv", index=False)
    validation = pd.read_csv(output / "validation_metrics.csv")
    validation_wide = pd.read_csv(output / "validation_predictions.csv.gz", parse_dates=["date"])
    rechecked = metrics_from_predictions(validation_wide)
    if not np.allclose(validation[["rmsle", "mae"]], rechecked[["rmsle", "mae"]], rtol=0, atol=1e-9):
        raise ValueError("Saved validation metrics do not recompute")
    comparison = validation[["model", "rmsle"]].merge(metrics, on="model", suffixes=("_validation", "_test"))
    comparison.to_csv(output / "comparison.csv", index=False)
    ablation = comparison.loc[comparison.model != "Seasonal naive"].copy()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    y = np.arange(len(ablation))
    ax.scatter(ablation.rmsle_validation, y+.12, label="Validation", marker="o")
    ax.scatter(ablation.rmsle_test, y-.12, label="Test exploratory", marker="s")
    ax.set_yticks(y, ablation.model)
    ax.invert_yaxis()
    ax.set(xlabel="RMSLE (lower is better)", title="Full Favorita: paired ID / retrospective USGS ablation")
    ax.legend()
    ax.grid(axis="x", alpha=.2)
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)
    blend_rows = pd.DataFrame([{"model": name, **value} for name, value in lock["blends"].items()])
    model_rows = pd.DataFrame([{"model": name, "features": len(value["features"]),
                 "best_iteration": value["best_iteration"], "fit_seconds": value["fit_seconds"]}
                 for name, value in lock["models"].items()])
    source_rows = pd.DataFrame([{"input": name, "SHA-256": value} for name, value in lock["input_hashes"].items()])
    index = metrics.set_index("model")
    delta_plain = index.loc["LightGBM + USGS", "rmsle"]-index.loc["LightGBM", "rmsle"]
    delta_ids = index.loc["LightGBM + IDs + USGS", "rmsle"]-index.loc["LightGBM + IDs", "rmsle"]
    chosen = lock["selected_on_validation"]
    text = rf"""# Train/test toàn bộ Favorita với dữ liệu USGS tự thu thập

Lượt này dùng toàn bộ **1.782 chuỗi / 3.000.888 nhãn sales**, bốn LightGBM theo thiết kế 2×2 và hai ensemble. Đây là **hồi cứu/exploratory** vì giai đoạn test đã được xem và USGS là danh mục có revision, chưa xác minh thời điểm công bố lịch sử. [Nguồn và bước thu thập](DATA_COLLECTION.md).

## Tiền xử lý và split

Giữ nguyên feature sales/lịch/promotion của pipeline cũ. Chỉ ghép allowlist gồm `{MODEL_FEATURES}`, many-to-one theo ngày, kiểm tra phủ đầy đủ và không tăng số nhãn. Lag sales theo đúng ngày lịch; rolling chỉ dùng quá khứ. Động đất chuyển UTC về UTC-05 Ecuador lục địa trước khi lấy ngày. Count=0 là không có bản ghi thỏa truy vấn, magnitude ngày không có event để NaN; LightGBM xử lý NaN. Chỉ thêm store_nbr/family dạng categorical ở hai nhánh IDs; không target encoding hay fit scaler trên test. Không dùng cột doanh số tổng hợp của bảng EDA hoặc event cùng ngày làm feature.

$$x'_{{g,t}}=[x_{{g,t}},C_{{t-1}},S_{{t-1}},M_{{t-1}},sum_{{j=1}}^7 C_{{t-j}}].$$

Train 2013-01-01–2015-12-31: **1.945.944** dòng; validation 2016-01-01–03-31: **162.162**; test 2016-04-01–2017-08-15: **892.782**. Tất cả chuỗi dùng chung một LightGBM mỗi variant. Actual-lag feedback trong test chỉ dùng nhãn trước ngày dự báo. Promotion cùng ngày vẫn cần giả định đã lên kế hoạch. Feature USGS có nhân quả theo **origin time**, nhưng không khẳng định as-of availability của revision.

## Train và khóa validation

Seed {config['seed']}, {config['num_threads']} CPU threads; objective squared error trên log1p(sales); tối đa {config['model']['n_estimators']} cây, learning rate {config['model']['learning_rate']}, tối đa {config['model']['num_leaves']} lá/cây; early stopping {config['early_stopping_rounds']} vòng, chung ngân sách cho mọi variant. Validation dùng chung cho early stopping và chọn ensemble nên chưa là đánh giá độc lập. Không có lần retrain sau khi xem test.

{markdown(model_rows, 2)}

Hai ensemble trộn dự báo đã chặn không âm trên thang log:

$$hat z=(1-eta)hat z_A+etahat z_B,qquad
eta^star=operatorname{{clip}}_{{[0,1]}}rac{{sum_{{val}}(z-hat z_A)(hat z_B-hat z_A)}}{{sum_{{val}}(hat z_B-hat z_A)^2}}.$$

{markdown(blend_rows)}

Validation chọn **{chosen}** theo RMSLE, tie break theo thứ tự candidate cố định. Khóa UTC `{lock['locked_at_utc']}` được ghi trước test, gồm hash code/config/input/model và trọng số ensemble. Test có marker chống lặp; report stage chỉ tính lại từ predictions đã lưu. Thời gian validation tổng {lock['validation_seconds']:.1f} giây; test preprocessing/predict/save {prediction_meta['seconds']:.1f} giây.

## Kết quả

{markdown(comparison)}

![So sánh validation và test](../{output.relative_to(ROOT).as_posix()}/comparison.png)

Tác dụng riêng của USGS trên test: ΔRMSLE **USGS − không USGS** = **{delta_plain:+.6f}** khi chưa có IDs và **{delta_ids:+.6f}** khi đã có IDs. Dấu âm là giảm sai số; dấu dương là tăng sai số. Dùng cặp có cùng IDs để tách ảnh hưởng dữ liệu thu thập khỏi ảnh hưởng nhận diện chuỗi. Metric trung bình theo chuỗi được báo cùng pooled RMSLE/MAE. Chỉ mô hình chọn trên validation được coi là quyết định của lượt này; không chọn lại theo thứ hạng test.

## Paired uncertainty

Bootstrap {config['bootstrap']['repetitions']} lần, block {config['bootstrap']['block_days']} ngày, giữ tất cả chuỗi cùng block, seed {config['seed']}; Δ = model − reference. Interval mô tả test cố định, không hiệu chỉnh multiple comparisons và không chứng minh tác động nhân quả của động đất.

{markdown(ci, 7)}

## Artifact và provenance

Kết quả tại `../{output.relative_to(ROOT).as_posix()}/`: selected_configs.json, preprocessing_audit.csv, validation_metrics.csv, validation_predictions.csv.gz, test_started.json, test_prediction_manifest.json, test_metrics.csv, test_predictions.csv.gz, paired_bootstrap.csv, comparison.csv, comparison.png và 4 LightGBM text models. Source/model/validation hashes đã kiểm tra lại; metric tính lại từ file dự báo. Kết quả các lượt cũ không bị ghi đè.

{markdown(source_rows)}

## Tái lập

```powershell
./.venv/Scripts/python.exe -m scripts.run_usgs_ablation --stage validate
./.venv/Scripts/python.exe -m scripts.run_usgs_ablation --stage test
./.venv/Scripts/python.exe -m scripts.run_usgs_ablation --stage report
```

Hai lệnh đầu từ chối lặp khi đã khóa/đã bắt đầu test. Lệnh report chỉ đọc model và predictions đã khóa. Danh mục động đất vùng chữ nhật không phải ground-truth latent jump của sales, không mô tả mức rung tại từng cửa hàng. Một lượt seed và test đã xem không đủ xác nhận hiệu quả tổng quát hay triển khai realtime.
"""
    (ROOT / "reports/USGS_MODEL_ABLATION.md").write_text(text, encoding="utf-8")
    (output / "report.md").write_text(text.replace("(DATA_COLLECTION.md)", "(../../reports/DATA_COLLECTION.md)")
            .replace("../"+output.relative_to(ROOT).as_posix()+"/comparison.png", "comparison.png"), encoding="utf-8")
    for row in metrics.itertuples():
        print(f"Test {row.model}: RMSLE={row.rmsle:.6f}, MAE={row.mae:.3f}, n={row.n:,}", flush=True)
    print(f"Validation-selected architecture: {chosen}; report={ROOT / 'reports/USGS_MODEL_ABLATION.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("validate", "test", "report"))
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output"]
    {"validate": validate, "test": test, "report": build_report}[args.stage](config, output)


if __name__ == "__main__":
    main()
