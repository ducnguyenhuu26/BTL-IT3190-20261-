"""Read-only evidence audit and BTL report builder from locked local artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import lightgbm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from src.beijing_data import load_beijing

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ED = ROOT / "results/eda"
DIAG = ROOT / "results/diagnostics"
REVIEW = ROOT / "results/model_review"
BEIJING = ROOT / "results/external_validation"
FULL = ROOT / "results/full_favorita_exploratory"

INPUTS = {
    "results/eda/summary.csv": "Favorita full-data coverage and zero counts",
    "results/eda/sales_by_weekday.csv": "Favorita weekday sales summaries",
    "results/eda/sales_by_month.csv": "Favorita monthly sales summaries",
    "results/eda/promotion_summary.csv": "Favorita promotion association",
    "results/eda/series_profile.csv": "Favorita series distribution",
    "results/baselines/metrics.csv": "Archived Favorita EMA and seasonal baselines",
    "results/baselines/predictions.csv.gz": "Archived Favorita baseline predictions",
    "results/diagnostics/leakage_audit.csv": "Favorita timing and leakage audit",
    "results/diagnostics/diagnostics.md": "Task 02 diagnostics and protocol",
    "results/model_review/review.md": "Task 03 interpretation and protocol",
    "results/model_review/selected_configs.json": "Favorita validation selections",
    "results/model_review/validation_grid.csv": "Favorita filter candidate grid",
    "results/model_review/real_validation_comparison.csv": "Favorita selected validation scores",
    "results/model_review/real_test_comparison.csv": "Favorita Task 03 test metrics",
    "results/model_review/test_predictions.csv": "Favorita Task 03 paired predictions",
    "results/model_review/synthetic_comparison.csv": "Synthetic evaluation means",
    "results/model_review/synthetic_seed_comparison.csv": "Synthetic per-seed values",
    "results/model_review/selected_synthetic_thresholds.json": "Synthetic calibration/evaluation seed lists",
    "results/external_validation/report.md": "Task 04 Beijing protocol and interpretation",
    "results/external_validation/selected_configs.json": "Beijing validation lock",
    "results/external_validation/dataset_profile.csv": "Beijing split coverage",
    "results/external_validation/missingness_by_station.csv": "Beijing missingness and invalid values",
    "results/external_validation/missingness_by_month.csv": "Beijing monthly missingness",
    "results/external_validation/station_train_groups.csv": "Train-only Beijing station groups",
    "results/external_validation/group_rmse_delta.csv": "Test RMSE deltas by fixed train-only station group",
    "results/external_validation/rolling_target_residual.csv": "Beijing validation/test residual trends",
    "results/external_validation/validation_metrics.csv": "Beijing validation scores",
    "results/external_validation/validation_grid.csv": "Beijing filter candidate grid",
    "results/external_validation/test_metrics.csv": "Beijing locked test scores",
    "results/external_validation/test_predictions.csv.gz": "Beijing paired test predictions",
    "results/external_validation/paired_bootstrap.csv": "Beijing paired seven-day uncertainty",
    "results/external_validation/beijing_lightgbm.txt": "Locked Beijing LightGBM model",
    "data/raw/beijing/PRSA2017_Data_20130301-20170228.zip": "Official UCI source archive",
    "data/raw/train.csv": "Local Favorita labeled source from mirror noted in README",
}


def table(frame: pd.DataFrame, digits: int = 4) -> str:
    def fmt(value):
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "\\|")
    cols = list(frame.columns)
    return "\n".join(["| " + " | ".join(cols) + " |",
                      "| " + " | ".join("---" for _ in cols) + " |",
                      *("| " + " | ".join(fmt(v) for v in row) + " |"
                        for row in frame.itertuples(index=False, name=None))])


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score(actual: np.ndarray, predicted: np.ndarray, metric: str) -> float:
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual >= 0)
    a, p = actual[valid], np.maximum(predicted[valid], 0)
    if metric == "mae":
        return float(np.mean(np.abs(a - p)))
    error = (np.log1p(a) - np.log1p(p)) if metric == "rmsle" else (a - p)
    return float(np.sqrt(np.mean(error**2)))


def align_predictions(long: pd.DataFrame, key: list[str], actual_col: str,
                      names: list[str]) -> tuple[pd.DataFrame, int]:
    base = long.loc[long.model == names[0], key + [actual_col]].sort_values(key).reset_index(drop=True)
    if base.duplicated(key).any():
        raise ValueError("Duplicate prediction keys")
    wide = base.copy()
    for name in names:
        part = long.loc[long.model == name, key + [actual_col, "prediction"]].sort_values(key).reset_index(drop=True)
        if len(part) != len(base) or not part[key].equals(base[key]):
            raise ValueError(f"Misaligned prediction rows: {name}")
        if not np.allclose(part[actual_col], base[actual_col], equal_nan=True):
            raise ValueError(f"Targets differ across models: {name}")
        wide[name] = part.prediction.to_numpy(float)
    return wide, len(base)


def assert_metric_rows(wide: pd.DataFrame, actual_col: str, saved: pd.DataFrame,
                       metric: str, names: list[str]) -> None:
    for name in names:
        row = saved.loc[saved.model == name].iloc[0]
        a, p = wide[actual_col].to_numpy(float), wide[name].to_numpy(float)
        n = int((np.isfinite(a) & (a >= 0) & np.isfinite(p)).sum())
        if n != int(row.n) or not np.isclose(score(a, p, metric), row[metric], rtol=0, atol=1e-9):
            raise ValueError(f"Saved metric does not recompute: {name}")


def audit() -> dict:
    missing = [path for path in INPUTS if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError("Required artifacts NOT AVAILABLE: " + ", ".join(missing))
    original_hashes = {path: hash_file(ROOT / path) for path in INPUTS}
    favor = pd.read_csv(REVIEW / "real_test_comparison.csv")
    favor_pred = pd.read_csv(REVIEW / "test_predictions.csv")
    f_names = favor.model.tolist()
    fwide, frows = align_predictions(favor_pred, ["date", "store_nbr", "family"], "target", f_names)
    assert_metric_rows(fwide, "target", favor, "rmsle", f_names)
    archived = pd.read_csv(ROOT / "results/baselines/metrics.csv")
    archived_pred = pd.read_csv(ROOT / "results/baselines/predictions.csv.gz",
                                usecols=["date", "store_nbr", "family", "actual", "prediction", "model"])
    a_names = ["seasonal_naive", "lgbm_ema"]
    awide, arows = align_predictions(archived_pred, ["date", "store_nbr", "family"], "actual", a_names)
    assert_metric_rows(awide, "actual", archived, "rmsle", a_names)
    if frows != arows or not fwide[["date", "store_nbr", "family"]].equals(
            awide[["date", "store_nbr", "family"]]):
        raise ValueError("Archived Favorita baselines have different keys")
    if not np.allclose(fwide.target, awide.actual):
        raise ValueError("Archived Favorita targets differ")

    beijing = pd.read_csv(BEIJING / "test_metrics.csv")
    bpred = pd.read_csv(BEIJING / "test_predictions.csv.gz", parse_dates=["timestamp"])
    b_names = beijing.model.tolist()
    bwide, brows = align_predictions(bpred, ["timestamp", "station"], "actual", b_names)
    assert_metric_rows(bwide, "actual", beijing, "rmse", b_names)
    if not all((bwide.timestamp.min() == pd.Timestamp("2016-07-01"),
                bwide.timestamp.max() == pd.Timestamp("2017-02-28 23:00"))):
        raise ValueError("Unexpected Beijing test coverage")

    fconfig = json.loads((REVIEW / "selected_configs.json").read_text(encoding="utf-8"))["selected"]
    block = json.loads((BEIJING / "selected_configs.json").read_text(encoding="utf-8"))
    bconfig = block["selected_configs"]
    if hash_file(BEIJING / "beijing_lightgbm.txt") != block["model_sha256"] or hash_file(
            ROOT / "data/raw/beijing/PRSA2017_Data_20130301-20170228.zip") != block["source_zip_sha256"]:
        raise ValueError("Beijing model/source changed after validation lock")
    ci = pd.read_csv(BEIJING / "paired_bootstrap.csv")
    rkl = ci.loc[(ci.model == "LightGBM + Robust Kalman") & (ci.reference == "LightGBM")].iloc[0]
    point = float(beijing.set_index("model").loc["LightGBM + Robust Kalman", "rmse"] -
                  beijing.set_index("model").loc["LightGBM", "rmse"])
    if not np.isclose(point, rkl.delta_rmse, atol=1e-9) or rkl.ci_low > point or rkl.ci_high < point:
        raise ValueError("Beijing CI sign or point difference is inconsistent")
    equal = {
        "beijing_jump_vs_kalman": float(np.max(np.abs(bwide["LightGBM + Jump Filter"] - bwide["LightGBM + Kalman"]))),
        "beijing_robustjump_vs_robustkalman": float(np.max(np.abs(bwide["LightGBM + Robust Jump Filter"] - bwide["LightGBM + Robust Kalman"]))),
        "favorita_jump_vs_kalman": float(np.max(np.abs(fwide["Jump Filter"] - fwide["Kalman"]))),
        "favorita_robustjump_vs_robustkalman": float(np.max(np.abs(fwide["Robust Jump Filter"] - fwide["Robust Kalman"]))),
    }
    if equal["beijing_jump_vs_kalman"] != 0 or equal["beijing_robustjump_vs_robustkalman"] != 0:
        raise ValueError("Beijing p_jump=0 counterparts should match")
    if bconfig["LightGBM + Jump Filter"]["p_jump"] != 0 or bconfig["LightGBM + Robust Jump Filter"]["p_jump"] != 0:
        raise ValueError("Beijing jump selection differs from expected lock")
    synthetic = pd.read_csv(REVIEW / "synthetic_comparison.csv")
    seeds = pd.read_csv(REVIEW / "synthetic_seed_comparison.csv")
    thresholds = json.loads((REVIEW / "selected_synthetic_thresholds.json").read_text(encoding="utf-8"))
    if set(seeds.seed.unique()) != set(thresholds["evaluation_seeds"]):
        raise ValueError("Synthetic evaluation seeds differ from metadata")
    if not set(thresholds["calibration_seeds"]).isdisjoint(thresholds["evaluation_seeds"]):
        raise ValueError("Synthetic calibration/evaluation seeds overlap")
    return locals()


def audit_full() -> dict | None:
    """Audit the separately locked full-cohort run, when it is complete."""
    required = ("selected_configs.json", "lightgbm_full.txt", "validation_grid.csv",
                "validation_metrics.csv", "test_metrics.csv", "test_predictions.csv.gz",
                "paired_bootstrap.csv", "report.md")
    if not FULL.exists():
        return None
    absent = [name for name in required if not (FULL / name).is_file()]
    if absent:
        raise FileNotFoundError("Incomplete full Favorita run: " + ", ".join(absent))
    original_hashes = {name: hash_file(FULL / name) for name in required}
    lock = json.loads((FULL / "selected_configs.json").read_text(encoding="utf-8"))
    if (hash_file(ROOT / "data/raw/train.csv") != lock["source_sha256"] or
            hash_file(FULL / "lightgbm_full.txt") != lock["model_sha256"]):
        raise ValueError("Full Favorita source/model differs from validation lock")
    val = pd.read_csv(FULL / "validation_metrics.csv")
    grid = pd.read_csv(FULL / "validation_grid.csv")
    for name, count in lock["candidate_counts"].items():
        if len(grid.loc[grid.model == name]) != count:
            raise ValueError(f"Incomplete validation grid for {name}")
    metrics = pd.read_csv(FULL / "test_metrics.csv")
    wide = pd.read_csv(FULL / "test_predictions.csv.gz", parse_dates=["date"])
    names = metrics.model.tolist()
    if len(wide) != 892782 or wide.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError("Full Favorita test keys/coverage invalid")
    if wide.actual.isna().any() or not np.isfinite(wide[names].to_numpy(float)).all():
        raise ValueError("Full Favorita saved predictions are incomplete")
    assert_metric_rows(wide, "actual", metrics, "rmsle", names)
    for row in metrics.itertuples():
        if not np.isclose(score(wide.actual.to_numpy(float), wide[row.model].to_numpy(float), "mae"),
                          row.mae, rtol=0, atol=2e-8):
            raise ValueError(f"Full Favorita MAE does not recompute: {row.model}")
    for jump, reference in (("LightGBM + Jump Filter", "LightGBM + Kalman"),
                            ("LightGBM + Robust Jump Filter", "LightGBM + Robust Kalman")):
        if lock["selected_configs"][jump]["p_jump"] != 0 or not np.array_equal(wide[jump], wide[reference]):
            raise ValueError(f"Full Favorita jump limiting case failed: {jump}")
    ci = pd.read_csv(FULL / "paired_bootstrap.csv")
    for row in ci.itertuples():
        delta = float(metrics.set_index("model").loc[row.model, "rmsle"] -
                      metrics.set_index("model").loc[row.reference, "rmsle"])
        if not np.isclose(delta, row.delta_rmsle, atol=1e-9):
            raise ValueError(f"Full Favorita bootstrap point differs: {row.model}")
    return locals()


def model_table(a: dict) -> pd.DataFrame:
    rows = []
    for model in ("seasonal_naive", "lgbm_ema"):
        row = a["archived"].set_index("model").loc[model]
        rows.append({"dataset": "Favorita", "scope": "100 series; exploratory test", "model": model,
                     "n_valid": int(row.n), "rmsle": row.rmsle, "rmse_pm25": np.nan,
                     "mae": row.mae, "rmse_log": row.rmse_log,
                     "source": "results/baselines/metrics.csv"})
    for row in a["favor"].itertuples():
        rows.append({"dataset": "Favorita", "scope": "100 series; exploratory test", "model": row.model,
                     "n_valid": int(row.n), "rmsle": row.rmsle, "rmse_pm25": np.nan,
                     "mae": row.mae, "rmse_log": row.rmse_log,
                     "source": "results/model_review/real_test_comparison.csv"})
    for row in a["beijing"].itertuples():
        rows.append({"dataset": "Beijing", "scope": "12 stations; locked one-time test", "model": row.model,
                     "n_valid": int(row.n), "rmsle": np.nan, "rmse_pm25": row.rmse,
                     "mae": row.mae, "rmse_log": row.rmse_log,
                     "source": "results/external_validation/test_metrics.csv"})
    return pd.DataFrame(rows)


def synthetic_table(a: dict) -> pd.DataFrame:
    picks = {
        "stationary": [("Kalman", "oracle"), ("Robust Jump Filter", "oracle")],
        "gaussian_drift": [("Kalman", "oracle"), ("Robust Jump Filter", "oracle")],
        "jump_only": [("Kalman", "oracle"), ("Robust Jump Filter", "oracle")],
        "mixed": [("Kalman", "oracle"), ("Robust Jump Filter", "oracle")],
        "isolated_outliers": [("Jump Filter", "reference"), ("Robust Jump Filter", "reference")],
    }
    rows = []
    source = a["synthetic"]
    for scenario, pairs in picks.items():
        for model, mode in pairs:
            x = source.loc[(source.scenario == scenario) & (source.model_family == model) &
                           (source.parameter_mode == mode)]
            if len(x) != 1:
                raise ValueError(f"Synthetic row NOT AVAILABLE: {scenario}/{model}/{mode}")
            x = x.iloc[0]
            rows.append({"scenario": scenario, "model": model, "parameters": mode,
                         "seeds": int(x.n_seeds), "forecast_rmse_log": x.prediction_rmse_log_at_0_5_mean,
                         "latent_rmse": x.latent_tracking_rmse_at_0_5_mean,
                         "precision": x.jump_precision_at_decision_mean,
                         "recall": x.jump_recall_at_decision_mean,
                         "fpr": x.false_positive_rate_at_decision_mean,
                         "delay_steps": x.mean_detection_delay_at_decision_mean,
                         "outlier_false_jump_rate": x.outlier_false_jump_rate_at_decision_mean})
    return pd.DataFrame(rows)


def data_mining(a: dict) -> dict:
    eda = pd.read_csv(ED / "summary.csv").iloc[0]
    weekday = pd.read_csv(ED / "sales_by_weekday.csv")
    month = pd.read_csv(ED / "sales_by_month.csv")
    promotion = pd.read_csv(ED / "promotion_summary.csv")
    series = pd.read_csv(ED / "series_profile.csv")
    groups = pd.read_csv(BEIJING / "station_train_groups.csv")
    group_delta = pd.read_csv(BEIJING / "group_rmse_delta.csv")
    missing = pd.read_csv(BEIJING / "missingness_by_station.csv")
    monthly_missing = pd.read_csv(BEIJING / "missingness_by_month.csv")
    rolling = pd.read_csv(BEIJING / "rolling_target_residual.csv", parse_dates=["timestamp"])
    coverage = pd.read_csv(BEIJING / "dataset_profile.csv")
    raw = load_beijing(ROOT / "data/raw/beijing")
    raw["PM2.5"] = raw["PM2.5"].where(raw["PM2.5"] >= 0)
    hourly = raw.groupby(raw.timestamp.dt.hour)["PM2.5"].mean()
    beijing_month = raw.groupby(raw.timestamp.dt.month)["PM2.5"].mean()
    return locals()


def build_figures(a: dict, dm: dict, syn: pd.DataFrame) -> list[tuple[str, str, str]]:
    folder = REPORTS / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    index = []

    def finish(fig, name, purpose, source):
        fig.tight_layout()
        fig.savefig(folder / name)
        plt.close(fig)
        index.append((name, purpose, source))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(dm["weekday"].dow, dm["weekday"]["mean"])
    ax.set(xlabel="Thứ (0=Thứ Hai, 6=Chủ Nhật)", ylabel="Doanh số trung bình / dòng",
           title="Favorita: mùa vụ theo thứ, toàn bộ dữ liệu có nhãn")
    finish(fig, "favorita_weekday.png", "Mùa vụ doanh số theo thứ, mô tả toàn bộ dữ liệu.",
           "results/eda/sales_by_weekday.csv")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["Không khuyến mại", "Có khuyến mại"], dm["promotion"]["mean"])
    ax.set(ylabel="Doanh số trung bình / dòng", title="Favorita: liên hệ quan sát với khuyến mại, toàn bộ dữ liệu")
    finish(fig, "favorita_promotion.png", "Liên hệ mô tả; không suy ra tác động nhân quả.",
           "results/eda/promotion_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(np.log1p(dm["series"]["mean"]), dm["series"].zero_rate, s=6, alpha=.35)
    ax.set(xlabel="log(1 + doanh số trung bình / chuỗi)", ylabel="Tỷ lệ ngày doanh số bằng 0",
           title="Favorita: khác biệt 1.782 chuỗi, toàn bộ dữ liệu")
    finish(fig, "favorita_series.png", "Độ thưa và quy mô khác nhau giữa các chuỗi.",
           "results/eda/series_profile.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    station = dm["groups"].sort_values("train_mean")
    ax.barh(station.station, station.train_mean)
    ax.set(xlabel="PM2.5 trung bình train (µg/m³)", title="Beijing: 12 trạm, chỉ thống kê train")
    finish(fig, "beijing_train_stations.png", "Khác biệt mức PM2.5 giữa 12 trạm; nhóm định nghĩa trên train.",
           "results/external_validation/station_train_groups.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(dm["hourly"].index, dm["hourly"].to_numpy(), marker=".")
    ax[0].set(xlabel="Giờ trong ngày", ylabel="PM2.5 trung bình (µg/m³)",
              title="Beijing: giờ, toàn bộ dữ liệu gốc")
    ax[1].plot(dm["beijing_month"].index, dm["beijing_month"].to_numpy(), marker="o")
    ax[1].set(xlabel="Tháng", ylabel="PM2.5 trung bình (µg/m³)",
              title="Beijing: tháng, toàn bộ dữ liệu gốc")
    finish(fig, "beijing_seasonality.png", "Mùa vụ theo giờ và tháng; mô tả nguồn gốc, không chọn model.",
           "data/raw/beijing/PRSA_Data_20130301-20170228/*.csv")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    rolling = dm["rolling"]
    ax.plot(rolling.timestamp, rolling.residual_roll7, label="Trung bình trượt 7 ngày")
    ax.axvline(pd.Timestamp("2016-07-01"), color="black", linestyle="--", label="Bắt đầu test")
    ax.set(xlabel="Ngày (2016-2017)", ylabel="Residual LightGBM trên thang log",
           title="Beijing: residual validation/test, mô tả sau đánh giá")
    ax.legend()
    finish(fig, "beijing_rolling_residual.png", "Độ lệch dự báo theo thời gian; không xác nhận latent jump.",
           "results/external_validation/rolling_target_residual.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    rows = a["beijing"].sort_values("rmse", ascending=False)
    ax.barh(rows.model, rows.rmse)
    ax.set(xlabel="Test RMSE PM2.5 (µg/m³)", title="Beijing: 68.536 nhãn test hợp lệ, cấu hình đã khóa")
    finish(fig, "beijing_test_models.png", "So sánh bảy mô hình trên cùng nhãn test đã khóa.",
           "results/external_validation/test_metrics.csv")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    selected = syn.loc[syn.scenario.isin(["jump_only", "mixed", "isolated_outliers"])].copy()
    selected["label"] = selected.scenario + " / " + selected.model.str.replace(" Filter", "", regex=False)
    ax.barh(selected.label, selected.forecast_rmse_log)
    ax.set(xlabel="Synthetic forecast RMSE (log units)",
           title="Synthetic: oracle (jump/mixed), reference (outlier), 10 seeds")
    finish(fig, "synthetic_cases.png", "Giới hạn của phép thử oracle/reference trên synthetic.",
           "results/model_review/synthetic_comparison.csv")
    return index


def selected_table(a: dict) -> pd.DataFrame:
    rows = []
    beijing_validation = pd.read_csv(BEIJING / "validation_metrics.csv").set_index("model")
    for dataset, configs in (("Favorita", a["fconfig"]), ("Beijing", a["bconfig"])):
        for family, cfg in configs.items():
            if family == "LightGBM + EMA":
                continue
            rows.append({"dataset": dataset, "model": family, "rho": cfg.get("rho"),
                         "q": cfg.get("q"), "R": cfg.get("R"),
                         "p_jump": cfg.get("p_jump", 0), "q_jump": cfg.get("q_jump", 0),
                         "p_outlier": cfg.get("p_outlier", 0),
                         "R_out/R": ((cfg.get("R_out", np.nan)/cfg["R"] if "R_out" in cfg
                                       else cfg.get("R_out_ratio", np.nan))
                                      if cfg.get("p_outlier", 0) > 0 else np.nan),
                         "validation_log_score": (cfg["validation"]["rmsle"] if dataset == "Favorita"
                                                  else float(beijing_validation.loc[family, "rmse_log"]))})
    return pd.DataFrame(rows)


def git_info() -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    if commit.returncode:
        return "N/A — thư mục làm việc không có Git repository; không thể xác nhận commit hoặc trạng thái uncommitted."
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    state = "uncommitted changes" if status.stdout.strip() else "working tree clean"
    return f"`{commit.stdout.strip()}`; {state}."


def write_evidence(a: dict, dm: dict) -> None:
    inventory = pd.DataFrame([{"Artifact": f"[{path}](../{path})", "Vai trò": role,
                               "Trạng thái": "AVAILABLE"}
                              for path, role in INPUTS.items()])
    f = a["fconfig"]
    b = a["bconfig"]
    eq = a["equal"]
    coverage = dm["coverage"].sort_values("period")
    text = f"""# Kiểm toán bằng chứng trước khi chốt BTL

## Phạm vi và nguồn artifact

Tập kết quả sau đây có thật và được đọc trong lượt audit. Script `scripts/build_report.py` kiểm tra checksum đầu vào trước/sau, tính lại điểm từ dự báo đã lưu và chỉ ghi trong `reports/`.

{table(inventory)}

Không có artifact bắt buộc nào trong danh sách trên ở trạng thái `NOT AVAILABLE`. Precision/recall/delay synthetic ở scenario không có true jump là `N/A` theo định nghĩa, không được điền 0. Không có nhãn jump thực cho Favorita hay Beijing.

## Môi trường và provenance

- Python `{sys.version.split()[0]}`; LightGBM `{lightgbm.__version__}`; pandas `{pd.__version__}`; NumPy `{np.__version__}`; scikit-learn `{sklearn.__version__}`; Matplotlib `{matplotlib.__version__}`.
- Git: {git_info()}
- Favorita: `data/raw/train.csv` trong workspace là bản từ mirror được README ghi rõ, chưa xác minh byte-for-byte với Kaggle chính thức. EDA gốc có {int(dm['eda']['rows']):,} dòng, 54 cửa hàng, 33 family, 1.782 chuỗi. Task 03 chỉ đánh giá 100 chuỗi cố định.
- Beijing: nguồn UCI ZIP SHA-256 `{a['block']['source_zip_sha256']}`, model SHA-256 `{a['block']['model_sha256']}` đều kiểm tra trùng file local. Lock ghi `{a['block']['locked_at_utc']}`; Task 04 đã chạy test một lần sau lock. Lượt Task 05 chỉ đọc dự báo đã lưu, không thực hiện `--stage test`.

## Dữ liệu và giao thức

Favorita: train 2013-01-01–2015-12-31 (109.200 dòng của 100 chuỗi); validation 2016-01-01–2016-03-31 (9.100); test 2016-04-01–2017-08-15 ({a['frows']:,}). Target doanh số ngày `t`, biến đổi `log1p`, dự báo tất cả chuỗi trước khi cập nhật nhãn cùng ngày. Không có nhãn test thiếu trong tập chọn. Báo cáo test là **exploratory**: test đã được xem nhiều lần từ Task 02/03, không còn là ước lượng held-out unbiased.

Beijing: 12 trạm, giờ địa phương 2013-03-01–2017-02-28, {int(coverage.rows.sum()):,} dòng gốc. Train/validation/test:

{table(coverage[['period','rows','start','end','pm_missing_or_invalid']])}

Test có {a['brows']:,} station-hour và {int(a['bwide'].actual.notna().sum()):,} nhãn PM2.5 hợp lệ; {int(a['bwide'].actual.isna().sum()):,} nhãn thiếu không tính metric/không cập nhật filter. Target `log1p(PM2.5)` để train/filter, xuất dự báo `max(0, expm1(z_hat))`. Mỗi giờ dự báo đủ 12 trạm rồi mới cập nhật bằng nhãn của giờ đó. Test đã khóa và đánh giá một lần theo Task 04; **hiện đã được xem**, không gọi là blind tại thời điểm viết báo cáo.

## Kiểm tra trực tiếp chọn nhánh jump và dự báo

- Favorita Jump Filter hai nhánh: validation chọn `p_jump={f['Jump Filter']['p_jump']}`. Đây là **khác số liệu bàn giao Task 05** vốn nói cả hai nhánh chọn 0. Robust Jump chọn `p_jump={f['Robust Jump Filter']['p_jump']}`. File khóa Task 03 là bằng chứng ưu tiên.
- Favorita Robust Jump `p_jump=0` nhưng `R_out/R={f['Robust Jump Filter']['R_out']/f['Robust Jump Filter']['R']:.0f}`; Robust Kalman có `R_out/R={f['Robust Kalman']['R_out']/f['Robust Kalman']['R']:.0f}`. Vì tham số outlier khác nhau, dự báo test đã lưu **không trùng**: max |Δ|={eq['favorita_robustjump_vs_robustkalman']:.4f} sales. Không được quy chênh lệch điểm của hai cấu hình đã chọn cho jump.
- Favorita Jump Filter `p_jump>0`, dự báo khác Kalman (max |Δ|={eq['favorita_jump_vs_kalman']:.4f} sales), nhưng validation RMSLE {f['Jump Filter']['validation']['rmsle']:.5f} > {f['Kalman']['validation']['rmsle']:.5f}; test exploratory cũng kém hơn.
- Beijing Jump và Robust Jump cùng chọn `p_jump=0`. Dự báo test tương ứng trùng Kalman/Robust Kalman từng hàng (max |Δ|={eq['beijing_jump_vs_kalman']:.1f} và {eq['beijing_robustjump_vs_robustkalman']:.1f} µg/m³). Đây là kiểm tra độc lập từ `test_predictions.csv.gz`.
- Các giới hạn `p_jump=0`, `p_outlier=0` với **cùng các tham số còn lại** được unit tests kiểm tra bằng filter states. Kết quả Favorita với `R_out` khác không phải phép thử giới hạn bằng tham số khớp.

## Chỉ số, uncertainty và availability

- Favorita: RMSLE trên sales (không đơn vị) và MAE theo sales; Beijing: RMSE/MAE PM2.5 (µg/m³) và RMSE log (không đơn vị). Không so trực tiếp 0.41987 Favorita với 19.8906 Beijing.
- Paired bootstrap Beijing trong artifact lấy 35 block liên tục dài 7 ngày, giữ 12 trạm cùng block, 500 lần với seed 42. Chênh lệch định nghĩa là **model − reference**. Robust Kalman − LightGBM: {a['rkl']['delta_rmse']:+.4f} µg/m³; CI 95% [{a['rkl']['ci_low']:+.4f}, {a['rkl']['ci_high']:+.4f}]. Dấu âm là RMSE thấp hơn. Điểm đã kiểm tra lại từ `test_metrics.csv`.
- Audit code Favorita có `same_day_promotion_availability=NOT VERIFIED`: `onpromotion[t]` chỉ hợp lệ dưới giả định đã lên kế hoạch trước ngày t; CSV không chứa thời điểm công bố. Beijing chỉ dùng các phép đo `t-1`/trước đó trong feature, nhưng dữ liệu không chứng minh thời điểm công bố thực tế của khí tượng/chất ô nhiễm. Unit test về lag không thay thế bằng chứng vận hành nguồn dữ liệu.
- PM2.5 residual spike và posterior `w_jump` trên dữ liệu quan sát không có ground truth jump. Không diễn giải chúng như phát hiện physical regime change.
"""
    (REPORTS / "evidence_audit.md").write_text(text, encoding="utf-8")


def write_summary(a: dict, models: pd.DataFrame, selections: pd.DataFrame,
                  syn: pd.DataFrame) -> None:
    f = models.loc[models.dataset == "Favorita", ["model","n_valid","rmsle","mae","source"]]
    b = models.loc[models.dataset == "Beijing", ["model","n_valid","rmse_pm25","mae","rmse_log"]]
    selected = selections.copy()
    ci = a["ci"].loc[a["ci"].model.isin(["LightGBM + Kalman", "LightGBM + Robust Kalman",
                                          "LightGBM + Jump Filter", "LightGBM + Robust Jump Filter"]),
                     ["model","reference","delta_rmse","ci_low","ci_high","blocks","bootstrap_reps"]]
    text = f"""# Tổng hợp định lượng và quyết định kiến trúc

Nguồn kiểm toán: [evidence_audit.md](evidence_audit.md); bảng máy đọc được: [model_comparison.csv](tables/model_comparison.csv), [synthetic_comparison.csv](tables/synthetic_comparison.csv). Tất cả giá trị sau tính/đọc từ artifact có sẵn, không train/test mới.

## Favorita — 100 chuỗi, test exploratory

{table(f, 5)}

`seasonal_naive` và `lgbm_ema` là baseline Task 01 lưu tại `results/baselines/`; năm model còn lại là lựa chọn Task 03. Chúng có cùng 50.100 khóa/ngày và nhãn, nhưng thuộc các lần chọn cấu hình khác nhau. LightGBM RMSLE 0.41987; Robust Kalman 0.42027; Robust Jump 0.42027. Không báo cả hai jump là `p_jump=0`: bản hai nhánh chọn 0.001, bản robust chọn 0. Mẫu Favorita chỉ là 100/1.782 chuỗi EDA; test đã được xem nhiều lần.

## Beijing — 12 trạm, 68.536 nhãn test hợp lệ

{table(b, 5)}

RMSE/MAE có đơn vị µg/m³; RMSE log = RMSLE theo định nghĩa `log1p` trong Task 04. Cả hai nhánh jump chọn `p_jump=0`; 12-trạm test forecasts trùng Gaussian counterpart. Robust Kalman cải thiện RMSE so với LightGBM tĩnh, nhưng **không có đóng góp tăng thêm từ jump**.

## Cấu hình chọn trên validation

{table(selected, 5)}

`validation_log_score` là RMSLE validation với Favorita và RMSE log validation với Beijing (đọc từ [`validation_metrics.csv`](../results/external_validation/validation_metrics.csv)). `R_out/R` là N/A khi `p_outlier=0` vì nhánh này không hoạt động.

## Chênh lệch paired Beijing

Chênh lệch là model − reference trên PM2.5 gốc; âm là tốt hơn. Khoảng bootstrap 95% từ 35 block 7 ngày × 12 trạm, 500 mẫu lặp, seed 42 (artifact gốc). Không coi đây là bằng chứng nhân quả.

{table(ci, 4)}

Robust Kalman − LightGBM = −0.5300 [−1.0424, −0.1462] µg/m³. Robust Kalman − Kalman = −0.0082 [−0.0663, +0.0444], chưa có bằng chứng gain độc lập rõ của outlier branch trên Beijing. Jump − counterpart bằng 0 trên Beijing. Các cặp Favorita không có paired CI trực tiếp với counterpart trong artifact; không gán ý nghĩa thống kê cho khác biệt số điểm gần 0.

## Synthetic — 10 seed evaluation mỗi scenario

RMSE dự báo/latent lấy từ các trace lưu ở cột `at_0_5` (bản thân RMSE không phụ thuộc ngưỡng báo động); precision, recall, FPR và delay dùng ngưỡng đã hiệu chỉnh trên seed calibration tách biệt. `oracle` nghĩa là biết tham số quá trình sinh, **không biết thời điểm jump**; `reference` là cấu hình tham chiếu cố định. N/A nghĩa là metric không xác định khi không có true jump hoặc không có detection, không phải 0.

{table(syn, 4)}

Trong `jump_only`, robust jump oracle có forecast RMSE log ≈0.2496 so với Kalman oracle ≈0.7308; recall đã hiệu chỉnh ≈0.4654, nên còn bỏ sót nhiều event. Ở `isolated_outliers`, robust reference giảm latent RMSE ≈0.0993→0.0313 và tỷ lệ outlier bị gọi sai là jump 1.000→0.033 so với jump reference. Đây là kết quả điều kiện trên synthetic, không phải bằng chứng jump có ích trên dữ liệu thực.

## Kết luận kỹ thuật

Chốt lõi pipeline one-step online và LightGBM cố định. **Gaussian Kalman residual correction** là adapter gọn có bằng chứng cải thiện RMSE gốc trên Beijing; Favorita 100 chuỗi vẫn lấy LightGBM tĩnh làm mốc tốt hơn trên RMSLE. Bật/tắt adapter phải dựa trên validation riêng từng domain. Robust outlier branch giữ như ablation vì lợi ích độc lập chưa rõ. Giữ two-/three-branch jump filter như giả thuyết và kết quả âm/conditional trong BTL, không triển khai nó như nhánh mặc định. `p_jump=0` chỉ cho thấy không cải thiện mục tiêu validation với dữ liệu, model và grid hiện tại; không chứng minh thế giới thực không có jump.
"""
    (REPORTS / "results_summary.md").write_text(text, encoding="utf-8")


def write_figure_index(figures: list[tuple[str, str, str]]) -> None:
    rows = pd.DataFrame([{"Hình": f"[figures/{name}](figures/{name})", "Mục đích": purpose,
                          "Nguồn đọc (repo root)": f"`{source}`"}
                         for name, purpose, source in figures])
    text = "# Danh mục hình và bảng\n\nTám hình được xuất lại từ CSV/raw đã có, chỉ mang tính mô tả; không thay đổi model, split hay test lock.\n\n"
    text += table(rows) + "\n\n"
    text += "| Bảng | Vai trò |\n| --- | --- |\n"
    text += "| [tables/model_comparison.csv](tables/model_comparison.csv) | Mọi baseline Favorita và Beijing, giữ nguyên đơn vị. |\n"
    text += "| [tables/synthetic_comparison.csv](tables/synthetic_comparison.csv) | Scenario, tham số oracle/reference, seed và detection metric có thật. |\n"
    text += "| [../results/external_validation/group_rmse_delta.csv](../results/external_validation/group_rmse_delta.csv) | Chênh RMSE test theo nhóm trạm định nghĩa bằng train, mô tả hậu nghiệm. |\n"
    text += "| [results_summary.md](results_summary.md) | Bảng định lượng trình bày trong Markdown, cấu hình validation, paired CI. |\n"
    text += "| [evidence_audit.md](evidence_audit.md) | Danh mục artifact, đối chiếu prediction/metric, giả định availability. |\n"
    (REPORTS / "figure_index.md").write_text(text, encoding="utf-8")


def write_btl(a: dict, dm: dict, models: pd.DataFrame, selections: pd.DataFrame,
              syn: pd.DataFrame) -> None:
    favor = models.loc[models.dataset == "Favorita", ["model","n_valid","rmsle","mae"]]
    beijing = models.loc[models.dataset == "Beijing", ["model","n_valid","rmse_pm25","mae","rmse_log"]]
    selected = selections.loc[selections.model.isin(["Kalman", "Robust Kalman", "Jump Filter",
                                                      "Robust Jump Filter", "LightGBM + Kalman",
                                                      "LightGBM + Robust Kalman", "LightGBM + Jump Filter",
                                                      "LightGBM + Robust Jump Filter"]),
                          ["dataset","model","rho","q","R","p_jump","p_outlier","R_out/R"]]
    c = dm["coverage"].set_index("period")
    fzero = int(dm["eda"].zero_sales)/int(dm["eda"].rows)
    promo = dm["promotion"].set_index("onpromotion")
    promomean = float(promo.loc[True, "mean"])
    nopromomean = float(promo.loc[False, "mean"])
    station = dm["groups"]
    hour = dm["hourly"]
    month = dm["beijing_month"]
    pm_missing = int(dm["missing"].loc[dm["missing"].variable == "PM2.5", "missing"].sum())
    ci = a["ci"]
    rkl = a["rkl"]
    rk_k = ci.loc[(ci.model == "LightGBM + Robust Kalman") &
                   (ci.reference == "LightGBM + Kalman")].iloc[0]
    isolated = syn.loc[syn.scenario == "isolated_outliers"].set_index("model")
    jump_only = syn.loc[syn.scenario == "jump_only"].set_index("model")
    mixed = syn.loc[syn.scenario == "mixed"].set_index("model")
    rolling = dm["rolling"]
    residual = rolling.groupby("period").residual_mean.agg(["mean", "std"])
    report = rf"""# Dự đoán thích nghi trên dữ liệu phi dừng bằng mô hình trạng thái ẩn có drift, jump và outlier

*Adaptive Prediction under Non-Stationarity with Jump-Aware Latent-State Filtering*

## Tóm tắt

Bài tập lớn xây dựng dự báo một bước với phản hồi theo thời gian, kết hợp LightGBM cố định và bộ lọc trạng thái ẩn trên phần dư `log1p(target)`. Hai nguồn thực được đánh giá: 100 chuỗi doanh số Favorita ở mức exploratory và 12 trạm chất lượng không khí Beijing ở một test đã khóa. Trên Beijing, LightGBM có RMSE PM2.5 {a['beijing'].set_index('model').loc['LightGBM','rmse']:.4f} µg/m³; Robust Kalman có {a['beijing'].set_index('model').loc['LightGBM + Robust Kalman','rmse']:.4f}, chênh −0.5300 với CI paired block 95% [−1.0424, −0.1462]. Cả hai cấu hình jump Beijing chọn `p_jump=0` và trùng Gaussian counterpart. Favorita Robust Jump cũng chọn `p_jump=0`, trong khi bản jump hai nhánh chọn 0.001 nhưng không thắng LightGBM. Synthetic oracle cho thấy jump có thể giúp khi quá trình sinh chứa jump; recall còn hạn chế. Vì vậy, kết quả thực chưa chứng minh lợi ích tăng thêm của thành phần Lévy jump. [Nguồn số liệu và kiểm toán](results_summary.md).

## 1. Đặt vấn đề

Với chuỗi `g` và thời điểm `t`, mô hình phải phát dự báo trước khi nhận `y_{{g,t}}`. Favorita dùng ngày; Beijing dùng giờ. Toàn bộ chuỗi/trạm cùng thời điểm được dự báo trước, sau đó nhãn hợp lệ mới cập nhật state cho lần kế tiếp. Đây là backtest một bước có phản hồi, không phải dự báo đồng thời nhiều bước khi nhãn tương lai chưa có.

Sai số lớn có thể do **outlier quan sát** nhất thời (lần đo/ghi nhận bất thường) hoặc **thay đổi latent dynamics** có ảnh hưởng kéo dài. Một residual spike đơn lẻ không phân biệt được hai nguyên nhân. Báo cáo do đó xem nhánh jump như giả thuyết thống kê, không tuyên bố phát hiện bước nhảy vật lý trên dữ liệu quan sát.

## 2. Dữ liệu và khai phá dữ liệu

**Favorita.** File local có {int(dm['eda'].rows):,} dòng, 54 cửa hàng, 33 family và 1.782 chuỗi; Task 03 chọn cố định 100 chuỗi. Tỷ lệ doanh số bằng 0 là {fzero:.1%}; median tỷ lệ 0 theo chuỗi là {dm['series'].zero_rate.median():.1%}, có {int((dm['series'].zero_rate == 1).sum())} chuỗi toàn 0. Khác biệt này làm metric pooled dễ bị nhóm doanh số lớn chi phối; vì vậy báo cáo đọc thêm metric trung bình theo chuỗi trong artifact. Chủ Nhật có mean sales {float(dm['weekday'].set_index('dow').loc[6,'mean']):.1f}, thứ Năm {float(dm['weekday'].set_index('dow').loc[3,'mean']):.1f} [Hình 1](figures/favorita_weekday.png). Mean tháng 12 là {float(dm['month'].set_index('month').loc[12,'mean']):.1f}, tháng 2 là {float(dm['month'].set_index('month').loc[2,'mean']):.1f} sales/dòng ([bảng tháng](../results/eda/sales_by_month.csv)). Ngày có promotion có mean {promomean:.1f}, ngày không có {nopromomean:.1f} sales/dòng [Hình 2](figures/favorita_promotion.png); đây là liên hệ thô, không phải tác động nhân quả. Phân bố mean/zero rate của 1.782 chuỗi ở [Hình 3](figures/favorita_series.png). Nguồn dữ liệu local lấy từ mirror được ghi trong README, chưa đối chiếu checksum với Kaggle gốc.

**Beijing.** UCI có {int(c.rows.sum()):,} dòng, 12 trạm theo giờ từ 2013-03-01 đến 2017-02-28. PM2.5 thiếu {pm_missing:,} ({pm_missing/int(c.rows.sum()):.2%}) trên toàn bộ nguồn; test thiếu {int(c.loc['test','pm_missing_or_invalid']):,}/{int(c.loc['test','rows']):,}. Không biến NA thành 0. Missingness theo trạm/biến/split và tháng được lưu tại [`missingness_by_station.csv`](../results/external_validation/missingness_by_station.csv) và [`missingness_by_month.csv`](../results/external_validation/missingness_by_month.csv). Trung bình PM2.5 trên train theo trạm trải từ {station.train_mean.min():.1f} đến {station.train_mean.max():.1f} µg/m³; SD theo trạm từ {station.train_std.min():.1f} đến {station.train_std.max():.1f} [Hình 4](figures/beijing_train_stations.png). Nhóm `low/middle/high` là tam phân vị trung bình PM2.5 **chỉ từ train**, không chọn lại theo test. Trên toàn bộ nguồn mô tả, giờ {int(hour.idxmax())} có mean cao nhất {hour.max():.1f} và giờ {int(hour.idxmin())} thấp nhất {hour.min():.1f}; tháng {int(month.idxmax())} cao nhất {month.max():.1f} µg/m³ [Hình 5](figures/beijing_seasonality.png). Các số toàn nguồn chỉ để mô tả, không tạo feature hậu nghiệm.

Residual LightGBM log trung bình theo ngày có mean {residual.loc['validation','mean']:+.4f} trên validation và {residual.loc['test','mean']:+.4f} trên test; đồ thị trung bình trượt bảy ngày ở [Hình 6](figures/beijing_rolling_residual.png). Những đoạn biến động là dấu hiệu sai số dự báo, không có nhãn sự kiện jump để xác thực. [Danh mục hình và nguồn](figure_index.md) ghi rõ mục đích từng hình.

Chênh RMSE test so với LightGBM tĩnh theo nhóm trạm đã định nghĩa từ train (âm là cải thiện), chỉ là mô tả hậu nghiệm và không dùng chọn model:

{table(dm['group_delta'][['group','LightGBM + Kalman','LightGBM + Robust Kalman']], 3)}

## 3. Phương pháp và thiết kế thuật toán

LightGBM từ thư viện được huấn luyện **một lần** trên train để ước lượng `f_φ(x_{{g,t}}) ≈ log(1+y_{{g,t}})`; dự báo này giữ cố định trong backtest. Phần dự án tự triển khai là `LevyOU`, `JumpAwareFilter`, `RobustJumpFilter`, EMA và bộ đánh giá online. Trạng thái ẩn một chiều `θ_{{g,t}}` sửa bias trên thang log:

$$
z_{{g,t}}=\log(1+y_{{g,t}})=f_\phi(x_{{g,t}})+\theta_{{g,t}}+\epsilon_{{g,t}},\qquad \epsilon_{{g,t}}\sim\mathcal N(0,R),
$$

$$
\theta_{{g,t}}=\mu+\rho(\theta_{{g,t-1}}-\mu)+\eta_{{g,t}}+I_{{g,t}}J_{{g,t}},\quad
\eta_{{g,t}}\sim\mathcal N(0,q),\ I_{{g,t}}\sim\mathrm{{Bernoulli}}(p_J),\ J_{{g,t}}\sim\mathcal N(0,q_J).
$$

`p_J` là xác suất trên **một bước thời gian của dataset** (ngày hoặc giờ); không so trực tiếp giá trị giữa hai tần suất. Đây là OU rời rạc hóa với xấp xỉ jump compound-Poisson, không phải một Lévy process thuần và không suy ra quá trình sinh thật từ quan sát.

## 4. Công thức bộ lọc

Trước khi thấy nhãn tại `t`, dùng `m^- = μ+ρ(m_{{t-1}}−μ)` và `P^-_0=ρ²P_{{t-1}}+q`. Nhánh jump có `P^-_J=P^-_0+q_J`; nhánh outlier giữ `P^-_O=P^-_0` nhưng tăng variance quan sát thành `R_O>R`. Các prior hypothesis là `π_0=1−p_J−p_O`, `π_J=p_J`, `π_O=p_O`. Với residual `r_t=z_t−f_φ(x_t)` và mỗi nhánh `k`:

$$
S_k=P^-_k+R_k,\quad
\ell_k=\log\pi_k-\tfrac12\left[\log(2\pi S_k)+\frac{{(r_t-m^-)^2}}{{S_k}}\right],\quad
w_k=\frac{{e^{{\ell_k-\max_j\ell_j}}}}{{\sum_j e^{{\ell_j-\max_h\ell_h}}}}.
$$

$$
K_k=\frac{{P^-_k}}{{S_k}},\quad m_k=m^-+K_k(r_t-m^-),\quad P_k=(1-K_k)P^-_k,
$$

$$
m_t=\sum_k w_k m_k,\qquad P_t=\sum_k w_k\left[P_k+(m_k-m_t)^2\right].
$$

Đây là **moment matching xấp xỉ** sau mỗi bước, không phải exact full Bayesian filtering qua toàn lịch sử. Nếu `p_J=0`, nhánh jump tắt; nếu thêm `p_O=0` và mọi tham số còn lại khớp thì trở về Kalman Gaussian. Giá trị forecast dùng **prior** `m^-`, xuất thang gốc `max(0, expm1(f_φ+m^-))`; cập nhật chỉ sau khi đủ forecast của ngày/giờ. Nhãn thiếu không cập nhật.

## 5. Giao thức thực nghiệm

Favorita chia train đến 2015-12-31, validation 2016-01-01–03-31, test 2016-04-01–2017-08-15; 100 chuỗi được chọn cố định. Dự báo dùng lag/rolling chỉ từ ngày trước; `onpromotion[t]` giả định là kế hoạch đã biết trước, nhưng CSV không có timestamp công bố. LightGBM dùng validation early stopping; filter chọn trên cùng validation, nên điểm validation không phải kiểm tra độc lập. Test Favorita đã được xem ở Task 02/03 và là **exploratory**.

Beijing chia train đến 2015-12-31, validation 2016-01-01–06-30, test 2016-07-01–2017-02-28. 39 feature gồm station/lịch, PM2.5 lag 1/3/24/168 giờ, rolling 24/168 giờ đã shift, và chỉ số khác lag ít nhất một giờ với missing indicator. Test có 68.536 nhãn hợp lệ; trạng thái filter reset cùng prior đầu split, mọi mô hình dùng chung LightGBM prediction. Mặc dù code bảo đảm không dùng phép đo cùng giờ, **thời điểm công bố thực tế** của các phép đo `t−1` chưa được nguồn dữ liệu chứng minh. Cấu hình được khóa trong JSON trước test; Task 05 không chạy lại test.

Metric Favorita là RMSLE không đơn vị và MAE sales; Beijing là RMSE/MAE PM2.5 µg/m³ và RMSE log không đơn vị. Test Beijing có paired bootstrap block bảy ngày, 500 lần, giữ 12 trạm cùng block. Synthetic dùng 10 seed calibration và 10 seed evaluation tách biệt, 1.200 bước/scenario; bảng phân biệt oracle/reference. Chi tiết khóa và kiểm tra được ghi tại [audit](evidence_audit.md).

## 6. Kết quả thực và thảo luận

**Favorita — 100 chuỗi, test exploratory** ([CSV gốc Task 03](../results/model_review/real_test_comparison.csv), [baseline lưu trước đó](../results/baselines/metrics.csv)):

{table(favor, 5)}

**Beijing — 12 trạm, 68.536 nhãn test hợp lệ** ([CSV gốc](../results/external_validation/test_metrics.csv), [Hình 7](figures/beijing_test_models.png)):

{table(beijing, 5)}

Cấu hình chọn trên validation:

{table(selected, 5)}

Trên Beijing, Robust Kalman − LightGBM có ΔRMSE {rkl.delta_rmse:+.4f} µg/m³, CI 95% [{rkl.ci_low:+.4f}, {rkl.ci_high:+.4f}]. Robust Kalman − Kalman chỉ {rk_k.delta_rmse:+.4f}, CI [{rk_k.ci_low:+.4f}, {rk_k.ci_high:+.4f}]; chưa có bằng chứng lợi ích riêng rõ của nhánh outlier trên test Beijing. Jump hai nhánh và robust jump đều `p_jump=0`, dự báo trùng counterpart Gaussian tương ứng, nên **không có cải thiện thêm từ jump**. Trên Favorita, jump hai nhánh thực tế chọn `p_jump=0.001` nhưng validation/test đều kém Kalman/LightGBM; robust jump chọn 0 và khác robust Kalman do `R_out` khác, không do jump. Không so độ lớn RMSLE Favorita với RMSE PM2.5 Beijing vì đơn vị khác nhau.

## 7. Synthetic có nhãn jump/outlier

Những thí nghiệm này có nhãn sinh sẵn nên mới tính precision/recall/FPR/delay. Giá trị oracle biết tham số generator, không biết jump times; không đại diện cho cách tune trên dữ liệu thực. Bảng dưới ghi RMSE dự báo/latent từ trace lưu ở cột `at_0_5` (RMSE không phụ thuộc ngưỡng báo động) và chỉ số detection tại ngưỡng calibration tách seed ([artifact](../results/model_review/synthetic_comparison.csv), [Hình 8](figures/synthetic_cases.png)):

{table(syn, 4)}

Trong `jump_only`, robust jump oracle có forecast RMSE log {jump_only.loc['Robust Jump Filter','forecast_rmse_log']:.4f} so với Kalman oracle {jump_only.loc['Kalman','forecast_rmse_log']:.4f}, nhưng recall {jump_only.loc['Robust Jump Filter','recall']:.3f}. Ở `mixed`, recall chỉ {mixed.loc['Robust Jump Filter','recall']:.3f}. Với outlier đơn lẻ và cấu hình reference, robust jump giảm latent RMSE từ {isolated.loc['Jump Filter','latent_rmse']:.4f} xuống {isolated.loc['Robust Jump Filter','latent_rmse']:.4f}, tỷ lệ gọi sai outlier là jump từ {isolated.loc['Jump Filter','outlier_false_jump_rate']:.3f} xuống {isolated.loc['Robust Jump Filter','outlier_false_jump_rate']:.3f}. Các kết quả này hỗ trợ vai trò mô hình hóa có điều kiện trên synthetic, chưa xác nhận jump thực.

## 8. Giới hạn và diễn giải

Favorita chỉ 100 chuỗi và nguồn local từ mirror; test đã được phân tích lặp lại. Promotion cùng ngày chưa có bằng chứng thời điểm công bố. Beijing có đủ 12 trạm nhưng chỉ kiểm chứng one-step feedback trong một thành phố/giai đoạn; công bố phép đo `t−1` chưa được xác minh. Cả hai tập thực thiếu ground-truth latent jump và outlier; residual/posterior là biến mô hình, không phải nhãn vật lý. Paired interval phản ánh biến động theo block thời gian trong test cố định, không chứng minh nhân quả. Cấu hình chọn `p_jump=0` chỉ nói nhánh jump không cải thiện **tiêu chí validation trong phạm vi model, feature, split và grid đã thử**, không chứng minh thế giới thực không tồn tại jump.

## 9. Kết luận và hướng mở rộng

Đóng góp ở mức BTL là pipeline causal online trên hai domain, bộ lọc Gaussian/jump/outlier tự cài đặt với limiting-case tests, benchmark có provenance và kết quả âm được báo cáo thẳng. Kiến trúc chốt là **LightGBM cố định + adapter residual Gaussian tùy chọn**, quyết định bật adapter theo validation của từng domain: trên Beijing Kalman cải thiện RMSE, còn trên Favorita 100 chuỗi LightGBM tĩnh có RMSLE tốt hơn. Robust outlier là ablation hợp lý; jump-aware filter giữ như giả thuyết nghiên cứu. Một nghiên cứu kế tiếp cần dữ liệu có nhãn thay đổi hoặc thí nghiệm can thiệp và giao thức availability rõ hơn trước khi đánh giá lợi ích jump; không dùng test hiện tại để thiết kế lại mô hình.

## 10. Tái lập và tài liệu tham khảo

Lệnh đã chạy ở Task 05: `.\\.venv\\Scripts\\python.exe -m scripts.build_report` hai lần và `.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v`; script chỉ đọc `results/`/`data/raw/`, ghi trong `reports/`. Hash các artifact đầu vào được kiểm tra không đổi trước/sau. Các lệnh train/test gốc nằm trong README và artifact Task 02–04; **không chạy lại** ở lượt này.

1. [Kaggle, Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data). File Favorita local lấy từ mirror được ghi trong [README](../README.md).
2. [UCI, Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing), DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), CC BY 4.0.
3. Ke và cộng sự, [*LightGBM: A Highly Efficient Gradient Boosting Decision Tree*](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), NeurIPS 2017.
4. Kalman, [*A New Approach to Linear Filtering and Prediction Problems*](https://doi.org/10.1115/1.3662552), *Journal of Basic Engineering*, 1960.

Mã filter đang dùng nằm trong `src/jump_filter.py` và `src/robust_jump_filter.py`; pipeline báo cáo không import FilterPy hoặc River.
"""
    (REPORTS / "BTL_REPORT_DRAFT.md").write_text(report, encoding="utf-8")


def write_full_addendum(full: dict) -> None:
    """Keep the later user-authorized full run visibly distinct from Task 01–04."""
    metrics = full["metrics"]
    validation = full["val"]
    selected = full["lock"]["selected_configs"]
    compared = metrics.merge(validation[["model", "rmsle"]], on="model", suffixes=("_test", "_validation"))
    compared = compared[["model", "n", "rmsle_validation", "rmsle_test", "mae", "mean_series_rmsle"]]
    c = full["ci"]
    paired = c.loc[(c.reference == "LightGBM") | c.reference.isin(
        ["LightGBM + Kalman", "LightGBM + Robust Kalman"]),
        ["model", "reference", "delta_rmsle", "ci_low", "ci_high"]]
    ema = paired.loc[paired.model == "LightGBM + EMA"].iloc[0]
    text = f"""# Bổ sung: một lượt Favorita toàn bộ 1.782 chuỗi

Lượt này được chạy theo yêu cầu trực tiếp sau khi bản Task 05 ban đầu đã hoàn thành. Nó train một LightGBM mới trên toàn bộ 1.945.944 dòng train; chọn cấu hình filter trên 162.162 dòng validation; ghi [khóa cấu hình](../results/full_favorita_exploratory/selected_configs.json) trước khi chạy test một lần trên 892.782 dòng. Kết quả nằm riêng trong [`results/full_favorita_exploratory`](../results/full_favorita_exploratory/report.md), không thay thế các artifact Task 01–04. Khoảng test đã được xem trong các task trước nên đây là **exploratory**, không phải xác nhận blind độc lập.

Nguồn `train.csv` SHA-256 `{full['lock']['source_sha256']}` và model SHA-256 `{full['lock']['model_sha256']}` đã đối chiếu file local. Seed 42, 4 CPU threads, LightGBM tối đa 500 cây, early stopping 50 vòng trên validation, best iteration {full['lock']['best_iteration']}. Cùng bộ feature causal hiện có; `onpromotion[t]` cần giả định kế hoạch đã biết trước ngày t vì file không có thời điểm công bố. Dự báo toàn bộ chuỗi mỗi ngày trước khi cập nhật nhãn ngày đó. State khởi tạo mean 0, variance 1 ở đầu mỗi split. Grid gồm {sum(full['lock']['candidate_counts'].values())} cấu hình adapter, chọn **RMSLE validation pooled** với tie break theo thứ tự candidate; [mọi thử nghiệm validation](../results/full_favorita_exploratory/validation_grid.csv) được lưu.

## Điểm validation và test

{table(compared, 5)}

Validation chọn LightGBM tĩnh (RMSLE {validation.set_index('model').loc['LightGBM','rmsle']:.5f}) trước EMA ({validation.set_index('model').loc['LightGBM + EMA','rmsle']:.5f}), Robust Kalman ({validation.set_index('model').loc['LightGBM + Robust Kalman','rmsle']:.5f}) và Kalman ({validation.set_index('model').loc['LightGBM + Kalman','rmsle']:.5f}). Vì vậy kiến trúc mặc định cho Favorita là LightGBM tĩnh. Trên test exploratory, EMA có RMSLE thấp hơn LightGBM {abs(ema.delta_rmsle):.5f}; đây là quan sát hậu nghiệm, không dùng để chọn lại sau test.

## Sai khác theo cặp và cấu hình jump

Chênh lệch là **model − reference** trên RMSLE, âm nghĩa là tốt hơn. Paired bootstrap lấy 18 block thời gian 28 ngày, giữ mọi chuỗi cùng block, 500 lần với seed 42; CI chỉ mô tả test cố định sau chọn cấu hình.

{table(paired, 8)}

Hai cấu hình jump chọn `p_jump={selected['LightGBM + Jump Filter']['p_jump']:.0f}` và `p_jump={selected['LightGBM + Robust Jump Filter']['p_jump']:.0f}`. Dự báo lưu cho từng jump và Gaussian counterpart **trùng chính xác 892.782/892.782 hàng** với cùng rho/q/R và outlier parameters. Vì vậy lượt toàn bộ không có đóng góp riêng từ jump. Robust Kalman giảm RMSLE so với Kalman chỉ khoảng {metrics.set_index('model').loc['LightGBM + Kalman','rmsle']-metrics.set_index('model').loc['LightGBM + Robust Kalman','rmsle']:.6f}; chưa đủ cơ sở đổi kiến trúc mặc định Favorita. Dữ liệu không có ground-truth jump hoặc thời gian công bố promotion; `p_jump=0` trong grid không chứng minh thế giới thực không có jump.

Các [dự báo đã lưu](../results/full_favorita_exploratory/test_predictions.csv.gz), [điểm test](../results/full_favorita_exploratory/test_metrics.csv) và [paired bootstrap](../results/full_favorita_exploratory/paired_bootstrap.csv) đã được đọc lại để tính điểm, kiểm tra đủ khóa, chiều dấu CI và giới hạn `p_jump=0`. Hash mọi artifact của lượt mới được giữ nguyên khi sinh báo cáo.
"""
    text = text.replace("Hai cấu hình jump chọn `p_jump=0` và `p_jump=0`.",
                        "Cả hai cấu hình jump chọn `p_jump=0`.")
    text = text.replace("Grid gồm 39 cấu hình adapter,", "Validation dùng chung cho LightGBM early stopping và chọn filter; điểm validation là tiêu chí chọn, không phải đánh giá độc lập. Grid gồm 39 cấu hình adapter,")
    (REPORTS / "full_favorita_addendum.md").write_text(text, encoding="utf-8")

    additions = {
        "evidence_audit.md": "\n## Bổ sung sau audit Task 05 ban đầu\n\nLượt Favorita toàn bộ 1.782 chuỗi mới được audit riêng tại [phụ lục](full_favorita_addendum.md). Khóa cấu hình, hash nguồn/model, 892.782 hàng dự báo và metric được kiểm tra từ artifact. Lượt này là exploratory; Beijing test không chạy lại.\n",
        "results_summary.md": "\n## Bổ sung toàn bộ Favorita\n\n[Phụ lục 1.782 chuỗi](full_favorita_addendum.md) ghi validation/test đầy đủ và paired interval. Validation chọn LightGBM tĩnh; test exploratory EMA nhỉnh hơn nhẹ. Hai nhánh jump chọn `p_jump=0` và dự báo trùng Gaussian counterparts.\n",
        "BTL_REPORT_DRAFT.md": "\n## 11. Bổ sung thực nghiệm toàn bộ Favorita\n\nMột lượt train/validation/test riêng trên toàn bộ 1.782 chuỗi được trình bày trong [phụ lục](full_favorita_addendum.md). RMSLE validation chọn LightGBM tĩnh; test exploratory cho EMA nhỉnh hơn nhẹ nhưng không đổi quyết định sau khi xem test. Cả hai nhánh jump chọn `p_jump=0`, trùng dự báo Gaussian counterparts. Các con số 100 chuỗi ở phần chính giữ nguyên provenance Task 03.\n",
        "figure_index.md": "\n| [full_favorita_addendum.md](full_favorita_addendum.md) | Bảng validation/test và paired CI cho 1.782 chuỗi Favorita bổ sung. |\n",
    }
    for name, extra in additions.items():
        path = REPORTS / name
        content = path.read_text(encoding="utf-8")
        if name == "evidence_audit.md":
            content = content.replace("Lượt Task 05 chỉ đọc dự báo đã lưu, không thực hiện `--stage test`.",
                                      "Không chạy lại `--stage test` của Beijing.")
        if name == "results_summary.md":
            content = content.replace("Tất cả giá trị sau tính/đọc từ artifact có sẵn, không train/test mới.",
                                      "Các bảng Task 01–04 bên dưới đọc từ artifact đã lưu; lượt Favorita toàn bộ mới được báo riêng ở phần bổ sung.")
        if name == "BTL_REPORT_DRAFT.md":
            content = content.replace("Hai nguồn thực được đánh giá: 100 chuỗi doanh số Favorita ở mức exploratory và 12 trạm",
                                      "Hai nguồn thực được đánh giá: 100 chuỗi Favorita ở Task 03, bổ sung 1.782 chuỗi ở lượt exploratory mới, và 12 trạm")
            content = content.replace("Favorita chỉ 100 chuỗi và nguồn local từ mirror;",
                                      "Task 03 Favorita chỉ 100 chuỗi; lượt mới bao phủ 1.782 chuỗi nhưng vẫn exploratory, nguồn local từ mirror;")
            content = content.replace("còn trên Favorita 100 chuỗi LightGBM tĩnh có RMSLE tốt hơn.",
                                      "còn trên Favorita toàn bộ 1.782 chuỗi LightGBM tĩnh có RMSLE validation tốt nhất.")
            content = content.replace("**không chạy lại** ở lượt này.",
                                      "không chạy lại Beijing test. Lượt Favorita toàn bộ chạy `python -m scripts.run_full_favorita --stage validate`, sau đó `--stage test` một lần.")
        path.write_text(content.rstrip() + extra if name == "figure_index.md" else content + extra,
                        encoding="utf-8")


def validate_local_links() -> None:
    import re
    broken = []
    for document in REPORTS.glob("*.md"):
        content = document.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if target.startswith(("https://", "http://", "#")):
                continue
            if not (document.parent / target).exists():
                broken.append(f"{document.name}: {target}")
    if broken:
        raise FileNotFoundError("Broken report links: " + "; ".join(broken))


def link_collection_report() -> None:
    if not (REPORTS / "DATA_COLLECTION.md").exists():
        return
    section = """## 1.1. Thu thập dữ liệu web bổ sung

Nhóm triển khai chương trình thu thập HTTP API công khai USGS cho động đất trong vùng Ecuador và lân cận, lưu GeoJSON gốc, URL, thời điểm truy xuất và SHA-256; sau đó chuẩn hóa UTC về ngày Ecuador, kiểm tra schema/ID/coordinates, tạo lag/rolling và ghép với doanh số. [Báo cáo thu thập và số lượng thực tế](DATA_COLLECTION.md) có bảng nguồn, quality audit, EDA và lệnh tái lập. Đây là dữ liệu ngoại sinh tự thu thập; nhãn sales vẫn lấy từ Favorita có sẵn. Danh mục USGS hồi cứu có revision, chưa xác minh thời điểm công bố lịch sử; dữ liệu mới chưa tham gia các model/kết quả đã khóa bên dưới.

"""
    path = REPORTS / "BTL_REPORT_DRAFT.md"
    content = path.read_text(encoding="utf-8")
    content = content.replace("## 2. Dữ liệu và khai phá dữ liệu", section + "## 2. Dữ liệu và khai phá dữ liệu", 1)
    path.write_text(content, encoding="utf-8")
    path = REPORTS / "evidence_audit.md"
    content = path.read_text(encoding="utf-8")
    content += "\n## Thu thập web bổ sung\n\n[Báo cáo thu thập](DATA_COLLECTION.md) ghi manifest, raw GeoJSON, SHA-256, QA và bảng ghép USGS–Favorita. Các artifact thu thập được audit riêng bởi CLI collect/prepare/report; chưa thay đổi feature hoặc kết quả model Task 01–05.\n"
    path.write_text(content, encoding="utf-8")
    path = REPORTS / "figure_index.md"
    content = path.read_text(encoding="utf-8").rstrip()
    content += "\n| [DATA_COLLECTION.md](DATA_COLLECTION.md) | Thu thập USGS, QA, bảng ghép và hình phân bố sự kiện; mô tả bổ sung. |\n"
    path.write_text(content, encoding="utf-8")


def link_usgs_ablation_report() -> None:
    if not (REPORTS / "USGS_MODEL_ABLATION.md").exists():
        return
    from scripts.run_usgs_ablation import read_lock, metrics_from_predictions, intervals
    config = json.loads((ROOT / "configs/usgs_ablation.json").read_text(encoding="utf-8"))
    output = ROOT / config["output"]
    lock = read_lock(config, output, allow_archived_code=True)
    metadata = json.loads((output / "test_prediction_manifest.json").read_text(encoding="utf-8"))
    if (metadata["sha256"] != hash_file(output / "test_predictions.csv.gz") or
            metadata["lock_sha256"] != hash_file(output / "selected_configs.json")):
        raise ValueError("USGS ablation test checksum/lock changed")
    wide = pd.read_csv(output / "test_predictions.csv.gz", parse_dates=["date"])
    measured = metrics_from_predictions(wide)
    saved = pd.read_csv(output / "test_metrics.csv")
    if measured.model.tolist() != saved.model.tolist() or not np.allclose(
            measured[["n", "rmsle", "mae", "mean_series_rmsle"]],
            saved[["n", "rmsle", "mae", "mean_series_rmsle"]], rtol=0, atol=1e-9):
        raise ValueError("USGS ablation metrics do not recompute")
    val = pd.read_csv(output / "validation_metrics.csv")
    comparison = val[["model", "rmsle"]].merge(measured[["model", "n", "rmsle", "mae"]],
                 on="model", suffixes=("_validation", "_test"))
    paired = intervals(wide, config)
    saved_paired = pd.read_csv(output / "paired_bootstrap.csv")
    if not np.allclose(paired[["delta_rmsle", "ci_low", "ci_high"]],
                       saved_paired[["delta_rmsle", "ci_low", "ci_high"]], rtol=0, atol=1e-9):
        raise ValueError("USGS ablation paired intervals do not recompute")
    index = comparison.set_index("model")
    delta = index.loc["LightGBM + USGS", "rmsle_test"] - index.loc["LightGBM", "rmsle_test"]
    relative = 100*delta/index.loc["LightGBM", "rmsle_test"]
    ci_usgs = paired.loc[(paired.model == "LightGBM + USGS") & (paired.reference == "LightGBM")].iloc[0]
    finding = f"""\n<!-- usgs-findings-start -->
## Nhận xét sau đánh giá đã khóa

USGS giảm RMSLE validation từ {index.loc['LightGBM','rmsle_validation']:.6f} xuống {index.loc['LightGBM + USGS','rmsle_validation']:.6f}, nhưng test tăng từ {index.loc['LightGBM','rmsle_test']:.6f} lên {index.loc['LightGBM + USGS','rmsle_test']:.6f}. ΔRMSLE USGS − baseline = {delta:+.6f} (thay đổi tương đối {relative:+.3f}%), paired CI 95% [{ci_usgs.ci_low:+.7f}, {ci_usgs.ci_high:+.7f}]. Trong lượt này dữ liệu thu thập chưa chứng minh giúp tăng độ chính xác trên test. Khoảng CI là mô tả exploratory, chưa hiệu chỉnh multiple comparisons.

Thêm IDs cũng không cải thiện RMSLE trong ngân sách/model hiện tại. Hai ensemble đều chọn beta=1 trên validation nên trùng nhánh có USGS; không có lợi ích ensemble tăng thêm. Đây không phải bằng chứng IDs hoặc ensemble luôn vô ích: chỉ là kết quả với feature/split/ngân sách đã thử. Không đổi trọng số hoặc tune lại theo test.

Quyết định của lượt đã khóa vẫn là LightGBM + USGS theo validation; kết quả test cho thấy lợi ích đó không tổng quát sang giai đoạn sau. Chưa đủ cơ sở nâng USGS/IDs thành kiến trúc mặc định. Có thể trình bày BTL bằng pipeline thu thập–tiền xử lý–ML hoàn chỉnh và báo cáo thẳng kết quả âm. Nếu nghiên cứu tiếp, cần rolling-origin validation trong dữ liệu phát triển và đánh giá phiên bản mới riêng, không biến test đã xem thành bằng chứng blind.
"""
    path = REPORTS / "USGS_MODEL_ABLATION.md"
    original = path.read_text(encoding="utf-8").split("\n<!-- usgs-findings-start -->")[0]
    path.write_text(original + finding, encoding="utf-8")
    section = f"""\n## Bổ sung: train/test với dữ liệu USGS tự thu thập

[Thực nghiệm USGS 2×2](USGS_MODEL_ABLATION.md) dùng toàn bộ 1.782 chuỗi, bốn LightGBM có/không IDs và có/không USGS, cùng seed/split/ngân sách; hai convex ensemble chọn trọng số trên validation. Validation chọn **{lock['selected_on_validation']}** trước khi chạy test một lần. Đây là quyết định cho phiên bản feature mới; các bảng Task 01–05 phía trên giữ nguyên phạm vi model cũ.

{table(comparison, 6)}

USGS tốt hơn trên validation nhưng test RMSLE tăng {delta:+.6f} ({relative:+.3f}%), CI 95% [{ci_usgs.ci_low:+.7f}, {ci_usgs.ci_high:+.7f}]. Lượt này chưa xác nhận lợi ích tăng accuracy của dữ liệu thu thập; chưa nâng USGS/IDs thành mặc định.

USGS lag chỉ lấy origin time trước ngày dự báo; availability của revision lịch sử vẫn NOT VERIFIED. Test Favorita đã được xem nên lượt mới là hồi cứu/exploratory. Ensemble có beta=1 trùng nhánh USGS, không có lợi ích ensemble tăng thêm. Metric từ predictions đã lưu được tính lại khi tạo báo cáo; hash code/input/model/validation/test prediction được kiểm tra. Không suy ra động đất là ground-truth latent jump của sales.
"""
    for name in ("BTL_REPORT_DRAFT.md", "results_summary.md"):
        path = REPORTS / name
        path.write_text(path.read_text(encoding="utf-8") + section, encoding="utf-8")
    path = REPORTS / "evidence_audit.md"
    path.write_text(path.read_text(encoding="utf-8") +
        "\n## Audit lượt USGS ablation\n\n[Thực nghiệm dữ liệu thu thập](USGS_MODEL_ABLATION.md): xác minh hash code/input/model/validation/test predictions và tính lại RMSLE, MAE, mean-series RMSLE từ 892.782 hàng đã lưu. Không đánh giá lại model trên test.\n", encoding="utf-8")
    path = REPORTS / "figure_index.md"
    path.write_text(path.read_text(encoding="utf-8").rstrip() +
        "\n| [USGS_MODEL_ABLATION.md](USGS_MODEL_ABLATION.md) | Feature ablation 2×2, convex ensemble, validation/test và paired uncertainty. |\n", encoding="utf-8")
    path = REPORTS / "DATA_COLLECTION.md"
    content = path.read_text(encoding="utf-8").split("\n<!-- usgs-model-link -->")[0]
    content += f"\n<!-- usgs-model-link -->\n## Train/test sau thu thập\n\n[Thực nghiệm toàn bộ 1.782 chuỗi](USGS_MODEL_ABLATION.md) đã dùng feature USGS để train trên 1.945.944 dòng và test 892.782 dòng. Validation chọn USGS, nhưng test RMSLE tăng {delta:+.6f}; kết quả không xác nhận cải thiện accuracy. Pipeline thu thập–tiền xử lý–train–test đã thực hiện đầy đủ, với provenance và giới hạn availability được giữ rõ.\n"
    path.write_text(content, encoding="utf-8")


def publish_report_figures() -> None:
    """Keep small explanatory figures visible in GitHub without model artifacts."""
    import re
    import shutil
    pattern = re.compile(r"(!\[[^\]]*\]\()(?P<target>\.\./results/[^)]+\.png)(\))")
    for document in REPORTS.glob("*.md"):
        content = document.read_text(encoding="utf-8")
        def replace(match):
            source = (document.parent / match.group("target")).resolve()
            if not source.is_relative_to((ROOT / "results").resolve()) or not source.is_file():
                raise ValueError("Report figure must be an existing local result image")
            name = document.stem.lower()+"_"+source.name
            shutil.copyfile(source, REPORTS / "figures" / name)
            return match.group(1)+"figures/"+name+match.group(3)
        updated = pattern.sub(replace, content)
        if updated != content:
            document.write_text(updated, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "tables").mkdir(exist_ok=True)
    a = audit()
    full = audit_full()
    models = model_table(a)
    syn = synthetic_table(a)
    selections = selected_table(a)
    dm = data_mining(a)
    figures = build_figures(a, dm, syn)
    models.to_csv(REPORTS / "tables/model_comparison.csv", index=False)
    syn.to_csv(REPORTS / "tables/synthetic_comparison.csv", index=False)
    write_evidence(a, dm)
    write_summary(a, models, selections, syn)
    write_figure_index(figures)
    write_btl(a, dm, models, selections, syn)
    if full is not None:
        write_full_addendum(full)
    link_collection_report()
    link_usgs_ablation_report()
    publish_report_figures()
    validate_local_links()
    changed = [path for path, digest in a["original_hashes"].items()
               if hash_file(ROOT / path) != digest]
    if full is not None:
        changed += [str(FULL / name) for name, digest in full["original_hashes"].items()
                    if hash_file(FULL / name) != digest]
    if changed:
        raise RuntimeError("Source artifacts changed while building the report: " + ", ".join(changed))
    print(f"Audited {len(INPUTS) + (len(full['original_hashes']) if full else 0)} source artifacts; Favorita {a['frows']:,} paired test rows" +
          (f" + {len(full['wide']):,} full-cohort test rows; " if full else "; ") +
          f"Beijing {int(a['bwide'].actual.notna().sum()):,} valid test labels; "
          f"generated {5 if full else 4} reports, 2 CSV tables, {len(figures)} figures; source hashes unchanged.")


if __name__ == "__main__":
    main()
