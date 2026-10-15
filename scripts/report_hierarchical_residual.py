"""Audit saved predictions and write the mathematical-method report."""
from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

from scripts.run_hierarchical_residual import (BASE_MODEL, LOCK, METHOD_SOURCE,
                                                OOF_MODEL, OUT, SCRIPT_SOURCE,
                                                SOURCE, SUMMARY, TEST_PRED,
                                                VAL_PRED, sha256)
from src.hierarchical_residual import BiasPenalties, fit_hierarchical_bias

ROOT = Path(__file__).resolve().parents[1]


def diagnostics(actual: np.ndarray, base: np.ndarray, method: np.ndarray) -> dict[str, float]:
    y = np.log1p(actual)
    b = np.log1p(base)
    m = np.log1p(method)
    error = y - b
    change = m - b
    return {"n": len(y),
            "base_rmsle": float(np.sqrt(np.mean(error ** 2))),
            "method_rmsle": float(np.sqrt(np.mean((error - change) ** 2))),
            "correction_energy": float(np.mean(change ** 2)),
            "twice_alignment": float(2 * np.mean(error * change)),
            "delta_mse": float(np.mean((error - change) ** 2) - np.mean(error ** 2))}


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    checks = {"source_sha256": SOURCE, "base_model_sha256": BASE_MODEL,
              "calibration_model_sha256": OOF_MODEL, "summary_sha256": SUMMARY,
              "validation_predictions_sha256": VAL_PRED,
              "method_source_sha256": METHOD_SOURCE, "script_source_sha256": SCRIPT_SOURCE}
    for key, path in checks.items():
        if sha256(path) != lock[key]:
            raise ValueError(f"Locked input changed: {key}")
    marker = json.loads((OUT / "test_started.json").read_text(encoding="utf-8"))
    if marker["selected_config_sha256"] != sha256(LOCK) or marker["test_predictions_input_sha256"] != sha256(TEST_PRED):
        raise ValueError("Test marker does not match the locked inputs")
    summary = pd.read_csv(SUMMARY)
    fitted = fit_hierarchical_bias(summary, BiasPenalties(**lock["selected_penalties"]))
    val = pd.read_csv(VAL_PRED, usecols=["date", "store_nbr", "family", "actual", "LightGBM"],
                      parse_dates=["date"])
    val_method = np.expm1(np.maximum(0.0, np.log1p(val.LightGBM.to_numpy(float)) + fitted.predict(val)))
    jan_feb = val.date < "2016-03-01"
    parts = {}
    for name, mask in (("Jan–Feb 2016", jan_feb), ("March 2016", ~jan_feb)):
        parts[name] = diagnostics(val.loc[mask, "actual"].to_numpy(float),
                                  val.loc[mask, "LightGBM"].to_numpy(float), val_method[mask])
    test = pd.read_csv(OUT / "test_predictions.csv.gz", parse_dates=["date"])
    if len(test) != 892782 or test.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError("Unexpected test prediction coverage")
    expected = np.expm1(np.maximum(0.0, np.log1p(test.LightGBM.to_numpy(float)) + fitted.predict(test)))
    if not np.allclose(expected, test.hierarchical_prediction.to_numpy(float), rtol=1e-10, atol=1e-7):
        raise ValueError("Saved test forecasts do not reproduce")
    parts["Exploratory test"] = diagnostics(test.actual.to_numpy(float),
                                            test.LightGBM.to_numpy(float),
                                            test.hierarchical_prediction.to_numpy(float))
    saved = pd.read_csv(OUT / "test_metrics.csv")
    if not np.isclose(saved.loc[0, "rmsle"], parts["Exploratory test"]["base_rmsle"]) or not np.isclose(saved.loc[1, "rmsle"], parts["Exploratory test"]["method_rmsle"]):
        raise ValueError("Saved test metrics do not reproduce")
    squared = test[["store_nbr", "family", "date"]].copy()
    y = np.log1p(test.actual.to_numpy(float))
    squared["base_sq"] = (y - np.log1p(test.LightGBM.to_numpy(float))) ** 2
    squared["method_sq"] = (y - np.log1p(test.hierarchical_prediction.to_numpy(float))) ** 2
    by_series = squared.groupby(["store_nbr", "family"], observed=True)[["base_sq", "method_sq"]].mean()
    squared["block"] = ((squared.date - squared.date.min()).dt.days // 28).to_numpy(int)
    by_block = squared.groupby("block")[["base_sq", "method_sq"]].mean()
    ci = pd.read_csv(OUT / "paired_bootstrap.csv").iloc[0]
    gate = pd.read_csv(ROOT / "results/online_gate_development/grid.csv")
    gate_best = gate.loc[gate.method != "LightGBM"].sort_values("jan_feb_rmsle", kind="stable").iloc[0]
    lines = [
        "# Phương pháp: hiệu chỉnh phần dư phân cấp với hiệu chuẩn theo thời gian", "",
        "## Giao thức", "",
        f"Dùng {lock['fit_rows']:,} dòng 2013–2014 để train LightGBM hiệu chuẩn 110 cây; chỉ dùng {lock['calibration_rows']:,} dự báo ngoài mẫu trên 2015 để ước lượng độ lệch nhóm. LightGBM chính đã khóa từ lượt Favorita toàn bộ, dự báo validation/test được đọc từ artifact cùng model SHA-256. Chọn phạt ridge bằng RMSLE tháng 1–2/2016; điểm tháng 3 được tính cho mọi ứng viên nhưng không dùng trong quy tắc chọn. Test sử dụng dự báo LightGBM đã lưu và được đánh giá một lần cho phiên bản này.", "",
        "LightGBM chính đã dùng toàn bộ Q1/2016 để early stopping ở lượt trước, nên tháng 3 không độc lập hoàn toàn với việc chọn base model. Các ngày test Favorita cũng đã được xem trong nghiên cứu trước; mọi kết quả dưới đây là exploratory, không phải xác nhận blind.", "",
        f"Khóa: [`method_selection_lock.json`](method_selection_lock.json); ứng viên: [`hierarchical_development_grid.csv`](tables/hierarchical_development_grid.csv). Chọn candidate {lock['selected_candidate']}: $\\lambda_0={lock['selected_penalties']['global_']:g}$, $\\lambda_s={lock['selected_penalties']['store']:g}$, $\\lambda_f={lock['selected_penalties']['family']:g}$, $\\lambda_p={lock['selected_penalties']['pair']:g}$.", "",
        "## Mô hình và lời giải toán học", "",
        "Gọi $r_{g,t}=\\log(1+y_{g,t})-f(x_{g,t})$ là phần dư ngoài mẫu ở năm hiệu chuẩn; $g=(s,c)$. Giả sử $r_{g,t}=b_0+u_s+v_c+w_g+\\epsilon_{g,t}$ với $\\epsilon\\sim\\mathcal N(0,\\sigma^2)$; các hiệu ứng có prior Gaussian tâm 0, phương sai tương ứng $\\sigma^2/\\lambda_j$. Posterior mean/MAP tối thiểu hóa hàm lồi chặt:", "",
        "$$L=\\sum_g n_g(\\bar r_g-b_0-u_s-v_c-w_g)^2+\\lambda_0b_0^2+\\lambda_s\\sum_su_s^2+\\lambda_f\\sum_cv_c^2+\\lambda_p\\sum_gw_g^2.$$", "",
        "Với các hiệu ứng chung đã cố định, nghiệm chính xác theo từng cặp là", "",
        "$$w_g=\\frac{n_g}{n_g+\\lambda_p}(\\bar r_g-b_0-u_s-v_c).$$", "",
        "Thế ngược vào $L$ cho bài toán tuyến tính chỉ còn 88 ẩn (1 toàn cục + 54 cửa hàng + 33 nhóm hàng), trọng số hiệu dụng $n_g\\lambda_p/(n_g+\\lambda_p)$. Ma trận Hessian xác định dương vì mọi $\\lambda_j>0$, do đó nghiệm tìm được là **nghiệm tối ưu duy nhất**, không phải kết quả của heuristic. Dự báo là $\\widehat z=\\max(0,f(x)+\\widehat b_0+\\widehat u_s+\\widehat v_c+\\widehat w_g)$.", "",
        "Vì $z=\\log(1+y)\\ge0$, phép chiếu một dự báo log $a$ lên $[0,\\infty)$ không tăng lỗi bình phương: $(z-\\max(0,a))^2\\le(z-a)^2$ cho mọi $z\\ge0$. Phép chặn về doanh số không âm do đó phù hợp với mục tiêu RMSLE.", "",
        "Trong mô hình Gaussian đúng và prior đúng, posterior mean tối thiểu hóa kỳ vọng lỗi bình phương log có điều kiện trên dữ liệu hiệu chuẩn. Đây là bảo đảm có điều kiện theo giả định; nó không tự động chứng minh dữ liệu thật tuân theo Gaussian hoặc hiệu ứng bất biến theo thời gian.", "",
        "### Mệnh đề về mức lệch cần thiết để hiệu chỉnh có ích", "",
        "Xét một chuỗi đơn giản hơn: phần dư hiệu chuẩn $r_i=\\beta+\\epsilon_i$, $n$ quan sát độc lập, $\\mathbb E\\epsilon_i=0$, $\\operatorname{Var}(\\epsilon_i)=\\sigma^2$. Ước lượng co là $\\widehat\\beta=a\\bar r$, $a=n/(n+\\lambda)$; phần dư tương lai độc lập có trung bình $\\beta$. So với không hiệu chỉnh, chênh kỳ vọng lỗi bình phương là", "",
        "$$\\Delta=(a^2-2a)\\beta^2+a^2\\sigma^2/n.$$", "",
        "Vì $a>0$, hiệu chỉnh có lợi **khi và chỉ khi** $\\beta^2>\\sigma^2/(n+2\\lambda)$. Đây là điều kiện có thể diễn giải: sai lệch ổn định phải đủ lớn so với nhiễu ước lượng. Nếu trung bình tương lai dịch thành $\\beta+\\delta$ với $|\\delta|\\le D$, điều kiện đủ mạnh hơn là $(2a-a^2)\\beta^2-a^2\\sigma^2/n>2a|\\beta|D$. Điều này cho thấy rõ vì sao trôi phân phối có thể xóa lợi ích của hiệu chỉnh tĩnh. Mệnh đề là trường hợp một nhóm của mô hình phân cấp, không phải bảo đảm vô điều kiện cho Favorita.", "",
        "### Vì sao bài toán hiện tại không cần reinforcement learning", "",
        "Nếu hành động duy nhất là công bố dự báo $a_t$, còn doanh số thật và trạng thái dữ liệu tương lai không phụ thuộc vào con số dự báo, phương trình Bellman có dạng $Q(H_t,a_t)=-\\mathbb E[\\ell(a_t,Z_t)\\mid H_t]+\\gamma\\mathbb E[V(H_{t+1})\\mid H_t]$. Hạng thứ hai không phụ thuộc $a_t$; do đó $a_t^*=\\arg\\min_a\\mathbb E[\\ell(a,Z_t)\\mid H_t]$. Với lỗi bình phương trên log, nghiệm là kỳ vọng có điều kiện của $Z_t$. Bởi vậy học có giám sát/online regression là phương pháp đúng cho dữ liệu hiện có. RL chỉ trở nên cần thiết nếu dự báo được dùng để quyết định tồn kho/giá và quyết định ấy làm thay đổi doanh số hoặc reward tương lai; dataset này không ghi chuỗi quyết định đó.", "",
        "## Đẳng thức kiểm chứng hiệu quả, không cần giả định phân phối", "",
        "Với $e=z-\\widehat z_{base}$, $d=\\widehat z_{method}-\\widehat z_{base}$, tính sau phép chặn dự báo, ta có chính xác:", "",
        "$$\\Delta\\mathrm{MSE}_{log}=\\mathbb E[d^2]-2\\mathbb E[ed].$$", "",
        "Phương pháp giảm RMSLE khi và chỉ khi $2\\mathbb E[ed]>\\mathbb E[d^2]$. Hai vế được tính lại từ dự báo đã lưu:", "",
        "| Giai đoạn | n | RMSLE base | RMSLE hiệu chỉnh | $\\mathbb E[d^2]$ | $2\\mathbb E[ed]$ | $\\Delta\\mathrm{MSE}_{log}$ |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in parts.items():
        lines.append(f"| {name} | {row['n']:,} | {row['base_rmsle']:.6f} | {row['method_rmsle']:.6f} | {row['correction_energy']:.7f} | {row['twice_alignment']:.7f} | {row['delta_mse']:+.7f} |")
    lines += ["", "## Kết quả test và giới hạn", "",
              f"Test có {int((by_series.method_sq < by_series.base_sq).sum()):,}/{len(by_series):,} chuỗi và {int((by_block.method_sq < by_block.base_sq).sum())}/{len(by_block)} block 28 ngày giảm lỗi bình phương log. Chênh RMSLE = {ci.delta_rmsle:+.6f}; paired block bootstrap 95% [{ci.ci_low:+.6f}, {ci.ci_high:+.6f}] trên test cố định. MAE tăng từ {saved.loc[0, 'mae']:.3f} lên {saved.loc[1, 'mae']:.3f} đơn vị sales. Hiệu quả RMSLE nhỏ và chưa được xác nhận bằng test blind mới.", "",
              "Không dùng test để điều chỉnh phương pháp hoặc các phạt ridge. Bộ lọc jump ở các lượt trước không cho thêm lợi ích thực chứng; bản hiệu chỉnh phân cấp này là một giả thuyết khác, dựa trên sai lệch ổn định theo nhóm.", "",
              "## Nhánh online đã thử và loại", "",
              f"Đã chạy 60 cấu hình online OGD/ridge trên validation; kết quả tốt nhất trong lưới là **{gate_best['method']}** với RMSLE Jan–Feb {gate_best.jan_feb_rmsle:.6f}, trong khi LightGBM đạt {gate.iloc[0].jan_feb_rmsle:.6f}. Nhánh này không được đưa sang test. [`online_gate_development_grid.csv`](tables/online_gate_development_grid.csv).", "",
              "Với OGD theo từng chuỗi, $c_t$ là EMA phần dư quá khứ, $w_t\\in[0,1]$, $\\ell_t(w)=(r_t-wc_t)^2$ và $w_{t+1}=\\Pi_{[0,1]}(w_t-\\eta_t\\nabla\\ell_t(w_t))$. Tính lồi và phép chiếu cho mọi $u\\in[0,1]$ bảo đảm đường đi:\n$$\\sum_{t=1}^T[\\ell_t(w_t)-\\ell_t(u)]\\le\\frac{1}{2\\eta_T}+\\frac12\\sum_{t=1}^T\\eta_t\\|\\nabla\\ell_t(w_t)\\|^2,$$\nvới $\\eta_t$ không tăng. Đây là bảo đảm so với một hệ số cố định tốt nhất trên **dãy tín hiệu EMA đã chọn**, không bảo đảm thắng LightGBM trong một cửa sổ ngắn. Validation xác nhận giới hạn thực tế đó.", "",
              "## Kiểm tra và tái lập", "",
              "`python -m scripts.run_hierarchical_residual --stage develop` tạo OOF model, sufficient statistics, grid và khóa. `--stage test` chỉ chạy một lần rồi ghi marker. `python -m scripts.report_hierarchical_residual` kiểm tra hash và tái tính dự báo/metric từ artifact; không train hoặc đánh giá lại base model. Các nguồn data/model lớn nằm trong `results/` và `data/` bị Git ignore.", ""]
    path = ROOT / "reports/MATHEMATICAL_METHOD.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    public = ROOT / "reports/tables"
    public.mkdir(parents=True, exist_ok=True)
    for source, destination in ((LOCK, ROOT / "reports/method_selection_lock.json"),
                                (OUT / "development_grid.csv", public / "hierarchical_development_grid.csv"),
                                (ROOT / "results/online_gate_development/grid.csv", public / "online_gate_development_grid.csv"),
                                (OUT / "test_metrics.csv", public / "hierarchical_test_metrics.csv"),
                                (OUT / "paired_bootstrap.csv", public / "hierarchical_paired_bootstrap.csv")):
        copyfile(source, destination)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
