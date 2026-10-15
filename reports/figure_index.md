# Danh mục hình và bảng

Tám hình được xuất lại từ CSV/raw đã có, chỉ mang tính mô tả; không thay đổi model, split hay test lock.

| Hình | Mục đích | Nguồn đọc (repo root) |
| --- | --- | --- |
| [figures/favorita_weekday.png](figures/favorita_weekday.png) | Mùa vụ doanh số theo thứ, mô tả toàn bộ dữ liệu. | `results/eda/sales_by_weekday.csv` |
| [figures/favorita_promotion.png](figures/favorita_promotion.png) | Liên hệ mô tả; không suy ra tác động nhân quả. | `results/eda/promotion_summary.csv` |
| [figures/favorita_series.png](figures/favorita_series.png) | Độ thưa và quy mô khác nhau giữa các chuỗi. | `results/eda/series_profile.csv` |
| [figures/beijing_train_stations.png](figures/beijing_train_stations.png) | Khác biệt mức PM2.5 giữa 12 trạm; nhóm định nghĩa trên train. | `results/external_validation/station_train_groups.csv` |
| [figures/beijing_seasonality.png](figures/beijing_seasonality.png) | Mùa vụ theo giờ và tháng; mô tả nguồn gốc, không chọn model. | `data/raw/beijing/PRSA_Data_20130301-20170228/*.csv` |
| [figures/beijing_rolling_residual.png](figures/beijing_rolling_residual.png) | Độ lệch dự báo theo thời gian; không xác nhận latent jump. | `results/external_validation/rolling_target_residual.csv` |
| [figures/beijing_test_models.png](figures/beijing_test_models.png) | So sánh bảy mô hình trên cùng nhãn test đã khóa. | `results/external_validation/test_metrics.csv` |
| [figures/synthetic_cases.png](figures/synthetic_cases.png) | Giới hạn của phép thử oracle/reference trên synthetic. | `results/model_review/synthetic_comparison.csv` |

| Bảng | Vai trò |
| --- | --- |
| [tables/model_comparison.csv](tables/model_comparison.csv) | Mọi baseline Favorita và Beijing, giữ nguyên đơn vị. |
| [tables/synthetic_comparison.csv](tables/synthetic_comparison.csv) | Scenario, tham số oracle/reference, seed và detection metric có thật. |
| [../results/external_validation/group_rmse_delta.csv](../results/external_validation/group_rmse_delta.csv) | Chênh RMSE test theo nhóm trạm định nghĩa bằng train, mô tả hậu nghiệm. |
| [results_summary.md](results_summary.md) | Bảng định lượng trình bày trong Markdown, cấu hình validation, paired CI. |
| [evidence_audit.md](evidence_audit.md) | Danh mục artifact, đối chiếu prediction/metric, giả định availability. |
| [full_favorita_addendum.md](full_favorita_addendum.md) | Bảng validation/test và paired CI cho 1.782 chuỗi Favorita bổ sung. |
| [DATA_COLLECTION.md](DATA_COLLECTION.md) | Thu thập USGS, QA, bảng ghép và hình phân bố sự kiện; mô tả bổ sung. |
| [USGS_MODEL_ABLATION.md](USGS_MODEL_ABLATION.md) | Feature ablation 2×2, convex ensemble, validation/test và paired uncertainty. |
