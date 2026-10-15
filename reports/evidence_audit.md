# Kiểm toán bằng chứng trước khi chốt BTL

## Phạm vi và nguồn artifact

Tập kết quả sau đây có thật và được đọc trong lượt audit. Script `scripts/build_report.py` kiểm tra checksum đầu vào trước/sau, tính lại điểm từ dự báo đã lưu và chỉ ghi trong `reports/`.

| Artifact | Vai trò | Trạng thái |
| --- | --- | --- |
| [results/eda/summary.csv](../results/eda/summary.csv) | Favorita full-data coverage and zero counts | AVAILABLE |
| [results/eda/sales_by_weekday.csv](../results/eda/sales_by_weekday.csv) | Favorita weekday sales summaries | AVAILABLE |
| [results/eda/sales_by_month.csv](../results/eda/sales_by_month.csv) | Favorita monthly sales summaries | AVAILABLE |
| [results/eda/promotion_summary.csv](../results/eda/promotion_summary.csv) | Favorita promotion association | AVAILABLE |
| [results/eda/series_profile.csv](../results/eda/series_profile.csv) | Favorita series distribution | AVAILABLE |
| [results/baselines/metrics.csv](../results/baselines/metrics.csv) | Archived Favorita EMA and seasonal baselines | AVAILABLE |
| [results/baselines/predictions.csv.gz](../results/baselines/predictions.csv.gz) | Archived Favorita baseline predictions | AVAILABLE |
| [results/diagnostics/leakage_audit.csv](../results/diagnostics/leakage_audit.csv) | Favorita timing and leakage audit | AVAILABLE |
| [results/diagnostics/diagnostics.md](../results/diagnostics/diagnostics.md) | Task 02 diagnostics and protocol | AVAILABLE |
| [results/model_review/review.md](../results/model_review/review.md) | Task 03 interpretation and protocol | AVAILABLE |
| [results/model_review/selected_configs.json](../results/model_review/selected_configs.json) | Favorita validation selections | AVAILABLE |
| [results/model_review/validation_grid.csv](../results/model_review/validation_grid.csv) | Favorita filter candidate grid | AVAILABLE |
| [results/model_review/real_validation_comparison.csv](../results/model_review/real_validation_comparison.csv) | Favorita selected validation scores | AVAILABLE |
| [results/model_review/real_test_comparison.csv](../results/model_review/real_test_comparison.csv) | Favorita Task 03 test metrics | AVAILABLE |
| [results/model_review/test_predictions.csv](../results/model_review/test_predictions.csv) | Favorita Task 03 paired predictions | AVAILABLE |
| [results/model_review/synthetic_comparison.csv](../results/model_review/synthetic_comparison.csv) | Synthetic evaluation means | AVAILABLE |
| [results/model_review/synthetic_seed_comparison.csv](../results/model_review/synthetic_seed_comparison.csv) | Synthetic per-seed values | AVAILABLE |
| [results/model_review/selected_synthetic_thresholds.json](../results/model_review/selected_synthetic_thresholds.json) | Synthetic calibration/evaluation seed lists | AVAILABLE |
| [results/external_validation/report.md](../results/external_validation/report.md) | Task 04 Beijing protocol and interpretation | AVAILABLE |
| [results/external_validation/selected_configs.json](../results/external_validation/selected_configs.json) | Beijing validation lock | AVAILABLE |
| [results/external_validation/dataset_profile.csv](../results/external_validation/dataset_profile.csv) | Beijing split coverage | AVAILABLE |
| [results/external_validation/missingness_by_station.csv](../results/external_validation/missingness_by_station.csv) | Beijing missingness and invalid values | AVAILABLE |
| [results/external_validation/missingness_by_month.csv](../results/external_validation/missingness_by_month.csv) | Beijing monthly missingness | AVAILABLE |
| [results/external_validation/station_train_groups.csv](../results/external_validation/station_train_groups.csv) | Train-only Beijing station groups | AVAILABLE |
| [results/external_validation/group_rmse_delta.csv](../results/external_validation/group_rmse_delta.csv) | Test RMSE deltas by fixed train-only station group | AVAILABLE |
| [results/external_validation/rolling_target_residual.csv](../results/external_validation/rolling_target_residual.csv) | Beijing validation/test residual trends | AVAILABLE |
| [results/external_validation/validation_metrics.csv](../results/external_validation/validation_metrics.csv) | Beijing validation scores | AVAILABLE |
| [results/external_validation/validation_grid.csv](../results/external_validation/validation_grid.csv) | Beijing filter candidate grid | AVAILABLE |
| [results/external_validation/test_metrics.csv](../results/external_validation/test_metrics.csv) | Beijing locked test scores | AVAILABLE |
| [results/external_validation/test_predictions.csv.gz](../results/external_validation/test_predictions.csv.gz) | Beijing paired test predictions | AVAILABLE |
| [results/external_validation/paired_bootstrap.csv](../results/external_validation/paired_bootstrap.csv) | Beijing paired seven-day uncertainty | AVAILABLE |
| [results/external_validation/beijing_lightgbm.txt](../results/external_validation/beijing_lightgbm.txt) | Locked Beijing LightGBM model | AVAILABLE |
| [data/raw/beijing/PRSA2017_Data_20130301-20170228.zip](../data/raw/beijing/PRSA2017_Data_20130301-20170228.zip) | Official UCI source archive | AVAILABLE |
| [data/raw/train.csv](../data/raw/train.csv) | Local Favorita labeled source from mirror noted in README | AVAILABLE |

Không có artifact bắt buộc nào trong danh sách trên ở trạng thái `NOT AVAILABLE`. Precision/recall/delay synthetic ở scenario không có true jump là `N/A` theo định nghĩa, không được điền 0. Không có nhãn jump thực cho Favorita hay Beijing.

## Môi trường và provenance

- Python `3.11.9`; LightGBM `4.7.0`; pandas `3.0.6`; NumPy `2.4.6`; scikit-learn `1.9.1`; Matplotlib `3.11.2`.
- Git: `f3f3b76959986bc928ef7ec4caf46582254d76a9`; uncommitted changes.
- Favorita: `data/raw/train.csv` trong workspace là bản từ mirror được README ghi rõ, chưa xác minh byte-for-byte với Kaggle chính thức. EDA gốc có 3,000,888 dòng, 54 cửa hàng, 33 family, 1.782 chuỗi. Task 03 chỉ đánh giá 100 chuỗi cố định.
- Beijing: nguồn UCI ZIP SHA-256 `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`, model SHA-256 `784c5ce1e336b02a87ab6642b4127490d05b7b4c958c326f5e8ea646ecb5f1be` đều kiểm tra trùng file local. Lock ghi `2026-10-14T02:06:00.674358+00:00`; Task 04 đã chạy test một lần sau lock. Không chạy lại `--stage test` của Beijing.

## Dữ liệu và giao thức

Favorita: train 2013-01-01–2015-12-31 (109.200 dòng của 100 chuỗi); validation 2016-01-01–2016-03-31 (9.100); test 2016-04-01–2017-08-15 (50,100). Target doanh số ngày `t`, biến đổi `log1p`, dự báo tất cả chuỗi trước khi cập nhật nhãn cùng ngày. Không có nhãn test thiếu trong tập chọn. Báo cáo test là **exploratory**: test đã được xem nhiều lần từ Task 02/03, không còn là ước lượng held-out unbiased.

Beijing: 12 trạm, giờ địa phương 2013-03-01–2017-02-28, 420,768 dòng gốc. Train/validation/test:

| period | rows | start | end | pm_missing_or_invalid |
| --- | --- | --- | --- | --- |
| test | 69984 | 2016-07-01 | 2017-02-28 23:00:00 | 1448 |
| train | 298368 | 2013-03-01 | 2015-12-31 23:00:00 | 6218 |
| validation | 52416 | 2016-01-01 | 2016-06-30 23:00:00 | 1073 |

Test có 69,984 station-hour và 68,536 nhãn PM2.5 hợp lệ; 1,448 nhãn thiếu không tính metric/không cập nhật filter. Target `log1p(PM2.5)` để train/filter, xuất dự báo `max(0, expm1(z_hat))`. Mỗi giờ dự báo đủ 12 trạm rồi mới cập nhật bằng nhãn của giờ đó. Test đã khóa và đánh giá một lần theo Task 04; **hiện đã được xem**, không gọi là blind tại thời điểm viết báo cáo.

## Kiểm tra trực tiếp chọn nhánh jump và dự báo

- Favorita Jump Filter hai nhánh: validation chọn `p_jump=0.001`. Đây là **khác số liệu bàn giao Task 05** vốn nói cả hai nhánh chọn 0. Robust Jump chọn `p_jump=0.0`. File khóa Task 03 là bằng chứng ưu tiên.
- Favorita Robust Jump `p_jump=0` nhưng `R_out/R=9`; Robust Kalman có `R_out/R=25`. Vì tham số outlier khác nhau, dự báo test đã lưu **không trùng**: max |Δ|=33.6648 sales. Không được quy chênh lệch điểm của hai cấu hình đã chọn cho jump.
- Favorita Jump Filter `p_jump>0`, dự báo khác Kalman (max |Δ|=6896.5324 sales), nhưng validation RMSLE 0.45601 > 0.45491; test exploratory cũng kém hơn.
- Beijing Jump và Robust Jump cùng chọn `p_jump=0`. Dự báo test tương ứng trùng Kalman/Robust Kalman từng hàng (max |Δ|=0.0 và 0.0 µg/m³). Đây là kiểm tra độc lập từ `test_predictions.csv.gz`.
- Các giới hạn `p_jump=0`, `p_outlier=0` với **cùng các tham số còn lại** được unit tests kiểm tra bằng filter states. Kết quả Favorita với `R_out` khác không phải phép thử giới hạn bằng tham số khớp.

## Chỉ số, uncertainty và availability

- Favorita: RMSLE trên sales (không đơn vị) và MAE theo sales; Beijing: RMSE/MAE PM2.5 (µg/m³) và RMSE log (không đơn vị). Không so trực tiếp 0.41987 Favorita với 19.8906 Beijing.
- Paired bootstrap Beijing trong artifact lấy 35 block liên tục dài 7 ngày, giữ 12 trạm cùng block, 500 lần với seed 42. Chênh lệch định nghĩa là **model − reference**. Robust Kalman − LightGBM: -0.5300 µg/m³; CI 95% [-1.0424, -0.1462]. Dấu âm là RMSE thấp hơn. Điểm đã kiểm tra lại từ `test_metrics.csv`.
- Audit code Favorita có `same_day_promotion_availability=NOT VERIFIED`: `onpromotion[t]` chỉ hợp lệ dưới giả định đã lên kế hoạch trước ngày t; CSV không chứa thời điểm công bố. Beijing chỉ dùng các phép đo `t-1`/trước đó trong feature, nhưng dữ liệu không chứng minh thời điểm công bố thực tế của khí tượng/chất ô nhiễm. Unit test về lag không thay thế bằng chứng vận hành nguồn dữ liệu.
- PM2.5 residual spike và posterior `w_jump` trên dữ liệu quan sát không có ground truth jump. Không diễn giải chúng như phát hiện physical regime change.

## Bổ sung sau audit Task 05 ban đầu

Lượt Favorita toàn bộ 1.782 chuỗi mới được audit riêng tại [phụ lục](full_favorita_addendum.md). Khóa cấu hình, hash nguồn/model, 892.782 hàng dự báo và metric được kiểm tra từ artifact. Lượt này là exploratory; Beijing test không chạy lại.

## Thu thập web bổ sung

[Báo cáo thu thập](DATA_COLLECTION.md) ghi manifest, raw GeoJSON, SHA-256, QA và bảng ghép USGS–Favorita. Các artifact thu thập được audit riêng bởi CLI collect/prepare/report; chưa thay đổi feature hoặc kết quả model Task 01–05.

## Audit lượt USGS ablation

[Thực nghiệm dữ liệu thu thập](USGS_MODEL_ABLATION.md): xác minh hash code/input/model/validation/test predictions và tính lại RMSLE, MAE, mean-series RMSLE từ 892.782 hàng đã lưu. Không đánh giá lại model trên test.
