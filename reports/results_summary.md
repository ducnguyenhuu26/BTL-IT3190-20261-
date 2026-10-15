# Tổng hợp định lượng và quyết định kiến trúc

Nguồn kiểm toán: [evidence_audit.md](evidence_audit.md); bảng máy đọc được: [model_comparison.csv](tables/model_comparison.csv), [synthetic_comparison.csv](tables/synthetic_comparison.csv). Các bảng Task 01–04 bên dưới đọc từ artifact đã lưu; lượt Favorita toàn bộ mới được báo riêng ở phần bổ sung.

## Favorita — 100 chuỗi, test exploratory

| model | n_valid | rmsle | mae | source |
| --- | --- | --- | --- | --- |
| seasonal_naive | 50100 | 0.56698 | 110.79490 | results/baselines/metrics.csv |
| lgbm_ema | 50100 | 0.43040 | 97.63515 | results/baselines/metrics.csv |
| LightGBM | 50100 | 0.41987 | 102.04009 | results/model_review/real_test_comparison.csv |
| Kalman | 50100 | 0.42028 | 102.36786 | results/model_review/real_test_comparison.csv |
| Jump Filter | 50100 | 0.42140 | 102.27453 | results/model_review/real_test_comparison.csv |
| Robust Kalman | 50100 | 0.42027 | 101.37252 | results/model_review/real_test_comparison.csv |
| Robust Jump Filter | 50100 | 0.42027 | 101.37217 | results/model_review/real_test_comparison.csv |

`seasonal_naive` và `lgbm_ema` là baseline Task 01 lưu tại `results/baselines/`; năm model còn lại là lựa chọn Task 03. Chúng có cùng 50.100 khóa/ngày và nhãn, nhưng thuộc các lần chọn cấu hình khác nhau. LightGBM RMSLE 0.41987; Robust Kalman 0.42027; Robust Jump 0.42027. Không báo cả hai jump là `p_jump=0`: bản hai nhánh chọn 0.001, bản robust chọn 0. Mẫu Favorita chỉ là 100/1.782 chuỗi EDA; test đã được xem nhiều lần.

## Beijing — 12 trạm, 68.536 nhãn test hợp lệ

| model | n_valid | rmse_pm25 | mae | rmse_log |
| --- | --- | --- | --- | --- |
| Seasonal naive | 68536 | 93.78176 | 61.68840 | 1.29739 |
| LightGBM | 68536 | 20.42065 | 10.26198 | 0.29332 |
| LightGBM + EMA | 68536 | 20.41725 | 10.26056 | 0.29277 |
| LightGBM + Kalman | 68536 | 19.89883 | 10.04509 | 0.29305 |
| LightGBM + Robust Kalman | 68536 | 19.89063 | 10.04049 | 0.29248 |
| LightGBM + Jump Filter | 68536 | 19.89883 | 10.04509 | 0.29305 |
| LightGBM + Robust Jump Filter | 68536 | 19.89063 | 10.04049 | 0.29248 |

RMSE/MAE có đơn vị µg/m³; RMSE log = RMSLE theo định nghĩa `log1p` trong Task 04. Cả hai nhánh jump chọn `p_jump=0`; 12-trạm test forecasts trùng Gaussian counterpart. Robust Kalman cải thiện RMSE so với LightGBM tĩnh, nhưng **không có đóng góp tăng thêm từ jump**.

## Cấu hình chọn trên validation

| dataset | model | rho | q | R | p_jump | q_jump | p_outlier | R_out/R | validation_log_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Favorita | Kalman | 0.90000 | 0.00000 | 0.60000 | 0.00000 | 0.00000 | 0.00000 | N/A | 0.45491 |
| Favorita | Jump Filter | 0.90000 | 0.00000 | 0.60000 | 0.00100 | 0.30000 | 0.00000 | N/A | 0.45601 |
| Favorita | Robust Kalman | 0.90000 | 0.00100 | 0.60000 | 0.00000 | 0.00000 | 0.00500 | 25.00000 | 0.45542 |
| Favorita | Robust Jump Filter | 0.90000 | 0.00100 | 0.60000 | 0.00000 | 0.00000 | 0.00500 | 9.00000 | 0.45530 |
| Beijing | LightGBM + Kalman | 0.80000 | 0.00200 | 0.04780 | 0.00000 | 0.00000 | 0.00000 | N/A | 0.30805 |
| Beijing | LightGBM + Robust Kalman | 0.80000 | 0.00200 | 0.04780 | 0.00000 | 0.00000 | 0.02000 | 25.00000 | 0.30796 |
| Beijing | LightGBM + Jump Filter | 0.80000 | 0.00200 | 0.04780 | 0.00000 | 0.50000 | 0.00000 | N/A | 0.30805 |
| Beijing | LightGBM + Robust Jump Filter | 0.80000 | 0.00200 | 0.04780 | 0.00000 | 0.50000 | 0.02000 | 25.00000 | 0.30796 |

`validation_log_score` là RMSLE validation với Favorita và RMSE log validation với Beijing (đọc từ [`validation_metrics.csv`](../results/external_validation/validation_metrics.csv)). `R_out/R` là N/A khi `p_outlier=0` vì nhánh này không hoạt động.

## Chênh lệch paired Beijing

Chênh lệch là model − reference trên PM2.5 gốc; âm là tốt hơn. Khoảng bootstrap 95% từ 35 block 7 ngày × 12 trạm, 500 mẫu lặp, seed 42 (artifact gốc). Không coi đây là bằng chứng nhân quả.

| model | reference | delta_rmse | ci_low | ci_high | blocks | bootstrap_reps |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM + Kalman | LightGBM | -0.5218 | -1.0680 | -0.1139 | 35 | 500 |
| LightGBM + Robust Kalman | LightGBM | -0.5300 | -1.0424 | -0.1462 | 35 | 500 |
| LightGBM + Robust Kalman | LightGBM + Kalman | -0.0082 | -0.0663 | 0.0444 | 35 | 500 |
| LightGBM + Jump Filter | LightGBM | -0.5218 | -1.0680 | -0.1139 | 35 | 500 |
| LightGBM + Jump Filter | LightGBM + Kalman | 0.0000 | 0.0000 | 0.0000 | 35 | 500 |
| LightGBM + Robust Jump Filter | LightGBM | -0.5300 | -1.0424 | -0.1462 | 35 | 500 |
| LightGBM + Robust Jump Filter | LightGBM + Robust Kalman | 0.0000 | 0.0000 | 0.0000 | 35 | 500 |

Robust Kalman − LightGBM = −0.5300 [−1.0424, −0.1462] µg/m³. Robust Kalman − Kalman = −0.0082 [−0.0663, +0.0444], chưa có bằng chứng gain độc lập rõ của outlier branch trên Beijing. Jump − counterpart bằng 0 trên Beijing. Các cặp Favorita không có paired CI trực tiếp với counterpart trong artifact; không gán ý nghĩa thống kê cho khác biệt số điểm gần 0.

## Synthetic — 10 seed evaluation mỗi scenario

RMSE dự báo/latent lấy từ các trace lưu ở cột `at_0_5` (bản thân RMSE không phụ thuộc ngưỡng báo động); precision, recall, FPR và delay dùng ngưỡng đã hiệu chỉnh trên seed calibration tách biệt. `oracle` nghĩa là biết tham số quá trình sinh, **không biết thời điểm jump**; `reference` là cấu hình tham chiếu cố định. N/A nghĩa là metric không xác định khi không có true jump hoặc không có detection, không phải 0.

| scenario | model | parameters | seeds | forecast_rmse_log | latent_rmse | precision | recall | fpr | delay_steps | outlier_false_jump_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stationary | Kalman | oracle | 10 | 0.1995 | 0.0003 | N/A | N/A | 0.0000 | N/A | N/A |
| stationary | Robust Jump Filter | oracle | 10 | 0.1995 | 0.0003 | N/A | N/A | 0.0000 | N/A | N/A |
| gaussian_drift | Kalman | oracle | 10 | 0.2534 | 0.1219 | N/A | N/A | 0.0000 | N/A | N/A |
| gaussian_drift | Robust Jump Filter | oracle | 10 | 0.2534 | 0.1219 | N/A | N/A | 0.0000 | N/A | N/A |
| jump_only | Kalman | oracle | 10 | 0.7308 | 0.7034 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| jump_only | Robust Jump Filter | oracle | 10 | 0.2496 | 0.0821 | 0.7035 | 0.4654 | 0.0045 | 1.0937 | N/A |
| mixed | Kalman | oracle | 10 | 0.2462 | 0.1182 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| mixed | Robust Jump Filter | oracle | 10 | 0.2429 | 0.1101 | 0.5433 | 0.2264 | 0.0017 | 1.6083 | N/A |
| isolated_outliers | Jump Filter | reference | 10 | 0.2436 | 0.0993 | 0.0000 | N/A | 0.0061 | N/A | 1.0000 |
| isolated_outliers | Robust Jump Filter | reference | 10 | 0.2276 | 0.0313 | 0.0000 | N/A | 0.0010 | N/A | 0.0333 |

Trong `jump_only`, robust jump oracle có forecast RMSE log ≈0.2496 so với Kalman oracle ≈0.7308; recall đã hiệu chỉnh ≈0.4654, nên còn bỏ sót nhiều event. Ở `isolated_outliers`, robust reference giảm latent RMSE ≈0.0993→0.0313 và tỷ lệ outlier bị gọi sai là jump 1.000→0.033 so với jump reference. Đây là kết quả điều kiện trên synthetic, không phải bằng chứng jump có ích trên dữ liệu thực.

## Kết luận kỹ thuật

Chốt lõi pipeline one-step online và LightGBM cố định. **Gaussian Kalman residual correction** là adapter gọn có bằng chứng cải thiện RMSE gốc trên Beijing; Favorita 100 chuỗi vẫn lấy LightGBM tĩnh làm mốc tốt hơn trên RMSLE. Bật/tắt adapter phải dựa trên validation riêng từng domain. Robust outlier branch giữ như ablation vì lợi ích độc lập chưa rõ. Giữ two-/three-branch jump filter như giả thuyết và kết quả âm/conditional trong BTL, không triển khai nó như nhánh mặc định. `p_jump=0` chỉ cho thấy không cải thiện mục tiêu validation với dữ liệu, model và grid hiện tại; không chứng minh thế giới thực không có jump.

## Bổ sung toàn bộ Favorita

[Phụ lục 1.782 chuỗi](full_favorita_addendum.md) ghi validation/test đầy đủ và paired interval. Validation chọn LightGBM tĩnh; test exploratory EMA nhỉnh hơn nhẹ. Hai nhánh jump chọn `p_jump=0` và dự báo trùng Gaussian counterparts.

## Bổ sung: train/test với dữ liệu USGS tự thu thập

[Thực nghiệm USGS 2×2](USGS_MODEL_ABLATION.md) dùng toàn bộ 1.782 chuỗi, bốn LightGBM có/không IDs và có/không USGS, cùng seed/split/ngân sách; hai convex ensemble chọn trọng số trên validation. Validation chọn **LightGBM + USGS** trước khi chạy test một lần. Đây là quyết định cho phiên bản feature mới; các bảng Task 01–05 phía trên giữ nguyên phạm vi model cũ.

| model | rmsle_validation | n | rmsle_test | mae |
| --- | --- | --- | --- | --- |
| Seasonal naive | 0.869816 | 892782 | 0.653260 | 96.339911 |
| LightGBM | 0.464533 | 892782 | 0.437766 | 77.815815 |
| LightGBM + USGS | 0.455867 | 892782 | 0.439166 | 78.079079 |
| LightGBM + IDs | 0.486228 | 892782 | 0.449127 | 98.223261 |
| LightGBM + IDs + USGS | 0.481116 | 892782 | 0.451337 | 96.930139 |
| Ensemble (base, USGS) | 0.455867 | 892782 | 0.439166 | 78.079079 |
| Ensemble (IDs, IDs+USGS) | 0.481116 | 892782 | 0.451337 | 96.930139 |

USGS tốt hơn trên validation nhưng test RMSLE tăng +0.001400 (+0.320%), CI 95% [+0.0000646, +0.0028807]. Lượt này chưa xác nhận lợi ích tăng accuracy của dữ liệu thu thập; chưa nâng USGS/IDs thành mặc định.

USGS lag chỉ lấy origin time trước ngày dự báo; availability của revision lịch sử vẫn NOT VERIFIED. Test Favorita đã được xem nên lượt mới là hồi cứu/exploratory. Ensemble có beta=1 trùng nhánh USGS, không có lợi ích ensemble tăng thêm. Metric từ predictions đã lưu được tính lại khi tạo báo cáo; hash code/input/model/validation/test prediction được kiểm tra. Không suy ra động đất là ground-truth latent jump của sales.
