# Bổ sung: một lượt Favorita toàn bộ 1.782 chuỗi

Lượt này được chạy theo yêu cầu trực tiếp sau khi bản Task 05 ban đầu đã hoàn thành. Nó train một LightGBM mới trên toàn bộ 1.945.944 dòng train; chọn cấu hình filter trên 162.162 dòng validation; ghi [khóa cấu hình](../results/full_favorita_exploratory/selected_configs.json) trước khi chạy test một lần trên 892.782 dòng. Kết quả nằm riêng trong [`results/full_favorita_exploratory`](../results/full_favorita_exploratory/report.md), không thay thế các artifact Task 01–04. Khoảng test đã được xem trong các task trước nên đây là **exploratory**, không phải xác nhận blind độc lập.

Nguồn `train.csv` SHA-256 `99a1b7f4241821df10fa71f23abdd71c634f71f901e3cab1400c0f0c583f862c` và model SHA-256 `9c3a82949a0d05aab558339d5a8b29caf2a19f09fa14c93cfa4ce7bef04bc2b6` đã đối chiếu file local. Seed 42, 4 CPU threads, LightGBM tối đa 500 cây, early stopping 50 vòng trên validation, best iteration 110. Cùng bộ feature causal hiện có; `onpromotion[t]` cần giả định kế hoạch đã biết trước ngày t vì file không có thời điểm công bố. Dự báo toàn bộ chuỗi mỗi ngày trước khi cập nhật nhãn ngày đó. State khởi tạo mean 0, variance 1 ở đầu mỗi split. Validation dùng chung cho LightGBM early stopping và chọn filter; điểm validation là tiêu chí chọn, không phải đánh giá độc lập. Grid gồm 39 cấu hình adapter, chọn **RMSLE validation pooled** với tie break theo thứ tự candidate; [mọi thử nghiệm validation](../results/full_favorita_exploratory/validation_grid.csv) được lưu.

## Điểm validation và test

| model | n | rmsle_validation | rmsle_test | mae | mean_series_rmsle |
| --- | --- | --- | --- | --- | --- |
| Seasonal naive | 892782 | 0.86982 | 0.65326 | 96.33991 | 0.61422 |
| LightGBM | 892782 | 0.46453 | 0.43777 | 77.81582 | 0.40014 |
| LightGBM + EMA | 892782 | 0.46559 | 0.43698 | 75.86602 | 0.39895 |
| LightGBM + Kalman | 892782 | 0.47960 | 0.43843 | 77.86005 | 0.40070 |
| LightGBM + Robust Kalman | 892782 | 0.47511 | 0.43841 | 77.85881 | 0.40069 |
| LightGBM + Jump Filter | 892782 | 0.47960 | 0.43843 | 77.86005 | 0.40070 |
| LightGBM + Robust Jump Filter | 892782 | 0.47511 | 0.43841 | 77.85881 | 0.40069 |

Validation chọn LightGBM tĩnh (RMSLE 0.46453) trước EMA (0.46559), Robust Kalman (0.47511) và Kalman (0.47960). Vì vậy kiến trúc mặc định cho Favorita là LightGBM tĩnh. Trên test exploratory, EMA có RMSLE thấp hơn LightGBM 0.00078; đây là quan sát hậu nghiệm, không dùng để chọn lại sau test.

## Sai khác theo cặp và cấu hình jump

Chênh lệch là **model − reference** trên RMSLE, âm nghĩa là tốt hơn. Paired bootstrap lấy 18 block thời gian 28 ngày, giữ mọi chuỗi cùng block, 500 lần với seed 42; CI chỉ mô tả test cố định sau chọn cấu hình.

| model | reference | delta_rmsle | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| Seasonal naive | LightGBM | 0.21549354 | 0.15814530 | 0.29042454 |
| LightGBM + EMA | LightGBM | -0.00078313 | -0.00141628 | -0.00021592 |
| LightGBM + Kalman | LightGBM | 0.00066647 | -0.00000010 | 0.00201716 |
| LightGBM + Robust Kalman | LightGBM | 0.00064820 | -0.00000013 | 0.00196227 |
| LightGBM + Robust Kalman | LightGBM + Kalman | -0.00001828 | -0.00005496 | -0.00000000 |
| LightGBM + Jump Filter | LightGBM | 0.00066647 | -0.00000010 | 0.00201716 |
| LightGBM + Jump Filter | LightGBM + Kalman | 0.00000000 | 0.00000000 | 0.00000000 |
| LightGBM + Robust Jump Filter | LightGBM | 0.00064820 | -0.00000013 | 0.00196227 |
| LightGBM + Robust Jump Filter | LightGBM + Robust Kalman | 0.00000000 | 0.00000000 | 0.00000000 |

Cả hai cấu hình jump chọn `p_jump=0`. Dự báo lưu cho từng jump và Gaussian counterpart **trùng chính xác 892.782/892.782 hàng** với cùng rho/q/R và outlier parameters. Vì vậy lượt toàn bộ không có đóng góp riêng từ jump. Robust Kalman giảm RMSLE so với Kalman chỉ khoảng 0.000018; chưa đủ cơ sở đổi kiến trúc mặc định Favorita. Dữ liệu không có ground-truth jump hoặc thời gian công bố promotion; `p_jump=0` trong grid không chứng minh thế giới thực không có jump.

Các [dự báo đã lưu](../results/full_favorita_exploratory/test_predictions.csv.gz), [điểm test](../results/full_favorita_exploratory/test_metrics.csv) và [paired bootstrap](../results/full_favorita_exploratory/paired_bootstrap.csv) đã được đọc lại để tính điểm, kiểm tra đủ khóa, chiều dấu CI và giới hạn `p_jump=0`. Hash mọi artifact của lượt mới được giữ nguyên khi sinh báo cáo.
