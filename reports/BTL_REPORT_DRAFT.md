# Dự đoán thích nghi trên dữ liệu phi dừng bằng mô hình trạng thái ẩn có drift, jump và outlier

*Adaptive Prediction under Non-Stationarity with Jump-Aware Latent-State Filtering*

## Tóm tắt

Bài tập lớn xây dựng dự báo một bước với phản hồi theo thời gian, kết hợp LightGBM cố định và bộ lọc trạng thái ẩn trên phần dư `log1p(target)`. Hai nguồn thực được đánh giá: 100 chuỗi Favorita ở Task 03, bổ sung 1.782 chuỗi ở lượt exploratory mới, và 12 trạm chất lượng không khí Beijing ở một test đã khóa. Trên Beijing, LightGBM có RMSE PM2.5 20.4207 µg/m³; Robust Kalman có 19.8906, chênh −0.5300 với CI paired block 95% [−1.0424, −0.1462]. Cả hai cấu hình jump Beijing chọn `p_jump=0` và trùng Gaussian counterpart. Favorita Robust Jump cũng chọn `p_jump=0`, trong khi bản jump hai nhánh chọn 0.001 nhưng không thắng LightGBM. Synthetic oracle cho thấy jump có thể giúp khi quá trình sinh chứa jump; recall còn hạn chế. Vì vậy, kết quả thực chưa chứng minh lợi ích tăng thêm của thành phần Lévy jump. [Nguồn số liệu và kiểm toán](results_summary.md).

## 1. Đặt vấn đề

Với chuỗi `g` và thời điểm `t`, mô hình phải phát dự báo trước khi nhận `y_{g,t}`. Favorita dùng ngày; Beijing dùng giờ. Toàn bộ chuỗi/trạm cùng thời điểm được dự báo trước, sau đó nhãn hợp lệ mới cập nhật state cho lần kế tiếp. Đây là backtest một bước có phản hồi, không phải dự báo đồng thời nhiều bước khi nhãn tương lai chưa có.

Sai số lớn có thể do **outlier quan sát** nhất thời (lần đo/ghi nhận bất thường) hoặc **thay đổi latent dynamics** có ảnh hưởng kéo dài. Một residual spike đơn lẻ không phân biệt được hai nguyên nhân. Báo cáo do đó xem nhánh jump như giả thuyết thống kê, không tuyên bố phát hiện bước nhảy vật lý trên dữ liệu quan sát.

## 1.1. Thu thập dữ liệu web bổ sung

Nhóm triển khai chương trình thu thập HTTP API công khai USGS cho động đất trong vùng Ecuador và lân cận, lưu GeoJSON gốc, URL, thời điểm truy xuất và SHA-256; sau đó chuẩn hóa UTC về ngày Ecuador, kiểm tra schema/ID/coordinates, tạo lag/rolling và ghép với doanh số. [Báo cáo thu thập và số lượng thực tế](DATA_COLLECTION.md) có bảng nguồn, quality audit, EDA và lệnh tái lập. Đây là dữ liệu ngoại sinh tự thu thập; nhãn sales vẫn lấy từ Favorita có sẵn. Danh mục USGS hồi cứu có revision, chưa xác minh thời điểm công bố lịch sử; dữ liệu mới chưa tham gia các model/kết quả đã khóa bên dưới.

## 2. Dữ liệu và khai phá dữ liệu

**Favorita.** File local có 3,000,888 dòng, 54 cửa hàng, 33 family và 1.782 chuỗi; Task 03 chọn cố định 100 chuỗi. Tỷ lệ doanh số bằng 0 là 31.3%; median tỷ lệ 0 theo chuỗi là 30.6%, có 53 chuỗi toàn 0. Khác biệt này làm metric pooled dễ bị nhóm doanh số lớn chi phối; vì vậy báo cáo đọc thêm metric trung bình theo chuỗi trong artifact. Chủ Nhật có mean sales 463.1, thứ Năm 283.5 [Hình 1](figures/favorita_weekday.png). Mean tháng 12 là 453.7, tháng 2 là 320.9 sales/dòng ([bảng tháng](../results/eda/sales_by_month.csv)). Ngày có promotion có mean 1137.7, ngày không có 158.2 sales/dòng [Hình 2](figures/favorita_promotion.png); đây là liên hệ thô, không phải tác động nhân quả. Phân bố mean/zero rate của 1.782 chuỗi ở [Hình 3](figures/favorita_series.png). Nguồn dữ liệu local lấy từ mirror được ghi trong README, chưa đối chiếu checksum với Kaggle gốc.

**Beijing.** UCI có 420,768 dòng, 12 trạm theo giờ từ 2013-03-01 đến 2017-02-28. PM2.5 thiếu 8,739 (2.08%) trên toàn bộ nguồn; test thiếu 1,448/69,984. Không biến NA thành 0. Missingness theo trạm/biến/split và tháng được lưu tại [`missingness_by_station.csv`](../results/external_validation/missingness_by_station.csv) và [`missingness_by_month.csv`](../results/external_validation/missingness_by_month.csv). Trung bình PM2.5 trên train theo trạm trải từ 68.0 đến 87.3 µg/m³; SD theo trạm từ 71.8 đến 86.1 [Hình 4](figures/beijing_train_stations.png). Nhóm `low/middle/high` là tam phân vị trung bình PM2.5 **chỉ từ train**, không chọn lại theo test. Trên toàn bộ nguồn mô tả, giờ 22 có mean cao nhất 88.9 và giờ 7 thấp nhất 73.3; tháng 12 cao nhất 104.6 µg/m³ [Hình 5](figures/beijing_seasonality.png). Các số toàn nguồn chỉ để mô tả, không tạo feature hậu nghiệm.

Residual LightGBM log trung bình theo ngày có mean +0.0016 trên validation và -0.0079 trên test; đồ thị trung bình trượt bảy ngày ở [Hình 6](figures/beijing_rolling_residual.png). Những đoạn biến động là dấu hiệu sai số dự báo, không có nhãn sự kiện jump để xác thực. [Danh mục hình và nguồn](figure_index.md) ghi rõ mục đích từng hình.

Chênh RMSE test so với LightGBM tĩnh theo nhóm trạm đã định nghĩa từ train (âm là cải thiện), chỉ là mô tả hậu nghiệm và không dùng chọn model:

| group | LightGBM + Kalman | LightGBM + Robust Kalman |
| --- | --- | --- |
| high | -0.626 | -0.641 |
| low | -0.212 | -0.240 |
| middle | -0.659 | -0.644 |

## 3. Phương pháp và thiết kế thuật toán

LightGBM từ thư viện được huấn luyện **một lần** trên train để ước lượng `f_φ(x_{g,t}) ≈ log(1+y_{g,t})`; dự báo này giữ cố định trong backtest. Phần dự án tự triển khai là `LevyOU`, `JumpAwareFilter`, `RobustJumpFilter`, EMA và bộ đánh giá online. Trạng thái ẩn một chiều `θ_{g,t}` sửa bias trên thang log:

$$
z_{g,t}=\log(1+y_{g,t})=f_\phi(x_{g,t})+\theta_{g,t}+\epsilon_{g,t},\qquad \epsilon_{g,t}\sim\mathcal N(0,R),
$$

$$
\theta_{g,t}=\mu+\rho(\theta_{g,t-1}-\mu)+\eta_{g,t}+I_{g,t}J_{g,t},\quad
\eta_{g,t}\sim\mathcal N(0,q),\ I_{g,t}\sim\mathrm{Bernoulli}(p_J),\ J_{g,t}\sim\mathcal N(0,q_J).
$$

`p_J` là xác suất trên **một bước thời gian của dataset** (ngày hoặc giờ); không so trực tiếp giá trị giữa hai tần suất. Đây là OU rời rạc hóa với xấp xỉ jump compound-Poisson, không phải một Lévy process thuần và không suy ra quá trình sinh thật từ quan sát.

## 4. Công thức bộ lọc

Trước khi thấy nhãn tại `t`, dùng `m^- = μ+ρ(m_{t-1}−μ)` và `P^-_0=ρ²P_{t-1}+q`. Nhánh jump có `P^-_J=P^-_0+q_J`; nhánh outlier giữ `P^-_O=P^-_0` nhưng tăng variance quan sát thành `R_O>R`. Các prior hypothesis là `π_0=1−p_J−p_O`, `π_J=p_J`, `π_O=p_O`. Với residual `r_t=z_t−f_φ(x_t)` và mỗi nhánh `k`:

$$
S_k=P^-_k+R_k,\quad
\ell_k=\log\pi_k-\tfrac12\left[\log(2\pi S_k)+\frac{(r_t-m^-)^2}{S_k}\right],\quad
w_k=\frac{e^{\ell_k-\max_j\ell_j}}{\sum_j e^{\ell_j-\max_h\ell_h}}.
$$

$$
K_k=\frac{P^-_k}{S_k},\quad m_k=m^-+K_k(r_t-m^-),\quad P_k=(1-K_k)P^-_k,
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

| model | n_valid | rmsle | mae |
| --- | --- | --- | --- |
| seasonal_naive | 50100 | 0.56698 | 110.79490 |
| lgbm_ema | 50100 | 0.43040 | 97.63515 |
| LightGBM | 50100 | 0.41987 | 102.04009 |
| Kalman | 50100 | 0.42028 | 102.36786 |
| Jump Filter | 50100 | 0.42140 | 102.27453 |
| Robust Kalman | 50100 | 0.42027 | 101.37252 |
| Robust Jump Filter | 50100 | 0.42027 | 101.37217 |

**Beijing — 12 trạm, 68.536 nhãn test hợp lệ** ([CSV gốc](../results/external_validation/test_metrics.csv), [Hình 7](figures/beijing_test_models.png)):

| model | n_valid | rmse_pm25 | mae | rmse_log |
| --- | --- | --- | --- | --- |
| Seasonal naive | 68536 | 93.78176 | 61.68840 | 1.29739 |
| LightGBM | 68536 | 20.42065 | 10.26198 | 0.29332 |
| LightGBM + EMA | 68536 | 20.41725 | 10.26056 | 0.29277 |
| LightGBM + Kalman | 68536 | 19.89883 | 10.04509 | 0.29305 |
| LightGBM + Robust Kalman | 68536 | 19.89063 | 10.04049 | 0.29248 |
| LightGBM + Jump Filter | 68536 | 19.89883 | 10.04509 | 0.29305 |
| LightGBM + Robust Jump Filter | 68536 | 19.89063 | 10.04049 | 0.29248 |

Cấu hình chọn trên validation:

| dataset | model | rho | q | R | p_jump | p_outlier | R_out/R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Favorita | Kalman | 0.90000 | 0.00000 | 0.60000 | 0.00000 | 0.00000 | N/A |
| Favorita | Jump Filter | 0.90000 | 0.00000 | 0.60000 | 0.00100 | 0.00000 | N/A |
| Favorita | Robust Kalman | 0.90000 | 0.00100 | 0.60000 | 0.00000 | 0.00500 | 25.00000 |
| Favorita | Robust Jump Filter | 0.90000 | 0.00100 | 0.60000 | 0.00000 | 0.00500 | 9.00000 |
| Beijing | LightGBM + Kalman | 0.80000 | 0.00200 | 0.04780 | 0.00000 | 0.00000 | N/A |
| Beijing | LightGBM + Robust Kalman | 0.80000 | 0.00200 | 0.04780 | 0.00000 | 0.02000 | 25.00000 |
| Beijing | LightGBM + Jump Filter | 0.80000 | 0.00200 | 0.04780 | 0.00000 | 0.00000 | N/A |
| Beijing | LightGBM + Robust Jump Filter | 0.80000 | 0.00200 | 0.04780 | 0.00000 | 0.02000 | 25.00000 |

Trên Beijing, Robust Kalman − LightGBM có ΔRMSE -0.5300 µg/m³, CI 95% [-1.0424, -0.1462]. Robust Kalman − Kalman chỉ -0.0082, CI [-0.0663, +0.0444]; chưa có bằng chứng lợi ích riêng rõ của nhánh outlier trên test Beijing. Jump hai nhánh và robust jump đều `p_jump=0`, dự báo trùng counterpart Gaussian tương ứng, nên **không có cải thiện thêm từ jump**. Trên Favorita, jump hai nhánh thực tế chọn `p_jump=0.001` nhưng validation/test đều kém Kalman/LightGBM; robust jump chọn 0 và khác robust Kalman do `R_out` khác, không do jump. Không so độ lớn RMSLE Favorita với RMSE PM2.5 Beijing vì đơn vị khác nhau.

## 7. Synthetic có nhãn jump/outlier

Những thí nghiệm này có nhãn sinh sẵn nên mới tính precision/recall/FPR/delay. Giá trị oracle biết tham số generator, không biết jump times; không đại diện cho cách tune trên dữ liệu thực. Bảng dưới ghi RMSE dự báo/latent từ trace lưu ở cột `at_0_5` (RMSE không phụ thuộc ngưỡng báo động) và chỉ số detection tại ngưỡng calibration tách seed ([artifact](../results/model_review/synthetic_comparison.csv), [Hình 8](figures/synthetic_cases.png)):

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

Trong `jump_only`, robust jump oracle có forecast RMSE log 0.2496 so với Kalman oracle 0.7308, nhưng recall 0.465. Ở `mixed`, recall chỉ 0.226. Với outlier đơn lẻ và cấu hình reference, robust jump giảm latent RMSE từ 0.0993 xuống 0.0313, tỷ lệ gọi sai outlier là jump từ 1.000 xuống 0.033. Các kết quả này hỗ trợ vai trò mô hình hóa có điều kiện trên synthetic, chưa xác nhận jump thực.

## 8. Giới hạn và diễn giải

Task 03 Favorita chỉ 100 chuỗi; lượt mới bao phủ 1.782 chuỗi nhưng vẫn exploratory, nguồn local từ mirror; test đã được phân tích lặp lại. Promotion cùng ngày chưa có bằng chứng thời điểm công bố. Beijing có đủ 12 trạm nhưng chỉ kiểm chứng one-step feedback trong một thành phố/giai đoạn; công bố phép đo `t−1` chưa được xác minh. Cả hai tập thực thiếu ground-truth latent jump và outlier; residual/posterior là biến mô hình, không phải nhãn vật lý. Paired interval phản ánh biến động theo block thời gian trong test cố định, không chứng minh nhân quả. Cấu hình chọn `p_jump=0` chỉ nói nhánh jump không cải thiện **tiêu chí validation trong phạm vi model, feature, split và grid đã thử**, không chứng minh thế giới thực không tồn tại jump.

## 9. Kết luận và hướng mở rộng

Đóng góp ở mức BTL là pipeline causal online trên hai domain, bộ lọc Gaussian/jump/outlier tự cài đặt với limiting-case tests, benchmark có provenance và kết quả âm được báo cáo thẳng. Kiến trúc chốt là **LightGBM cố định + adapter residual Gaussian tùy chọn**, quyết định bật adapter theo validation của từng domain: trên Beijing Kalman cải thiện RMSE, còn trên Favorita toàn bộ 1.782 chuỗi LightGBM tĩnh có RMSLE validation tốt nhất. Robust outlier là ablation hợp lý; jump-aware filter giữ như giả thuyết nghiên cứu. Một nghiên cứu kế tiếp cần dữ liệu có nhãn thay đổi hoặc thí nghiệm can thiệp và giao thức availability rõ hơn trước khi đánh giá lợi ích jump; không dùng test hiện tại để thiết kế lại mô hình.

## 10. Tái lập và tài liệu tham khảo

Lệnh đã chạy ở Task 05: `.\\.venv\\Scripts\\python.exe -m scripts.build_report` hai lần và `.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v`; script chỉ đọc `results/`/`data/raw/`, ghi trong `reports/`. Hash các artifact đầu vào được kiểm tra không đổi trước/sau. Các lệnh train/test gốc nằm trong README và artifact Task 02–04; không chạy lại Beijing test. Lượt Favorita toàn bộ chạy `python -m scripts.run_full_favorita --stage validate`, sau đó `--stage test` một lần.

1. [Kaggle, Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data). File Favorita local lấy từ mirror được ghi trong [README](../README.md).
2. [UCI, Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing), DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), CC BY 4.0.
3. Ke và cộng sự, [*LightGBM: A Highly Efficient Gradient Boosting Decision Tree*](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), NeurIPS 2017.
4. Kalman, [*A New Approach to Linear Filtering and Prediction Problems*](https://doi.org/10.1115/1.3662552), *Journal of Basic Engineering*, 1960.

Mã filter đang dùng nằm trong `src/jump_filter.py` và `src/robust_jump_filter.py`; pipeline báo cáo không import FilterPy hoặc River.

## 11. Bổ sung thực nghiệm toàn bộ Favorita

Một lượt train/validation/test riêng trên toàn bộ 1.782 chuỗi được trình bày trong [phụ lục](full_favorita_addendum.md). RMSLE validation chọn LightGBM tĩnh; test exploratory cho EMA nhỉnh hơn nhẹ nhưng không đổi quyết định sau khi xem test. Cả hai nhánh jump chọn `p_jump=0`, trùng dự báo Gaussian counterparts. Các con số 100 chuỗi ở phần chính giữ nguyên provenance Task 03.

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
