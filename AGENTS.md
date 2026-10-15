# Codex instructions — levy-drift-forecasting

## Mục tiêu

Triển khai BTL **Jump-Aware Adaptive Learning for Non-Stationary Demand Forecasting** bằng Python, chạy trên CPU và làm việc trực tiếp trong VS Code. Bài toán là **dự báo doanh số ngày tiếp theo** cho từng cặp `(store_nbr, family)` của bộ dữ liệu Corporación Favorita. Sau khi dự báo ngày `t`, hệ thống mới được dùng doanh số thực tế của ngày đó để cập nhật mô hình cho ngày `t+1`.

Sản phẩm cần có: pipeline dữ liệu tái lập được, phân tích dữ liệu có ý nghĩa, LightGBM baseline, các baseline thích nghi, một **Bayesian Jump-Aware Filter** tự triển khai, bộ thực nghiệm so sánh, biểu đồ và bảng kết quả. Không làm RL, POMDP, deep learning, giao diện web hay lý thuyết regret ở giai đoạn này.

## Công nghệ và mã tham khảo

- Python 3.11+, NumPy, Pandas, LightGBM, scikit-learn, Matplotlib; dùng PyArrow để lưu Parquet nếu có sẵn. Chạy CPU, đặt seed và số luồng qua config.
- Dùng các file `.py` và CLI. **Không tạo Jupyter Notebook**.
- Repo tham khảo chính: https://github.com/avsngh-git/Favorita-store-sales-prediction . Đọc `src/favorita_forecasting/`, ưu tiên tận dụng ý tưởng/code hợp lệ từ `data.py`, `features.py`, `splitting.py`, `metrics.py`, `evaluation.py`. Kiểm tra kỹ logic và leakage trước khi dùng. Repo này **chưa tái lập đầy đủ** pipeline LightGBM của notebook cũ, nên tự triển khai bước train nếu cần.
- Baseline lọc: https://github.com/rlabbe/filterpy . Baseline online/drift detection, nếu đủ thời gian: https://github.com/online-ml/river . Đây là thư viện tham khảo/độc lập, không cần clone vào source tree.
- Nếu sao chép code từ repo bên ngoài, giữ thông tin bản quyền/giấy phép MIT và ghi rõ nguồn trong README. Không fork hoặc chép nguyên dự án khi không cần.
- Môi trường có thể không có Internet. Không tự động tải dataset, `git clone` hay cài đặt online nếu không được phép. Nếu thiếu file/thư viện, nêu rõ và hoàn thành được phần nào thì làm phần đó.

## Dữ liệu và định nghĩa dự báo

Dataset: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

Đọc dữ liệu từ `data/raw/` (tối thiểu `train.csv`; tận dụng `stores.csv`, `holidays_events.csv`, `oil.csv` khi có). **Không dùng `test.csv` của Kaggle làm tập đánh giá**, vì không có nhãn; tự chia tập có nhãn theo ngày:

- Train: đến hết `2015-12-31`.
- Validation: `2016-01-01` đến `2016-03-31`.
- Test: `2016-04-01` đến ngày cuối của `train.csv` (dự kiến `2017-08-15`).

Dùng các mốc này làm cấu hình mặc định; kiểm tra phạm vi ngày thực tế trước khi chạy. Đây là **one-step-ahead online backtest**, không phải bài toán dự báo đồng thời 16 ngày của cuộc thi Kaggle.

Định nghĩa target: `z = log1p(sales)`. Model dự đoán `z`; khi báo cáo trên doanh số dùng `max(0, expm1(z_pred))`. Đánh giá RMSLE và MAE trên thang doanh số, thêm RMSE trên thang log nếu cần. Cần báo cáo cả metric toàn bộ mẫu và metric trung bình theo chuỗi để không che khuất nhóm bán ít.

Quy tắc bắt buộc để tránh leakage:

1. Tách train/validation/test **theo ngày**, tuyệt đối không random split.
2. Feature tại thời điểm dự báo ngày `t` chỉ dùng dữ liệu đã biết trước ngày đó: lịch, thông tin cửa hàng, khuyến mại dự kiến (nếu đã có trong dữ liệu), `sales` lag và rolling statistics đã `shift(1)` theo từng chuỗi.
3. Không dùng `sales[t]`, doanh số tổng hợp ngày `t`, hoặc `transactions[t]` làm feature của dự báo ngày `t`. Giá dầu chỉ dùng giá đã công bố trước thời điểm dự báo, chẳng hạn `oil_price_lag1`; không tự ý dùng giá dầu cùng ngày.
4. Dự đoán **tất cả chuỗi trong ngày `t` trước**, rồi mới dùng nhãn thực của ngày `t` để cập nhật filter. Không cho chuỗi dự đoán sau nhìn thấy target cùng ngày của chuỗi trước.
5. Mọi encoder, phép điền dữ liệu dựa trên thống kê và model hyperparameter đều fit/tune chỉ từ train/validation. Tuyệt đối không dùng test để chọn tham số. Feature lag trong test được dùng nhãn ngày trước theo đúng giả định feedback online.
6. Không coi mọi ngày thiếu bản ghi là doanh số bằng 0 một cách mặc định; phân biệt missing và zero sau khi kiểm tra dữ liệu. Không xóa âm thầm chuỗi ngắn hoặc toàn zero.

## Mô hình cần xây dựng

Đặt `g = (store_nbr, family)`, `x[g,t]` là các feature ngày `t`, `z[g,t] = log1p(sales[g,t])`.

**Base predictor:**

```text
f_phi(x[g,t]) = LightGBM prediction on log1p(sales)
```

LightGBM được train offline trên train split và **giữ cố định khi backtest**. Baseline này và tất cả mô hình thích nghi phải dùng cùng một base prediction. Không train lại LightGBM trong vòng lặp từng ngày.

**Latent correction (MVP 1D):**

```text
z[g,t] = f_phi(x[g,t]) + theta[g,t] + observation_noise

theta[g,t] = mu + rho * (theta[g,t-1] - mu)
             + Gaussian_drift + jump_indicator * jump_size
```

- `Gaussian_drift ~ N(0, q)`.
- `jump_indicator ~ Bernoulli(p)`; `p = 1 - exp(-lambda * delta_t)`.
- `jump_size ~ N(0, q_jump)` độc lập với Gaussian drift.
- Observation noise có variance `R > 0` trên thang log; mặc định `mu=0` và `delta_t=1` ngày. Ràng buộc `0 <= rho < 1`, `q >= 0`, `q_jump >= 0`, `0 <= p < 1`.

Đây là **mô hình Lévy-driven OU rời rạc hóa với compound-Poisson jump approximation**, không gọi nó là một Lévy process thuần túy. Đơn vị của mọi variance phải nhất quán.

**Jump-Aware Filter:** mỗi chuỗi giữ posterior mean `m` và variance `P` cho `theta`; trước khi quan sát `z[g,t]` phải tính dự đoán dùng prior mean `m_minus = mu + rho * (m - mu)`.

Với `P_base = rho^2 * P + q` và residual `r = z - f_phi(x)`, tính hai nhánh:

```text
No-jump: P0_minus = P_base
Jump:    P1_minus = P_base + q_jump
S[k]   = Pk_minus + R
L[k]   = NormalPDF(r; m_minus, S[k])
w1     = p*L[1] / ((1-p)*L[0] + p*L[1])
w0     = 1 - w1
K[k]   = Pk_minus / S[k]
m[k]   = m_minus + K[k]*(r - m_minus)
P[k]   = (1-K[k])*Pk_minus
m      = w0*m[0] + w1*m[1]
P      = sum_k w[k]*(P[k] + (m[k]-m)^2)
```

Tính mixture weights bằng log-likelihood/log-sum-exp để tránh underflow; chặn variance bởi epsilon dương. Lưu `w1` (estimated jump probability), prior/corrected latent estimate và forecast error để phân tích. Đây là **approximate two-component Bayesian filter** (moment matching), không mô tả là exact full Bayesian filtering qua nhiều ngày. Với `p=0` hoặc `q_jump=0`, kết quả phải phù hợp Gaussian Kalman counterpart trong điều kiện tương ứng.

Hyperparameters `rho, q, R, p, q_jump` phải được xác định bằng train/validation, không chọn bằng test. Bắt đầu cấu hình global dùng chung cho mọi chuỗi; không tạo hàng nghìn model riêng. Mỗi chuỗi có state `(m, P)` riêng, đặt prior thống nhất và có cơ chế fallback cho chuỗi ít dữ liệu.

## Baselines và kiểm chứng

Triển khai và đánh giá trên cùng test period, cùng information budget:

1. Seasonal naive: doanh số cùng thứ tuần trước, với fallback xác định rõ khi thiếu lag.
2. LightGBM cố định.
3. LightGBM + EMA trên residual log-scale (chỉ cập nhật sau khi nhận nhãn).
4. LightGBM + Kalman 1D (Gaussian drift, không jump).
5. LightGBM + Jump-Aware Filter (mô hình chính).

Nếu còn thời gian, thêm một robust-noise baseline (ví dụ clip residual hoặc Student-t inspired) để kiểm tra liệu kết quả chỉ do xử lý outlier. River/ADWIN là **tùy chọn**, không được làm chậm MVP.

Thực hiện thêm synthetic controlled experiment bằng cách dùng feature thực hoặc feature sinh đơn giản, sau đó tạo `theta_t` và target bằng quá trình trên với seed cố định. Các case bắt buộc: stationary, Gaussian drift only, jump only, mixed drift+jump, isolated outliers. Chỉ ở synthetic experiment mới báo cáo jump detection precision/recall, false alarm, detection delay, latent tracking RMSE vì có ground truth. **Không được coi jump time hoặc theta trên dữ liệu Favorita là ground truth.** Không dùng nhãn synthetic thay cho real-data results.

## Data mining và báo cáo

Tạo script EDA chạy được không cần notebook và lưu hình/bảng vào `results/`:

- Số dòng, thời gian bao phủ, số cửa hàng và nhóm sản phẩm, missing/zero sales, phân phối doanh số.
- Seasonality theo thứ/ngày/tháng, quan hệ với promotion và holidays.
- Chênh lệch giữa các chuỗi, có thể phân nhóm theo doanh số trung bình, biến thiên và tỷ lệ ngày zero.
- Rolling error/residual của LightGBM trên validation/test; nêu các giai đoạn suy giảm chất lượng dự đoán. Không mặc định residual spike chính là regime shift.

Tạo bảng so sánh các baseline trên cùng test set và biểu đồ: forecast-vs-actual, rolling error, latent correction, jump posterior, error quanh một vài giai đoạn biến động. Lưu kết quả ra CSV/PNG và cấu hình chạy; không ghi số liệu giả vào README.

## Cấu trúc tối thiểu

```text
levy-drift-forecasting/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/raw/                  # data Kaggle, không commit
├── data/processed/            # file sinh ra, không commit
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── baselines.py
│   ├── levy_process.py
│   ├── jump_filter.py
│   └── evaluate.py
├── scripts/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── run_eda.py
│   ├── run_baselines.py
│   ├── run_jump_filter.py
│   └── run_experiments.py
├── configs/
├── tests/
└── results/                  # artifact sinh ra, không commit
```

Có thể điều chỉnh tên file khi hợp lý nhưng giữ module tách bạch và CLI ổn định. Dùng `argparse`, `logging` và config JSON/YAML đơn giản; không dựng framework phức tạp.

## Thứ tự triển khai

1. Kiểm tra repo hiện có và dữ liệu local; ghi nhận thiếu gì. Không ghi đè code có sẵn và không xóa file người dùng.
2. Tạo project skeleton, dependencies, `.gitignore`, README hướng dẫn chạy offline.
3. Tạo data pipeline, chronological split, feature engineering và EDA; test leakage.
4. Train LightGBM và chạy seasonal naive; lưu metrics và prediction files.
5. Cài EMA + Kalman baseline, dùng cùng dự đoán LightGBM.
6. Viết `levy_process.py` và `jump_filter.py`; trước hết chạy synthetic tests, sau đó chạy real-data backtest.
7. Chạy ablation `p=0`, `q=0`, `p>0 & q>0`; tổng hợp bảng và hình. Nếu chưa có Kaggle data, chỉ chạy synthetic/smoke tests và ghi rõ phần real-data đang chờ dữ liệu.
8. Hoàn thiện README: mô tả bài toán, data requirements, cách chạy, phương pháp, kết quả *đã thực sự đo được*, giới hạn nghiên cứu và nguồn code tái sử dụng.

Cung cấp chế độ chạy nhanh với số chuỗi giới hạn (ví dụ `--max-series 100`) và `--seed`; mặc định vẫn hỗ trợ chạy toàn bộ dữ liệu. Khi chọn subset, chọn theo train data và cố định cùng danh sách chuỗi qua các splits. Không tự khởi chạy full training tốn thời gian nếu chưa cần; trước hết chạy smoke test.

## Kiểm thử bắt buộc

- Features không truy cập target tương lai; split không giao nhau.
- Forecast ngày `t` không đổi khi sửa nhãn ngày `t` hoặc các ngày sau (trước bước update).
- Filter xử lý `p=0`, `q_jump=0`, residual cực lớn, variance nhỏ và chuỗi mới, không sinh NaN/inf.
- Khi `p=0`, Jump-Aware Filter trùng kết quả Kalman với cùng parameters, init state và observations.
- Synthetic generator có seed tái lập; jump event/count và kích thước noise đúng định nghĩa mô hình.
- Metrics thống nhất giữa baselines; predictions được lưu kèm `date`, `store_nbr`, `family`, `actual`, `prediction`, `model`.
- Unit tests chạy bằng `python -m pytest -q` hoặc `python -m unittest discover -s tests` theo dependencies đã chọn; ghi lệnh thực tế trong README.

## Cách làm việc và tiêu chí hoàn thành

Làm từng bước, chạy kiểm thử sau mỗi bước, sửa lỗi trước khi chuyển sang bước kế tiếp. Không thay đổi mục tiêu hay thêm tool/framework nếu không thực sự cần. Không commit dữ liệu thô, credentials, `.venv/`, model binaries hay artifact lớn. Không `git push` khi chưa được yêu cầu.

Kết thúc công việc phải có một đường chạy từ dữ liệu local đến bảng metrics và biểu đồ bằng CLI trên CPU, cùng mã nguồn filter tự viết và các test quan trọng. Nếu chưa có dataset, vẫn bàn giao project structure, synthetic experiment và test đang chạy được; ghi rõ những bước chưa thể kiểm chứng trên dữ liệu thật. Không tự nhận kết quả tốt hơn baseline khi chưa đo được.
