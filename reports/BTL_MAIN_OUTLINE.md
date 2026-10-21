# Đề cương BTL: Thu thập dữ liệu và dự báo nhu cầu Green Taxi

## Bài toán duy nhất

Với lịch sử chuyến đã hoàn thành, dự báo số chuyến Green Taxi đón khách tại mỗi khu vực của NYC trong ngày kế tiếp. Đơn vị quan sát là `(pickup_zone, date)`, nhãn là số chuyến có thời điểm pickup trong ngày và pickup zone tương ứng. Đây là số chuyến được ghi nhận, không phải toàn bộ nhu cầu chưa được phục vụ.

## Quy trình dữ liệu là phần đóng góp trọng tâm

1. **Thu thập web:** crawler tải trang danh mục chính thức của NYC Taxi and Limousine Commission (TLC), parse HTML để tìm link Parquet theo tháng. Khi trang danh mục thiếu một số tháng lịch sử, dùng URL archive pattern có cấu hình và ghi riêng các tháng fallback.
2. **Lưu dữ liệu gốc:** lưu Parquet theo snapshot ngoài Git; manifest lưu URL, thời điểm lấy, kích thước, SHA-256 và hash HTML/config.
3. **Làm sạch:** chỉ đọc timestamp pickup và pickup zone; loại hàng có timestamp không parse được, timestamp nằm ngoài tháng của file hoặc zone không hợp lệ. Lưu số lượng theo tháng và giữ nguyên file thô để truy vết.
4. **Tích hợp và xây cơ sở dữ liệu:** nạp chuyến sạch, nguồn tháng, danh mục khu vực và panel ngày-khu vực vào SQLite. Khóa tự nhiên `(source_month, source_row)` giúp lần ngược dòng dữ liệu về đúng tệp gốc.
5. **Khai phá dữ liệu:** khảo sát xu hướng theo năm, tháng, ngày trong tuần và khu vực; đánh giá missing/zero và chất lượng theo tháng trước khi chọn 12 khu vực từ năm 2023.
6. **Mô hình:** tạo đặc trưng từ lịch sử trước ngày dự báo, huấn luyện LightGBM cố định, rồi so với seasonal naive và hiệu chỉnh phần dư trực tuyến.

## Snapshot hiện có

Nguồn TLC cung cấp bản ghi theo tháng ở định dạng Parquet. Pipeline đã xử lý giai đoạn 2023-01 đến 2025-12: **36 tệp, 2.038.653 dòng thô, 2.032.098 dòng hợp lệ**. Crawler tìm được 27/36 link ngay trong trang HTML ở snapshot này; 9 tháng không còn được liệt kê nên được lấy từ URL archive pattern đã cấu hình và đánh dấu là fallback. Panel mô hình có 12 khu vực × 1.096 ngày = **13.152 dòng**. Database SQLite được lưu local và không commit.

Chi tiết thao tác, schema, audit và lệnh tái lập ở [DATA_PIPELINE.md](DATA_PIPELINE.md). Kết quả mô hình và giới hạn diễn giải ở [TLC_GREEN_DEMAND.md](TLC_GREEN_DEMAND.md).

## Thiết kế thời gian

| Khoảng | Vai trò |
|---|---|
| 2023 | Huấn luyện LightGBM và chọn khu vực từ dữ liệu train |
| 01–06/2024 | Chọn tham số hiệu chỉnh phần dư |
| 07–12/2024 | Theo dõi tiếp diễn, không chọn tham số |
| 2025 | Test khóa |

Tất cả khu vực trong ngày được dự báo trước khi nhãn thực của ngày đó cập nhật bộ lọc. Đặc trưng lag/rolling kết thúc trước ngày dự báo. Tham số không chọn lại theo test.

## Kết quả đã đo

Trên 4.380 khu vực-ngày năm 2025, RMSLE của seasonal naive, LightGBM và LightGBM + Kalman lần lượt là **0,321347**, **0,302883** và **0,263461**. Báo cáo chính có MAE, khoảng bất định và kết quả từng khu vực. Kết luận giới hạn ở tập dữ liệu và giao thức này; TLC xuất bản dữ liệu theo tháng nên giả định phản hồi nhãn hằng ngày chỉ phù hợp đơn vị có log chuyến nội bộ.

## Lệnh demo quy trình

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_tlc_green --stage all
.\.venv\Scripts\python.exe -m scripts.build_tlc_database
.\.venv\Scripts\python.exe -m scripts.query_tlc_database summary
.\.venv\Scripts\python.exe -m scripts.query_tlc_database monthly --year 2025
.\.venv\Scripts\python.exe -m scripts.run_tlc_green_forecast --stage develop
.\.venv\Scripts\python.exe -m scripts.run_tlc_green_forecast --stage test
.\.venv\Scripts\python.exe -m scripts.report_tlc_green
```
