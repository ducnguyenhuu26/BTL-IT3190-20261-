# Quy trình thu thập, làm sạch và khai phá dữ liệu Green Taxi

## Bài toán

Dự báo **số chuyến Green Taxi đón khách ngày kế tiếp tại từng khu vực của NYC**. Nhãn là chuyến đã hoàn thành và được ghi nhận trong dữ liệu TLC. Đây là đại lượng phản ánh hoạt động dịch vụ thực tế, không phải số yêu cầu đi xe chưa được phục vụ.

Phần dữ liệu là một sản phẩm chính của dự án: crawler lấy nguồn từ website TLC, lưu dấu vết thu thập, làm sạch chuyến, xây cơ sở dữ liệu SQLite và tạo bảng ngày–khu vực cho khai phá dữ liệu và dự báo.

## Nguồn và cách thu thập

Nguồn là [trang NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). TLC công bố Green Taxi Trip Records theo tháng ở định dạng Parquet. Tài liệu TLC nêu dữ liệu thường được công bố trễ khoảng hai tháng và được gửi từ các nhà cung cấp công nghệ; TLC không bảo đảm mọi bản ghi đều chính xác.

`scripts/collect_tlc_green.py` gửi HTTP request đến trang danh mục rồi parse các thẻ liên kết HTML có tên `green_tripdata_YYYY-MM.parquet`. Trong snapshot 2023-01 đến 2025-12, trang có 27/36 liên kết tháng cần lấy. Chín tháng cũ không còn xuất hiện trên trang; pipeline dùng URL archive pattern của TLC được khai báo trong config và đánh dấu riêng các tháng này là `configured_archive_template`. Báo cáo không gộp link tìm thấy trong HTML với URL fallback.

Các tệp Parquet gốc được lưu ngoài Git. Manifest lưu URL, thời điểm lấy, kích thước, SHA-256 và hash phản hồi HTML. Tệp có sẵn chỉ được dùng lại sau khi đối chiếu SHA-256. [Manifest dữ liệu gốc](../provenance/tlc_green_manifest.json) bảo toàn dấu vết của các tệp tháng; [crawl manifest](../provenance/tlc_green_crawl_manifest.json) lưu hash trang HTML, các link thu được và các tháng fallback.

## Làm sạch và kiểm soát chất lượng

Mỗi tệp chỉ đọc hai cột phục vụ nhãn: `lpep_pickup_datetime` và `PULocationID`. Một dòng được giữ khi thời gian pickup parse được và nằm đúng tháng của tệp, đồng thời mã pickup zone là số nguyên từ 1 đến 263.

| Chỉ tiêu | Số lượng |
|---|---:|
| Tệp tháng | 36 |
| Dòng chuyến ban đầu | 2.038.653 |
| Dòng hợp lệ sau lọc | 2.032.098 |
| Dòng bị loại | 6.555 |
| Pickup timestamp bị thiếu | 0 |
| Dòng lệch tháng trong tệp | 553 |
| Dòng có pickup zone không hợp lệ | 6.011 |

Hai số đếm lý do loại có giao nhau ở 9 dòng, vì vậy tổng dòng loại được tính bằng số dòng thô trừ số dòng giữ lại, không cộng hai lý do. Trước khi nhận file tải mới, collector xác nhận Parquet đọc được, có footer, có hai cột bắt buộc và có dòng dữ liệu. Khi xử lý, SHA-256 được kiểm tra lại; builder so số dòng đã đọc với metadata của Parquet.

## Tích hợp và cơ sở dữ liệu

`scripts/build_tlc_database.py` đọc theo batch và tạo SQLite tại `data/processed/tlc_green/2026-10-16/green_taxi.sqlite`. Cơ sở dữ liệu có năm bảng:

| Bảng | Nội dung và khóa |
|---|---|
| `collection_run` | Thời điểm snapshot, trang nguồn và hash HTML, config, manifest cùng tổng số dòng. |
| `source_file` | Một bản ghi mỗi tháng, gồm URL, kiểu link HTML/fallback, kích thước, hash và thống kê chất lượng. |
| `taxi_zone` | Zone ID, borough, tên khu vực, service zone và cờ khu vực được chọn cho mô hình. |
| `trip_record` | Chuyến đã làm sạch; mỗi chuyến truy ngược được về `(source_month, source_row)`. |
| `daily_zone_demand` | Số pickup theo ngày và khu vực, khóa `(pickup_date, zone_id)`, kèm cờ khu vực mô hình. |

Chọn 12 khu vực có nhiều chuyến nhất **chỉ từ năm 2023**, sau đó cố định danh sách cho các năm sau. Giai đoạn 2023-01-01 đến 2025-12-31 có 1.096 ngày; panel mô hình gồm 1.096 × 12 = **13.152 khu vực-ngày**. Có một ô bằng 0; đây là ngày không có chuyến hợp lệ trong tháng nguồn đã tải và kiểm tra đầy đủ. Pipeline không điền 0 khi thiếu tệp nguồn hoặc thiếu ngày do lỗi thu thập.

Database là artifact local và bị Git ignore. `scripts/query_tlc_database.py` cung cấp truy vấn chỉ đọc để xem tổng quan, dữ liệu ngày hoặc tổng hợp tháng; có thể xuất kết quả truy vấn sang CSV. Bảng CSV nén từ cùng snapshot vẫn được giữ làm đầu vào gọn cho script mô hình.

## Khai phá dữ liệu và chuyển sang mô hình

`scripts/report_tlc_green.py` tổng hợp xu hướng theo năm và khu vực, đồng thời tạo hình trực quan. 12 khu vực được chọn ghi nhận 553.564 chuyến năm 2023, 481.227 năm 2024 và 423.062 năm 2025. Đây là biến động số chuyến được ghi nhận; dữ liệu này không tự cho biết nguyên nhân hay lượng nhu cầu tiềm ẩn.

Đặc trưng dự báo gồm khu vực, lịch, lag 1/7/28 ngày và trung bình trượt 7/28 ngày đã dịch để chỉ dùng quá khứ. LightGBM huấn luyện trên 2023 và được giữ cố định. Bộ lọc sai lệch được chọn tham số trên nửa đầu 2024; nửa cuối 2024 dùng quan sát tiếp diễn; năm 2025 là test đã khóa. Dự báo tất cả khu vực của ngày trước, sau đó mới dùng nhãn ngày đó để cập nhật bộ lọc cho ngày kế tiếp.

Kết quả dự báo được trình bày riêng trong [báo cáo nghiên cứu](TLC_GREEN_DEMAND.md). Điểm số là bằng chứng thực nghiệm cho NYC Green Taxi, 12 khu vực và giao thức thời gian này; không phải bảo đảm cho thành phố hoặc giai đoạn khác.

## Chạy lại pipeline dữ liệu

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_tlc_green --stage all
.\.venv\Scripts\python.exe -m scripts.build_tlc_database
.\.venv\Scripts\python.exe -m scripts.query_tlc_database summary
.\.venv\Scripts\python.exe -m scripts.query_tlc_database daily --start-date 2025-01-01 --end-date 2025-01-07
.\.venv\Scripts\python.exe -m scripts.report_tlc_green
```

Stage `all` crawl trang nguồn, tải/kiểm tra cache và tạo panel cho mô hình. Dựng SQLite là stage riêng để có thể tạo lại warehouse từ các file đã lưu. Nhãn Green Taxi trong quy trình này không lấy từ Kaggle.

## Giới hạn và các module mở rộng

- Dữ liệu TLC công khai được phát hành theo tháng với độ trễ. Backtest phản hồi theo ngày giả định đơn vị vận hành có nhật ký chuyến nội bộ mỗi ngày.
- Chín tháng trong snapshot phải dùng URL fallback vì trang HTML không còn liệt kê; manifest đánh dấu rõ nguồn link của từng tháng.
- Mô hình chỉ dùng 12 khu vực chọn từ năm 2023; chưa kết luận cho toàn bộ 263 zone hay dịch vụ ở đô thị khác.
- Các module có thể phát triển tiếp cho demo sản phẩm gồm quản lý phiên bản schema/migration SQLite, job nạp tháng mới có retry và cảnh báo, cùng API/dashboard đọc panel và dự báo. Hiện crawler, warehouse và query CLI đã chạy độc lập; các module vận hành bổ sung này không cần để tái lập kết quả môn học.
