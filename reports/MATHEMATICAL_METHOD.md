# Phương pháp: hiệu chỉnh phần dư phân cấp với hiệu chuẩn theo thời gian

## Giao thức

Dùng 1,297,296 dòng 2013–2014 để train LightGBM hiệu chuẩn 110 cây; chỉ dùng 648,648 dự báo ngoài mẫu trên 2015 để ước lượng độ lệch nhóm. LightGBM chính đã khóa từ lượt Favorita toàn bộ, dự báo validation/test được đọc từ artifact cùng model SHA-256. Chọn phạt ridge bằng RMSLE tháng 1–2/2016; điểm tháng 3 được tính cho mọi ứng viên nhưng không dùng trong quy tắc chọn. Test sử dụng dự báo LightGBM đã lưu và được đánh giá một lần cho phiên bản này.

LightGBM chính đã dùng toàn bộ Q1/2016 để early stopping ở lượt trước, nên tháng 3 không độc lập hoàn toàn với việc chọn base model. Các ngày test Favorita cũng đã được xem trong nghiên cứu trước; mọi kết quả dưới đây là exploratory, không phải xác nhận blind.

Khóa: [`method_selection_lock.json`](method_selection_lock.json); ứng viên: [`hierarchical_development_grid.csv`](tables/hierarchical_development_grid.csv). Chọn candidate 12: $\lambda_0=1000$, $\lambda_s=100$, $\lambda_f=1e+09$, $\lambda_p=300$.

## Mô hình và lời giải toán học

Gọi $r_{g,t}=\log(1+y_{g,t})-f(x_{g,t})$ là phần dư ngoài mẫu ở năm hiệu chuẩn; $g=(s,c)$. Giả sử $r_{g,t}=b_0+u_s+v_c+w_g+\epsilon_{g,t}$ với $\epsilon\sim\mathcal N(0,\sigma^2)$; các hiệu ứng có prior Gaussian tâm 0, phương sai tương ứng $\sigma^2/\lambda_j$. Posterior mean/MAP tối thiểu hóa hàm lồi chặt:

$$L=\sum_g n_g(\bar r_g-b_0-u_s-v_c-w_g)^2+\lambda_0b_0^2+\lambda_s\sum_su_s^2+\lambda_f\sum_cv_c^2+\lambda_p\sum_gw_g^2.$$

Với các hiệu ứng chung đã cố định, nghiệm chính xác theo từng cặp là

$$w_g=\frac{n_g}{n_g+\lambda_p}(\bar r_g-b_0-u_s-v_c).$$

Thế ngược vào $L$ cho bài toán tuyến tính chỉ còn 88 ẩn (1 toàn cục + 54 cửa hàng + 33 nhóm hàng), trọng số hiệu dụng $n_g\lambda_p/(n_g+\lambda_p)$. Ma trận Hessian xác định dương vì mọi $\lambda_j>0$, do đó nghiệm tìm được là **nghiệm tối ưu duy nhất**, không phải kết quả của heuristic. Dự báo là $\widehat z=\max(0,f(x)+\widehat b_0+\widehat u_s+\widehat v_c+\widehat w_g)$.

Vì $z=\log(1+y)\ge0$, phép chiếu một dự báo log $a$ lên $[0,\infty)$ không tăng lỗi bình phương: $(z-\max(0,a))^2\le(z-a)^2$ cho mọi $z\ge0$. Phép chặn về doanh số không âm do đó phù hợp với mục tiêu RMSLE.

Trong mô hình Gaussian đúng và prior đúng, posterior mean tối thiểu hóa kỳ vọng lỗi bình phương log có điều kiện trên dữ liệu hiệu chuẩn. Đây là bảo đảm có điều kiện theo giả định; nó không tự động chứng minh dữ liệu thật tuân theo Gaussian hoặc hiệu ứng bất biến theo thời gian.

### Mệnh đề về mức lệch cần thiết để hiệu chỉnh có ích

Xét một chuỗi đơn giản hơn: phần dư hiệu chuẩn $r_i=\beta+\epsilon_i$, $n$ quan sát độc lập, $\mathbb E\epsilon_i=0$, $\operatorname{Var}(\epsilon_i)=\sigma^2$. Ước lượng co là $\widehat\beta=a\bar r$, $a=n/(n+\lambda)$; phần dư tương lai độc lập có trung bình $\beta$. So với không hiệu chỉnh, chênh kỳ vọng lỗi bình phương là

$$\Delta=(a^2-2a)\beta^2+a^2\sigma^2/n.$$

Vì $a>0$, hiệu chỉnh có lợi **khi và chỉ khi** $\beta^2>\sigma^2/(n+2\lambda)$. Đây là điều kiện có thể diễn giải: sai lệch ổn định phải đủ lớn so với nhiễu ước lượng. Nếu trung bình tương lai dịch thành $\beta+\delta$ với $|\delta|\le D$, điều kiện đủ mạnh hơn là $(2a-a^2)\beta^2-a^2\sigma^2/n>2a|\beta|D$. Điều này cho thấy rõ vì sao trôi phân phối có thể xóa lợi ích của hiệu chỉnh tĩnh. Mệnh đề là trường hợp một nhóm của mô hình phân cấp, không phải bảo đảm vô điều kiện cho Favorita.

### Vì sao bài toán hiện tại không cần reinforcement learning

Nếu hành động duy nhất là công bố dự báo $a_t$, còn doanh số thật và trạng thái dữ liệu tương lai không phụ thuộc vào con số dự báo, phương trình Bellman có dạng $Q(H_t,a_t)=-\mathbb E[\ell(a_t,Z_t)\mid H_t]+\gamma\mathbb E[V(H_{t+1})\mid H_t]$. Hạng thứ hai không phụ thuộc $a_t$; do đó $a_t^*=\arg\min_a\mathbb E[\ell(a,Z_t)\mid H_t]$. Với lỗi bình phương trên log, nghiệm là kỳ vọng có điều kiện của $Z_t$. Bởi vậy học có giám sát/online regression là phương pháp đúng cho dữ liệu hiện có. RL chỉ trở nên cần thiết nếu dự báo được dùng để quyết định tồn kho/giá và quyết định ấy làm thay đổi doanh số hoặc reward tương lai; dataset này không ghi chuỗi quyết định đó.

## Đẳng thức kiểm chứng hiệu quả, không cần giả định phân phối

Với $e=z-\widehat z_{base}$, $d=\widehat z_{method}-\widehat z_{base}$, tính sau phép chặn dự báo, ta có chính xác:

$$\Delta\mathrm{MSE}_{log}=\mathbb E[d^2]-2\mathbb E[ed].$$

Phương pháp giảm RMSLE khi và chỉ khi $2\mathbb E[ed]>\mathbb E[d^2]$. Hai vế được tính lại từ dự báo đã lưu:

| Giai đoạn | n | RMSLE base | RMSLE hiệu chỉnh | $\mathbb E[d^2]$ | $2\mathbb E[ed]$ | $\Delta\mathrm{MSE}_{log}$ |
|---|---:|---:|---:|---:|---:|---:|
| Jan–Feb 2016 | 106,920 | 0.484650 | 0.484254 | 0.0004440 | 0.0008278 | -0.0003838 |
| March 2016 | 55,242 | 0.422887 | 0.422492 | 0.0004439 | 0.0007777 | -0.0003339 |
| Exploratory test | 892,782 | 0.437766 | 0.437142 | 0.0004440 | 0.0009900 | -0.0005460 |

## Kết quả test và giới hạn

Test có 1,104/1,782 chuỗi và 18/18 block 28 ngày giảm lỗi bình phương log. Chênh RMSLE = -0.000624; paired block bootstrap 95% [-0.000750, -0.000504] trên test cố định. MAE tăng từ 77.816 lên 77.844 đơn vị sales. Hiệu quả RMSLE nhỏ và chưa được xác nhận bằng test blind mới.

Không dùng test để điều chỉnh phương pháp hoặc các phạt ridge. Bộ lọc jump ở các lượt trước không cho thêm lợi ích thực chứng; bản hiệu chỉnh phân cấp này là một giả thuyết khác, dựa trên sai lệch ổn định theo nhóm.

## Nhánh online đã thử và loại

Đã chạy 60 cấu hình online OGD/ridge trên validation; kết quả tốt nhất trong lưới là **ogd** với RMSLE Jan–Feb 0.484654, trong khi LightGBM đạt 0.484650. Nhánh này không được đưa sang test. [`online_gate_development_grid.csv`](tables/online_gate_development_grid.csv).

Với OGD theo từng chuỗi, $c_t$ là EMA phần dư quá khứ, $w_t\in[0,1]$, $\ell_t(w)=(r_t-wc_t)^2$ và $w_{t+1}=\Pi_{[0,1]}(w_t-\eta_t\nabla\ell_t(w_t))$. Tính lồi và phép chiếu cho mọi $u\in[0,1]$ bảo đảm đường đi:
$$\sum_{t=1}^T[\ell_t(w_t)-\ell_t(u)]\le\frac{1}{2\eta_T}+\frac12\sum_{t=1}^T\eta_t\|\nabla\ell_t(w_t)\|^2,$$
với $\eta_t$ không tăng. Đây là bảo đảm so với một hệ số cố định tốt nhất trên **dãy tín hiệu EMA đã chọn**, không bảo đảm thắng LightGBM trong một cửa sổ ngắn. Validation xác nhận giới hạn thực tế đó.

## Kiểm tra và tái lập

`python -m scripts.run_hierarchical_residual --stage develop` tạo OOF model, sufficient statistics, grid và khóa. `--stage test` chỉ chạy một lần rồi ghi marker. `python -m scripts.report_hierarchical_residual` kiểm tra hash và tái tính dự báo/metric từ artifact; không train hoặc đánh giá lại base model. Các nguồn data/model lớn nằm trong `results/` và `data/` bị Git ignore.
