# DEEPLEARNING 
##  TUẦN 2 : NUMPY, PANDAS, Matplotlib

---

**Sinh viên thực hiện:** Nguyễn Hạn Vũ  
**MSSV:** 2374802010571  
**Môn học:** DEEPLEARNING  
**Giảng viên:** Thầy Nguyễn Thái Anh

---
# BÁO CÁO THỰC HÀNH: XỬ LÝ DỮ LIỆU VỚI NUMPY VÀ PANDAS

Nội dung này ghi lại các bài tập thực hành về thư viện NumPy, cách thao tác mảng và quy trình tiền xử lý dữ liệu trong Machine Learning.

---
### 1. Thao tác nền tảng với NumPy
* **Khởi tạo mảng**: Chuyển đổi từ Python List sang NumPy array bằng `np.array()`.
* **Ép kiểu dữ liệu**: Sử dụng `.astype(int)` để chuyển đổi kiểu dữ liệu của phần tử trong mảng.
* **Truy xuất dữ liệu (Indexing & Slicing)**:
    * Lấy phần tử theo chỉ số: `x[3]` hoặc `np_list[0, 1]`.
    * Cắt lát mảng đa chiều: Lấy toàn bộ hàng/cột hoặc lấy vùng dữ liệu bằng `np_list[1:, :]`.
    * Kỹ thuật nâng cao: Sử dụng bước nhảy `np_list[0::2, 0]` và lấy đảo ngược phần tử `np_list[-1, ::-1]`.
* **Lọc dữ liệu (Boolean Indexing)**: Sử dụng các điều kiện logic như `np_x % 2 == 0` và kết hợp nhiều điều kiện bằng toán tử `&` (AND), `|` (OR).

### 2. Bài tập (BTVN)

#### BTVN 1: Logic trò chơi Tic-Tac-Toe
* **Khởi tạo bàn cờ**: Tạo ma trận $3 \times 3$ chứa giá trị mặc định là 99.
* **Hàm kiểm tra thắng cuộc**: 
    * Sử dụng `np.all()` để kiểm tra hàng và cột.
    * Sử dụng `np.diag()` và `np.fliplr()` để kiểm tra hai đường chéo.
* **Vòng lặp điều khiển**: Sử dụng `while True` để nhận tọa độ `(x, y)` từ người chơi, kiểm tra ô trống và thông báo thắng cuộc.

#### BTVN 2 & 3: Xử lý danh sách
* **Lọc số chẵn**: 
    * Cách 1: Sử dụng vòng lặp `for` và câu lệnh `if` truyền thống.
    * Cách 2: Sử dụng **List Comprehension** để tối ưu mã nguồn trên một dòng.

#### BTVN 4: Tiền xử lý dữ liệu Machine Learning
* **Tạo dữ liệu**: Khởi tạo mảng ngẫu nhiên kích thước $150 \times 5$ đại diện cho các thông số sinh viên (chiều cao, cân nặng, tuổi, lương, điểm TB).
* **Chia tách biến**: 
    * Tách 4 cột đầu tiên thành biến độc lập `X`.
    * Tách cột cuối cùng thành biến mục tiêu `y`.
* **Phân tập Train/Test**: Sử dụng `train_test_split` để chia dữ liệu theo tỷ lệ 70% huấn luyện và 30% kiểm tra.
* **K-Fold Cross Validation**: Sử dụng `KFold(n_splits=10)` để tạo ra 10 tập dữ liệu kiểm tra không chồng chéo nhau.

---

##  Công nghệ và Thư viện sử dụng
* **NumPy**: Thư viện chính để xử lý mảng đa chiều và toán học.
* **Pandas**: Hỗ trợ hiển thị bàn cờ dưới dạng DataFrame trực quan.
* **Scikit-learn**: Sử dụng các module `train_test_split` và `KFold` để quản lý tập dữ liệu.

---
