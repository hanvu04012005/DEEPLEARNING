#  DEEPLEARNING
## Pandas

---

**Sinh viên thực hiện:** Nguyễn Hạn Vũ  
**MSSV:** 2374802010571  
**Môn học:** DEEPLEARNING  
**Giảng viên:** Thầy Nguyễn Thái Anh  

---

## 1. Làm quen với thư viện Pandas và các thư viện liên quan

Trong bài lab này, em đã thực hành với các thư viện quan trọng:

- **Pandas (pd):** xử lý dữ liệu dạng bảng (Series, DataFrame)
- **NumPy (np):** hỗ trợ mảng số học và các giá trị đặc biệt như `np.nan`

---

## 2. Làm việc với Pandas Series

Em đã học và thực hành các nội dung chính về **Series**.

### Tạo Series

- Tạo Series từ **list**
- Tạo Series từ **numpy array**
- Tạo Series từ **dictionary**
- Tạo Series từ **scalar**

### Thuộc tính của Series

- `.values`
- `.index`

### Truy xuất dữ liệu (Indexing)

- Indexing cơ bản bằng số
- Indexing theo nhãn (label)
- Indexing kết hợp (combined indexing)
- Indexing dạng chữ (letter indexing)

---

## 3. Làm việc với DataFrame

Em đã thực hành các thao tác quan trọng với **DataFrame**.

### Tạo DataFrame

- Tạo từ **2 dictionary**
- Các cách tạo DataFrame khác nhau

### Thuộc tính DataFrame

- `.index`
- `.columns`
- `.values`

### Truy xuất dữ liệu

- Truy xuất theo kiểu dictionary: `df["col"]`
- Truy xuất theo kiểu attribute: `df.col`
- Hiểu trường hợp attribute style **không hoạt động đúng**

---

## 4. Indexing nâng cao trong DataFrame

Em đã thực hành các kỹ thuật quan trọng:

- **Masking**
- **Fancy indexing**
- So sánh **iloc** và **loc**
- Indexing vs slicing
- Các dạng truy xuất dữ liệu theo hàng/cột

---

## 5. Broadcasting trong Pandas

Em đã học về cơ chế **broadcasting** và sự khác nhau trong Pandas:

- Broadcasting cơ bản
- **Index alignment** (căn chỉnh index tự động)
- Broadcasting giữa **DataFrame** và **Series**

---

## 6. Xử lý dữ liệu thiếu (Missing Data)

Em đã thực hành cách xử lý missing data trong Pandas:

- Hiểu ý nghĩa của `np.nan` và `None`
- Kiểm tra dữ liệu thiếu bằng:
  - `isnull()`
  - `notnull()`
- Xóa dữ liệu thiếu:
  - `dropna()`
- Điền dữ liệu thiếu:
  - `fillna()`

---

## 7. Nối và gộp dữ liệu

Em đã thực hành các thao tác quan trọng khi làm việc với nhiều bảng dữ liệu:

- **Concatenation** (nối DataFrame)
- **Join inner**
- **Merging datasets với ID**

---

## 8. Tổng hợp dữ liệu (Aggregation)

Em đã học cách tổng hợp dữ liệu bằng các thao tác aggregation trong Pandas để phục vụ phân tích dữ liệu.

---

## 9. Bài tập thực hành

Trong notebook có các phần luyện tập:

- **Bài tập 1**
- **Bài tập 2**

Nhằm giúp em củng cố các kiến thức về Series, DataFrame, Indexing, Missing Data và Merge.

---

## 10. Những gì em rút ra được sau buổi lab

Sau buổi thực hành, em rút ra được:

- Hiểu rõ cách sử dụng **Series** và **DataFrame**
- Nắm được các thao tác xử lý dữ liệu quan trọng trong Pandas
- Biết cách indexing nâng cao với `loc`, `iloc`, masking, fancy indexing
- Biết cách xử lý dữ liệu bị thiếu và làm sạch dữ liệu
- Hiểu cách nối/gộp nhiều bảng dữ liệu phục vụ phân tích
- Biết xử lý dữ liệu đầu vào cho các bài toán Machine Learning / Deep Learning

---

## Kết luận

Buổi lab Pandas giúp em xây dựng nền tảng quan trọng trong việc xử lý dữ liệu. Đây là bước chuẩn bị cần thiết trước khi đưa dữ liệu vào huấn luyện mô hình Deep Learning, giúp việc phân tích và tiền xử lý dữ liệu trở nên dễ dàng và hiệu quả hơn.

---

