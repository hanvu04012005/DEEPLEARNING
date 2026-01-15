# README – DEEPLEARNING 
## Lab1_PyTorch 

---

**Sinh viên thực hiện:** Nguyễn Hạn Vũ  
**MSSV:** 2374802010571  
**Môn học:** DEEPLEARNING  
**Giảng viên:** Thầy Nguyễn Thái Anh

---

## 1. Làm quen với thư viện PyTorch và các thư viện liên quan

Em đã học cách import và sử dụng các thư viện phổ biến:

* **PyTorch (torch)**: làm việc với tensor, tự động đạo hàm
* **NumPy**: xử lý mảng số học
* **Pandas**: đọc và xử lý dữ liệu dạng bảng
* **Matplotlib**: trực quan hóa dữ liệu

---

## 2. Đọc và xử lý dữ liệu

* Đọc dữ liệu từ file CSV (bộ dữ liệu **Iris**) bằng `pandas.read_csv`
* Quan sát dữ liệu và hiểu cấu trúc của tập dữ liệu
* Tách đặc trưng (X) và nhãn (y)

---

## 3. Tiền xử lý dữ liệu

* Sử dụng **LabelEncoder** để mã hóa nhãn dạng chuỗi sang số
* Chia dữ liệu thành tập **train** và **test** bằng `train_test_split`
* Kiểm tra số lượng mẫu trong tập huấn luyện
* Phân tích sự phân bố nhãn trong tập train

---

## 4. Tính đạo hàm tự động với PyTorch (Autograd)

* Khai báo tensor với `requires_grad=True`
* Xây dựng hàm toán học (ví dụ: (y = 2x^4 + x^3 + 3x^2 + 1))
* Sử dụng `backward()` để tính đạo hàm
* Hiểu được cách PyTorch hỗ trợ **tự động đạo hàm** cho việc huấn luyện mô hình

---

## 5. Làm việc với Tensor trong PyTorch

* Tạo tensor từ NumPy bằng `torch.from_numpy()`
* Tạo tensor độc lập bằng `torch.tensor()`
* Hiểu sự khác nhau giữa **chia sẻ bộ nhớ** và **không chia sẻ bộ nhớ** giữa NumPy và PyTorch

### Kết luận quan trọng:

* `torch.from_numpy()` chia sẻ chung vùng nhớ với NumPy
* `torch.tensor()` tạo một bản sao độc lập

---

## 6. Tạo và khởi tạo Tensor

Em đã thực hành tạo tensor với nhiều cách khác nhau:

* Tensor rỗng: `torch.empty()`
* Tensor toàn số 0: `torch.zeros()`
* Tensor toàn số 1: `torch.ones()`
* Tensor ngẫu nhiên: `torch.rand()`

---

## 7. Thay đổi kích thước Tensor (Reshape)

* Sử dụng `view()` và `view_as()` để reshape tensor
* Hiểu điều kiện để reshape hợp lệ (tổng số phần tử không đổi)

---

## 8. Những gì em rút ra được sau buổi lab

* Hiểu rõ khái niệm **tensor** và cách thao tác với tensor trong PyTorch
* Biết cách xử lý dữ liệu đầu vào cho bài toán Machine Learning
* Nắm được cơ chế **tự động đạo hàm** – nền tảng của Deep Learning
* Phân biệt được sự khác nhau giữa NumPy array và PyTorch tensor
* Tự tin hơn khi làm việc với PyTorch cho các bài toán tiếp theo

---

## Kết luận

Buổi lab này đã giúp em xây dựng nền tảng quan trọng để tiếp cận **Deep Learning**. Những kiến thức học được không chỉ mang tính lý thuyết mà còn rất thực tế, hỗ trợ tốt cho việc phát triển và huấn luyện các mô hình học máy trong tương lai.

---


