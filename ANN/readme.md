#  DEEPLEARNING
## Lab_ANN 

---

**Sinh viên thực hiện:** Nguyễn Hạn Vũ  
**MSSV:** 2374802010571  
**Môn học:** DEEPLEARNING  
**Giảng viên:** Thầy Nguyễn Thái Anh  

---

## 1. Giới thiệu về bài lab

Trong bài lab này, em được làm quen với **Mạng Nơ-ron Nhân tạo (ANN)** bằng thư viện **PyTorch**.

Bài lab giúp em hiểu được cách một mô hình ANN đơn giản hoạt động như thế nào, từ khâu:
- tạo dữ liệu
- xây dựng mô hình
- huấn luyện
- dự đoán và trực quan hóa kết quả

---

## 2. Các thư viện sử dụng trong bài

Em đã sử dụng các thư viện chính sau:

- **NumPy:** tạo dữ liệu ngẫu nhiên và xử lý mảng số học
- **PyTorch (torch):** xây dựng và huấn luyện mạng nơ-ron
- **torch.nn:** xây dựng các lớp mạng (Linear, ReLU, Sigmoid)
- **torch.optim:** tối ưu mô hình bằng thuật toán Gradient Descent (Adam)
- **sklearn.model_selection:** chia dữ liệu train/test bằng `train_test_split`
- **Matplotlib:** trực quan hóa dữ liệu và kết quả dự đoán

---

## 3. Tạo dữ liệu (Dataset)

Trong lab này, em tạo dữ liệu giả lập gồm 2 lớp:

- **Lớp 0:** các điểm nằm trong vùng **vòng tròn** (radius nhỏ)
- **Lớp 1:** các điểm nằm trong vùng **vành đai** (radius lớn hơn)

Dữ liệu được tạo ngẫu nhiên bằng `numpy.random` và được biểu diễn theo tọa độ (x, y).

---

## 4. Chia dữ liệu Train/Test

Sau khi tạo xong dữ liệu, em thực hiện:

- Gộp dữ liệu thành tập `X` và nhãn `y`
- Chia thành 2 tập:
  - **Train set**
  - **Test set**
- Dùng `train_test_split` để đảm bảo mô hình có thể kiểm tra được độ chính xác trên dữ liệu chưa thấy bao giờ.

---

## 5. Xây dựng mô hình ANN bằng PyTorch

Em xây dựng một mô hình ANN đơn giản gồm:

- **Input layer:** 2 đầu vào (x, y)
- **Hidden layer:** 4 neuron
- **Output layer:** 1 neuron (dự đoán xác suất thuộc lớp 1)

Các hàm kích hoạt được dùng:
- **ReLU** cho hidden layer
- **Sigmoid** cho output layer (đưa kết quả về khoảng 0 → 1)

---

## 6. Hàm loss và optimizer

Trong bài lab, em sử dụng:

- **Loss function:** `BCELoss()`  
  (vì đây là bài toán phân loại nhị phân)

- **Optimizer:** `Adam`  
  giúp mô hình học nhanh và ổn định hơn so với SGD cơ bản.

---

## 7. Huấn luyện mô hình (Training)

Em thực hiện quá trình training gồm nhiều epoch:

- Forward: đưa dữ liệu vào mô hình
- Tính loss
- Backward: tính gradient bằng `loss.backward()`
- Cập nhật trọng số bằng `optimizer.step()`

Trong quá trình huấn luyện, em có theo dõi loss để xem mô hình học tốt hay không.

---

## 8. Dự đoán và đánh giá mô hình

Sau khi huấn luyện xong, em tiến hành:

- Dự đoán trên tập test
- Chuyển xác suất về nhãn (0 hoặc 1)
- Tính độ chính xác (accuracy)

---

## 9. Trực quan hóa kết quả

Một phần khá thú vị trong lab là em được trực quan hóa:

- Dữ liệu 2 lớp ban đầu (vòng tròn và vành đai)
- Kết quả mô hình phân loại
- Quan sát được ANN học cách phân tách dữ liệu phi tuyến

---

## 10. Những gì rút ra được sau buổi lab

Sau bài lab này, em rút ra được:

- Hiểu rõ hơn ANN là gì và hoạt động ra sao
- Biết cách xây dựng một mô hình ANN đơn giản bằng PyTorch
- Biết cách tạo dữ liệu giả lập cho bài toán phân loại
- Hiểu được pipeline cơ bản của Deep Learning:
  - dataset → model → loss → optimizer → training → evaluation
- Tự tin hơn khi tiếp cận các mô hình sâu hơn sau này

---

## Kết luận

Buổi lab này giúp em có cái nhìn rõ hơn về cách một mô hình ANN học dữ liệu.  
Tuy mô hình khá đơn giản nhưng em thấy đây là nền tảng quan trọng để học tiếp các phần như:

- MLP nhiều tầng hơn
- CNN
- RNN
- các mô hình Deep Learning phức tạp khác

---

