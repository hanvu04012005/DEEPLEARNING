#  DEEP LEARNING

##  TUẦN 9: RNN 

---

###  Thông tin sinh viên

* **Họ tên:** Nguyễn Hạn Vũ
* **MSSV:** 2374802010571
* **Môn học:** Deep Learning
* **Giảng viên:** Thầy Nguyễn Thái Anh

---

##  Giới thiệu

Báo cáo này trình bày việc xây dựng mô hình *RNN** để dự đoán chuỗi thời gian .

Nội dung chính bao gồm:

* Tiền xử lý dữ liệu chuỗi
* Xây dựng mô hình RNN bằng PyTorch
* Huấn luyện và đánh giá mô hình
* Thử nghiệm các tham số để cải thiện hiệu suất

---

##  1. Tiền xử lý dữ liệu

###  Tạo dữ liệu

* Sử dụng hàm sin, cos để tạo dữ liệu chuỗi thời gian
* Bao gồm **3 features**:

```python
data = np.stack([
    np.sin(x),
    np.cos(x),
    np.sin(2*x)
], axis=1)
```

---

###  Chuẩn hóa dữ liệu

* Chuẩn hóa về khoảng `[0, 1]`:

```python
data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
```

---

###  Tạo chuỗi con (Sequence)

* Sử dụng `seq_length = 20`

```python
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
```

---

###  Chia tập dữ liệu

* 70%: Training
* 15%: Validation
* 15%: Test

---

##  2. Xây dựng mô hình RNN

###  Kiến trúc mô hình

* `input_size = 3`
* `hidden_size = 32`
* `output_size = 1`

```python
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(3, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
```

---

###  Cấu hình huấn luyện

* Loss: `MSELoss`
* Optimizer: `Adam`
* Epochs: `150`

---

###  Theo dõi loss

* Lưu:

  * Train Loss
  * Validation Loss

---

##  3. Đánh giá mô hình

###  Dự đoán trên tập test

```python
y_pred = model(X_test)
```

---

###  Chỉ số đánh giá

* **MSE (Mean Squared Error)**
* **MAE (Mean Absolute Error)**

```python
mse = nn.MSELoss()(y_pred, y_test)
mae = torch.mean(torch.abs(y_pred - y_test))
```

---

###  Trực quan hóa

* So sánh:

  * Giá trị thực
  * Giá trị dự đoán

* Biểu đồ sai số theo thời gian

---

##  4. Thực nghiệm nâng cao

###  Thay đổi `seq_length`

* Thử: 10, 20, 30

 Nhận xét:

* Chuỗi ngắn → thiếu thông tin
* Chuỗi dài → dự đoán tốt hơn nhưng dễ overfit

---

###  Thay đổi `hidden_size`

* Thử: 16, 32, 64

 Nhận xét:

* Nhỏ → underfitting
* Lớn → học tốt hơn nhưng dễ overfitting

---

###  Multi-step prediction

* Dự đoán 3 bước tiếp theo

 Sai số tăng dần theo số bước dự đoán

---

###  Thử Dropout và nhiều layer

```python
nn.RNN(input_size=3, hidden_size=32, num_layers=2, dropout=0.2)
```

 Giúp giảm overfitting

---

###  Thay đổi Learning Rate

* Thử: `0.001`, `0.01`, `0.1`

 Nhận xét:

* LR lớn → không ổn định
* LR nhỏ → hội tụ chậm

---

###  Quan sát Loss

* Train loss giảm dần
* Validation loss giúp phát hiện overfitting

---

##  Công nghệ sử dụng

* PyTorch
* NumPy
* Matplotlib

---

##  Kết luận

* Mô hình RNN có khả năng học tốt dữ liệu chuỗi thời gian
* Các yếu tố ảnh hưởng mạnh:

  * `seq_length`
  * `hidden_size`
  * `learning rate`
* Multi-step prediction làm tăng sai số theo thời gian
* Cần cân bằng giữa độ phức tạp mô hình và khả năng tổng quát

---


