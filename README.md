# NLP - Job Career Level Classification (Phân loại cấp bậc nghề nghiệp)

Dự án này là một ứng dụng của Xử lý Ngôn ngữ Tự nhiên (NLP) và Machine Learning để phân loại cấp bậc nghề nghiệp (`career_level`) dựa trên thông tin mô tả công việc. 

## Video Demo
[![Video Demo](https://img.youtube.com/vi/L6ANXTB3IBA/0.jpg)](https://youtu.be/L6ANXTB3IBA)

##  Mô tả chung
Mục tiêu của dự án là xây dựng một 파ypline dự đoán cấp bậc của một công việc (ví dụ: thực tập sinh, nhân viên, quản lý, v.v.) thông qua các thông tin văn bản như tiêu đề công việc, mô tả, ngành nghề, và vị trí địa lý.

Dự án sử dụng thư viện **Scikit-Learn** cho quá trình trích xuất đặc trưng văn bản, lựa chọn đặc trưng và huấn luyện mô hình học máy.

## Cấu trúc dữ liệu
*   **Dữ liệu đầu vào:** Tập dữ liệu dạng bảng tính `final_project.ods`. (Link tải dữ liệu: [Google Drive](https://drive.google.com/file/d/1BZasDYJAucvcxFsiZ1zQRSp0IPjCuFhp/view?usp=sharing))
*   **Nhãn mục tiêu (Target):** `career_level` - Cấp bậc nghề nghiệp của công việc.
*   **Các đặc trưng (Features):**
    *   `title` (Tiêu đề công việc): Xử lý bằng TF-IDF.
    *   `location` (Vị trí): Trích xuất mã vùng với Regex và chuyển đổi qua One-Hot Encoding.
    *   `description` (Mô tả công việc): Trích xuất văn bản với TF-IDF (hỗ trợ bi-grams).
    *   `function` (Chức năng/Vai trò): Xử lý bằng One-Hot Encoding.
    *   `industry` (Ngành nghề): Xử lý bằng TF-IDF.

##  Các bước xử lý trong Pipeline (Quy trình)

1.  **Tiền xử lý dữ liệu (Data Preprocessing):**
    *   Loại bỏ các dòng chứa giá trị null (`dropna`).
    *   Rút trích mã bang/vùng của `location` thông qua biểu thức chính quy (Regex).
2.  **Trích xuất đặc trưng (Feature Extraction):**
    *   Sử dụng `ColumnTransformer` để áp dụng song song:
        *   `TfidfVectorizer` cho các cột chứa văn bản (`title`, `description`, `industry`).
        *   `OneHotEncoder` cho các cột danh mục (`location`, `function`).
3.  **Lựa chọn đặc trưng (Feature Selection):**
    *   Sử dụng `SelectPercentile` kết hợp kiểm định Chi-bình phương (`chi2`) để giữ lại top 5% đặc trưng quan trọng nhất.
4.  **Mô hình huấn luyện (Modeling):**
    *   Sử dụng mô hình **Random Forest Classifier**.
    *   Sử dụng `GridSearchCV` với 4-fold Cross Validation để tìm ra bộ siêu tham số tốt nhất (tối ưu hóa theo `recall_weighted`):
        *   Tiêu chí phân chia (`criterion`): `gini`, `entropy`, `log_loss`.
        *   Tỷ lệ phần trăm đặc trưng được chọn: 1%, 5%, 10%.
5.  **Xử lý mất cân bằng dữ liệu (Tùy chọn):**
    *   Code có hỗ trợ tích hợp `SMOTEN` từ thư viện `imbalanced-learn` để oversampling cho các nhóm cấp bậc thiểu số. (Hiện đang được comment lại).

## 🛠️ Yêu cầu cài đặt (Requirements)
Đảm bảo bạn đã cài đặt các thư viện Python sau:
```bash
pip install pandas scikit-learn imbalanced-learn openpyxl odfpy
```
*(Ghi chú: Cần cài thêm `odfpy` vì file dữ liệu là định dạng `.ods`)*

## Hướng dẫn sử dụng
1. Đặt file dữ liệu `final_project.ods` vào thư mục `D:/NLP/Job/`.
2. Chạy đoạn mã huấn luyện bằng lệnh:
```bash
python Job/job.py
```
3. Kết quả đầu ra sẽ hiển thị bảng báo cáo đánh giá mô hình (Classification Report) trên tập dữ liệu kiểm tra.
