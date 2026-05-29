# 🛡️ ML-Based Network Intrusion Detection System (NIDS)

Dự án phát hiện xâm nhập mạng (NIDS) ứng dụng học máy (Machine Learning), sử dụng thuật toán XGBoost để phân loại lưu lượng mạng và phát hiện các nỗ lực tấn công mạng dựa trên tập dữ liệu NSL-KDD.

## 🚀 Công nghệ sử dụng
* **Ngôn ngữ:** Python 3.12
* **Machine Learning:** XGBoost, Scikit-learn, Pandas, Numpy
* **Môi trường:** Linux (Ubuntu)

## ⚙️ Quy trình xử lý
1. **Tiền xử lý dữ liệu:** Xử lý các nhãn phân loại (Categorical Data), chuyển bài toán thành phân loại nhị phân (Binary Classification - Normal vs Attack).
2. **Huấn luyện mô hình:** Sử dụng XGBoost Classifier với `max_depth=5` và `learning_rate=0.1`.
3. **Đánh giá:** Mô hình đạt độ chính xác (Accuracy) **~80.7%** trên tập kiểm thử chưa từng thấy (`KDDTest+`), với độ chính xác trên nhóm cảnh báo tấn công (Precision) lên tới **97%**.

## 💻 Cài đặt và Sử dụng
**1. Clone dự án**
```bash
git clone [https://github.com/TÊN_GITHUB_CỦA_BẠN/nids-xgboost-project.git](https://github.com/TÊN_GITHUB_CỦA_BẠN/nids-xgboost-project.git)
cd nids-xgboost-project