import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

print("[1/3] Dang tai va tien xu ly du lieu...")

# 1. Xử lý đường dẫn động (Dynamic Path)
current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_dir, '..', 'data', 'KDDTrain+.txt')
test_path = os.path.join(current_dir, '..', 'data', 'KDDTest+.txt')

# Đọc dữ liệu từ file
train_df = pd.read_csv(train_path, header=None)
test_df = pd.read_csv(test_path, header=None)

# 2. Tách Features (X) và Label (y)
X_train = train_df.iloc[:, 0:41]
y_train = train_df.iloc[:, 41]

X_test = test_df.iloc[:, 0:41]
y_test = test_df.iloc[:, 41]

# 3. Chuyển đổi nhãn thành Phân loại nhị phân (0: Bình thường, 1: Tấn công)
y_train = y_train.apply(lambda x: 0 if x == 'normal' else 1)
y_test = y_test.apply(lambda x: 0 if x == 'normal' else 1)

# 4. Mã hóa các cột chứa chữ (Categorical) thành số
X_combined = pd.concat([X_train, X_test])
encoder = LabelEncoder()

# Tìm và mã hóa tất cả các cột KHÔNG phải là số
for col in X_combined.select_dtypes(exclude=['number']).columns:
    X_combined[col] = encoder.fit_transform(X_combined[col].astype(str))

# Ép kiểu toàn bộ dữ liệu về số thực (float) để đảm bảo XGBoost không báo lỗi
X_combined = X_combined.astype(float)

# Tách lại ra tập Train và Test sau khi đã mã hóa xong
X_train = X_combined.iloc[:len(X_train)]
X_test = X_combined.iloc[len(X_train):]

print("[2/3] Dang huan luyen mo hinh XGBoost. Qua trinh nay co the mat vai chuc giay...")

# Khởi tạo mô hình XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)

# Huấn luyện
model.fit(X_train, y_train)

print("[3/3] Dang danh gia mo hinh tren tap kiem thu...")

# Dự đoán
y_pred = model.predict(X_test)

# In kết quả
print("\n--- BAO CAO KET QUA (CLASSIFICATION REPORT) ---")
print(f"Do chinh xac tong the (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=['Binh thuong (0)', 'Tan cong (1)']))
# Lưu mô hình vào thư mục models/
print("\n[4] Dang luu mo hinh...")
model.save_model(os.path.join(current_dir, '..', 'models', 'nids_xgboost_v1.json'))
print("Hoan tat! Mo hinh da duoc luu tai thu muc 'models/'.")