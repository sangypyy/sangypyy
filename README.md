# ============================================
# 1 NHẬP THƯ VIỆN
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# 2 HIỂU & BIỂU DIỄN DỮ LIỆU
# ============================================

df = pd.read_csv(r"C:\Users\ADMIN\Downloads\advertising.csv")
print("Kích thước dữ liệu:", df.shape)
display(df.head())

target_col = 'Sales'
num_cols = df.select_dtypes(include=np.number).columns

# --- Biểu đồ scatter: TV vs Sales
plt.figure(figsize=(6,4))
plt.scatter(df['TV'], df[target_col], color='blue', alpha=0.6)
plt.title("Quan hệ giữa chi tiêu TV và doanh số")
plt.xlabel("TV")
plt.ylabel("Sales")
plt.grid(alpha=0.5)
plt.show()

# ============================================
# 3 HUẤN LUYỆN MÔ HÌNH HỒI QUY TUYẾN TÍNH
# ============================================

X = df[['TV', 'Radio', 'Newspaper']]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ============================================
# 4 ĐÁNH GIÁ MÔ HÌNH
# ============================================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R2 Score: {r2:.3f}")

# --- Biểu đồ thực tế vs dự đoán
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color='teal', alpha=0.6)
plt.title("Giá trị thực tế vs Dự đoán")
plt.xlabel("Thực tế")
plt.ylabel("Dự đoán")
plt.grid(True)
plt.show()

# ============================================
# 5 TINH CHỈNH MÔ HÌNH
# ============================================
# Có thể thử thêm PolynomialFeatures hoặc Regularization (Ridge/Lasso)

# ============================================
# 6 KẾT LUẬN
# ============================================
print("\n Hồi quy hoàn tất.")
if r2 >= 0.8:
    print("Mô hình có độ phù hợp tốt (R2 > 0.8)!")
else:
    print("Cần thử thêm biến hoặc mô hình phức tạp hơn (ví dụ Ridge/Lasso/RandomForest).")
