# ============================================
# 1 NHẬP THƯ VIỆN
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================
# 2 HIỂU & BIỂU DIỄN DỮ LIỆU
# ============================================

df = pd.read_csv(r"C:\Users\ADMIN\Downloads\Mall_Customers.csv")
print("Kích thước dữ liệu:", df.shape)
display(df.head())

# Lấy các cột số để phân cụm
num_cols = df.select_dtypes(include=np.number).columns
display(df[num_cols].describe())

# --- Biểu đồ scatter 2 đặc trưng
plt.figure(figsize=(6,5))
plt.scatter(df[num_cols[0]], df[num_cols[1]], alpha=0.6, color='teal')
plt.title(f"Quan hệ giữa {num_cols[0]} và {num_cols[1]}")
plt.xlabel(num_cols[0])
plt.ylabel(num_cols[1])
plt.show()

# ============================================
# 3 HUẤN LUYỆN MÔ HÌNH PHÂN CỤM (KMEANS)
# ============================================

X = df[num_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Thử K từ 2 đến 10
inertias = []
for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

# --- Biểu đồ Elbow
plt.figure(figsize=(6,4))
plt.plot(range(2,11), inertias, marker='o', color='purple')
plt.title("Biểu đồ Elbow - chọn số cụm tối ưu")
plt.xlabel("Số cụm (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Chọn K tối ưu (ví dụ 4)
k_opt = 4
kmeans = KMeans(n_clusters=k_opt, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels

# ============================================
# 4 ĐÁNH GIÁ PHÂN CỤM
# ============================================

score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.3f}")

# --- Biểu đồ scatter thể hiện cụm
plt.figure(figsize=(6,5))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='rainbow', alpha=0.6)
plt.title(f"Phân cụm KMeans (K={k_opt})")
plt.xlabel(num_cols[0])
plt.ylabel(num_cols[1])
plt.show()

# ============================================
# 5 TINH CHỈNH (nếu cần)
# ============================================
# Có thể thử thêm PCA hoặc StandardScaler khác để xem kết quả

# ============================================
# 6 KẾT LUẬN
# ============================================
print(f"✅ Phân cụm hoàn tất. Số cụm: {k_opt}, Silhouette = {score:.3f}")
if score > 0.5:
    print("Kết quả phân cụm khá tốt!")
else:
    print("Phân cụm chưa rõ ràng, nên thử K khác hoặc chọn đặc trưng phù hợp hơn.")
