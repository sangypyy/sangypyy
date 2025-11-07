# ============================================
# 1. NHẬP THƯ VIỆN
# ============================================

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ============================================
# 2. ĐỌC & KHÁM PHÁ DỮ LIỆU
# ============================================

# ⚠️ Thay đường dẫn và tên cột mục tiêu
file_path = r"C:\Users\ADMIN\Downloads\ten_file.csv"

target_col = "TenCotMucTieu"

df = pd.read_csv(file_path)

print("✅ Kích thước dữ liệu:", df.shape)

print("✅ Các cột:", df.columns.tolist())

display(df.head())

# --- Thống kê mô tả
print("\n📊 Thống kê mô tả:")

display(df.describe())


# ============================================
# 3. BIỂU DIỄN DỮ LIỆU VỚI MATPLOTLIB
# ============================================

# --- Phân bố nhãn mục tiêu
plt.figure(figsize=(6,4))

label_counts = df[target_col].value_counts()

plt.bar(label_counts.index.astype(str), label_counts.values, color=['skyblue','salmon'])

plt.title('Phân bố nhãn mục tiêu')

plt.xlabel(target_col)

plt.ylabel('Số lượng mẫu')

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# --- Histogram của 1 đặc trưng số
num_col = df.select_dtypes(include=np.number).columns[0]

plt.figure(figsize=(6,4))

plt.hist(df[num_col], bins=20, color='orange', edgecolor='black')

plt.title(f'Phân phối đặc trưng {num_col}')

plt.xlabel(num_col)

plt.ylabel('Tần suất')

plt.grid(alpha=0.5)

plt.show()


# --- Scatter giữa 2 đặc trưng đầu tiên
num_cols = df.select_dtypes(include=np.number).columns[:2]

if len(num_cols) >= 2:
    
    plt.figure(figsize=(6,5))
    
    plt.scatter(
        df[num_cols[0]],
        df[num_cols[1]],
        c=df[target_col].astype('category').cat.codes,
        cmap='coolwarm',
        alpha=0.6
    )
    
    plt.title(f'Quan hệ giữa {num_cols[0]} và {num_cols[1]}')
    
    plt.xlabel(num_cols[0])
    
    plt.ylabel(num_cols[1])
    
    plt.colorbar(label=target_col)
    
    plt.show()


# ============================================
# 4. CHUẨN BỊ DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH
# ============================================

X = df.select_dtypes(include=np.number)

y = df[target_col]

# --- Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Chuẩn hóa dữ liệu
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# --- Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)


# ============================================
# 5. ĐÁNH GIÁ MÔ HÌNH (ACCURACY, PRECISION, RECALL, F1)
# ============================================

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)

rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n🎯 **ĐÁNH GIÁ MÔ HÌNH BAN ĐẦU**")

print(f"✅ Accuracy  : {acc:.3f}")

print(f"✅ Precision : {prec:.3f}")

print(f"✅ Recall    : {rec:.3f}")

print(f"✅ F1-score  : {f1:.3f}")

print("\n📋 Báo cáo chi tiết:")

print(classification_report(y_test, y_pred))

# --- Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)

classes = np.unique(y_test)

plt.figure(figsize=(5,4))

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

plt.title("Ma trận nhầm lẫn - KNN")

plt.colorbar()

plt.xticks(np.arange(len(classes)), classes)

plt.yticks(np.arange(len(classes)), classes)

for i in range(len(classes)):
    
    for j in range(len(classes)):
        
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.ylabel('Thực tế')

plt.xlabel('Dự đoán')

plt.tight_layout()

plt.show()


# ============================================
# 6. TINH CHỈNH MÔ HÌNH (GRIDSEARCHCV)
# ============================================

param_grid = {'n_neighbors': range(1, 21)}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

grid.fit(X_train_scaled, y_train)

print("\n🔎 K tối ưu:", grid.best_params_['n_neighbors'])

print(f"✅ Điểm trung bình cross-validation: {grid.best_score_:.3f}")

# --- Dự đoán với mô hình tốt nhất
best_model = grid.best_estimator_

y_pred_best = best_model.predict(X_test_scaled)

best_acc = accuracy_score(y_test, y_pred_best)

best_f1 = f1_score(y_test, y_pred_best, average='weighted', zero_division=0)

print(f"\n📈 Độ chính xác sau tinh chỉnh: {best_acc:.3f}")

print(f"📊 F1-score sau tinh chỉnh: {best_f1:.3f}")

# --- Biểu đồ độ chính xác theo K
k_values = range(1, 21)

accuracies = []

for k in k_values:
    
    model = KNeighborsClassifier(n_neighbors=k)
    
    model.fit(X_train_scaled, y_train)
    
    y_pred_k = model.predict(X_test_scaled)
    
    accuracies.append(accuracy_score(y_test, y_pred_k))

plt.figure(figsize=(7,4))

plt.plot(k_values, accuracies, marker='o', color='purple')

plt.title('Độ chính xác theo số lượng láng giềng (K)')

plt.xlabel('K')

plt.ylabel('Accuracy')

plt.grid(True)

plt.show()


# ============================================
# 7. KẾT LUẬN
# ============================================

if best_acc >= 0.9:
    
    print("🎉 Kết quả đạt yêu cầu (>= 90%)!")

else:
    
    print("⚠️ Kết quả chưa đạt, cần thử mô hình khác hoặc tinh chỉnh thêm.")

print("\n✅ Bài thực hành hoàn tất!")
