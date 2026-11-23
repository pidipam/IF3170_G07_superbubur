import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.decision_tree import DecisionTree

# --- TES 1: Dataset Mainan (Play Tennis) ---
X_toy = np.array([
    ["Sunny", 85],
    ["Overcast", 83],
    ["Rain", 70],
    ["Sunny", 72],
    ["Overcast", 64],
    ["Rain", 75]
], dtype=object)
y_toy = np.array([0, 1, 1, 0, 1, 1])

dt_toy = DecisionTree(max_depth=5, categorical_idx=[0])
dt_toy.fit(X_toy, y_toy)
print("Tes 1 (Dataset Mainan) Prediksi:", dt_toy.predict(X_toy))

# --- TES 2: Null & Kategori Baru ---
X_edge = np.array([
    [None, 80],          # Null di kategorikal
    ["Rain", np.nan],    # NaN di numerik
    ["Cloudy", 70]       # Kategori baru ("Cloudy") yang tidak ada saat training
], dtype=object)
print("Tes 2 (Null & Kategori Baru) Prediksi:", dt_toy.predict(X_edge))
print("Tes 2 Probabilitas:\n", dt_toy.predict_probability(X_edge))

# --- TES 3: Save & Load ---
dt_toy.save("temp_model.pkl")
dt_loaded = DecisionTree.load("temp_model.pkl")
assert np.array_equal(dt_toy.predict(X_toy), dt_loaded.predict(X_toy)), "Save/Load berbeda!"
print("Tes 3 (Save/Load): LOLOS (Identik 100%)")

# --- TES 4: Subsample 5.000 Baris Data Hotel & Perbandingan Scikit-Learn ---
# Sesuaikan path CSV data train kamu di bawah ini
df = pd.read_csv("data/train.csv").sample(n=5000, random_state=42)
target_col = "is_canceled"
X = df.drop(columns=[target_col, "id"], errors="ignore")
y = df[target_col].values

# Cari indeks kolom kategorikal (tipe object/string)
cat_idx = [i for i, dtype in enumerate(X.dtypes) if dtype == "object"]

X_train, X_val, y_train, y_val = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

# Ukur waktu fit DecisionTree from scratch
t0 = time.time()
dt_hotel = DecisionTree(
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    categorical_idx=cat_idx,
    max_thresholds=32
)
dt_hotel.fit(X_train, y_train)
durasi = time.time() - t0

preds = dt_hotel.predict(X_val)
print(f"\nTes 4 (5.000 Baris Hotel):")
print(f"Waktu fit : {durasi:.2f} detik")
print(f"Akurasi   : {accuracy_score(y_val, preds):.4f}")
print(f"F1-Score  : {f1_score(y_val, preds):.4f}")