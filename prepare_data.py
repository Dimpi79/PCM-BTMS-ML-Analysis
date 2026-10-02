import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv("pcm_btms_dataset.csv")

# ==========================================================
# 2. DEFINE INPUTS AND OUTPUTS
# ==========================================================

input_features = [
    "C_rate",
    "Tm",
    "rho",
    "cp",
    "k",
    "Lf",
    "Aavg"
]

target_features = [
    "Tmax",
    "dTmax",
    "phi_avg"
]

X = df[input_features].copy()
y = df[target_features].copy()

# ==========================================================
# 3. CHECK SHAPES
# ==========================================================

print("=" * 60)
print("DATASET PREPARATION")
print("=" * 60)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nInput features:")
print(input_features)

print("\nTarget features:")
print(target_features)

# ==========================================================
# 4. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# ==========================================================
# 5. STANDARDIZATION
# ==========================================================

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# ==========================================================
# 6. CHECK STANDARDIZED DATA
# ==========================================================

print("\n" + "=" * 60)
print("SCALED DATA CHECK")
print("=" * 60)

print("\nX_train mean:")
print(np.round(X_train_scaled.mean(axis=0), 4))

print("\nX_train standard deviation:")
print(np.round(X_train_scaled.std(axis=0), 4))

print("\ny_train mean:")
print(np.round(y_train_scaled.mean(axis=0), 4))

print("\ny_train standard deviation:")
print(np.round(y_train_scaled.std(axis=0), 4))

# ==========================================================
# 7. DISPLAY FIRST FEW SAMPLES
# ==========================================================

print("\n" + "=" * 60)
print("FIRST 5 TRAINING INPUTS")
print("=" * 60)

print(X_train.head())

print("\n" + "=" * 60)
print("FIRST 5 TRAINING TARGETS")
print("=" * 60)

print(y_train.head())