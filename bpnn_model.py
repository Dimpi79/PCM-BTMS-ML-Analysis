import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("pcm_btms_dataset.csv")


# ============================================================
# 2. INPUTS AND OUTPUTS
# ============================================================

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

X = df[input_features]
y = df[target_features]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 4. STANDARDIZATION
# ============================================================

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# ============================================================
# 5. BPNN MODEL
# ============================================================

model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=2000,
    batch_size=16,
    early_stopping=True,
    validation_fraction=0.20,
    n_iter_no_change=50,
    random_state=42
)


# ============================================================
# 6. TRAIN
# ============================================================

print("=" * 70)
print("TRAINING BPNN")
print("=" * 70)

model.fit(
    X_train_scaled,
    y_train_scaled
)

print("Training completed.")
print("Iterations:", model.n_iter_)


# ============================================================
# 7. PREDICTION
# ============================================================

y_pred_scaled = model.predict(X_test_scaled)

# Convert predictions back to original units
y_pred = y_scaler.inverse_transform(y_pred_scaled)


# ============================================================
# 8. PERFORMANCE METRICS
# ============================================================

print("\n" + "=" * 70)
print("BPNN PERFORMANCE")
print("=" * 70)

for i, target in enumerate(target_features):

    actual = y_test.iloc[:, i]
    predicted = y_pred[:, i]

    r2 = r2_score(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    print(f"\n{target}")
    print(f"R²   : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")


# ============================================================
# 9. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "bpnn_pcm_btms.pkl"
)

joblib.dump(
    X_scaler,
    "X_scaler.pkl"
)

joblib.dump(
    y_scaler,
    "y_scaler.pkl"
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("bpnn_pcm_btms.pkl")
print("X_scaler.pkl")
print("y_scaler.pkl")