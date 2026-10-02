
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from prepare_data import X_scaler
# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv("pcm_btms_dataset.csv")

inputs = [
    "C_rate",
    "Tm",
    "rho",
    "cp",
    "k",
    "Lf",
    "Aavg"
]

outputs = [
    "Tmax",
    "dTmax",
    "phi_avg"
]

X = df[inputs]
y = df[outputs]

# --------------------------------------------------
# Same split used during training
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# Load model and scalers
# --------------------------------------------------

model = joblib.load("bpnn_pcm_btms.pkl")
X_scaler = joblib.load("X_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# --------------------------------------------------
# Scale
# --------------------------------------------------

X_test_scaled = X_scaler.transform(X_test)

# --------------------------------------------------
# Predict
# --------------------------------------------------

y_pred_scaled = model.predict(X_test_scaled)

y_pred = y_scaler.inverse_transform(y_pred_scaled)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

print("=" * 60)
print("BPNN TEST PERFORMANCE")
print("=" * 60)

for i, target in enumerate(outputs):

    r2 = r2_score(
        y_test.iloc[:, i],
        y_pred[:, i]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test.iloc[:, i],
            y_pred[:, i]
        )
    )

    mae = mean_absolute_error(
        y_test.iloc[:, i],
        y_pred[:, i]
    )

    print(f"\n{target}")
    print(f"R²   : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

# --------------------------------------------------
# Actual vs Predicted
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, target in enumerate(outputs):

    axes[i].scatter(
        y_test.iloc[:, i],
        y_pred[:, i],
        alpha=0.7
    )

    minimum = min(
        y_test.iloc[:, i].min(),
        y_pred[:, i].min()
    )

    maximum = max(
        y_test.iloc[:, i].max(),
        y_pred[:, i].max()
    )

    axes[i].plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    axes[i].set_xlabel(f"Actual {target}")
    axes[i].set_ylabel(f"Predicted {target}")

    axes[i].set_title(
        f"{target}: Actual vs Predicted"
    )

    axes[i].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "bpnn_predictions.png",
    dpi=300
)

plt.show()