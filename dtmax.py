import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib



from sklearn.model_selection import train_test_split

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

X = df[inputs]
y = df["dTmax"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = joblib.load("bpnn_pcm_btms.pkl")
X_scaler = joblib.load("X_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

X_test_scaled = X_scaler.transform(X_test)

pred_scaled = model.predict(X_test_scaled)

# Because model predicts 3 outputs
pred = y_scaler.inverse_transform(pred_scaled)

# dTmax is output index 1
pred_dTmax = pred[:, 1]

actual = y_test.values

error = np.abs(actual - pred_dTmax)

sample = np.arange(len(actual))

plt.figure(figsize=(12, 6))

plt.plot(
    sample,
    actual,
    marker="s",
    linestyle="",
    label="Calculated"
)

plt.plot(
    sample,
    pred_dTmax,
    marker="o",
    linestyle="",
    label="Predicted"
)

plt.bar(
    sample,
    error,
    alpha=0.2,
    label="Absolute Error"
)

plt.xlabel("Test Sample")
plt.ylabel("ΔTmax (K)")

plt.title(
    "BPNN Prediction of Maximum Temperature Difference"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "bpnn_dTmax_comparison.png",
    dpi=300
)

plt.show()