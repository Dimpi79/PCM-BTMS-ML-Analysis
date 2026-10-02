import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("pcm_btms_dataset.csv")

columns = [
    "Tm",
    "rho",
    "cp",
    "k",
    "Lf",
    "Aavg",
    "C_rate",
    "Tmax",
    "dTmax",
    "phi_avg"
]

corr = df[columns].corr(method="pearson")

plt.figure(figsize=(11, 9))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True
)

plt.title("Pearson Correlation Matrix of PCM-BTMS Parameters")
plt.tight_layout()

plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()