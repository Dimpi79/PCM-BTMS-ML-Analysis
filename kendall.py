import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

kendall_values = []

for feature in inputs:

    row = []

    for target in outputs:

        value = df[feature].corr(
            df[target],
            method="kendall"
        )

        row.append(value)

    kendall_values.append(row)

kendall_df = pd.DataFrame(
    kendall_values,
    index=inputs,
    columns=outputs
)

print("\nKENDALL RANK CORRELATION")
print("=" * 60)
print(kendall_df.round(3))

plt.figure(figsize=(8, 7))

sns.heatmap(
    kendall_df,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0
)

plt.title("Kendall Rank Correlation: PCM-BTMS Inputs vs Thermal Outputs")

plt.xlabel("Thermal Output")
plt.ylabel("Input Parameter")

plt.tight_layout()

plt.savefig("kendall_heatmap.png", dpi=300)
plt.show()