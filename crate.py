import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pcm_btms_dataset.csv")

mean_values = df.groupby("C_rate")[
    ["Tmax", "dTmax", "phi_avg"]
].mean()

print(mean_values)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Tmax
axes[0].plot(
    mean_values.index,
    mean_values["Tmax"],
    marker="o"
)

axes[0].set_xlabel("C-rate")
axes[0].set_ylabel("Mean Tmax (K)")
axes[0].set_title("Maximum Temperature vs C-rate")
axes[0].grid(True)

# dTmax
axes[1].plot(
    mean_values.index,
    mean_values["dTmax"],
    marker="o"
)

axes[1].set_xlabel("C-rate")
axes[1].set_ylabel("Mean ΔTmax (K)")
axes[1].set_title("Temperature Rise vs C-rate")
axes[1].grid(True)

# phi
axes[2].plot(
    mean_values.index,
    mean_values["phi_avg"],
    marker="o"
)

axes[2].set_xlabel("C-rate")
axes[2].set_ylabel("Mean Liquid Fraction")
axes[2].set_title("Liquid Fraction vs C-rate")
axes[2].grid(True)

plt.tight_layout()

plt.savefig(
    "crate_analysis.png",
    dpi=300
)

plt.show()