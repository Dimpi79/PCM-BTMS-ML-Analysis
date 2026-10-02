import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pcm_btms_dataset.csv")

inputs = [
    "Tm",
    "rho",
    "cp",
    "k",
    "Lf",
    "Aavg"
]

target = "Tmax"

fig, axes = plt.subplots(3, 2, figsize=(12, 14))

axes = axes.flatten()

for ax, feature in zip(axes, inputs):

    ax.scatter(
        df[feature],
        df[target],
        alpha=0.7
    )

    ax.set_xlabel(feature)
    ax.set_ylabel("Tmax (K)")
    ax.set_title(f"{feature} vs Tmax")
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "Input Parameters vs Maximum Battery Temperature",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "input_output_scatter.png",
    dpi=300
)

plt.show()