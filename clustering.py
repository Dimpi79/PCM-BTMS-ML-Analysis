import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv("topsis_results.csv")

features = [
    "C_rate",
    "Tm",
    "rho",
    "cp",
    "k",
    "Lf",
    "Aavg",
    "Tmax",
    "dTmax",
    "phi_avg"
]

X = df[features]

# --------------------------------------------------
# Standardize
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# K-Means
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("=" * 60)
print("CLUSTER COUNTS")
print("=" * 60)

print(
    df["Cluster"].value_counts().sort_index()
)

# --------------------------------------------------
# Cluster statistics
# --------------------------------------------------

print("\nCLUSTER CHARACTERISTICS")
print("=" * 60)

print(
    df.groupby("Cluster")[
        [
            "C_rate",
            "Tmax",
            "dTmax",
            "phi_avg",
            "TOPSIS_score"
        ]
    ].mean().round(4)
)

# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    "clustering_results.csv",
    index=False
)

# --------------------------------------------------
# TOPSIS distribution by cluster
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="TOPSIS_score",
    hue="Cluster",
    bins=20,
    multiple="layer",
    alpha=0.5
)

plt.title(
    "TOPSIS Score Distribution by Cluster"
)

plt.xlabel("TOPSIS Score")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "clustering.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(9, 7))

for cluster in sorted(df["Cluster"].unique()):

    subset = df[df["Cluster"] == cluster]

    plt.scatter(
        subset["Tmax"],
        subset["dTmax"],
        label=f"Cluster {cluster}",
        alpha=0.7
    )

plt.xlabel("Tmax (K)")
plt.ylabel("ΔTmax (K)")

plt.title(
    "K-Means Clustering of PCM-BTMS Thermal Cases"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "cluster_visualization.png",
    dpi=300
)

plt.show()