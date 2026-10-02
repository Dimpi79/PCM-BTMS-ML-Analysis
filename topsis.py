import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("pcm_btms_dataset.csv")

# --------------------------------------------------
# TOPSIS criteria
# --------------------------------------------------

criteria = [
    "Tmax",
    "dTmax",
    "phi_avg"
]

# 1 = benefit, -1 = cost
types = np.array([
    -1,   # Tmax: lower is better
    -1,   # dTmax: lower is better
     1    # phi_avg: higher is better
])

weights = np.array([
    0.40,
    0.35,
    0.25
])

X = df[criteria].values.astype(float)

# --------------------------------------------------
# Vector normalization
# --------------------------------------------------

norm = np.sqrt((X ** 2).sum(axis=0))

R = X / norm

# --------------------------------------------------
# Weighted normalized matrix
# --------------------------------------------------

V = R * weights

# --------------------------------------------------
# Ideal solutions
# --------------------------------------------------

ideal_best = np.zeros(len(criteria))
ideal_worst = np.zeros(len(criteria))

for j in range(len(criteria)):

    if types[j] == 1:

        ideal_best[j] = V[:, j].max()
        ideal_worst[j] = V[:, j].min()

    else:

        ideal_best[j] = V[:, j].min()
        ideal_worst[j] = V[:, j].max()

# --------------------------------------------------
# Distances
# --------------------------------------------------

distance_best = np.sqrt(
    ((V - ideal_best) ** 2).sum(axis=1)
)

distance_worst = np.sqrt(
    ((V - ideal_worst) ** 2).sum(axis=1)
)

# --------------------------------------------------
# TOPSIS score
# --------------------------------------------------

score = distance_worst / (
    distance_best + distance_worst
)

df["TOPSIS_score"] = score

df["TOPSIS_rank"] = (
    df["TOPSIS_score"]
    .rank(
        ascending=False,
        method="min"
    )
)

# --------------------------------------------------
# Results
# --------------------------------------------------

result = df.sort_values(
    "TOPSIS_score",
    ascending=False
)

print("=" * 60)
print("TOPSIS RESULTS")
print("=" * 60)

print(
    result[
        [
            "C_rate",
            "Tmax",
            "dTmax",
            "phi_avg",
            "TOPSIS_score",
            "TOPSIS_rank"
        ]
    ].head(10)
)

# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    "topsis_results.csv",
    index=False
)

# --------------------------------------------------
# Distribution
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.hist(
    df["TOPSIS_score"],
    bins=20
)

plt.xlabel("TOPSIS Score")
plt.ylabel("Frequency")

plt.title(
    "TOPSIS Score Distribution of PCM-BTMS Cases"
)

plt.tight_layout()

plt.savefig(
    "topsis_distribution.png",
    dpi=300
)

plt.show()