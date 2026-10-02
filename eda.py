import pandas as pd

df = pd.read_csv("pcm_btms_dataset.csv")

# ==========================================
# BASIC STATISTICS
# ==========================================

print("=" * 60)
print("BASIC STATISTICS")
print("=" * 60)

print(df.describe().T)

# ==========================================
# CORRELATION MATRIX
# ==========================================

print("\n" + "=" * 60)
print("PEARSON CORRELATION MATRIX")
print("=" * 60)

print(df.corr(numeric_only=True).round(3))

# ==========================================
# CORRELATION WITH TARGETS
# ==========================================

targets = ["Tmax", "dTmax", "phi_avg"]

print("\n" + "=" * 60)
print("CORRELATION WITH TARGETS")
print("=" * 60)

corr = df.corr(numeric_only=True)

for target in targets:
    print(f"\n--- {target} ---")
    print(
        corr[target]
        .drop(target)
        .sort_values(ascending=False)
        .round(3)
    )

# ==========================================
# MEAN OUTPUTS BY C-RATE
# ==========================================

print("\n" + "=" * 60)
print("MEAN OUTPUTS BY C-RATE")
print("=" * 60)

print(
    df.groupby("C_rate")[["Tmax", "dTmax", "phi_avg"]]
    .mean()
    .round(4)
)

# ==========================================
# MEDIAN OUTPUTS BY C-RATE
# ==========================================

print("\n" + "=" * 60)
print("MEDIAN OUTPUTS BY C-RATE")
print("=" * 60)

print(
    df.groupby("C_rate")[["Tmax", "dTmax", "phi_avg"]]
    .median()
    .round(4)
)