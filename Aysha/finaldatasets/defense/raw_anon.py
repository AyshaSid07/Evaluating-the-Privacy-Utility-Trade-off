"""
Compare k-Anonymity across all defenses using multiple k thresholds.
Shows % of records satisfying k=2, k=5, k=10 for each defense.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_raw = pd.read_csv("../csvfiles/telecom_dataset.csv")

defenses = {
    "Generalization": pd.read_csv("../defense/generalized_telecom.csv"),
    "Attribute\nSuppression": pd.read_csv("../defense/attribute_suppressed_telecom.csv"),
    "Cell\nSuppression": pd.read_csv("../defense/cell_suppressed_telecom.csv"),
    "Data\nSwapping": pd.read_csv("../defense/swapped_telecom.csv"),
    "Generalization\n+ Suppression": pd.read_csv("../defense/generalized_and_suppressed_telecom.csv"),
    "Row Suppression\nk-Anon (k=5)": pd.read_csv("../defense/row_suppressed_telecom.csv"),
}

quasi_identifiers = ['Device_Type', 'Network_Type', 'PLMN', 'LAI']
k_thresholds = [2, 5, 10]

def pct_records_satisfying_k(df, qis, k):
    qi_cols = [c for c in qis if c in df.columns]
    if len(qi_cols) == 0 or len(df) == 0:
        return 0
    group_sizes = df.groupby(qi_cols).size()
    safe_records = group_sizes[group_sizes >= k].sum()
    return round((safe_records / len(df)) * 100, 1)

methods = ["Raw\n(No Defense)"] + list(defenses.keys())
all_dfs = [df_raw] + list(defenses.values())

# Calculate for each k threshold
results = {k: [] for k in k_thresholds}
for df in all_dfs:
    for k in k_thresholds:
        results[k].append(pct_records_satisfying_k(df, quasi_identifiers, k))

# Plot grouped bar chart
x = np.arange(len(methods))
width = 0.25
colors_k = ['#3498db', '#2ecc71', '#e67e22']

fig, ax = plt.subplots(figsize=(14, 7))

for i, k in enumerate(k_thresholds):
    bars = ax.bar(x + i * width, results[k], width, label=f'k ≥ {k}',
                  color=colors_k[i], edgecolor='black')
    for bar, val in zip(bars, results[k]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{val}%", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('% of Records Satisfying k-Anonymity', fontsize=12)
ax.set_xlabel('Defense Method', fontsize=12)
ax.set_title('k-Anonymity Comparison Across Defense Methods (k=2, k=5, k=10)', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylim(0, 115)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("../defense/k_anonymity_comparison.png", dpi=300)
plt.show()

# Print summary
print(f"\n{'Method':<30}", end="")
for k in k_thresholds:
    print(f"{'k≥'+str(k):<12}", end="")
print()
print("-" * 66)
for i, m in enumerate(methods):
    print(f"{m.replace(chr(10), ' '):<30}", end="")
    for k in k_thresholds:
        print(f"{results[k][i]:<12}", end="")
    print()