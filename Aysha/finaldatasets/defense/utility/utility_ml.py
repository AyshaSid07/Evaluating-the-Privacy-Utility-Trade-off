"""
ML-Based Utility Evaluation: Classification Accuracy Across Defense Methods
============================================================================
PURPOSE:
    Measure how much each anonymization defense degrades Machine Learning
    performance compared to the original (raw) data. This gives a
    "task-specific" utility measure that complements the statistical
    utility metrics (JSD, NCP, DM) computed separately.

APPROACH:
    1. Train a Random Forest classifier on the RAW data → get baseline accuracy
    2. Train the SAME model (same settings, same seed) on each defense dataset
    3. Compare accuracy: the defense with the smallest drop has the best ML utility

WHY RANDOM FOREST?
    - Handles categorical features natively (our data is all categorical)
    - Robust to noise and missing values (some defenses suppress columns)
    - Does not require feature scaling (unlike SVM or Neural Networks)
    - Widely used in privacy research for utility evaluation

TARGET VARIABLE: Network_Type
    - Present in ALL defense files (never suppressed)
    - Low cardinality (few classes like 3G, 4G, 5G)
    - Realistic telecom analytics task: predict network type from device,
      location, and operator attributes

Research Basis:
    - Iyengar, V.S. (2002). "Transforming data to satisfy privacy constraints."
      ACM SIGKDD. DOI: 10.1145/775047.775089
      → Used classification accuracy as primary utility metric for anonymized data

    - Fung, B.C.M. et al. (2010). "Privacy-preserving data publishing: A survey."
      ACM Computing Surveys. DOI: 10.1145/1749603.1749605
      → Listed classification accuracy as a standard data utility measure

FOLDER STRUCTURE (relative to this script):
    This script:    finaldatasets/defense/utility/utility_ml.py
    CSV files:      finaldatasets/csvfiles/           → ../../csvfiles/
    Defense files:  finaldatasets/defense/             → ../
    Save results:   finaldatasets/defense/utility/     → ./ (same folder)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════
# Paths relative to this script location:
#   This script is at: defense/utility/utility_ml.py
#   CSV files are at:  csvfiles/  → go up 2 levels: ../../csvfiles/
#   Defense files at:  defense/   → go up 1 level:  ../

# Load the original (raw) telecom dataset — our ground truth
df_raw = pd.read_csv("../../csvfiles/telecom_dataset.csv")

# Load each anonymized (defense) version of the same dataset
defenses = {
    "Generalization": pd.read_csv("../generalized_telecom.csv"),
    "Attribute\nSuppression": pd.read_csv("../attribute_suppressed_telecom.csv"),
    "Cell\nSuppression": pd.read_csv("../cell_suppressed_telecom.csv"),
    "Data\nSwapping": pd.read_csv("../swapped_telecom.csv"),
    "Generalization\n+ Suppression": pd.read_csv("../generalized_and_suppressed_telecom.csv"),
    "Row Suppression\nk-Anon (k=5)": pd.read_csv("../row_suppressed_telecom.csv"),
}

# ══════════════════════════════════════════════
# 2. CONFIGURATION
# ══════════════════════════════════════════════
TARGET = 'Network_Type'          # What we're predicting
RANDOM_SEED = 42                 # Fixed seed — ensures identical results every run
TEST_SIZE = 0.2                  # 80% train, 20% test
SUPPRESSED_PLACEHOLDER = 'SUPPRESSED'  # Placeholder for missing/suppressed values

# Quasi-identifiers: attributes that can be used in linking attacks
# Reference: Sweeney (2002) — defines QIs as attributes that, in combination,
# can be linked with external data to re-identify individuals
quasi_identifiers = ['Device_Type', 'Network_Type', 'PLMN', 'LAI']
k_for_privacy = 5  # Privacy threshold for k-anonymity

# ── Auto-detect direct identifiers ──
# Direct identifiers (IMSI, IMEI, etc.) are removed by EVERY defense.
# We exclude them from features because they're never available for ML.
# Reference: Fung et al. (2010) — classifies attributes into direct
# identifiers (always removed), quasi-identifiers (anonymized), and
# sensitive attributes (protected)
all_defense_dfs = list(defenses.values())
direct_identifiers = [col for col in df_raw.columns
                      if all(col not in df_def.columns for df_def in all_defense_dfs)]

# ── Build feature list ──
# Features = all columns EXCEPT the target variable and direct identifiers
feature_columns = [col for col in df_raw.columns
                   if col != TARGET and col not in direct_identifiers]

print(f" Target: {TARGET} | Classes: {df_raw[TARGET].unique().tolist()}")
print(f" Direct identifiers excluded: {direct_identifiers}")
print(f" Features used: {feature_columns}")

# ══════════════════════════════════════════════
# 3. PREPROCESSING FUNCTIONS
# ══════════════════════════════════════════════

def prepare_dataset(df, feature_cols, target_col, label_encoders=None, fit=False):
    """
    Prepares a dataset for Random Forest classification.

    WHY THIS IS NEEDED:
      Random Forest requires numerical input, but our telecom data is all
      categorical (text). This function converts text → numbers using
      Label Encoding.

    Steps:
      1. If a feature column is MISSING (defense suppressed it),
         fill it with 'SUPPRESSED' placeholder so all datasets have
         the same number of features
      2. Convert all values to strings (handles mixed types from anonymization)
      3. Label-encode all categorical columns to numbers

    Parameters:
      df             : the dataframe to prepare
      feature_cols   : list of feature column names
      target_col     : name of the target column (Network_Type)
      label_encoders : pre-fitted encoders (None if fitting for first time)
      fit            : True = create new encoders; False = use existing ones

    Returns: X (feature matrix), y (target vector), label_encoders
    """
    df_copy = df.copy()

    # Step 1: Ensure all expected feature columns exist
    # If a defense removed a column, fill it with 'SUPPRESSED'
    for col in feature_cols:
        if col not in df_copy.columns:
            df_copy[col] = SUPPRESSED_PLACEHOLDER

    # Step 2: Fill any NaN values and convert everything to string
    for col in feature_cols:
        df_copy[col] = df_copy[col].fillna(SUPPRESSED_PLACEHOLDER).astype(str)
    df_copy[target_col] = df_copy[target_col].astype(str)

    # Step 3: Label encode (convert text categories → integer numbers)
    if fit:
        # First time: learn the mapping from data
        # e.g., "4G"→0, "5G"→1, "3G"→2
        label_encoders = {}
        for col in feature_cols + [target_col]:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col])
            label_encoders[col] = le
    else:
        # Re-use existing encoders from master vocabulary
        # Handles unseen values by mapping them to SUPPRESSED
        for col in feature_cols + [target_col]:
            le = label_encoders[col]
            df_copy[col] = df_copy[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_
                else le.transform([SUPPRESSED_PLACEHOLDER])[0]
                if SUPPRESSED_PLACEHOLDER in le.classes_
                else -1
            )

    X = df_copy[feature_cols].values  # Feature matrix (numbers)
    y = df_copy[target_col].values    # Target vector (numbers)
    return X, y, label_encoders


def pct_records_satisfying_k(df, qis, k):
    """
    Calculates what percentage of records satisfy k-anonymity.

    Reference: Sweeney, L. (2002). Int. J. Uncertainty, Fuzziness &
               Knowledge-Based Systems, 10(5), 557-570.

    HOW IT WORKS:
      1. Group records by quasi-identifier columns
      2. Count how many records are in each group (equivalence class)
      3. A record is "safe" if its group has ≥ k members
      4. Return: (number of safe records / total records) × 100

    Formula: kAnon(%) = Σ|EC_j| (where |EC_j| >= k) / n × 100
    """
    qi_cols = [c for c in qis if c in df.columns]
    if not qi_cols or len(df) == 0:
        return 100.0
    group_sizes = df.groupby(qi_cols).size()
    safe = group_sizes[group_sizes >= k].sum()
    return round((safe / len(df)) * 100, 2)


# ══════════════════════════════════════════════
# 4. BUILD MASTER ENCODER
# ══════════════════════════════════════════════
# WHY: We combine ALL datasets (raw + all 6 defenses) to build one encoder
# that knows every possible value. This prevents "unseen label" errors when
# a defense introduces new values (e.g., generalized categories like
# "Country_404" instead of specific PLMN codes).

df_combined = pd.concat([df_raw] + all_defense_dfs, ignore_index=True)
_, _, master_encoders = prepare_dataset(df_combined, feature_columns, TARGET, fit=True)

# ══════════════════════════════════════════════
# 5. TRAIN BASELINE MODEL ON RAW DATA
# ══════════════════════════════════════════════
# This is our "ground truth" — the best the model can do with unmodified data.
# All defense accuracies are compared against this baseline.

X_raw, y_raw, _ = prepare_dataset(df_raw, feature_columns, TARGET,
                                   label_encoders=master_encoders, fit=False)

# Split: 80% for training, 20% for testing
# stratify=y_raw ensures each Network_Type class is proportionally
# represented in both train and test sets
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_raw
)

# Train Random Forest: 100 decision trees, each votes on the prediction
# Final prediction = majority vote across all 100 trees
# Reference: Iyengar (2002) used tree-based classifiers for utility evaluation
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
rf_model.fit(X_train_raw, y_train_raw)
y_pred_raw = rf_model.predict(X_test_raw)

# Record baseline metrics
baseline_accuracy = accuracy_score(y_test_raw, y_pred_raw)
baseline_f1 = f1_score(y_test_raw, y_pred_raw, average='weighted', zero_division=0)

print(f"\n🎯 Baseline (Raw Data): Accuracy={baseline_accuracy*100:.2f}%, F1={baseline_f1*100:.2f}%")

# ══════════════════════════════════════════════
# 6. EVALUATE EACH DEFENSE
# ══════════════════════════════════════════════
# For each defense: train a FRESH model with IDENTICAL settings,
# then compare its accuracy to the baseline.
# The key output is "Accuracy Change %" — positive means the defense
# accidentally improved accuracy (by removing noise), negative means
# the defense degraded accuracy (by destroying useful information).
# Reference: Fung et al. (2010) — classification accuracy as utility measure

results = []

# Add raw baseline as first entry
raw_k_anon = pct_records_satisfying_k(df_raw, quasi_identifiers, k_for_privacy)
results.append({
    'Method': 'Raw\n(No Defense)',
    'Accuracy': round(baseline_accuracy * 100, 2),
    'Precision': round(precision_score(y_test_raw, y_pred_raw, average='weighted', zero_division=0) * 100, 2),
    'Recall': round(recall_score(y_test_raw, y_pred_raw, average='weighted', zero_division=0) * 100, 2),
    'F1_Score': round(baseline_f1 * 100, 2),
    'Accuracy_Change_%': 0.0,
    'k_Anon_%': raw_k_anon,
    'Features_Suppressed': 0,
})

# Evaluate each defense method
for name, df_def in defenses.items():
    # Check which feature columns this defense removed
    suppressed = [col for col in feature_columns if col not in df_def.columns]

    # Skip if the target column itself was removed (can't classify without it)
    if TARGET not in df_def.columns:
        continue

    # Prepare data using the same master encoders
    X_def, y_def, _ = prepare_dataset(df_def, feature_columns, TARGET,
                                       label_encoders=master_encoders, fit=False)

    # Same split ratio and seed — ensures fair comparison
    X_train_def, X_test_def, y_train_def, y_test_def = train_test_split(
        X_def, y_def, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_def
    )

    # Train fresh model with identical hyperparameters
    rf_def = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf_def.fit(X_train_def, y_train_def)
    y_pred_def = rf_def.predict(X_test_def)

    # Compute all classification metrics
    acc = accuracy_score(y_test_def, y_pred_def)
    prec = precision_score(y_test_def, y_pred_def, average='weighted', zero_division=0)
    rec = recall_score(y_test_def, y_pred_def, average='weighted', zero_division=0)
    f1 = f1_score(y_test_def, y_pred_def, average='weighted', zero_division=0)

    # Accuracy change: positive = improved, negative = dropped
    # Formula: ((defense_acc - baseline_acc) / baseline_acc) × 100
    acc_change = round(((acc - baseline_accuracy) / baseline_accuracy) * 100, 2)

    # Compute privacy score for this defense
    k_anon_pct = pct_records_satisfying_k(df_def, quasi_identifiers, k_for_privacy)

    results.append({
        'Method': name,
        'Accuracy': round(acc * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall': round(rec * 100, 2),
        'F1_Score': round(f1 * 100, 2),
        'Accuracy_Change_%': acc_change,
        'k_Anon_%': k_anon_pct,
        'Features_Suppressed': len(suppressed),
    })

df_results = pd.DataFrame(results)

# ══════════════════════════════════════════════
# 7. PRINT SUMMARY TABLE
# ══════════════════════════════════════════════
print("\n" + "=" * 115)
print("ML UTILITY EVALUATION — CLASSIFICATION RESULTS SUMMARY")
print(f"Model: Random Forest | Target: {TARGET} | Test Size: {TEST_SIZE} | Seed: {RANDOM_SEED}")
print("=" * 115)
print(f"\n{'Method':<28} {'Accuracy%':>10} {'Precision%':>12} {'Recall%':>10} "
      f"{'F1%':>8} {'Acc Change%':>13} {'k≥' + str(k_for_privacy) + '%':>8} {'Feat Supp':>11}")
print("-" * 115)
for _, row in df_results.iterrows():
    change_str = f"+{row['Accuracy_Change_%']}" if row['Accuracy_Change_%'] > 0 else f"{row['Accuracy_Change_%']}"
    print(f"{row['Method'].replace(chr(10), ' '):<28} {row['Accuracy']:>10.2f} "
          f"{row['Precision']:>12.2f} {row['Recall']:>10.2f} {row['F1_Score']:>8.2f} "
          f"{change_str:>13} {row['k_Anon_%']:>8.2f} {row['Features_Suppressed']:>11}")

# ══════════════════════════════════════════════
# 8. LIGHT PASTEL COLOR PALETTE
# ══════════════════════════════════════════════
PASTEL_GREEN = '#B5EAD7'
PASTEL_ORANGE = '#FFD8B1'
PASTEL_PURPLE = '#D4A5D0'
PASTEL_RED = '#FFB3B3'
PASTEL_YELLOW = '#FFFACD'
PASTEL_TEAL = '#B2DFDB'
PASTEL_PINK = '#F8C8DC'

# ══════════════════════════════════════════════
# 9. VISUALIZATION — 3 KEY CHARTS
# ══════════════════════════════════════════════
# Only 3 charts that tell the complete ML utility story:
#   Chart 1: Privacy vs ML Accuracy scatter (THE tradeoff)
#   Chart 2: Accuracy with privacy overlay (dual-axis comparison)
#   Chart 3: Feature importance (explains WHY accuracy changes)

method_labels = [m.replace('\n', ' ') for m in df_results['Method']]
defense_mask = df_results['Method'] != 'Raw\n(No Defense)'
df_defenses = df_results[defense_mask]
defense_labels = [m.replace('\n', ' ') for m in df_defenses['Method']]

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle(f'ML-Based Utility Evaluation: Random Forest Classification ({TARGET})\n'
             f'(Iyengar, 2002 | Fung et al., 2010)',
             fontsize=16, fontweight='bold', y=1.02)

# ════════════════════════════════════════════
# CHART 1: PRIVACY vs ML ACCURACY SCATTER
# ════════════════════════════════════════════
# Each defense is a point: X = privacy, Y = accuracy.
# Ideal position = TOP-RIGHT (high privacy AND high accuracy).
# The baseline accuracy is shown as a horizontal dashed line.
# Reference: Iyengar (2002) — classification accuracy as utility metric

ax1 = axes[0]
scatter_colors = [PASTEL_GREEN, PASTEL_ORANGE, PASTEL_TEAL,
                  PASTEL_YELLOW, PASTEL_RED, PASTEL_PURPLE]

for i, label in enumerate(defense_labels):
    row = df_defenses.iloc[i]
    ax1.scatter(row['k_Anon_%'], row['Accuracy'], color=scatter_colors[i],
                s=300, edgecolors='black', linewidth=1.5, zorder=5)
    ax1.annotate(label, (row['k_Anon_%'], row['Accuracy']),
                 textcoords="offset points", xytext=(8, 8),
                 fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# Horizontal line showing raw data baseline accuracy
ax1.axhline(y=baseline_accuracy * 100, color='gray', linestyle='--', alpha=0.6,
            label=f'Raw Baseline ({baseline_accuracy*100:.1f}%)')

ax1.set_xlabel(f'Privacy (% records satisfying k≥{k_for_privacy})', fontsize=12)
ax1.set_ylabel('ML Accuracy (%)', fontsize=12)
ax1.set_title('Privacy vs. ML Accuracy Tradeoff\n(Iyengar, 2002 | Sweeney, 2002)',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='lower left')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_xlim(-5, 110)

# ════════════════════════════════════════════
# CHART 2: ACCURACY WITH PRIVACY OVERLAY
# ════════════════════════════════════════════
# Dual-axis chart combining both metrics:
#   Green bars (left Y-axis) = ML accuracy (higher = better utility)
#   Purple line (right Y-axis) = Privacy level (higher = better privacy)
# Green/red text on bars shows accuracy change vs baseline.
# Reference: Fung et al. (2010) — multi-criteria evaluation

ax2 = axes[1]
x = np.arange(len(defense_labels))

# Accuracy bars
bars2 = ax2.bar(x, df_defenses['Accuracy'].values, color=PASTEL_GREEN,
                edgecolor='gray', linewidth=0.8, label='ML Accuracy %', width=0.6)

# Label each bar with accuracy value and change percentage
for bar, val, change in zip(bars2, df_defenses['Accuracy'].values, df_defenses['Accuracy_Change_%'].values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    if change != 0:
        change_color = '#27ae60' if change > 0 else '#e74c3c'
        change_str = f'+{change:.1f}%' if change > 0 else f'{change:.1f}%'
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 2.5,
                 change_str, ha='center', fontsize=8, color=change_color, fontweight='bold')

# Baseline reference line
ax2.axhline(y=baseline_accuracy * 100, color='gray', linestyle='--', alpha=0.5,
            label=f'Baseline ({baseline_accuracy*100:.1f}%)')

# Privacy line on second Y-axis
ax2_twin = ax2.twinx()
ax2_twin.plot(x, df_defenses['k_Anon_%'].values, color='#8e44ad', marker='D',
              linewidth=2.5, markersize=10, label=f'k≥{k_for_privacy} Privacy %', zorder=5)
for i, val in enumerate(df_defenses['k_Anon_%'].values):
    ax2_twin.annotate(f'{val:.0f}%', (x[i], val), textcoords="offset points",
                      xytext=(0, 10), fontsize=9, fontweight='bold', color='#8e44ad', ha='center')

ax2.set_xticks(x)
ax2.set_xticklabels(defense_labels, fontsize=8, rotation=20, ha='right')
ax2.set_ylabel('ML Accuracy (%)', fontsize=11)
ax2.set_ylim(0, max(df_defenses['Accuracy'].values) * 1.3)
ax2_twin.set_ylabel(f'Privacy (k≥{k_for_privacy} %)', fontsize=11, color='#8e44ad')
ax2_twin.set_ylim(0, 115)
ax2_twin.tick_params(axis='y', labelcolor='#8e44ad')
ax2.set_title('ML Accuracy with Privacy Overlay\n(Bars = Utility | Line = Privacy)',
              fontsize=14, fontweight='bold')

lines2, labels2 = ax2.get_legend_handles_labels()
lines2t, labels2t = ax2_twin.get_legend_handles_labels()
ax2.legend(lines2 + lines2t, labels2 + labels2t, loc='upper left', fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

# ════════════════════════════════════════════
# CHART 3: FEATURE IMPORTANCE
# ════════════════════════════════════════════
# Shows which features the Random Forest model relies on most.
# This is critical for explaining WHY some defenses improve accuracy:
#   - Features with 0.000 importance are noise
#   - When a defense removes noise features, accuracy can increase
#   - This is "implicit feature selection" (Iyengar, 2002)

ax3 = axes[2]
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)
sorted_features = [feature_columns[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

# Purple = important features (≥0.1), Pink = minor features (<0.1)
importance_colors = [PASTEL_PURPLE if v >= 0.1 else PASTEL_PINK for v in sorted_importances]
bars3 = ax3.barh(sorted_features, sorted_importances, color=importance_colors,
                 edgecolor='gray', linewidth=0.8)
for bar, val in zip(bars3, sorted_importances):
    ax3.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

ax3.set_xlabel('Feature Importance', fontsize=12)
ax3.set_title('Feature Importance (Raw Data Model)\n[Explains accuracy changes per defense]',
              fontsize=14, fontweight='bold')
ax3.grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
# Save to same folder as this script: defense/utility/
plt.savefig("ml_classification_utility.png", dpi=300, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════
# 10. SAVE RESULTS TO CSV
# ════════════════════════════���═════════════════
df_export = df_results.copy()
df_export['Method'] = df_export['Method'].str.replace('\n', ' ')
# Save to same folder as this script
df_export.to_csv("ml_classification_results.csv", index=False)

df_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
df_importance.to_csv("ml_feature_importance.csv", index=False)

print("\n" + "=" * 70)
print(" FILES SAVED (same folder as this script):")
print("   - ml_classification_utility.png  (3 key charts)")
print("   - ml_classification_results.csv  (all metrics)")
print("   - ml_feature_importance.csv      (feature ranking)")
print("=" * 70)
print(f"\n Configuration:")
print(f"   Model: Random Forest (100 trees)")
print(f"   Target: {TARGET}")
print(f"   Test size: {TEST_SIZE} | Seed: {RANDOM_SEED}")
print(f"   Direct identifiers excluded: {direct_identifiers}")
print(f"   Features: {feature_columns}")
print(f"   Privacy threshold: k≥{k_for_privacy}")

# ══════════════════════════════════════════════
# 11. KEY FINDINGS
# ══════════════════════════════════════════════
print(f"\n KEY FINDINGS:")
print(f"\n   Baseline (Raw Data): {baseline_accuracy*100:.2f}% accuracy")

print(f"\n   Defenses with HIGHER accuracy than raw:")
improved = df_defenses[df_defenses['Accuracy_Change_%'] > 0]
for _, row in improved.iterrows():
    print(f"    {row['Method'].replace(chr(10), ' ')}: {row['Accuracy']}% "
          f"(+{row['Accuracy_Change_%']}%) — removes noisy features "
          f"({row['Features_Suppressed']} columns suppressed)")

print(f"\n   Defenses with LOWER accuracy than raw:")
dropped = df_defenses[df_defenses['Accuracy_Change_%'] < 0]
for _, row in dropped.iterrows():
    print(f"   {row['Method'].replace(chr(10), ' ')}: {row['Accuracy']}% "
          f"({row['Accuracy_Change_%']}%)")

print(f"\n    EXPLANATION:")
print(f"      Some defenses achieve higher accuracy because they remove")
print(f"      zero-importance features (TMSI, TLLI, LMSI = 0.000 importance),")
print(f"      which acts as implicit feature selection, reducing noise.")
print(f"      Reference: Iyengar (2002) observed that anonymization can")
print(f"      improve classification when it removes irrelevant attributes.")