"""
Statistical Privacy-Utility Tradeoff Analysis Across Defense Methods
=====================================================================
PURPOSE:
    Measure how much each anonymization defense degrades data quality
    using established statistical metrics from the privacy research literature.
    This is a DATA-CENTRIC evaluation — it measures how much the data itself
    changed, regardless of any specific downstream task.

METRICS AND THEIR RESEARCH BASIS:

    1. Jensen-Shannon Divergence (JSD)
       Paper: Lin, J. (1991). "Divergence measures based on the Shannon entropy."
              IEEE Transactions on Information Theory, 37(1), 145-151.
              DOI: 10.1109/18.61115
       What:  Measures how much the frequency distribution of each column changed.
       Range: 0 (identical) to 1 (maximally different)

    2. Normalized Certainty Penalty (NCP)
       Paper: Xu, J. et al. (2006). "Utility-based anonymization using local recoding."
              ACM SIGKDD, pp. 785-790. DOI: 10.1145/1150402.1150503
       What:  Measures what fraction of distinct values were lost in quasi-identifiers.
       Range: 0 (no loss) to 1 (all values collapsed)

    3. Discernibility Metric (DM)
       Paper: Bayardo, R.J. & Agrawal, R. (2005). "Data privacy through optimal
              k-anonymization." IEEE ICDE, pp. 217-228. DOI: 10.1109/ICDE.2005.42
       What:  Penalizes large equivalence classes where records become indistinguishable.
       Range: 0 (all records unique) to 1 (all records in one group)

    4. Record Loss Rate
       Paper: Li, N. et al. (2007). "t-Closeness: Privacy beyond k-anonymity and
              l-diversity." IEEE ICDE, pp. 106-115. DOI: 10.1109/ICDE.2007.367856
       What:  Percentage of records deleted during anonymization.
       Range: 0% (no records removed) to 100% (all records removed)

    5. k-Anonymity Satisfaction
       Paper: Sweeney, L. (2002). "k-Anonymity: A model for protecting privacy."
              Int. J. Uncertainty, Fuzziness and Knowledge-Based Systems, 10(5), 557-570.
              DOI: 10.1142/S0218488502001648
       What:  Percentage of records in equivalence classes of size >= k.
       Range: 0% (no record is safe) to 100% (all records satisfy k-anonymity)

    6. Average Equivalence Class Size
       Paper: Fung, B.C.M. et al. (2010). "Privacy-preserving data publishing: A survey."
              ACM Computing Surveys, 42(4), Article 14. DOI: 10.1145/1749603.1749605
       What:  Mean number of indistinguishable records per group.

COMPOSITE UTILITY SCORE:
    Weighted combination of the above metrics into a single 0-100 score.
    Weights follow the relative importance suggested in the literature:
      - JSD  = 40%  (distribution preservation is most critical for analysis)
      - NCP  = 30%  (distinct value preservation impacts query accuracy)
      - DM   = 20%  (record distinguishability affects individual-level analysis)
      - RecLoss = 10% (completeness matters but only for suppression methods)
    Reference for multi-criteria framework: Fung et al. (2010)

FOLDER STRUCTURE (relative to this script):
    This script:   finaldatasets/defense/utility/utility_static/utility_static.py
    CSV files:     finaldatasets/csvfiles/
    Defense files:  finaldatasets/defense/
    Save results:  finaldatasets/defense/utility/utility_static/  (same folder)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

# ══════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════

# Load the original (raw) telecom dataset — our ground truth
df_raw = pd.read_csv("../../../csvfiles/telecom_dataset.csv")

# Load each anonymized (defense) version of the same dataset
defenses = {
    "Generalization": pd.read_csv("../../generalized_telecom.csv"),
    "Attribute\nSuppression": pd.read_csv("../../attribute_suppressed_telecom.csv"),
    "Cell\nSuppression": pd.read_csv("../../cell_suppressed_telecom.csv"),
    "Data\nSwapping": pd.read_csv("../../swapped_telecom.csv"),
    "Generalization\n+ Suppression": pd.read_csv("../../generalized_and_suppressed_telecom.csv"),
    "Row Suppression\nk-Anon (k=5)": pd.read_csv("../../row_suppressed_telecom.csv"),
}

# ══════════════════════════════════════════════
# 2. CONFIGURATION
# ══════════════════════════════════════════════

# Quasi-identifiers: attributes that can be used in linking attacks
# Reference: Sweeney (2002) defines these as attributes that, in combination,
# can be linked with external data to re-identify individuals
quasi_identifiers = ['Device_Type', 'Network_Type', 'PLMN', 'LAI']

# Privacy threshold: minimum equivalence class size for k-anonymity
# k=5 means every record must be indistinguishable from at least 4 others
k_for_privacy = 5

# Composite utility weights (must sum to 100)
# These reflect the relative importance of each metric for data analysis tasks
WEIGHT_JSD = 40          # Distribution preservation (Lin, 1991)
WEIGHT_NCP = 30          # Distinct value preservation (Xu et al., 2006)
WEIGHT_DM = 20           # Record distinguishability (Bayardo & Agrawal, 2005)
WEIGHT_RECORD_LOSS = 10  # Data completeness (Li et al., 2007)

methods = list(defenses.keys())
all_defense_dfs = list(defenses.values())
all_original_columns = df_raw.columns.tolist()

# ══════════════════════════════════════════════
# 3. AUTO-DETECT DIRECT IDENTIFIERS
# ══════════════════════════════════════════════
# Direct identifiers (like IMSI, IMEI) uniquely identify a person and are
# ALWAYS removed by every defense method. Including them in JSD would add
# a fixed penalty to all methods, compressing the differences between them.
# Reference: Fung et al. (2010) classifies attributes into:
#   - Direct identifiers (always removed)
#   - Quasi-identifiers (anonymized)
#   - Sensitive attributes (protected)

direct_identifiers = []
for col in all_original_columns:
    # If a column is missing from EVERY defense file, it's a direct identifier
    if all(col not in df_def.columns for df_def in all_defense_dfs):
        direct_identifiers.append(col)

# JSD is computed only on columns where defenses actually differ
jsd_columns = [c for c in all_original_columns if c not in direct_identifiers]

print(f" Direct identifiers (removed by ALL defenses): {direct_identifiers}")
print(f" Columns used for JSD comparison: {jsd_columns}")
print(f" Quasi-identifiers for k-anonymity: {quasi_identifiers}")


# 4. UTILITY METRIC FUNCTIONS

# Each function implements one metric from the research literature.
# All functions take the original and anonymized dataframes as input
# and return a single number measuring information loss.

def compute_jsd(df_orig, df_anon, columns):
    """
    Jensen-Shannon Divergence (JSD) — measures distribution change.

    Reference: Lin, J. (1991). IEEE Trans. Information Theory, 37(1), 145-151.

    Formula:
        JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        where M = 0.5 * (P + Q)
        and KL is Kullback-Leibler divergence

    For each column:
        - Build frequency distribution P from original data
        - Build frequency distribution Q from anonymized data
        - Compute JSD between P and Q
        - If column is MISSING (suppressed), assign JSD = 1.0 (max penalty)

    Returns: Average JSD across all specified columns
    """
    jsd_values = []
    for col in columns:
        if col not in df_anon.columns:
            # Column was entirely suppressed — maximum information loss
            jsd_values.append(1.0)
        else:
            # Get all unique categories from both datasets
            all_cats = set(df_orig[col].astype(str).unique()) | set(df_anon[col].astype(str).unique())
            # Build normalized frequency distributions
            p = df_orig[col].astype(str).value_counts(normalize=True).reindex(all_cats, fill_value=0).sort_index()
            q = df_anon[col].astype(str).value_counts(normalize=True).reindex(all_cats, fill_value=0).sort_index()
            # Compute JSD using scipy (base=2 ensures JSD ∈ [0, 1])
            jsd_val = jensenshannon(p.values, q.values, base=2)
            jsd_values.append(jsd_val)
    return np.mean(jsd_values) if jsd_values else 1.0


def compute_per_column_jsd(df_orig, df_anon, columns):
    """
    Per-column JSD for heatmap visualization.
    Same logic as compute_jsd() but returns a dict of {column: JSD}.
    """
    jsd_per_col = {}
    for col in columns:
        if col not in df_anon.columns:
            jsd_per_col[col] = 1.0
        else:
            all_cats = set(df_orig[col].astype(str).unique()) | set(df_anon[col].astype(str).unique())
            p = df_orig[col].astype(str).value_counts(normalize=True).reindex(all_cats, fill_value=0).sort_index()
            q = df_anon[col].astype(str).value_counts(normalize=True).reindex(all_cats, fill_value=0).sort_index()
            jsd_per_col[col] = round(jensenshannon(p.values, q.values, base=2), 4)
    return jsd_per_col


def compute_ncp(df_orig, df_anon, qis):
    """
    Normalized Certainty Penalty (NCP) — measures distinct value loss.

    Reference: Xu, J. et al. (2006). ACM SIGKDD, pp. 785-790.

    Original formula (with taxonomy tree):
        NCP_A(EC) = |leaves of generalized node| / |total leaves of attribute A|

    Our approximation (without taxonomy tree):
        NCP(A) = 1 - (distinct values after / distinct values before)

    This approximation is valid when no predefined generalization hierarchy exists.
    NCP = 0 means all distinct values preserved; NCP = 1 means all collapsed.
    """
    ncp_values = []
    for qi in qis:
        if qi not in df_anon.columns:
            # Column entirely suppressed — maximum penalty
            ncp_values.append(1.0)
        elif qi in df_orig.columns:
            orig_unique = df_orig[qi].nunique()
            anon_unique = df_anon[qi].nunique()
            if orig_unique <= 1:
                ncp_values.append(0.0)
            else:
                ncp_values.append(1.0 - (anon_unique / orig_unique))
        else:
            ncp_values.append(1.0)
    return np.mean(ncp_values) if ncp_values else 0.0


def compute_discernibility_metric(df, qis):
    """
    Discernibility Metric (DM) — penalizes large equivalence classes.

    Reference: Bayardo, R.J. & Agrawal, R. (2005). IEEE ICDE, pp. 217-228.

    Formula:
        DM = Σ |EC_j|²     for all equivalence classes EC_j
        DM_normalized = DM / n²    (scaled to [0, 1])

    Interpretation:
        - Many small groups → DM close to 0 (good utility, records distinguishable)
        - One giant group   → DM = 1 (bad utility, all records look the same)
    """
    qi_cols = [c for c in qis if c in df.columns]
    n = len(df)
    if not qi_cols or n == 0:
        return 1.0
    group_sizes = df.groupby(qi_cols).size()
    dm = (group_sizes ** 2).sum()
    return dm / (n ** 2)


def compute_avg_eq_class_size(df, qis):
    """
    Average Equivalence Class Size — mean group size after anonymization.

    Reference: Fung, B.C.M. et al. (2010). ACM Computing Surveys, 42(4).

    Larger average = more records are indistinguishable = more privacy but less utility.
    """
    qi_cols = [c for c in qis if c in df.columns]
    if not qi_cols or len(df) == 0:
        return len(df) if len(df) > 0 else 1
    group_sizes = df.groupby(qi_cols).size()
    return group_sizes.mean()


def compute_record_loss_rate(df_orig, df_anon):
    """
    Record Loss Rate — percentage of records removed by anonymization.

    Reference: Li, N. et al. (2007). IEEE ICDE, pp. 106-115.

    Formula: RecordLoss(%) = (1 - |D_anon| / |D_orig|) × 100
    Only Row Suppression methods remove records; others keep all rows.
    """
    if len(df_orig) == 0:
        return 0.0
    return max(0, (1 - len(df_anon) / len(df_orig)) * 100)


def pct_records_satisfying_k(df, qis, k):
    """
    k-Anonymity Satisfaction — percentage of records in safe groups.

    Reference: Sweeney, L. (2002). Int. J. Uncertainty, Fuzziness &
               Knowledge-Based Systems, 10(5), 557-570.

    A record is "safe" if its equivalence class has >= k members.
    Formula: kAnon(%) = Σ|EC_j| (for all EC_j where |EC_j| >= k) / n × 100
    """
    qi_cols = [c for c in qis if c in df.columns]
    if not qi_cols or len(df) == 0:
        return 100.0
    group_sizes = df.groupby(qi_cols).size()
    safe = group_sizes[group_sizes >= k].sum()
    return round((safe / len(df)) * 100, 2)



# 5. COMPUTE ALL METRICS FOR EACH DEFENSE

results = []
for name, df_def in zip(methods, all_defense_dfs):
    # Compute each metric
    jsd = compute_jsd(df_raw, df_def, jsd_columns)
    avg_ec = compute_avg_eq_class_size(df_def, quasi_identifiers)
    dm = compute_discernibility_metric(df_def, quasi_identifiers)
    ncp = compute_ncp(df_raw, df_def, quasi_identifiers)
    record_loss = compute_record_loss_rate(df_raw, df_def)
    k_anon_pct = pct_records_satisfying_k(df_def, quasi_identifiers, k_for_privacy)

    # Composite Utility Score (0-100, higher = better utility)
    # Formula: weighted sum of (1 - metric) for each metric
    # Reference: Multi-criteria evaluation framework from Fung et al. (2010)
    utility_score = (
        (1 - jsd) * WEIGHT_JSD +
        (1 - ncp) * WEIGHT_NCP +
        (1 - dm) * WEIGHT_DM +
        (1 - record_loss / 100) * WEIGHT_RECORD_LOSS
    )

    results.append({
        'Method': name,
        'JSD': round(jsd, 4),
        'Avg_EC_Size': round(avg_ec, 2),
        'DM_Normalized': round(dm, 4),
        'NCP': round(ncp, 4),
        'Record_Loss_%': round(record_loss, 2),
        'k_Anon_%': k_anon_pct,
        'Utility_Score': round(utility_score, 2),
    })

df_results = pd.DataFrame(results)


# 6. PRINT SUMMARY TABLE

print("\n" + "=" * 110)
print("STATISTICAL PRIVACY-UTILITY TRADEOFF ANALYSIS")
print(f"Composite Weights: JSD={WEIGHT_JSD}% | NCP={WEIGHT_NCP}% | DM={WEIGHT_DM}% | RecLoss={WEIGHT_RECORD_LOSS}%")
print("=" * 110)
print(f"\n{'Method':<28} {'JSD':>8} {'Avg EC':>8} {'DM(norm)':>10} {'NCP':>8} "
      f"{'Rec Loss%':>10} {'k≥' + str(k_for_privacy) + '%':>8} {'Utility':>8}")
print("-" * 110)
for _, row in df_results.iterrows():
    print(f"{row['Method'].replace(chr(10), ' '):<28} {row['JSD']:>8.4f} {row['Avg_EC_Size']:>8.2f} "
          f"{row['DM_Normalized']:>10.4f} {row['NCP']:>8.4f} {row['Record_Loss_%']:>10.2f} "
          f"{row['k_Anon_%']:>8.2f} {row['Utility_Score']:>8.2f}")


# 7. LIGHT PASTEL COLOR PALETTE

PASTEL_GREEN = '#B5EAD7'
PASTEL_ORANGE = '#FFD8B1'
PASTEL_PURPLE = '#D4A5D0'
PASTEL_RED = '#FFB3B3'
PASTEL_YELLOW = '#FFFACD'
PASTEL_TEAL = '#B2DFDB'
PASTEL_PINK = '#F8C8DC'
PASTEL_BLUE = '#AEC6CF'


# 8. VISUALIZATION — 3 KEY CHARTS

# Only 3 charts that tell the complete story:
#   Chart 1: Privacy vs Utility scatter (THE tradeoff)
#   Chart 2: Utility score with privacy overlay (dual-axis comparison)
#   Chart 3: JSD bar chart (distribution preservation detail)

method_labels = [m.replace('\n', ' ') for m in df_results['Method']]
privacy_scores = df_results['k_Anon_%'].values
utility_scores = df_results['Utility_Score'].values

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Statistical Privacy-Utility Tradeoff Analysis\n'
             f'[Direct identifiers {direct_identifiers} excluded from JSD]',
             fontsize=16, fontweight='bold', y=1.02)


# CHART 1: PRIVACY vs UTILITY SCATTER PLOT

# Each defense is a point: X = privacy level, Y = utility score.
# Ideal position = TOP-RIGHT (high privacy AND high utility).
# Reference: Fung et al. (2010) multi-criteria evaluation framework

ax1 = axes[0]
scatter_colors = [PASTEL_GREEN, PASTEL_ORANGE, PASTEL_TEAL,
                  PASTEL_YELLOW, PASTEL_RED, PASTEL_PURPLE]

for i, label in enumerate(method_labels):
    ax1.scatter(privacy_scores[i], utility_scores[i], color=scatter_colors[i],
                s=300, edgecolors='black', linewidth=1.5, zorder=5)
    ax1.annotate(label, (privacy_scores[i], utility_scores[i]),
                 textcoords="offset points", xytext=(8, 8),
                 fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax1.set_xlabel(f'Privacy (% records satisfying k≥{k_for_privacy})', fontsize=12)
ax1.set_ylabel('Composite Utility Score (0–100)', fontsize=12)
ax1.set_title('Privacy vs. Utility Tradeoff\n(Sweeney, 2002 | Fung et al., 2010)',
              fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_xlim(-5, 110)

# ════════════════════════════════════════════
# CHART 2: COMPOSITE UTILITY + PRIVACY OVERLAY
# ════════════════════════════════════════════
# Green bars = utility (higher = better)
# Purple line = privacy (higher = better)

ax2 = axes[1]

bar_colors = [PASTEL_GREEN if u >= 75 else PASTEL_ORANGE if u >= 60 else PASTEL_RED
              for u in utility_scores]
bars2 = ax2.bar(method_labels, utility_scores, color=bar_colors,
                edgecolor='gray', linewidth=0.8, label='Utility Score', width=0.6)

for bar, val in zip(bars2, utility_scores):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

ax2_twin = ax2.twinx()
ax2_twin.plot(method_labels, privacy_scores, color='#8e44ad', marker='D',
              linewidth=2.5, markersize=10, label=f'k≥{k_for_privacy} Privacy %', zorder=5)
for i, val in enumerate(privacy_scores):
    ax2_twin.annotate(f'{val:.0f}%', (method_labels[i], val),
                      textcoords="offset points", xytext=(0, 10),
                      fontsize=9, fontweight='bold', color='#8e44ad', ha='center')

ax2.set_ylabel('Utility Score (0–100)', fontsize=11)
ax2.set_ylim(0, max(utility_scores) * 1.25)
ax2_twin.set_ylabel(f'Privacy (k≥{k_for_privacy} %)', fontsize=11, color='#8e44ad')
ax2_twin.set_ylim(0, 115)
ax2_twin.tick_params(axis='y', labelcolor='#8e44ad')
ax2.set_title('Composite Utility with Privacy Overlay\n(Bars = Utility | Line = Privacy)',
              fontsize=14, fontweight='bold')
ax2.set_xticklabels(method_labels, fontsize=8, rotation=20, ha='right')

lines2, labels2 = ax2.get_legend_handles_labels()
lines2t, labels2t = ax2_twin.get_legend_handles_labels()
ax2.legend(lines2 + lines2t, labels2 + labels2t, loc='upper left', fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.4)


# CHART 3: JSD BAR CHART

# Lower JSD = better (less information loss).
# Reference: Lin (1991)

ax3 = axes[2]
jsd_values = df_results['JSD'].values

jsd_colors = [PASTEL_GREEN if v <= 0.20 else PASTEL_ORANGE if v <= 0.40 else PASTEL_RED
              for v in jsd_values]

bars3 = ax3.barh(method_labels, jsd_values, color=jsd_colors, edgecolor='gray', linewidth=0.8)
for bar, val in zip(bars3, jsd_values):
    ax3.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
             f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

ax3.set_xlabel('Jensen-Shannon Divergence (lower = better utility)', fontsize=11)
ax3.set_title('Distribution Preservation (JSD)\n(Suppressed columns = 1.0)',
              fontsize=14, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
# Save to same folder as this script: defense/utility/utility_static/
plt.savefig("privacy_utility_tradeoff.png", dpi=300, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════
# 9. PER-COLUMN JSD HEATMAP
# ══════════════════════════════════════════════
# Shows JSD for EVERY column × EVERY defense combination.
# Dark red (1.0) = column completely destroyed.
# Light yellow (0.0) = column perfectly preserved.

jsd_matrix = []
for df_def in all_defense_dfs:
    jsd_per_col = compute_per_column_jsd(df_raw, df_def, jsd_columns)
    jsd_matrix.append([jsd_per_col[col] for col in jsd_columns])

df_jsd_heatmap = pd.DataFrame(jsd_matrix, index=method_labels, columns=jsd_columns)

fig2, ax_heat = plt.subplots(figsize=(max(14, len(jsd_columns) * 1.8), 7))
im = ax_heat.imshow(df_jsd_heatmap.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

ax_heat.set_xticks(np.arange(len(jsd_columns)))
ax_heat.set_yticks(np.arange(len(method_labels)))
ax_heat.set_xticklabels(jsd_columns, rotation=45, ha='right', fontsize=10)
ax_heat.set_yticklabels(method_labels, fontsize=10)

for i in range(len(method_labels)):
    for j in range(len(jsd_columns)):
        val = df_jsd_heatmap.values[i, j]
        text_color = 'white' if val > 0.4 else 'black'
        ax_heat.text(j, i, f'{val:.3f}', ha='center', va='center',
                     fontsize=9, color=text_color, fontweight='bold')

ax_heat.set_title('Per-Column Jensen-Shannon Divergence Heatmap\n'
                  f'(Direct identifiers {direct_identifiers} excluded | '
                  'Suppressed columns = 1.0)',
                  fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax_heat, label='JSD (0 = identical, 1 = max divergence)')
plt.tight_layout()
# Save to same folder as this script
plt.savefig("jsd_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════
# 10. SAVE RESULTS TO CSV
# ══════════════════════════════════════════════
df_export = df_results.copy()
df_export['Method'] = df_export['Method'].str.replace('\n', ' ')
# Save to same folder as this script
df_export.to_csv("privacy_utility_results.csv", index=False)
df_jsd_heatmap.to_csv("jsd_per_column_heatmap.csv")

print("\n" + "=" * 70)
print(" FILES SAVED (same folder as this script):")
print("   - privacy_utility_tradeoff.png  (3 key charts)")
print("   - jsd_heatmap.png               (per-column detail)")
print("   - privacy_utility_results.csv   (all metrics)")
print("   - jsd_per_column_heatmap.csv    (heatmap data)")
print("=" * 70)
print(f"\n Configuration:")
print(f"   Composite weights: JSD={WEIGHT_JSD}% | NCP={WEIGHT_NCP}% | DM={WEIGHT_DM}% | RecLoss={WEIGHT_RECORD_LOSS}%")
print(f"   Direct identifiers excluded: {direct_identifiers}")
print(f"   JSD columns: {jsd_columns}")
print(f"   Quasi-identifiers: {quasi_identifiers}")
print(f"   Privacy threshold: k≥{k_for_privacy}")