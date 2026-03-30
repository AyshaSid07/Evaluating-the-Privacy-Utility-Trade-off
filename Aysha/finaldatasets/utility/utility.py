import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ========== 1. Load Data ==========
df = pd.read_csv('telecom_dataset.csv')

# ========== 2. Feature Engineering ==========
# IP splitting if present
if 'IP_Address' in df.columns:
    df[['IP_Octet_1','IP_Octet_2','IP_Octet_3','IP_Octet_4']] = df['IP_Address'].str.split('.',expand=True).astype(int)

# LAI/RAI splitting if present
if 'LAI' in df.columns:
    df['LAI_Area_Code'] = df['LAI'].str.split('-').str[2].astype(int)
if 'RAI' in df.columns:
    df['RAI_Routing_Code'] = df['RAI'].str.split('-').str[3].astype(int)
    
# PLMN split into country & operator
if 'PLMN' in df.columns:
    df['PLMN_Country_Code'] = df['PLMN'].astype(str).str[:3].astype(int)
    df['PLMN_Operator_Code'] = df['PLMN'].astype(str).str[3:].astype(int)

# IMSI country code split
if 'IMSI' in df.columns:
    df['IMSI_Country'] = df['IMSI'].astype(str).str[:3].astype(int)
if 'MCC' in df.columns:
    df['MCC'] = pd.to_numeric(df['MCC'], errors='coerce').astype(int)
if 'MNC' in df.columns:
    df['MNC'] = pd.to_numeric(df['MNC'], errors='coerce').astype(int)

# Choose feature columns (adapt if some not used!)
feature_cols = [
    'MCC', 'MNC', 'LAI_Area_Code', 'RAI_Routing_Code',
    'PLMN_Country_Code', 'PLMN_Operator_Code', 'IMSI_Country',
    'IP_Octet_1', 'IP_Octet_2', 'IP_Octet_3', 'IP_Octet_4'
]
# Drop columns missing in data
features = [f for f in feature_cols if f in df.columns]
X = df[features].values

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['Network_Type'])

# ========== 3. Train/Test Split & Scaling ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ========== 4. Fit Utility Model ==========
rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)

# ========== 5. Calculate Metrics ==========
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
cm = confusion_matrix(y_test, y_pred)
classes = encoder.classes_

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# ========== 6. Utility Dashboard Plot ==========
fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=100)

# Feature Importance
ax1 = axes[0, 0]
top_10 = feature_importance.head(10)
ax1.barh(range(len(top_10)), top_10['Importance'].values, color='#66B2FF', edgecolor='black', alpha=0.8)
ax1.set_yticks(range(len(top_10)))
ax1.set_yticklabels(top_10['Feature'].values, fontsize=9)
ax1.set_xlabel('Importance', fontweight='bold')
ax1.set_title('Feature Importance (Top 10)', fontweight='bold', fontsize=11)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Performance Metrics
ax2 = axes[0, 1]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [accuracy, precision, recall, f1, roc_auc]
colors_metrics = ['#66B2FF', '#99FF99', '#FFD700', '#FF9999', '#FFB6C1']
bars = ax2.bar(metrics_names, metrics_values, color=colors_metrics, edgecolor='black', alpha=0.8)
ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('Performance Metrics', fontweight='bold', fontsize=11)
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, metrics_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

# Test Set Class Distribution
ax3 = axes[0, 2]
class_counts = [np.sum(y_test == i) for i in range(len(classes))]
colors_class = ['#FF9999', '#66B2FF', '#99FF99']
bars = ax3.bar(classes, class_counts, color=colors_class, edgecolor='black', alpha=0.8)
ax3.set_ylabel('Count', fontweight='bold')
ax3.set_title('Test Set Class Distribution', fontweight='bold', fontsize=11)
ax3.grid(axis='y', alpha=0.3)
for bar, count in zip(bars, class_counts):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height, f'{int(count)}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Per-Class Accuracy
ax4 = axes[1, 0]
per_class_acc = []
for i in range(len(classes)):
    mask = y_test == i
    if np.sum(mask) > 0:
        acc = accuracy_score(y_test[mask], y_pred[mask])
        per_class_acc.append(acc)
    else:
        per_class_acc.append(0)
bars = ax4.bar(classes, per_class_acc, color=colors_class, edgecolor='black', alpha=0.8)
ax4.set_ylabel('Accuracy', fontweight='bold')
ax4.set_title('Per-Class Accuracy', fontweight='bold', fontsize=11)
ax4.set_ylim([0, 1])
ax4.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, per_class_acc):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height, f'{acc:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Confusion Matrix
ax5 = axes[1, 1]
im = ax5.imshow(cm, cmap='Blues', aspect='auto')
ax5.set_xticks(range(len(classes)))
ax5.set_yticks(range(len(classes)))
ax5.set_xticklabels(classes, fontsize=10)
ax5.set_yticklabels(classes, fontsize=10)
ax5.set_xlabel('Predicted Label', fontweight='bold')
ax5.set_ylabel('True Label', fontweight='bold')
ax5.set_title('Confusion Matrix', fontweight='bold', fontsize=11)
for i in range(len(classes)):
    for j in range(len(classes)):
        text = ax5.text(j, i, cm[i, j], ha="center", va="center", color="black", fontweight='bold', fontsize=11)

plt.colorbar(im, ax=ax5, label='Count')

# Summary Box
ax6 = axes[1, 2]
ax6.axis('off')
summary_text = f"""
BASELINE MODEL SUMMARY

Total Samples: {len(df)}
Training: {len(X_train)} (80%)
Testing: {len(X_test)} (20%)

Features Used: {len(features)}
Classes: {', '.join(classes)}

KEY METRICS:
Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1-Score:  {f1:.4f}
ROC-AUC:   {roc_auc:.4f}

INTERPRETATION:
- Model correctly predicts
  network type {accuracy*100:.2f}% of time
- Balanced performance across
  all metrics
- Ready for anonymization
  comparison in Step 6
"""
ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Random Forest Classifier - Baseline Model Analysis\nNetwork Type Classification (3G, 4G, 5G)',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('baseline_model_analysis.png', dpi=100, bbox_inches='tight')
print("Plot saved: baseline_model_analysis.png")
plt.show()