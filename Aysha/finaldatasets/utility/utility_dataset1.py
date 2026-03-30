import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sns.set(style="whitegrid", context="talk")


# STEP 2: LOAD DATA

df = pd.read_csv('telecom_dataset.csv')
print("Original Shape:", df.shape)



# STEP 3: PREPROCESSING


# ---- Remove PII ----
pii_cols = ['IMSI', 'MSISDN', 'IMEI', 'TMSI', 'LMSI', 'TLLI']
df = df.drop(columns=[c for c in pii_cols if c in df.columns], errors='ignore')


# ---- Handle Timestamp (ROBUST) ----
if 'Timestamp' in df.columns:
    df['Timestamp'] = df['Timestamp'].astype(str).str.strip()

    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'],
        errors='coerce',
        utc=True
    )

    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month

    df = df.drop(columns=['Timestamp'])


# ---- IP Address Handling ----
if 'IP_Address' in df.columns:
    df['IP_Address'] = df['IP_Address'].fillna("0.0.0.0")

    ip_parts = df['IP_Address'].str.split('.', expand=True)
    df[['IP_1','IP_2','IP_3','IP_4']] = ip_parts.astype(float)


# ---- LAC Extraction ----
if 'LAI' in df.columns:
    df['LAI'] = df['LAI'].fillna("000-00-0000")
    df['LAC'] = df['LAI'].str.split('-').str[2].astype(float)


# ---- Encode categorical ----
categorical_cols = ['Device_Type', 'RAI', 'PLMN']

for col in categorical_cols:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)


# ---- Encode Target ----
le = LabelEncoder()
df['Target'] = le.fit_transform(df['Network_Type'])


# ==========================================================
# STEP 4: FEATURE SELECTION
# ==========================================================
features = [col for col in df.columns if col not in ['Network_Type', 'Target', 'IP_Address', 'LAI']]
X = df[features]
y = df['Target']

# Keep numeric only
X = X.select_dtypes(include=[np.number])

print("Final Features:", X.columns)


# ==========================================================
# STEP 5: TRAIN-TEST SPLIT
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# STEP 6: DEFINE PIPELINE MODELS 

models = {
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=150, random_state=42))
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(n_estimators=150, random_state=42))
    ]),

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
}



# STEP 7: TRAIN + EVALUATE

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, average='weighted'),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted')
    })

results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)

print("\n=== UTILITY RESULTS ===")
print(results_df)


# ==========================================================
# STEP 8: CROSS-VALIDATION (NOW FIXED ✅)
# ==========================================================
cv_scores = {}

for name, model in models.items():
    score = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
    cv_scores[name] = score.mean()

cv_df = pd.DataFrame(list(cv_scores.items()), columns=["Model", "CV_F1"])


# ==========================================================
# STEP 9: FEATURE IMPORTANCE
# ==========================================================
rf_model = models["Random Forest"]
rf_model.fit(X_train, y_train)

# Access inner model from pipeline
rf = rf_model.named_steps["model"]

feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)


# ==========================================================
# STEP 10: ALL GRAPHS IN ONE FIGURE ⭐🔥
# ==========================================================
# Use light theme
sns.set_theme(style="whitegrid")

# Create figure with better spacing
fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)

# ------------------------------------------
# Plot 1: Model Performance
# ------------------------------------------
perf_data = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

sns.barplot(
    data=perf_data,
    x="Model",
    y="Score",
    hue="Metric",
    palette="pastel",
    ax=axes[0, 0]
)

axes[0, 0].set_title("Model Performance (Utility)", fontsize=14, weight='bold')
axes[0, 0].set_ylim(0, 1)
axes[0, 0].tick_params(axis='x', rotation=20)
axes[0, 0].legend(title="Metric", fontsize=9)


# ------------------------------------------
# Plot 2: Cross-validation
# ------------------------------------------
sns.barplot(
    data=cv_df,
    x="Model",
    y="CV_F1",
    palette="light:blue",
    ax=axes[0, 1]
)

axes[0, 1].set_title("Cross-Validation Stability", fontsize=14, weight='bold')
axes[0, 1].set_ylim(0, 1)
axes[0, 1].tick_params(axis='x', rotation=20)


# ------------------------------------------
# Plot 3: Feature Importance
# ------------------------------------------
sns.barplot(
    data=feat_imp.head(8),
    x="Importance",
    y="Feature",
    palette="light:green",
    ax=axes[1, 0]
)

axes[1, 0].set_title("Top Utility Features", fontsize=14, weight='bold')


# ------------------------------------------
# Plot 4: Correlation Heatmap
# ------------------------------------------
corr = pd.DataFrame(X, columns=X.columns).corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    annot=False,          # remove numbers → cleaner
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    ax=axes[1, 1]
)

axes[1, 1].set_title("Feature Correlation", fontsize=14, weight='bold')

# Rotate labels for clarity
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].tick_params(axis='y', rotation=0)


# ------------------------------------------
# Final adjustments
# ------------------------------------------
plt.suptitle("Data Utility Evaluation Dashboard", fontsize=18, weight='bold')

plt.savefig("final_thesis_figure_clean.png", dpi=300, bbox_inches='tight')
plt.show()