# ==========================================================
# STEP 1: IMPORT LIBRARIES
# ==========================================================
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

sns.set_theme(style="whitegrid")


# ==========================================================
# STEP 2: LOAD DATA
# ==========================================================
df = pd.read_csv('../finaldatasets/mendeley_dataset.csv')
print("Dataset loaded successfully ")
print("Shape:", df.shape)


# ==========================================================
# STEP 3: PREPROCESSING
# ==========================================================

# Remove direct identifier
if 'IP_Address' in df.columns:
    df = df.drop(columns=['IP_Address'])

# Replace "N/A"
df = df.replace("N/A", np.nan)

# Convert latitude/longitude to numeric
for col in ['Latitude', 'Longitude']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# Encode categorical columns
categorical_cols = [
    'City', 'Region', 'Country', 'Postal Code',
    'Timezone', 'ISP', 'Organization', 'Autonomous System'
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)
        df = pd.get_dummies(df, columns=[col], drop_first=True)


# ==========================================================
# STEP 4: TARGET CREATION
# ==========================================================
target_column = 'Country' if 'Country' in df.columns else df.columns[0]

le = LabelEncoder()
df['Target'] = le.fit_transform(df[target_column])


# ==========================================================
# STEP 5: FEATURE SELECTION
# ==========================================================
X = df.drop(columns=['Target'])
y = df['Target']

# Keep only numeric
X = X.select_dtypes(include=[np.number])

# Save feature names (IMPORTANT FIX)
feature_names = X.columns

print("Number of features:", len(feature_names))


# ==========================================================
# STEP 6: TRAIN-TEST SPLIT
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==========================================================
# STEP 7: PIPELINE MODELS (NO NaN ISSUES)
# ==========================================================
models = {
    "RF": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),

    "GB": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),

    "LR": Pipeline([
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


# ==========================================================
# STEP 8: TRAIN & EVALUATE
# ==========================================================
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

print("\n=== FINAL RESULTS ===")
print(results_df)


# ==========================================================
# STEP 9: CROSS-VALIDATION
# ==========================================================
cv_results = []

for name, model in models.items():
    score = cross_val_score(model, X, y, cv=5, scoring='f1_weighted').mean()
    cv_results.append({"Model": name, "CV_F1": score})

cv_df = pd.DataFrame(cv_results)


# ==========================================================
# STEP 10: FEATURE IMPORTANCE
# ==========================================================
rf_model = models["RF"]
rf_model.fit(X_train, y_train)

rf = rf_model.named_steps["model"]

feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)


# ==========================================================
# STEP 11: FINAL CLEAN PLOT
# ==========================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)

# Performance
perf = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=perf, x="Model", y="Score", hue="Metric",
            palette="pastel", ax=axes[0, 0])
axes[0, 0].set_title("Model Performance")
axes[0, 0].set_ylim(0, 1)

# CV
sns.barplot(data=cv_df, x="Model", y="CV_F1",
            palette="light:blue", ax=axes[0, 1])
axes[0, 1].set_title("Cross Validation")

# Feature importance
sns.barplot(data=feat_imp.head(8),
            x="Importance", y="Feature",
            palette="light:green", ax=axes[1, 0])
axes[1, 0].set_title("Top Features")

# Correlation
corr = pd.DataFrame(X, columns=feature_names).corr()
sns.heatmap(corr, cmap="coolwarm", center=0,
            cbar_kws={"shrink": 0.8}, ax=axes[1, 1])
axes[1, 1].set_title("Correlation")

plt.suptitle("Dataset 2 Utility Evaluation", fontsize=18, weight='bold')

plt.savefig("dataset2_final.png", dpi=300, bbox_inches='tight')
plt.show()