# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import LabelEncoder


# # Dataset paths
# datasets = {
#     "Raw": "../defense/raw_dataset.csv",
#     "Suppressed": "../defense/suppressed_dataset.csv",
#     "Generalized": "../defense/generalized_dataset.csv",
#     "Masked": "../defense/masked_dataset.csv"
# }

# # Target and features
# target = "Network_Type"
# features = ["MCC", "MNC", "PLMN", "LAI", "RAI"]

# results = []


# for name, path in datasets.items():

#     # Load dataset
#     df = pd.read_csv(path)

#     # Encode categorical columns
#     for col in features + [target]:
#         df[col] = LabelEncoder().fit_transform(df[col].astype(str))

#     # Define features and target
#     X = df[features]
#     y = df[target]

#     # Train-test split (80/20)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # Train model
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_train)

#     # Predict
#     y_pred = model.predict(X_test)

#     # Evaluation metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average="weighted")
#     recall = recall_score(y_test, y_pred, average="weighted")
#     f1 = f1_score(y_test, y_pred, average="weighted")

#     # Store results
#     results.append([name, accuracy, precision, recall, f1])


# # Create results dataframe
# results_df = pd.DataFrame(
#     results,
#     columns=["Dataset", "Accuracy", "Precision", "Recall", "F1"]
# )

# print("\nUtility Evaluation Results")
# print(results_df)


# # --------- Bar Graph ---------

# colors = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f"]

# results_df.set_index("Dataset").plot(
#     kind="bar",
#     figsize=(8, 5),
#     color=colors
# )

# plt.ylabel("Score")
# plt.title("Data Utility Comparison")
# plt.ylim(0, 1)

# plt.tight_layout()
# plt.savefig("utility_comparison.png", dpi=300)
# plt.show()

# utility_evaluation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Datasets produced by your anonymization scripts
datasets = {
    "Raw": "../defense/raw_dataset.csv",
    "Masking": "../defense/masked_dataset.csv",
    "Suppression": "../defense/suppressed_dataset.csv",
    "Generalization": "../defense/generalized_dataset.csv",
    "Aggregation": "../defense/aggregated_dataset.csv"
}

results = {}

def evaluate_dataset(path):
    df = pd.read_csv(path)

    # Target: Network_Type
    y = df["Network_Type"].astype(str)
    y = LabelEncoder().fit_transform(y)

    # Features: ONLY location/operator identifiers
    X = df[["MCC", "MNC", "PLMN", "LAI", "RAI"]].astype(str)

    # Encode categorical features
    for col in X.columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # RandomForest classifier
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "F1": f1_score(y_test, pred, average="macro", zero_division=0)
    }

# Run evaluation
for name, path in datasets.items():
    results[name] = evaluate_dataset(path)

print("\nUtility Evaluation Results (Network_Type Classification)")
for name, m in results.items():
    print(f"{name:14s} Acc={m['Accuracy']:.4f} Prec={m['Precision']:.4f} "
          f"Rec={m['Recall']:.4f} F1={m['F1']:.4f}")

# --------- Bar graph ---------

metrics = ["Accuracy", "Precision", "Recall", "F1"]
methods = list(results.keys())
colors = ["#A8DADC", "#BDE0FE", "#FFC8DD", "#E2ECE9"]

plt.figure(figsize=(14, 8))
bar_width = 0.18
x = np.arange(len(methods))

for i, metric in enumerate(metrics):
    plt.bar(
        x + i * bar_width,
        [results[m][metric] for m in methods],
        width=bar_width,
        label=metric,
        color=colors[i],
        edgecolor="black",
        linewidth=0.8
    )

plt.xticks(x + bar_width * 1.5, methods, fontsize=13)
plt.ylabel("Score", fontsize=14)
plt.ylim(0, 1.05)
plt.title("Privacy–Utility Comparison (Network_Type Classification)", fontsize=18)
plt.legend(title="Metric", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("privacy_utility_networktype_classification.png", dpi=300)
plt.show()
