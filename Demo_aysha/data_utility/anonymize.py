import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_dataset(path):

    print("\nEvaluating dataset:", path)

# Load dataset
    df = pd.read_csv(path)

    print(df.columns)
# Remove direct identifiers if they exist
    identifiers = ["IMSI", "MSISDN", "IMEI", "IP_Address"]
    df = df.drop(columns=[c for c in identifiers if c in df.columns], errors="ignore")

# Encode categorical features
    encoder = LabelEncoder()

    if "LAI" in df.columns:
        df["LAI"] = encoder.fit_transform(df["LAI"])

    if "RAI" in df.columns:
        df["RAI"] = encoder.fit_transform(df["RAI"])

    if "Network_Type" in df.columns:
        df["Network_Type"] = encoder.fit_transform(df["Network_Type"])

    # Remove constant columns
    constant_cols = ["MCC", "MNC", "PLMN"]
    df = df.drop(columns=[c for c in constant_cols if c in df.columns], errors="ignore")

    # Separate features and target
    X = df.drop("Network_Type", axis=1)
    y = df["Network_Type"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return accuracy, precision, recall, f1

masked_results = evaluate_dataset("../defense/masked_dataset.csv")
suppressed_results = evaluate_dataset("../defense/suppressed_dataset.csv")
swapped_results = evaluate_dataset("../defense/swapped_dataset.csv")

results = pd.DataFrame({
    "Dataset": ["Masked", "Suppressed", "Swapped"],
    "Accuracy": [masked_results[0], suppressed_results[0], swapped_results[0]],
    "Precision": [masked_results[1], suppressed_results[1], swapped_results[1]],
    "Recall": [masked_results[2], suppressed_results[2], swapped_results[2]],
    "F1 Score": [masked_results[3], suppressed_results[3], swapped_results[3]]
})

print("\nUtility Comparison:")
print(results)

import matplotlib.pyplot as plt
import numpy as np

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
datasets = results["Dataset"]

x = np.arange(len(datasets))
width = 0.18

colors = ["#6fa8dc", "#f6b26b", "#93c47d", "#e06666"]

plt.figure(figsize=(9,6))

for i, metric in enumerate(metrics):
    bars = plt.bar(
        x + i*width,
        results[metric],
        width,
        label=metric,
        color=colors[i],
        edgecolor="black"
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.001,
            f"{height:.3f}",
            ha='center',
            fontsize=9
        )

plt.xlabel("Anonymization Method", fontsize=12)
plt.ylabel("Score", fontsize=12)

plt.title(
    "Utility Evaluation of Anonymized Datasets\n(Random Forest Prediction of Network_Type)",
    fontsize=13
)

plt.xticks(x + width*1.5, datasets)

# IMPORTANT improvement
plt.ylim(0.30, 0.37)

plt.grid(axis='y', linestyle="--", alpha=0.5)

plt.legend(title="Metrics")

plt.tight_layout()

plt.savefig("anonymized_dataset_performance.png", dpi=300)

plt.show()

