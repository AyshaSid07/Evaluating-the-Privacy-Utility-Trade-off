import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def evaluate_dataset(path):

    print("\nEvaluating dataset:", path)

    df = pd.read_csv(path)

    # Target
    y = df["Network_Type"]

    # Features
    X = df.drop("Network_Type", axis=1)

    # Encode categorical columns
    encoder = LabelEncoder()

    for col in X.columns:
        X[col] = encoder.fit_transform(X[col].astype(str))

    y = encoder.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", round(accuracy,4))

    return accuracy


#paths

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(CURRENT_DIR)

DEFENSE_PATH = os.path.join(BASE_DIR, "defense")

print("Dataset directory:", DEFENSE_PATH)


#datasets

raw_accuracy = evaluate_dataset(
    os.path.join(DEFENSE_PATH, "raw_dataset.csv")
)

suppressed_accuracy = evaluate_dataset(
    os.path.join(DEFENSE_PATH, "suppressed_dataset.csv")
)

generalized_accuracy = evaluate_dataset(
    os.path.join(DEFENSE_PATH, "generalized_dataset.csv")
)

masked_accuracy = evaluate_dataset(
    os.path.join(DEFENSE_PATH, "masked_dataset.csv")
)


# results

print("\nUtility Comparison")
print("----------------------------")
print("Raw Dataset Accuracy:", raw_accuracy)
print("Suppressed Dataset Accuracy:", suppressed_accuracy)
print("Generalized Dataset Accuracy:", generalized_accuracy)
print("Masked Dataset Accuracy:", masked_accuracy)


#plot

datasets = ["Raw", "Suppressed", "Generalized", "Masked"]

scores = [
    raw_accuracy,
    suppressed_accuracy,
    generalized_accuracy,
    masked_accuracy
]

colors = ["#4CAF50", "#2196F3", "#FFC107", "#FF5722"]

plt.figure(figsize=(7,5))

bars = plt.bar(datasets, scores, color=colors)

plt.title("Data Utility Comparison", fontsize=14)
plt.ylabel("Accuracy")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.005,
        f"{height:.3f}",
        ha="center"
    )

plt.ylim(0,0.5)

plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()

plt.savefig("utility_comparison.png")

print("\nGraph saved as utility_comparison.png")

plt.show()