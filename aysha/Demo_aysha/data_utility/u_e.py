#Train - Raw dataset
#Test  - Re-identified dataset 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


# Dataset paths
datasets = {
    "Raw": "../defense/raw_dataset.csv",
    "Suppressed": "../defense/suppressed_dataset.csv",
    "Generalized": "../defense/generalized_dataset.csv",
    "Masked": "../defense/masked_dataset.csv"
}

target = "Network_Type"
features = ["MCC","MNC","PLMN","LAI","RAI"]


# TRAIN ON RAW DATASET

raw_df = pd.read_csv(datasets["Raw"])

for col in features + [target]:
    raw_df[col] = LabelEncoder().fit_transform(raw_df[col].astype(str))

X_train = raw_df[features]
y_train = raw_df[target]

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


#  TEST ON ALL DATASETS

results = []

for name, path in datasets.items():

    df = pd.read_csv(path)

    for col in features + [target]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X_test = df[features]
    y_test = df[target]

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    results.append([name, accuracy, precision, recall, f1])


#  RESULTS TABLE 

results_df = pd.DataFrame(
    results,
    columns=["Dataset","Accuracy","Precision","Recall","F1"]
)

print("\nUtility Evaluation Results")
print(results_df)


#  BAR GRAPH 

colors = ["#a6cee3","#b2df8a","#fb9a99","#fdbf6f"]

results_df.set_index("Dataset").plot(
    kind="bar",
    figsize=(8,5),
    color=colors
)

plt.ylabel("Score")
plt.title("ML Data Utility Comparison")
plt.ylim(0,1)

plt.tight_layout()
plt.savefig("u_e.png", dpi=300)

plt.show()