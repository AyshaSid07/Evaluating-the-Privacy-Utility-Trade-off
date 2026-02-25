import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# file paths
raw_path = "data/mendeley_data.csv"
v1_path = "anonymize/anonymized_v1.csv"
v2_path = "anonymize/anonymized_v2.csv"
v3_path = "anonymize/anonymized_v3.csv"

# load datasets
raw = pd.read_csv(raw_path)
v1 = pd.read_csv(v1_path)
v2 = pd.read_csv(v2_path)
v3 = pd.read_csv(v3_path)

print("Datasets loaded")

target = "Organization"

# remove rare classes
min_samples = 3
valid_orgs = raw[target].value_counts()
valid_orgs = valid_orgs[valid_orgs >= min_samples].index

raw = raw[raw[target].isin(valid_orgs)]
v1 = v1[v1[target].isin(valid_orgs)]
v2 = v2[v2[target].isin(valid_orgs)]
v3 = v3[v3[target].isin(valid_orgs)]

# reset index
raw = raw.reset_index(drop=True)
v1 = v1.reset_index(drop=True)
v2 = v2.reset_index(drop=True)
v3 = v3.reset_index(drop=True)

# preprocessing
def preprocess(df):
    X = df.drop(columns=[target])
    y = df[target]
    X = pd.get_dummies(X)
    return X, y

# label encoding
all_labels = pd.concat([raw[target], v1[target], v2[target], v3[target]])
le = LabelEncoder()
le.fit(all_labels)

X_raw, y_raw = preprocess(raw)
y_raw_encoded = le.transform(y_raw)

# split raw dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_raw,
    y_raw_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_raw_encoded
)

# train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# baseline accuracy
y_pred_raw = model.predict(X_test)
acc_raw = accuracy_score(y_test, y_pred_raw)

print("\nBaseline Accuracy:", round(acc_raw, 4))

# evaluate anonymized datasets
def evaluate(data):
    X_anon, y_anon = preprocess(data)
    X_anon = X_anon.reindex(columns=X_raw.columns, fill_value=0)
    y_anon_encoded = le.transform(y_anon)

    X_anon_test = X_anon.iloc[X_test.index]
    y_anon_test = y_anon_encoded[X_test.index]

    y_pred = model.predict(X_anon_test)
    return accuracy_score(y_anon_test, y_pred)

acc_v1 = evaluate(v1)
acc_v2 = evaluate(v2)
acc_v3 = evaluate(v3)

print("\nAnonymized V1 Accuracy:", round(acc_v1, 4))
print("Anonymized V2 Accuracy:", round(acc_v2, 4))
print("Anonymized V3 Accuracy:", round(acc_v3, 4))

# compute retention
ret_v1 = (acc_v1 / acc_raw) * 100
ret_v2 = (acc_v2 / acc_raw) * 100
ret_v3 = (acc_v3 / acc_raw) * 100

print("\nUtility Retention V1:", round(ret_v1, 2), "%")
print("Utility Retention V2:", round(ret_v2, 2), "%")
print("Utility Retention V3:", round(ret_v3, 2), "%")

# accuracy plot (saved)
labels = ["Raw", "V1", "V2", "V3"]
accuracies = [acc_raw, acc_v1, acc_v2, acc_v3]

plt.figure()
bars = plt.bar(labels, accuracies)

for i in range(len(bars)):
    height = bars[i].get_height()
    plt.text(i, height, round(height, 4), ha='center', va='bottom')

plt.xlabel("Dataset")
plt.ylabel("Accuracy")
plt.title("Model Accuracy: Raw vs Anonymized")
plt.savefig("accuracy_plot.png")
plt.close()

# retention plot (saved)
retentions = [100, ret_v1, ret_v2, ret_v3]

plt.figure()
bars = plt.bar(labels, retentions)

for i in range(len(bars)):
    height = bars[i].get_height()
    plt.text(i, height, round(height, 2), ha='center', va='bottom')

plt.xlabel("Dataset")
plt.ylabel("Utility Retention (%)")
plt.title("Utility Retention Comparison")
plt.savefig("retention_plot.png")
plt.close()

print("\nPlots saved as accuracy_plot.png and retention_plot.png")
