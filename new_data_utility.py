import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split   # Splits data into train/test sets
from sklearn.ensemble import RandomForestClassifier    # Machine learning model
from sklearn.preprocessing import OrdinalEncoder       # Encodes text into numbers
from sklearn.metrics import accuracy_score             # Measures model accuracy
from pandas.api.types import is_numeric_dtype          # Helps check if a column is numeric

# Load the raw dataset
raw = pd.read_csv("data/mendeley_data.csv")

# Load the anonymized versions
v1  = pd.read_csv("anonymize/anonymized_v1.csv")
v2  = pd.read_csv("anonymize/anonymized_v2.csv")
v3  = pd.read_csv("anonymize/anonymized_v3.csv")

#deletes the IP address from all the dataset (as its sensitive for prediction)
for df in [raw, v1, v2, v3]:
    if "IP Address" in df.columns:
        df.drop(columns=["IP Address"], inplace=True)

# Remove any rows from raw data if the region is missing (reason - the model cannot learn from missing labels)
raw = raw.dropna(subset=["Region"])
v1  = v1.dropna(subset=["Region"])
v2  = v2.dropna(subset=["Region"])
v3  = v3.dropna(subset=["Region"])

# Create an encoder that converts text like city names to  numbers
# It also changes new/unseen values by assigning them -1
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

# Function to encode all non-numeric columns
def encode(df, fit=False):
    df = df.copy()
#find all the columns contain text
    non_numeric_cols = [c for c in df.columns if not is_numeric_dtype(df[c])]  # Find text columns
    
    if fit:
        # Fit the encoder on the raw dataset and transform it
        df[non_numeric_cols] = encoder.fit_transform(df[non_numeric_cols].astype(str))
    else:
        # Use the already-fitted encoder to transform anonymized datasets
        df[non_numeric_cols] = encoder.transform(df[non_numeric_cols].astype(str))
    
    return df

# Encode the raw dataset (this trains the encoder)
raw_enc = encode(raw, fit=True) #it learn the categories from raw dataset

# Encode the anonymized datasets using the same encoder
v1_enc = encode(v1)
v2_enc = encode(v2)
v3_enc = encode(v3)

# Choose the target column we want to predict
target = "Region"

# Separate features (X) and target (y)
X = raw_enc.drop(columns=[target])
y = raw_enc[target]

# Split the raw data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

# Function to evaluate accuracy on any dataset
def evaluate(df, label):
    X = df.drop(columns=[target])   # Features
    y = df[target]                  # True labels
    acc = accuracy_score(y, model.predict(X))  # Compare predictions to true labels
    print(f"{label} accuracy:", acc)
    return acc

# Evaluate accuracy on raw test data
acc_raw = accuracy_score(y_test, model.predict(X_test))
print("Raw accuracy:", acc_raw)

# Evaluate accuracy on anonymized datasets
acc_v1 = evaluate(v1_enc, "V1")
acc_v2 = evaluate(v2_enc, "V2")
acc_v3 = evaluate(v3_enc, "V3")

# Create a bar chart to compare accuracies
labels = ["Raw", "V1", "V2", "V3"]
scores = [acc_raw, acc_v1, acc_v2, acc_v3]

plt.figure(figsize=(10, 6))
plt.bar(labels, scores, color=["#4CAF50", "#FF9800", "#03A9F4", "#9C27B0"])
plt.title("Utility Comparison")              # Title of the plot
plt.ylabel("Accuracy")
plt.ylim(0, 1)                                # Accuracy ranges from 0 to 1
plt.tight_layout()

# Save the plot as an image file 
plt.savefig("utility_plot.png", dpi=300)
print("Plot saved as utility_plot.png")

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, scores, color=["#4CAF50", "#FF9800", "#03A9F4", "#9C27B0"])

# Add accuracy labels on top of each bar
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{score:.2f}", ha='center', fontsize=12)

plt.title("Utility Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("utility_plot_labeled.png", dpi=300)
print("Plot saved as utility_plot_labeled.png")
