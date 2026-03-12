#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.patches import Patch
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#step 1-load dataset

dataset_path = "../datasets/telecom_dataset.csv"
df = pd.read_csv(dataset_path)

print("Dataset loaded successfully")
print("\nDataset shape:")
print("\nFirst 5 rows:")
print(df.shape)
print(df.head())

#step 2-data exploration

print("\n Dataset Information ")
print(df.info())

print("\n Missing Values ")
print(df.isnull().sum())

print("\n Unique Values per Column ")
print(df.nunique())

#I analyzed the target variable distribution using value_counts() 
#observed that the dataset was balanced across the three classes (3G, 4G, and 5G). 
#This ensured that the machine learning model would not be biased toward a particular class
print("\n Target Variable Distribution")
print(df["Network_Type"].value_counts())

#step 3 -defining attributes

# Direct identifiers
direct_identifiers = ["IMSI", "MSISDN", "IMEI", "IP_Address"]

# Quasi identifiers
quasi_identifiers = ["LAI", "RAI", "MCC", "MNC", "PLMN"]

# Temporary identifiers
temporary_identifiers = ["TMSI", "LMSI", "TLLI"]

#Target variable
target = "Network_Type"

print("Direct Identifiers:", direct_identifiers)
print("Quasi Identifiers:", quasi_identifiers)
print("Temporary Identifiers:", temporary_identifiers)

#Step 4 -Feature preprocessing
#from exploratory data analysis we see IMSI,MSISDN,IMEI,IP_Address are direct identifiers the ML model may memorize instead of learning patterns so we roemve it
#Remove direct identifiers

direct_identifiers = ["IMSI", "MSISDN", "IMEI", "IP_Address"]

df_clean = df.drop(columns=direct_identifiers)

print("\nColumns after removing direct identifiers:")
print(df_clean.columns)

#check the remaining identifier count after removing direct identifiers
print(df_clean.shape)

#Step 5 - Encode categorical variables
#ML models dont understand categorical variables so we nee dto encode them using label encoding 
#LabelEncoder converts categorical string values (LAI, RAI, Network_Type) into numeric labels.
# Each unique category is assigned an integer(3G→0, 4G→1, 5G→2) so machine learning models can process the data.
# LAI and RAI were originally categorical location identifiers (strings).
# Using LabelEncoder, each unique LAI and RAI value in the dataset is converted
# into a unique integer label (LAI "240-01-1048" → 26, RAI "240-01-1048-64" → 35)
# so they can be used as numerical inputs for the ML model.
# Create encoder
encoder = LabelEncoder()

# Encode LAI
df_clean["LAI"] = encoder.fit_transform(df_clean["LAI"])

# Encode RAI
df_clean["RAI"] = encoder.fit_transform(df_clean["RAI"])

# Encode target variable
df_clean["Network_Type"] = encoder.fit_transform(df_clean["Network_Type"])

print("\nEncoded dataset preview:")
print(df_clean.head())
print(encoder.classes_)


#Step 5.1: Remove constant features

df_clean = df_clean.drop(columns=["MCC", "MNC", "PLMN"])

print("Remaining columns:")
print(df_clean.columns)

#Step 6 - Split dataset for training and testing
#separate features and target variable
X = df_clean.drop("Network_Type", axis=1)
#target variable
y = df_clean["Network_Type"]

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

#step 7 - Train machine larning model- Random forest classifier 
#Random Forest is an ensemble learning algorithm that builds multiple decision trees.
#model learns patterns in the training data to predict the target varaible Network_type based on the features (MCC, MNC, PLMN, TMSI, LMSI, TLLI, LAI, RAI)
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("Model training completed")

#step 8 - Make predictions
#predict the network type for the unseen data (predicted ->4G,actual ->4G)
y_pred = model.predict(X_test)

print("Predictions completed")

#step 9- evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nModel Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

#step 10 - visualize the model performance

# Model performance metrics
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [0.3495, 0.350173062576163, 0.3495, 0.3497072457474897]

# Light colors for each metric
colors = ["#A7C7E7", "#FFD8A8", "#B5EAD7", "#FFB7B2"]

plt.figure(figsize=(7,5))

bars = plt.bar(metrics, values, color=colors, edgecolor="black")

# Updated title with target column
plt.title("Model Performance on Raw Telecom Dataset (Target: Network_Type)")
plt.ylabel("Score")
plt.ylim(0,1)

# Display values above bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01,
             round(yval,4), ha='center')

# Legend showing color meaning
legend_elements = [
    Patch(facecolor="#A7C7E7", label="Accuracy"),
    Patch(facecolor="#FFD8A8", label="Precision"),
    Patch(facecolor="#B5EAD7", label="Recall"),
    Patch(facecolor="#FFB7B2", label="F1 Score")
]

plt.legend(handles=legend_elements, loc="upper right", title="Metrics")

plt.savefig("telecom_data_performance.png", dpi=300, bbox_inches="tight")

plt.show()
