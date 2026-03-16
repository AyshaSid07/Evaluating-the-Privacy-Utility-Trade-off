# ============================================================
# TELECOM USER BEHAVIOUR ANOMALY DETECTION
# ============================================================

# Import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("telecom_dataset.csv")

print("Dataset preview:")
print(df.head())

print("\nDataset size:", df.shape)


# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())

df = df.drop_duplicates()

print("\nDataset size after removing duplicates:", df.shape)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# -------- IP behaviour --------

ip_count = df.groupby("IMSI")["IP_Address"].nunique()

df["IP_Subnet"] = df["IP_Address"].apply(lambda x: ".".join(x.split(".")[:3]))

subnet_count = df.groupby("IMSI")["IP_Subnet"].nunique()


# -------- Device behaviour --------

device_count = df.groupby("IMSI")["IMEI"].nunique()


# -------- Location behaviour --------

location_count = df.groupby("IMSI")["LAI"].nunique()


# -------- Network behaviour --------

network_count = df.groupby("IMSI")["Network_Type"].nunique()


# ============================================================
# 4. CREATE USER BEHAVIOUR PROFILE
# ============================================================

user_profile = pd.DataFrame({

    "unique_ip_count": ip_count,
    "subnet_variation": subnet_count,
    "device_count": device_count,
    "location_variation": location_count,
    "network_variation": network_count

}).reset_index()

print("\nUser behaviour profile:")
print(user_profile.head())


# ============================================================
# 5. VISUALIZE USER BEHAVIOUR DISTRIBUTIONS
# ============================================================

plt.figure()
plt.hist(user_profile["unique_ip_count"], bins=20)
plt.title("Distribution of Unique IP Usage per User")
plt.xlabel("Number of Unique IPs")
plt.ylabel("Number of Users")
plt.show()


plt.figure()
plt.hist(user_profile["device_count"], bins=10)
plt.title("Device Usage Behaviour")
plt.xlabel("Number of Devices per User")
plt.ylabel("Number of Users")
plt.show()


plt.figure()
plt.hist(user_profile["location_variation"], bins=10)
plt.title("Location Variation Among Users")
plt.xlabel("Number of Locations")
plt.ylabel("Number of Users")
plt.show()


plt.figure()
plt.hist(user_profile["network_variation"], bins=5)
plt.title("Network Type Variation")
plt.xlabel("Number of Network Types Used")
plt.ylabel("Number of Users")
plt.show()


# ============================================================
# 6. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

X = user_profile.drop("IMSI", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# 7. TRAIN ANOMALY DETECTION MODEL
# ============================================================

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)


# ============================================================
# 8. PREDICT ANOMALIES
# ============================================================

predictions = model.predict(X_scaled)

user_profile["anomaly"] = predictions

user_profile["anomaly"] = user_profile["anomaly"].map({1:0, -1:1})

print("\nAnomaly detection results:")
print(user_profile.head())


# ============================================================
# 9. VISUALIZE ANOMALY DETECTION RESULTS
# ============================================================

counts = user_profile["anomaly"].value_counts()

plt.figure()
plt.bar(["Normal Users", "Anomalous Users"], counts)
plt.title("Detected Telecom Behaviour Anomalies")
plt.ylabel("Number of Users")
plt.show()


# ============================================================
# 10. ANALYZE SUSPICIOUS USERS
# ============================================================

anomalies = user_profile[user_profile["anomaly"] == 1]

print("\nNumber of suspicious users:", len(anomalies))

print("\nExample suspicious users:")
print(anomalies.head())

suspicious_data = df[df["IMSI"].isin(anomalies["IMSI"])]

print("\nRaw telecom data for suspicious users:")
print(suspicious_data.head())


# ============================================================
# 11. USER BEHAVIOUR MAP (PCA VISUALIZATION)
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

user_profile["pca1"] = X_pca[:,0]
user_profile["pca2"] = X_pca[:,1]

plt.figure()

normal = user_profile[user_profile["anomaly"] == 0]
anomaly = user_profile[user_profile["anomaly"] == 1]

plt.scatter(normal["pca1"], normal["pca2"], label="Normal Users", alpha=0.6)
plt.scatter(anomaly["pca1"], anomaly["pca2"], label="Anomalous Users", marker="x")

plt.title("Telecom User Behaviour Map (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

plt.show()


# ============================================================
# 12. FEATURE CORRELATION ANALYSIS
# ============================================================

plt.figure()

corr = user_profile.drop(["IMSI","anomaly","pca1","pca2"], axis=1).corr()

sns.heatmap(corr, annot=True)

plt.title("Correlation Between User Behaviour Features")

plt.show()


# ============================================================
# 13. USER BEHAVIOUR PATTERN ANALYSIS
# ============================================================

plt.figure()

plt.scatter(user_profile["unique_ip_count"],
            user_profile["device_count"],
            c=user_profile["anomaly"])

plt.xlabel("Unique IP Count")
plt.ylabel("Device Count")
plt.title("IP Usage vs Device Usage Behaviour")

plt.show()


# ============================================================
# 14. NETWORK MOBILITY BEHAVIOUR
# ============================================================

plt.figure()

normal = user_profile[user_profile["anomaly"] == 0]
anomaly = user_profile[user_profile["anomaly"] == 1]

plt.scatter(normal["subnet_variation"],
            normal["location_variation"],
            label="Normal Users",
            alpha=0.6)

plt.scatter(anomaly["subnet_variation"],
            anomaly["location_variation"],
            label="Anomalous Users",
            marker="x",
            s=100)

plt.xlabel("Subnet Variation")
plt.ylabel("Location Variation")
plt.title("Network Mobility Behaviour vs Anomalies")
plt.legend()

plt.show()


# ============================================================
# 15. SAVE ANOMALY RESULT PLOT
# ============================================================



