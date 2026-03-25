import pandas as pd

# Step 1: Load the dataset
data = pd.read_csv("newtelecom_dataset.csv")

# Step 2: Count total rows
total_rows = len(data)

# Step 3: Find unique subscribers (using IMSI, MSISDN,IMEI)
unique_users = data[["IMSI", "MSISDN","IMEI"]].drop_duplicates()
num_unique_users = len(unique_users)

# Step 4: Calculate percentage
percentage = (num_unique_users / total_rows) * 100

# Step 5: Count total columns
total_columns = len(data.columns)

# Step 6: Print results
print("Total records:", total_rows)
print("Unique subscribers:", num_unique_users)
print("Unique subscribers %:", round(percentage, 2), "%")
print("Total attributes:", total_columns)