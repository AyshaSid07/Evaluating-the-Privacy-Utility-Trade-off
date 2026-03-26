import pandas as pd
import numpy as np
import ipaddress

# Load dataset
df = pd.read_csv("mendeley_dataset.csv")

print("Original shape:", df.shape)

# Step 1: Remove duplicates 
df = df.drop_duplicates()
df = df.drop_duplicates(subset=['IP Address'])

# Step 2: Convert 'N/A' → NaN 
df.replace('N/A', np.nan, inplace=True)

# Step 3: Handle missing values 
# Fill non-critical columns
fill_cols = ['City', 'Region', 'Postal Code', 
             'Organization', 'Autonomous System']

for col in fill_cols:
    df[col] = df[col].fillna('Unknown')

# Drop rows with critical missing values
df = df.dropna(subset=['Latitude', 'Longitude', 'Country'])

# Step 4: Standardize text 
for col in ['City', 'Region', 'Country']:
    df[col] = df[col].str.strip().str.title()

# Step 5: Convert IP Address to numeric (optional but useful)
df['IP_int'] = df['IP Address'].apply(lambda x: int(ipaddress.ip_address(x)))

# Step 6: Final check 
print("Cleaned shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Step 7: Save cleaned dataset 
df.to_csv("cleaned_mendeley_dataset.csv", index=False)

print("\nCleaned dataset saved as 'cleaned_mendeley_dataset.csv'")