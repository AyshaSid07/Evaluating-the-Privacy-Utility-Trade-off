import pandas as pd

# Read the dataset CSV file
df = pd.read_csv("mendeley_data.csv")

# Define PII classification for each column
# This is based on how sensitive or identifying the data is
pii_classification = {
    "IP Address": "Direct personal identifier",
    "City": "Indirect identifier",
    "Region": "Indirect identifier",
    "Country": "Indirect identifier",
    "Postal Code": "Indirect identifier",
    "Latitude": "High-risk indirect identifier",
    "Longitude": "High-risk indirect identifier",
    "ISP": "Indirect identifier",
    "Organization": "Indirect identifier",
    "Autonomous System": "Indirect identifier",
    "Timezone": "Low-risk contextual data"
}

# Create a table showing each column and its PII classification
pii_table = pd.DataFrame({
    "Column": df.columns,
    "PII Classification": [pii_classification.get(col, "Unknown") for col in df.columns]
})

# Print the classification result
print(pii_table)

# Save the result to a CSV file
pii_table.to_csv("pii_classification_result.csv", index=False)
