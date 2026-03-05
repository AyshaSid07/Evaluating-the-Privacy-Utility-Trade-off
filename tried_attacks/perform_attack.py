import pandas as pd

#Load the anonymized data
df = pd.read_csv('ip_only_safe.csv')

#Count occurrences of each attribute combination
# This finds how many people share the same City and Organization
counts = df.groupby(['City', 'Organization']).size().reset_index(name='count')

#Identify unique individuals (K=1)
# These are the records that have been successfully re-identified
reidentified = counts[counts['count'] == 1]

#Display the results
print("RE-IDENTIFICATION ATTACK RESULTS")
if not reidentified.empty:
    print(f"Successfully re-identified {len(reidentified)} unique individuals.")
    print(reidentified[['City', 'Organization']])
else:
    print("Attack failed: No unique individuals found.")

