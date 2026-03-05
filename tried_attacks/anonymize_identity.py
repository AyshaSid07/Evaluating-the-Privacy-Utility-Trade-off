import pandas as pd
import hashlib

# Load your dataset
df = pd.read_csv('mendeley_data.csv')

#Secret Salt (private!)
SALT = "Vera_Security_2026"

def make_token(ip):
    # Combine IP + Salt and scramble it
    return hashlib.sha256((str(ip) + SALT).encode()).hexdigest()[:12]

#Create Token
df['User_Token'] = df['IP Address'].apply(make_token)

#Delete the original IP(GDPR Data Minimization)
df_hidden = df.drop(columns=['IP Address'])

#result
df_hidden.to_csv('data_phase1.csv', index=False)
print("Real IPs are gone")
