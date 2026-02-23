import pandas as pd
import hashlib

df = pd.read_csv('datasets/mendeley_data.csv')

SALT = "Vera_Security_2026" 

def make_token(ip):
    # Combine IP + Salt and scramble it
    return hashlib.sha256((str(ip) + SALT).encode()).hexdigest()[:12]

#Create Token
df['User_Token'] = df['IP Address'].apply(make_token)

df_hidden = df.drop(columns=['IP Address'])

#result
df_hidden.to_csv('datasets/hashed_ip.csv', index=False)
