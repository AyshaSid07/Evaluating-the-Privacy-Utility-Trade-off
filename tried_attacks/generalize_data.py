import pandas as pd

# Load your dataset
df = pd.read_csv('mendeley_data.csv')

# Func mask the IP
# Eg: 192.168.1.45 -> 192.168.1.xxx
def mask_ip(ip):
    parts = str(ip).split('.')
    if len(parts) == 4:
        # keep the first part (the network) 
        # hide the last part (the specific device)
        return f"{parts[0]}.xxx.xxx.xxx"
    return ip

# Apply the mask to the 'IP Address' column
if 'IP Address' in df.columns:
    df['IP_Masked'] = df['IP Address'].apply(mask_ip)
    # Drop the original dangerous column
    df_safe = df.drop(columns=['IP Address'])
    # Save the safe version
    df_safe.to_csv('ip_only_safe.csv', index=False)
    print("Success! Only IP addresses were processed.")
else:
    print("Error: Could not find 'IP Address' column.")
