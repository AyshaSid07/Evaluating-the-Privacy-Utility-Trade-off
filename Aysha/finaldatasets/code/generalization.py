import pandas as pd

def generalization(df):
    protected = df.copy()
    
    # 1. LAI Generalization: Remove the Cell-ID
    #splitting '240-01-5012' and keeping only '240-01' (Country and Operator)
    if 'LAI' in protected.columns:
        protected["LAI"] = protected["LAI"].astype(str).str.split('-').str[:2].str.join('-')
    
    # 2. Timestamp Generalization: Reduce precision to the Date/Hour
    # stripping the minutes and seconds (e.g., '231027143005' becomes '231027')
    #  stops an attacker from 'syncing' events with high-precision external logs.
    if 'Timestamp' in protected.columns:
        protected["Timestamp"] = protected["Timestamp"].astype(str).str[:6]
        
    print("LAI and Timestamp Generalized.")
    return protected

# 1. Load the result from Layer 1 (Suppression)
df_layer_1 = pd.read_csv('suppression.csv')

# 2. Apply Generalization
df_layer_2 = generalization(df_layer_1)

# 3. Save to CSV
df_layer_2.to_csv('generalization.csv', index=False)