import pandas as pd
from faker import Faker
import numpy as np

fake = Faker()

def generate_external_knowledge(raw_file="trial.csv", fraction=0.5):
    df_raw = pd.read_csv(raw_file)
    
    # Attacker knows 50% of the population
    subset = df_raw.sample(frac=fraction, random_state=99)
    
    external_data = []
    
    for _, row in subset.iterrows():
        # Attacker extracts TAC from IMEI (first 8 digits)
        known_tac = str(row['IMEI'])[:8]
        
        # Attacker knows Location Area (LAI) roughly
        known_lai = row['LAI']
        
        external_data.append({
            'Real_Name': fake.name(),
            'MSISDN': row['MSISDN'],      # The Key Link
            'Known_TAC': known_tac,       # Device Info
            'Known_LAI': known_lai,       # Location Info
            'Known_Network': row['Network_Type']
        })
        
    df_ext = pd.DataFrame(external_data)
    df_ext.to_csv("telecom_external.csv", index=False)
    print(f"Generated {len(df_ext)} rows of External Knowledge.")
    return df_ext

if __name__ == "__main__":
    generate_external_knowledge()