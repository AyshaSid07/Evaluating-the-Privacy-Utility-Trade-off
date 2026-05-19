import os
import pandas as pd

dataset_dir = "../datasets/"

qis = ["AGEP", "SCHL", "SEX", "MAR", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY", "ESR"]


for filename in os.listdir(dataset_dir):
    
    if filename.startswith("ARX_acs") and filename.endswith(".csv"):
        file_path = os.path.join(dataset_dir, filename)
        
        try:
            df = pd.read_csv(file_path, dtype=str)
            
            qis_in_df = [qi for qi in qis if qi in df.columns]
            
            if not qis_in_df:
                print(f"Found no matching QIs in {filename}, skipping...")
                continue

            original_len = len(df)
            
            suppressed_rows = (df[qis_in_df] == '*').all(axis=1)
            
            df_clean = df[~suppressed_rows]
            
            clean_len = len(df_clean)
            removed_count = original_len - clean_len
            
            if removed_count > 0:
                df_clean.to_csv(file_path, index=False)
                print(f"Cleaned {filename}: Removed {removed_count} rows.")
            else:
                print(f"{filename} was already clean (0 rows removed).")
                
        except Exception as e:
            print(f"Something went wrong with {filename}: {e}")
