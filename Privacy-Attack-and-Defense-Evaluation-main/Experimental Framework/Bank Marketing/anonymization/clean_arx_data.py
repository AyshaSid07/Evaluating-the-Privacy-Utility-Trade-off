import os
import pandas as pd
# path of the folder containing anonymized datasets
dataset_dir = "../datasets/"
# list of quasi-identifiers used in the anonymized datasets
qis = ["age", "job", "marital", "education"]


for filename in os.listdir(dataset_dir):
    # select only ARX anonymized CSV files
    if filename.startswith("ARX_bank") and filename.endswith(".csv"):
        file_path = os.path.join(dataset_dir, filename)
        
        try:
            df = pd.read_csv(file_path, dtype=str)
            # find quasi-identifiers that exist in the dataset
            qis_in_df = [qi for qi in qis if qi in df.columns]
            # skip file if no quasi-identifiers are found
            if not qis_in_df:
                print(f"Found no matching QIs in {filename}, skipping...")
                continue

            original_len = len(df)
            #detect rows where all QIs are dully suppressed "*"
            suppressed_rows = (df[qis_in_df] == '*').all(axis=1)
            # remove fully suppressed rows
            df_clean = df[~suppressed_rows]
            # count removing and removed rows
            clean_len = len(df_clean)
            removed_count = original_len - clean_len
            # save cleaned dataset if rows were removed
            if removed_count > 0:
                df_clean.to_csv(file_path, index=False)
                print(f"Cleaned {filename}: Removed {removed_count} rows.")
            else:
                print(f"{filename} was already clean (0 rows removed).")
                
        except Exception as e:
            print(f"Something went wrong with {filename}: {e}")
