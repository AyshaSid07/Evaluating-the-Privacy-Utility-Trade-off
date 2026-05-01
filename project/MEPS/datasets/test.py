import pandas as pd

def check_attribute_balance(csv_path, columns_to_check):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    print(f"Total rows in dataset: {len(df)}")
    
    for col in columns_to_check:
        if col not in df.columns:
            print(f"\n[Warning] Column '{col}' not found in the dataset.")
            continue
            
        print(f"\n--- Balance check for: {col} ---")
        
        # Get counts and percentages (dropna=False ensures we also see if there are missing/NaN values)
        counts = df[col].value_counts(dropna=False)
        percentages = df[col].value_counts(normalize=True, dropna=False) * 100
        
        # Combine into a neat DataFrame for printing
        balance_df = pd.DataFrame({
            'Count': counts,
            'Percentage (%)': percentages.round(2)
        })
        
        # Sort by the category name/number so it's strictly ordered (e.g., 1, 2, 3)
        balance_df = balance_df.sort_index()
        
        print(balance_df)

if __name__ == "__main__":
    # Update this path if your file is located somewhere else
    DATASET_PATH = 'MEPS.csv'
    
    # Add any columns you want to check here
    ATTRIBUTES_TO_CHECK = ["PCS42","MCS42","K6SUM42", "MIDX","RTHLTH","ADHDADDX","CHBRON","DFSEE42","OHRTDX","COGLIM","HIBPDX","CANCERDX","HONRDC","ANGIDX","DIABDX","EMPHDX","ARTHTYPE","ASTHDX","POVCAT","MNHLTH","JTPAIN","WLKLIM","STRKDX","ACTLIM","ADSMOK42","CHOLDX","FTSTU","ACTDTY","SOCLIM","PHQ242","DFHEAR42","INSCOV","MARRY","CHDDX","SEX","ARTHDX","REGION","EMPST","PREGNT"]
    
    check_attribute_balance(DATASET_PATH, ATTRIBUTES_TO_CHECK)