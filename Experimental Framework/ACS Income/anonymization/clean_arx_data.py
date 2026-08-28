import os
import pandas as pd

dataset_dir = "../datasets/"

datasets_config = {
    "ARX_acs_income": ["AGEP", "SCHL", "SEX", "COW", "MAR", "RELP", "WKHP", "POBP", "OCCP"],
}

metrics_log = []

for filename in os.listdir(dataset_dir):
    if not filename.endswith(".csv") or filename.endswith("_cleaned.csv"):
        continue

    for prefix, qis in datasets_config.items():
        if filename.startswith(prefix):
            file_path = os.path.join(dataset_dir, filename)
            
            try:
                df = pd.read_csv(file_path, dtype=str)
                
                qis_in_df = [qi for qi in qis if qi in df.columns]
                if not qis_in_df:
                    print(f"Skipping {filename}: No matching QIs found.")
                    continue
                    
                original_len = len(df)
                
                suppressed_mask = (df[qis_in_df] == '*').all(axis=1)
                supp_count = suppressed_mask.sum()
                supp_rate = supp_count / original_len if original_len > 0 else 0
                
                hit_cap = "Yes" if 0.0495 <= supp_rate <= 0.0505 else "No"
                
                df_clean = df[~suppressed_mask]
                clean_len = len(df_clean)
                
                clean_filename = filename.replace(".csv", "_cleaned.csv")
                clean_file_path = os.path.join(dataset_dir, clean_filename)
                df_clean.to_csv(clean_file_path, index=False)
                
                unique_vals = {qi: df_clean[qi].nunique() for qi in qis_in_df}
                
                metrics_log.append({
                    "Dataset": filename,
                    "Original Rows": original_len,
                    "Suppressed Rows": supp_count,
                    "Suppression Rate (%)": round(supp_rate * 100, 2),
                    "Hit 5% Cap?": hit_cap,
                    "Cleaned Rows": clean_len,
                    "Distinct Vals per QI": unique_vals
                })
                
                print(f"{filename} -> {clean_filename} | Removed {supp_count} rows ({supp_rate:.2%})")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if metrics_log:
    df_metrics = pd.DataFrame(metrics_log)
    report_path = os.path.join(dataset_dir, "ARX_Metrics_Report.csv")
    df_metrics.to_csv(report_path, index=False)
    
    print("ARX METRICS REPORT GENERATED")
    print(f"Saved complete metrics log to: {report_path}")