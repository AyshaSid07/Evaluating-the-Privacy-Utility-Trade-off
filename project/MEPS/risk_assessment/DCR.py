import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

TARGET_COL = 'UTILIZATION'
RANDOM_STATE = 123
NUMERIC_COLS = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PHQ242']

def dcr(df_orig_raw, df_anon_raw, numeric_cols):
    """
    Quantifies the risk for Distance-Based Record Linkage.
    Measures the Euclidean distance from each anonymized/synthetic row to its 
    nearest neighbor in the original data (DCR).
    """
    df_orig = df_orig_raw.copy()
    df_anon = df_anon_raw.copy()

    # 1. Ensure we only have common columns and drop indices/targets
    cols_to_use = [c for c in df_orig.columns if c not in ['index', TARGET_COL]]
    df_orig = df_orig[cols_to_use]
    df_anon = df_anon[cols_to_use]

    # Clean numeric columns (replace '*' with NaN, then fill with 0)
    for col in numeric_cols:
        df_orig[col] = pd.to_numeric(df_orig[col].astype(str).str.replace('*', 'NaN', regex=False), errors='coerce').fillna(0)
        df_anon[col] = pd.to_numeric(df_anon[col].astype(str).str.replace('*', 'NaN', regex=False), errors='coerce').fillna(0)

    # 2. Create a ColumnTransformer (OneHot + Scaling [0,1])
    # MinMax-scaling is extremely important so 'AGE' (0-100) doesn't dominate over 'K6SUM42'
    categorical_cols = [c for c in cols_to_use if c not in numeric_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])

    # 3. Fit on the original data, transform both
    X_orig = preprocessor.fit_transform(df_orig)
    X_anon = preprocessor.transform(df_anon)

    # 4. Use kNN to efficiently find the shortest distance (DCR)
    # DCR uses euclidan distance
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean', n_jobs=-1)
    knn.fit(X_orig)
    
    # distances will be an array with the shortest distance for each evaluated row
    distances, _ = knn.kneighbors(X_anon)
    min_distances = distances.flatten()

    return {
        'mean_distance':        float(np.mean(min_distances)),
        'median_distance':      float(np.median(min_distances)),
        
        'min_distance':         float(np.min(min_distances)),
        'max_distance':         float(np.max(min_distances)),
        'std_distance':         float(np.std(min_distances)), # Hur utspritt är bruset?
        
        'percentile_1st':       float(np.percentile(min_distances, 1)),  # De 1% mest sårbara
        'percentile_5th':       float(np.percentile(min_distances, 5)),
        'percentile_10th':      float(np.percentile(min_distances, 10)),
        'percentile_25th':      float(np.percentile(min_distances, 25)),
        'percentile_75th':      float(np.percentile(min_distances, 75)),
        'percentile_90th':      float(np.percentile(min_distances, 90)),
        'percentile_95th':      float(np.percentile(min_distances, 95)),
        'percentile_99th':      float(np.percentile(min_distances, 99)), # De 1% bäst skyddade
        
        'exact_matches_%':      float((min_distances == 0).mean() * 100),
        'near_matches_1pct_%':  float((min_distances < 0.01).mean() * 100),
        'near_matches_5pct_%':  float((min_distances < 0.05).mean() * 100),
        'near_matches_10pct_%': float((min_distances < 0.10).mean() * 100)
    }

def plot_results(results_dict):
    plt.figure(figsize=(10, 6)) 
    methods = list(results_dict.keys())
    rates = list(results_dict.values())
    
    for i in range(len(methods)):
        plt.bar(methods[i], rates[i], color=plt.cm.Set3(i), edgecolor='black')
        plt.text(methods[i], rates[i] + 0.02, f"{rates[i]:.4f}", ha='center', va='bottom', fontsize=10)
        
    plt.xticks(rotation=15, ha='right', fontsize=10)    
    plt.ylabel('Mean DCR Value', fontsize=10)
    plt.xlabel('Anonymization Value', fontsize=10)

    plt.title('Average Distance to Closest Record (Mean DCR) \nAcross different epsilon values of Differential Privacy on the MEPS Dataset', fontsize=13, pad=15)
    # plt.title('Average Distance to Closest Record (Mean DCR) \nAcross different k-values of k-Anonymity on the MEPS Dataset', fontsize=10)
    # plt.title('Average Distance to Closest Record (Mean DCR) \nAcross different k-, l- and t-values of k-Anonymity, l-Diversity and t-Closeness on the MEPS Dataset', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.ylim(1.5, 2.6)
    plt.tight_layout()
    plt.savefig("../plots/dcr_mean_distance_meps_DP.png", dpi=300)
    # plt.savefig("../plots/dcr_mean_distance_meps_ARX_k_only.png", dpi=300)
    # plt.savefig("../plots/dcr_mean_distance_meps_ARX_k_l_t.png", dpi=300)
    plt.show()


df_real = pd.read_csv('../datasets/MEPS.csv')
df_train_real, df_test_real = train_test_split(df_real, test_size=0.3, random_state=RANDOM_STATE)
datasets_to_test = {
    "No anonymization (Baseline)": df_test_real,  # attacker has real data — upper bound
    "ARX MEPS, k = 3": pd.read_csv('../datasets/ARX_meps_k3.csv'),
    "ARX MEPS, k = 5": pd.read_csv('../datasets/ARX_meps_k5.csv'),
    "ARX MEPS, k = 10": pd.read_csv('../datasets/ARX_meps_k10.csv'),
    "ARX MEPS, k = 15": pd.read_csv('../datasets/ARX_meps_k15.csv'),
    "ARX MEPS, k = 5, l = 2": pd.read_csv('../datasets/ARX_meps_k5_l2.csv'),
    "ARX MEPS, k = 5, l = 3": pd.read_csv('../datasets/ARX_meps_k5_l3.csv'),
    "ARX MEPS, k = 5, t = 0.3": pd.read_csv('../datasets/ARX_meps_k5_t0.3.csv'),
    "ARX MEPS, k = 5, t = 0.15": pd.read_csv('../datasets/ARX_meps_k5_t0.15.csv'),
    "DP MEPS, epsilon = 10.0": pd.read_csv('../datasets/DP_meps_epsilon_10_0.csv'),
    "DP MEPS, epsilon = 5.0":  pd.read_csv('../datasets/DP_meps_epsilon_5_0.csv'),
    "DP MEPS, epsilon = 3.0":  pd.read_csv('../datasets/DP_meps_epsilon_3_0.csv'),
    "DP MEPS, epsilon = 1.0":  pd.read_csv('../datasets/DP_meps_epsilon_1_0.csv'),
    "DP MEPS, epsilon = 0.5":  pd.read_csv('../datasets/DP_meps_epsilon_0_5.csv'),
    "Combined MEPS, k = 3 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_meps.csv'),
    "Combined MEPS, k = 3 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_meps.csv'),
    "Combined MEPS, k = 3 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_meps.csv'),
    "Combined MEPS, k = 5 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_meps.csv'),
    "Combined MEPS, k = 5 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_meps.csv'),
    "Combined MEPS, k = 5 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_meps.csv'),
}

results = {}

for name, df in datasets_to_test.items():
    print(f"\n--- Evaluating DCR for: {name} ---")
    results[name] = dcr(df_orig_raw=df_train_real, df_anon_raw=df, numeric_cols=['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PHQ242'])
    # print(f"Mean DCR: {results[name]['mean_distance']:.4f}")
    # print(f"Median DCR: {results[name]['median_distance']:.4f}")
    # print(f"Exact Matches (%): {results[name]['exact_matches_%']:.2f}%")
    # print(f"Near Matches <1% DCR (%): {results[name]['near_matches_1pct_%']:.2f}%")

# plot_DP_mean_dcr({k: v['mean_distance'] for k, v in results.items()})
df_results = pd.DataFrame.from_dict(results, orient='index')

# 2. Döp om index-kolumnen så det ser snyggt ut i CSV:n
df_results.index.name = 'Anonymization_Configuration'

# 3. Definiera var du vill spara filen (skapa en mapp som heter "results" om du inte har det)
csv_filepath = "../plots/dcr/dcr_results_meps.csv" 

# 4. Spara till CSV
df_results.to_csv(csv_filepath)
print(df_results)
# plot_results({k: v['mean_distance'] for k, v in results.items()})