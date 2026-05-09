import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

TARGET_COL = 'default payment'
RANDOM_STATE = 123
NUMERIC_COLS = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

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
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
    knn.fit(X_orig)
    
    # distances will be an array with the shortest distance for each evaluated row
    distances, _ = knn.kneighbors(X_anon)
    min_distances = distances.flatten()

    return {
        'mean_distance':    float(min_distances.mean()),
        'median_distance':  float(np.median(min_distances)),
        'exact_matches_%':  float((min_distances == 0).mean() * 100),   # % perfect copies
        'near_matches_1pct_%': float((min_distances < 0.01).mean() * 100) # % very close (threshold)
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

    # plt.title('Average Distance to Closest Record (Mean DCR) \nAcross different epsilon values of Differential Privacy on the Credit Card Clients Dataset', fontsize=13, pad=15)
    plt.title('Average Distance to Closest Record (Mean DCR) \nAcross different k-values of k-Anonymity on the Credit Card Clients Dataset', fontsize=10)
    # plt.title('Average Distance to Closest Record (Mean DCR) \nAcross different k-, l- and t-values of k-Anonymity, l-Diversity and t-Closeness on the Credit Card Clients Dataset', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    # plt.ylim(0, 1)
    plt.tight_layout()
    # plt.savefig("../plots/dcr_mean_distance_credit_card_clients_DP.png", dpi=300)
    plt.savefig("../plots/dcr_mean_distance_credit_card_clients_ARX_k_only.png", dpi=300)
    # plt.savefig("../plots/dcr_mean_distance_credit_card_clients_ARX_k_l_t.png", dpi=300)
    plt.show()


df_real = pd.read_csv('../datasets/credit-card-clients.csv')
df_train_real, df_test_real = train_test_split(df_real, test_size=0.3, random_state=RANDOM_STATE)

datasets_to_test = {
    "No anonymization (Baseline)": df_test_real,  # signals: use X_train_real directly
    # "ARX Credit Card Clients, k = 5": pd.read_csv('../datasets/ARX_credit_card_clients_k=5.csv'),
    # "ARX Credit Card Clients, k = 5, l = 2": pd.read_csv('../datasets/ARX_credit_card_clients_k=5_l=2.csv'),
    # "ARX Credit Card Clients, k = 5, l = 3": pd.read_csv('../datasets/ARX_credit_card_clients_k=5_l=3.csv'),
    # "ARX Credit Card Clients, k = 5, t = 0.20": pd.read_csv('../datasets/ARX_credit_card_clients_k=5_t=0.2.csv'),
    # "ARX Credit Card Clients, k = 5, t = 0.10": pd.read_csv('../datasets/ARX_credit_card_clients_k=5_t=0.1.csv'),
    "ARX Credit Card Clients k = 5": pd.read_csv('../datasets/ARX_credit_card_clients_k=5.csv'),
    "ARX Credit Card Clients k = 10": pd.read_csv('../datasets/ARX_credit_card_clients_k=10.csv'),
    "ARX Credit Card Clients k = 20": pd.read_csv('../datasets/ARX_credit_card_clients_k=20.csv'),
    # "DP Credit Card Clients, epsilon = 10.0": pd.read_csv('../datasets/credit_card_clients_dp_epsilon_10_0.csv'),
    # "DP Credit Card Clients, epsilon = 5.0":  pd.read_csv('../datasets/credit_card_clients_dp_epsilon_5_0.csv'),
    # "DP Credit Card Clients, epsilon = 3.0":  pd.read_csv('../datasets/credit_card_clients_dp_epsilon_3_0.csv'),
    # "DP Credit Card Clients, epsilon = 1.0":  pd.read_csv('../datasets/credit_card_clients_dp_epsilon_1_0.csv'),
    # "DP Credit Card Clients, epsilon = 0.5":  pd.read_csv('../datasets/credit_card_clients_dp_epsilon_0_5.csv'),
}

results = {}

for name, df in datasets_to_test.items():
    print(f"\n--- Evaluating DCR for: {name} ---")
    results[name] = dcr(df_orig_raw=df_train_real, df_anon_raw=df, numeric_cols=NUMERIC_COLS)
    print(f"Mean DCR: {results[name]['mean_distance']:.4f}")
    print(f"Median DCR: {results[name]['median_distance']:.4f}")
    print(f"Exact Matches (%): {results[name]['exact_matches_%']:.2f}%")
    print(f"Near Matches <1% DCR (%): {results[name]['near_matches_1pct_%']:.2f}%")

plot_results({k: v['mean_distance'] for k, v in results.items()})