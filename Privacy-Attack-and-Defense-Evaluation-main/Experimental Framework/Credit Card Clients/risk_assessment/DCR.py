import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

TARGET_COL = 'default payment'
RANDOM_STATE = 123
NUMERIC_COLS = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

def convert_val(val):
# Parse ARX intervals (e.g. "[20, 40[") into their numerical midpoint to avoid losing generalized data
    val = str(val).strip()
    if val == '*': return 0.0
    if val.startswith('[') and val.endswith('['):
        parts = val[1:-1].split(',')
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except:
            return 0.0
    try:
        return float(val)
    except:
        return 0.0

def dcr(df_orig_raw, df_anon_raw, numeric_cols):

    df_orig = df_orig_raw.copy()
    df_anon = df_anon_raw.copy()

    # 1. Ensure we only have common columns and drop indices/targets
    cols_to_use = [c for c in df_orig.columns if c not in ['Linkage_Index', TARGET_COL]]
    df_orig = df_orig[cols_to_use]
    df_anon = df_anon[cols_to_use]

    # Clean numeric columns and parse ARX interval generalizations
    for col in numeric_cols:
        df_orig[col] = df_orig[col].apply(convert_val)
        df_anon[col] = df_anon[col].apply(convert_val)

        # Use original training median instead of 0 for missing/suppressed values
        median_val = df_orig[col].median()

        if pd.isna(median_val):
            median_val = 0.0

        df_orig[col] = df_orig[col].fillna(median_val)
        df_anon[col] = df_anon[col].fillna(median_val)

    # 2. Create a ColumnTransformer (OneHot + Scaling [0,1])
    categorical_cols = [c for c in cols_to_use if c not in numeric_cols]

    for col in categorical_cols:
        df_orig[col] = df_orig[col].astype(str)
        df_anon[col] = df_anon[col].astype(str)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])

    # 3. Fit on the original data, transform both
    X_orig = preprocessor.fit_transform(df_orig)
    X_anon = preprocessor.transform(df_anon)

    # 4. Use kNN to efficiently find the shortest distance (DCR)
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean', n_jobs=-1)
    knn.fit(X_orig)
    
    distances, _ = knn.kneighbors(X_anon)
    min_distances = distances.flatten()

    return {
        'mean_distance':        float(np.mean(min_distances)),
        'median_distance':      float(np.median(min_distances)),
        
        'min_distance':         float(np.min(min_distances)),
        'max_distance':         float(np.max(min_distances)),
        'std_distance':         float(np.std(min_distances)),
        
        'percentile_1st':       float(np.percentile(min_distances, 1)),
        'percentile_5th':       float(np.percentile(min_distances, 5)),
        'percentile_10th':      float(np.percentile(min_distances, 10)),
        'percentile_25th':      float(np.percentile(min_distances, 25)),
        'percentile_75th':      float(np.percentile(min_distances, 75)),
        'percentile_90th':      float(np.percentile(min_distances, 90)),
        'percentile_95th':      float(np.percentile(min_distances, 95)),
        'percentile_99th':      float(np.percentile(min_distances, 99)),
        
        'exact_matches_%':      float((min_distances == 0).mean() * 100),
        'near_matches_1pct_%':  float((min_distances < 0.01).mean() * 100),
        'near_matches_5pct_%':  float((min_distances < 0.05).mean() * 100),
        'near_matches_10pct_%': float((min_distances < 0.10).mean() * 100)
    }

df_real = pd.read_csv('../datasets/credit_card_clients_binned.csv')
df_train_real, df_test_real = train_test_split(df_real, test_size=0.3, random_state=RANDOM_STATE)

datasets_to_test = {
    "No anonymization (Baseline)": df_test_real, 
    "ARX Credit Card Clients, k = 3": pd.read_csv('../datasets/ARX_credit_card_clients_k3.csv'),
    "ARX Credit Card Clients, k = 5": pd.read_csv('../datasets/ARX_credit_card_clients_k5.csv'),
    "ARX Credit Card Clients, k = 10": pd.read_csv('../datasets/ARX_credit_card_clients_k10.csv'),
    "ARX Credit Card Clients, k = 15": pd.read_csv('../datasets/ARX_credit_card_clients_k15.csv'),
    "ARX Credit Card Clients, k = 5, l = 2": pd.read_csv('../datasets/ARX_credit_card_clients_k5_l2.csv'),
    "ARX Credit Card Clients, k = 5, l = 4": pd.read_csv('../datasets/ARX_credit_card_clients_k5_l4.csv'),
    "ARX Credit Card Clients, k = 5, t = 0.3": pd.read_csv('../datasets/ARX_credit_card_clients_k5_t0.3.csv'),
    "ARX Credit Card Clients, k = 5, t = 0.15": pd.read_csv('../datasets/ARX_credit_card_clients_k5_t0.15.csv'),
    "DP Credit Card Clients, epsilon = 10.0": pd.read_csv('../datasets/DP_credit_card_clients_epsilon_10_0.csv'),
    "DP Credit Card Clients, epsilon = 5.0":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_5_0.csv'),
    "DP Credit Card Clients, epsilon = 3.0":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_3_0.csv'),
    "DP Credit Card Clients, epsilon = 1.0":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_1_0.csv'),
    "DP Credit Card Clients, epsilon = 0.5":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_0_5.csv'),
    "Combined Credit Card Clients, k = 3 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_credit_card_clients.csv'),
    "Combined Credit Card Clients, k = 3 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_credit_card_clients.csv'),
    "Combined Credit Card Clients, k = 3 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_credit_card_clients.csv'),
    "Combined Credit Card Clients, k = 5 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_credit_card_clients.csv'),
    "Combined Credit Card Clients, k = 5 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_credit_card_clients.csv'),
    "Combined Credit Card Clients, k = 5 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_credit_card_clients.csv'),
}

results = {}

for name, df in datasets_to_test.items():
    print(f"\n--- Evaluating DCR for: {name} ---")
    results[name] = dcr(df_orig_raw=df_train_real, df_anon_raw=df, numeric_cols=NUMERIC_COLS)

df_results = pd.DataFrame.from_dict(results, orient='index')

df_results.index.name = 'Anonymization_Configuration'

csv_filepath = "../results/dcr/dcr_results_credit_card_clients.csv" 

df_results.to_csv(csv_filepath)
print(df_results)