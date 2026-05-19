import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

TARGET_COLUMN = 'default payment'
RANDOM_STATE = 123
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
df_real = pd.read_csv('../datasets/credit_card_clients_binned.csv')

if 'Linkage_Index' in df_real.columns:
    df_real = df_real.drop(columns=['Linkage_Index'])

df_real = df_real[df_real[TARGET_COLUMN].astype(str) != '*'].copy()

if 'SEX' in df_real.columns:
    df_real['SEX'] = ["Female" if str(v) == '2' else "Male" for v in df_real['SEX']]

FEATURE_COLS = [c for c in df_real.columns if c != TARGET_COLUMN]

NUMERIC_COLS = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

CATEGORICAL_COLS = [c for c in FEATURE_COLS if c not in NUMERIC_COLS]

y_real = df_real[TARGET_COLUMN].astype(int).values
X_real = df_real.drop(columns=[TARGET_COLUMN])

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.2, random_state=RANDOM_STATE
)

datasets_to_test = {
    "No anonymization (Baseline)": None,  
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

results = []

for name, df in datasets_to_test.items():
    print(f"Evaluating: {name}...")
    if df is None:
        X_train = X_train_real.copy()
        y_train = y_train_real
    else:
        if 'Linkage_Index' in df.columns:
            df = df.set_index('Linkage_Index')
            surviving_train_indices = X_train_real.index.intersection(df.index)
            df_train = df.loc[surviving_train_indices].copy()
        else:
            df = df.reset_index(drop=True)
            df_train = df.copy()
        
        df_train = df_train[df_train[TARGET_COLUMN].astype(str) != '*'].copy()
        if 'SEX' in df_train.columns:
            df_train['SEX'] = ["Female" if str(v) == '2' else "Male" for v in df_train['SEX']]

        df_train = df_train[FEATURE_COLS + [TARGET_COLUMN]]

        y_train = df_train[TARGET_COLUMN].astype(int).values
        X_train = df_train.drop(columns=[TARGET_COLUMN]).copy()

    for col in NUMERIC_COLS:
        X_train[col] = X_train[col].apply(convert_val)

        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = 0.0

        X_train[col] = X_train[col].fillna(median_val)

    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype(str)

    X_test = X_test_real.copy()

    for col in NUMERIC_COLS:
        X_test[col] = X_test[col].apply(convert_val)

        median_val = X_test[col].median()
        if pd.isna(median_val):
            median_val = 0.0

        X_test[col] = X_test[col].fillna(median_val)

    for col in CATEGORICAL_COLS:
        X_test[col] = X_test[col].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS),
            ('num', 'passthrough', NUMERIC_COLS)
        ])

    model = Pipeline([
        ('preprocessor', preprocessor), 
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=123, n_jobs=-1))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        'Dataset': name,
        'Accuracy': accuracy_score(y_test_real, y_pred),
        'F1-Score': f1_score(y_test_real, y_pred, zero_division=0),
        'Precision': precision_score(y_test_real, y_pred, zero_division=0),
        'Recall': recall_score(y_test_real, y_pred, zero_division=0)
    })

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("../results/utility/credit_card_clients_utility_results.csv", index=False)