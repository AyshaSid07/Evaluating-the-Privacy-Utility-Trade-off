import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

TARGET_COLUMN = 'UTILIZATION'
RANDOM_STATE = 123

df_real = pd.read_csv('../datasets/MEPS.csv')

if 'index' in df_real.columns:
    df_real = df_real.drop(columns=['index'])

df_real = df_real[df_real[TARGET_COLUMN].astype(str) != '*'].copy()
FEATURE_COLS = [c for c in df_real.columns if c != TARGET_COLUMN]

NUMERIC_COLS = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PHQ242']

CATEGORICAL_COLS = [c for c in FEATURE_COLS if c not in NUMERIC_COLS]

y_real = df_real[TARGET_COLUMN].astype(int).values
X_real = df_real.drop(columns=[TARGET_COLUMN])

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.2, random_state=RANDOM_STATE
)
datasets_to_test = {
    "No anonymization (Baseline)": None,  # signals: use X_train_real directly
    # "ARX MEPS, k = 5": pd.read_csv('../datasets/ARX_meps_k=5.csv'),
    # "ARX MEPS, k = 5, l = 2": pd.read_csv('../datasets/ARX_meps_k=5_l=2.csv'),
    # "ARX MEPS, k = 5, l = 3": pd.read_csv('../datasets/ARX_meps_k=5_l=3.csv'),
    # "ARX MEPS, k = 5, t = 0.2": pd.read_csv('../datasets/ARX_meps_k=5_t=0.2.csv'),
    # "ARX MEPS, k = 5, t = 0.1": pd.read_csv('../datasets/ARX_meps_k=5_t=0.1.csv'),
    "ARX MEPS k = 5": pd.read_csv('../datasets/ARX_meps_k=5.csv'),
    "ARX MEPS k = 10": pd.read_csv('../datasets/ARX_meps_k=10.csv'),
    "ARX MEPS k = 20": pd.read_csv('../datasets/ARX_meps_k=20.csv'),
    # "DP MEPS, epsilon = 10.0": pd.read_csv('../datasets/meps_dp_epsilon_10_0.csv'),
    # "DP MEPS, epsilon = 5.0":  pd.read_csv('../datasets/meps_dp_epsilon_5_0.csv'),
    # "DP MEPS, epsilon = 3.0":  pd.read_csv('../datasets/meps_dp_epsilon_3_0.csv'),
    # "DP MEPS, epsilon = 1.0":  pd.read_csv('../datasets/meps_dp_epsilon_1_0.csv'),
    # "DP MEPS, epsilon = 0.5":  pd.read_csv('../datasets/meps_dp_epsilon_0_5.csv'),
}

results = []

for name, df in datasets_to_test.items():
    print(f"Evaluating: {name}...")
    if df is None:
        X_train = X_train_real
        y_train = y_train_real
    else:
        if 'index' in df.columns:
            df = df.set_index('index')
        else:
            df.index = df_real.index 
        
        surviving_train_indices = X_train_real.index.intersection(df.index)
        
        df_train = df.loc[surviving_train_indices].copy()
        
        df_train = df_train[df_train[TARGET_COLUMN].astype(str) != '*'].copy()
        
        if 'SEX' in df_train.columns:
            df_train['SEX'] = ["Female" if str(v) == '2' else "Male" for v in df_train['SEX']]

        df_train = df_train[FEATURE_COLS + [TARGET_COLUMN]]

        y_train = df_train[TARGET_COLUMN].astype(int).values
        X_train = df_train.drop(columns=[TARGET_COLUMN]).copy()

    for col in NUMERIC_COLS:
        X_train[col] = pd.to_numeric(X_train[col].astype(str).str.replace('*', 'NaN', regex=False), errors='coerce').fillna(0)
    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype(str)
        
    if df is None: 
        for col in NUMERIC_COLS:
            X_test_real[col] = pd.to_numeric(X_test_real[col], errors='coerce').fillna(0)
        for col in CATEGORICAL_COLS:
            X_test_real[col] = X_test_real[col].astype(str)

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
    y_pred = model.predict(X_test_real)

    results.append({
        'Dataset': name,
        'Accuracy': accuracy_score(y_test_real, y_pred),
        'F1-Score': f1_score(y_test_real, y_pred, zero_division=0),
        'Precision': precision_score(y_test_real, y_pred, zero_division=0),
        'Recall': recall_score(y_test_real, y_pred, zero_division=0)
    })

results_df = pd.DataFrame(results)

def plot_utility_results(results_df):
    plt.figure(figsize=(14, 7)) 
    
    methods = results_df['Dataset'].tolist()
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    
    x = np.arange(len(methods))
    width = 0.2  
    
    colors = ['#0072B2', '#E69F00', '#56B4E9', '#009E73']
    
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width 
        values = results_df[metric].tolist()
        
        bars = plt.bar(x + offset, values, width, label=metric, color=colors[i], edgecolor='black')
        
        for j, val in enumerate(values):
            plt.text(x[j] + offset, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=9)
            
    plt.xticks(x, methods, rotation=15, ha='right', fontsize=10)    
    plt.ylabel('Score (0.0 - 1.0)', fontsize=10)
    plt.xlabel('Anonymization Setting', fontsize=10)
    plt.title('Utility Evaluation on MEPS Data (Random Forest)', fontsize=12)

    plt.ylim(0, 1.1)
    
    plt.legend(loc='lower right', framealpha=1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("../plots/utility_meps_TSTR_ARX_k_only.png", dpi=300)
    # plt.savefig("../plots/utility_meps_TSTR_ARX_k_l_t.png", dpi=300)
    # plt.savefig("../plots/utility_meps_TSTR_DP.png", dpi=300)
    plt.show()

plot_utility_results(results_df)