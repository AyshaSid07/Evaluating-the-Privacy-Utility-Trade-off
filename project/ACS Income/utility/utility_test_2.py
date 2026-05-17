import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

TARGET_COLUMN = 'PINCP' 
RANDOM_STATE = 123

df_real = pd.read_csv('../datasets/folktables_income_RAW.csv')
df_real = df_real[df_real[TARGET_COLUMN].astype(str) != '*'].copy()

FEATURE_COLS = [c for c in df_real.columns if c != TARGET_COLUMN]

y_real = df_real[TARGET_COLUMN].astype(int).values
X_real = df_real.drop(columns=[TARGET_COLUMN]).astype(str)

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.2, random_state=RANDOM_STATE
)

datasets_to_test = {
    "No anonymization (Baseline)": None,
    "ARX Income, k = 3": pd.read_csv('../datasets/ARX_acs_income_k3.csv'),
    "ARX Income, k = 5": pd.read_csv('../datasets/ARX_acs_income_k5.csv'),
    "ARX Income, k = 10": pd.read_csv('../datasets/ARX_acs_income_k10.csv'),
    "ARX Income, k = 15": pd.read_csv('../datasets/ARX_acs_income_k15.csv'),
    "ARX Income, k = 5, l = 3": pd.read_csv('../datasets/ARX_acs_income_k5_l3.csv'),
    "ARX Income, k = 5, l = 5": pd.read_csv('../datasets/ARX_acs_income_k5_l5.csv'),
    "ARX Income, k = 5, t = 0.3": pd.read_csv('../datasets/ARX_acs_income_k5_t0.3.csv'),
    "ARX Income, k = 5, t = 0.15": pd.read_csv('../datasets/ARX_acs_income_k5_t0.15.csv'),
    "DP Income, epsilon = 10.0": pd.read_csv('../datasets/DP_income_epsilon_10_0.csv'),
    "DP Income, epsilon = 5.0":  pd.read_csv('../datasets/DP_income_epsilon_5_0.csv'),
    "DP Income, epsilon = 1.0":  pd.read_csv('../datasets/DP_income_epsilon_1_0.csv'),
    "DP Income, epsilon = 0.5":  pd.read_csv('../datasets/DP_income_epsilon_0_5.csv'),
    "DP Income, epsilon = 0.1":  pd.read_csv('../datasets/DP_income_epsilon_0_1.csv'),
    "Combined Income, k = 3 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_income.csv'),
    "Combined Income, k = 3 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_income.csv'),
    "Combined Income, k = 3 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_income.csv'),
    "Combined Income, k = 5 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_income.csv'),
    "Combined Income, k = 5 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_income.csv'),
    "Combined Income, k = 5 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_income.csv'),
}

results = []

for name, df in datasets_to_test.items():
    print(f"Evaluating: {name}...")

    # Baseline: train directly on the real training split
    if df is None:
        X_train = X_train_real
        y_train = y_train_real
    else:
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        df = df[df[TARGET_COLUMN].astype(str) != '*'].copy()

        # Align column order to real data
        df = df[FEATURE_COLS + [TARGET_COLUMN]]

        # Use ALL rows for training — no re-splitting needed for synthetic data
        y_train = df[TARGET_COLUMN].astype(int).values
        X_train = df.drop(columns=[TARGET_COLUMN]).astype(str)

    model = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', LogisticRegression(max_iter=1000, n_jobs=-1, class_weight='balanced'))
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
print(results_df)
results_df.to_csv('../plots/utility/acs_income_utility_results.csv', index=False)

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
    plt.title('Utility Evaluation on ACS Income Data (Logistic Regression) ', fontsize=12)

    plt.ylim(0, 1.1)
    
    plt.legend(loc='lower right', framealpha=1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.savefig("../plots/utility_income_TSTR_ARX_k_only.png", dpi=300)
    # plt.savefig("../plots/utility_income_TSTR_ARX_k_l_t.png", dpi=300)
    plt.savefig("../plots/utility_income_TSTR_DP.png", dpi=300)
    plt.show()

# plot_utility_results(results_df)