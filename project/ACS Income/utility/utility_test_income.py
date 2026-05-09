import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

datasets_to_test = {
    "No anonymization (Baseline)": pd.read_csv('../datasets/folktables_income_RAW.csv'),
    "DP Income, epsilon = 1.0": pd.read_csv('../datasets/income_dp_eps_1.0.csv'),
    "DP Income, epsilon = 5.0": pd.read_csv('../datasets/income_dp_eps_5.0.csv'),
    "DP Income, epsilon = 10.0": pd.read_csv('../datasets/income_dp_eps_10.0.csv'),
    # "ARX ACS Income k = 10": pd.read_csv('../datasets/ARX_income_k=10.csv'),
    # "ARX ACS Income k = 20": pd.read_csv('../datasets/ARX_income_k=20.csv'),
    # "ARX ACS Income k = 50": pd.read_csv('../datasets/ARX_income_k=50.csv'),
    # "ARX Income k = 10": pd.read_csv('../datasets/ARX_income_k=10_l=2.csv'),
    # "ARX Income k = 10, l = 2": pd.read_csv('../datasets/ARX_income_k=10_l=2.csv'),
    # "ARX Income k = 10, t = 0.2": pd.read_csv('../datasets/ARX_income_k=10_t=0.2.csv'),
    # "ARX Income k = 10, t = 0.10": pd.read_csv('../datasets/ARX_income_k=10_t=0.1.csv'),
    # "ARX Income k = 10, t = 0.05": pd.read_csv('../datasets/ARX_income_k=10_t=0.05.csv'),
}

TARGET_COLUMN = 'PINCP' 

results = []

for name, df in datasets_to_test.items():
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    df = df[df[TARGET_COLUMN].astype(str) != '*'].copy()
    
    y = df[TARGET_COLUMN].astype(int).values
    X = df.drop(columns=[TARGET_COLUMN])
    
    X = X.astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ('scaler', StandardScaler(with_mean=False)), 
        ('classifier', LogisticRegression(max_iter=1000, n_jobs=-1)) # n_jobs=-1 för att använda alla processorkärnor
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results.append({
        'Dataset': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
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
    plt.xlabel('Anonymization Value', fontsize=10)
    plt.title('Utility Evaluation on ACS Income Data (Logistic Regression)', fontsize=12)

    plt.ylim(0, 1.1)
    
    plt.legend(loc='lower right', framealpha=1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("../plots/utility_income.png", dpi=300)
    plt.show()

plot_utility_results(results_df)