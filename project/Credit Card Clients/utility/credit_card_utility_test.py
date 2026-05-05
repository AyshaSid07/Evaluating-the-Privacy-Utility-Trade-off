import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

datasets_to_test = {
    "No anonymization (Baseline)": pd.read_csv('../datasets/credit-card-clients.csv'),
    "ARX Credit Card Clients, k = 5": pd.read_csv('../datasets/ARX_credit_card_clients_k=5.csv'),
    "ARX Credit Card Clients, k = 5, l = 2": pd.read_csv('../datasets/ARX_credit_card_clients_k=5_l=2.csv'),
    "ARX Credit Card Clients, k = 5, l = 3": pd.read_csv('../datasets/ARX_credit_card_clients_k=5_l=3.csv'),
    "ARX Credit Card Clients, k = 5, t = 0.20": pd.read_csv('../datasets/ARX_credit_card_clients_k=5_t=0.2.csv'),
    "ARX Credit Card Clients, k = 5, t = 0.10": pd.read_csv('../datasets/ARX_credit_card_clients_k=5_t=0.1.csv'),
    # "ARX Credit Card Clients k = 5": pd.read_csv('../datasets/ARX_credit_card_clients_k=5.csv'),
    # "ARX Credit Card Clients k = 10": pd.read_csv('../datasets/ARX_credit_card_clients_k=10.csv'),
    # "ARX Credit Card Clients k = 20": pd.read_csv('../datasets/ARX_credit_card_clients_k=20.csv'),
}

TARGET_COLUMN = 'default payment'

results = []

for name, df in datasets_to_test.items():
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    df.drop_duplicates(keep='first', inplace=True)
    
    df = df[df[TARGET_COLUMN].astype(str) != '*'].copy()

    if 'SEX' in df.columns:
        df['SEX'] = ["Female" if str(v) == '2' else "Male" for v in df['SEX']]

    y = df[TARGET_COLUMN].astype(int).values
    X = df.drop(columns=[TARGET_COLUMN])
    
    X = X.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) 

    model = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=123, n_jobs=-1))
    ])
    
    model.fit(X_train, y_train)
    y_predicts = model.predict(X_test)
    
    results.append({
        'Dataset': name,
        'Accuracy': accuracy_score(y_test, y_predicts),
        'F1-Score': f1_score(y_test, y_predicts, zero_division=0),
        'Precision': precision_score(y_test, y_predicts, zero_division=0),
        'Recall': recall_score(y_test, y_predicts, zero_division=0)
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
    
    plt.title('Utility Evaluation on Credit Card Clients Data (Random Forest)', fontsize=12)

    plt.ylim(0, 1.1)
    plt.legend(loc='lower right', framealpha=1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig("../plots/utility_credit_card_clients.png", dpi=300)
    plt.show()

plot_utility_results(results_df)