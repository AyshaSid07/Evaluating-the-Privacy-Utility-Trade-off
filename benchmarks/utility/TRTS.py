from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
TRTS: Train on Real, Test on (de-identified) Synthetic 
"""

def preprocess_data(df):

    df_clean = df.copy()
    df_clean = df.drop(columns=['IP Address'], errors='ignore') 
    df_clean = df_clean.fillna("Unknown")
    
    le = LabelEncoder()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            
    return df_clean

def plot_utility_results(accuracies_scores, f1_scores, target_col):
    names = list(accuracies_scores.keys())
    accuracies = list(accuracies_scores.values())
    f1s = list(f1_scores.values())

    x = np.arange(len(names))  
    bar_width = 0.35              

    plt.figure(figsize=(10, 6))
    
    plt.bar(x - bar_width/2, accuracies, bar_width, label='Accuracy', color='skyblue')
    plt.bar(x + bar_width/2, f1s, bar_width, label='F1 Score', color='salmon')
    for i in range(len(names)):
        plt.text(x[i] - bar_width/2, accuracies[i] + 0.01, f"{accuracies[i]:.4f}", ha='center', va='bottom', fontsize=10)
        plt.text(x[i] + bar_width/2, f1s[i] + 0.01, f"{f1s[i]:.4f}", ha='center', va='bottom', fontsize=10)

    plt.ylabel('Utility Score (Accuracy and F1)', fontsize=12)
    plt.xlabel('De-identification method', fontsize=12)

    plt.title(f'Utility evaluation: Train on Real data, Test on de-identified data of different de-identification techniques on target: {target_col}', fontsize=12)
    plt.xticks(x, names)
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()    
    
    plt.tight_layout()
    plt.savefig("../plots/trts_plot.png", dpi=300)
    plt.show()

def evaluate_dataset_utility(df_original, df_protected, target_col):
    model = RandomForestClassifier()
    df_protected = preprocess_data(df_protected)

    different_cols_original=list(set(df_original.columns) - set(df_protected.columns))
    different_cols_protected=list(set(df_protected.columns) - set(df_original.columns))

    cols_to_drop_train = different_cols_original + [target_col]
    X_train = df_original.drop(columns=cols_to_drop_train) 
    y_train = df_original[target_col]
    model.fit(X_train, y_train)

    cols_to_drop_test = different_cols_protected + [target_col]
    X_test = df_protected.drop(columns=cols_to_drop_test) 
    y_test = df_protected[target_col]
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracies_scores[defense_name] = accuracy
    f1_scores[defense_name] = f1
    return accuracy, f1

if __name__ == "__main__":
    df_original = pd.read_csv('../../datasets/mendeley_data.csv') # original data always
    df_original = preprocess_data(df_original)

    target_col = 'Organization'  # Choose a target column that is relevant for utility evaluation, e.g., 'ISP'

    datasets_to_test = {
        "Generalized": pd.read_csv('../../datasets/de-identified-datasets/generalization.csv'),
        "Masking": pd.read_csv('../../datasets/de-identified-datasets/masking.csv'),
        "Masking and generalization" : pd.read_csv('../../datasets/de-identified-datasets/generalization_and_masking.csv'),
        "Data Swapping" : pd.read_csv('../../datasets/de-identified-datasets/swapped_data.csv')
    }

    accuracies_scores = {}
    f1_scores = {}

    for defense_name, df_protected in datasets_to_test.items():
        accuracy, f1 = evaluate_dataset_utility(df_original, df_protected, target_col)
        accuracies_scores[defense_name] = accuracy
        f1_scores[defense_name] = f1

    plot_utility_results(accuracies_scores, f1_scores, target_col)
