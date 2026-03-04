  
from tkinter.font import names

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

"""
ML performance evaluation: Train and test on the same data
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

def evaluate_dataset_utility(df, target_col):
    df_processed = preprocess_data(df)
    
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1

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
    plt.title(f'Utility evaluation: Machine Learning Performance of different de-identification methods for target column: {target_col}', fontsize=12)
    
    plt.xticks(x, names)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()    
    
    plt.tight_layout()
    plt.savefig("../plots/ml_performance_plot.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    target_col = 'Organization'  # Choose a target column that is relevant for utility evaluation, e.g., 'ISP'
    datasets_to_test = {
        "Generalized": pd.read_csv('../../datasets/de-identified-datasets/generalization.csv'),
        "Masking": pd.read_csv('../../datasets/de-identified-datasets/masking.csv'),
        "Masking and generalization" : pd.read_csv('../../datasets/de-identified-datasets/generalization_and_masking.csv'),
        "Data Swapping" : pd.read_csv('../../datasets/de-identified-datasets/swapped_data.csv')
    }

    accuracies_scores = {}
    f1_scores = {}

    for defense_name, df in datasets_to_test.items():
        accuracy, f1 = evaluate_dataset_utility(df, target_col)
        accuracies_scores[defense_name] = accuracy
        f1_scores[defense_name] = f1
    
    plot_utility_results(accuracies_scores, f1_scores, target_col)

