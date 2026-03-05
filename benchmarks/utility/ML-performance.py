  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
"""
ML performance evaluation: Train and test on the same data
"""

def evaluate_dataset_utility(df, target_col):
    df_clean = df.drop(columns=['IP Address'], errors='ignore').fillna("Unknown")
    
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    X_train[cat_cols] = X_train[cat_cols].astype(str)
    X_test[cat_cols] = X_test[cat_cols].astype(str)
    
    # 'use_encoded_value' ensures that if a rare category only ends up in the test set, 
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Fit the encoder ONLY on the training data, then transform both
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])
    
    model = RandomForestClassifier(random_state=42)
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

