from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
TSTR: Train on (de-identified) Synthetic, Test on Real
"""

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
    plt.title(f'Utility evaluation: Train on de-identified data, Test on Real data of different de-identification techniques on target: {target_col}', fontsize=12)
    
    plt.xticks(x, names)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()    
    
    plt.tight_layout()
    plt.savefig("../plots/tstr_plot.png", dpi=300)
    plt.show()

def evaluate_dataset_utility(df_original, df_de_identified, target_col):
    different_cols_original = list(set(df_original.columns) - set(df_de_identified.columns))
    different_cols_protected = list(set(df_de_identified.columns) - set(df_original.columns))
    
    cols_to_drop = list(set(different_cols_original + different_cols_protected))
    if target_col in cols_to_drop:
        cols_to_drop.remove(target_col)

    df_train = df_de_identified.drop(columns=cols_to_drop + ['IP Address'], errors='ignore').fillna("Unknown")
    df_test = df_original.drop(columns=cols_to_drop + ['IP Address'], errors='ignore').fillna("Unknown")

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    text_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    X_train[text_cols] = X_train[text_cols].astype(str)
    X_test[text_cols] = X_test[text_cols].astype(str)
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    X_train[text_cols] = encoder.fit_transform(X_train[text_cols])
    
    X_test[text_cols] = encoder.transform(X_test[text_cols])

    model = RandomForestClassifier(random_state=21) # we use a fixed random state for reproducibility
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, f1


if __name__ == "__main__":
    df_original = pd.read_csv('../../datasets/mendeley_data.csv') # original data always

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
