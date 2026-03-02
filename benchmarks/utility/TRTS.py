from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
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


    plt.figure(figsize=(10, 6))
    accuracies_bars = plt.bar(names, accuracies, color=plt.cm.Set3.colors, edgecolor='black')
    f1_bars = plt.bar(names, f1s, color=plt.cm.Set2.colors, edgecolor='black')
    accuracies_text = [plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center', va='bottom', fontsize=10) for i, acc in enumerate(accuracies)]
    f1_text = [plt.text(i, f1 + 0.01, f"{acc:.4f}", ha='center', va='bottom', fontsize=10) for i, f1 in enumerate(f1s)]
    
    plt.ylabel('Accuracy (usability)', fontsize=12)
    plt.xlabel('Defense method', fontsize=12)
    plt.title(f'Utility evaulation: Train on Real data test, Test on de-identified data of different de-identification on target column: {target_col}', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("../plots/trts_plot.png", dpi=300)
    plt.show()
    

if __name__ == "__main__":
    model = RandomForestClassifier()
    df_original = pd.read_csv('../../datasets/mendeley_data.csv') # original data always
    target_col = 'Organization'  # Choose a target column that is relevant for utility evaluation, e.g., 'ISP'
    df_original = preprocess_data(df_original)
    datasets_to_test = {
        "Generalized": pd.read_csv('../../datasets/de-identified-datasets/generalization.csv'),
        "Masking": pd.read_csv('../../datasets/de-identified-datasets/masking.csv'),
        "Masking and generalization" : pd.read_csv('../../datasets/de-identified-datasets/generalization_and_masking.csv'),
        "Data Swapping" : pd.read_csv('../../datasets/de-identified-datasets/swapped_data.csv')
    }

    accuracies_scores = {}
    f1_scores = {}

    for defense_name, df_protected in datasets_to_test.items():
        df_clean = preprocess_data(df_protected)
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
        f1 = f1_score(y_test, y_pred)

        accuracies_scores[defense_name] = accuracy
        f1_scores[defense_name] = f1
    plot_utility_results(accuracies_scores, f1_scores, target_col)
