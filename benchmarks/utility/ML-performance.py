  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

"""
ML performance evaluation: Train and test on the same data
"""

def preprocess_data(df):

    df_clean = df.copy()
    
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
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def plot_utility_results(results_dict, target_col):
    names = list(results_dict.keys())
    accuracies = list(results_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, color=plt.cm.Set3.colors, edgecolor='black')
    text = [plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center', va='bottom', fontsize=10) for i, acc in enumerate(accuracies)]
    
    plt.ylabel('Accuracy (utility)', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.title(f'Utility evaluation: Machine Learning Performance of different de-identification methods for target column: {target_col}', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("../plots/ml_performance_plot.png", dpi=300)

    plt.show()

if __name__ == "__main__":
    df_original = pd.read_csv('../../datasets/mendeley_data.csv') # original data always
    df_generalized = pd.read_csv('../../datasets/de-identified-datasets/generalization.csv')
    df_masked = pd.read_csv('../../datasets/de-identified-datasets/masking.csv')
    df_generalized_masked = pd.read_csv('../../datasets/de-identified-datasets/generalization_and_masking.csv')
    target_col = 'Organization'  # Choose a target column that is relevant for utility evaluation, e.g., 'ISP'

    results = {
        "Original Data": evaluate_dataset_utility(df_original, target_col),
        "Generalized": evaluate_dataset_utility(df_generalized, target_col),
        "Masking": evaluate_dataset_utility(df_masked, target_col),
        "Masking and generalization" : evaluate_dataset_utility(df_generalized_masked, target_col)
    }
    
    for method, acc in results.items():
        print(f"{method}: Accuracy = {acc:.4f}")
        
    plot_utility_results(results, target_col)
