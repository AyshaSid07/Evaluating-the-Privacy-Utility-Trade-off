# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder

# def preprocess_data(df, target_col):

#     df_clean = df.copy()
    
#     df_clean = df_clean.dropna(subset=[target_col])
    
#     df_clean = df_clean.fillna("Unknown")
    
#     le = LabelEncoder()
#     for col in df_clean.columns:
#         if df_clean[col].dtype == 'object':
#             df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            
#     return df_clean

# def evaluate_dataset_utility(df, target_col):
# #https://jisem-journal.com/index.php/journal/article/view/1198/456
#     df_processed = preprocess_data(df, target_col)
    
#     X = df_processed.drop(columns=[target_col])
#     y = df_processed[target_col]
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     clf = RandomForestClassifier(n_estimators=50, random_state=42)
#     clf.fit(X_train, y_train)
    
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
    
#     return accuracy

# def plot_utility_results(results_dict, target_col):
#     names = list(results_dict.keys())
#     accuracies = list(results_dict.values())
    
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(names, accuracies, color=plt.cm.Set3.colors, edgecolor='black')
#     text = [plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center', va='bottom', fontsize=10) for i, acc in enumerate(accuracies)]
    
#     plt.ylabel('Accuracy (utility)', fontsize=12)
#     plt.xlabel('Dataset', fontsize=12)
#     plt.title(f'Comparisons of Data Utility after different de-identification methods for target column: {target_col}')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
    
#     plt.show()

# if __name__ == "__main__":
#     df_original = pd.read_csv('datasets/mendeley_data.csv') # original data always
#     #data_anon1 = pd.read_csv('datasets/k2.csv') # Choose one or more de-identification methods, e.g., K-anonymity k = 2
#     #data_anon2 = pd.read_csv('datasets/k4.csv') 
#     # ip_masked = pd.read_csv('datasets/masked_ip.csv')
#     # ip_hashed = pd.read_csv('datasets/hashed_ip.csv')
#     # v1 = pd.read_csv('datasets/anonymized_v1.csv')
#     # v2 = pd.read_csv('datasets/anonymized_v2.csv')
#     v3 = pd.read_csv('datasets/generalized.csv')
#     masked = pd.read_csv('datasets/masked.csv')
#     target_col = 'Organization'  # Choose a target column that is relevant for utility evaluation, e.g., 'ISP'

#     #quasi_identifiers = ["Region", "City", "ISP", "Organization", "Country"]  # Adjust based on your dataset's structure


#     results = {
#         "Original Data": evaluate_dataset_utility(df_original, target_col),
#         "Anonymized v1": evaluate_dataset_utility(v3, target_col),
#         # "Anonymized v2": evaluate_dataset_utility(v2, target_col),
#         # "Anonymized v3": evaluate_dataset_utility(v3, target_col),
#         "Masked": evaluate_dataset_utility(masked, target_col)
#     }
    
#     for method, acc in results.items():
#         print(f"{method}: Accuracy = {acc:.4f}")
        
#     plot_utility_results(results, target_col)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(df, target_col):

    df_clean = df.copy()
    
    df_clean = df_clean.dropna(subset=[target_col])
    
    df_clean = df_clean.fillna("Unknown")
    
    le = LabelEncoder()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            
    return df_clean

# def plot_results(results_dict):
#     plt.figure(figsize=(10, 6)) 
#     methods = list(results_dict.keys())
#     accuracies = list(results_dict.values())
    
#     for i in range(len(methods)):
#         bars = plt.bar(methods[i], accuracies[i], color=plt.cm.Set3(i), edgecolor='black')
#         texts = plt.text(methods[i], accuracies[i] + 1, f"{accuracies[i]:.2f}%", ha='center', va='bottom', fontsize=10)
#     plt.ylabel('Hit Accuracy (%)', fontsize=12)
#     plt.xlabel('Defense Method', fontsize=12)
#     plt.title('Dataset Utility Across Defenses', fontsize=14)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
                 
#     plt.tight_layout()
#     plt.show()
def plot_utility_results(results_dict, target_col):
    names = list(results_dict.keys())
    accuracies = list(results_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, color=plt.cm.Set3.colors, edgecolor='black')
    text = [plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center', va='bottom', fontsize=10) for i, acc in enumerate(accuracies)]
    
    plt.ylabel('Accuracy (utility)', fontsize=12)
    plt.xlabel('Defense method', fontsize=12)
    plt.title(f'Comparisons of Data Utility after different de-identification methods for target column: {target_col}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()

model = RandomForestClassifier()
df_original = pd.read_csv('datasets/mendeley_data.csv') # original data always
target_col = 'Organization'  # Choose a target column that is relevant for utility evaluation, e.g., 'ISP'
df_original = preprocess_data(df_original, target_col)
datasets_to_test = {
    "generalized": pd.read_csv('datasets/generalized.csv'),
    "masked": pd.read_csv('datasets/masked.csv'),
    "k4": pd.read_csv('datasets/k4.csv')
}

final_results = {}

for defense_name, df_protected in datasets_to_test.items():
    df_clean = preprocess_data(df_protected, target_col)
    df_protected = preprocess_data(df_protected, target_col)

    different_cols_original=list(set(df_original.columns) - set(df_protected.columns))
    different_cols_protected=list(set(df_protected.columns) - set(df_original.columns))

    different_cols_original.append(target_col)
    different_cols_protected.append(target_col)

    X_train = df_original.drop(columns=different_cols_original) 
    y_train = df_original[target_col]
    model.fit(X_train, y_train)

    X_test = df_protected.drop(columns=different_cols_protected) 
    y_test = df_protected[target_col]
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    final_results[defense_name] = accuracy
plot_utility_results(final_results, target_col)
# quasi_identifiers = ["Region", "City", "ISP", "Organization", "Country"]  # Adjust based on your dataset's structure


# results = {
#     "Original Data": evaluate_dataset_utility(df_original, target_col),
#     "IP Masked": evaluate_dataset_utility(ip_masked, target_col),
#     "IP Hashed": evaluate_dataset_utility(ip_hashed, target_col)
# }

# for method, acc in results.items():
#     print(f"{method}: Accuracy = {acc:.4f}")
    
# plot_utility_results(results, target_col)