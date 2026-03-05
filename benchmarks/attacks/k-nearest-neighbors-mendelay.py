import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

def perform_attack(df_protected, df_attacker, quasi_identifiers):

    numerical_cols = df_protected[quasi_identifiers].select_dtypes(include=['number']).columns.tolist()
    categorical_cols = [col for col in quasi_identifiers if col not in numerical_cols]

    # Pipeline for numeric data
    # We use StandardScaler so that numbers scale correctly, since they can have outliers 
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])

    # Pipeline for text data
    # One-hot encoding transforms text into categories, based on the unique values in the protected dataset.
    # 'ignore' handles cases where a category was completely masked out in the protected set.
    text_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([('num', num_pipeline, [c for c in numerical_cols if c in quasi_identifiers]),('cat', text_pipeline, categorical_cols)])

    # Transform the data. We fit on the protected data because an attacker 
    # would only have access to the released dataset to base their scales/categories on.
    X_protected = preprocessor.fit_transform(df_protected[quasi_identifiers])
    X_attacker = preprocessor.transform(df_attacker[quasi_identifiers])

    # Setup the KNN model
    knn = NearestNeighbors(n_neighbors=1, metric='manhattan')
    knn.fit(X_protected)

    # Find the single nearest neighbor for each attacker record
    distances, indices = knn.kneighbors(X_attacker)

    results = []
    for i, attacker_row in df_attacker.reset_index().iterrows():
        true_ip = attacker_row['IP Address']
        best_match_index = indices[i][0]
        results.append((true_ip, best_match_index))

    return results

def evaluate_attack(results, df_original, defense_name):
    correct_links = 0
    
    for row in results:
        predicted_idx = row[1]  # predicted index from the tuple
        true_ip = row[0]        # true IP address from the tuple
        
        if predicted_idx != -1: # if we found a match
            predicted_ip = df_original.loc[predicted_idx, 'IP Address']
            if predicted_ip == true_ip:
                correct_links += 1
                
    total_attacks = len(results) # total number of attacker attempts, can't be 0
    if total_attacks == 0:
        print("No attacks were performed.")
        return
    hit_precision = (correct_links / total_attacks) * 100

    print(f"Defense Method: {defense_name} - Hit Precision: {hit_precision:.2f}% ({correct_links}/{total_attacks} correct links)")
    return hit_precision

def plot_results(results_dict):
    plt.figure(figsize=(10, 6)) 
    methods = list(results_dict.keys())
    accuracies = list(results_dict.values())
    
    for i in range(len(methods)):
        plt.bar(methods[i], accuracies[i], color=plt.cm.Set3(i), edgecolor='black')
        plt.text(methods[i], accuracies[i] + 1, f"{accuracies[i]:.2f}%", ha='center', va='bottom', fontsize=10)
        
    plt.ylim(0, 100)
    plt.ylabel('Hit Accuracy (%)', fontsize=12)
    plt.xlabel('De-identified Method', fontsize=12)
    plt.title('K-Nearest Neighbors Attack Accuracy Across De-identified Datasets on the Mendeley Dataset', fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("../plots/KNN_attack_accuracy_mendeley.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    df_original = pd.read_csv('../../datasets/mendeley_data.csv')

    quasi_identifiers = ['Postal Code', 'Latitude', 'Longitude']

    df_attacker = df_original.sample(50, random_state=42)[['IP Address'] + quasi_identifiers] 

    datasets_to_test = {
        "No Defense (Baseline)": df_original,
        "Generalization" : pd.read_csv('../../datasets/de-identified-datasets/generalization.csv'),
        "Masking": pd.read_csv('../../datasets/de-identified-datasets/masking.csv'),
        "Masking & Generalization": pd.read_csv('../../datasets/de-identified-datasets/generalization_and_masking.csv'),
        "Data Swapping" : pd.read_csv('../../datasets/de-identified-datasets/swapped_data.csv')
    }

    final_results = {}
    
    for defense_name, df_protected in datasets_to_test.items():
        results = perform_attack(df_protected, df_attacker, quasi_identifiers)
        
        hit_accuracy = evaluate_attack(results, df_original, defense_name)
        final_results[defense_name] = hit_accuracy
        
    plot_results(final_results)