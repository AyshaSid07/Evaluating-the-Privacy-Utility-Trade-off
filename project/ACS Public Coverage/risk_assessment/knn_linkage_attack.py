import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def align_attacker_data(df_protected, df_attacker, quasi_identifiers):
    df_attacker_aligned = df_attacker.copy()
    
    for col in quasi_identifiers:
        if df_protected[col].dtype == object:
            df_attacker_aligned[col] = df_attacker_aligned[col].astype(object)
            unique_protected_vals = df_protected[col].unique()
            
            for index, row in df_attacker_aligned.iterrows():
                val_a = row[col]
                
                for val_p in unique_protected_vals:
                    str_p = str(val_p).strip()
                    
                    if str_p.startswith('[') and str_p.endswith('[') or str_p.startswith('[') and str_p.endswith(']'):
                        bounds = str_p[1:-1].split(',')
                        lower = float(bounds[0].strip())
                        upper = float(bounds[1].strip())
                        val_a_float = float(val_a)
                        
                        if lower <= val_a_float < upper:
                            df_attacker_aligned.at[index, col] = str_p
                            break
                    elif str_p == '*':
                        pass
                        
    return df_attacker_aligned

def perform_attack(df_protected, df_attacker, quasi_identifiers, k):
    df_attacker = align_attacker_data(df_protected, df_attacker, quasi_identifiers)

    numerical_cols = df_protected[quasi_identifiers].select_dtypes(include=['number']).columns.tolist()
    text_cols = [col for col in quasi_identifiers if col not in numerical_cols]
    for col in text_cols:
        df_protected[col] = df_protected[col].astype(str)
        df_attacker[col] = df_attacker[col].astype(str)
    # Pipeline for numeric data
    # We use StandardScaler so that numbers scale correctly, since they can have outliers 
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])

    # Pipeline for text data
    # One-hot encoding transforms text into categories, based on the unique values in the protected dataset.
    # 'ignore' handles cases where a category was completely masked out in the protected set.
    text_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([('num', num_pipeline, [c for c in numerical_cols if c in quasi_identifiers]),('text', text_pipeline, text_cols)])

    # Transform the data. We fit on the protected data because an attacker 
    # would only have access to the released dataset to base their scales/categories on.
    X_protected = preprocessor.fit_transform(df_protected[quasi_identifiers])
    X_attacker = preprocessor.transform(df_attacker[quasi_identifiers])

    # Setup the KNN model
    knn = NearestNeighbors(n_neighbors=k, metric='manhattan')
    knn.fit(X_protected)

    # Find the single nearest neighbor for each attacker record
    distances, indices = knn.kneighbors(X_attacker)

    results = []
    for i, attacker_idx in enumerate(df_attacker.index):
        best_match_index = indices[i][0]
        results.append((attacker_idx, best_match_index))

    return results

        
def evaluate_attack(results, df_attacker, df_protected, defense_name):
    correct_links = 0
    total_attacks = len(df_attacker) # total number of attacker attempts, can't be 0
    if total_attacks == 0:
        print("No attacks were performed.")
        return
    
    for attacker_idx, predicted_idx in results:
        
        if predicted_idx != -1: # if we found a match
            true_attacker_id = df_attacker.loc[attacker_idx, 'Linkage_Index']
            guessed_protected_id = df_protected.loc[predicted_idx, 'Linkage_Index']
            if true_attacker_id == guessed_protected_id: # if the predicted index matches the true index, it's a correct link
                correct_links += 1
                
    hit_precision = (correct_links / total_attacks) * 100

    print(f"Defense Method: {defense_name} - Hit Precision: {hit_precision:.2f}% ({correct_links}/{total_attacks} correct links)")
    return hit_precision


def plot_results(results_dict, k):
    plt.figure(figsize=(10, 6)) 
    methods = list(results_dict.keys())
    accuracies = list(results_dict.values())
    
    for i in range(len(methods)):
        plt.bar(methods[i], accuracies[i], color=plt.cm.Set3(i), edgecolor='black')
        plt.text(methods[i], accuracies[i], f"{accuracies[i]:.2f}%", ha='center', va='bottom', fontsize=10)
        
    plt.xticks(rotation=10, ha='right', fontsize=10)    
    plt.ylabel('Re-identification Rate (%)', fontsize=10)
    plt.xlabel('De-identification Method', fontsize=10)
    plt.title(f'K-Nearest Neighbors with k={k} Re-Identification Rate Across De-identified Datasets on the "Sensitive telecom attributes" Dataset', fontsize=9)
        

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("../plots/KNN_attack_accuracy_telecom.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    df_original = pd.read_csv('../datasets/folktables_public_coverage_RAW.csv')
    
    df_original['Linkage_Index'] = df_original.index

    quasi_identifiers = ['AGEP', 'SCHL', 'MAR', 'SEX', 'ESP', 'CIT', 'MIL', 'ANC', 'ESR']

    df_attacker = df_original.sample(n=2000, random_state=123).copy()

    datasets_to_test = {
        "Baseline (no anonymization)": df_original,
        "Anjana k-anonymity (k=2)":          pd.read_csv('../datasets/public_coverage_k=2.csv'),
        "Anjana k-anonymity (k=5)":          pd.read_csv('../datasets/public_coverage_k=5.csv'),
        "Anjana k-anonymity (k=10)":          pd.read_csv('../datasets/public_coverage_k=10.csv'),
        "Anjana k-anonymity (k=20)":          pd.read_csv('../datasets/public_coverage_k=20.csv'),
        "Anjana k-anonymity (k=50)":          pd.read_csv('../datasets/public_coverage_k=50.csv'),
        "Arx k-anonymity (k=2)":          pd.read_csv('../datasets/ARX_public_coverage_k=2.csv'),
        "Arx k-anonymity (k=5)":          pd.read_csv('../datasets/ARX_public_coverage_k=5.csv'),
        "Arx k-anonymity (k=10)":          pd.read_csv('../datasets/ARX_public_coverage_k=10.csv'),
        "Arx k-anonymity (k=20)":          pd.read_csv('../datasets/ARX_public_coverage_k=20.csv'),
        "Arx k-anonymity (k=50)":          pd.read_csv('../datasets/ARX_public_coverage_k=50.csv'),
    }
    
    final_results = {}

    k = 1 # baseline k value

    for defense_name, df_protected in datasets_to_test.items():
        if 'index' in df_protected.columns:
            df_protected = df_protected.rename(columns={'index': 'Linkage_Index'})
        else:
            df_protected['Linkage_Index'] = df_protected.index
        results = perform_attack(df_protected, df_attacker, quasi_identifiers, k)
        hit_accuracy = evaluate_attack(results, df_attacker, df_protected, defense_name)

        final_results[defense_name] = hit_accuracy
        
    # plot_results(final_results, k)