import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

def perform_attack(df_protected, df_attacker, quasi_identifiers, k):
    # can include the outcommonted ones to make it more dynamic
    numerical_cols = df_protected[quasi_identifiers].select_dtypes(include=['number']).columns.tolist()
    text_cols = [col for col in quasi_identifiers if col not in numerical_cols]
    #numerical_cols =  ['Latitude', 'Longitude']
    #text_cols = ['Postal Code'] 

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

def evaluate_attack(results, defense_name):
    correct_links = 0
    total_attacks = len(results) # total number of attacker attempts, can't be 0
    if total_attacks == 0:
        print("No attacks were performed.")
        return
    
    for attacker_idx, predicted_idx in results:
        
        if predicted_idx != -1: # if we found a match
            if predicted_idx == attacker_idx: # if the predicted index matches the true index, it's a correct link
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
        plt.text(methods[i], accuracies[i] + 1, f"{accuracies[i]:.2f}%", ha='center', va='bottom', fontsize=10)
        
    plt.ylabel('Hit Accuracy (%)', fontsize=12)
    plt.xlabel('De-identified Method', fontsize=12)
    plt.title(f'K-Nearest Neighbors with k={k} Attack Accuracy Across De-identified Datasets on the Mendeley Dataset', fontsize=12)
        

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("../plots/KNN_attack_accuracy_mendeley.png", dpi=300)
    plt.show()

def mask_LAI_RAI(value): # since we assume the attacker has the access to the masked dataset, we can assume they know the masking method and can try to handle it in their attack. This is a simple example of how they could do that for the LAI and RAI fields.
    if pd.isna(value):
        return value
    parts = str(value).split('-')
    if len(parts) == 3:
        return f"{parts[0]}-{parts[1]}-{parts[2][:2]}**"
    elif len(parts) == 4:
        return f"{parts[0]}-{parts[1]}-{parts[2][:2]}**-**"

if __name__ == "__main__":
    df_original = pd.read_csv('../../datasets/synthetic_telecom_data.csv')

    quasi_identifiers = ['LAI', 'RAI', 'Network_Type']

    df_attacker =  pd.read_csv('../../datasets/external_telecom_data.csv')
    # only use 500 rows to speed up the attack
    # using df.sample(500, random_state=42) would be more realistic, but for simplicity we just take the first 500 rows here.
    df_original = df_original.head(500)
    df_attacker = df_attacker.head(500)
    df_masked = pd.read_csv('../../datasets/de-identified-datasets/masked_dataset_telecom.csv')
    df_masked = df_masked.head(500)
    df_masked['LAI'] = df_masked['LAI'].apply(mask_LAI_RAI)
    df_masked['RAI'] = df_masked['RAI'].apply(mask_LAI_RAI)
    df_swapped = pd.read_csv('../../datasets/de-identified-datasets/swapped_telecom_data.csv')
    df_swapped = df_swapped.head(500)
    df_suppressed_network_type = pd.read_csv('../../datasets/de-identified-datasets/suppressed_network_type_telecom.csv')
    df_suppressed_network_type = df_suppressed_network_type.head(500)
    df_swapped_and_suppressed_network_type = pd.read_csv('../../datasets/de-identified-datasets/swapped_and_suppressed_network_type_telecom.csv')
    df_swapped_and_suppressed_network_type = df_swapped_and_suppressed_network_type.head(500)

    k = 1

    datasets_to_test = {
        "No Defense (Baseline)": df_original,
        # "Generalization" : pd.read_csv('../../datasets/de-identified-datasets/generalization.csv'),
        "Masking": df_masked,
        "Data Swapping" : df_swapped,
        "Suppressed network type" : df_suppressed_network_type
        # "Masking & Generalization": pd.read_csv('../../datasets/de-identified-datasets/generalization_and_masking.csv'),
        # "Data Swapping" : pd.read_csv('../../datasets/de-identified-datasets/swapped_data.csv')
    }

    final_results = {}
    
    for defense_name, df_protected in datasets_to_test.items():
        if defense_name.__contains__("Suppressed"):

            quasi_identifiers = ['LAI', 'RAI']

        
        results = perform_attack(df_protected, df_attacker, quasi_identifiers, k)
        
        hit_accuracy = evaluate_attack(results, defense_name)
        final_results[defense_name] = hit_accuracy
        
    plot_results(final_results, k)