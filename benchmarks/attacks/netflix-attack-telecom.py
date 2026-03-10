import pandas as pd
from math import log
from matplotlib import pyplot as plt

def calculate_weights(df_protected, quasi_identifiers):
    # Calculate weights based on the frequency of each value in the quasi-identifiers in the dataset
    # The rarer a value is, the higher weight it get
    weights = {}
    total_records = len(df_protected)
    for col in quasi_identifiers:
        weights[col] = {}
        cleaned_col = df_protected[col].astype(str).str.strip().str.lower()
        value_counts = cleaned_col.value_counts()
        
        for value, count in value_counts.items():
            weights[col][value] = log(total_records / count)
            
    return weights

def perform_attack(df_protected, df_attacker, quasi_identifiers, weights):
    results = []
    
    for attacker_idx, attacker_row in df_attacker.iterrows():
        best_score = -1.0
        best_match_index = -1
        
        # compare this attacker row against all protected rows
        for target_idx, protected_row in df_protected.iterrows():
            score = 0.0
            
            for qi in quasi_identifiers:
                val_a = attacker_row[qi]
                val_p = protected_row[qi]
                
                if pd.isna(val_a) or pd.isna(val_p): # ignore missing values
                    continue
                
                str_a = str(val_a).strip().lower()
                str_p = str(val_p).strip().lower()
                
                if str_a == str_p:
                    score += weights[qi].get(str_p, 0.0)
                elif '*' in str_p: # this is how the attacker could handle masked values, by checking if the non-masked part matches
                    prefix = str_p.replace('*', '')
                    if str_a.startswith(prefix) and prefix != '':
                        score += weights[qi].get(str_p, 0.0)
            
            #save the best match for this attacker row
            if score > best_score:
                best_score = score
                best_match_index = target_idx
                
        results.append((attacker_idx, best_match_index)) # save the predicted index for this attacker row, -1 if no match found
        
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

def plot_results(results_dict):
    plt.figure(figsize=(10, 6)) 
    methods = list(results_dict.keys())
    accuracies = list(results_dict.values())
    
    for i in range(len(methods)):
        plt.bar(methods[i], accuracies[i], color=plt.cm.Set3(i), edgecolor='black')
        plt.text(methods[i], accuracies[i], f"{accuracies[i]:.2f}%", ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=10, ha='right', fontsize=10)    
    plt.ylabel('Hit Accuracy (%)', fontsize=12)
    plt.xlabel('De-identified Method', fontsize=12)
    plt.title('Netflix Attack Accuracy Across De-identified Datasets on the Telecom Dataset', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
                 
    plt.tight_layout()
    plt.savefig("../plots/Netflix_attack_accuracy_telecom.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    df_original = pd.read_csv('../../datasets/synthetic_telecom_data.csv')

    quasi_identifiers = ['LAI', 'RAI', 'Network_Type']

    df_attacker =  pd.read_csv('../../datasets/external_telecom_data.csv')

    # add more defenses here
    datasets_to_test = {
        "No Defense (Baseline)": df_original,
        "Masked": pd.read_csv('../../datasets/de-identified-datasets/masked_dataset_telecom.csv'),
        "Data Swapping" : pd.read_csv('../../datasets/de-identified-datasets/swapped_dataset_telecom.csv'),
        "Suppressed network type" : pd.read_csv('../../datasets/de-identified-datasets/suppressed_network_type_telecom.csv'),
        "Swapped and Suppressed network type" : pd.read_csv('../../datasets/de-identified-datasets/swapped_and_suppressed_network_type_telecom.csv'),
        "Masked and Suppressed network type" : pd.read_csv('../../datasets/de-identified-datasets/masked_and_suppressed_network_type_telecom.csv')
    }

    final_results = {}
    # only use 500 rows to speed up the attack
    # using df.sample(500, random_state=42) would be more realistic, but for simplicity we just take the first 500 rows here.
    df_attacker = df_attacker.head(500)
    for defense_name, df_protected in datasets_to_test.items():
        df_protected = df_protected.head(500) # only use 500 rows to speed up the attack, and to be consistent with the attacker's dataset size
        if defense_name.__contains__("Suppressed"):
            quasi_identifiers = ['LAI', 'RAI']

        weights = calculate_weights(df_protected, quasi_identifiers)
        
        results = perform_attack(df_protected, df_attacker, quasi_identifiers, weights)
        
        hit_accuracy = evaluate_attack(results, defense_name)
        final_results[defense_name] = hit_accuracy
        
    plot_results(final_results)