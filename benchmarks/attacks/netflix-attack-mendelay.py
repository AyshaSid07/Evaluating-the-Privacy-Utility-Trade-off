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
        value_counts = df_protected[col].value_counts()
        for value, count in value_counts.items():
            weights[col][value] = log(total_records / count)
            
    return weights

def perform_attack(df_protected, df_attacker, quasi_identifiers, weights):
    results = []
    
    for _, attacker_row in df_attacker.iterrows():
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
                
                if isinstance(val_a, str): # if value is a string, do exact match
                    if str(val_a).strip().lower() == str(val_p).strip().lower():
                        score += weights[qi].get(val_p, 0.0) # add weight if it's a match
                else: # if value is numerical, calculate distance score
                    diff = abs(float(val_a) - float(val_p))
                    threshold = 1.0 # threshold of latitude/longitude, as we only care about close matches
                    if diff < threshold: # we add a threshold of 1.0
                        # base_weight = weights[qi].get(val_p, 0.0)
                        # distance_penalty = (threshold - diff) / threshold
                        # score += base_weight * distance_penalty

                        # for some reason, this performs better for this dataset.
                        score += (threshold - diff) 
            
            #save the best match for this attacker row
            if score > best_score:
                best_score = score
                best_match_index = target_idx
                
        results.append((attacker_row['IP Address'], best_match_index)) # save the predicted index for this attacker row, -1 if no match found
        
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
    plt.title('Netflix Attack Accuracy Across De-identified Datasets on the Mendeley Dataset', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
                 
    plt.tight_layout()
    plt.savefig("../plots/Netflix_attack_accuracy_mendeley.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    df_original = pd.read_csv('../../datasets/mendeley_data.csv')

    quasi_identifiers = ['Postal Code', 'Latitude', 'Longitude']

    # we assume the attacker only has access to some of the quasi-identifiers, and not the IP address, and also only 50 random records to link.
    # we add the IP-address just so we can evaluate how well the attack went. The attacker is only using the quasi-identifiers
    df_attacker = df_original.sample(50, random_state=42)[['IP Address'] + quasi_identifiers] 
    
        
    # add more defenses here
    datasets_to_test = {
        "No Defense (Baseline)": df_original,
        "Generalization" : pd.read_csv('../../datasets/de-identified-datasets/generalization.csv'),
        "Masking": pd.read_csv('../../datasets/de-identified-datasets/masking.csv'),
        "Masking and generalization" : pd.read_csv('../../datasets/de-identified-datasets/generalization_and_masking.csv'),
        "Data Swapping" : pd.read_csv('../../datasets/de-identified-datasets/swapped_data.csv')
        # "Suppression" : pd.read_csv("../datasets/de-identified-datasets/suppressed_city.csv")
        # "v2": pd.read_csv('../datasets/anonymized_v2.csv'),
        # "v3": pd.read_csv('../datasets/anonymized_v3.csv')
    }

    final_results = {}
    
    for defense_name, df_protected in datasets_to_test.items():
        
        weights = calculate_weights(df_protected, quasi_identifiers)
        
        results = perform_attack(df_protected, df_attacker, quasi_identifiers, weights)
        
        hit_accuracy = evaluate_attack(results, df_original, defense_name)
        final_results[defense_name] = hit_accuracy
        
    plot_results(final_results)