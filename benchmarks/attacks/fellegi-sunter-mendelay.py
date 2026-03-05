import pandas as pd
import recordlinkage as rl
from matplotlib import pyplot as plt
from pandas.api.types import is_numeric_dtype

def perform_attack(df_protected, df_attacker, quasi_identifiers):    
    df_attacker = df_attacker.copy()
    df_protected = df_protected.copy()
    df_attacker.index.name = 'attacker_idx'
    df_protected.index.name = 'protected_idx'

    indexer = rl.Index()
    indexer.full() # compare every row in attacker with every row in protected, this is computationally expensive but we have a small dataset. very slow for large data sets.
    candidate_links = indexer.index(df_attacker, df_protected) 

    comp = rl.Compare()
    
    for qi in quasi_identifiers:
        if is_numeric_dtype(df_protected[qi]):
            comp.numeric(qi, qi, method='step', offset=1.0, label=qi)
        else:
            comp.string(qi, qi, method='jarowinkler', threshold=0.85, label=qi)

    features = comp.compute(candidate_links, df_attacker, df_protected)

    ecm = rl.ECMClassifier() # ECM = Expectation conditional maximization, based on Fellegi-Sunter model
    ecm.fit(features)
    
    match_probs = ecm.prob(features)

    results = []
    
    for attacker_idx, attacker_row in df_attacker.iterrows():
        probs_for_attacker = match_probs.loc[attacker_idx]
        
        best_target_idx = probs_for_attacker.idxmax()
        
        results.append((attacker_row['IP Address'], best_target_idx))
        
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
    plt.title('Fellegi-Sunter Attack Accuracy Across De-identified Datasets on the Mendeley Dataset', fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("../plots/Fellegi_Sunter_attack_accuracy_mendeley.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    df_original = pd.read_csv('../../datasets/mendeley_data.csv')

    quasi_identifiers = ['Organization', 'Latitude', 'Longitude']
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