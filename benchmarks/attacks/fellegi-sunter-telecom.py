import pandas as pd
import recordlinkage as rl
from matplotlib import pyplot as plt
from pandas.api.types import is_numeric_dtype

def perform_attack(df_protected, df_attacker, quasi_identifiers):    
    df_attacker.index.name = 'attacker_idx'
    df_protected.index.name = 'protected_idx'

    indexer = rl.Index()
    # A full index creates a Cartesian product (N x M pairs).
    indexer.full() # compare every row in attacker with every row in protected, this is computationally expensive but we have a small dataset. very slow for large data sets.
    candidate_links = indexer.index(df_attacker, df_protected) 

    comp = rl.Compare()


    for qi in quasi_identifiers:
            if qi in ['LAI', 'RAI']: # handle LAI and RAI as special cases, since they have a hierarchical structure that can be partially matched
                # jarowinkler is a string similarity metric that gives a score between 0 and 1 based on how closely the strings match
                # the threshold is based on how much the LAI and RAI has been masked/generalized.
                comp.string(qi, qi, method='jarowinkler', threshold=0.92, label=qi)
            else:
                comp.exact(qi, qi, label=qi)

    # Execute comparisons to build the feature vectors for the ML model
    features = comp.compute(candidate_links, df_attacker, df_protected)

    # ECM = Expectation conditional maximization, based on Fellegi-Sunter model
    # It estimates m- and u-probabilities to calculate 
    # the likelihood of a true match based on the agreement patterns.
    ecm = rl.ECMClassifier() 
    ecm.fit(features)
    
    # Output the final calculated probability (0.0 to 1.0) for each evaluated pair
    match_probs = ecm.prob(features)

    results = []
    
    for attacker_idx in df_attacker.index:
        probs_for_attacker = match_probs.loc[attacker_idx]
        # Get the index of the protected record with the highest match probability
        best_target_idx = probs_for_attacker.idxmax()  
        
        results.append((attacker_idx, best_target_idx))
        
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
    plt.title('Fellegi-Sunter Attack Accuracy Across De-identified Datasets on the Mendeley Dataset', fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("../plots/Fellegi_Sunter_attack_accuracy_mendeley.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    df_original = pd.read_csv('../../datasets/synthetic_telecom_data.csv')

    quasi_identifiers = ['LAI', 'RAI', 'Network_Type']

    df_attacker =  pd.read_csv('../../datasets/external_telecom_data.csv')

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
        results = perform_attack(df_protected, df_attacker, quasi_identifiers)
        hit_accuracy = evaluate_attack(results, defense_name)
        final_results[defense_name] = hit_accuracy
        
    plot_results(final_results)