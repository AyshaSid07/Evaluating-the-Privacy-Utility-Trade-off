import pandas as pd
import recordlinkage as rl
from matplotlib import pyplot as plt

def perform_attack(df_protected, df_attacker, quasi_identifiers, defense_name): # include defense name to handle masked values
    df_attacker.index.name = 'attacker_idx'
    df_protected.index.name = 'protected_idx'

    indexer = rl.Index()
    # A full index creates a Cartesian product (N x M pairs).
    indexer.full() # compare every row in attacker with every row in protected, this is computationally expensive but we have a small dataset. very slow for large data sets.
    candidate_links = indexer.index(df_attacker, df_protected) 

    comp = rl.Compare()


    for qi in quasi_identifiers:
        # Determine what the column is called in the protected dataset
        if qi in df_protected.columns:
            right_qi = qi
        else:
                continue
    if defense_name == "Masking (LAI/RAI)":
        # handle these as special cases, since they have a hierarchical structure that can be partially matched
        # jarowinkler is a string similarity metric that gives a score between 0 and 1 based on how closely the strings match
        # the threshold is based on how much the QI has been masked/generalized.
        if qi == 'LAI':
            comp.string(qi, qi, method='jarowinkler', threshold=0.84, label=qi)
        elif qi == 'RAI':
            comp.string(qi, qi, method='jarowinkler', threshold=0.714, label=qi) 
        else:
            comp.exact(qi, right_qi, label=qi)
    else:
        comp.exact(qi, right_qi, label=qi)

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
    plt.savefig("../plots/Fellegi_Sunter_attack_accuracy_telecom.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    df_original = pd.read_csv('../../datasets/telecom_dataset.csv')

    quasi_identifiers = ['Device_Type','Network_Type', 'PLMN', 'LAI'] 
    df_attacker =  pd.read_csv('../../datasets/external_dataset_telecom.csv')

    datasets_to_test = {
        "No Defense (Baseline)": df_original,
        "Row Suppression (k=2)": pd.read_csv("../../datasets/de_identified_datasets/row_suppression_telecom.csv"),
        "Data Swapping (30%)": pd.read_csv("../../datasets/de_identified_datasets/swapped_dataset_telecom.csv"),
        "Masking (LAI/RAI)": pd.read_csv("../../datasets/de_identified_datasets/masking_telecom.csv")
    }

    final_results = {}
    for defense_name, df_protected in datasets_to_test.items():
        results = perform_attack(df_protected, df_attacker, quasi_identifiers, defense_name)
        hit_accuracy = evaluate_attack(results, df_attacker, df_protected, defense_name)
        final_results[defense_name] = hit_accuracy
        
    plot_results(final_results)