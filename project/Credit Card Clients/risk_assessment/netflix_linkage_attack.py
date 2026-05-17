from time import time

import pandas as pd
from math import log
import numpy as np
import matplotlib.pyplot as plt
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

def is_match(val_attacker, val_protected):
    # help function to get through anonymization
    str_a = str(val_attacker).strip().lower()
    str_p = str(val_protected).strip().lower()
    
    if str_a == str_p:
        return True
        
    if str_p == '*':
        return True 
        
    if str_p.startswith('[') and str_p.endswith('['):
        
        bounds = str_p[1:-1].split(',')
        lower = float(bounds[0].strip())
        upper = float(bounds[1].strip())
        val_a_float = float(val_attacker)
        
        if lower <= val_a_float < upper:
            return True

    return False

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
                
                if pd.isna(val_a) or pd.isna(val_p):
                    continue
                
                if is_match(val_a, val_p):
                    str_p = str(val_p).strip().lower()
                    score += weights[qi].get(str_p, 0.0)

            #save the best match for this attacker row
            if score > best_score:
                best_score = score
                best_match_index = target_idx
                
        results.append((attacker_idx, best_match_index))
        
    return results 
        
def evaluate_attack(results, df_attacker, df_protected, defense_name):
    correct_links = 0
    total_attacks = len(df_attacker)
    if total_attacks == 0:
        print("No attacks were performed.")
        return 0.0
    
    for attacker_idx, predicted_idx in results:
        if predicted_idx != -1:  # if we found a match
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
    plt.ylabel('Re-identification Rate (%)', fontsize=10)
    plt.xlabel('Epsilon Value', fontsize=10)
    # plt.xlabel('k-Anonymity Value', fontsize=10)
    plt.title('Netflix Attack Re-Identification Rate Across different epsilon values of Differential Privacy\n on the MEPS Dataset', fontsize=10)
    # plt.title('Netflix Attack Re-Identification Rate Across different k-values of k-Anonymity\n on the MEPS Dataset', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.savefig("../plots/netflix_attack_accuracy_meps_ARX.png", dpi=300)
    plt.savefig("../plots/netflix_attack_accuracy_meps_DP.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    df_original = pd.read_csv('../datasets/credit-card-clients.csv')

    df_original['Linkage_Index'] = df_original.index

    quasi_identifiers = ["SEX", "EDUCATION", "AGE", "MARRIAGE"]


    datasets_to_test = {
        "No anonymization (Baseline)": df_original,
        "ARX Credit Card Clients, k = 3": pd.read_csv('../datasets/ARX_credit_card_clients_k3.csv'),
        "ARX Credit Card Clients, k = 5": pd.read_csv('../datasets/ARX_credit_card_clients_k5.csv'),
        # "ARX Credit Card Clients, k = 10": pd.read_csv('../datasets/ARX_credit_card_clients_k10.csv'),
        # "ARX Credit Card Clients, k = 15": pd.read_csv('../datasets/ARX_credit_card_clients_k15.csv'),
        # "ARX Credit Card Clients, k = 5, l = 2": pd.read_csv('../datasets/ARX_credit_card_clients_k5_l2.csv'),
        "ARX Credit Card Clients, k = 5, l = 4": pd.read_csv('../datasets/ARX_credit_card_clients_k5_l4.csv'),
        # "ARX Credit Card Clients, k = 5, t = 0.3": pd.read_csv('../datasets/ARX_credit_card_clients_k5_t0.3.csv'),
        "ARX Credit Card Clients, k = 5, t = 0.15": pd.read_csv('../datasets/ARX_credit_card_clients_k5_t0.15.csv'),
        "DP Credit Card Clients, epsilon = 10.0": pd.read_csv('../datasets/DP_credit_card_clients_epsilon_10_0.csv'),
        # "DP Credit Card Clients, epsilon = 5.0":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_5_0.csv'),
        # "DP Credit Card Clients, epsilon = 3.0":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_3_0.csv'),
        "DP Credit Card Clients, epsilon = 1.0":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_1_0.csv'),
        "DP Credit Card Clients, epsilon = 0.5":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_0_5.csv'),
        "Combined Credit Card Clients, k = 3 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_credit_card_clients.csv'),
        "Combined Credit Card Clients, k = 3 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_credit_card_clients.csv'),
        # "Combined Credit Card Clients, k = 3 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_credit_card_clients.csv'),
        "Combined Credit Card Clients, k = 5 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_credit_card_clients.csv'),
        "Combined Credit Card Clients, k = 5 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_credit_card_clients.csv'),
        # "Combined Credit Card Clients, k = 5 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_credit_card_clients.csv'),
    }

    final_results = {}
    start = time()
    
    for defense_name, df_protected in datasets_to_test.items():
        print(f"\nEvaluating defense: {defense_name}")
        
        if 'index' in df_protected.columns:
            df_protected = df_protected.rename(columns={'index': 'Linkage_Index'})
        elif 'Linkage_Index' not in df_protected.columns:
            df_protected['Linkage_Index'] = df_original.index 

        # n_sample = 1000
        # if len(df_protected) > n_sample:
        #     df_protected_sample = df_protected.sample(n=n_sample, random_state=42).copy()
        # else:
        #     df_protected_sample = df_protected.copy()
        df_protected_sample = df_protected.copy()
        attacker_indices = df_protected_sample['Linkage_Index'].sample(n=300, random_state=123)
        
        df_attacker = df_original[df_original['Linkage_Index'].isin(attacker_indices)].copy()
        

        weights = calculate_weights(df_protected_sample, quasi_identifiers)
        
        results = perform_attack(df_protected_sample, df_attacker, quasi_identifiers, weights)
        
        hit_accuracy = evaluate_attack(results, df_attacker, df_protected_sample, defense_name)
        final_results[defense_name] = hit_accuracy

    # plot_results(final_results)
    end = time()
    print(f"Total execution time: {end - start} seconds")
    data = [{'Dataset': name, 'Hit_Precision': precision} for name, precision in final_results.items()]
    df_results = pd.DataFrame(data)
    df_results.to_csv("../plots/linkage/linkage_attack_results_credit_card_clients.csv", index=False)
