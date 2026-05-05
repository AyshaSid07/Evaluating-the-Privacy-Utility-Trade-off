import pandas as pd
import recordlinkage as rl
import numpy as np

def align_attacker_data(df_protected, df_attacker, quasi_identifiers):
    df_attacker_aligned = df_attacker.copy()
    
    for col in quasi_identifiers:
        if df_protected[col].dtype == object or df_protected[col].dtype == str:
            unique_protected_vals = df_protected[col].dropna().unique()
            
            for index, row in df_attacker_aligned.iterrows():
                val_a = row[col]
                if pd.isna(val_a):
                    continue
                
                for val_p in unique_protected_vals:
                    str_p = str(val_p).strip()
                    
                    if str_p.startswith('[') and str_p.endswith('['):
                        try:
                            bounds = str_p[1:-1].split(',')
                            lower = float(bounds[0].strip())
                            upper = float(bounds[1].strip())
                            val_a_float = float(val_a)
                            
                            if lower <= val_a_float < upper:
                                df_attacker_aligned.at[index, col] = str_p
                                break
                        except ValueError:
                            pass
    return df_attacker_aligned


def perform_attack(df_protected, df_attacker_aligned, quasi_identifiers): 
    df_attacker_aligned.index.name = 'attacker_idx'
    df_protected.index.name = 'protected_idx'

    indexer = rl.Index()
    # A full index creates a Cartesian product (N x M pairs).
    indexer.full() # compare every row in attacker with every row in protected, this is computationally expensive but we have a small dataset. very slow for large data sets.
    candidate_links = indexer.index(df_attacker, df_protected) 

    comp = rl.Compare()

    for qi in quasi_identifiers:
        comp.exact(qi, qi, label=qi)

    # Execute comparisons to build the feature vectors for the ML model
    features = comp.compute(candidate_links, df_attacker_aligned, df_protected)

    # ECM = Expectation conditional maximization, based on Fellegi-Sunter model
    # It estimates m- and u-probabilities to calculate 
    # the likelihood of a true match based on the agreement patterns.
    ecm = rl.ECMClassifier() 
    
    ecm.fit(features)
    # Output the final calculated probability (0.0 to 1.0) for each evaluated pair
    match_probs = ecm.prob(features)

    results = []
    
    for attacker_idx in df_attacker_aligned.index:
        if attacker_idx in match_probs.index.get_level_values('attacker_idx'):
            probs_for_attacker = match_probs.loc[attacker_idx]
            best_target_idx = probs_for_attacker.idxmax()  
            results.append((attacker_idx, best_target_idx))
        
    return results

        
def evaluate_attack(results, df_attacker, df_protected, defense_name):
    correct_links = 0
    total_attacks = len(df_attacker)
    if total_attacks == 0 or len(results) == 0:
        print(f"Defense Method: {defense_name} - Hit Precision: 0.00%")
        return 0.0
    
    for attacker_idx, predicted_idx in results:
        if predicted_idx != -1: 
            true_attacker_id = df_attacker.loc[attacker_idx, 'Linkage_Index']
            guessed_protected_id = df_protected.loc[predicted_idx, 'Linkage_Index']
            
            if true_attacker_id == guessed_protected_id: 
                correct_links += 1
                
    hit_precision = (correct_links / total_attacks) * 100
    print(f"Defense Method: {defense_name} - Hit Precision: {hit_precision:.2f}% ({correct_links}/{total_attacks} correct links)")
    return hit_precision


if __name__ == "__main__":
    df_original = pd.read_csv('../datasets/folktables_income_RAW.csv')
    
    df_original['Linkage_Index'] = df_original.index

    quasi_identifiers = ['AGEP', 'COW', 'SCHL', 'MAR', 'SEX']

    sample_size = 2000 
    df_attacker = df_original.sample(n=sample_size, random_state=123).copy()

    datasets_to_test = {
        "No Defense (Baseline)": df_original,
        # "Anjana (k=2)": pd.read_csv('../datasets/.csv'),
    }

    final_results = {}
    for defense_name, df_protected in datasets_to_test.items():
        if 'index' in df_protected.columns:
            df_protected = df_protected.rename(columns={'index': 'Linkage_Index'})
        else:
            df_protected['Linkage_Index'] = df_protected.index
        if defense_name != "No Defense (Baseline)":
            df_attacker_input = align_attacker_data(df_protected, df_attacker, quasi_identifiers)
        else:
            df_attacker_input = df_attacker.copy()
            
        results = perform_attack(df_protected, df_attacker_input, quasi_identifiers)
        evaluate_attack(results, df_attacker, df_protected, defense_name)