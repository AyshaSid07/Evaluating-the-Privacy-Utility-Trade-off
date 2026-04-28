import pandas as pd
from math import log
import numpy as np

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
        weights = calculate_weights(df_protected, quasi_identifiers)
        
        results = perform_attack(df_protected, df_attacker, quasi_identifiers, weights)
        
        hit_accuracy = evaluate_attack(results, df_attacker, df_protected, defense_name)
        final_results[defense_name] = hit_accuracy
        