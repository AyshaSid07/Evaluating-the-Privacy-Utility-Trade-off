import pandas as pd
from math import log
from matplotlib import pyplot as plt

"""
THIS IS NOT WORKING CURRENTLY, might try again some other time
"""

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

def calculate_age_group(age):
    return 0

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
                
                is_match = False
                if qi == 'age':
                    is_match = age_matches(val_a, val_p)
                elif qi == 'zipcode':
                    is_match = zip_matches(val_a, val_p)
                elif qi == 'gender':
                    is_match = (str(val_a).strip() == str(val_p).strip())
                
                if is_match:
                    score += weights[qi].get(val_p, 1.0)
                # if isinstance(val_a, str): # if value is a string, do exact match
                #     if val_a == val_p:
                #         score += weights[qi].get(val_p, 0.0) # add weight if it's a match
                # else: # if value is numerical, calculate distance score
                #     diff = abs(float(val_a) - float(val_p))
                #     threshold = 1.0 # threshold of latitude/longitude, as we only care about close matches
                #     if diff < threshold: # we add a threshold of 1.0
                #         score += (threshold - diff) 
            
            #save the best match for this attacker row
            if score > best_score:
                best_score = score
                best_match_index = target_idx
                
        results.append((attacker_row['name'], best_match_index)) # save the predicted index for this attacker row, -1 if no match found
        if best_match_index != -1:
            print(f"Attacker row: {attacker_row['name']}, Best match index: {df_protected.iloc[best_match_index]}, Score: {best_score:.4f}")
    return results 
        
def evaluate_attack(results, ground_truth, defense_name):
    correct_links = 0
    total_valid_attacks = 0
    
    for row in results:
        true_name = row[0]        # Namnet från attacker-tuple
        predicted_idx = row[1]    # Indexet algoritmen gissade på
        
        # Hämta det sanna indexet från facit
        correct_idx = ground_truth.get(true_name, -1)
        
        # Vi rättar bara om personen faktiskt var unik/fanns i databasen
        if correct_idx != -1: 
            total_valid_attacks += 1
            if predicted_idx == correct_idx:
                correct_links += 1
                
    if total_valid_attacks == 0:
        print(f"Defense Method: {defense_name} - No valid targets found in Ground Truth.")
        return 0.0
        
    hit_precision = (correct_links / total_valid_attacks) * 100

    print(f"Defense Method: {defense_name:<25} - Hit Precision: {hit_precision:.2f}% ({correct_links}/{total_valid_attacks} correct links)")
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
    plt.xlabel('Defense Method', fontsize=12)
    plt.title('Netflix Attack Accuracy Across Defenses on the Sweeney Dataset', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
                 
    plt.tight_layout()
    plt.savefig("../plots/Netflix_attack_accuracy_sweeney.png", dpi=300)
    plt.show()

def age_matches(exact_age, age):
    if pd.isna(exact_age) or pd.isna(age):
        return False
        
    age = str(age).strip()
    
    if "to" in age:
        # e.g. "50 to 69"
        parts = age.split("to")
        low = int(parts[0].strip())
        high = int(parts[1].strip())
        return low <= exact_age <= high
        
    elif "or Older" in age:
        # e.g. "70 or Older"
        low = int(age.replace("or Older", "").strip())
        return exact_age >= low
        
    else:
        try:
            return exact_age == int(age)
        except:
            return False

def zip_matches(exact_zip, masked_zip):
    if pd.isna(exact_zip) or pd.isna(masked_zip):
        return False
        
    exact_zip = str(exact_zip).strip()
    masked_zip = str(masked_zip).strip()
    
    prefix = masked_zip.replace('*', '')
    return exact_zip.startswith(prefix)

if __name__ == "__main__":
    df_original = pd.read_csv('../../datasets/sparcs_medical_records.csv')
    df_attacker = pd.read_csv('../../datasets/voters.csv')
    df_attacker['zipcode'] = df_attacker['zipcode'].astype(str)
    df_attacker['dob'] = pd.to_datetime(df_attacker['dob'])
    df_attacker['age'] = 2015 - df_attacker['dob'].dt.year
    quasi_identifiers = ['age','zipcode', 'gender']
        

    # ground_truth = {}
    
    # for _, voter_row in df_attacker.iterrows():
    #     matches = []
        
    #     for idx, med_row in df_original.iterrows():
    #         gender_match = (str(voter_row['gender']).strip().upper() == str(med_row['gender']).strip().upper())
    #         age_match = age_matches(voter_row['age'], med_row['age'])
    #         zip_match = zip_matches(voter_row['zipcode'], med_row['zipcode'])
            
    #         if gender_match and age_match and zip_match:
    #             matches.append(idx)
                
    #     if len(matches) < 5 and len(matches) != 0:
    #         ground_truth[voter_row['name']] = matches[0]
    #     elif len(matches) > 1:
    #         print(f"Hittade {len(matches)} möjliga patienter för {voter_row['name']}. Inte unik!")
    #         ground_truth[voter_row['name']] = -1
    #     else:
    #         print(f"Hittade 0 patienter för {voter_row['name']}.")
    #         ground_truth[voter_row['name']] = -1

    # add more defenses here
    datasets_to_test = {
        "No Defense (Baseline)": df_original,
        # "Generalization" : pd.read_csv('../../datasets/de-identified-datasets/generalization.csv'),
        # "Masking": pd.read_csv('../../datasets/de-identified-datasets/masking.csv'),
        # "Masking and generalization" : pd.read_csv('../../datasets/de-identified-datasets/generalization_and_masking.csv'),
        # "Data Swapping" : pd.read_csv('../../datasets/de-identified-datasets/swapped_data.csv')
        # "Suppression" : pd.read_csv("../datasets/de-identified-datasets/suppressed_city.csv")
        # "v2": pd.read_csv('../datasets/anonymized_v2.csv'),
        # "v3": pd.read_csv('../datasets/anonymized_v3.csv')
    }

    final_results = {}
    
    for defense_name, df_protected in datasets_to_test.items():
        if 'zipcode' in df_protected.columns:
            df_protected['zipcode'] = df_protected['zipcode'].astype(str)
        weights = calculate_weights(df_protected, quasi_identifiers)
        
        results = perform_attack(df_protected, df_attacker, quasi_identifiers, weights)
        
        # hit_accuracy = evaluate_attack(results, ground_truth, defense_name)
        # final_results[defense_name] = hit_accuracy
        
    plot_results(final_results)