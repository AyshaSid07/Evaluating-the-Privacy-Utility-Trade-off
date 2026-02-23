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
                    if val_a == val_p:
                        score += weights[qi].get(val_p, 0.0) # add weight if it's a match
                else: # if value is numerical, calculate distance score
                    diff = abs(float(val_a) - float(val_p))
                    threshold = 1.0 # threshold of latitude/longitude, as we only care about close matches
                    # 1.0 is around 111kmb (i think), so should be possible to be lower
                    if diff < threshold: # we add a threshold of 1.0
                        score += (threshold - diff) 
            
            #save the best match for this attacker row
            if score > best_score:
                best_score = score
                best_match_index = target_idx
                
        results.append((attacker_row['IP Address'], best_match_index)) # save the predicted index for this attacker row, -1 if no match found
        
    return results 
        
def evaluate_attack(results, df_original):
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

    print(f"Attack Precision: {hit_precision:.2f}% ({correct_links}/{total_attacks} correct links)")
    return hit_precision

def plot_results(hit_accuracy):
    plt.figure(figsize=(6, 4))
    plt.bar(['Attack Hit Accuracy'], [hit_accuracy], color='red')
    plt.ylim(0, 100)
    plt.ylabel('Hit accuracy (%)')
    plt.title('Netflix Attack Hit Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    df_original = pd.read_csv('../datasets/mendeley_data.csv')
    #df_protected = pd.read_csv('../datasets/masked_ip.csv') # take a protected dataset
    #df_protected = pd.read_csv('../datasets/hashed_ip.csv') # take a protected dataset
    df_protected = pd.read_csv('../datasets/generalized_lat_long.csv') # take a protected dataset, with generalized lat/long, but still with IP address, should be interesting to see how much the attack is affected by this
    # simulated dataset for the attacker, with only quasi-identifiers and IP, 50 random samples at the moment
    df_attacker = df_original.sample(50, random_state=42)[['IP Address', 'City', 'ISP', 'Latitude', 'Longitude']] 
    QIs = ['City', 'ISP', 'Latitude', 'Longitude']
    
    # first calculate the weights based on the protected dataset, which the attacker have access to
    weights = calculate_weights(df_protected, ['City', 'ISP'])
    
    # then perform the attack using these weights
    results = perform_attack(df_protected, df_attacker, QIs, weights)
    
    hit_accuracy = evaluate_attack(results, df_original)
    plot_results(hit_accuracy)