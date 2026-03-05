import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

def get_comparison_features(row_a, row_p, quasi_identifiers):
    """
    Compares two rows and creates a 'feature vector' describing their differences.
    - Categorical QIs (City, ISP): 1 if exact match, 0 if different.
    - Numerical QIs (Lat, Lon): Absolute mathematical difference.
    """
    features = {}
    
    for qi in quasi_identifiers:
        val_a = row_a[qi]
        val_p = row_p[qi]
        
        if pd.isna(val_a) or pd.isna(val_p):
            if qi in ['Latitude', 'Longitude']:
                features[f'{qi}_diff'] = 999.0 # Penalize heavily
            else:
                features[f'{qi}_match'] = 0    # No match
            continue
            
        if qi in ['Latitude', 'Longitude']:
            try:
                features[f'{qi}_diff'] = abs(float(val_a) - float(val_p))
            except ValueError:
                features[f'{qi}_diff'] = 999.0
                
        else:
            match = 1 if str(val_a).strip().lower() == str(val_p).strip().lower() else 0
            features[f'{qi}_match'] = match
            
    return pd.Series(features)

def train_linkage_model(df_train, quasi_identifiers):
    """
    Trains a Random Forest classifier to distinguish between matches and non-matches.
    We generate positive pairs (Match=1) and negative pairs (Match=0) to teach the AI.
    """
    X_train = []
    y_train = []
    
    for _, row in df_train.iterrows():
        features = get_comparison_features(row, row, quasi_identifiers)
        X_train.append(features)
        y_train.append(1) # 1 is a match
        
    df_shuffled = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    for i, row in df_train.iterrows():
        random_row = df_shuffled.iloc[i]
        
        if row['IP Address'] != random_row['IP Address']:
            features = get_comparison_features(row, random_row, quasi_identifiers)
            X_train.append(features)
            y_train.append(0) # 0 is not a match
            
    X_df = pd.DataFrame(X_train)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_df, y_train)
    
    return clf

def perform_ml_attack(clf, df_protected, df_attacker, quasi_identifiers):
    """
    Performs the Machine Learning linkage attack.
    The AI calculates the probability of a match for every combination.
    """
    results = []
    
    for _, attacker_row in df_attacker.iterrows():
        best_prob = -1.0
        best_match_index = -1
        
        for target_idx, protected_row in df_protected.iterrows():
            features = get_comparison_features(attacker_row, protected_row, quasi_identifiers)
            
            prob_match = clf.predict_proba(pd.DataFrame([features]))[0][1]
            
            if prob_match > best_prob:
                best_prob = prob_match
                best_match_index = target_idx
                
        results.append((attacker_row['IP Address'], best_match_index))
        
    return results

def evaluate_attack(results, df_original, defense_name):
    """Evaluates the precision of the attack"""
    correct_links = 0
    total_attacks = len(results)
    
    if total_attacks == 0: return
    
    for row in results:
        true_ip = row[0]
        predicted_idx = row[1]
        
        if predicted_idx != -1:
            predicted_ip = df_original.loc[predicted_idx, 'IP Address']
            if predicted_ip == true_ip:
                correct_links += 1
                
    hit_precision = (correct_links / total_attacks) * 100
    print(f"Defense Method: {defense_name:<25} - Hit Precision: {hit_precision:.2f}% ({correct_links}/{total_attacks})")
    return hit_precision

def plot_results(results_dict):
    """Plots the bar chart"""
    plt.figure(figsize=(10, 6))
    methods = list(results_dict.keys())
    accuracies = list(results_dict.values())
    
    for i in range(len(methods)):
        plt.bar(methods[i], accuracies[i], color=plt.cm.Set3(i), edgecolor='black')
        plt.text(methods[i], accuracies[i] + 1, f"{accuracies[i]:.2f}%", ha='center', va='bottom', fontsize=10)
        
    plt.ylim(0, 100)
    plt.ylabel('Hit Accuracy (%)', fontsize=12)
    plt.xlabel('Defense Method', fontsize=12)
    plt.title('Machine Learning-based Linkage Attack Accuracy', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_original = pd.read_csv('../../datasets/mendeley_data.csv')

    quasi_identifiers = ['City', 'ISP', 'Latitude', 'Longitude']
    df_attacker = df_original.sample(50, random_state=42)[['IP Address'] + quasi_identifiers] 

    ml_model = train_linkage_model(df_original, quasi_identifiers)

    datasets_to_test = {
        "No Defense (Baseline)": df_original,
        "Generalization" : pd.read_csv('../../datasets/de-identified-datasets/generalization.csv'),
        "Masking": pd.read_csv('../../datasets/de-identified-datasets/masking.csv'),
        "Masking & Generalization": pd.read_csv('../../datasets/de-identified-datasets/generalization_and_masking.csv'),
        "Data Swapping" : pd.read_csv('../../datasets/de-identified-datasets/swapped_data.csv')
    }

    final_results = {}
    
    for defense_name, df_protected in datasets_to_test.items():
        results = perform_ml_attack(ml_model, df_protected, df_attacker, quasi_identifiers)
        hit_accuracy = evaluate_attack(results, df_original, defense_name)
        final_results[defense_name] = hit_accuracy
        
    plot_results(final_results)