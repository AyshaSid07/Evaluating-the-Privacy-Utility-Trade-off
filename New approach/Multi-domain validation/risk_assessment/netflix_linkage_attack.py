import pandas as pd
from math import log
import numpy as np
# from matplotlib import pyplot as plt # Utkommenterad tills vidare

def calculate_weights(df_protected, quasi_identifiers):
    # Räknar ut TF-IDF-liknande vikter. Unika/ovanliga värden ger högre poäng.
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
    """
    Hjälpfunktion för att hantera Anjanas generaliseringar (t.ex. intervaller och stjärnor).
    """
    str_a = str(val_attacker).strip().lower()
    str_p = str(val_protected).strip().lower()
    
    # 1. Exakt matchning (t.ex. samma kön, eller om ålder inte generaliserats)
    if str_a == str_p:
        return True
        
    # 2. Matchning mot undertryckt data (*)
    if str_p == '*':
        return True # Maskerad data matchar allt (men ger extremt låg poäng via vikterna)
        
    # 3. Matchning mot Anjana-intervall (t.ex. "[20, 40[")
    if str_p.startswith('[') and str_p.endswith('['):
        try:
            # Plocka ut siffrorna från strängen "[20, 40[" -> 20.0 och 40.0
            bounds = str_p[1:-1].split(',')
            lower = float(bounds[0].strip())
            upper = float(bounds[1].strip())
            val_a_float = float(val_attacker)
            
            # Kontrollera om hackerns exakta siffra ligger inom intervallet
            if lower <= val_a_float < upper:
                return True
        except ValueError:
            pass # Om det inte gick att konvertera till float, gå vidare
            
    # Lägg till fler "if"-satser här om ni har hierarkier typ "Higher Ed" vs "University"
    # Men för tillfället räcker detta för siffror och exakta strängar.
    return False

def perform_attack(df_protected, df_attacker, quasi_identifiers, weights):
    results = []
    
    for attacker_idx, attacker_row in df_attacker.iterrows():
        best_score = -1.0
        best_match_index = -1
        
        # Jämför denna hacker-rad mot ALLA skyddade rader
        for target_idx, protected_row in df_protected.iterrows():
            score = 0.0
            
            for qi in quasi_identifiers:
                val_a = attacker_row[qi]
                val_p = protected_row[qi]
                
                if pd.isna(val_a) or pd.isna(val_p):
                    continue
                
                # Om värdena matchar (antingen exakt, via * eller via intervall)
                if is_match(val_a, val_p):
                    # Ge poäng baserat på hur ovanligt det skyddade värdet är
                    str_p = str(val_p).strip().lower()
                    score += weights[qi].get(str_p, 0.0)
            
            # Spara den skyddade rad som får HÖGST poäng för denna hacker-rad
            if score > best_score:
                best_score = score
                best_match_index = target_idx
            # Om vi får oavgjort, behåll den första vi hittade (konservativ attack)
                
        results.append((attacker_idx, best_match_index))
        
    return results 
        
def evaluate_attack(results, df_attacker, df_protected, defense_name):
    correct_links = 0
    total_attacks = len(df_attacker)
    if total_attacks == 0:
        print("No attacks were performed.")
        return 0.0
    
    for attacker_idx, predicted_idx in results:
        if predicted_idx != -1: 
            # Hämta de sanna identiteterna (Linkage_Index)
            true_attacker_id = df_attacker.loc[attacker_idx, 'Linkage_Index']
            guessed_protected_id = df_protected.loc[predicted_idx, 'Linkage_Index']
            
            # Blev det rätt person?
            if true_attacker_id == guessed_protected_id:
                correct_links += 1
                
    hit_precision = (correct_links / total_attacks) * 100
    print(f"Defense Method: {defense_name} - Hit Precision: {hit_precision:.2f}% ({correct_links}/{total_attacks} correct links)")
    return hit_precision

# def plot_results(results_dict):
#     # Utkommenterad tillsvidare!
#     pass

if __name__ == "__main__":
    print("--- Startar Option A: Länkning (Netflix-Attack) ---")
    
    # 1. Ladda in originaldatan (Se till att sökvägen stämmer för dig!)
    df_original = pd.read_csv('../datasets/credit-card-clients.csv')
    
    # Skapa facit: Spara det ursprungliga rad-numret som ett unikt ID för varje person
    df_original['Linkage_Index'] = df_original.index

    # 2. Definiera Quasi-Identifiers (De kolumner angriparen känner till)
    quasi_identifiers = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'LIMIT_BAL']

    # 3. OPTION A: Skapa angriparens dataset genom att sampla från originalet.
    # Hackern vet t.ex. informationen om 100 slumpmässiga kunder.
    sample_size = 300
    df_attacker = df_original.sample(n=sample_size, random_state=42).copy()
    print(f"Hackern har background knowledge om {len(df_attacker)} personer.")

    # 4. Ladda in era skyddade dataset
    # OBS! Anjana brukar generera en egen kolumn som heter 'index' som motsvarar ursprungsraden.
    # Vi döper om den till 'Linkage_Index' så att facit matchar!
    
    df_k2 = pd.read_csv('credit_card_k=2.csv')
    if 'index' in df_k2.columns:
        df_k2 = df_k2.rename(columns={'index': 'Linkage_Index'})
    else:
        # Om Anjana inte genererade ett index, antar vi att radordningen är bevarad
        df_k2['Linkage_Index'] = df_k2.index

    datasets_to_test = {
        "No Defense (Baseline)": df_original,
        "Anjana (k=2)": df_k2
    }

    # 5. Kör experimenten
    final_results = {}
    for defense_name, df_protected in datasets_to_test.items():
        print(f"\nUtvärderar: {defense_name}...")
        
        # Beräkna TF-IDF-vikterna för detta dataset
        weights = calculate_weights(df_protected, quasi_identifiers)
        
        # Kör Netflix-länkningen
        results = perform_attack(df_protected, df_attacker, quasi_identifiers, weights)
        
        # Utvärdera resultatet
        hit_accuracy = evaluate_attack(results, df_attacker, df_protected, defense_name)
        final_results[defense_name] = hit_accuracy
        
    # plot_results(final_results)