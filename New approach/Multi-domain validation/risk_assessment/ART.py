import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from art.estimators.classification import SklearnClassifier
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox

#df_anon = pd.read_csv("../datasets/credit_card_k=2.csv")
df_anon = pd.read_csv("../datasets/credit-card-clients.csv")

# Ta bort 'index'-kolumnen som Anjana lade till
if 'index' in df_anon.columns:
    df_anon = df_anon.drop(columns=['index'])

# 2. ENCODING: Konvertera strängar ("[20, 40[", "*", "Higher Ed") till siffror
print("--- Utför Encoding av kategorisk data ---")
encoders = {}
for col in df_anon.columns:
    # Vi gör om allt till strängar tillfälligt för att LabelEncoder ska klara av '*' och NaN
    df_anon[col] = df_anon[col].astype(str)
    le = LabelEncoder()
    df_anon[col] = le.fit_transform(df_anon[col])
    encoders[col] = le # Spara ifall vi vill översätta tillbaka siffran till text senare

# 3. Separera Features (X) och Målvariabel (y)
target_col = 'default payment'
X = df_anon.drop(columns=[target_col]).values
y = df_anon[target_col].values

# 4. Träna Baseline-modellen (Utility)
print("--- Tränar den ursprungliga ML-modellen ---")
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# 5. Konfigurera IBM's ART
# Vi "wrappar" scikit-learn-modellen så ART kan förstå den
art_classifier = SklearnClassifier(model=model)

# 6. Definiera Attacken (Attribute Inference)
# Låt oss säga att angriparen vill gissa personens civilstånd (MARRIAGE). 
# MARRIAGE är kolumn index 3 i vår X-matris (LIMIT_BAL=0, SEX=1, EDUCATION=2, MARRIAGE=3).
attack_feature_index = 3

print(f"--- Sätter upp Attribute Inference Attack (Mål-index: {attack_feature_index}) ---")
# BlackBox-attacken bygger en egen maskininlärningsmodell som försöker lära sig 
# hur din modell beter sig för att kunna återskapa den gömda kolumnen.
attack = AttributeInferenceBlackBox(art_classifier, attack_feature=attack_feature_index)

# Angriparen "tränar" sin attack på ett shadow dataset (vi använder X här för enkelhetens skull)
attack.fit(X)

# 7. Utför attacken! (Infer)
# Angriparen gissar värdet för 'MARRIAGE' för alla personer.
# 7. Utför attacken! (Infer)
# Först måste vi låta vår vanliga ML-modell gissa (predictions) 
# eftersom BlackBox-angriparen använder dessa gissningar som ledtrådar.
# 7. Utför attacken! (Infer)
# Först: Target-modellen gör sina gissningar (den fick se all data när den användes)
# 7. Utför attacken! (Infer)
# Först: Vi låter original-modellen gissa. Vi använder Scikit-learns vanliga .predict()
# eftersom den ger oss en rak klass (t.ex. 0 eller 1) istället för sannolikheter.
# .reshape(-1, 1) tvingar det att bli exakt 1 snygg kolumn.
model_predictions = model.predict(X).reshape(-1, 1)

# Sedan: Hackern samlar in datan, men saknar 'MARRIAGE'-kolumnen.
# Vi raderar kolumnen (axis=1 betyder kolumn) från hackerns dataset, vilket ger 22 kolumner.
X_hacker = np.delete(X, attack_feature_index, axis=1)

# Till sist: Hackern pusslar ihop datan (22) med modellgissningen (1) = 23! (Ingen krasch)
inferred_values = attack.infer(X_hacker, pred=model_predictions)

# 8. Utvärdera resultatet
# Hämta de sanna värdena för att se hur bra hackern var
true_values = X[:, attack_feature_index]

print("\n=== ATTACK RESULTAT ===")
print("Hackerns gissningar (Kodade):", inferred_values)
print("De sanna värdena    (Kodade):", true_values)

# Översätt tillbaka till läsbar text för att se vad de faktiskt gissade!
inferred_text = encoders['MARRIAGE'].inverse_transform(np.round(inferred_values).astype(int))
true_text = encoders['MARRIAGE'].inverse_transform(true_values)

print("\nHackerns gissningar (Text)  :", inferred_text)
print("De sanna värdena    (Text)  :", true_text)

# Enkel uträkning av attackens Accuracy
accuracy = np.mean(inferred_values == true_values) * 100
print(f"\nAttackens Success Rate: {accuracy:.2f}%")