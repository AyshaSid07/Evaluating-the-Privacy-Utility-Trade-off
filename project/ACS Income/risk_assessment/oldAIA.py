import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from art.estimators.classification import SklearnClassifier
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox

def plot_results(results_dict):
    plt.figure(figsize=(10, 6)) 
    methods = list(results_dict.keys())
    accuracies = list(results_dict.values())
    
    for i in range(len(methods)):
        plt.bar(methods[i], accuracies[i], color=plt.cm.Set3(i), edgecolor='black')
        plt.text(methods[i], accuracies[i], f"{accuracies[i]:.2f}%", ha='center', va='bottom', fontsize=10)
        
    plt.xticks(rotation=10, ha='right', fontsize=10)    
    plt.ylabel('Re-identification Rate (%)', fontsize=10)
    plt.xlabel('Anonymization Method', fontsize=10)
    plt.title('Attribute Inference Attack Success Rate across k-anonymity, l-diversity\nand t-closeness using the ARX tool on the ACS Income Dataset', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("../plots/attribute_inference_accuracy_income.png", dpi=300)
    plt.show()

datasets_to_test = {
    "No anonymization (Baseline)": pd.read_csv('../datasets/folktables_income_RAW.csv'),
    "ARX Income k = 10": pd.read_csv('../datasets/ARX_income_k=10_l=2.csv'),
    "ARX Income k = 10, l = 2": pd.read_csv('../datasets/ARX_income_k=10_l=2.csv'),
    "ARX Income k = 10, t = 0.2": pd.read_csv('../datasets/ARX_income_k=10_t=0.2.csv'),
    "ARX Income k = 10, t = 0.10": pd.read_csv('../datasets/ARX_income_k=10_t=0.1.csv'),
    "ARX Income k = 10, t = 0.05": pd.read_csv('../datasets/ARX_income_k=10_t=0.1.csv'),

    
    # "Anjana Income k = 2, l = 2": pd.read_csv('../datasets/anjana_income_k=2_l=2.csv'),
    # "Anjana Income k = 5, l = 2": pd.read_csv('../datasets/anjana_income_k=5_l=2.csv'),
    # "Anjana Income k = 20, l = 2": pd.read_csv('../datasets/anjana_income_k=20_l=2.csv'),
    # "Anjana Income k = 50, l = 2": pd.read_csv('../datasets/anjana_income_k=50_l=2.csv'),
}
# datasets_to_test = {
#     "No anonymization (Baseline)": pd.read_csv('../datasets/folktables_income_RAW.csv'), 
#     "ARX Income k = 2, l = 2": pd.read_csv('../datasets/ARX_income_k=2_l=2.csv'),
#     "ARX Income k = 5, l = 2": pd.read_csv('../datasets/ARX_income_k=5_l=2.csv'),
#     "ARX Income k = 10, l = 2": pd.read_csv('../datasets/ARX_income_k=10_l=2.csv'),
#     "ARX Income k = 20, l = 2": pd.read_csv('../datasets/ARX_income_k=20_l=2.csv'),
#     "ARX Income k = 50, l = 2": pd.read_csv('../datasets/ARX_income_k=50_l=2.csv'),
# }
final_results = {}
for defense_name, df_anon in datasets_to_test.items():
    df_anon = df_anon.sample(n=2000, random_state=123).copy()  # Sample 2000 records for faster attack execution
    if 'index' in df_anon.columns:
        df_anon = df_anon.drop(columns=['index'])

    encoders = {}
    for col in df_anon.columns:
        df_anon[col] = df_anon[col].astype(str)
        le = LabelEncoder()
        df_anon[col] = le.fit_transform(df_anon[col])
        encoders[col] = le

    target_col = 'PINCP'

    X = df_anon.drop(columns=[target_col]).values
    y = df_anon[target_col].values

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    art_classifier = SklearnClassifier(model=model)

    attack_feature_index = 9 # index 3 in this case is RAC1P (Race)

    attack = AttributeInferenceBlackBox(art_classifier, attack_feature=attack_feature_index)

    attack.fit(X)

    model_predictions = model.predict(X).reshape(-1, 1)

    X_hacker = np.delete(X, attack_feature_index, axis=1)

    inferred_values = attack.infer(X_hacker, pred=model_predictions)

    true_values = X[:, attack_feature_index]
    acc_balanced = balanced_accuracy_score(true_values.flatten(), inferred_values.flatten()) * 100
    
    print(f"{defense_name} - Balanced Success Rate: {acc_balanced:.2f}%")
    final_results[defense_name] = acc_balanced

plot_results(final_results)