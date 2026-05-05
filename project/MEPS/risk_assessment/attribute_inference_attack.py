import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from art.estimators.classification import SklearnClassifier
from art.attacks.inference.attribute_inference import AttributeInferenceBaseline, AttributeInferenceBlackBox

def plot_results(results_df):
    plt.figure(figsize=(12, 6)) 
    
    methods = results_df['Dataset'].tolist()
    blackbox_acc = results_df['BlackBox_Accuracy'].tolist()
    
    x = np.arange(len(methods))
    width = 0.7 
    
    colors = [plt.cm.Set3(i) for i in range(len(methods))]
    
    plt.bar(x, blackbox_acc, width, color=colors, edgecolor='black')
    
    for i in range(len(methods)):
        plt.text(x[i], blackbox_acc[i], f"{blackbox_acc[i]:.2f}", ha='center', va='bottom', fontsize=10)
        
    plt.xticks(x, methods, rotation=15, ha='right', fontsize=10)    
    plt.ylabel('Re-identification Rate (Accuracy)', fontsize=10)
    plt.xlabel('Anonymization Value', fontsize=10)
    plt.title('Attribute Inference Attack Results on different anonymization values on the MEPS Data', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("../plots/attribute_inference_MEPS.png", dpi=300)
    plt.show()

datasets_to_test = {
    "No anonymization (Baseline)": pd.read_csv('../datasets/MEPS.csv'),
    "ARX MEPS, k = 5": pd.read_csv('../datasets/ARX_meps_k=5.csv'),
    "ARX MEPS, k = 5, l = 2": pd.read_csv('../datasets/ARX_meps_k=5_l=2.csv'),
    "ARX MEPS, k = 5, l = 3": pd.read_csv('../datasets/ARX_meps_k=5_l=3.csv'),
    "ARX MEPS, k = 5, t = 0.2": pd.read_csv('../datasets/ARX_meps_k=5_t=0.2.csv'),
    "ARX MEPS, k = 5, t = 0.1": pd.read_csv('../datasets/ARX_meps_k=5_t=0.1.csv'),
}

PREDICTION_TARGET = 'UTILIZATION' 
SENSITIVE_ATTR = 'INSCOV'

results = []

for name, df in datasets_to_test.items():
    print(f"\n--- Running attacks on: {name} ---")
    
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    y = df[PREDICTION_TARGET].astype(int).values
    sens_y = df[SENSITIVE_ATTR].astype(int).values
    
    X_raw = df.drop(columns=[PREDICTION_TARGET, SENSITIVE_ATTR])
    
    categorical_cols = X_raw.select_dtypes(include=['object', 'string', 'category']).columns
    numerical_cols = X_raw.select_dtypes(include=['number']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Fit and transform the features into a purely numeric array
    X_processed = preprocessor.fit_transform(X_raw)
    
    # 3. Train / Test Split
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X_processed, y, sens_y, test_size=0.3, random_state=42
    )
    
    # 4. ART Data Formatting
    X_train_with_sens = np.column_stack((sens_train, X_train))
    
    # 5. Train the Target Model (Predicting PUBCOV)
    target_model = RandomForestClassifier(n_estimators=50, random_state=42)
    target_model.fit(X_train_with_sens, y_train) 
    
    # Wrap model in ART
    art_classifier = SklearnClassifier(model=target_model)
    
    # Get test predictions for the black-box attack
    X_test_with_sens = np.column_stack((sens_test, X_test))
    
    # Reshape the 1D prediction array into a 2D column vector for ART
    predictions_test = target_model.predict(X_test_with_sens).reshape(-1, 1)
    
    # # 6. Execute Baseline Attack (Control)
    # print("Training Baseline Attack Model...")
    # baseline_attack = AttributeInferenceBaseline(
    #     attack_model_type='rf', 
    #     attack_feature=0,      
    #     is_continuous=False    
    # )
    # baseline_attack.fit(x=X_train_with_sens) 
    
    # 7. Execute Black-Box Attack (Potent)
    print("Training Black-Box Attack Model...")
    blackbox_attack = AttributeInferenceBlackBox(
        estimator=art_classifier,
        attack_model_type='rf',
        attack_feature=0,
        is_continuous=False
    )
    blackbox_attack.fit(x=X_train_with_sens, y=y_train)
    
    # 8. Evaluate Attacks
    possible_sensitive_values = np.unique(sens_train).tolist()
    
    # infer_baseline = baseline_attack.infer(
    #     x=X_test, 
    #     values=possible_sensitive_values
    # )
    
    infer_blackbox = blackbox_attack.infer(
        x=X_test, 
        y=y_test, 
        pred=predictions_test, 
        values=possible_sensitive_values
    )
    
    # Calculate metrics
    # acc_baseline = accuracy_score(sens_test, infer_baseline)
    acc_blackbox = accuracy_score(sens_test, infer_blackbox)
    
    # print(f"Baseline Accuracy:  {acc_baseline:.4f}")
    print(f"Black-Box Accuracy: {acc_blackbox:.4f}")
    
    results.append({
        'Dataset': name,
        # 'Baseline_Accuracy': acc_baseline,
        'BlackBox_Accuracy': acc_blackbox 
    })

results_df = pd.DataFrame(results)
print("\n=== Final Privacy Leakage Results ===")
print(results_df)

plot_results(results_df)





