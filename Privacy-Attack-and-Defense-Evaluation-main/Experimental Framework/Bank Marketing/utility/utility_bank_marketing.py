import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Target attribute used for prediction 
TARGET_COLUMN = 'y'
RANDOM_STATE = 123

#Load original dataset
df_real = pd.read_csv('../datasets/bank-additional-full.csv', sep=';')
# Remove linkage index column if present
if 'Linkage_Index' in df_real.columns:
    df_real = df_real.drop(columns=['Linkage_Index'])
# Remove suppressed target values
df_real = df_real[df_real[TARGET_COLUMN].astype(str) != '*'].copy()

if 'duration' in df_real.columns:
    df_real = df_real.drop(columns=['duration'])
target_map = {'yes': 1, 'no': 0, '1': 1, '0': 0}
df_real[TARGET_COLUMN] = df_real[TARGET_COLUMN].astype(str).str.lower().map(target_map)

df_real = df_real.dropna(subset=[TARGET_COLUMN]) 
FEATURE_COLS = [c for c in df_real.columns if c != TARGET_COLUMN]
# Separate features and target attribute
y_real = df_real[TARGET_COLUMN].astype(int).values
X_real = df_real.drop(columns=[TARGET_COLUMN]).astype(str)

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.2, random_state=RANDOM_STATE
)
# Load datasets for utility evaluation
datasets_to_test = {
    "No anonymization (Baseline)": None,  
    "ARX Bank Marketing, k = 3": pd.read_csv('../datasets/ARX_bank_marketing_k3.csv'),
    "ARX Bank Marketing, k = 5": pd.read_csv('../datasets/ARX_bank_marketing_k5.csv'),
    "ARX Bank Marketing, k = 10": pd.read_csv('../datasets/ARX_bank_marketing_k10.csv'),
    "ARX Bank Marketing, k = 15": pd.read_csv('../datasets/ARX_bank_marketing_k15.csv'),
    "ARX Bank Marketing, k = 5, l = 2": pd.read_csv('../datasets/ARX_bank_marketing_k5_l2.csv'),
    "ARX Bank Marketing, k = 5, l = 3": pd.read_csv('../datasets/ARX_bank_marketing_k5_l3.csv'),
    "ARX Bank Marketing, k = 5, t = 0.3": pd.read_csv('../datasets/ARX_bank_marketing_k5_t0.3.csv'),
    "ARX Bank Marketing, k = 5, t = 0.15": pd.read_csv('../datasets/ARX_bank_marketing_k5_t0.15.csv'),
    "DP Bank Marketing, epsilon = 10.0": pd.read_csv('../datasets/DP_bank_marketing_epsilon_10_0.csv'),
    "DP Bank Marketing, epsilon = 5.0":  pd.read_csv('../datasets/DP_bank_marketing_epsilon_5_0.csv'),
    "DP Bank Marketing, epsilon = 3.0":  pd.read_csv('../datasets/DP_bank_marketing_epsilon_3_0.csv'),
    "DP Bank Marketing, epsilon = 1.0":  pd.read_csv('../datasets/DP_bank_marketing_epsilon_1_0.csv'),
    "DP Bank Marketing, epsilon = 0.5":  pd.read_csv('../datasets/DP_bank_marketing_epsilon_0_5.csv'),
    "DP Bank Marketing, epsilon = 0.1":  pd.read_csv('../datasets/DP_bank_marketing_epsilon_0_1.csv'),
    "Combined Bank Marketing, k = 3 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_bank_marketing.csv'),
    "Combined Bank Marketing, k = 3 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_bank_marketing.csv'),
    "Combined Bank Marketing, k = 3 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_bank_marketing.csv'),
    "Combined Bank Marketing, k = 5 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_bank_marketing.csv'),
    "Combined Bank Marketing, k = 5 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_bank_marketing.csv'),
    "Combined Bank Marketing, k = 5 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_bank_marketing.csv'),
}

results = []

for name, df in datasets_to_test.items():
    print(f"Evaluating: {name}...")
    
    # Baseline: train directly on the real training split
    if df is None:
        X_train = X_train_real.copy().astype(str)
        y_train = y_train_real
    else:
        # ARX datasets: align surviving rows with the real training split
        if 'duration' in df.columns:
            df = df.drop(columns=['duration'])
        if 'Linkage_Index' in df.columns:
            df = df.set_index('Linkage_Index')
            surviving_train_indices = X_train_real.index.intersection(df.index)
            df_train = df.loc[surviving_train_indices].copy()
        else:
             # DP or combined: no row correspondence, use all generated rows
            df_train = df.reset_index(drop=True).copy()

        df_train = df_train[df_train[TARGET_COLUMN].astype(str) != '*'].copy()

        df_train[TARGET_COLUMN] = (
            df_train[TARGET_COLUMN]
            .astype(str)
            .str.lower()
            .map(target_map)
        )

        df_train = df_train.dropna(subset=[TARGET_COLUMN])
        df_train = df_train[FEATURE_COLS + [TARGET_COLUMN]]

        y_train = df_train[TARGET_COLUMN].astype(int).values
        X_train = df_train.drop(columns=[TARGET_COLUMN]).astype(str)
    
    # Clean copy of real test set for each configuration
    X_test = X_test_real.copy().astype(str)

    model = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ('classifier', RandomForestClassifier(n_jobs=-1,random_state=2,class_weight='balanced'))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        'Dataset': name,
        'Accuracy': accuracy_score(y_test_real, y_pred),
        'F1-Score': f1_score(y_test_real, y_pred, zero_division=0),
        'Precision': precision_score(y_test_real, y_pred, zero_division=0),
        'Recall': recall_score(y_test_real, y_pred, zero_division=0)
    })

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("../results/utility/bank_marketing_utility_results.csv", index=False)