import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

TARGET_COLUMN = 'PUBCOV'
RANDOM_STATE = 123

df_real = pd.read_csv('../datasets/folktables_public_coverage_RAW.csv')
if 'Linkage_Index' in df_real.columns:
    df_real = df_real.drop(columns=['Linkage_Index'])
df_real = df_real[df_real[TARGET_COLUMN].astype(str) != '*'].copy()

FEATURE_COLS = [c for c in df_real.columns if c != TARGET_COLUMN]

y_real = df_real[TARGET_COLUMN].astype(int).values
X_real = df_real.drop(columns=[TARGET_COLUMN]).astype(str)

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.2, random_state=RANDOM_STATE
)

datasets_to_test = {
    "No anonymization (Baseline)": None,  
    "ARX Public Coverage, k = 3": pd.read_csv('../datasets/ARX_acs_public_coverage_k3.csv'),
    "ARX Public Coverage, k = 5": pd.read_csv('../datasets/ARX_acs_public_coverage_k5.csv'),
    "ARX Public Coverage, k = 10": pd.read_csv('../datasets/ARX_acs_public_coverage_k10.csv'),
    "ARX Public Coverage, k = 15": pd.read_csv('../datasets/ARX_acs_public_coverage_k15.csv'),
    "ARX Public Coverage, k = 5, l = 3": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_l3.csv'),
    "ARX Public Coverage, k = 5, l = 5": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_l5.csv'),
    "ARX Public Coverage, k = 5, t = 0.3": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_t0.3.csv'),
    "ARX Public Coverage, k = 5, t = 0.15": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_t0.15.csv'),
    "DP Public Coverage, epsilon = 10.0": pd.read_csv('../datasets/DP_public_coverage_epsilon_10_0.csv'),
    "DP Public Coverage, epsilon = 5.0":  pd.read_csv('../datasets/DP_public_coverage_epsilon_5_0.csv'),
    "DP Public Coverage, epsilon = 1.0":  pd.read_csv('../datasets/DP_public_coverage_epsilon_1_0.csv'),
    "DP Public Coverage, epsilon = 0.5":  pd.read_csv('../datasets/DP_public_coverage_epsilon_0_5.csv'),
    "DP Public Coverage, epsilon = 0.1":  pd.read_csv('../datasets/DP_public_coverage_epsilon_0_1.csv'),
    "Combined Public Coverage, k = 3 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_public_coverage.csv'),
    "Combined Public Coverage, k = 3 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_public_coverage.csv'),
    "Combined Public Coverage, k = 3 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_public_coverage.csv'),
    "Combined Public Coverage, k = 5 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_public_coverage.csv'),
    "Combined Public Coverage, k = 5 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_public_coverage.csv'),
    "Combined Public Coverage, k = 5 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_public_coverage.csv'),
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
        if 'Linkage_Index' in df.columns:
            df = df.set_index('Linkage_Index')
            surviving_train_indices = X_train_real.index.intersection(df.index)
            df_train = df.loc[surviving_train_indices].copy()
        else:
            # DP or combined: no row correspondence, use all generated rows
            df_train = df.reset_index(drop=True).copy()

        df_train = df_train[df_train[TARGET_COLUMN].astype(str) != '*'].copy()

        # Align column order to real data
        df_train = df_train[FEATURE_COLS + [TARGET_COLUMN]]

        y_train = df_train[TARGET_COLUMN].astype(int).values
        X_train = df_train.drop(columns=[TARGET_COLUMN]).astype(str)

    # Clean copy of real test set for each configuration
    X_test = X_test_real.copy().astype(str)

    model = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', LogisticRegression(max_iter=1000, n_jobs=-1, class_weight='balanced'))
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
results_df.to_csv('../results/utility/acs_public_coverage_utility_results.csv', index=False)