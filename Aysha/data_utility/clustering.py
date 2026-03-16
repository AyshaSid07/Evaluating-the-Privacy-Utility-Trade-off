import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Direct Paths (Relative to data_utility folder)
RAW_PATH = "../datasets/telecomb_dataset.csv"
MASKED_PATH = "../defense/masked_dataset.csv"
SWAPPED_PATH = "../defense/swapped_dataset.csv"
SUPPRESSED_PATH = "../defense/suppressed_dataset.csv"

def get_utility(file_path, original_labels=None):
    # Load the file
    df = pd.read_csv(file_path)
    
    # Features used to identify user patterns
    features = ["MNC", "TMSI", "LMSI", "TLLI", "LAI", "RAI"]
    
    # Dynamically select only existing columns (handles Suppression)
    existing_features = [c for c in features if c in df.columns]
    X = df[existing_features].copy()
    
    # Preprocessing: Convert strings to numbers (Fixed for Pandas 4 warnings)
    for col in X.select_dtypes(include=['object', 'string']):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # Scale data for K-Means
    X_scaled = StandardScaler().fit_transform(X)
    
    # Clustering (K=50 matches your data generation logic)
    kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    if original_labels is None:
        return labels
    
    # Return the similarity score (ARS)
    return adjusted_rand_score(original_labels, labels)

# --- EXECUTION ---
print("--- Generating Utility Report ---")

# 1. Get Baseline
raw_labels = get_utility(RAW_PATH)

# 2. Get Scores
masked_score = get_utility(MASKED_PATH, raw_labels)
swapped_score = get_utility(SWAPPED_PATH, raw_labels)
suppressed_score = get_utility(SUPPRESSED_PATH, raw_labels)

# 3. Create Results DataFrame for Plotting
results_df = pd.DataFrame({
    "Method": ["Masked", "Swapped", "Suppressed"],
    "Utility_Score": [masked_score, swapped_score, suppressed_score]
})

print("\n--- Final Results ---")
print(results_df)

# 4. Generate the Plot
plt.figure(figsize=(10, 6))
colors = ['#3498db', '#e74c3c', '#2ecc71'] # Blue, Red, Green
bars = plt.bar(results_df["Method"], results_df["Utility_Score"], color=colors)

# Add baseline line
plt.axhline(y=1.0, color='black', linestyle='--', label='Original Data (100%)')

# Add text labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2%}', ha='center', va='bottom', fontweight='bold')

plt.title("Data Utility Retention Post-Anonymization", fontsize=14)
plt.ylabel("Adjusted Rand Score (Clustering Quality)", fontsize=12)
plt.ylim(0, 1.15)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)

# Save the plot
plt.savefig("utility_comparison_plot.png")
print("\nSuccess: Plot saved as 'utility_comparison_plot.png'")