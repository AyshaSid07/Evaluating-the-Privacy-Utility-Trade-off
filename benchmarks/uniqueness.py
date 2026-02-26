import pandas as pd
import matplotlib.pyplot as plt

def plot_results(results_dict):
    plt.figure(figsize=(10, 6)) 
    
    methods = list(results_dict.keys())
    uniqueness = list(results_dict.values())
    
    # Skapa staplarna
    for i in range(len(methods)):
        bars = plt.bar(methods[i], uniqueness[i], color=plt.cm.Set3(i), edgecolor='black')
        texts = plt.text(methods[i], uniqueness[i] + 1, f"{uniqueness[i]:.2f}%", ha='center', va='bottom', fontsize=10)
    plt.ylim(0, 100)
    plt.ylabel('Uniqueness (%)', fontsize=12)
    plt.xlabel('Defense Method', fontsize=12)
    plt.title('Uniqueness Across Defenses (Risk assessment)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
                 
    plt.tight_layout()
    plt.savefig("plots/uniqueness.png", dpi=300)
    plt.show()

if __name__ == "__main__":

    datasets_to_test = {
            "No Defense (Baseline)": pd.read_csv('../datasets/mendeley_data.csv'),
            "Generalization": pd.read_csv('../datasets/de-identified-datasets/generalization.csv'),
            "Masking": pd.read_csv('../datasets/de-identified-datasets/masking.csv'),
            "Masking and generalization" : pd.read_csv('../datasets/de-identified-datasets/generalization_and_masking.csv')
            # "Suppression" : pd.read_csv("../datasets/de-identified-datasets/suppressed_city.csv")

            # "v2": pd.read_csv('../datasets/anonymized_v2.csv'),
            # "v3": pd.read_csv('../datasets/anonymized_v3.csv')
        }

    # 2. Selecting specific quasi-identifiers
    QI_cols = ["City", "Region", "Country", "Postal Code", "ISP", "Organization", "Latitude", "Longitude", "Timezone", "Autonomous System"]

    print("Quasi-identifiers used:", QI_cols)

    final_results = {}

    for defense_name, df in datasets_to_test.items():
        group_sizes = df.groupby(QI_cols).size()

        unique_records = (group_sizes == 1).sum()
        total_records = len(df)

        percent_unique = (unique_records / total_records) * 100

        print(f"% Unique Records: {percent_unique:.2f}%")
        print(f"Unique Records: {unique_records}")
        print(f"Total Records: {total_records}")

        # 4. Re-identification success rate
        reid_success_rate = percent_unique

        print(f"Re-identification Success Rate: {reid_success_rate:.2f}%")
        final_results[defense_name] = reid_success_rate
    
    plot_results(final_results)

# plt.figure(figsize=(10,6))

# plt.title(f"Re-identification Risk (uniqueness) for {defense_name}")
# plt.ylabel("Number of Records")
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()

#     #plt.savefig(f"baseline_bar_chart_{defense_name}.png", dpi=300)
# plt.show()

# # 6. Save metrics
# metrics = pd.DataFrame({
#     "metric": ["total_records", "unique_records", "percent_unique", "reid_success_rate"],
#     "value": [total_records, unique_records, percent_unique, reid_success_rate]
# })

# metrics.to_csv("baseline_metrics.csv", index=False)
