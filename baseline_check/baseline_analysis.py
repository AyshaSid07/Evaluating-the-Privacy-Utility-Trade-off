import pandas as pd
import matplotlib.pyplot as plt

# 1. Load raw dataset
df = pd.read_csv("data/mendeley_data.csv")

# 2. Selecting specific quasi-identifiers
QI_cols = ["City", "Postal Code", "ISP", "Organization"]

print("Quasi-identifiers used:", QI_cols)

# 3. Compute uniqueness
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

# 5. Graph 1: Total vs Re-identified
plt.figure(figsize=(6,4))
plt.bar(["Total Records", "Re-identified"],
        [total_records, unique_records],
        color=["gray", "red"])

plt.title("Baseline Re-Identification Risk (No Defense)")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig("baseline_bar_chart.png", dpi=300)
plt.show()

# 6. Save metrics
metrics = pd.DataFrame({
    "metric": ["total_records", "unique_records", "percent_unique", "reid_success_rate"],
    "value": [total_records, unique_records, percent_unique, reid_success_rate]
})

metrics.to_csv("baseline_metrics.csv", index=False)
