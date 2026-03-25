import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("../datasets/newtelecom_dataset.csv")

# Quasi-identifiers
qi = ["MCC", "MNC", "LAI"]

# Group by QI to find uniqueness
groups = df.groupby(qi).size().reset_index(name="count")

# Records in groups of size 1 → uniquely re-identifiable
unique_records = groups[groups["count"] == 1]["count"].sum()
total_records = len(df)
uniqueness_percent = (unique_records / total_records) * 100

# Plot labels and values
labels = [f"Total Records\n(100%)", f"Re-identified\n({uniqueness_percent:.2f}%)"]
values = [total_records, unique_records]

plt.figure(figsize=(7, 5))

# Plot bars 
colors = ["#87CEEB", "#FA8072"]  
bars = plt.bar(labels, values, color=colors)

plt.title("Baseline Re-identification Risk (No Defense)", fontsize=12)
plt.ylabel("Number of Records")

# Annotate the count above each bar 
for i, v in enumerate(values):
    plt.text(i, v + (total_records * 0.015), str(v),
             ha='center', fontsize=10)

# Adjust Y-axis 
plt.ylim(0, total_records * 1.1)
plt.tight_layout()
plt.savefig("uniqueness_plot.png", dpi=300)
plt.show()