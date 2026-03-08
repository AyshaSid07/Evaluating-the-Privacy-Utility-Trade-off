import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../raw_dataset.csv")

#Quasi-identifiers
qi = ["LAI","RAI","Network_Type"]

#grouping data
groups = df.groupby(qi).size().reset_index(name="count")

unique_records = groups[groups["count"] == 1]["count"].sum()
total_records = len(df)

#percentage calculation
reid_percentage = (unique_records / total_records) * 100
labels = ["Total Records\n(100%)", f"Re-identified\n({reid_percentage:.2f}%)"]
values = [total_records, unique_records]

plt.figure(figsize=(6,4))
colors = ["skyblue", "salmon"]
plt.bar(labels, values, color=colors)
plt.title("Baseline Re-identification Risk (No Defense)")
plt.ylabel("Number of Records")

# Adding record numbers above bars
for i, v in enumerate(values):
    plt.text(i, v + 50, str(v), ha="center")

# Save graph
plt.savefig("baseline_risk_graph.png", bbox_inches="tight")

print("Graph saved as baseline_risk_graph.png")