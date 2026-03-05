import matplotlib.pyplot as plt

# Data
methods = ["Original", "Defense 1 (k=2)", "Defense 2 (k=3)"]
records = [2732, 697, 1776]

# Create bar chart
plt.figure(figsize=(8,5))
plt.bar(methods, records)

plt.title("Data Retention After Applying Defenses")
plt.ylabel("Number of Records")
plt.xticks(rotation=20)

# Save graph
plt.tight_layout()
plt.savefig("defense_graph.png")

print("Graph saved successfully as defense_graph.png")
