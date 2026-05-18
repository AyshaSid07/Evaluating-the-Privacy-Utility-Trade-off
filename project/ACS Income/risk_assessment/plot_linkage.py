import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_linkage_plot():
    # 1. Standardized Order based on your Dataset Summary Table
    datasets = [
        'ACS Income\n(195655 records)', 
        'ACS Public Coverage\n(138218 records)', 
        'ACS Employment\n(378818 records)', 
        'Bank Marketing\n(41189 records)', 
        'Credit Card Clients\n(30000 records)', 
        'MEPS\n(15931 records)'
    ]
    
    # Reordered data to match the standard
    data = {
        'Dataset': datasets,
        'Baseline': [95.487, 33.791, 31.842, 13.000, 2.667, 35.333],
        'k-anonymity (k=3)': [0.051, 0.722, 0.553, 3.250, 0.333, 5.333],
        'k-anonymity (k=5)': [0.000, 0.217, 0.263, 0.750, 0.000, 0.667],
        'DP (ε=1.0)': [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        'Combined (k=3 + ε=1.0)': [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    }
    
    df = pd.DataFrame(data)
    configs = ['Baseline', 'k-anonymity (k=3)', 'k-anonymity (k=5)', 'DP (ε=1.0)', 'Combined (k=3 + ε=1.0)']
    
    # 2. Plot Setup
    # Adjusted figsize to 14x7 to reduce empty horizontal space, helping LaTeX scale it better
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(df['Dataset'])) 
    width = 0.15 
    
    colors = [plt.cm.Set3(i) for i in range(len(configs))]
    
    # 3. Plotting Loop
    for i, config in enumerate(configs):
        offset = (i - 2) * width 
        bars = ax.bar(x + offset, df[config], width, label=config, 
                      color=colors[i], edgecolor='black', linewidth=1)
        
        for bar in bars:
            height = bar.get_height()
            y_pos = height + 2 
            text_str = f"{height:.2f}%"
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor=colors[i], edgecolor='black', linewidth=0.8, alpha=0.9)
            
            # INCREASED fontsize for the badges to 10
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    text_str, ha='center', va='bottom', fontsize=10, rotation=90, bbox=bbox_props, fontweight='bold') 
    
    # 4. Customization and Formatting
    # INCREASED label and title font sizes
    ax.set_ylabel('Re-identification Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Datasets', fontsize=14, fontweight='bold')
    ax.set_title('Netflix Record Linkage Attack Re-identification Success Rates Across Privacy Mechanisms', fontsize=16, fontweight='bold', pad=20)
    
    # INCREASED tick font sizes
    ax.set_xticks(x)
    ax.set_xticklabels(df['Dataset'], rotation=15, ha='right', fontsize=13)
    
    ax.set_ylim(0, 115)
    
    # THE FIX: Legend moved INSIDE the plot. Upper right is empty because the tallest bar is on the far left.
    ax.legend(title='Anonymization Configuration', loc='upper right', fontsize=12, title_fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("../plots/linkage/linkage_attack_master_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

generate_linkage_plot()