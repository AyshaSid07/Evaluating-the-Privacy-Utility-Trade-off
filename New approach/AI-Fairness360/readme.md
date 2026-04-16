## Prediction code for "AI Fairness 360: An Extensible Toolkit for Detecting and Mitigating Algorithmic Bias", by Bellamy et al.

Paper source: https://arxiv.org/abs/1810.01943

Original Github Repository: https://github.com/Trusted-AI/AIF360

#### Contains the dataset:
- **Medical Expenditure Panel Survey (MEPS) Panel 19:** Predicting high healthcare utilization, meaning their total number of medical events in a year is greater than or equal to 10 for the 2014-2015 panel.

#### Installation Instructions for MEPS Datasets:

AIF360 does not download the MEPS data automatically due to its size and licensing. To run the prediction code, you must download the data manually:

1. Follow the official instructions on the AIF360 GitHub: [AIF360 MEPS README](https://github.com/Trusted-AI/AIF360/blob/main/aif360/data/raw/meps/README.md), and get the 'h181' datafile in csv format. 
2. Place that `h181.csv` file exactly where AIF360 expects it on your computer. This is usually located inside your Python virtual environment folder at: `site-packages/aif360/data/raw/meps/`.
