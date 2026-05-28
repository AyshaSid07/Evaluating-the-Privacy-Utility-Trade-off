## Prediction code for "AI Fairness 360: An Extensible Toolkit for Detecting and Mitigating Algorithmic Bias", by Bellamy et al.

[Paper Source](https://arxiv.org/abs/1810.01943)

[Original GitHub Repository](https://github.com/Trusted-AI/AIF360)

MEPS Dataset Source: See "Installation Instructions for MEPS Datasets" down below.

#### Contains the datasets:
- **Medical Expenditure Panel Survey (MEPS) Panel 19:** Predicting high healthcare utilization, meaning their total number of medical events in a year is greater than or equal to 10 for the 2014-2015 panel.


*(Privacy Risk: This dataset contain standard demographic quasi-identifiers like age, sex, and race. However, they link these demographics to highly sensitive real-world outcomes—chronic medical needs and financial healthcare burdens in MEP)*

#### Installation Instructions for MEPS Datasets:

AIF360 does not download the MEPS data automatically due to its size and licensing. To run the prediction code, you must download the data manually:

1. Follow the official instructions on the AIF360 GitHub: [AIF360 MEPS README](https://github.com/Trusted-AI/AIF360/blob/main/aif360/data/raw/meps/README.md), and get the 'h181' datafile in csv format. 
2. Place that `h181.csv` file exactly where AIF360 expects it on your computer. This is usually located inside your Python virtual environment folder at: `site-packages/aif360/data/raw/meps/`.

## Adaptations and changes from the original code

Original Jupyter Notebook: `tutorial_medical_expenditure.ipynb` from the [AIF360 GitHub Repository](https://github.com/Trusted-AI/AIF360/blob/main/examples/tutorial_medical_expenditure.ipynb)

#### Code Modifications for our Study:

We adapted the MEPS dataset setup and Random Forest model directly from the AIF360 tutorial. To ensure our baseline matches the original study, we kept their specific data split `[0.5, 0.8]`, census `instance_weights`, and exact pipeline architecture (`StandardScaler` and `RandomForestClassifier` with 500 estimators).

We made the following changes to fit our privacy-focused methodology:

1. **Removed Bias Mitigation:** The original notebook focuses on algorithmic fairness, using techniques like Reweighing, Prejudice Removers, and LIME explainers. We removed these steps and used the model solely to measure baseline classification accuracy before anonymization.
2. **Data Extraction:** Because AIF360 loads data into custom objects, we used `.convert_to_dataframe()` to extract the raw Pandas DataFrame and export it as a CSV file