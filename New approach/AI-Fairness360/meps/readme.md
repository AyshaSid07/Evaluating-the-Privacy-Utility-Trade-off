## Adaptations and changes from the original code

Original Jupyter Notebook: `tutorial_medical_expenditure.ipynb` from the [AIF360 GitHub Repository](https://github.com/Trusted-AI/AIF360/blob/main/examples/tutorial_medical_expenditure.ipynb)

#### Code Modifications for our Study:

We adapted the MEPS dataset setup and Random Forest model directly from the AIF360 tutorial. To ensure our baseline matches the original study, we kept their specific data split `[0.5, 0.8]`, census `instance_weights`, and exact pipeline architecture (`StandardScaler` and `RandomForestClassifier` with 500 estimators).

We made the following changes to fit our privacy-focused methodology:

1. **Removed Bias Mitigation:** The original notebook focuses on algorithmic fairness, using techniques like Reweighing, Prejudice Removers, and LIME explainers. We removed these steps and used the model solely to measure baseline classification accuracy before anonymization.
2. **Data Extraction:** Because AIF360 loads data into custom objects, we used `.convert_to_dataframe()` to extract the raw Pandas DataFrame and export it as a CSV file