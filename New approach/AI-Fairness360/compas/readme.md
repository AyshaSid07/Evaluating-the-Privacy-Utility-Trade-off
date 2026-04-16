## Adaptations and changes from the original code

Original Jupyter Notebook: `demo_mdss_classifier_metric_sklearn.ipynb` from the [AIF360 GitHub Repository](https://github.com/Trusted-AI/AIF360)

#### Code Modifications for our Study:

We adapted the COMPAS dataset setup and Logistic Regression model directly from the AIF360 tutorial. To ensure our baseline matches the original study, we kept their exact hyperparameters, data split sizes, and categorical encoding (`.cat.codes`).

We made the following changes to fit our privacy-focused methodology:

1. **Removed Fairness Metrics:** The original notebook uses the model to run a Multi-Dimensional Subset Scan (MDSS) for bias detection. We removed these fairness scans and used the model solely to measure baseline classification accuracy before anonymization.
2. **Data Extraction:** Because AIF360 loads data into custom objects, we used `.convert_to_dataframe()` to extract the raw Pandas DataFrame and export it as a CSV file