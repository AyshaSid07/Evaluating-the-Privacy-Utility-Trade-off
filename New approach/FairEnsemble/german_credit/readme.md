## Adaptations and changes from the original code

Original Python Script: `19-credit-risk-classification-modeling-and-metrics.py` from the [FairEnsemble GitHub Repository](https://github.com/UsmanGohar/FairEnsemble/blob/main/GermanCredit/Kernels/RF/19-credit-risk-classification-modeling-and-metrics.py)

#### Code Modifications for our Study:

We adapted the German Credit preprocessing and model training directly from the FairEnsemble repository. To ensure our baseline exactly matches their environment, we kept their custom data imputation (filling missing accounts with "Others"), their specific 80/20 train-test split, their `StandardScaler` implementation, and their exact Random Forest hyperparameters (`max_depth=10`, `min_samples_split=5`, etc.).

We made the following changes to fit our privacy-focused methodology:

1. **Removed Exploratory Data Analysis (EDA):** The original script contains extensive code for generating Plotly and Seaborn visual graphs. We stripped out all visualization code to isolate the predictive model.
2. **Simplified Evaluation Metrics:** The original code calculates a wide array of metrics including ROC curves, AUC, sensitivity, and specificity. We reduced this to just the baseline accuracy score to keep our utility measurements standardized across all datasets.