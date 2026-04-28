## Prediction code for "Towards Understanding Fairness and its Composition in Ensemble Machine Learning", by Gohar et al.

[Paper Source](https://ieeexplore.ieee.org/abstract/document/10172501)

[Original GitHub Repository](https://github.com/UsmanGohar/FairEnsemble)

[Bank Marketing Dataset Source](https://github.com/UsmanGohar/FairEnsemble/tree/main/BankMarketingNoteBook/Data)

#### Contains the dataset:
- **Bank Marketing:** Predicting whether a client will subscribe to a term deposit.

*(The Bank Marketing sensitive, real-world tabular which contains demographic protected attributes and quasi-identifiers such as Age, Job status, Marital Status, and Education level).*

## Adaptations and changes from the original code

Original Python Script: `4-subscription-prediction-for-bank-marketing-data.py` from the [FairEnsemble GitHub Repository](https://github.com/UsmanGohar/FairEnsemble/blob/main/BankMarketingNoteBook/Kernels/RF/4-subscription-prediction-for-bank-marketing-data.py)

#### Code Modifications for our Study:

We adapted the Bank Marketing preprocessing and model training directly from the FairEnsemble repository. To ensure our baseline matches their environment, we kept their specific duplicate-dropping logic, categorical encoding array, and test-train split configuration.

We made the following changes to fit our privacy-focused methodology:

1. **Removed Exploratory Data Analysis (EDA):** The original script contains extensive code for profiling the data, generating correlation matrices, and plotting Seaborn charts. We stripped out all visualization and profiling code to isolate the predictive model.
2. **Isolated the Baseline Model:** The original code trains and evaluates five different machine learning algorithms simultaneously (Logistic Regression, KNN, Decision Tree, Random Forest, and XGBoost). To prevent confounding variables when measuring utility drop, we commented out the other models and kept only the Random Forest classifier as our baseline metric.