## Adaptations and changes from the original code

Original Python Script: `5-eda-to-prediction-dietanic.py` from the [FairEnsemble GitHub Repository](https://github.com/UsmanGohar/FairEnsemble/blob/main/Titanic/Kernels/RF/5-eda-to-prediction-dietanic.py)

#### Code Modifications for our Study:

We adapted the Titanic dataset preprocessing and model training directly from the FairEnsemble repository. To ensure our baseline exactly matches their environment, we kept their custom feature engineering (e.g., extracting Name titles to impute missing ages, calculating family size) and their stratified `train_test_split`.

We made the following changes to fit our privacy-focused methodology:

1. **Model Bloat Reduction:** The original script trains and evaluates seven different machine learning algorithms sequentially (including SVMs, Naive Bayes, and KNN). To prevent confounding variables when measuring utility drop, we isolated the standard Random Forest classifier (`n_estimators=100`) as our sole baseline metric.
