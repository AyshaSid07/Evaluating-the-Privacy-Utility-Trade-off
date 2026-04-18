## Prediction code for "Recall-based Machine Learning approach for early detection of Cervical Cancer" by Gupta et al.

[Paper source](https://ieeexplore.ieee.org/abstract/document/9418099)

[Original Github Repository](https://github.com/mr-sesquipedalian/cervical_cancer_prediction)

[Dataset Source](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors)

#### Contains the dataset:
- Cervical Cancer (Risk Factors)
    - What we try to predict: The target variable is Biopsy (a binary 0 or 1). The model attempts to predict whether a patient will test positive for cervical cancer during a biopsy examination. (Note: The dataset also contains Hinselmann, Schiller, and Cytology test results, but Biopsy is the primary ground-truth target for the illness).

    - The Features: The prediction is based on highly sensitive behavioral and demographic quasi-identifiers, including age, smoking history, contraceptive use, number of pregnancies, and historical STD diagnoses.

## Execution steps:
First clean the raw data by running preprocess.py, then run the code by running cancer_prediction.py. 

## Adaptations and changes from the original code

Original Jupyter Notebook: `cervical_cancer_classifier_final.ipynb` from the [cervical_cancer_prediction GitHub Repository](https://github.com/mr-sesquipedalian/cervical_cancer_prediction)

#### Code Modifications for our Study:

We adapted the data cleaning process directly from the original notebook, explicitly dropping the heavily missing `STDs: Time since first/last diagnosis` columns and removing alternate medical target variables (like Schiller and Citology) to isolate the `Biopsy` prediction.

We made the following changes to fit our privacy-focused methodology:

1. **Replaced Deep Learning with Standard Baseline:** The original notebook utilizes a complex TensorFlow/Keras neural network architecture paired with Principal Component Analysis (PCA). To prevent confounding variables in our utility comparison, we replaced this with our standardized Random Forest classifier (`n_estimators=100`).
2. **Removed Synthetic Data (SMOTE):** The original authors used SMOTE to generate synthetic patient profiles to balance the dataset. Because our privacy defense ($k$-anonymity) evaluates the risk of re-identifying *real* human profiles, we removed SMOTE entirely. 