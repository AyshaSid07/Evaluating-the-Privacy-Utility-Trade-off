## Prediction code for "Recall-based Machine Learning approach for early detection of Cervical Cancer" by Gupta et al.

Paper source:  https://ieeexplore.ieee.org/abstract/document/9418099

Original Github Repository: https://github.com/mr-sesquipedalian/cervical_cancer_prediction/tree/master

Dataset Source: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors

#### Contains the dataset:
- Cervical Cancer (Risk Factors)
    - What we try to predict: The target variable is Biopsy (a binary 0 or 1). The model attempts to predict whether a patient will test positive for cervical cancer during a biopsy examination. (Note: The dataset also contains Hinselmann, Schiller, and Cytology test results, but Biopsy is the primary ground-truth target for the illness).

    - The Features: The prediction is based on highly sensitive behavioral and demographic quasi-identifiers, including age, smoking history, contraceptive use, number of pregnancies, and historical STD diagnoses.

## Execution steps:
First clean the raw data by running preprocess.py, then run the code by running cancer_prediction.py. 