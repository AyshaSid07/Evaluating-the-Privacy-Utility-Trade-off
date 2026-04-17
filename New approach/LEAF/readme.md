## Prediction code for "To trust or not to trust an explanation: using LEAF to evaluate local linear XAI methods" by Amparore et al.

[Paper source](https://peerj.com/articles/cs-479/)

[Original Github Repository](https://github.com/amparore/leaf)

[Dataset Source](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)

#### Contains the dataset:
- **Heart Risk (Framingham):** Predicting a patient's 10-year risk of developing coronary heart disease (CHD) based on their demographic profile, behavioral habits, and medical history.

*(This dataset originates from the real-world Framingham Heart Study. It contains highly sensitive demographic quasi-identifiers such as age, gender, education, and BMI, paired with extreme privacy-sensitive medical diagnostics like stroke history, diabetes, and blood pressure.)*

*The paper does not state that they use this dataset, but rather the "heart risk problem" dataset, which is equal to the Framingham dataset.*
---

## Adaptations and changes from the original code

Original Jupyter Notebook: `LEAF_test.ipynb` from the [LEAF GitHub Repository](https://github.com/amparore/leaf/blob/master/LEAF_test.ipynb)

The code for `leaf.py` can also be found in the  [LEAF GitHub Repository](https://github.com/amparore/leaf/blob/master/leaf.py)

#### The Role of `leaf.py` in Our Methodology:
To ensure our baseline utility perfectly mirrors the original authors' mathematical methodology, we integrated their custom `leaf.py` script directly into our pipeline. Specifically, we utilize their `leaf.train_model()` function to establish our baseline. This function enforces the authors' exact experimental environment by:
1. Performing an automatic, hardcoded 80/20 `train_test_split` (strictly using `random_state=1234`).
2. Automatically balancing the dataset by calculating and applying `sample_weights` during the model fitting phase.
3. Generating the strictly isolated test-set accuracy and classification report, entirely preventing data leakage in our baseline utility metrics.

#### Code Modifications for our Study:
While we retained the `leaf` training architecture to ensure baseline validity, we made the following changes to adapt the code for our privacy-focused evaluation:

1. **Isolation of Baseline Utility (XAI Removal):** The primary purpose of the original repository is to use predictive models to evaluate the stability of Local Linear Explanations (LIME and SHAP) via their custom `LEAF` class. We completely bypassed the `LEAF` explanation generation and metric plotting, utilizing the `LogisticRegression` baseline strictly to capture the pre-anonymization predictive utility score.
3. **Missing Value Management:** Because real-world medical data frequently contains null values (e.g., missing glucose or BMI readings), we retained the authors' standard mean imputation logic (`hr_X.fillna(hr_X.mean())`) prior to model training so the Scikit-Learn baseline could execute successfully.
4. **Target Normalization:** We explicitly renamed the dataset's native `TenYearCHD` column to `TARGET` to successfully interface with the `leaf.py` variable extraction logic.