## Prediction code for "An Experimental Study on Fairness-aware Machine Learning for Credit Scoring Problems", by Giang et al.

[Paper Source](https://arxiv.org/abs/2412.20298)

[Original GitHub Repository](https://github.com/tailequy/faircredit)

[Dataset Source](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

#### Contains one dataset:
- **Credit Card Clients:** Predicting whether a customer in Taiwan will face a default payment situation next month based on their financial history and demographic profile.

*(This dataset contains highly sensitive quasi-identifiers such as Sex, Education, and Marital status, paired with detailed financial billing records).*

---

## Adaptations and changes from the original code

Original Jupyter Notebook: `exp_credit_card.ipynb` from the [faircredit GitHub Repository](https://github.com/tailequy/faircredit/blob/main/exp_credit_card.ipynb)

#### Code Modifications for our Study:

We adapted the baseline methodology directly from the peer-reviewed `faircredit` repository. To ensure our baseline matches the authors' environment, we maintained their explicit responsive preprocessing logic, their 70/30 `train_test_split`, and strictly utilized the standard Scikit-Learn `DecisionTreeClassifier(random_state=0)`.

We made the following changes to fit our privacy-focused methodology:

1. **Isolation of Baseline Utility:** The original study heavily evaluates pre-processing (DIR, LFR) and post-processing fairness algorithms (EOP, CEP) to balance predictive accuracy with fairness measures. We stripped out all AIF360 fairness dependencies and ABROCA metric generators, utilizing only the traditional Decision Tree baseline to establish a pre-anonymization utility score.
2. **Model Bloat Reduction:** The original code loops through multiple algorithms simultaneously (Decision Tree, Naive Bayes, MLP, and kNN). To prevent confounding variables when measuring utility drop, we commented out the other algorithms and isolated the Decision Tree classifier as our sole baseline metric.
