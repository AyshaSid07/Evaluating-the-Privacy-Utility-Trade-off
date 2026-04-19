## Prediction code for "Comparative Study on the Performance of Categorical Variable Encoders in Classification and Regression Tasks", by Zhu et al.

[Paper Source](https://arxiv.org/abs/2401.09682)

[Original GitHub Repository](https://github.com/QiuRunwen/CategoryEncoderComparison)

[Employee Salaries Dataset Source (OpenML ID: 42125)](https://www.openml.org/t/34165)

[Student Performance Dataset Source (UCI)](https://archive.ics.uci.edu/dataset/320/student+performance)

[Cholesterol Dataset Source (OpenML ID: 204)](https://www.openml.org/d/204)

[Autism Screening Dataset Source (UCI)](https://archive.ics.uci.edu/dataset/426/autism+screening+adult)

[Churn Dataset Source (IBM / Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

#### Contains five datasets:
- **Employee Salaries:** Predicting the exact annual salary of public-sector employees based on their job title, department, and tenure (Regression).
- **Student Performance:** Predicting a high school student's final grade without using previous test scores, relying instead on family background and lifestyle (Regression).
- **Cholesterol:** Predicting a patient's serum cholesterol level based on demographic and resting clinical metrics (Regression).
- **Autism Screening Adult:** Predicting whether an adult has Autistic Spectrum Disorder (ASD) based on demographic profiles and app-based screening metrics (Classification).
- **Churn:** Predicting whether a telecommunications customer will cancel their service based on their demographic profile and contract status (Classification).

*(All datasets are highly sensitive, real-world tabular benchmarks used to evaluate machine learning utility. They represent extreme privacy risks in the domains of employment/finance, pediatric lifestyle, medical diagnostics, and corporate billing. They contain prominent demographic quasi-identifiers such as Age, Sex, Address, Job Title, and Family Size, making them prime candidates for evaluating linkage attacks and k-anonymity frameworks).*

---

#### Methodology & Code Attribution

To ensure a scientifically rigorous and peer-reviewed baseline for our privacy experiments, the modeling pipelines in these scripts are directly adapted from the original authors' codebase. 

Specifically, the following logic was extracted and preserved from the [CategoryEncoderComparison](https://github.com/QiuRunwen/CategoryEncoderComparison) repository:

1. **Data Loading & Feature Definition (`src/data/*.py` & `src/datasets.py`)**
   - The logic for loading each dataset, mapping the target variables (`y_col`), and categorizing numerical versus categorical features (`get_num_cat_cols`) was taken directly from their respective source scripts (e.g., `src/data/autism.py`, `src/data/employee_salaries.py`).
   - Their exact `util.py` script was utilized to dynamically identify and drop useless columns (such as IDs or single-value columns) via the `find_useless_colum` and `drop_useless` functions.

2. **Preprocessing Pipeline (`src/preprocess.py`)**
   - The imputation, scaling, and encoding steps were strictly copied from their `split_impute_encode_scale` function. 
   - This includes mapping categorical variables through `SimpleImputer(strategy="most_frequent")` followed by `OrdinalEncoder` (to establish the unconstrained baseline), and numerical variables through `SimpleImputer(strategy="median")` followed by `StandardScaler()`. 

3. **Model Architecture & Evaluation (`src/models.py`)**
   - The model instantiation (`RandomForestClassifier` and `RandomForestRegressor`) was extracted from their `train_model` function.
   - The utility evaluation metrics—Accuracy for classification tasks, and MSE, RMSE, MAE, and $R^2$ for regression tasks—were extracted identically from their `evaluate` function, aiming for an exact baseline comparison for later privacy analysis.