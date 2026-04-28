## Prediction code for "Retiring Adult: New Datasets for Fair Machine Learning", by Ding et al.

[Paper Source](https://arxiv.org/pdf/2108.04884)

[Original Github Repository](https://github.com/socialfoundations/folktables)

Dataset Source: Can be fetched from the *Folktables* library

#### Contains the dataset:
- ACS Public Coverage: Predicting if someone has public health coverage.


*(The ACS Public Coverage dataset are derived from the real US Census American Community Survey and use demographic sensitive features like Age, Education, Sex, Race).*

---

## Adaptations and changes from the original code

Original Code: Quickstart tutorial from the [folktables GitHub Repository README](https://github.com/socialfoundations/folktables/blob/main/README.md)

#### Code Modifications for our Study:

We adapted the data ingestion and modeling pipeline directly from the official `folktables` documentation. To ensure our baseline matches the authors' intended usage, we kept the `ACSDataSource` API, the 80/20 train-test split, and their exact pipeline architecture (`StandardScaler` paired with `LogisticRegression`). This exact adaptation process was uniformly applied to all five ACS datasets.

We made the following changes to fit our privacy-focused methodology:

1. **Removed Fairness Metrics:** The original code evaluates algorithmic bias by calculating True Positive Rates (TPR) across demographic groups to find Equality of Opportunity violations. We removed these demographic fairness checks and used the model solely to measure baseline classification accuracy before anonymization.
2. **Data Extraction:** The original tutorial immediately converts the data into NumPy arrays for model training. To enable our privacy defense, we utilized the `df_to_pandas()` function to extract the human-readable features and labels, combined them into a single DataFrame, and exported the raw CSV file for the ARX privacy tool.