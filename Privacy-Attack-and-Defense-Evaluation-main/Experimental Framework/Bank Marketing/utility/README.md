# Utility Evaluation

This folder contains the utility evaluation implementation used in the Bank Marketing experiments.

The utility evaluation measures the predictive performance of privacy-preserving datasets using machine learning models.

------------------------------------------------------------------------

## `utility_bank_marketing.py

Implements the utility evaluation pipeline.

The script:
- loads original and privacy-preserving datasets
- trains a Random Forest model
- evaluates predictive performance on real test data
- compares anonymized, differential privacy, and combined datasets
- stores utility evaluation results

### Evaluation Metrics
- accuracy
- F1-score
- precision
- recall

### Run

```bash
python utility_bank_marketing.py
```

------------------------------------------------------------------------

# REQUIREMENTS

```bash
pip install pandas scikit-learn numpy
```

------------------------------------------------------------------------

# OUTPUT

- `bank_marketing_utility_results.csv`