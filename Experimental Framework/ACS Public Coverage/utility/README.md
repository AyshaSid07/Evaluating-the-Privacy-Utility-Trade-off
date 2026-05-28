# Utility Evaluation

This folder contains the utility evaluation implementation used in the ACS Coverage experiments.

The utility evaluation measures the predictive performance of privacy-preserving datasets using machine learning models.

------------------------------------------------------------------------

## `utility_coverage.py

Implements the utility evaluation pipeline.

The script:
- loads original and privacy-preserving datasets
- trains a Logistic Regression model
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
python utility_coevrage.py
```

------------------------------------------------------------------------

# REQUIREMENTS

```bash
pip install pandas scikit-learn numpy
```

------------------------------------------------------------------------

# OUTPUT

- `acs_coverage_utility_results.csv`