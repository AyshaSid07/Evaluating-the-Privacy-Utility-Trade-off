# Evaluating the Privacy-Utility Trade-off: Benchmarking Privacy-Preserving Mechanisms Using an Experimental Framework

This repository contains the implementation, experiments, and evaluation framework developed for privacy-preserving data publishing using anonymization and differential privacy techniques.

The project evaluates the trade-off between:
- data utility
- privacy protection
- re-identification risk

across multiple real-world datasets.

------------------------------------------------------------------------

# Project Overview

The framework applies and evaluates:
- k-anonymity
- l-diversity
- t-closeness
- differential privacy
- combined anonymization + differential privacy approaches

The experiments include:
- utility evaluation
- attribute inference attacks
- linkage attacks
- distance-to-closest-record (DCR) analysis

------------------------------------------------------------------------

# Datasets

The repository contains experiments for the following datasets:

## 1. ACS Income
- Census-based income prediction dataset from Folktables

## 2. ACS Public Coverage
- Public health insurance coverage prediction dataset from Folktables

## 3. Bank Marketing
- Bank subscription prediction dataset

## 4. Credit Card Clients
- Credit default prediction dataset

## 5. MEPS
- Medical Expenditure Panel Survey dataset

------------------------------------------------------------------------

# Repository Structure

```text
project/
│
├── datasets/
│
├── ACS_Income/
│   ├── Datasets/
│   ├── Anonymization/
│   ├── Risk_Assessment/
│   ├── Utility/
│   └── Results/
│
├── ACS_Public_Coverage/
│   ├── Datasets/
│   ├── Anonymization/
│   ├── Risk_Assessment/
│   ├── Utility/
│   └── Results/
│
├── Bank_Marketing/
│   ├── Datasets/
│   ├── Anonymization/
│   ├── Risk_Assessment/
│   ├── Utility/
│   └── Results/
│
├── Credit_Card_Clients/
│   ├── Dataset/
│   ├── Anonymization/
│   ├── Risk_Assessment/
│   ├── Utility/
│   └── Results/
│
├── MEPS/
│   ├── Dataset/
│   ├── Anonymization/
│   ├── Risk_Assessment/
│   ├── Utility/
│   └── Results/
│
└── README.md
```

------------------------------------------------------------------------

# Privacy Models

## k-Anonymity

Protects against identity disclosure by ensuring that each record is indistinguishable from at least k−1 other records.

## l-Diversity

Extends k-anonymity by ensuring diversity in sensitive attribute values within equivalence classes.

## t-Closeness

Limits the distance between sensitive attribute distributions in equivalence classes and the full dataset distribution.

## Differential Privacy

Generates synthetic datasets with formal privacy guarantees using the SmartNoise Synthesizer framework.

------------------------------------------------------------------------

# Risk Assessment Methods

## Attribute Inference Attack (AIA)

Evaluates whether sensitive attributes can be inferred from protected datasets using black-box machine learning attacks.

### Metrics
- accuracy
- balanced accuracy
- precision
- recall
- F1-score

------------------------------------------------------------------------

## Linkage Attack

Evaluates re-identification risk by attempting to match attacker records with protected records using quasi-identifiers.

### Metric
- re-identification success rate

------------------------------------------------------------------------

## Distance to Closest Record (DCR)

Measures similarity between original and protected datasets using nearest-neighbor distances.

### Metrics
- mean distance
- median distance
- nearest-neighbor similarity
- exact match percentage
- near match percentage

------------------------------------------------------------------------

# Utility Evaluation

Utility evaluation measures how well machine learning models trained on privacy-preserving datasets perform on real test data.

### Metrics
- accuracy
- precision
- recall
- F1-score

### Models Used
- Logistic Regression
- Random Forest

------------------------------------------------------------------------

# Technologies Used

## Python Libraries

```bash
pip install pandas numpy scikit-learn snsynth folktables aif360 adversarial-robustness-toolbox
```

## Java
- Java JDK
- ARX Data Anonymization Framework

ARX website:

https://arx.deidentifier.org/

------------------------------------------------------------------------

# Differential Privacy

Differential privacy experiments use:
- SmartNoise Synthesizer
- MST (Maximum Spanning Tree) synthesizer

The experiments evaluate multiple epsilon values to analyze the privacy–utility trade-off.

------------------------------------------------------------------------

# Experimental Workflow

1. Dataset preparation
2. Anonymization using ARX
3. Differential privacy synthetic data generation
4. Utility evaluation
5. Privacy risk assessment
6. Result analysis

------------------------------------------------------------------------

# Output Files

The framework generates:
- anonymized datasets
- differentially private datasets
- combined privacy-preserving datasets
- utility evaluation results
- attack evaluation results
- DCR evaluation results

------------------------------------------------------------------------

# Research Goal

The primary goal of this project is to evaluate the effectiveness of privacy-preserving techniques in balancing:
- data utility
- privacy protection
- resistance to inference and linkage attacks

across multiple tabular datasets.