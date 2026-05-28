# ACS Public Coverage

This folder contains the experimental framework used for evaluating
privacy-preserving techniques on the ACS Public Coverage dataset.

------------------------------------------------------------------------

# Folder Structure

```text
ACS Public Coverage/
│
├── anonymization/
├── datasets/
├── results/
├── risk_assessment/
├── utility/
└── README.md
```

------------------------------------------------------------------------

# anonymization

This folder contains implementations for:
- k-anonymity
- l-diversity
- t-closeness
- differential privacy

Main files:
- `add_index.py`
- `ARX_public_coverage.java`
- `clean_arx_data.py`
- `coverage_dp.py`

------------------------------------------------------------------------

# datasets

This folder contains dataset preparation scripts and raw datasets.

Main file:
- `fetch_public_coverage_data.py`

Output:
- `folktables_public_coverage_RAW.csv`

------------------------------------------------------------------------

# risk_assessment

This folder contains privacy risk evaluation methods such as:
- Attribute Inference Attack (AIA)
- Linkage Attack
- Distance to Closest Record (DCR)

Main files:
- `attribute_inference_attack.py`
- `DCR.py`
- `income_linkage_attack.py`

------------------------------------------------------------------------

# utility

This folder contains utility evaluation experiments for measuring the
performance of privacy-preserving datasets.

Main file:
- `utility_coverage.py`

Evaluation metrics:
- accuracy
- precision
- recall
- F1-score

------------------------------------------------------------------------

# results

This folder contains:
- utility evaluation results
- AIA results
- DCR results
- linkage attack results

------------------------------------------------------------------------

# How to Run

## Step 1: Prepare Dataset

```bash
python fetch_public_coverage_data.py
```

## Step 2: Add Linkage Index

```bash
python add_index.py
```

## Step 3: Run Anonymization

```bash
javac ARX_public_coverage.java
java ARX_public_coverage
```

## Step 4: Clean Anonymized Data

```bash
python clean_arx_data.py
```

## Step 5: Generate Differentially Private Data

```bash
python coverage_dp.py
```

## Step 6: Run Risk Assessment

```bash
python attribute_inference_attack.py
python DCR.py
python income_linkage_attack.py
```

## Step 7: Run Utility Evaluation

```bash
python utility_coverage.py
```

------------------------------------------------------------------------

# Requirements

## Python

```bash
pip install pandas numpy scikit-learn folktables snsynth adversarial-robustness-toolbox
```

## Java

- Java JDK
- ARX Data Anonymization Framework

ARX website:
https://arx.deidentifier.org/

------------------------------------------------------------------------