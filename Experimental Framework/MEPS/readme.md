# MEPS

This folder contains the experimental framework used for evaluating
privacy-preserving techniques on the MEPS dataset.

------------------------------------------------------------------------

# Folder Structure

```text
MEPS/
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
- `ARX_meps.java`
- `clean_arx_data.py`
- `meps_dp.py`

------------------------------------------------------------------------

# datasets

This folder contains dataset preparation scripts and raw datasets.

Main files:
- `fetch_meps.py`
- `reverse_one_hot.py`

Outputs:
- `MEPS19_RAW.csv`
- `MEPS.csv`

------------------------------------------------------------------------

# risk_assessment

This folder contains privacy risk evaluation methods such as:
- Attribute Inference Attack (AIA)
- Linkage Attack
- Distance to Closest Record (DCR)

Main files:
- `attribute_inference_attack.py`
- `DCR.py`
- `meps_linkage_attack.py`

------------------------------------------------------------------------

# utility

This folder contains utility evaluation experiments for measuring the
performance of privacy-preserving datasets.

Main file:
- `utility_meps.py`

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
python fetch_meps.py
python reverse_one_hot.py
```

## Step 2: Add Linkage Index

```bash
python add_index.py
```

## Step 3: Run Anonymization

```bash
javac ARX_meps.java
java ARX_meps
```

## Step 4: Clean Anonymized Data

```bash
python clean_arx_data.py
```

## Step 5: Generate Differentially Private Data

```bash
python meps_dp.py
```

## Step 6: Run Risk Assessment

```bash
python attribute_inference_attack.py
python DCR.py
python meps_linkage_attack.py
```

## Step 7: Run Utility Evaluation

```bash
python utility_meps.py
```

------------------------------------------------------------------------

# Requirements

## Python

```bash
pip install pandas numpy scikit-learn snsynth aif360 adversarial-robustness-toolbox
```

## Java

- Java JDK
- ARX Data Anonymization Framework

ARX website:
https://arx.deidentifier.org/

------------------------------------------------------------------------