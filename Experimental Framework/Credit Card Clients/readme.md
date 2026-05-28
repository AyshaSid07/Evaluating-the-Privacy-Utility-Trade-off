# Credit Card Clients

This folder contains the experimental framework used for evaluating
privacy-preserving techniques on the Credit Card Clients dataset.

------------------------------------------------------------------------

# Folder Structure

```text
Credit Card Clients/
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
- `ARX_credit_card_clients.java`
- `clean_arx_data.py`
- `credit_card_clients_dp.py`

------------------------------------------------------------------------

# datasets

This folder contains dataset preparation scripts and raw datasets.

Main file:
- `bin_limit_bal.py`

Output:
- `credit_card_clients_binned.csv`

------------------------------------------------------------------------

# risk_assessment

This folder contains privacy risk evaluation methods such as:
- Attribute Inference Attack (AIA)
- Linkage Attack
- Distance to Closest Record (DCR)

Main files:
- `attribute_inference_attack.py`
- `DCR.py`
- `credit_card_clients_linkage_attack.py`

------------------------------------------------------------------------

# utility

This folder contains utility evaluation experiments for measuring the
performance of privacy-preserving datasets.

Main file:
- `utility_credit_card_clients.py`

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
python bin_limit_bal.py
```

## Step 2: Add Linkage Index

```bash
python add_index.py
```

## Step 3: Run Anonymization

```bash
javac ARX_credit_card_clients.java
java ARX_credit_card_clients
```

## Step 4: Clean Anonymized Data

```bash
python clean_arx_data.py
```

## Step 5: Generate Differentially Private Data

```bash
python credit_card_clients_dp.py
```

## Step 6: Run Risk Assessment

```bash
python attribute_inference_attack.py
python DCR.py
python credit_card_clients_linkage_attack.py
```

## Step 7: Run Utility Evaluation

```bash
python utility_credit_card_clients.py
```

------------------------------------------------------------------------

# Requirements

## Python

```bash
pip install pandas numpy scikit-learn snsynth adversarial-robustness-toolbox
```

## Java

- Java JDK
- ARX Data Anonymization Framework

ARX website:
https://arx.deidentifier.org/

------------------------------------------------------------------------