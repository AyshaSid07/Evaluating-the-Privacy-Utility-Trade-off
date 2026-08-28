# Bank Marketing

This folder contains the experimental framework used for evaluating 

privacy-preserving techniques on the Bank Marketing dataset.

------------------------------------------------------------------------

# Folder Structure

```text
Bank Marketing/

│
├── anonymization/
│   └── hierarchies/
├── datasets/
├── results/
│   ├── aia/
│   ├── dcr/
│   ├── linkage/
│   └── utility/
├── risk_assessment/
├── utility/
└── readme.md
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
- `ARX_bank_marketing.java`
- `clean_arx_data.py`
- `bank_marketing_dp.py`

The `hierarchies` folder contains the hierarchy files used for the quasi-identifiers.

------------------------------------------------------------------------

# datasets

This folder contains dataset preparation scripts and the original dataset.

Main files:

- `bank-additional-full.csv`
- `split_dataset.py`

------------------------------------------------------------------------

# risk_assessment

This folder contains privacy risk evaluation methods

- Attribute Inference Attack (AIA)
- Distance to Closest Record (DCR)
- Netflix Linkage Attack

Main files:

- `attribute_inference_attack.py`
- `DCR.py`
- `netflix_linkage_attack.py`

------------------------------------------------------------------------

# utility

This folder contains utility evaluation experiments for measuring the performance of privacy-preserving datasets.

Main file:

- `utility_bank_marketing.py`

Evaluation metrics:

- accuracy
- balanced accuracy
- ROC-AUC
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

## Step 1: Split Dataset

```bash
python split_dataset.py
```

## Step 2: Add Linkage Index

```bash
python add_index.py
```

## Step 3: Run Anonymization

```bash
javac -cp ".:../../libarx.jar" ARX_bank_marketing.java
java -cp ".:../../libarx.jar" ARX_bank_marketing.java
```

## Step 4: Clean Anonymized Data

```bash
python clean_arx_data.py
```

## Step 5: Generate Differentially Private Data

```bash
python bank_marketing_dp.py
```

## Step 6: Run Risk Assessment

```bash
python attribute_inference_attack.py

python DCR.py

python netflix_linkage_attack.py
```

## Step 7: Run Utility Evaluation

```bash
python utility_bank_marketing.py
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

[https://arx.deidentifier.org/]

------------------------------------------------------------------------