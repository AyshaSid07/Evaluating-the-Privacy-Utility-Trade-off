# ACS Public Coverage

This folder contains the experimental framework used for evaluating

privacy-preserving techniques on the ACS Public Coverage dataset.

------------------------------------------------------------------------

# Folder Structure

```text
ACS Public Coverage/

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
└── Readme.md
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
- `public_coverage_dp.py`

The `hierarchies` folder contains the hierarchy files used for the quasi-identifiers.

------------------------------------------------------------------------

# datasets

This folder contains dataset preparation scripts.

Main files:

- `fetch_public_coverage_data.py`
- `split_dataset.py`

Outputs:

- `folktables_public_coverage_RAW.csv`
- `folktables_public_coverage_train.csv`
- `folktables_public_coverage_test.csv`

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

- `utility_public_coverage.py`

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

## Step 1: Prepare Dataset

```bash
python fetch_public_coverage_data.py
```

## Step 2: Split Dataset

```bash
python split_dataset.py
```

## Step 3: Add Linkage Index

```bash
python add_index.py
```

## Step 4: Run Anonymization

```bash
javac -cp ".:../../libarx.jar" ARX_public_coverage.java
java -cp ".:../../libarx.jar" ARX_public_coverage.java
```

## Step 5: Clean Anonymized Data

```bash
python clean_arx_data.py
```

## Step 6: Generate Differentially Private Data

```bash
python public_coverage_dp.py
```

## Step 7: Run Risk Assessment

```bash
python attribute_inference_attack.py

python DCR.py

python netflix_linkage_attack.py
```

## Step 8: Run Utility Evaluation

```bash
python utility_public_coverage.py
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