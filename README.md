# Evaluating the Privacy-Utility Trade-off

This repository contains the experimental framework developed for evaluating privacy-preserving techniques on multiple real-world datasets.

The framework includes:
- anonymization
- differential privacy
- privacy risk assessment
- utility evaluation

------------------------------------------------------------------------

# Datasets Included

- ACS Employment
- ACS Income
- ACS Public Coverage
- Bank Marketing
- Credit Card Clients
- MEPS

------------------------------------------------------------------------

# Repository Structure

```text
Evaluating-the-Privacy-Utility-Trade-off/
│
├── ACS Employment/
├── ACS Income/
├── ACS Public Coverage/
├── Bank Marketing/
├── Credit Card Clients/
├── MEPS/
└── README.md
```

------------------------------------------------------------------------

# Installation Instructions

## 1. Clone Repository

```bash
git clone https://github.com/AyshaSid07/Evaluating-the-Privacy-Utility-Trade-off.git
cd Evaluating-the-Privacy-Utility-Trade-off
```

------------------------------------------------------------------------

# Requirements

## Python Dependencies

Install the required Python libraries:

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install folktables
pip install snsynth
pip install aif360
pip install adversarial-robustness-toolbox
```

------------------------------------------------------------------------

## Java Requirements

Install:
- Java JDK
- ARX Data Anonymization Framework

Check Java installation:

```bash
java -version
javac -version
```

ARX website:

```text
https://arx.deidentifier.org/
```

------------------------------------------------------------------------

# Setting Up ARX

## Step 1: Download ARX

Download the **Cross-platform Java Library (Including Dependencies)** from:

```text
https://arx.deidentifier.org/
```

## Step 2: Extract ARX

Extract the downloaded ARX package.

## Step 3: Add ARX Library

Rename the downloaded library to:

```text
libarx.jar
```

Place it in the repository root directory:

```text
Evaluating-the-Privacy-Utility-Trade-off/
│
├── libarx.jar
├── ACS Employment/
├── ACS Income/
├── ACS Public Coverage/
├── Bank Marketing/
├── Credit Card Clients/
├── MEPS/
└── README.md
```

## Step 4: Compile and Run ARX Programs

Navigate to the dataset's anonymization folder and compile the corresponding ARX Java file.

Example for ACS Employment:

### Compile

```bash
javac -cp "../../libarx.jar" ARX_employment.java
```

### Run

```bash
java -cp "../../libarx.jar" ARX_employment
```

To run another dataset, replace `ARX_employment` with the corresponding Java file name:

- ARX_income
- ARX_public_coverage
- ARX_bank_marketing
- ARX_credit_card_clients
- ARX_meps

## Step 5: Clean Anonymized Datasets

After the anonymization process completes, run:

```bash
python clean_arx_data.py
```

This removes fully suppressed records from the generated ARX datasets.

------------------------------------------------------------------------

# Dataset README Files

Each dataset folder contains a separate README file with:
- dataset-specific instructions
- execution steps
- anonymization details
- utility evaluation details
- privacy risk assessment details

------------------------------------------------------------------------

# How to Run Experiments

Go to a dataset folder:

```bash
cd "ACS Employment"
```

Then follow the instructions provided in:

```text
ACS Employment/README.md
```

------------------------------------------------------------------------

# Experimental Workflow

The workflow used in the experiments is:

1. Prepare dataset
2. Add linkage index
3. Apply anonymization using ARX
4. Clean anonymized datasets
5. Generate differentially private datasets
6. Perform privacy risk assessment
7. Evaluate utility
8. Store results

------------------------------------------------------------------------

# Results

The framework generates:
- utility evaluation results
- Attribute Inference Attack results
- linkage attack results
- DCR evaluation results

Results are stored inside the `results/` folder of each dataset.

------------------------------------------------------------------------
