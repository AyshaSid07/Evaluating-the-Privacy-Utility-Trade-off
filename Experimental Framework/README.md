# Evaluating the Privacy-Utility Trade-off

This repository contains the experimental framework developed for evaluating
privacy-preserving techniques on multiple real-world datasets.

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

Install all required Python libraries:

```bash
pip install pandas numpy scikit-learn folktables snsynth aif360 adversarial-robustness-toolbox
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

Download the ARX Data Anonymization Tool from:

```text
https://arx.deidentifier.org/
```

## Step 2: Extract ARX

Extract the downloaded ARX package.

## Step 3: Add ARX JAR File

Add the ARX JAR file to your Java project classpath before compiling the Java programs.

### Compile

```bash
javac -cp arx.jar ARX_income.java
```

### Run (Windows)

```bash
java -cp ".;arx.jar" ARX_income
```

### Run (Linux/macOS)

```bash
java -cp ".:arx.jar" ARX_income
```

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
ACS Employment/readme.md
```

------------------------------------------------------------------------

# Running Experiments for Each Dataset

## ACS Employment

```bash
cd "ACS Employment"
```

Follow:

```text
ACS Employment/readme.md
```

------------------------------------------------------------------------

## ACS Income

```bash
cd "ACS Income"
```

Follow:

```text
ACS Income/readme.md
```

------------------------------------------------------------------------

## ACS Public Coverage

```bash
cd "ACS Public Coverage"
```

Follow:

```text
ACS Public Coverage/readme.md
```

------------------------------------------------------------------------

## Bank Marketing

```bash
cd "Bank Marketing"
```

Follow:

```text
Bank Marketing/readme.md
```

------------------------------------------------------------------------

## Credit Card Clients

```bash
cd "Credit Card Clients"
```

Follow:

```text
Credit Card Clients/readme.md
```

------------------------------------------------------------------------

## MEPS

```bash
cd "MEPS"
```

Follow:

```text
MEPS/readme.md
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