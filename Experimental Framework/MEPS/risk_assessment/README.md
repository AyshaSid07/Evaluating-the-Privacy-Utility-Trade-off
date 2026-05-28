# Risk_Assessment 

This folder contains the risk assessment implementation used to evaluate privacy risks in the MEPS dataset.

The evaluation includes:
- Attribute Inference Attack (AIA)
- Distance to Closest Record (DCR)
- Linkage Attack

------------------------------------------------------------------------

## `attribute_inference_attack.py`

Implements the Attribute Inference Attack (AIA) evaluation.

The script:
- loads original and privacy-preserving datasets
- preprocesses categorical and numerical attributes
- trains a target machine learning model
- performs black-box attribute inference attacks
- evaluates attack performance
- stores the attack evaluation results

### Evaluation Metrics
- accuracy
- balanced accuracy
- precision
- recall
- F1-score

### Run

```bash
python attribute_inference_attack.py
```

------------------------------------------------------------------------

## `DCR.py`

Implements the Distance to Closest Record (DCR) evaluation.

The script:
- preprocesses original and protected datasets
- computes nearest-neighbor distances between records
- measures similarity between original and protected data
- stores DCR evaluation results

### Run

```bash
python DCR.py
```

------------------------------------------------------------------------

## `meps_linkage_attack.py`

Implements the linkage attack evaluation.

The script:
- compares attacker records against protected datasets
- performs weighted record matching using quasi-identifiers
- evaluates re-identification risk
- stores linkage attack evaluation results

### Run

```bash
python meps_linkage_attack.py
```

------------------------------------------------------------------------

# REQUIREMENTS

```bash
pip install pandas numpy scikit-learn adversarial-robustness-toolbox
```

------------------------------------------------------------------------

# OUTPUT

The scripts generate:
- `meps_aia_results.csv`
- `dcr_results_meps.csv`
- `linkage_attack_results_meps.csv`