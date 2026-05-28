# Dataset Preparation 

This folder contains the dataset preparation scripts used to generate the raw MEPS dataset for the experiments.

-------------------------------------------------------------------------

## `fetch_meps.py`

Downloads and prepares the MEPS dataset using the AIF360 framework.

The script:
- loads the MEPS Panel 19 dataset
- converts the dataset into pandas DataFrame format
- exports the dataset as `MEPS19_RAW.csv`

### Run

```bash
python fetch_meps.py
```

-------------------------------------------------------------------------

## `reverse_one_hot.py`

Processes the raw MEPS dataset by reversing one-hot encoded attributes.

The script:
- loads the raw MEPS dataset
- identifies one-hot encoded categorical attributes
- reconstructs categorical columns from one-hot encoded values
- removes the original one-hot encoded columns
- exports the processed dataset as `MEPS.csv`

### Run

```bash
python reverse_one_hot.py
```

--------------------------------------------------------------------------

# REQUIREMENTS

```bash
pip install pandas numpy aif360
```

---------------------------------------------------------------------------

# OUTPUT

- `MEPS19_RAW.csv`
- `MEPS.csv`