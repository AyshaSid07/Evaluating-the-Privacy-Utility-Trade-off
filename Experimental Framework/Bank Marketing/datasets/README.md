# Dataset Preparation 

This folder contains the dataset preparation script used to generate the raw Bank Marketing dataset for the experiments.

-------------------------------------------------------------------------

## `fetch_bank_marketing_data.py`

Loads and prepares the Bank Marketing dataset for the experiments.

The script:
- loads the Bank Marketing dataset
- converts the dataset into pandas DataFrame format
- prepares the dataset for anonymization experiments
- exports the dataset as `bank-additional-full.csv`

### Run

```bash
python fetch_bank_marketing_data.py
```

--------------------------------------------------------------------------

# REQUIREMENTS

```bash
pip install pandas
```

---------------------------------------------------------------------------

# OUTPUT

- `bank-additional-full.csv`