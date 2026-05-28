# Dataset Preparation 

This folder contains the dataset preparation script used to generate the raw Credit Card Clients dataset for the experiments.

-------------------------------------------------------------------------

## `bin_limit_bal.py`

Loads and prepares the Credit Card Clients dataset for the experiments.

The script:
- loads the Credit Card Clients dataset
- bins the `LIMIT_BAL` attribute into categorical ranges
- creates a new attribute called `LIMIT_CATEGORY`
- removes the original `LIMIT_BAL` column
- exports the processed dataset as `credit_card_clients_binned.csv`

### LIMIT_CATEGORY Bins
- 0 = 0 - 50,000
- 1 = 50,000 - 100,000
- 2 = 100,000 - 200,000
- 3 = above 200,000

### Run

```bash
python bin_limit_bal.py
```

--------------------------------------------------------------------------

# REQUIREMENTS

```bash
pip install pandas numpy
```

---------------------------------------------------------------------------

# OUTPUT

- `credit_card_clients_binned.csv`