# Dataset Preparation 

This folder contains the dataset preparation script used to generate the raw ACS Income dataset for the experiments.

-------------------------------------------------------------------------

## `fetch_income_data.py`

Downloads and prepares the ACS Income dataset using the Folktables framework.

The script:
- loads ACS income data
- converts the dataset into pandas DataFrame format
- combines features and labels
- exports the dataset as `folktables_income_RAW.csv`

### Run

```bash
python fetch_income_data.py
```

--------------------------------------------------------------------------

# REQUIREMENTS

```bash
pip install pandas folktables
```

---------------------------------------------------------------------------

# OUTPUT

- `folktables_income_RAW.csv`