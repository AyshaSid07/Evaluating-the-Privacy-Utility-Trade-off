# Dataset Preparation 

This folder contains the dataset preparation script used to generate the raw ACS Employment dataset for the experiments.

-------------------------------------------------------------------------

## `fetch_employment_data.py`

Downloads and prepares the ACS Employment dataset using the Folktables framework.

The script:
- loads ACS employment data
- converts the dataset into pandas DataFrame format
- combines features and labels
- exports the dataset as `folktables_employment_RAW.csv`

### Run

```bash
python fetch_employment_data.py
```

--------------------------------------------------------------------------

# REQUIREMENTS

```bash
pip install pandas folktables
```

---------------------------------------------------------------------------

# OUTPUT

- `folktables_employment_RAW.csv`