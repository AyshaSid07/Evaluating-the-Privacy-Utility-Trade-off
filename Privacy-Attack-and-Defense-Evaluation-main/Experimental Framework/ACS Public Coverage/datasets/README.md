# Dataset Preparation 

This folder contains the dataset preparation script used to generate the raw ACS Public Coverage dataset for the experiments.

-------------------------------------------------------------------------

## `fetch_income_data.py`

Downloads and prepares the ACS Public Coverage dataset using the Folktables framework.

The script:
- loads ACS Public Coverage data
- converts the dataset into pandas DataFrame format
- combines features and labels
- exports the dataset as `folktables_public_coverage_RAW.csv`

### Run

```bash
python fetch_public_coverage_data.py
```

--------------------------------------------------------------------------

# REQUIREMENTS

```bash
pip install pandas folktables
```

---------------------------------------------------------------------------

# OUTPUT

- `folktables_public_coverage_RAW.csv`