import pandas as pd
from folktables import ACSDataSource, ACSPublicCoverage

# Load the ACS data for California and prepare the features, labels, and group information
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA"], download=True)
features, label, group = ACSPublicCoverage.df_to_numpy(acs_data)

# Save the raw data to a CSV file
al_features, al_labels, _ = ACSPublicCoverage.df_to_pandas(acs_data)
combined_df = pd.concat([al_features, al_labels], axis=1)
combined_df.to_csv('folktables_public_coverage_RAW.csv', index=False)