from aif360.datasets import MEPSDataset19

dataset_orig_panel19 = MEPSDataset19()

meps_full_df, _ = dataset_orig_panel19.convert_to_dataframe()

meps_full_df.to_csv("MEPS19_RAW.csv", index=False)