import numpy as np
import pandas as pd
from aif360.datasets import MEPSDataset19
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

np.random.seed(1)

(dataset_orig_panel19_train,
 dataset_orig_panel19_val,
 dataset_orig_panel19_test) = MEPSDataset19().split([0.5, 0.8], shuffle=True)

meps_df, _ = dataset_orig_panel19_train.convert_to_dataframe()
meps_df.to_csv("MEPS19_RAW.csv", index=False)


dataset = dataset_orig_panel19_train
model = make_pipeline(StandardScaler(),
                      RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}
rf_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

dataset_test = dataset_orig_panel19_test
y_test_pred = model.predict(dataset_test.features)

print(f"Accuracy Score: {accuracy_score(dataset_test.labels.ravel(), y_test_pred) * 100:.2f}%")