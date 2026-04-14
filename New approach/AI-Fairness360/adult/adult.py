import pandas as pd
import seaborn as sns

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aif360.sklearn.datasets import fetch_adult, standardize_dataset
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error
from aif360.sklearn.metrics import base_rate, ratio

X, y, sample_weight = fetch_adult(binary_race=False)

X_num, y_num, _ = fetch_adult(dropna=False)

(X_train, X_test,
 y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=1234567)

import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone


class PandasMeta(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        output = self.estimator_.transform(X)
        if not isinstance(output, pd.DataFrame):
            output = pd.DataFrame(output, index=X.index)
            try:
                columns = self.estimator_.get_feature_names_out()
                output.columns = columns
            except:
                pass
        return output
    
    def fit_transform(self, X, y=None, **fit_params):
        assert isinstance(X, pd.DataFrame)
        self.estimator_ = clone(self.estimator)
        output = self.estimator_.fit_transform(X, y, **fit_params)
        if not isinstance(output, pd.DataFrame):
            output = pd.DataFrame(output, index=X.index)
            try:
                columns = self.estimator_.get_feature_names_out()
                output.columns = columns
            except:
                pass
        return output
    
pre = make_column_transformer(
    (OneHotEncoder(sparse_output=False), X_train.dtypes == 'category'),
    (StandardScaler(), X_train.dtypes != 'category'),
    verbose_feature_names_out=False)
pre = PandasMeta(pre)
pre.fit_transform(X_train)

di_train = disparate_impact_ratio(y_train, prot_attr='sex', priv_group='Male', pos_label='>50K')

ratio(base_rate, y_train, prot_attr='sex', priv_group='Male', pos_label='>50K')

model = LogisticRegression(max_iter=1000)
grid = GridSearchCV(model, param_grid={'C': [1, 10]})
pipe = make_pipeline(pre, grid)

y_pred = pipe.fit(X_train, y_train).predict(X_test)
print(accuracy_score(y_test, y_pred))
