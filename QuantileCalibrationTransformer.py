from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class QuantileCalibrationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, quantile=10):
        self.quantile = quantile

    def fit(self, X, y):
        """
        Fit the quantile calibration transformer.
        :param X: Array like which contains the predicted values.
        :param y: Array like which contains the ground truth values.
        :return: self
        """

        self.lookup_table_ = pd.Series(y).groupby(pd.qcut(X, self.quantile)).mean()

        return self

    def transform(self, X, y=None):
        return np.array([self._lookup(a) for a in X])

    def _lookup(self, val):
        if val >= self.lookup_table_.index[-1].right:
            return self.lookup_table_.iloc[-1]
        elif val <= self.lookup_table_.index[0].left:
            return self.lookup_table_.iloc[0]
        else:
            return self.lookup_table_[val]
