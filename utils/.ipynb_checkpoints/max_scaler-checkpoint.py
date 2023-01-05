"""
Max scaler.
Author: JiaWei Jiang

This file contains the definition of max scaler, which divides all
values in dataset by the maximum, either global (i.e., across entities)
or within each entity (e.g., a sensor).
"""
from __future__ import annotations
from typing import Union

import numpy as np
import pandas as pd
from torch import Tensor


class MaxScaler:
    """Scale dataset values by dividing them by the maximum.

    The scaled value of a sample `x` is computed as:
        z = x / x_m

    where `x_m` is the maximum of all the samples and it's extracted
    independently from each feature if `local=True`. Or, `x_m` will be
    the maximum across all the features (e.g., time series, nodes).

    Parameters:
        local: whether maximum value is extracted within each series
            (i.e., nodes) or across all series (globally, see TPA-LSTM)

    Attributes:
        max_: ndarray, the maximum value for each feature, with shape
              (n_features, )
    """

    def __init__(self, local: bool = True):
        self._local = local

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> MaxScaler:
        """Compute the maximums to be used for later scaling.

        Parameters:
            X: ndarray or pd.DataFrame, data used to calculate maximum
               for later scaling, with shape (n_samples, n_features)

        Return:
            self: obj, fitted scaler
        """
        n_features = X.shape[1]
        if self._local:
            self.max_ = np.max(np.abs(X), axis=0)
        else:
            self.max_ = np.full(shape=(n_features,), fill_value=np.max(np.abs(X)))

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Perform scaling by dividing values by maximum.

        Parameters:
            X: data to scale, with shape (n_samples, n_features)

        Return:
            X_tr: transformed (scaled) array, with shape as input
        """
        if isinstance(X, pd.DataFrame):
            # Temporary workaround for np.ndarray in Union has no attr
            # values
            X_tr = X.values / self.max_
        else:
            X_tr = X / self.max_

        return X_tr

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute the maximums and scale the data."""
        self.fit(X)

        return self.transform(X)

    def inverse_transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Undo scaling of input data.

        It's commonly used when users want to get the prediction and
        groundtruths at the original scale.

        Parameters:
            X: input data to be inversely transformed, with shape
                (n_sampels, n_features)

        Return:
            Xt: inversely transformed data, with shape as input
        """
        Xt = X * self.max_

        return Xt
