import pandas as pd
import numpy as np
import warnings
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

def get_index_or_col_from_df(df, name):
    # Check if the DataFrame's index is a MultiIndex.
    is_multi = isinstance(df.index, pd.MultiIndex)
    
    # Check for existence in columns.
    col_exists = name in df.columns
    
    # Check for existence in the index.
    # For a MultiIndex, use df.index.names; for a regular index, wrap the name in a list.
    index_names = df.index.names if is_multi else [df.index.name]
    index_exists = name in index_names

    # When the name is found in both the index and the columns, warn and default to the index.
    if col_exists and index_exists:
        msg = (f"'{name}' is found in both the DataFrame's "
               f"{'MultiIndex levels' if is_multi else 'index'} and its columns. "
               "Defaulting to the index.")
        warnings.warn(msg)
        return df.index.get_level_values(name) if is_multi else df.index

    # When the name is only in the columns, return the column.
    elif col_exists:
        return df[name]

    # When the name is only in the index, return the index values.
    elif index_exists:
        return df.index.get_level_values(name) if is_multi else df.index

    # If the name is not found in either, raise an error.
    else:
        raise KeyError(f"'{name}' was not found in the DataFrame's columns or index names.")


def check_periods(periods, obj_name = 'periods'):
    if isinstance(periods, pd.Series):
        return periods
    elif isinstance(periods, pd.MultiIndex):
        raise ValueError(f'{obj_name} must be a level of an index. Got a pd.MultiIndex instead.')
    elif isinstance(periods, pd.Index):
        return pd.Series(periods)
    elif isinstance(periods, np.ndarray):
        if len(periods.shape) > 1:
            raise ValueError(f'{obj_name} array must be one-dimensional. Got an array of shape {periods.shape} instead')
        return pd.Series(periods)
    else:
        raise ValueError(f'{obj_name} type not supported.')
    
def check_labels(labels):
    if not isinstance(labels, (pd.Series, pd.DataFrame, pd.Index, np.ndarray)):
        raise TypeError(f"labels object type {type(labels)} not supported. labels must be a Pandas Series, DataFrame, Index, or NumPy array")
    
def check_method(fitted_estimators, method):
    for estimator in fitted_estimators:
        if not hasattr(estimator, method):
            raise ValueError(f"Invalid method provided. {estimator.__class__.__name__} does not have method {method}.")

def check_fitted_estimators(fitted_estimators):
    for estimator in fitted_estimators:
        try:
            check_is_fitted(estimator)
        except NotFittedError as exc:
            print(f"One or more of the estimators haven't been fitted yet. Please fit all estimators before using cross_val_predict.")
