import narwhals as nw
from narwhals.typing import IntoDataFrame, IntoSeries
import numpy as np
import warnings
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from collections.abc import Iterable

# Keep pandas import for fallback compatibility
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    _PANDAS_AVAILABLE = False


__pdoc__ = {
    'get_index_or_col_from_df': False,
    'check_cv': False,
    'check_periods':False,
    'check_labels':False,
    'check_fitted_estimators':False,
    'check_method':False,
    '_check_X_y':False
}

def get_index_or_col_from_df(df, name):
    """Get column or index from dataframe in a dataframe-agnostic way."""
    try:
        # Use narwhals for dataframe-agnostic operations
        df_nw = nw.from_native(df, pass_through=True)
        
        # Check if it's a column
        if hasattr(df_nw, 'columns') and name in df_nw.columns:
            from ..application import _safe_to_native
            # Use get_column instead of select().to_series()
            return _safe_to_native(df_nw.get_column(name))
        
        # For index operations, we still need pandas-specific logic as narwhals doesn't support index access
        # This is a limitation but necessary for backward compatibility
        if _PANDAS_AVAILABLE and hasattr(df, 'index'):
            # Check if the DataFrame's index is a MultiIndex.
            is_multi = hasattr(df.index, 'names') and len(df.index.names) > 1
            
            # Check for existence in the index.
            index_names = df.index.names if is_multi else [df.index.name]
            index_exists = name in index_names
            
            # Check for existence in columns.
            col_exists = hasattr(df, 'columns') and name in df.columns
            
            # When the name is found in both the index and the columns, warn and default to the index.
            if col_exists and index_exists:
                msg = (f"'{name}' is found in both the DataFrame's "
                       f"{'MultiIndex levels' if is_multi else 'index'} and its columns. "
                       "Defaulting to the index.")
                warnings.warn(msg)
                return df.index.get_level_values(name) if is_multi else df.index
            
            # When the name is only in the index, return the index values.
            elif index_exists:
                return df.index.get_level_values(name) if is_multi else df.index
        
        # If the name is not found, raise an error.
        raise KeyError(f"'{name}' was not found in the DataFrame's columns or index names.")
        
    except Exception as e:
        # Re-raise the original error if it's a KeyError we created
        if isinstance(e, KeyError):
            raise e
        # For other exceptions, try simple column access
        try:
            df_nw = nw.from_native(df, pass_through=True)
            from ..application import _safe_to_native
            return _safe_to_native(df_nw.get_column(name))
        except:
            raise KeyError(f"'{name}' was not found in the DataFrame's columns.")

def check_cv(cv, X= None, y =None, groups=None):
    if hasattr(cv, "split"):  # If cv is a class with split() method
        splits = cv.split(X=X, y=y, groups=groups)
    elif isinstance(cv, Iterable):  # If cv is an iterable
        splits = cv
    else:
        raise ValueError("cv should be a cross-validation splitter or an iterable of splits.")
    return splits

# def check_y(y):
#     if y.isna().any():
#         warnings.warn('y contains observations that are NA and will be dropped in order to fit the estimator.')
#     y_train = y.loc[train_indices].dropna()
#     return y_train

def check_periods(periods, obj_name = 'periods'):
    """Check and convert periods to a compatible format using narwhals."""
    try:
        # Try narwhals first for dataframe-agnostic operations
        periods_nw = nw.from_native(periods, pass_through=True)
        if hasattr(periods_nw, 'to_series'):
            return periods_nw.to_series()
        elif hasattr(periods_nw, 'to_numpy'):
            # Convert to numpy and create a series-like object
            periods_array = periods_nw.to_numpy()
            if len(periods_array.shape) > 1:
                raise ValueError(f'{obj_name} array must be one-dimensional. Got an array of shape {periods_array.shape} instead')
            return periods_array
        else:
            return periods
    except Exception:
        # Fallback to pandas-specific logic for compatibility
        import pandas as pd
        if isinstance(periods, pd.Series):
            return periods
        elif hasattr(periods, 'names') and len(getattr(periods, 'names', [])) > 1:
            raise ValueError(f'{obj_name} must be a level of an index. Got a MultiIndex instead.')
        elif hasattr(periods, '__iter__') and not isinstance(periods, str):
            # Convert iterable to numpy array first
            periods_array = np.array(periods)
            if len(periods_array.shape) > 1:
                raise ValueError(f'{obj_name} array must be one-dimensional. Got an array of shape {periods_array.shape} instead')
            return pd.Series(periods_array)
        else:
            raise ValueError(f'{obj_name} type not supported.')
    
def check_labels(labels):
    """Check if labels are in a supported format using narwhals."""
    try:
        # Try narwhals for dataframe-agnostic validation
        labels_nw = nw.from_native(labels, pass_through=True)
        if hasattr(labels_nw, 'to_numpy') or hasattr(labels_nw, 'to_series'):
            return  # Valid narwhals-compatible object
    except Exception:
        pass
    
    # Fallback to check specific types
    import pandas as pd
    if not isinstance(labels, (pd.Series, pd.DataFrame, pd.Index, np.ndarray)) and not hasattr(labels, '__array__'):
        raise TypeError(f"labels object type {type(labels)} not supported. labels must be a Series, DataFrame, Index, or array-like object")
    
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


def _check_X_y(X, y = None):
    """
    Validate that X and y are supported data types using narwhals.

    Parameters
    ----------
    X : array-like, IntoDataFrame, or IntoSeries
        Input features.
    y : array-like, IntoDataFrame, or IntoSeries
        Target values.

    Returns
    -------
    tuple
        The validated inputs (X, y).

    Raises
    ------
    TypeError
        If X or y is not a supported data type.
    """
    # Try narwhals validation first
    try:
        X_nw = nw.from_native(X, pass_through=True)
        if not (hasattr(X_nw, 'to_numpy') or hasattr(X_nw, 'shape') or isinstance(X, np.ndarray)):
            raise TypeError("X should be a dataframe, series, or array-like object")
    except Exception:
        # Fallback validation
        import pandas as pd
        valid_types = (np.ndarray, pd.DataFrame, pd.Series)
        if not isinstance(X, valid_types) and not hasattr(X, '__array__'):
            raise TypeError("X should be a numpy array, dataframe, series, or array-like object")
    
    if y is not None:
        try:
            y_nw = nw.from_native(y, pass_through=True)
            if not (hasattr(y_nw, 'to_numpy') or hasattr(y_nw, 'shape') or isinstance(y, np.ndarray)):
                raise TypeError("y should be a dataframe, series, or array-like object")
        except Exception:
            # Fallback validation
            import pandas as pd
            valid_types = (np.ndarray, pd.DataFrame, pd.Series)
            if not isinstance(y, valid_types) and not hasattr(y, '__array__'):
                raise TypeError("y should be a numpy array, dataframe, series, or array-like object")
    