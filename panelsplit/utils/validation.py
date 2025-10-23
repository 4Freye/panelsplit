import warnings
import inspect
import importlib

from collections.abc import Iterable

import narwhals as nw
import numpy as np
from narwhals.dependencies import (
    is_numpy_array,
    is_pandas_dataframe,
    is_pandas_series,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


# Keep pandas import for fallback compatibility
def _get_pandas():
    try:
        return importlib.import_module("pandas")
    except ImportError:
        return None


pd = _get_pandas()
_PANDAS_AVAILABLE = False if pd is None else True


def _safe_indexing(obj, indices=None, to_native=False):
    """
    Unified safe indexing and conversion function for dataframe-agnostic operations.

    Parameters
    ----------
    obj : pandas.DataFrame/Series or narwhals-compliant object
        The object to index and/or convert
    indices : array-like, optional
        Integer positions to select. If None, no indexing is performed
    to_native : bool, optional
        Whether to convert to native format. Default is False

    Returns
    -------
    obj : same type as input or native format
        Processed object (indexed and/or converted)
    """
    # Handle indexing if indices provided
    if indices is not None:
        if is_numpy_array(obj):
            result = obj[indices]
        elif hasattr(obj, "iloc"):
            result = obj.iloc[indices]
        else:
            result = obj[indices]
    else:
        result = obj

    # Handle narwhals conversion
    if to_native and (
        hasattr(result, "_compliant_frame") or hasattr(result, "_compliant_series")
    ):
        return nw.to_native(result)
    return result


def _to_numpy_array(data):
    """Convert any data structure to numpy array using narwhals."""
    if is_numpy_array(data):
        return data

    # Use narwhals to handle conversion
    try:
        return nw.from_native(data, pass_through=True).to_numpy()
    except Exception:
        # Final fallback
        return np.array(data)


def _is_valid_data_type(data, data_name="data"):
    """Unified data type validation leveraging narwhals dependencies."""
    # Direct numpy array check
    if is_numpy_array(data):
        return True

    # Narwhals-compatible check
    try:
        data_nw = nw.from_native(data, pass_through=True)
        return hasattr(data_nw, "to_numpy") or hasattr(data_nw, "shape")
    except Exception:
        # Final fallback for array-like objects
        return hasattr(data, "__array__") or hasattr(data, "__iter__")


def _supports_sample_weights(estimator, sample_weight=None):
    """
    Check whether an estimator supports sample weights in its fit method.
    Issues a warning if not supported.

    Parameters
    ----------
    estimator : object
        A scikit-learn style estimator class or instance.

    Returns
    -------
    bool
        True if the estimator supports sample_weight, False otherwise.
    """
    if sample_weight is None:
        return True

    try:
        fit_signature = inspect.signature(estimator.fit)
        if "sample_weight" not in fit_signature.parameters:
            warnings.warn(
                f"Estimator {estimator.__class__.__name__} does not support sample_weight. sample_weight will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            return False
    except (TypeError, ValueError):
        warnings.warn(
            f"Estimator {estimator.__class__.__name__} has no inspectable fit signature; "
            "cannot verify sample_weight support.",
            UserWarning,
            stacklevel=2,
        )
        return False

    return True


__pdoc__ = {
    "get_index_or_col_from_df": False,
    "check_cv": False,
    "check_periods": False,
    "check_labels": False,
    "check_fitted_estimators": False,
    "check_method": False,
    "_check_X_y": False,
}


def get_index_or_col_from_df(df, name):
    """Get column or index from dataframe in a dataframe-agnostic way."""
    try:
        # Use narwhals for dataframe-agnostic operations
        df_nw = nw.from_native(df, pass_through=True)

        # Check if it's a column
        col_exists = hasattr(df_nw, "columns") and name in df_nw.columns

        # For index operations, we need pandas-specific logic as narwhals doesn't support index access
        index_exists = False
        is_multi = False
        if _PANDAS_AVAILABLE and hasattr(df, "index"):
            # Check if the DataFrame's index is a MultiIndex.
            is_multi = hasattr(df.index, "names") and len(df.index.names) > 1
            # Check for existence in the index.
            index_names = df.index.names if is_multi else [df.index.name]
            index_exists = name in index_names

        # When the name is found in both the index and the columns, warn and default to the index.
        if col_exists and index_exists:
            msg = (
                f"'{name}' is found in both the DataFrame's "
                f"{'MultiIndex levels' if is_multi else 'index'} and its columns. "
                "Defaulting to the index."
            )
            warnings.warn(msg)
            return df.index.get_level_values(name) if is_multi else df.index

        # When the name is only in the index, return the index values.
        elif index_exists:
            return df.index.get_level_values(name) if is_multi else df.index

        # When the name is only in columns, return the column
        elif col_exists:
            column = df_nw.get_column(name)
            return (
                nw.to_native(column) if hasattr(column, "_compliant_series") else column
            )

        # If the name is not found, raise an error.
        raise KeyError(
            f"'{name}' was not found in the DataFrame's columns or index names."
        )

    except KeyError as e:
        # Re-raise the original error if it's a KeyError we created
        raise KeyError(f"'{name}' was not found in the DataFrame's columns. {e}")


def check_cv(cv, X=None, y=None, groups=None):
    if hasattr(cv, "split"):  # If cv is a class with split() method
        splits = cv.split(X=X, y=y, groups=groups)
    elif isinstance(cv, Iterable):  # If cv is an iterable
        splits = cv
    else:
        raise ValueError(
            "cv should be a cross-validation splitter or an iterable of splits."
        )
    return splits


# def check_y(y):
#     if y.isna().any():
#         warnings.warn('y contains observations that are NA and will be dropped in order to fit the estimator.')
#     y_train = y.loc[train_indices].dropna()
#     return y_train


def check_periods(periods, obj_name="periods"):
    """Check and convert periods to a compatible format using narwhals."""
    try:
        # Try narwhals first for dataframe-agnostic operations
        periods_nw = nw.from_native(periods, pass_through=True)
        if hasattr(periods_nw, "to_series"):
            return periods_nw.to_series()
        elif hasattr(periods_nw, "to_numpy"):
            # Convert to numpy and create a series-like object
            periods_array = periods_nw.to_numpy()
            if len(periods_array.shape) > 1:
                raise ValueError(
                    f"{obj_name} array must be one-dimensional. Got an array of shape {periods_array.shape} instead"
                )
            return periods_array
        else:
            return periods
    except Exception:
        # Fallback to pandas-specific logic for compatibility
        import pandas as pd

        if is_pandas_series(periods):
            return periods
        elif hasattr(periods, "names") and len(getattr(periods, "names", [])) > 1:
            raise ValueError(
                f"{obj_name} must be a level of an index. Got a MultiIndex instead."
            )
        elif hasattr(periods, "__iter__") and not isinstance(periods, str):
            # Convert iterable to numpy array first
            periods_array = np.array(periods)
            if len(periods_array.shape) > 1:
                raise ValueError(
                    f"{obj_name} array must be one-dimensional. Got an array of shape {periods_array.shape} instead"
                )
            return pd.Series(periods_array)
        else:
            raise ValueError(f"{obj_name} type not supported.")


def check_labels(labels):
    """Check if labels are in a supported format using narwhals."""
    try:
        # Try narwhals for dataframe-agnostic validation
        labels_nw = nw.from_native(labels, pass_through=True)
        if hasattr(labels_nw, "to_numpy") or hasattr(labels_nw, "to_series"):
            return  # Valid narwhals-compatible object
    except Exception:
        pass

    # Fallback to check specific types
    if not (
        is_pandas_series(labels)
        or is_pandas_dataframe(labels)
        or is_numpy_array(labels)
        or (hasattr(labels, "index") and hasattr(labels, "names"))  # pandas Index check
        or hasattr(labels, "__array__")
    ):
        raise TypeError(
            f"labels object type {type(labels)} not supported. labels must be a Series, DataFrame, Index, or array-like object"
        )


def check_method(fitted_estimators, method):
    for estimator in fitted_estimators:
        if not hasattr(estimator, method):
            raise ValueError(
                f"Invalid method provided. {estimator.__class__.__name__} does not have method {method}."
            )


def check_fitted_estimators(fitted_estimators):
    for estimator in fitted_estimators:
        try:
            check_is_fitted(estimator)
        except NotFittedError:
            print(
                "One or more of the estimators haven't been fitted yet. Please fit all estimators before using cross_val_predict."
            )


def _handle_data_input(data):
    """Handle input data with narwhals-first approach."""
    # Quick numpy check
    if is_numpy_array(data):
        return data, "numpy"

    # Try narwhals conversion
    try:
        data_nw = nw.from_native(data, pass_through=True)
        return data_nw, "narwhals"
    except Exception:
        # For unsupported types, try to make array-like
        if hasattr(data, "__array__"):
            return np.asarray(data), "numpy"
        raise TypeError(f"Unsupported data type: {type(data)}")


def _check_X_y(X, y=None):
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
    if not _is_valid_data_type(X, "X"):
        raise TypeError("X should be a dataframe, series, or array-like object")

    if y is not None and not _is_valid_data_type(y, "y"):
        raise TypeError("y should be a dataframe, series, or array-like object")
