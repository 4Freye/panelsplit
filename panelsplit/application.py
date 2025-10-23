import inspect
from typing import Tuple, List, Union, Optional

import narwhals as nw
import numpy as np
from joblib import Parallel, delayed
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.base import clone, BaseEstimator

from .utils.utils import _split_wrapper
from .utils.validation import (
    _safe_indexing,
    _to_numpy_array,
    check_cv,
    check_fitted_estimators,
    _supports_sample_weights,
)


def _get_non_null_mask(data):
    """Get non-null mask for any data type."""
    return ~nw.from_native(data, series_only=True).is_null()


def _predict_split(model, X_test: IntoDataFrame, method: str = "predict") -> np.ndarray:
    """
    Perform predictions for a single split.

    Parameters
    ----------
    model : object
        The machine learning model used for prediction.
    X_test : IntoDataFrame
        The input features for testing.
    method : str, optional
        The method to use for prediction. It can be 'predict', 'predict_proba',
        or 'predict_log_proba'. Default is 'predict'.

    Returns
    -------
    np.ndarray
        Array containing predicted values.
    """
    predict_func = getattr(model, method)
    return predict_func(X_test)


def _fit_split(
    estimator,
    X: IntoDataFrame,
    y: Optional[IntoSeries],
    train_indices: np.ndarray,
    sample_weight: Optional[Union[IntoSeries, np.ndarray]] = None,
    drop_na_in_y: bool = False,
):
    """
    Fit a cloned estimator on the given training indices.

    Parameters
    ----------
    estimator : object
        The machine learning model to be fitted.
    X : IntoDataFrame
        The input features for the estimator.
    y : IntoSeries or None
        The target variable for the estimator.
    train_indices : np.ndarray
        Integer indices indicating the training data.
    sample_weight : IntoSeries or np.ndarray, optional
        Sample weights for the training data. Default is None.
    drop_na_in_y : bool, default=False
        Whether to drop rows with null values in y.

    Returns
    -------
    object
        A fitted estimator.
    """
    local_estimator = clone(estimator)

    # Use narwhals for dataframe-agnostic operations
    X_nw = nw.from_native(X, pass_through=True)

    # Use safe position-based indexing (train_indices are already integers)
    X_subset = _safe_indexing(X_nw, train_indices)

    # Handle y=None case (for transformers)
    if y is not None:
        y_nw = nw.from_native(y, pass_through=True)
        y_subset = _safe_indexing(y_nw, train_indices)

        if drop_na_in_y:
            # Get mask of non-null values
            non_null_mask = _get_non_null_mask(y_subset)
            non_null_indices = np.where(_to_numpy_array(non_null_mask))[0]

            # Filter both X and y using the non-null indices
            X_filtered = _safe_indexing(X_subset, non_null_indices)
            y_filtered = _safe_indexing(y_subset, non_null_indices)
        else:
            X_filtered = X_subset
            y_filtered = y_subset

        # Convert back to native format for sklearn
        X_native = _safe_indexing(X_filtered, to_native=True)
        y_native = _safe_indexing(y_filtered, to_native=True)
    else:
        # When y is None, we can't drop nulls based on y
        X_filtered = X_subset
        X_native = _safe_indexing(X_filtered, to_native=True)
        y_native = None

    if sample_weight is not None:
        sw_nw = nw.from_native(sample_weight, pass_through=True)
        sw_subset = _safe_indexing(sw_nw, train_indices)

        # Only filter by non_null_indices if y is not None and drop_na_in_y is True
        if y is not None and drop_na_in_y:
            sw_filtered = _safe_indexing(sw_subset, non_null_indices)
        else:
            sw_filtered = sw_subset

        sw_native = _safe_indexing(sw_filtered, to_native=True)

        # Check if the estimator supports sample_weight
        fit_signature = inspect.signature(local_estimator.fit)
        if "sample_weight" in fit_signature.parameters:
            return local_estimator.fit(X_native, y_native, sample_weight=sw_native)
        else:
            # Estimator doesn't support sample_weight, fit without it
            return local_estimator.fit(X_native, y_native)
    else:
        return local_estimator.fit(X_native, y_native)


def _prediction_order_to_original_order(indices: List[np.ndarray]) -> np.ndarray:
    """
    Convert the concatenated predictions back to the original order.

    Parameters
    ----------
    indices : List[np.ndarray]
        List of integer index arrays corresponding to the test/train splits.

    Returns
    -------
    np.ndarray
        Array of indices representing the sorted order to restore the original data order.
    """
    indices = np.concatenate(indices)
    return np.argsort(indices)


def cross_val_fit(
    estimator,
    X: IntoDataFrame,
    y: IntoSeries,
    cv,
    sample_weight: Optional[Union[IntoSeries, np.ndarray]] = None,
    n_jobs: int = 1,
    progress_bar: bool = False,
    drop_na_in_y: bool = False,
) -> List[BaseEstimator]:
    """
    Fit the estimator using cross-validation.

    Parameters
    ----------
    estimator : object
        The machine learning model to be fitted.
    X : IntoDataFrame
        The input features for the estimator.
    y : IntoSeries
        The target variable for the estimator.
    cv : object or iterable
        Cross-validation splitter; either an object that generates train/test splits (e.g., an instance of PanelSplit)
        or an iterable of splits.
    sample_weight : IntoSeries or np.ndarray, optional
        Sample weights for the training data. Default is None.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is 1.
    progress_bar : bool, optional
        Whether to display a progress bar. Default is False.
    drop_na_in_y : bool, optional
        Whether to drop observations where y is na. Default is False.

    Returns
    -------
    list
        List containing fitted models for each split.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from panelsplit.cross_validation import PanelSplit
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     'feature': [1, 2, 3, 4, 5, 6],
    ...     'period': [1, 1, 2, 2, 3, 3]
    ... })
    >>> X = df[['feature']]
    >>> y = pd.Series([2, 4, 6, 8, 10, 12])
    >>> # Create a PanelSplit instance for cross-validation
    >>> ps = PanelSplit(periods=df['period'], n_splits=2)
    >>> fitted_models = cross_val_fit(LinearRegression(), X, y, ps)
    >>> len(fitted_models)
    2
    """
    _supports_sample_weights(estimator, sample_weight)
    splits = check_cv(cv)

    fitted_estimators = Parallel(n_jobs=n_jobs)(
        delayed(_fit_split)(
            estimator, X, y, train_indices, sample_weight, drop_na_in_y=drop_na_in_y
        )
        for train_indices, _ in _split_wrapper(
            indices=splits, progress_bar=progress_bar
        )
    )

    return fitted_estimators


def cross_val_predict(
    fitted_estimators,
    X: IntoDataFrame,
    cv,
    method: str = "predict",
    n_jobs: int = 1,
    return_train_preds: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Perform cross-validated predictions using a given predictor model.

    Parameters
    ----------
    fitted_estimators : list
        List of fitted machine learning models used for prediction.
    X : IntoDataFrame
        The input features for prediction.
    cv : object or iterable
        Cross-validation splitter; either an object that generates train/test splits or an iterable of splits.
    method : str, optional
        The method to use for prediction. It can be whatever methods are available to the estimator
        (e.g. predict_proba in the case of a classifier or transform in the case of a transformer). Default is 'predict'.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is 1.
    return_train_preds : bool, optional
        If True, return predictions for the training set as well. Default is False.

    Returns
    -------
    test_preds : np.ndarray
        Array containing test predictions made by the model during cross-validation.
    train_preds : np.ndarray, optional
        Array containing train predictions made by the model during cross-validation.
        Returned only if `return_train_preds` is True.
    """
    check_fitted_estimators(fitted_estimators)
    splits = check_cv(cv)

    test_splits = [split[1] for split in splits]
    test_indices = _prediction_order_to_original_order(test_splits)

    # Use narwhals for dataframe-agnostic operations
    X_nw = nw.from_native(X, pass_through=True)

    test_preds = Parallel(n_jobs=n_jobs)(
        delayed(_predict_split)(
            fitted_estimators[i],
            _safe_indexing(X_nw, test_idx, to_native=True),
            method,
        )
        for i, test_idx in enumerate(test_splits)
    )

    if return_train_preds:
        train_splits = [split[0] for split in splits]
        train_indices = _prediction_order_to_original_order(train_splits)

        train_preds = Parallel(n_jobs=n_jobs)(
            delayed(_predict_split)(
                fitted_estimators[i],
                _safe_indexing(X_nw, train_idx, to_native=True),
                method,
            )
            for i, train_idx in enumerate(train_splits)
        )

        return np.concatenate(test_preds, axis=0)[test_indices], np.concatenate(
            train_preds, axis=0
        )[train_indices]
    else:
        return np.concatenate(test_preds, axis=0)[test_indices]


def cross_val_fit_predict(
    estimator,
    X: IntoDataFrame,
    y: IntoSeries,
    cv,
    method: str = "predict",
    sample_weight: Optional[Union[IntoSeries, np.ndarray]] = None,
    n_jobs: int = 1,
    return_train_preds: bool = False,
    drop_na_in_y=False,
) -> Union[
    Tuple[np.ndarray, List[BaseEstimator]],
    Tuple[np.ndarray, np.ndarray, List[BaseEstimator]],
]:
    """
    Fit the estimator using cross-validation and then make predictions.

    Parameters
    ----------
    estimator : object
        The machine learning model to be fitted.
    X : IntoDataFrame
        The input features for the estimator.
    y : IntoSeries
        The target variable for the estimator.
    cv : object
        Cross-validation splitter; an object that generates train/test splits.
     method : str, optional
        The method to use for prediction. It can be whatever methods are available to the estimator
        (e.g. predict_proba in the case of a classifier or transform in the case of a transformer). Default is 'predict'.
    sample_weight : IntoSeries or np.ndarray, optional
        Sample weights for the training data. Default is None.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is 1.
    return_train_preds : bool, optional
        If True, return predictions for the training set as well. Default is False.
    drop_na_in_y : bool, optional
        Whether to drop observations where y is na. Default is False.

    Returns
    -------
    tuple
        If `return_train_preds` is False, returns a tuple of:
            - preds (np.ndarray): Array containing predictions made by the model during cross-validation.
            - fitted_estimators (list): List containing fitted models for each split.
        If `return_train_preds` is True, returns a tuple of:
            - preds (np.ndarray): Array containing test predictions made by the model during cross-validation.
            - train_preds (np.ndarray): Array containing train predictions made by the model during cross-validation.
            - fitted_estimators (list): List containing fitted models for each split.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from panelsplit.cross_validation import PanelSplit  # assuming PanelSplit is imported from your module
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     'feature': [1, 2, 3, 4, 5, 6],
    ...     'period': [1, 1, 2, 2, 3, 3]
    ... })
    >>> X = df[['feature']]
    >>> y = pd.Series([2, 4, 6, 8, 10, 12])
    >>> # Create a PanelSplit instance for cross-validation
    >>> ps = PanelSplit(periods=df['period'], n_splits=2)
    >>> # Get test predictions and fitted models
    >>> preds, models = cross_val_fit_predict(LinearRegression(), X, y, ps)
    >>> preds.shape
    (2,)
    """

    fitted_estimators = cross_val_fit(
        estimator, X, y, cv, sample_weight, n_jobs, drop_na_in_y=drop_na_in_y
    )

    res = cross_val_predict(
        fitted_estimators, X, cv, method, n_jobs, return_train_preds
    )

    if return_train_preds:
        # res should be Tuple[np.ndarray, np.ndarray]
        if isinstance(res, tuple):
            preds, train_preds = res
        else:
            # defensive: unexpected type at runtime
            raise TypeError("cross_val_predict returned ndarray but expected tuple")
        return preds, train_preds, fitted_estimators
    else:
        # res should be np.ndarray
        if isinstance(res, tuple):
            preds = res[0]
        else:
            preds = res
        return preds, fitted_estimators
