import numpy as np
import narwhals as nw
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.base import clone
from joblib import Parallel, delayed
from .utils.utils import _split_wrapper
from .utils.validation import check_method, check_fitted_estimators, check_cv, _to_numpy_array
from typing import List, Union


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
        result = obj.iloc[indices] if hasattr(obj, "iloc") else obj[indices]
    else:
        result = obj
    
    # Handle conversion if needed
    if to_native and (hasattr(result, "_compliant_frame") or hasattr(result, "_compliant_series")):
        return nw.to_native(result)
    return result


def _get_non_null_mask(data):
    """Get non-null mask for any data type."""
    for method in ['isnull', 'is_null']:
        if hasattr(data, method):
            return ~getattr(data, method)()
    # Fallback for numpy arrays
    import pandas as pd
    return ~pd.isnull(data)




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
    y: IntoSeries,
    train_indices: List[bool],
    sample_weight: Union[IntoSeries, np.ndarray] = None,
    drop_na_in_y=False,
):
    """
    Fit a cloned estimator on the given training indices.

    Parameters
    ----------
    estimator : object
        The machine learning model to be fitted.
    X : IntoDataFrame
        The input features for the estimator.
    y : IntoSeries
        The target variable for the estimator.
    train_indices : list of bool
        Boolean mask or indices indicating the training data.
    sample_weight : IntoSeries or np.ndarray, optional
        Sample weights for the training data. Default is None.

    Returns
    -------
    object
        A fitted estimator.
    """
    local_estimator = clone(estimator)

    # Use narwhals for dataframe-agnostic operations
    X_nw = nw.from_native(X, pass_through=True)

    # Convert boolean indices to row numbers for narwhals
    train_row_indices = np.where(train_indices)[0]

    # Use safe position-based indexing
    X_subset = _safe_indexing(X_nw, train_row_indices)

    # Handle y=None case (for transformers)
    if y is not None:
        y_nw = nw.from_native(y, pass_through=True)
        y_subset = _safe_indexing(y_nw, train_row_indices)

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
        sw_subset = _safe_indexing(sw_nw, train_row_indices)

        # Only filter by non_null_indices if y is not None and drop_na_in_y is True
        if y is not None and drop_na_in_y:
            sw_filtered = _safe_indexing(sw_subset, non_null_indices)
        else:
            sw_filtered = sw_subset

        sw_native = _safe_indexing(sw_filtered, to_native=True)

        # Check if the estimator supports sample_weight
        import inspect

        fit_signature = inspect.signature(local_estimator.fit)
        if "sample_weight" in fit_signature.parameters:
            return local_estimator.fit(X_native, y_native, sample_weight=sw_native)
        else:
            # Estimator doesn't support sample_weight, fit without it
            return local_estimator.fit(X_native, y_native)
    else:
        return local_estimator.fit(X_native, y_native)


def _prediction_order_to_original_order(indices: List[bool]) -> List[int]:
    """
    Convert the concatenated predictions back to the original order.

    Parameters
    ----------
    indices : list of array-like
        List of boolean arrays or index arrays corresponding to the test/train splits.

    Returns
    -------
    np.ndarray
        Array of indices representing the sorted order to restore the original data order.
    """
    indices = np.concatenate([np.where(indices_)[0] for indices_ in indices])
    return np.argsort(indices)


def cross_val_fit(
    estimator,
    X: IntoDataFrame,
    y: IntoSeries,
    cv,
    sample_weight: Union[IntoSeries, np.ndarray] = None,
    n_jobs: int = 1,
    progress_bar: bool = False,
    drop_na_in_y=False,
):
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
) -> np.ndarray:
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
            _safe_indexing(X_nw, np.where(test_idx)[0], to_native=True),
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
                _safe_indexing(X_nw, np.where(train_idx)[0], to_native=True),
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
    sample_weight: Union[IntoSeries, np.ndarray] = None,
    n_jobs: int = 1,
    return_train_preds: bool = False,
    drop_na_in_y=False,
) -> np.ndarray:
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

    if return_train_preds:
        preds, train_preds = cross_val_predict(
            fitted_estimators, X, cv, method, n_jobs, return_train_preds
        )
        return preds, train_preds, fitted_estimators
    else:
        preds = cross_val_predict(fitted_estimators, X, cv, method, n_jobs)
        return preds, fitted_estimators
