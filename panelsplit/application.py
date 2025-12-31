import inspect
from typing import Tuple, List, Optional, Iterable, Union
from numpy.typing import NDArray

import narwhals as nw
import numpy as np
from joblib import Parallel, delayed
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.base import clone
from .utils.typing import ArrayLike, EstimatorLike
from .cross_validation import PanelSplit
from typing import Literal

from .utils.utils import _split_wrapper
from .utils.validation import (
    _safe_indexing,
    _to_numpy_array,
    check_cv,
    check_fitted_estimators,
    _supports_sample_weights,
)


def _get_non_null_mask(data: IntoSeries) -> IntoSeries:
    """Get non-null mask for any data type."""
    return ~nw.from_native(data, series_only=True).is_null()


def _predict_split(
    model: EstimatorLike, X_test: ArrayLike, method: str = "predict"
) -> np.ndarray:
    """
    Perform predictions for a single split.

    Parameters
    ----------
    model : EstimatorLike
        The machine learning model used for prediction.
    X_test : ArrayLike
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
    estimator: EstimatorLike,
    X: IntoDataFrame,
    y: Optional[IntoSeries],
    train_indices: NDArray,
    sample_weight: Optional[Union[IntoSeries, NDArray]] = None,
    drop_na_in_y: bool = False,
) -> EstimatorLike:
    """
    Fit a cloned estimator on the given training indices.

    Parameters
    ----------
    estimator : EstimatorLike
        The machine learning model to be fitted.
    X : IntoDataFrame
        The input features for the estimator.
    y : Optional[IntoSeries]
        The target variable for the estimator. Default is None.
    train_indices : NDArray
        Integer indices indicating the training data.
    sample_weight : Optional[Union[IntoSeries, NDArray]]
        Sample weights for the training data. Default is None.
    drop_na_in_y : bool
        Whether to drop rows with null values in y. Default is False

    Returns
    -------
    EstimatorLike
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
    estimator: EstimatorLike,
    X: IntoDataFrame,
    y: IntoSeries,
    cv: Union[PanelSplit, Iterable],
    sample_weight: Optional[Union[IntoSeries, np.ndarray]] = None,
    n_jobs: int = 1,
    progress_bar: bool = False,
    drop_na_in_y: bool = False,
) -> List[EstimatorLike]:
    """
    Fit the estimator using cross-validation.

    Parameters
    ----------
    estimator : EstimatorLike
        The machine learning model to be fitted.
    X : IntoDataFrame
        The input features for the estimator.
    y : IntoSeries
        The target variable for the estimator.
    cv : Union[PanelSplit, Iterable]
        Cross-validation splitter; either an object that generates train/test splits (e.g., an instance of PanelSplit)
        or an iterable of splits.
    sample_weight : Optional[Union[IntoSeries, np.ndarray]]
        Sample weights for the training data. Default is None.
    n_jobs : int
        The number of jobs to run in parallel. Default is 1.
    progress_bar : bool
        Whether to display a progress bar. Default is False.
    drop_na_in_y : bool
        Whether to drop observations where y is na. Default is False.

    Returns
    -------
    List[EstimatorLike]
        List containing fitted models for each split.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from panelsplit.cross_validation import PanelSplit
    >>> # Create sample data
    >>> df = pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6], "period": [1, 1, 2, 2, 3, 3]})
    >>> X = df[["feature"]]
    >>> y = pd.Series([2, 4, 6, 8, 10, 12])
    >>> # Create a PanelSplit instance for cross-validation
    >>> ps = PanelSplit(periods=df["period"], n_splits=2)
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
    fitted_estimators: List[EstimatorLike],
    X: IntoDataFrame,
    cv: Union[PanelSplit, Iterable],
    method: str = "predict",
    n_jobs: int = 1,
    return_group: Literal["test", "train"] = "test",
) -> np.ndarray:
    """
    Perform cross-validated predictions using a given predictor model.

    Parameters
    ----------
    fitted_estimators : List[EstimatorLike]
        List of fitted machine learning models used for prediction.
    X : IntoDataFrame
        The input features for prediction.
    cv : Union[PanelSplit, Iterable]
        Cross-validation splitter; either an object that generates train/test splits or an iterable of splits.
    method : str
        The method to use for prediction. It can be whatever methods are available to the estimator.
        (e.g. predict_proba in the case of a classifier or transform in the case of a transformer). Default is 'predict'.
    n_jobs : int
        The number of jobs to run in parallel. Default is 1.
    return_group : {"test","train"}
        Whether to return the train or test predictions. Default is "test".

    Returns
    -------
    np.ndarray
        Predictions (either train or test depending on return_group).

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> import numpy as np
    >>> from panelsplit.cross_validation import PanelSplit,
    >>> from panelsplit.application import cross_val_predict, cross_val_fit
    >>> X = np.arange(12).reshape(6, 2)
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> ps = PanelSplit(periods=np.array([1, 1, 2, 2, 3, 3]), n_splits=2)
    >>> estimators = cross_val_fit(LinearRegression(), X, y, ps)
    >>> preds = cross_val_predict(estimators, X, ps)
    >>> preds.shape
    (4,)
    """
    check_fitted_estimators(fitted_estimators)
    splits = check_cv(cv)
    if return_group not in ["train", "test"]:
        raise ValueError(
            f"return_group must be train or test. Got {return_group} instead."
        )

    group = 0 if return_group == "train" else 1
    group_splits = [split[group] for split in splits]
    group_indices = _prediction_order_to_original_order(group_splits)

    # Use narwhals for dataframe-agnostic operations
    X_nw = nw.from_native(X, pass_through=True)

    preds = Parallel(n_jobs=n_jobs)(
        delayed(_predict_split)(
            fitted_estimators[i],
            _safe_indexing(X_nw, test_idx, to_native=True),
            method,
        )
        for i, test_idx in enumerate(group_splits)
    )

    return np.concatenate(preds, axis=0)[group_indices]


def cross_val_fit_predict(
    estimator: EstimatorLike,
    X: IntoDataFrame,
    y: IntoSeries,
    cv: Union[PanelSplit, Iterable],
    method: str = "predict",
    sample_weight: Optional[Union[IntoSeries, np.ndarray]] = None,
    n_jobs: int = 1,
    return_group: Literal["test", "train"] = "test",
    drop_na_in_y: bool = False,
) -> Tuple[np.ndarray, List[EstimatorLike]]:
    """
    Fit the estimator using cross-validation and then make predictions.

    Parameters
    ----------
    estimator : EstimatorLike
        The machine learning model to be fitted.
    X : IntoDataFrame
        The input features for the estimator.
    y : IntoSeries
        The target variable for the estimator.
    cv : Union[PanelSplit, Iterable]
        Cross-validation splitter; an object that generates train/test splits.
    method : str
        The method to use for prediction. It can be any method available on the estimator
        (e.g., ``predict_proba`` for classifiers or ``transform`` for transformers). Default is predict.
    sample_weight : Optional[Union[IntoSeries, np.ndarray]]
        Sample weights for the training data. Default is None.
    n_jobs : int
        The number of jobs to run in parallel. Default is 1.
    return_group : {"test","train"}
        Whether to return the train or test predictions. Default is test.
    drop_na_in_y : bool
        Whether to drop observations where ``y`` is NA. Default is False.

    Returns
    -------
    Tuple[np.ndarray, List[EstimatorLike]]
        (predictions (either train or test depending on return_group), fitted_estimators).

    Raises
    ------
    TypeError
        If the provided estimator does not implement the specified ``method`` or has invalid type.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from panelsplit.cross_validation import (
    ...     PanelSplit,
    ... )  # assuming PanelSplit is imported from your module
    >>> # Create sample data
    >>> df = pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6], "period": [1, 1, 2, 2, 3, 3]})
    >>> X = df[["feature"]]
    >>> y = pd.Series([2, 4, 6, 8, 10, 12])
    >>> # Create a PanelSplit instance for cross-validation
    >>> ps = PanelSplit(periods=df["period"], n_splits=2)
    >>> # Get test predictions and fitted models
    >>> preds, models = cross_val_fit_predict(LinearRegression(), X, y, ps)
    >>> preds.shape
    (2,)
    """

    fitted_estimators = cross_val_fit(
        estimator, X, y, cv, sample_weight, n_jobs, drop_na_in_y=drop_na_in_y
    )

    preds = cross_val_predict(fitted_estimators, X, cv, method, n_jobs, return_group)

    return preds, fitted_estimators
