import numpy as np
import pandas as pd
from sklearn.base import clone
from joblib import Parallel, delayed
from .utils.utils import _split_wrapper
from .utils.validation import check_method, check_fitted_estimators, check_cv
from typing import List, Union

def _predict_split(model, X_test: pd.DataFrame, method: str = 'predict') -> np.ndarray:
    """
    Perform predictions for a single split.

    Parameters
    ----------
    model : object
        The machine learning model used for prediction.
    X_test : pd.DataFrame
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


def _fit_split(estimator, X: pd.DataFrame, y: pd.Series, train_indices: List[bool],
               sample_weight: Union[pd.Series, np.ndarray] = None,
               drop_na_in_y=False):
    """
    Fit a cloned estimator on the given training indices.

    Parameters
    ----------
    estimator : object
        The machine learning model to be fitted.
    X : pd.DataFrame
        The input features for the estimator.
    y : pd.Series
        The target variable for the estimator.
    train_indices : list of bool
        Boolean mask or indices indicating the training data.
    sample_weight : pd.Series or np.ndarray, optional
        Sample weights for the training data. Default is None.

    Returns
    -------
    object
        A fitted estimator.
    """
    local_estimator = clone(estimator)

    if drop_na_in_y:
        y = y.loc[train_indices].dropna()
        X = X.loc[y.index]

    if sample_weight is not None:
        sw = sample_weight.loc[y.index]
        return local_estimator.fit(X, y, sample_weight=sw)
    else:
        return local_estimator.fit(X, y)


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


def cross_val_fit(estimator, X: pd.DataFrame, y: pd.Series, cv, 
                  sample_weight: Union[pd.Series, np.ndarray] = None, n_jobs: int = 1, 
                  progress_bar: bool = False, drop_na_in_y=False):
    """
    Fit the estimator using cross-validation.

    Parameters
    ----------
    estimator : object
        The machine learning model to be fitted.
    X : pd.DataFrame
        The input features for the estimator.
    y : pd.Series
        The target variable for the estimator.
    cv : object or iterable
        Cross-validation splitter; either an object that generates train/test splits (e.g., an instance of PanelSplit)
        or an iterable of splits.
    sample_weight : pd.Series or np.ndarray, optional
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
        delayed(_fit_split)(estimator, X, y, train_indices, sample_weight, drop_na_in_y = drop_na_in_y)
        for train_indices, _ in _split_wrapper(indices=splits, progress_bar=progress_bar)
    )
    
    return fitted_estimators


def cross_val_predict(fitted_estimators, X: pd.DataFrame, cv, method: str = 'predict', 
                      n_jobs: int = 1, return_train_preds: bool = False) -> np.ndarray:
    """
    Perform cross-validated predictions using a given predictor model.

    Parameters
    ----------
    fitted_estimators : list
        List of fitted machine learning models used for prediction.
    X : pd.DataFrame
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

    test_preds = Parallel(n_jobs=n_jobs)(
        delayed(_predict_split)(fitted_estimators[i], X.loc[test_idx], method)
        for i, test_idx in enumerate(test_splits)
    )

    if return_train_preds:
        train_splits = [split[0] for split in splits]
        train_indices = _prediction_order_to_original_order(train_splits)

        train_preds = Parallel(n_jobs=n_jobs)(
            delayed(_predict_split)(fitted_estimators[i], X.loc[train_idx], method)
            for i, train_idx in enumerate(train_splits)
        )

        return np.concatenate(test_preds, axis=0)[test_indices], np.concatenate(train_preds, axis=0)[train_indices]
    else:
        return np.concatenate(test_preds, axis=0)[test_indices]


def cross_val_fit_predict(estimator, X: pd.DataFrame, y: pd.Series, cv, method: str = 'predict',
                            sample_weight: Union[pd.Series, np.ndarray] = None, n_jobs: int = 1,
                            return_train_preds: bool = False,
                            drop_na_in_y=False) -> np.ndarray:
    """
    Fit the estimator using cross-validation and then make predictions.

    Parameters
    ----------
    estimator : object
        The machine learning model to be fitted.
    X : pd.DataFrame
        The input features for the estimator.
    y : pd.Series
        The target variable for the estimator.
    cv : object
        Cross-validation splitter; an object that generates train/test splits.
     method : str, optional
        The method to use for prediction. It can be whatever methods are available to the estimator 
        (e.g. predict_proba in the case of a classifier or transform in the case of a transformer). Default is 'predict'.
    sample_weight : pd.Series or np.ndarray, optional
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

    fitted_estimators = cross_val_fit(estimator, X, y, cv, sample_weight, n_jobs,  drop_na_in_y = drop_na_in_y)

    if return_train_preds:
        preds, train_preds = cross_val_predict(fitted_estimators, X, cv, method, n_jobs, return_train_preds)
        return preds, train_preds, fitted_estimators
    else:
        preds = cross_val_predict(fitted_estimators, X, cv, method, n_jobs)
        return preds, fitted_estimators
