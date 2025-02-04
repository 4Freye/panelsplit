import numpy as np
from sklearn.base import clone
from joblib import Parallel, delayed
from .utils.utils import _split_wrapper
from .utils.validation import check_method, check_fitted_estimators

def _predict_split(model, X_test, method='predict'):
    """
    Perform predictions for a single split.

    Parameters:
    - model: The machine learning model used for prediction.
    - X_test: pandas DataFrame. The input features for testing.
    - method: Optional str (default='predict'). The method to use.
    It can be 'predict', 'predict_proba', or 'predict_log_proba'.

    Returns:
    pd.Series: Series containing predicted values.
    """
    predict_func = getattr(model, method)
    return predict_func(X_test)

def _fit_split(estimator, X, y, train_indices, sample_weight=None):
    """
    Fit a cloned estimator on the given training indices.

    Parameters:
    - estimator: The machine learning model to be fitted.
    - X: pandas DataFrame. The input features for the estimator.
    - y: pandas Series. The target variable for the estimator.
    - train_indices: Indices for the training data.
    - sample_weight: Optional pandas Series or numpy array (default=None). Sample weights for the training data.

    Returns:
    - A fitted model.
    """
    local_estimator = clone(estimator)
    y_train = y.loc[train_indices].dropna()
    X_train = X.loc[y_train.index]

    if sample_weight is not None:
        sw = sample_weight.loc[y_train.index]
        return local_estimator.fit(X_train, y_train, sample_weight=sw)
    else:
        return local_estimator.fit(X_train, y_train)

def _prediction_order_to_original_order(indices): 
    """
    Convert the concatenated predictions back to the original order provided.
    
    Parameters:
    - indices: List of boolean or index arrays corresponding to the test/train splits.

    Returns:
    - A NumPy array representing the sorted index order to restore original data order.
    """
    indices = np.concatenate([np.where(indices_)[0] for indices_ in indices])
    return np.argsort(indices)
def cross_val_fit(estimator, X, y, cv, sample_weight=None, n_jobs=1, progress_bar=False):
    """
    Fit the estimator using cross-validation.

    Parameters:
    - estimator: The machine learning model to be fitted.
    - X: pandas DataFrame. The input features for the estimator.
    - y: pandas Series. The target variable for the estimator.
    - cv: cross-validation splitter. An object that generates train/test splits.
    - sample_weight: Optional pandas Series or numpy array (default=None). Sample weights for the training data.
    - n_jobs: Optional int (default=1). The number of jobs to run in parallel.
    - progress_bar: Optional bool (default=False). Whether to display a progress bar.

    Returns:
    - list of fitted models: List containing fitted models for each split.
    """
    fitted_estimators = Parallel(n_jobs=n_jobs)(
        delayed(_fit_split)(estimator, X, y, train_indices, sample_weight)
        for train_indices, _ in _split_wrapper(indices=cv.split(), progress_bar=progress_bar)
    )
    return fitted_estimators


def cross_val_predict(fitted_estimators, X, cv, method='predict', n_jobs=1, return_train_preds=False):
    """
    Perform cross-validated predictions using a given predictor model.

    Parameters:
    - fitted_estimators: A list of machine learning models used for prediction.
    - X: pandas DataFrame. The input features for the predictor.
    - cv: cross-validation splitter. An object that generates train/test splits.
    - method: Optional str (default='predict'). The method to use.
      It can be 'predict', 'predict_proba', or 'predict_log_proba'.
    - n_jobs: Optional int (default=1). The number of jobs to run in parallel.
    - return_train_preds: Optional bool (default=False). If True, return predictions for the training set as well.

    Returns:
    - test_preds: NumPy array containing test predictions made by the model during cross-validation.
      If return_train_preds is True, train predictions will also be returned.
    - train_preds: NumPy array containing train predictions made by the model during cross-validation.
    """
    check_fitted_estimators(fitted_estimators)

    test_splits = [split[1] for split in cv.split()]
    test_indices = _prediction_order_to_original_order(test_splits)

    test_preds = Parallel(n_jobs=n_jobs)(
        delayed(_predict_split)(fitted_estimators[i], X.loc[test_idx], method)
        for i, test_idx in enumerate(test_splits)
    )

    if return_train_preds:
        train_splits = [split[0] for split in cv.split()]
        train_indices = _prediction_order_to_original_order(train_splits)

        train_preds = Parallel(n_jobs=n_jobs)(
            delayed(_predict_split)(fitted_estimators[i], X.loc[train_idx], method)
            for i, train_idx in enumerate(train_splits)
        )

        return np.concatenate(test_preds, axis=0)[test_indices], np.concatenate(train_preds, axis=0)[train_indices]
    else:
        return np.concatenate(test_preds, axis=0)[test_indices]


def cross_val_fit_predict(estimator, X, y, cv, method='predict', sample_weight=None, n_jobs=1, return_train_preds=False):
    """
    Fit the estimator using cross-validation and then make predictions.

    Parameters:
    - estimator: The machine learning model to be fitted.
    - X: pandas DataFrame. The input features for the estimator.
    - y: pandas Series. The target variable for the estimator.
    - cv: cross-validation splitter. An object that generates train/test splits.
    - method: Optional str (default='predict'). The method to use. For example, 'predict', 'predict_proba', or 'predict_log_proba'.
    - sample_weight: Optional pandas Series or numpy array (default=None). Sample weights for the training data.
    - n_jobs: Optional int (default=1). The number of jobs to run in parallel.
    - return_train_preds: Optional bool (default=False). If True, return predictions for the training set as well.

    Returns:
    - pd.DataFrame: Concatenated DataFrame containing predictions made by the model during cross-validation. 
      It includes the original indices joined with the predicted values.
    - (Optional) pd.DataFrame: Concatenated DataFrame containing predictions made by the model during cross-validation for the training set. 
      It includes the original indices joined with the predicted values.
    - list of fitted models: List containing fitted models for each split.
    """
    fitted_estimators = cross_val_fit(estimator, X, y, cv, sample_weight, n_jobs)

    if return_train_preds:
        preds, train_preds = cross_val_predict(fitted_estimators, X, cv, method, n_jobs, return_train_preds)
        return preds, train_preds, fitted_estimators
    else:
        preds = cross_val_predict(fitted_estimators, X, cv, method, n_jobs)
        return preds, fitted_estimators
