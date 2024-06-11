from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from sklearn.base import clone
import warnings

class PanelSplit:
    def __init__(self, periods, unique_periods= None, snapshots = None, n_splits = 5, gap = 0, test_size = 1, max_train_size=None, plot=False, drop_splits=False, y=None, progress_bar=False):
        """
        A class for performing time series cross-validation with custom train/test splits based on unique periods.

        Parameters:
        - periods: A pandas Series containing all available training periods.
        - unique_periods: Optional. Pandas DataFrame or Series containing unique periods.
        - snapshots: Optional. A Pandas Series defining the snapshot for the observation (when it was updated)
        - n_splits: Number of splits for TimeSeriesSplit.
        - gap: Gap between train and test sets in TimeSeriesSplit.
        - test_size: Size of the test set in TimeSeriesSplit.
        - max_train_size: Maximum size for a single training set.
        - plot: Flag to visualize time series splits.
        - drop_splits: Whether to drop splits with empty or single-value train or test sets.
        - y: Target variable to assess whether splits contain empty or single-value train or test sets. Required if drop_splits is True.
        - progress_bar: Bool. Whether or not to use the tqdm progress bar.
        """

        if unique_periods is None:
            unique_periods = pd.Series(periods.unique()).sort_values()
        self.tss = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size, max_train_size = max_train_size)
        indices = self.tss.split(unique_periods.reset_index(drop=True))
        self.u_periods_cv = self._split_unique_periods(indices, unique_periods)
        self.all_periods = periods
        self.snapshots = snapshots
    
        if y is not None and drop_splits is False:
            warnings.warn("Ignoring y values because drop_splits is False.", stacklevel=2)
        
        if y is None and drop_splits is True:
            raise ValueError("Cannot drop splits without specifying y values.")
        
        self.drop_splits = drop_splits
        self.progress_bar = progress_bar
        self.n_splits = n_splits
        self.split(y=y, init=True)

        if plot:
            self._plot_time_series_splits(self.u_periods_cv)
        
    def _split_wrapper(self, indices):
        if self.progress_bar:
            return tqdm(indices)
        else:
            return indices
    def _split_unique_periods(self, indices, unique_periods):
        """
        Split unique periods into train/test sets based on TimeSeriesSplit indices.

        Parameters:
        - indices: TimeSeriesSplit indices.
        - unique_periods: Pandas DataFrame or Series containing unique periods.

        Returns: 
        List of tuples containing train and test periods.
        """

        u_periods_cv = []
        for i, (train_index, test_index) in enumerate(indices):
            unique_train_periods = unique_periods.iloc[train_index].values
            unique_test_periods = unique_periods.iloc[test_index].values
            u_periods_cv.append((unique_train_periods, unique_test_periods))
        return u_periods_cv

    def split(self, X = None, y = None, groups=None, init=False):
        """
        Generate train/test indices based on unique periods.
        """
        self.all_indices = []
       
        for i, (train_periods, test_periods) in enumerate(self.u_periods_cv):
            if self.snapshots is not None:
                if test_periods.max() >= self.snapshots.min():
                    snapshot_val = test_periods.max()  
                else:
                    snapshot_val = self.snapshots.min()
                    if init:
                        warnings.warn(f'The maximum period value {test_periods.max()} for split {i} is less than the minimum snapshot value {self.snapshots.min()}. Defaulting to minimum snapshot value for split {i}.', stacklevel=2)
                train_indices = self.all_periods.isin(train_periods).values & (self.snapshots == snapshot_val)
                test_indices = self.all_periods.isin(test_periods).values & (self.snapshots == snapshot_val)
            else:
                train_indices = self.all_periods.isin(train_periods).values
                test_indices = self.all_periods.isin(test_periods).values 

            if self.drop_splits and ((len(train_indices) == 0 or len(test_indices) == 0) or (y.loc[train_indices].nunique() == 1 or y.loc[test_indices].nunique() == 1)):
                if init:
                    self.n_splits -= 1
                    print(f'Dropping split {i} as either the test or train set is either empty or contains only one unique value.')
                else:
                    continue
            else:
                self.all_indices.append((train_indices, test_indices))
        return self.all_indices
   
    def get_n_splits(self, X=None, y =None, groups=None):
        """
        Returns: Number of splits
        """
        return self.n_splits
    
    def gen_snapshots(self, data, period_col = None):
        """
        Generate snapshots for each split.

        Parameters:
        - data: A pandas DataFrame from which snapshots are generated.
        - period_col: Optional. A str, the column in data from which the column snapshot_period is created.

        Returns: 
        A pandas DataFrame where each split has its own set of observations.
        """

        _data = data.copy()
        splits = self.split()
        snapshots = []
        for i, split in enumerate(splits):
            split_indices = np.array([split[0], split[1]]).any(axis = 0)
            if period_col is not None:
                last_period = _data.loc[split_indices, period_col].unique().max()
                snapshots.append(_data.loc[split_indices].assign(split = i, snapshot_period = last_period))
            else:
                snapshots.append(_data.loc[split_indices].assign(split = i))
        return pd.concat(snapshots)
    
    def gen_train_labels(self, labels):
        """
        Generate test labels using the DataFrame's labels.

        Parameters:
        - labels: Pandas Series or DataFrame. The labels used to identify observations.
        
        Returns:
        The labels of each fold's train set as a single DataFrame.
        """
        train_indices = np.stack([split[0] for split in self.split()], axis=1).any(axis=1)
        return labels.loc[train_indices].copy()

    def gen_test_labels(self, labels):
        """
        Generate test labels using the DataFrame's labels.

        Parameters:
        - labels: Pandas Series or DataFrame. The labels used to identify observations.
        
        Returns:
        The labels of each fold's test set as a single DataFrame.
        """
        test_indices = np.stack([split[1] for split in self.split()], axis=1).any(axis=1)
        return labels.loc[test_indices].copy()

    def _predict_split(self, model, X_test, prediction_method='predict'):
        """
        Perform predictions for a single split.

        Parameters:
        - estimator: The machine learning model used for prediction.
        - X_train: pandas DataFrame. The input features for training.
        - y_train: pandas Series. The target variable for training.
        - X_test: pandas DataFrame. The input features for testing.
        - prediction_method: Optional str (default='predict'). The prediction method to use. It can be 'predict', 'predict_proba', or 'predict_log_proba'.

        Returns:
        pd.Series: Series containing predicted values.
        """

        if prediction_method == 'predict':
            return model.predict(X_test)
        elif prediction_method == 'predict_proba':
            return model.predict_proba(X_test)
        elif prediction_method == 'predict_log_proba':
            return model.predict_log_proba(X_test)
        else:
            raise ValueError("Invalid prediction_method. Supported values are 'predict', 'predict_proba', or 'predict_log_proba'.")

    def cross_val_fit(self, estimator, X, y, sample_weight=None, n_jobs=1):
        """
        Fit the estimator using cross-validation.

        Parameters:
        - estimator: The machine learning model to be fitted.
        - X: pandas DataFrame. The input features for the estimator.
        - y: pandas Series. The target variable for the estimator.
        - sample_weight: Optional pandas Series or numpy array (default=None). Sample weights for the training data.
        - n_jobs: Optional int (default=1). The number of jobs to run in parallel.

        Returns:
        list of fitted models: List containing fitted models for each split.
        """
        def fit_split(train_indices):
            local_estimator = clone(estimator)
            y_train = y.loc[train_indices].dropna()
            X_train = X.loc[y_train.index]
            if sample_weight is not None:
                sw = sample_weight.loc[y_train.index]
                return local_estimator.fit(X_train, y_train, sample_weight=sw)
            else:
                return local_estimator.fit(X_train, y_train)

        fitted_estimators = Parallel(n_jobs=n_jobs)(
            delayed(fit_split)(train_indices)
            for train_indices, _ in self._split_wrapper(self.split())
        )
        return fitted_estimators

    def cross_val_predict(self, fitted_estimators, X, prediction_method='predict', n_jobs=1, return_train_preds=False):
        """
        Perform cross-validated predictions using a given predictor model.

        Parameters:
        - fitted_estimators: A list of machine learning models used for prediction.
        - X: pandas DataFrame. The input features for the predictor.
        - labels: Optional pandas DataFrame. Labels to identify the predictions, if provided will be returned along with predictions.
        - prediction_method: Optional str (default='predict'). The prediction method to use. It can be 'predict', 'predict_proba', or 'predict_log_proba'.
        - n_jobs: Optional int (default=1). The number of jobs to run in parallel.
        - return_train_preds: Optional bool (default=False). If True, return predictions for the training set as well.

        Returns:
        - test_preds: NumPy array containing test predictions made by the model during cross-validation. If return_train_preds is True, train predictions will also be returned.
        - train_preds: NumPy array containing train predictions made by the model during cross-validation.
        """

        def prediction_order_to_original_order(indices): 
            """
            To convert the concatenated predictions back to original order provided
            """
            indices = np.concatenate([np.where(indices_)[0] for indices_ in indices])
            return np.argsort(indices)

        def predict_split(model, test_indices):
            X_test = X.loc[test_indices]
            return self._predict_split(model, X_test, prediction_method)

        test_indices = prediction_order_to_original_order([split[1] for split in self.split()])

        test_preds = Parallel(n_jobs=n_jobs)(
            delayed(predict_split)(fitted_estimators[i], test_indices)
            for i, (_, test_indices) in enumerate(self.split())
        )

        if return_train_preds:
            train_preds = Parallel(n_jobs=n_jobs)(
                delayed(predict_split)(fitted_estimators[i], train_indices)
                for i, (train_indices, _) in enumerate(self.split())
            )
            train_indices = prediction_order_to_original_order([split[0] for split in self.split()])
            return np.concatenate(test_preds, axis = 0)[test_indices], np.concatenate(train_preds, axis = 0)[train_indices]
        else:
            return np.concatenate(test_preds, axis=0)[test_indices]

    def cross_val_fit_predict(self, estimator, X, y, prediction_method='predict', sample_weight=None, n_jobs=1, return_train_preds=False):
        """
        Fit the estimator using cross-validation and then make predictions.

        Parameters:
        - estimator: The machine learning model to be fitted.
        - X: pandas DataFrame. The input features for the estimator.
        - y: pandas Series. The target variable for the estimator.
        - prediction_method: Optional str (default='predict'). The prediction method to use. It can be 'predict', 'predict_proba', or 'predict_log_proba'.
        - sample_weight: Optional pandas Series or numpy array (default=None). Sample weights for the training data.
        - n_jobs: Optional int (default=1). The number of jobs to run in parallel.
        - return_train_preds: Optional bool (default=False). If True, return predictions for the training set as well.

        Returns:
        pd.DataFrame: Concatenated DataFrame containing predictions made by the model during cross-validation. It includes the original indices joined with the predicted values.
        (Optional) pd.DataFrame: Concatenated DataFrame containing predictions made by the model during cross-validation for the training set. It includes the original indices joined with the predicted values.
        list of fitted models: List containing fitted models for each split.
        """
        fitted_estimators = self.cross_val_fit(estimator, X, y, sample_weight, n_jobs)

        if return_train_preds:
            preds, train_preds = self.cross_val_predict(fitted_estimators, X, prediction_method, n_jobs, return_train_preds)
            return preds, train_preds, fitted_estimators
        else:
            preds = self.cross_val_predict(fitted_estimators, X, prediction_method, n_jobs)
            return preds, fitted_estimators
            
    def _plot_time_series_splits(self, split_output):
        """
        Visualize time series splits using a scatter plot.

        Parameters:
        - split_output: Output of time series splits.
        """
        splits = len(split_output)
        fig, ax = plt.subplots()
        
        for i, (train_index, test_index) in enumerate(split_output):
            ax.scatter(train_index, [i] * len(train_index), color='blue', marker='.', s=50)
            ax.scatter(test_index, [i] * len(test_index), color='red', marker='.', s=50)

        ax.set_xlabel('Periods')
        ax.set_ylabel('Split')
        ax.set_title('Cross-Validation Splits')
        ax.set_yticks(range(splits))  # Set the number of ticks on y-axis
        ax.set_yticklabels([f'{i}' for i in range(splits)])  # Set custom labels for y-axi
        plt.show()

    def _cross_val_fit(self, transformer, X, include_test_in_fit=False, n_jobs=1):
        """
        Perform cross-validated fitting using a given transformer.

        Parameters:
        - transformer: The transformer object used for fitting.
        - X: pandas DataFrame. The input features for the transformer.
        - include_test_in_fit: Optional bool (default=False). If True, include test data in fitting the transformer.

        Returns:
        list of fitted transformers: List containing fitted transformers for each split.
        """
        transformers = []

        def fit_split(train_indices, test_indices):
            X_train = X.loc[train_indices]
            local_transformer = clone(transformer)
            if include_test_in_fit:
                return local_transformer.fit(X_train, X.loc[test_indices])
            else:
                return local_transformer.fit(X_train)

        fitted_transformers = Parallel(n_jobs=n_jobs)(
            delayed(fit_split)(train_indices, test_indices)
            for train_indices, test_indices in self._split_wrapper(self.split())
        )

        return fitted_transformers

    def cross_val_transform(self, transformers, X, transform_train = False, n_jobs=1):
        """
        Perform cross-validated transformation using fitted transformers.

        Parameters:
        - transformers: List of fitted transformers.
        - X: pandas DataFrame. The input features for the transformation.
        - transform_train: Optional bool (default=False). If True, transform train set as well as the test set.

        Returns:
        pd.DataFrame: DataFrame containing transformed values during cross-validation.
        """
        _X = X.copy()

        if transform_train:
            if self.snapshots is None:
                raise ValueError("Cannot transform training sets without providing snapshots.")

        def transform_split(transformer, train_indices, test_indices):
            if transform_train:
                train_or_test = pd.concat([train_indices, test_indices], axis = 1).any(axis = 1)
                _X.loc[train_or_test] = transformer.transform(X.loc[train_or_test])
            else:
                _X.loc[test_indices] = transformer.transform(X.loc[test_indices])
            
        Parallel(n_jobs=n_jobs)(
            delayed(transform_split)(transformers[i], train_indices, test_indices)
            for i, (train_indices, test_indices) in enumerate(self.split())
        )

        return _X

    def cross_val_fit_transform(self, transformer, X, include_test_in_fit=False, transform_train=False, n_jobs = 1):
        """
        Perform cross-validated fitting and transformation using a given transformer.

        Parameters:
        - transformer: The transformer object used for fitting and transformation.
        - X: pandas DataFrame. The input features for the transformer.
        - include_test_in_fit: Optional bool (default=False). If True, include test data in fitting the transformer.
        - transform_train: Optional bool (default=False). If True, transform train set as well as the test set.

        Returns:
        pd.DataFrame: DataFrame containing transformed values during cross-validation.
        list of fitted transformers: List containing fitted transformers for each split.
        """
        transformers = self._cross_val_fit(transformer, X, include_test_in_fit, n_jobs)
        _X = self.cross_val_transform(transformers, X, transform_train, n_jobs)

        return _X, transformers