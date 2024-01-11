from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import numpy as np

class PanelSplit:
    def __init__(self, unique_periods, train_periods, n_splits=5, gap=None, test_size=None, max_train_size=None):
        """
        A class for performing time series cross-validation with custom train/test splits based on unique periods.

        Parameters:
        - n_splits: Number of splits for TimeSeriesSplit
        - gap: Gap between train and test sets in TimeSeriesSplit
        - test_size: Size of the test set in TimeSeriesSplit
        - unique_periods: NumPy array or list containing unique periods
        - train_periods: All available training periods
        - max_train_size: Maximum size for a single training set.
        """
        self.tss = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size, max_train_size=max_train_size)
        indices = list(self.tss.split(unique_periods))
        self.u_periods_cv = self.split_unique_periods(indices, unique_periods)
        self.all_periods = train_periods
        self.n_splits = n_splits

    def split_unique_periods(self, indices, unique_periods):
        """
        Split unique periods into train/test sets based on TimeSeriesSplit indices.

        Parameters:
        - indices: TimeSeriesSplit indices
        - unique_periods: NumPy array or list containing unique periods

        Returns: List of tuples containing train and test periods
        """
        u_periods_cv = []
        for i, (train_index, test_index) in enumerate(indices):
            unique_train_periods = unique_periods[train_index]
            unique_test_periods = unique_periods[test_index]
            u_periods_cv.append((unique_train_periods, unique_test_periods))
        return u_periods_cv

    def split(self, X=None, y=None, groups=None):
        """
        Generate train/test indices based on unique periods.
        """
        self.all_indices = []

        for i, (train_periods, test_periods) in enumerate(self.u_periods_cv):
            train_indices = np.where(np.isin(self.all_periods, train_periods))[0]
            test_indices = np.where(np.isin(self.all_periods, test_periods))[0]
            self.all_indices.append((train_indices, test_indices))

        return self.all_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns: Number of splits
        """
        return self.n_splits

    def cross_val_predict(self, estimator, X, y, indices, prediction_method='predict'):
        """
        Perform cross-validated predictions using a given predictor model.
    
        Parameters:
        -----------
        estimator : The machine learning model used for prediction.
    
        X : NumPy array
            The input features for the predictor.
    
        y : NumPy array
            The target variable to be predicted.
    
        indices : NumPy array
            Indices corresponding to the dataset.
    
        prediction_method : str, optional (default='predict')
            The prediction method to use. It can be 'predict', 'predict_proba', or 'predict_log_proba'.
    
        Returns:
        --------
        dict
            Dictionary containing predictions made by the model during cross-validation.
            Keys are indices, and values are the corresponding true target values.
        """
        predictions = {}
    
        for train_indices, test_indices in tqdm(self.split(X=X, y=y)):
            y_train = y[train_indices].dropna()
            X_train = X[y_train.index]
            X_test, y_test = X[test_indices], y[test_indices]
    
            model = estimator.fit(X_train, y_train)
    
            if prediction_method == 'predict':
                y_pred = model.predict(X_test)
            elif prediction_method == 'predict_proba':
                y_pred = model.predict_proba(X_test)[:, 1]
            elif prediction_method == 'predict_log_proba':
                y_pred = model.predict_log_proba(X_test)[:, 1]
            else:
                raise ValueError("Invalid prediction_method. Supported values are 'predict', 'predict_proba', or 'predict_log_proba'.")
    
            # Convert test indices to a list for cases where test_indices is a NumPy array
            test_indices_list = test_indices.tolist()
    
            # Update predictions dictionary with key-value pairs
            predictions.update(dict(zip(test_indices_list, y_test)))
    
        return predictions
    
