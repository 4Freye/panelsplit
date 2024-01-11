from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import pandas as pd

class PanelSplit:
    def __init__(self, unique_periods, train_periods, n_splits=5, gap=None, test_size=None, max_train_size=None):
        """
        A class for performing time series cross-validation with custom train/test splits based on unique periods.

        Parameters:
        - n_splits: Number of splits for TimeSeriesSplit
        - gap: Gap between train and test sets in TimeSeriesSplit
        - test_size: Size of the test set in TimeSeriesSplit
        - unique_periods: Pandas DataFrame or Series containing unique periods
        - train_periods: All available training periods
        - max_train_size: Maximum size for a single training set.
        """
        self.tss = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size, max_train_size=max_train_size)
        indices = self.tss.split(unique_periods)
        self.u_periods_cv = self.split_unique_periods(indices, unique_periods)
        self.all_periods = train_periods
        self.n_splits = n_splits

    def split_unique_periods(self, indices, unique_periods):
        """
        Split unique periods into train/test sets based on TimeSeriesSplit indices.

        Parameters:
        - indices: TimeSeriesSplit indices
        - unique_periods: Pandas DataFrame or Series containing unique periods

        Returns: List of tuples containing train and test periods
        """
        u_periods_cv = []
        for i, (train_index, test_index) in enumerate(indices):
            unique_train_periods = unique_periods.iloc[train_index].values
            unique_test_periods = unique_periods.iloc[test_index].values
            u_periods_cv.append((unique_train_periods, unique_test_periods))
        return u_periods_cv

    def split(self, X=None, y=None, groups=None):
        """
        Generate train/test indices based on unique periods.
        """
        self.all_indices = []

        for i, (train_periods, test_periods) in enumerate(self.u_periods_cv):
            train_indices = self.all_periods.loc[self.all_periods.isin(train_periods)].index
            test_indices = self.all_periods.loc[self.all_periods.isin(test_periods)].index
            self.all_indices.append((train_indices, test_indices))

        return self.all_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns: Number of splits
        """
        return self.n_splits

    def cross_val_predict(self, estimator, X, y):
        """
        Perform cross-validated predictions using a given predictor model.

        Parameters:
        -----------
        estimator : The machine learning model used for prediction.

        X : pandas DataFrame
            The input features for the predictor.

        y : pandas Series
            The target variable to be predicted.

        Returns:
        --------
        pd.DataFrame
            Concatenated DataFrame containing predictions made by the model during cross-validation.
            It includes the original indices joined with the predicted values.

        """
        predictions = []

        for train_indices, test_indices in tqdm(self.all_indices):
            # first drop nas with respect to y_train
            y_train = y.iloc[train_indices].dropna()
            # use y_train to filter for X_train
            X_train = X.iloc[y_train.index]
            X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

            pred = pd.DataFrame(index=test_indices)
            pred['y_true'] = y_test.values

            model = estimator.fit(X_train, y_train)

            pred['y_pred'] = model.predict_proba(X_test)[:, 1]

            predictions.append(pred)

        return pd.concat(predictions, axis=0)
