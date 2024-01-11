from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import numpy as np

class PanelSplit:
    def __init__(self, unique_periods, train_periods, n_splits=5, gap=None, test_size=None, max_train_size=None):
        self.tss = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size, max_train_size=max_train_size)
        indices = list(self.tss.split(unique_periods))
        self.u_periods_cv = self.split_unique_periods(indices, unique_periods)
        self.all_periods = train_periods
        self.n_splits = n_splits

    def split_unique_periods(self, indices, unique_periods):
        u_periods_cv = []
        for i, (train_index, test_index) in enumerate(indices):
            unique_train_periods = unique_periods.iloc[train_index].values
            unique_test_periods = unique_periods.iloc[test_index].values
            u_periods_cv.append((unique_train_periods, unique_test_periods))
        return u_periods_cv

    def split(self, X=None, y=None, groups=None):
        self.all_indices = []

        for i, (train_periods, test_periods) in enumerate(self.u_periods_cv):
            train_indices = np.where(np.isin(self.all_periods, train_periods))[0]
            test_indices = np.where(np.isin(self.all_periods, test_periods))[0]
            self.all_indices.append((train_indices, test_indices))

        return self.all_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def cross_val_predict(self, estimator, X, y, indices, prediction_method='predict'):
        predictions = []

        for train_indices, test_indices in tqdm(self.all_indices):
            y_train = y[train_indices].dropna()
            X_train = X[y_train.index]
            X_test, y_test = X[test_indices], y[test_indices]

            pred = {'index': indices[test_indices], 'y_true': y_test}

            model = estimator.fit(X_train, y_train)

            if prediction_method == 'predict':
                pred['y_pred'] = model.predict(X_test)
            elif prediction_method == 'predict_proba':
                pred['y_pred'] = model.predict_proba(X_test)[:, 1]
            elif prediction_method == 'predict_log_proba':
                pred['y_pred'] = model.predict_log_proba(X_test)[:, 1]
            else:
                raise ValueError("Invalid prediction_method. Supported values are 'predict', 'predict_proba', or 'predict_log_proba'.")

            predictions.append(pred)

        return predictions
