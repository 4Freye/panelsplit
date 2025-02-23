# this is still under development
import unittest
import pandas as pd
import numpy as np
from panelsplit.cross_validation import PanelSplit
from panelsplit.application import cross_val_fit_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class TestPanelSplit(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.periods = pd.Series(pd.date_range(start='2022-01-01', end='2022-01-10').tolist() * 2)
        self.panel_split = PanelSplit(periods = self.periods, n_splits=3)

    def test_split(self):
        # Test if the split method returns the correct number of splits
        splits = self.panel_split.split()
        self.assertEqual(len(splits), 3)

    def test_get_n_splits(self):
        # Test if the get_n_splits method returns the correct number of splits
        n_splits = self.panel_split.get_n_splits()
        self.assertEqual(n_splits, 3)

    def test_gen_train_labels(self):
        # Test if the gen_train_labels method returns the correct train labels
        vals =[ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        labels = pd.Series(vals, index= vals)
        train_labels = self.panel_split.gen_train_labels(pd.Series(range(len(self.periods))))
        pd.testing.assert_series_equal(train_labels, labels)  # Assuming one split has no train data

    def test_gen_test_labels(self):
        # Test if the gen_test_labels method returns the correct test labels
        vals = [ 7,  8,  9, 17, 18, 19]
        labels = pd.Series(vals, index=vals)
        test_labels = self.panel_split.gen_test_labels(pd.Series(range(len(self.periods))))
        pd.testing.assert_series_equal(test_labels, labels)  # Assuming one split has no test data

    def test_parallel_prediction(self):
        # Test if the predictions are the same in parallel and non-parallel
        np.random.seed(1)
        df = pd.DataFrame(np.random.random((40, 10)))
        df['entity'] = np.repeat(['A','B','C', 'D'], 10)
        df['time'] = list(range(10)) * 4
        df.set_index(['entity', 'time'], inplace=True)

        # Specify model 
        model = RandomForestRegressor(n_estimators=10, random_state=1)

        # Run panel split: initialize, generate test labels, and fit and predict on data with and without parallel
        ps = PanelSplit(periods=pd.Series(df.index.get_level_values('time')), n_splits=5)
        pred_df = ps.gen_test_labels(df.iloc[:, 0].reset_index())
        pred_df['pred'], _ = cross_val_fit_predict(model, X=df.iloc[:, 1:], y=df.iloc[:, 0], cv=ps)
        pred_df['pred_parallel'], _ = cross_val_fit_predict(model, X=df.iloc[:, 1:], y=df.iloc[:, 0],cv =  ps, n_jobs=-1)

        # Assert whether the mean squared errors are equal
        self.assertTrue(np.isclose(mean_squared_error(pred_df[0], pred_df['pred']), mean_squared_error(pred_df[0], pred_df['pred_parallel'])))

if __name__ == '__main__':
    unittest.main()
