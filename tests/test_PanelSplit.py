# this is still under development
import unittest
import pandas as pd
from panelsplit import PanelSplit

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

if __name__ == '__main__':
    unittest.main()
