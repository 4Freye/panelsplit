# this is still under development
import unittest
import pandas as pd
from ..panelsplit import PanelSplit

class TestPanelSplit(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        periods = pd.Series(pd.date_range(start='2022-01-01', end='2022-01-10'))
        unique_periods = pd.Series([pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-05'), pd.Timestamp('2022-01-10')])
        self.panel_split = PanelSplit(periods, unique_periods=unique_periods, n_splits=3)

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
        labels = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        train_labels = self.panel_split.gen_train_labels(labels)
        self.assertEqual(len(train_labels), 7)  # Assuming one split has no train data

    def test_gen_test_labels(self):
        # Test if the gen_test_labels method returns the correct test labels
        labels = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        test_labels = self.panel_split.gen_test_labels(labels)
        self.assertEqual(len(test_labels), 3)  # Assuming one split has no test data

if __name__ == '__main__':
    unittest.main()
