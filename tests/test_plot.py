import unittest
import pandas as pd
from panelsplit.cross_validation import PanelSplit
from panelsplit.plot import plot_splits

class TestPlot(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.periods = pd.Series(pd.date_range(start='2022-01-01', end='2022-01-10').tolist() * 2)
        self.panel_split = PanelSplit(periods = self.periods, n_splits=3)

    def test_plot(self):
        plot_splits(self.panel_split, show=False)
        plot_splits(self.panel_split, show=True)


if __name__ == '__main__':
    unittest.main()