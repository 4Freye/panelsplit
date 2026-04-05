"""
Test suite for setting and getting params from SequentialCVPipeline.

This module tests that SequentialCVPipeline.set_params and .get_params functions like an sklearn pipeline
"""

import unittest
from panelsplit.pipeline import SequentialCVPipeline
from sklearn.ensemble import RandomForestRegressor


class TestParamsInteraction(unittest.TestCase):
    def setUp(self):
        """Set up test data with different dataframe types."""
        # Create pandas test data
        self.pipe = SequentialCVPipeline(
            [("rf", RandomForestRegressor(n_estimators=10))], cv_steps=[None]
        )

    def test_get_params(self):
        params = self.pipe.get_params()
        n_estimators = params["rf__n_estimators"]
        self.assertEqual(10, n_estimators)  # should be 10.

    def test_set_params(self):
        self.pipe.set_params(rf__n_estimators=500)
        params = self.pipe.get_params()
        n_estimators = params["rf__n_estimators"]
        self.assertEqual(500, n_estimators)  # should be 500.
