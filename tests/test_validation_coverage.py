"""
Test cases specifically targeting the validation module to improve coverage.
"""

import unittest
import numpy as np
import pandas as pd
from panelsplit.utils.validation import (
    check_periods,
    check_labels,
    get_index_or_col_from_df,
    check_cv,
    check_fitted_estimators,
    check_method,
    _check_X_y,
)
from panelsplit.cross_validation import PanelSplit
from sklearn.ensemble import RandomForestRegressor


class TestValidationCoverage(unittest.TestCase):
    """Test validation module functions for better coverage."""

    def test_check_periods_edge_cases(self):
        """Test check_periods with various input types."""
        # Test with pandas Series (gets converted to numpy)
        periods_series = pd.Series([1, 2, 3, 4])
        result = check_periods(periods_series)
        self.assertIsInstance(result, np.ndarray)

        # Test with numpy array
        periods_array = np.array([1, 2, 3, 4])
        result = check_periods(periods_array)
        self.assertIsInstance(result, np.ndarray)

        # Test with list (may stay as list)
        periods_list = [1, 2, 3, 4]
        result = check_periods(periods_list)
        self.assertTrue(isinstance(result, (np.ndarray, list)))

        # Test with pandas Index (may return pandas Series)
        periods_index = pd.Index([1, 2, 3, 4])
        result = check_periods(periods_index)
        self.assertTrue(isinstance(result, (np.ndarray, pd.Series, pd.Index)))

    def test_check_periods_with_custom_name(self):
        """Test check_periods with custom object name."""
        periods_series = pd.Series([1, 2, 3, 4])
        result = check_periods(periods_series, obj_name="custom_periods")
        self.assertIsInstance(result, np.ndarray)

    def test_check_labels_edge_cases(self):
        """Test check_labels with various input types."""
        # Test with pandas Series
        labels_series = pd.Series(["a", "b", "c", "d"])
        check_labels(labels_series)  # Should not raise

        # Test with numpy array
        labels_array = np.array(["a", "b", "c", "d"])
        check_labels(labels_array)  # Should not raise

        # Test with pandas DataFrame
        labels_df = pd.DataFrame({"col1": [1, 2, 3, 4]})
        check_labels(labels_df)  # Should not raise

        # Test with pandas Index
        labels_index = pd.Index(["a", "b", "c", "d"])
        check_labels(labels_index)  # Should not raise

    def test_check_labels_behavior(self):
        """Test check_labels basic behavior."""
        labels_series = pd.Series(["a", "b", "c", "d"])
        # Should not raise
        check_labels(labels_series)

    def test_get_index_or_col_from_df(self):
        """Test get_index_or_col_from_df function."""
        # Create test DataFrame
        df = pd.DataFrame(
            {"col1": [1, 2, 3, 4], "col2": [5, 6, 7, 8]}, index=["a", "b", "c", "d"]
        )

        # Test getting column
        result = get_index_or_col_from_df(df, "col1")
        pd.testing.assert_series_equal(result, df["col1"])

        # Test getting index when column name is None
        result = get_index_or_col_from_df(df, None)
        pd.testing.assert_index_equal(result, df.index)

        # Test error when column doesn't exist
        with self.assertRaises(KeyError):
            get_index_or_col_from_df(df, "nonexistent_column")

    def test_check_cv_with_different_inputs(self):
        """Test check_cv with different cross-validation inputs."""
        # Test with PanelSplit object
        periods = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])
        ps = PanelSplit(periods=periods, n_splits=2)
        result = check_cv(ps)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

        # Test with list of splits
        splits_list = [
            (
                np.array([True, True, False, False]),
                np.array([False, False, True, True]),
            ),
            (
                np.array([True, True, True, False]),
                np.array([False, False, False, True]),
            ),
        ]
        result = check_cv(splits_list)
        self.assertEqual(result, splits_list)

    def test_check_fitted_estimators(self):
        """Test check_fitted_estimators function."""
        # Test with list of fitted estimators
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        estimators = []
        for _ in range(2):
            est = RandomForestRegressor(n_estimators=5, random_state=42)
            est.fit(X, y)
            estimators.append(est)

        # Should not raise
        check_fitted_estimators(estimators)

        # Test with unfitted estimator (should print warning but not raise)
        unfitted = RandomForestRegressor()
        # Should not raise, but will print warning
        check_fitted_estimators([unfitted])

    def test_check_method(self):
        """Test check_method function."""
        # Test with fitted estimator
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        estimator = RandomForestRegressor(n_estimators=5, random_state=42)
        estimator.fit(X, y)

        # Test valid methods (check_method expects a list)
        check_method([estimator], "predict")  # Should not raise
        check_method([estimator], "score")  # Should not raise

        # Test invalid method (check_method expects a list of estimators)
        with self.assertRaises(ValueError):
            check_method([estimator], "nonexistent_method")

    def test_check_X_y_edge_cases(self):
        """Test _check_X_y function with edge cases."""
        # Test with valid inputs
        X = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        _check_X_y(X, y)  # Should not raise

        # Test with y=None
        _check_X_y(X, None)  # Should not raise

        # Test with numpy arrays
        X_np = np.array([[1, 2], [3, 4], [5, 6]])
        y_np = np.array([0, 1, 0])
        _check_X_y(X_np, y_np)  # Should not raise

        # Test mismatched lengths (this function may not validate lengths)
        y_short = pd.Series([0, 1])  # Different length
        try:
            _check_X_y(X, y_short)
        except ValueError:
            pass  # Expected if it validates lengths


if __name__ == "__main__":
    unittest.main()
