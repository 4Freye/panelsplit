"""
Test suite for narwhals dataframe-agnostic functionality in PanelSplit.

This module tests that PanelSplit works with both pandas and other narwhals-supported
dataframe libraries to ensure dataframe-agnostic behavior.
"""

import unittest
import numpy as np
import pandas as pd
import narwhals as nw
from panelsplit.cross_validation import PanelSplit
from panelsplit.application import cross_val_fit_predict
from panelsplit.pipeline import SequentialCVPipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


class TestNarwhalsCompatibility(unittest.TestCase):
    """Test narwhals dataframe-agnostic functionality."""

    def setUp(self):
        """Set up test data with different dataframe types."""
        # Create pandas test data
        self.periods_pandas = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])
        self.X_pandas = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [2, 4, 6, 8, 10, 12, 14, 16]
        })
        self.y_pandas = pd.Series([10, 20, 30, 40, 50, 60, 70, 80])

        # Create numpy equivalents for comparison
        self.periods_numpy = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        self.X_numpy = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12], [7, 14], [8, 16]])
        self.y_numpy = np.array([10, 20, 30, 40, 50, 60, 70, 80])

    def test_panelsplit_with_pandas_series(self):
        """Test PanelSplit with pandas Series for periods."""
        ps = PanelSplit(periods=self.periods_pandas, n_splits=2)
        splits = ps.split()
        
        self.assertEqual(len(splits), 2)
        self.assertEqual(ps.get_n_splits(), 2)
        
        # Test that splits contain boolean arrays
        for train_idx, test_idx in splits:
            self.assertIsInstance(train_idx, np.ndarray)
            self.assertIsInstance(test_idx, np.ndarray)
            self.assertEqual(train_idx.dtype, bool)
            self.assertEqual(test_idx.dtype, bool)

    def test_panelsplit_with_numpy_array(self):
        """Test PanelSplit with numpy array for periods."""
        ps = PanelSplit(periods=self.periods_numpy, n_splits=2)
        splits = ps.split()
        
        self.assertEqual(len(splits), 2)
        self.assertEqual(ps.get_n_splits(), 2)

    def test_cross_validation_pandas_vs_numpy(self):
        """Test that results are consistent between pandas and numpy inputs."""
        # Test with pandas
        ps_pandas = PanelSplit(periods=self.periods_pandas, n_splits=2)
        splits_pandas = ps_pandas.split()
        
        # Test with numpy
        ps_numpy = PanelSplit(periods=self.periods_numpy, n_splits=2)
        splits_numpy = ps_numpy.split()
        
        # Results should be equivalent
        self.assertEqual(len(splits_pandas), len(splits_numpy))
        
        for (train_pd, test_pd), (train_np, test_np) in zip(splits_pandas, splits_numpy):
            np.testing.assert_array_equal(train_pd, train_np)
            np.testing.assert_array_equal(test_pd, test_np)

    def test_application_module_with_pandas(self):
        """Test application module functions with pandas DataFrames."""
        ps = PanelSplit(periods=self.periods_pandas, n_splits=2)
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        # Test cross_val_fit_predict
        try:
            preds, fitted_models = cross_val_fit_predict(
                model, self.X_pandas, self.y_pandas, ps, n_jobs=1
            )
            
            # Check that predictions have the right shape
            self.assertIsInstance(preds, np.ndarray)
            self.assertEqual(len(fitted_models), 2)  # 2 splits
            
        except Exception as e:
            self.fail(f"cross_val_fit_predict failed with pandas: {e}")

    def test_application_module_with_numpy(self):
        """Test application module functions with numpy arrays."""
        ps = PanelSplit(periods=self.periods_numpy, n_splits=2)
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        # Test cross_val_fit_predict with numpy arrays
        try:
            preds, fitted_models = cross_val_fit_predict(
                model, self.X_numpy, self.y_numpy, ps, n_jobs=1
            )
            
            # Check that predictions have the right shape
            self.assertIsInstance(preds, np.ndarray)
            self.assertEqual(len(fitted_models), 2)  # 2 splits
            
        except Exception as e:
            self.fail(f"cross_val_fit_predict failed with numpy: {e}")

    def test_pipeline_with_pandas(self):
        """Test SequentialCVPipeline with pandas DataFrames."""
        ps = PanelSplit(periods=self.periods_pandas, n_splits=2)
        
        pipeline = SequentialCVPipeline([
            ('scaler', StandardScaler(), None),
            ('regressor', RandomForestRegressor(n_estimators=5, random_state=42), ps)
        ])
        
        try:
            # Test fit_predict
            predictions = pipeline.fit_predict(self.X_pandas, self.y_pandas)
            self.assertIsInstance(predictions, np.ndarray)
            
        except Exception as e:
            self.fail(f"SequentialCVPipeline failed with pandas: {e}")

    def test_label_generation_with_pandas(self):
        """Test label generation functions with pandas inputs."""
        ps = PanelSplit(periods=self.periods_pandas, n_splits=2)
        
        # Test with pandas Series
        train_labels = ps.gen_train_labels(self.y_pandas)
        test_labels = ps.gen_test_labels(self.y_pandas)
        
        # Should return pandas objects
        self.assertIsInstance(train_labels, (pd.Series, np.ndarray))
        self.assertIsInstance(test_labels, (pd.Series, np.ndarray))

    def test_label_generation_with_numpy(self):
        """Test label generation functions with numpy arrays."""
        ps = PanelSplit(periods=self.periods_numpy, n_splits=2)
        
        # Test with numpy array
        train_labels = ps.gen_train_labels(self.y_numpy)
        test_labels = ps.gen_test_labels(self.y_numpy)
        
        # Should return numpy arrays
        self.assertIsInstance(train_labels, np.ndarray)
        self.assertIsInstance(test_labels, np.ndarray)

    def test_narwhals_native_conversion(self):
        """Test narwhals from_native and to_native functions work correctly."""
        # Test with pandas DataFrame
        df_nw = nw.from_native(self.X_pandas, pass_through=True)
        self.assertTrue(hasattr(df_nw, 'columns'))
        
        # Convert back to native
        df_native = nw.to_native(df_nw)
        pd.testing.assert_frame_equal(df_native, self.X_pandas)
        
        # Test with pandas Series
        series_nw = nw.from_native(self.y_pandas, pass_through=True)
        self.assertTrue(hasattr(series_nw, 'to_numpy'))
        
        # Convert back to native (check if it's already native first)
        from panelsplit.application import _safe_to_native
        series_native = _safe_to_native(series_nw)
        pd.testing.assert_series_equal(series_native, self.y_pandas)

    def test_snapshots_generation_with_pandas(self):
        """Test snapshot generation with pandas DataFrames."""
        data = pd.DataFrame({
            'value': [10, 20, 30, 40, 50, 60, 70, 80],
            'period': [1, 1, 2, 2, 3, 3, 4, 4]
        })
        
        ps = PanelSplit(periods=data['period'], n_splits=2)
        
        try:
            snapshots = ps.gen_snapshots(data, period_col='period')
            
            # Should return a DataFrame-like object
            self.assertTrue(hasattr(snapshots, 'columns') or isinstance(snapshots, pd.DataFrame))
            
            # Should have split column
            if hasattr(snapshots, 'columns'):
                self.assertIn('split', snapshots.columns)
            
        except Exception as e:
            self.fail(f"gen_snapshots failed with pandas: {e}")

    def test_error_handling_with_invalid_inputs(self):
        """Test that appropriate errors are raised for invalid inputs."""
        # Test with invalid periods type
        with self.assertRaises((ValueError, TypeError)):
            PanelSplit(periods="invalid", n_splits=2)
        
        # Test with invalid labels type
        ps = PanelSplit(periods=self.periods_pandas, n_splits=2)
        with self.assertRaises(TypeError):
            ps.gen_train_labels("invalid")

    def test_dataframe_agnostic_behavior(self):
        """Test that the same operations work consistently across different input types."""
        # Create equivalent data in different formats
        periods_list = [1, 1, 2, 2, 3, 3, 4, 4]
        
        # Test with pandas Series
        ps_pandas = PanelSplit(periods=pd.Series(periods_list), n_splits=2)
        splits_pandas = ps_pandas.split()
        
        # Test with numpy array
        ps_numpy = PanelSplit(periods=np.array(periods_list), n_splits=2)
        splits_numpy = ps_numpy.split()
        
        # Test with list (should be converted internally)
        ps_list = PanelSplit(periods=periods_list, n_splits=2)
        splits_list = ps_list.split()
        
        # All should produce equivalent results
        for i in range(len(splits_pandas)):
            np.testing.assert_array_equal(splits_pandas[i][0], splits_numpy[i][0])
            np.testing.assert_array_equal(splits_pandas[i][1], splits_numpy[i][1])
            np.testing.assert_array_equal(splits_pandas[i][0], splits_list[i][0])
            np.testing.assert_array_equal(splits_pandas[i][1], splits_list[i][1])


if __name__ == '__main__':
    unittest.main()