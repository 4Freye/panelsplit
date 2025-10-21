"""
Test suite for narwhals dataframe-agnostic functionality in PanelSplit.

This module tests that PanelSplit works with both pandas and other narwhals-supported
dataframe libraries to ensure dataframe-agnostic behavior.
"""

import unittest
import pytest
import numpy as np
import pandas as pd
import narwhals as nw
from panelsplit.cross_validation import PanelSplit
from panelsplit.application import cross_val_fit_predict
from panelsplit.pipeline import SequentialCVPipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


class TestNarwhalsCompatibility(unittest.TestCase):
    """Test narwhals dataframe-agnostic functionality."""

    def setUp(self):
        """Set up test data with different dataframe types."""
        # Create pandas test data
        self.periods_pandas = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])
        self.X_pandas = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16],
            }
        )
        self.y_pandas = pd.Series([10, 20, 30, 40, 50, 60, 70, 80])

        # Create numpy equivalents for comparison
        self.periods_numpy = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        self.X_numpy = np.array(
            [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12], [7, 14], [8, 16]]
        )
        self.y_numpy = np.array([10, 20, 30, 40, 50, 60, 70, 80])

    def test_panelsplit_with_pandas_series(self):
        """Test PanelSplit with pandas Series for periods."""
        ps = PanelSplit(periods=self.periods_pandas, n_splits=2)
        splits = ps.split()

        self.assertEqual(len(splits), 2)
        self.assertEqual(ps.get_n_splits(), 2)

        # Test that splits contain integer arrays
        for train_idx, test_idx in splits:
            self.assertIsInstance(train_idx, np.ndarray)
            self.assertIsInstance(test_idx, np.ndarray)
            self.assertEqual(train_idx.dtype, np.int64)
            self.assertEqual(test_idx.dtype, np.int64)

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

        for (train_pd, test_pd), (train_np, test_np) in zip(
            splits_pandas, splits_numpy
        ):
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

        pipeline = SequentialCVPipeline(
            [
                ("scaler", StandardScaler(), None),
                (
                    "regressor",
                    RandomForestRegressor(n_estimators=5, random_state=42),
                    ps,
                ),
            ]
        )

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
        self.assertTrue(hasattr(df_nw, "columns"))

        # Convert back to native
        df_native = nw.to_native(df_nw)
        pd.testing.assert_frame_equal(df_native, self.X_pandas)

        # Test with pandas Series
        series_nw = nw.from_native(self.y_pandas, pass_through=True)
        self.assertTrue(hasattr(series_nw, "to_numpy"))

        # Convert back to native (check if it's already native first)
        from panelsplit.utils.validation import _safe_indexing

        series_native = _safe_indexing(series_nw, to_native=True)
        pd.testing.assert_series_equal(series_native, self.y_pandas)

    def test_snapshots_generation_with_pandas(self):
        """Test snapshot generation with pandas DataFrames."""
        data = pd.DataFrame(
            {
                "value": [10, 20, 30, 40, 50, 60, 70, 80],
                "period": [1, 1, 2, 2, 3, 3, 4, 4],
            }
        )

        ps = PanelSplit(periods=data["period"], n_splits=2)

        try:
            snapshots = ps.gen_snapshots(data, period_col="period")

            # Should return a DataFrame-like object
            self.assertTrue(
                hasattr(snapshots, "columns") or isinstance(snapshots, pd.DataFrame)
            )

            # Should have split column
            if hasattr(snapshots, "columns"):
                self.assertIn("split", snapshots.columns)

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

    def test_transformer_with_y_none(self):
        """Test transformers that don't require y (like KNNImputer) work with y=None."""
        # Create data with missing values for imputation
        X_with_missing = self.X_pandas.copy()
        X_with_missing.iloc[1, 0] = np.nan  # Add missing value
        X_with_missing.iloc[3, 1] = np.nan  # Add missing value

        ps = PanelSplit(periods=self.periods_pandas, n_splits=2)

        # Test KNNImputer with y=None (transformers don't need y)
        imputer = KNNImputer(n_neighbors=2)

        try:
            # This should work without error even with y=None
            preds, fitted_models = cross_val_fit_predict(
                imputer, X_with_missing, y=None, cv=ps, method="transform", n_jobs=1
            )

            # Check that predictions have the right shape
            self.assertIsInstance(preds, np.ndarray)
            self.assertEqual(len(fitted_models), 2)  # 2 splits

            # Cross-validation returns out-of-fold predictions (only test sets)
            # With 2 splits and 2 test samples each, we expect 4 total predictions
            expected_test_samples = sum(len(test) for _, test in ps.split())
            self.assertEqual(
                preds.shape[0], expected_test_samples
            )  # Out-of-fold predictions only
            self.assertEqual(
                preds.shape[1], X_with_missing.shape[1]
            )  # Same number of features

            # Check that missing values were imputed (no NaN values in output)
            self.assertFalse(np.isnan(preds).any())

        except Exception as e:
            self.fail(f"cross_val_fit_predict failed with y=None and KNNImputer: {e}")

    def test_transformer_with_y_none_and_sample_weight(self):
        """Test transformers with y=None and sample_weight work correctly."""
        # Create data with missing values for imputation
        X_with_missing = self.X_pandas.copy()
        X_with_missing.iloc[1, 0] = np.nan  # Add missing value
        X_with_missing.iloc[3, 1] = np.nan  # Add missing value

        # Create sample weights
        sample_weight = np.array([1.0, 2.0, 1.5, 1.0, 2.0, 1.5, 1.0, 2.0])

        ps = PanelSplit(periods=self.periods_pandas, n_splits=2)

        # Test KNNImputer with y=None and sample_weight
        imputer = KNNImputer(n_neighbors=2)

        try:
            # This should work without error even with y=None and sample_weight
            with pytest.warns(
                UserWarning,
                match="does not support sample_weight. sample_weight will be ignored.",
            ):
                preds, fitted_models = cross_val_fit_predict(
                    imputer,
                    X_with_missing,
                    y=None,
                    cv=ps,
                    method="transform",
                    sample_weight=sample_weight,
                    n_jobs=1,
                )

            # Check that predictions have the right shape
            self.assertIsInstance(preds, np.ndarray)
            self.assertEqual(len(fitted_models), 2)  # 2 splits

            # Cross-validation returns out-of-fold predictions (only test sets)
            # With 2 splits and 2 test samples each, we expect 4 total predictions
            expected_test_samples = sum(len(test) for _, test in ps.split())
            self.assertEqual(
                preds.shape[0], expected_test_samples
            )  # Out-of-fold predictions only
            self.assertEqual(
                preds.shape[1], X_with_missing.shape[1]
            )  # Same number of features

            # Check that missing values were imputed (no NaN values in output)
            self.assertFalse(np.isnan(preds).any())

        except Exception as e:
            self.fail(
                f"cross_val_fit_predict failed with y=None, sample_weight and KNNImputer: {e}"
            )

    def test_pandas_non_default_index(self):
        """Test that narwhals works correctly with pandas objects having non-default indices."""
        # Create pandas data with non-default index
        custom_index = ["a", "b", "c", "d", "e", "f", "g", "h"]
        periods_custom = pd.Series([1, 1, 2, 2, 3, 3, 4, 4], index=custom_index)
        X_custom = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16],
            },
            index=custom_index,
        )
        y_custom = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], index=custom_index)

        ps = PanelSplit(periods=periods_custom, n_splits=2)
        model = RandomForestRegressor(n_estimators=5, random_state=42)

        try:
            # This should work without KeyError even with custom index
            preds, fitted_models = cross_val_fit_predict(
                model, X_custom, y_custom, ps, n_jobs=1
            )

            # Check that predictions have the right shape
            self.assertIsInstance(preds, np.ndarray)
            self.assertEqual(len(fitted_models), 2)  # 2 splits

            # Should have predictions for test samples only
            expected_test_samples = sum(len(test) for _, test in ps.split())
            self.assertEqual(len(preds), expected_test_samples)

        except Exception as e:
            self.fail(f"cross_val_fit_predict failed with custom pandas index: {e}")

    def test_label_generation_with_custom_index(self):
        """Test label generation with pandas objects having non-default indices."""
        # Create pandas data with non-default index
        custom_index = ["x", "y", "z", "w", "a", "b", "c", "d"]
        periods_custom = pd.Series([1, 1, 2, 2, 3, 3, 4, 4], index=custom_index)
        labels_custom = pd.Series(
            ["A", "B", "C", "D", "E", "F", "G", "H"], index=custom_index
        )

        ps = PanelSplit(periods=periods_custom, n_splits=2)

        try:
            # This should work without KeyError even with custom index
            train_labels = ps.gen_train_labels(labels_custom)
            test_labels = ps.gen_test_labels(labels_custom)

            # Should return pandas objects or arrays
            self.assertIsInstance(train_labels, (pd.Series, np.ndarray))
            self.assertIsInstance(test_labels, (pd.Series, np.ndarray))

        except Exception as e:
            self.fail(f"Label generation failed with custom pandas index: {e}")

    def test_pipeline_fit_score_method(self):
        """Test that fit_score method works correctly (GitHub issue fix)."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from panelsplit.pipeline import SequentialCVPipeline

        # Test data from the GitHub issue
        period = np.array([1, 2, 3, 4])
        X = np.array([[4, 1], [1, 3], [5, 7], [6, 7]])
        y = np.array([0, 1, 1, 0])

        ps_1 = PanelSplit(periods=period, n_splits=2, include_first_train_in_test=True)
        ps_2 = PanelSplit(periods=period, n_splits=2)

        pipeline = SequentialCVPipeline(
            [
                ("scaler", StandardScaler(), ps_1),
                ("classifier", LogisticRegression(), ps_2),
            ]
        )

        try:
            # This used to fail with: TypeError: ClassifierMixin.score() missing 1 required positional argument: 'y'
            result = pipeline.fit_score(X, y)

            # Check that the result is valid
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 2)  # Should have 2 scores (one per split)
            self.assertTrue(
                all(isinstance(score, (int, float, np.number)) for score in result)
            )

        except Exception as e:
            self.fail(f"fit_score method failed: {e}")

    def test_pipeline_score_method(self):
        """Test that score method works correctly after fitting."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from panelsplit.pipeline import SequentialCVPipeline

        # Test data
        period = np.array([1, 2, 3, 4])
        X = np.array([[4, 1], [1, 3], [5, 7], [6, 7]])
        y = np.array([0, 1, 1, 0])

        ps_1 = PanelSplit(periods=period, n_splits=2, include_first_train_in_test=True)
        ps_2 = PanelSplit(periods=period, n_splits=2)

        pipeline = SequentialCVPipeline(
            [
                ("scaler", StandardScaler(), ps_1),
                ("classifier", LogisticRegression(), ps_2),
            ]
        )

        try:
            # First fit the pipeline
            pipeline.fit(X, y)

            # Then test the score method
            result = pipeline.score(X, y)

            # Check that the result is valid
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 2)  # Should have 2 scores (one per split)
            self.assertTrue(
                all(isinstance(score, (int, float, np.number)) for score in result)
            )

        except Exception as e:
            self.fail(f"score method failed: {e}")

    def test_dynamic_method_inspection(self):
        """Test that dynamic method inspection works correctly for custom estimators."""
        from sklearn.base import BaseEstimator
        from panelsplit.pipeline import _call_method_with_correct_args

        class CustomEstimator(BaseEstimator):
            def fit(self, X, y=None):
                return self

            def predict(self, X):
                return np.ones(len(X))

            def score(self, X, y):  # Requires y
                return 0.95

            def transform(self, X):  # Doesn't require y
                return X * 2

            def custom_score(
                self, X, y, sample_weight=None
            ):  # y required, sample_weight optional
                return 0.8

        estimator = CustomEstimator()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        # Test methods that don't require y
        pred = _call_method_with_correct_args(estimator, "predict", X)
        self.assertEqual(pred.shape, (2,))

        transform = _call_method_with_correct_args(estimator, "transform", X)
        self.assertEqual(transform.shape, (2, 2))

        # Test methods that require y
        score = _call_method_with_correct_args(estimator, "score", X, y)
        self.assertEqual(score, 0.95)

        custom_score = _call_method_with_correct_args(estimator, "custom_score", X, y)
        self.assertEqual(custom_score, 0.8)

        # Test error case: method requires y but y is None
        with self.assertRaises(ValueError) as cm:
            _call_method_with_correct_args(estimator, "score", X, None)
        self.assertIn("requires y parameter but y is None", str(cm.exception))

    def test_multiindex_integration(self):
        """Test PanelSplit with pandas MultiIndex structures."""
        from panelsplit.utils.validation import get_index_or_col_from_df

        # Create MultiIndex data
        entities = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        periods = [1, 2, 3, 1, 2, 3, 1, 2, 3]

        index = pd.MultiIndex.from_arrays(
            [entities, periods], names=["entity", "period"]
        )

        data = {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            "target": [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0],
        }

        df = pd.DataFrame(data, index=index)

        # Test extracting index levels
        periods_extracted = get_index_or_col_from_df(df, "period")
        entities_extracted = get_index_or_col_from_df(df, "entity")

        self.assertIsInstance(periods_extracted, pd.Index)
        self.assertIsInstance(entities_extracted, pd.Index)
        self.assertEqual(len(periods_extracted), 9)
        self.assertEqual(len(entities_extracted), 9)

        # Test PanelSplit with MultiIndex periods
        X = df[["feature1", "feature2"]]
        y = df["target"]

        # Test with period level
        ps_periods = PanelSplit(periods=periods_extracted, n_splits=2, test_size=1)
        splits_periods = list(ps_periods.split(X, y))

        self.assertEqual(len(splits_periods), 2)
        for train_idx, test_idx in splits_periods:
            self.assertTrue(len(train_idx) > 0)
            self.assertTrue(len(test_idx) > 0)

        # Test with entity level
        ps_entities = PanelSplit(periods=entities_extracted, n_splits=2, test_size=1)
        splits_entities = list(ps_entities.split(X, y))

        self.assertEqual(len(splits_entities), 2)

        # Test cross-validation with MultiIndex
        preds, models = cross_val_fit_predict(
            RandomForestRegressor(n_estimators=5, random_state=42), X, y, ps_periods
        )

        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(len(preds), 6)  # Should have predictions for test sets
        self.assertEqual(len(models), 2)  # Should have 2 fitted models

        # Test edge case: name conflict (column and index level with same name)
        df_conflict = df.copy()
        df_conflict["period"] = df_conflict.index.get_level_values("period") * 10

        # Should warn and default to index level
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_index_or_col_from_df(df_conflict, "period")

            # Should have generated a warning
            self.assertTrue(len(w) > 0)
            self.assertIn("found in both", str(w[0].message))

            # Should return the index level (not the column)
            expected_index_values = df_conflict.index.get_level_values("period")
            np.testing.assert_array_equal(result.values, expected_index_values.values)


if __name__ == "__main__":
    unittest.main()
