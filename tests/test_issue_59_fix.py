#!/usr/bin/env python3
"""
Tests for the dynamic method calling fix in SequentialCVPipeline (Issue #59).

This test ensures that the pipeline properly handles methods with different
signatures, especially the score() method which requires y parameter.
"""

import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from panelsplit.pipeline import SequentialCVPipeline, _call_method_with_correct_args
from panelsplit.cross_validation import PanelSplit


class TestDynamicMethodCalling(unittest.TestCase):
    """Test cases for the dynamic method calling fix (Issue #59)."""

    def setUp(self):
        """Set up test data."""
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 1, 0])
        self.periods = np.array([1, 2, 3, 4])

    def test_call_method_with_correct_args_predict(self):
        """Test _call_method_with_correct_args with predict (doesn't require y)."""
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X, self.y)

        # Predict doesn't require y
        predictions = _call_method_with_correct_args(model, "predict", self.X)
        self.assertEqual(len(predictions), len(self.X))

        # Should also work when y is provided (it will be ignored)
        predictions_with_y = _call_method_with_correct_args(
            model, "predict", self.X, self.y
        )
        np.testing.assert_array_equal(predictions, predictions_with_y)

    def test_call_method_with_correct_args_score(self):
        """Test _call_method_with_correct_args with score (requires y)."""
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X, self.y)

        # Score requires y
        score = _call_method_with_correct_args(model, "score", self.X, self.y)
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_call_method_with_correct_args_score_missing_y(self):
        """Test _call_method_with_correct_args raises error when score called without y."""
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X, self.y)

        # Score requires y, should raise ValueError when y is None
        with self.assertRaises(ValueError) as context:
            _call_method_with_correct_args(model, "score", self.X, None)

        self.assertIn("requires y parameter", str(context.exception))

    def test_call_method_with_correct_args_transform(self):
        """Test _call_method_with_correct_args with transform (doesn't require y)."""
        scaler = StandardScaler()
        scaler.fit(self.X)

        # Transform doesn't require y
        transformed = _call_method_with_correct_args(scaler, "transform", self.X)
        self.assertEqual(transformed.shape, self.X.shape)

    def test_pipeline_fit_score(self):
        """Test that fit_score works correctly in pipeline."""
        pipeline = SequentialCVPipeline(
            [
                ("scaler", StandardScaler(), None),
                ("classifier", LogisticRegression(max_iter=1000), None),
            ]
        )

        # This should work without errors (issue #59)
        score = pipeline.fit_score(self.X, self.y)
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_pipeline_score_after_fit(self):
        """Test that score works after fitting pipeline."""
        pipeline = SequentialCVPipeline(
            [
                ("scaler", StandardScaler(), None),
                ("classifier", LogisticRegression(max_iter=1000), None),
            ]
        )

        # Fit first
        pipeline.fit(self.X, self.y)

        # Then score (issue #59)
        score = pipeline.score(self.X, self.y)
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_pipeline_predict_after_fit(self):
        """Test that predict works after fitting pipeline."""
        pipeline = SequentialCVPipeline(
            [
                ("scaler", StandardScaler(), None),
                ("classifier", LogisticRegression(max_iter=1000), None),
            ]
        )

        # Fit first
        pipeline.fit(self.X, self.y)

        # Then predict
        predictions = pipeline.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))

    def test_pipeline_fit_predict(self):
        """Test that fit_predict works correctly in pipeline."""
        pipeline = SequentialCVPipeline(
            [
                ("scaler", StandardScaler(), None),
                ("classifier", LogisticRegression(max_iter=1000), None),
            ]
        )

        # This should work without errors
        predictions = pipeline.fit_predict(self.X, self.y)
        self.assertEqual(len(predictions), len(self.X))

    def test_pipeline_with_cv_fit_score(self):
        """Test that fit_score works with cross-validation splits."""
        ps = PanelSplit(periods=self.periods, n_splits=2)

        pipeline = SequentialCVPipeline(
            [
                ("scaler", StandardScaler(), None),
                ("classifier", LogisticRegression(max_iter=1000), ps),
            ]
        )

        # This should work without errors (issue #59 with CV)
        # With CV, score returns an array of scores (one per fold)
        score = pipeline.fit_score(self.X, self.y)
        self.assertIsInstance(score, np.ndarray)
        self.assertEqual(len(score), 2)  # 2 folds

    def test_pipeline_with_cv_score_after_fit(self):
        """Test that score works after fitting pipeline with CV."""
        ps = PanelSplit(periods=self.periods, n_splits=2)

        pipeline = SequentialCVPipeline(
            [
                ("scaler", StandardScaler(), None),
                ("classifier", LogisticRegression(max_iter=1000), ps),
            ]
        )

        # Fit first
        pipeline.fit(self.X, self.y)

        # Then score (issue #59 with CV)
        # With CV, score returns an array of scores (one per fold)
        score = pipeline.score(self.X, self.y)
        self.assertIsInstance(score, np.ndarray)
        self.assertEqual(len(score), 2)  # 2 folds

    def test_regressor_score(self):
        """Test that score works with regressors."""
        y_reg = np.array([1.0, 2.0, 3.0, 4.0])

        pipeline = SequentialCVPipeline(
            [
                ("scaler", StandardScaler(), None),
                ("regressor", RandomForestRegressor(), None),
            ]
        )

        # This should work without errors
        score = pipeline.fit_score(self.X, y_reg)
        self.assertIsInstance(score, (int, float))
        self.assertLessEqual(score, 1.0)

    def test_method_caching(self):
        """Test that method signature caching works correctly."""
        from panelsplit.pipeline import _METHOD_SIGNATURE_CACHE

        # Clear cache
        _METHOD_SIGNATURE_CACHE.clear()

        model = LogisticRegression(max_iter=1000)
        model.fit(self.X, self.y)

        # First call should populate cache
        _call_method_with_correct_args(model, "predict", self.X)
        cache_key = (type(model).__name__, "predict")
        self.assertIn(cache_key, _METHOD_SIGNATURE_CACHE)

        # Second call should use cached value
        cached_value = _METHOD_SIGNATURE_CACHE[cache_key]
        _call_method_with_correct_args(model, "predict", self.X)
        self.assertEqual(_METHOD_SIGNATURE_CACHE[cache_key], cached_value)


if __name__ == "__main__":
    unittest.main()
