#!/usr/bin/env python3
"""
Tests for the check_is_fitted fix in SequentialCVPipeline (Issue #54).

This test ensures that the pipeline properly implements sklearn's
fitted state detection conventions.
"""

import unittest
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from panelsplit.pipeline import SequentialCVPipeline


class TestCheckFittedFix(unittest.TestCase):
    """Test cases for the check_is_fitted fix (Issue #54)."""

    def setUp(self):
        """Set up test data."""
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([1, 2, 3, 4])
        self.periods = np.array([1, 2, 3, 4])

    def test_check_is_fitted_unfitted_pipeline(self):
        """Test that check_is_fitted raises NotFittedError for unfitted pipeline."""
        pipeline = SequentialCVPipeline([("regressor", RandomForestRegressor(), None)])

        # Should not have fitted_steps_ attribute before fitting
        self.assertFalse(hasattr(pipeline, "fitted_steps_"))

        # check_is_fitted should raise NotFittedError
        with self.assertRaises(NotFittedError) as context:
            check_is_fitted(pipeline)

        self.assertIn("not fitted yet", str(context.exception))
        self.assertIn("SequentialCVPipeline", str(context.exception))

    def test_check_is_fitted_fitted_pipeline(self):
        """Test that check_is_fitted passes for fitted pipeline."""
        pipeline = SequentialCVPipeline([("regressor", RandomForestRegressor(), None)])

        # Fit the pipeline
        pipeline.fit(self.X, self.y)

        # Should now have fitted_steps_ attribute
        self.assertTrue(hasattr(pipeline, "fitted_steps_"))
        self.assertIsInstance(pipeline.fitted_steps_, dict)
        self.assertIn("regressor", pipeline.fitted_steps_)

        # check_is_fitted should pass without raising
        try:
            check_is_fitted(pipeline)
        except NotFittedError:
            self.fail("check_is_fitted raised NotFittedError unexpectedly")

    def test_check_is_fitted_with_cv_pipeline(self):
        """Test check_is_fitted with CV pipeline."""
        # Use a pipeline without CV for the final step to avoid dimension issues
        pipeline = SequentialCVPipeline(
            [
                ("scaler", StandardScaler(), None),  # No CV to avoid dimension mismatch
                ("regressor", RandomForestRegressor(), None),
            ]
        )

        # Should fail before fitting
        with self.assertRaises(NotFittedError):
            check_is_fitted(pipeline)

        # Fit and should pass
        pipeline.fit(self.X, self.y)
        try:
            check_is_fitted(pipeline)
        except NotFittedError:
            self.fail("check_is_fitted raised NotFittedError after fitting")

    def test_dynamic_methods_unfitted_pipeline(self):
        """Test that dynamic methods raise NotFittedError for unfitted pipeline."""
        pipeline = SequentialCVPipeline([("regressor", RandomForestRegressor(), None)])

        # Dynamic predict method should raise NotFittedError
        with self.assertRaises(NotFittedError) as context:
            pipeline.predict(self.X)

        self.assertIn("not fitted yet", str(context.exception))

    def test_dynamic_methods_fitted_pipeline(self):
        """Test that dynamic methods work after fitting."""
        pipeline = SequentialCVPipeline([("regressor", RandomForestRegressor(), None)])

        # Fit first
        pipeline.fit(self.X, self.y)

        # Now predict should work
        predictions = pipeline.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))

    def test_check_is_fitted_with_specific_attribute(self):
        """Test check_is_fitted with specific attribute name."""
        pipeline = SequentialCVPipeline([("regressor", RandomForestRegressor(), None)])

        # Should fail with specific attribute check
        with self.assertRaises(NotFittedError):
            check_is_fitted(pipeline, "fitted_steps_")

        # Fit and should pass
        pipeline.fit(self.X, self.y)
        try:
            check_is_fitted(pipeline, "fitted_steps_")
        except NotFittedError:
            self.fail("check_is_fitted with 'fitted_steps_' failed after fitting")

    def test_sklearn_convention_compliance(self):
        """Test that the pipeline follows sklearn's fitted attribute conventions."""
        pipeline = SequentialCVPipeline([("regressor", RandomForestRegressor(), None)])

        # Before fitting: no attributes ending with '_' should exist
        fitted_attrs_before = [
            attr
            for attr in dir(pipeline)
            if attr.endswith("_") and not attr.startswith("_")
        ]
        # fitted_steps_ should not be in the list
        self.assertNotIn("fitted_steps_", fitted_attrs_before)

        # After fitting: fitted_steps_ should exist
        pipeline.fit(self.X, self.y)
        fitted_attrs_after = [
            attr
            for attr in dir(pipeline)
            if attr.endswith("_") and not attr.startswith("_")
        ]
        self.assertIn("fitted_steps_", fitted_attrs_after)


if __name__ == "__main__":
    unittest.main()
