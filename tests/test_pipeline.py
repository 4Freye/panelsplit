import unittest
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin, clone
from panelsplit.pipeline import SequentialCVPipeline

# Dummy transformer that doubles the input.
class DoubleTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X * 2

# Dummy regressor that returns the sum of each row.
class DummyRegressor(BaseEstimator):

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self
    def predict(self, X):
        return np.sum(X, axis=1)
    def score(self, X, y = None):
        return 1.0

# Dummy classifier that supports predict_proba, predict_log_proba, and score.
class DummyClassifier(BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)
    def predict_proba(self, X):
        return np.tile(np.array([0.5, 0.5]), (X.shape[0], 1))
    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))
    def score(self, X, y):
        return 0.5

# Dummy CV splitter that yields two folds.
class DummyCV:
    def split(self, X, y=None, groups=None):
        n = len(X)
        indices = np.arange(n)
        # First fold: first half train, second half test.
        yield (indices[:n//2], indices[n//2:])
        # Second fold: second half train, first half test.
        yield (indices[n//2:], indices[:n//2])
    def get_n_splits(self, X, y=None, groups=None):
        return 2

class TestSequentialCVPipeline(unittest.TestCase):

    def setUp(self):
        # Create a simple dataset.
        self.X = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9],
                           [10, 11, 12]])
        self.y = np.array([1, 2, 3, 4])

    def test_non_cv_pipeline(self):
        # Pipeline with non-CV steps.
        pipe = SequentialCVPipeline([
            ('double', DoubleTransformer(), None),
            ('dummy', DummyRegressor(), None)
        ], verbose=False)

        # The transformer should double the input.
        Xt = pipe[:-1].fit_transform(self.X)
        np.testing.assert_array_equal(Xt, self.X * 2)
        # The regressor sums the doubled rows.
        pred = pipe.fit_predict(self.X)
        expected = np.sum(self.X * 2, axis=1)
        np.testing.assert_array_equal(pred, expected)

    def test_cv_pipeline(self):
        # Pipeline with CV on the first step.
        dummy_cv = DummyCV()
        pipe = SequentialCVPipeline([
            ('double', DoubleTransformer(), dummy_cv),
            ('dummy', DummyRegressor(), None)
        ], verbose=False)
        pipe.fit(self.X, self.y)
        # Check that the final estimator is fitted.
        self.assertIn('dummy', pipe.fitted_steps_)
        # Transform and predict.
        pred = pipe.predict(self.X)
        expected = np.sum(self.X * 2, axis=1)
        np.testing.assert_array_equal(pred, expected)

    def test_fit_predict(self):
        # Test the dynamic fit_predict method.
        pipe = SequentialCVPipeline([
            ('double', DoubleTransformer(), None),
            ('dummy', DummyRegressor(), None)
        ], verbose=False)
        pred = pipe.fit_predict(self.X, self.y)
        expected = np.sum(self.X * 2, axis=1)
        np.testing.assert_array_equal(pred, expected)

    def test_fit_predict_proba(self):
        # Test a classifier pipeline with dynamic fit_predict_proba.
        pipe = SequentialCVPipeline([
            ('double', DoubleTransformer(), None),
            ('dummy', DummyClassifier(), None)
        ], verbose=False)
        # Ensure the dynamic method was injected.
        self.assertTrue(hasattr(pipe, 'fit_predict_proba'))
        prob = pipe.fit_predict_proba(self.X, self.y)
        expected = np.tile(np.array([0.5, 0.5]), (self.X.shape[0], 1))
        np.testing.assert_array_almost_equal(prob, expected)

    def test_fit_score(self):
        # Test the dynamic fit_score method.
        pipe = SequentialCVPipeline([
            ('double', DoubleTransformer(), None),
            ('dummy', DummyRegressor(), None)
        ], verbose=False)
        score = pipe.fit_score(self.X, self.y)
        self.assertEqual(score, 1.0)

    def test_passthrough_step(self):
        # Test a step set to None (passthrough).
        pipe = SequentialCVPipeline([
            ('double', None, None),
            ('dummy', DummyRegressor(), None)
        ], verbose=False)
        pipe.fit(self.X, self.y)
        pred = pipe.predict(self.X)
        # In this case, X should be passed unchanged to DummyRegressor.
        expected = np.sum(self.X, axis=1)
        np.testing.assert_array_equal(pred, expected)

    def test_invalid_step(self):
        # Test that an invalid step raises a ValueError.
        with self.assertRaises(ValueError):
            SequentialCVPipeline([
                ('double', DoubleTransformer()),  # missing cv
                ('dummy', DummyRegressor(), None)
            ], verbose=False)

    def test_cv_single_value_output(self):
        # Test when a transformer returns a single value for a CV fold.
        class SingleValueTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                # Return the mean as a single value.
                return np.full_like(X, np.mean(X))
        dummy_cv = DummyCV()
        pipe = SequentialCVPipeline([
            ('single', SingleValueTransformer(), dummy_cv),
            ('dummy', DummyRegressor(), None)
        ], verbose=False)
        pipe.fit(self.X, self.y)
        pred = pipe.predict(self.X)
        # We don't assert a specific value here; just check the output length.
        self.assertEqual(pred.shape[0], self.X.shape[0])

#%%
#%%
if __name__ == '__main__':
    unittest.main()
