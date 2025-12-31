import unittest
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from panelsplit.pipeline import SequentialCVPipeline
from panelsplit.cross_validation import PanelSplit
import pandas as pd


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

    def score(self, X, y=None):
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


class X1ToYRegressor(BaseEstimator):
    """The first array in X is the output when predicting Y."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[:, 0]


# Dummy CV splitter that yields two folds.
class DummyCV:
    def split(self, X, y=None, groups=None):
        n = len(X)
        indices = np.arange(n)
        # First fold: first half train, second half test.
        yield (indices[: n // 2], indices[n // 2 :])
        # Second fold: second half train, first half test.
        yield (indices[n // 2 :], indices[: n // 2])

    def get_n_splits(self, X, y=None, groups=None):
        return 2


class TestSequentialCVPipeline(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset.
        self.X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.y = np.array([1, 2, 3, 4])

    def test_non_cv_pipeline(self):
        # Pipeline with non-CV steps.
        pipe = SequentialCVPipeline(
            steps=[("double", DoubleTransformer()), ("dummy", DummyRegressor())],
            cv_steps=[None, None],
            verbose=False,
        )

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
        pipe = SequentialCVPipeline(
            steps=[
                ("double", DoubleTransformer()),
                ("dummy", DummyRegressor()),
            ],
            cv_steps=[dummy_cv, None],
            verbose=False,
        )
        pipe.fit(self.X, self.y)
        # Check that the final estimator is fitted.
        self.assertIn("dummy", pipe.fitted_steps_)
        # Transform and predict.
        pred = pipe.predict(self.X)
        expected = np.sum(self.X * 2, axis=1)
        np.testing.assert_array_equal(pred, expected)

    def test_fit_predict(self):
        # Test the dynamic fit_predict method.
        pipe = SequentialCVPipeline(
            steps=[("double", DoubleTransformer()), ("dummy", DummyRegressor())],
            cv_steps=[None, None],
            verbose=False,
        )
        pred = pipe.fit_predict(self.X, self.y)
        expected = np.sum(self.X * 2, axis=1)
        np.testing.assert_array_equal(pred, expected)

    def test_fit_predict_proba(self):
        # Test a classifier pipeline with dynamic fit_predict_proba.
        pipe = SequentialCVPipeline(
            steps=[("double", DoubleTransformer()), ("dummy", DummyClassifier())],
            cv_steps=[None, None],
            verbose=False,
        )
        # Ensure the dynamic method was injected.
        self.assertTrue(hasattr(pipe, "fit_predict_proba"))
        prob = pipe.fit_predict_proba(self.X, self.y)
        expected = np.tile(np.array([0.5, 0.5]), (self.X.shape[0], 1))
        np.testing.assert_array_almost_equal(prob, expected)

    def test_fit_score(self):
        # Test the dynamic fit_score method.
        pipe = SequentialCVPipeline(
            steps=[("double", DoubleTransformer()), ("dummy", DummyRegressor())],
            cv_steps=[None, None],
            verbose=False,
        )
        score = pipe.fit_score(self.X, self.y)
        self.assertEqual(score, 1.0)

    def test_passthrough_step(self):
        # Test a step set to None (passthrough).
        pipe = SequentialCVPipeline(
            steps=[("double", None), ("dummy", DummyRegressor())],
            cv_steps=[None, None],
            verbose=False,
        )
        pipe.fit(self.X, self.y)
        pred = pipe.predict(self.X)
        # In this case, X should be passed unchanged to DummyRegressor.
        expected = np.sum(self.X, axis=1)
        np.testing.assert_array_equal(pred, expected)

    def test_invalid_step(self):
        # Test that an invalid step raises a ValueError.
        with self.assertRaises(ValueError):
            SequentialCVPipeline(
                steps=[
                    ("double", DoubleTransformer(), None),  # length of 3
                    ("dummy", DummyRegressor()),
                ],
                cv_steps=[None, None],
                verbose=False,
            )

    def test_cv_single_value_output(self):
        # Test when a transformer returns a single value for a CV fold.
        class SingleValueTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                # Return the mean as a single value.
                return np.full_like(X, np.mean(X))

        dummy_cv = DummyCV()
        pipe = SequentialCVPipeline(
            steps=[
                ("double", SingleValueTransformer()),
                ("dummy", DummyRegressor()),
            ],
            cv_steps=[dummy_cv, None],
            verbose=False,
        )
        pipe.fit(self.X, self.y)
        pred = pipe.predict(self.X)
        # We don't assert a specific value here; just check the output length.
        self.assertEqual(pred.shape[0], self.X.shape[0])

    def test_pipeline_indexing(self):
        pipe = SequentialCVPipeline(
            steps=[("double", DoubleTransformer()), ("dummy", DummyRegressor())],
            cv_steps=[None, 4],
            verbose=False,
        )
        # slice tests
        first_half = pipe[:1]
        second_half = pipe[-1:]

        for slice in [first_half, second_half]:
            for attr in ["cv_steps", "steps"]:
                self.assertTrue(len(getattr(slice, attr)) == 1)

        self.assertTrue(first_half.cv_steps == [None])
        self.assertTrue(second_half.cv_steps == [4])

        # int tests:
        self.assertTrue(isinstance(pipe[1], BaseEstimator))
        self.assertTrue(isinstance(pipe[0], BaseEstimator))


def test_indices_aligned():
    size = 100
    X = np.vstack([np.arange(size), np.arange(size)]).T
    n_sample = 100
    n_u_periods = 25
    y = np.arange(100)
    period = np.repeat([*range(n_u_periods)], n_sample // n_u_periods)
    # ensure things are properly shuffled:
    period = np.random.choice(period, size=n_sample, replace=False)
    model = X1ToYRegressor()
    ps = PanelSplit(period, n_splits=12)
    for cv_steps in [[None], [ps]]:
        pipe = SequentialCVPipeline(
            [("regression", model)], cv_steps=cv_steps, include_indices=False
        )

        pipe.fit(X, y)
        output = pipe.predict(X)
        if cv_steps != [None]:
            assert all(output == ps.gen_test_labels(y))
        else:
            assert all(output == y)

        pipe.set_params(**{"include_indices": True})
        pipe.fit(X, y)
        idx, preds = pipe.predict(X)
        if cv_steps == [None]:
            assert all(idx == y)
            assert all(preds == y)
            assert all(output == y)
        else:
            assert all(y[idx] == preds)


def test_score_order():
    # Expected behavior:
    # the score for the first test should be in the same position
    # as the order of appearance for the first test period
    # in periods.

    # setup the test- establish X, y, period, panelsplit, rf, and pipe
    n_sample = 12
    n_u_periods = 3
    X, y = (
        np.random.sample(size=(n_sample, 5)),
        np.random.choice([*range(5)], size=n_sample),
    )
    period = np.repeat([*range(3)], n_sample // n_u_periods)
    period = np.random.choice(period, size=n_sample, replace=False)

    ps = PanelSplit(period, n_splits=2)
    rf = RandomForestRegressor(n_estimators=2, random_state=1)

    pipe = SequentialCVPipeline(
        [("rf", rf)],
        cv_steps=[ps],
    )

    pipe.fit(X, y)

    _, first_test_indices, first_fitted_estimator = pipe.__dict__["fitted_steps_"][
        "rf"
    ][0]
    isolated_score = first_fitted_estimator.score(
        X[first_test_indices], y[first_test_indices]
    )

    # the first test indices are indeed all 1- the first test period.
    assert all(period[first_test_indices] == 1)

    scores = pipe.score(X, y)

    where_is_first_score = np.where(scores == isolated_score)[0]
    unique_periods = pd.Series(period[period != 0]).drop_duplicates().to_numpy()

    where_is_first_instance_of_first_test_period = np.where(unique_periods == 1)[0]
    assert where_is_first_instance_of_first_test_period == where_is_first_score


# %%
# %%
if __name__ == "__main__":
    unittest.main()
