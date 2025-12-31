# tests/test_metrics.py
import numpy as np
import pytest

from sklearn.base import BaseEstimator

from panelsplit.metrics import (
    _get_idx_from_last_cv,
    make_SequentialCV_scorer,
)


# ---------- Helpers / Fixtures ---------- #
class CVObjWithSplit:
    """Simple object exposing split() which yields (train_idx, test_idx)."""

    def __init__(self, splits):
        # splits: iterable of (train_idx, test_idx)
        self._splits = list(splits)

    def split(self):
        for s in self._splits:
            yield s


class DummyEstimator(BaseEstimator):
    """Minimal estimator used by make_SequentialCV_scorer tests.

    - must be deepcopy-able
    - expose cv_steps attribute (list-like)
    - provide set_params(include_indices=True)
    - implement predict / predict_proba / score as needed
    """

    def __init__(
        self, n_samples, preds, cv_steps=None, method="predict", return_group="test"
    ):
        # preds: 1d array-like with one entry per sample (for label preds)
        self.n_samples = n_samples
        self.preds = np.asarray(preds)
        self.cv_steps = list(cv_steps) if cv_steps is not None else [None]
        self.include_indices = False
        self.method = method
        self.return_group = return_group

    def set_params(self, **kwargs):
        # only care about include_indices
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def predict(self, X):
        # return (indices, predictions)
        idx = np.arange(self.n_samples)
        return idx, self.preds

    def predict_proba(self, X):
        idx = np.arange(self.n_samples)
        # assume preds already in 'prob' format for these tests
        return idx, self.preds

    def score(self, X, y):
        # trivial accuracy
        return float(np.mean(self.preds == y))


# ---------- Tests for _get_idx_from_last_cv ---------- #
def test_get_idx_from_last_cv_none():
    est = DummyEstimator(n_samples=3, preds=[1, 0, 1], cv_steps=[None])
    assert _get_idx_from_last_cv(est) is None


def test_get_idx_from_last_cv_with_split_method():
    # create two folds
    splits = [([0, 2], [1]), ([1], [0, 2])]
    cvobj = CVObjWithSplit(splits)
    est = DummyEstimator(n_samples=3, preds=[1, 0, 1], cv_steps=[None, cvobj])
    res = _get_idx_from_last_cv(est)
    assert isinstance(res, list)
    # expect list of test_idx arrays
    assert res == [[1], [0, 2]]


def test_get_idx_from_last_cv_iterable():
    # last cv is a plain iterable of (train_idx, test_idx)
    splits = [([0, 1], [2]), ([2], [0, 1])]
    est = DummyEstimator(n_samples=3, preds=[0, 1, 0], cv_steps=[None, splits])
    res = _get_idx_from_last_cv(est)
    assert res == [[2], [0, 1]]


def test_get_idx_from_last_cv_bad_type_raises():
    class WeirdCV:
        pass

    est = DummyEstimator(n_samples=2, preds=[0, 1], cv_steps=[None, WeirdCV()])
    with pytest.raises(ValueError):
        _get_idx_from_last_cv(est)


# ---------- Tests for make_SequentialCV_scorer (plain metric) ---------- #
def test_make_seqcv_scorer_plain_metric_no_cv():
    # simple metric that counts matches between y_true and y_pred
    def simple_metric(y_true, y_pred):
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])

    est = DummyEstimator(n_samples=len(y), preds=preds, cv_steps=[None])

    scorer = make_SequentialCV_scorer(simple_metric, response_method="predict")

    result = scorer(est, None, y)  # returns list containing a single value when no CV
    assert isinstance(result, list) and len(result) == 1
    # expected accuracy on full set
    assert pytest.approx(result[0]) == float(np.mean(y == preds))


def test_make_seqcv_scorer_plain_metric_with_cv_iterable():
    # metric that returns mean equality for a fold
    def simple_metric(y_true, y_pred):
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    # create 4 samples; two folds: fold1 -> test idx [0,1], fold2 -> [2,3]
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 1, 1, 0])

    splits = [([2, 3], [0, 1]), ([0, 1], [2, 3])]
    # we only care that the scorer reads the last cv, so cv_steps last entry should be splits
    est = DummyEstimator(n_samples=4, preds=preds, cv_steps=[None, splits])

    scorer = make_SequentialCV_scorer(simple_metric, response_method="predict")
    per_fold_scores = scorer(est, None, y)
    # should return a list of two scores (one per fold)
    assert isinstance(per_fold_scores, list) and len(per_fold_scores) == 2
    # compute expected manually: fold 1 test idx [0,1] -> preds [1,1] vs y [1,0] => mean([True, False]) = 0.5
    assert pytest.approx(per_fold_scores[0]) == 0.5
    # fold 2 test idx [2]()
