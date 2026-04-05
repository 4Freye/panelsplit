# tests/test_sequentialcvpipeline_indices.py
import numpy as np
import pytest

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from panelsplit.pipeline import SequentialCVPipeline
from panelsplit.cross_validation import PanelSplit
from panelsplit.metrics import neg_mean_squared_error_scorer as mse_scorer

# Small toy dataset used across tests
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
y_reg = np.array([1.0, 2.0, 3.0, 4.0])
y_clf = np.array([0, 0, 1, 1])  # simple binary labels

# PanelSplit used for CV variants
ps = PanelSplit([1, 1, 2, 3], n_splits=2, include_first_train_in_test=True)


@pytest.mark.parametrize(
    "cv_steps",
    [
        ([None, None]),
        ([ps, None]),
        ([None, ps]),
        ([ps, ps]),
    ],
)
@pytest.mark.parametrize("include_indices", [False, True])
def test_regressor_cv_variants(cv_steps, include_indices):
    """
    Ensure regressor pipeline fits, predicts, and scoring works for
    different cv_steps and include_indices combinations.
    """
    p = SequentialCVPipeline(
        steps=[
            ("impute", SimpleImputer()),
            ("regressor", RandomForestRegressor(random_state=0)),
        ],
        cv_steps=cv_steps,
        include_indices=include_indices,
    )

    # fit should not raise
    p.fit(X, y_reg)

    # predict should return a 1D array of length n_samples
    preds = p.predict(X)
    if include_indices:
        preds = preds[-1]

    assert isinstance(preds, np.ndarray), "predict must return numpy array"
    assert preds.shape[0] == X.shape[0], (
        "predict must return prediction for each sample"
    )

    # predictions should not contain tuples (no tuple leaked into final preds)
    # this checks that intermediate steps didn't produce (idx, data) that became final preds
    flat = np.atleast_1d(preds)
    assert not any(isinstance(v, tuple) for v in flat), (
        "predictions must not contain tuples"
    )

    # scorer must run and return a finite float (scorer may internally set include_indices)
    score = mse_scorer(p, X, y_reg)
    # assert isinstance(score, float), "scorer must return a float"
    # assert np.isfinite(score), "scorer result must be finite"
    assert not any(np.isnan(score))

    # additionally, compare with direct MSE of pipeline predictions (when cv_steps are None this must match)
    # If the scorer uses CV predictions, they may differ; so only compare when cv_steps have no split
    if all(s is None for s in cv_steps):
        mse_direct = -mean_squared_error(y_reg, preds)
        assert pytest.approx(mse_direct, rel=1e-6) == score[0]


@pytest.mark.parametrize(
    "cv_steps",
    [
        ([None, None]),
        ([ps, None]),
        ([None, ps]),
        ([ps, ps]),
    ],
)
@pytest.mark.parametrize("include_indices", [False, True])
def test_classifier_cv_variants(cv_steps, include_indices):
    """
    Ensure classifier pipeline fits, predicts, returns classes_, and scoring works.
    """
    clf = RandomForestClassifier(random_state=0)

    p = SequentialCVPipeline(
        steps=[("impute", SimpleImputer()), ("classifier", clf)],
        cv_steps=cv_steps,
        include_indices=include_indices,
    )

    # fit must not raise
    p.fit(X, y_clf)

    # classes_ should be present and be ndarray for classifier
    # (this property may raise for non-classifiers; for classifier it must succeed)
    classes = p.classes_
    assert isinstance(classes, np.ndarray), (
        "classes_ must be an ndarray for a fitted classifier"
    )
    assert set(classes).issubset(set(y_clf)), (
        "classes_ should reflect labels seen in training data"
    )

    # predict returns predictions with correct shape
    preds = p.predict(X)
    if include_indices:
        preds = preds[-1]

    assert preds.shape[0] == X.shape[0]
    assert not any(isinstance(v, tuple) for v in np.atleast_1d(preds)), (
        "predictions must not contain tuples"
    )

    # run a scorer that uses include_indices internally (we re-use the MSE scorer for shape & no-exception check)
    # We expect a finite float (even though MSE scorer isn't conceptually for classification, it should still run)
    score = mse_scorer(p, X, y_clf)
    # assert isinstance(score, float)
    # assert np.isfinite(score)
    assert not any(np.isnan(score))


def test_predict_and_scorer_consistency_for_no_cv():
    """
    Quick sanity check: when cv_steps = [None, None] the scorer's output must match
    a direct negative MSE computed from p.predict(X).
    """
    p = SequentialCVPipeline(
        steps=[
            ("impute", SimpleImputer()),
            ("regressor", RandomForestRegressor(random_state=0)),
        ],
        cv_steps=[None, None],
        include_indices=False,
    )
    p.fit(X, y_reg)
    preds = p.predict(X)
    score = mse_scorer(p, X, y_reg)
    assert pytest.approx(-mean_squared_error(y_reg, preds), rel=1e-9) == score[0]
