"""
Tests for make_SequentialCV_scorer
"""

from .df_generation import create_rf_friendly_dataset
import unittest
from panelsplit.cross_validation import PanelSplit
from panelsplit.pipeline import SequentialCVPipeline
from panelsplit.metrics import make_SequentialCV_scorer, get_scorer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import average_precision_score, r2_score, fbeta_score


class Testmake_SequentialCV_scorer(unittest.TestCase):
    def setUp(self):
        df = create_rf_friendly_dataset(random_state=42)
        self.X = df[[col for col in df.columns if "X" in col]]
        self.period = df["year"]
        self.y = df["y"]
        self.y_binary = df["y_binary"]
        self.params = {"n_estimators": 5, "random_state": 1}
        self.n_splits = 5
        self.ps = PanelSplit(self.period, n_splits=self.n_splits)

    def test_last_cv_None(self):
        pipe = SequentialCVPipeline(
            [("rf", RandomForestClassifier(**self.params))], cv_steps=[None]
        )
        pipe.fit(self.X, self.y_binary)
        ap_scorer = make_SequentialCV_scorer(average_precision_score)
        result = ap_scorer(pipe, self.X, self.y_binary)
        assert isinstance(result, list)
        assert len(result) == 1  # should only be one

    def test_classification(self):
        pipe = SequentialCVPipeline(
            [("rf", RandomForestClassifier(**self.params))], cv_steps=[self.ps]
        )
        average_precision_scorer = make_SequentialCV_scorer(average_precision_score)
        pipe.fit(self.X, self.y_binary)
        result = average_precision_scorer(pipe, self.X, self.y_binary)
        assert len(result) == self.n_splits  # should match n_splits
        assert all([res > 0.7 for res in result])  # should do reasonably well.

    def test_regression(self):
        pipe = SequentialCVPipeline(
            [("rf", RandomForestRegressor(**self.params))], cv_steps=[self.ps]
        )
        r2_scorer = make_SequentialCV_scorer(r2_score)
        pipe.fit(self.X, self.y)
        result = r2_scorer(pipe, self.X, self.y)
        assert all([res > 0.5 for res in result])  # should do reasonably well.

    def test_addtl_arg(self):
        pipe = SequentialCVPipeline(
            [("rf", RandomForestClassifier(**self.params))], cv_steps=[self.ps]
        )
        fb_scorer = make_SequentialCV_scorer(fbeta_score, beta=0.4)
        pipe.fit(self.X, self.y_binary)
        fb_scorer(pipe, self.X, self.y_binary)

    def test_scorer_uses_predict_proba(self):
        pipe = SequentialCVPipeline(
            [("rf", RandomForestClassifier(**self.params))], cv_steps=[None]
        )
        average_precision_scorer = get_scorer("average_precision")
        pipe.fit(self.X, self.y_binary)
        result = average_precision_scorer(pipe, self.X, self.y_binary)
        rf = RandomForestClassifier(**self.params)
        rf.fit(self.X, self.y_binary)
        print(self.y_binary)

        # this should be in this string.
        self.assertIn("predict_proba", average_precision_scorer._response_method)

        # average_precision_scorer should call predict_proba, not predict.
        # as a result these shouldn't be equal:
        self.assertNotEqual(
            result, [average_precision_score(self.y_binary, rf.predict(self.X))]
        )
        # and these should:
        rf_result = [
            average_precision_score(self.y_binary, rf.predict_proba(self.X)[:, 1])
        ]
        self.assertEqual(result, rf_result)
