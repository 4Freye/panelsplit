import unittest
import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from panelsplit.metrics import get_scorer
from sklearn.utils._param_validation import InvalidParameterError

from panelsplit.model_selection import GridSearch, RandomizedSearch
from panelsplit.cross_validation import PanelSplit
from panelsplit.pipeline import SequentialCVPipeline
from panelsplit.metrics import make_SequentialCV_scorer

from .df_generation import create_rf_friendly_dataset


class Test_searches(unittest.TestCase):
    def setUp(self):
        df = create_rf_friendly_dataset(random_state=42)
        self.X = df[[col for col in df.columns if "X" in col]]
        self.period = df["year"]
        self.y = df["y"]
        self.y_binary = df["y_binary"]
        self.rf_params = {"n_estimators": 5, "random_state": 1}
        self.n_splits = 5
        self.ps = PanelSplit(self.period, n_splits=self.n_splits)
        self.pipe = SequentialCVPipeline(
            [("rf", RandomForestClassifier(**self.rf_params))], cv_steps=[self.ps]
        )
        self.gs_params = {"rf__n_estimators": [5, 10]}

    def test_no_scoring_metric(self):
        gs = GridSearch(self.pipe, param_grid=self.gs_params, refit=False)
        gs.fit(self.X, self.y_binary)
        assert hasattr(gs, "cv_results_")

    def test_str_scoring_metric(self):
        gs = GridSearch(
            self.pipe, param_grid=self.gs_params, refit=False, scoring="roc_auc"
        )
        gs.fit(self.X, self.y_binary)
        assert hasattr(gs, "cv_results_")

    def test_scorer_scoring_metric(self):
        gs = GridSearch(
            self.pipe,
            param_grid=self.gs_params,
            refit=False,
            scoring=make_SequentialCV_scorer(average_precision_score),
        )
        gs.fit(self.X, self.y_binary)
        assert hasattr(gs, "cv_results_")

    def test_multi_scoring(self):
        gs = GridSearch(
            self.pipe,
            param_grid=self.gs_params,
            refit=False,
            scoring=["roc_auc", "average_precision"],
        )
        gs.fit(self.X, self.y_binary)
        assert hasattr(gs, "cv_results_")


class TestSearchesExtensive(unittest.TestCase):
    def setUp(self):
        # Reuse same dataset helper as in user's example
        df = create_rf_friendly_dataset(random_state=42)
        self.X = df[[col for col in df.columns if "X" in col]]
        self.period = df["year"]
        self.y = df["y"]
        self.y_binary = df["y_binary"]
        self.rf_params = {"n_estimators": 5, "random_state": 1}
        self.n_splits = 5
        self.ps = PanelSplit(self.period, n_splits=self.n_splits)

        # classifier pipeline and regressor pipeline for tests
        self.pipe_clf = SequentialCVPipeline(
            [("rf", RandomForestClassifier(**self.rf_params))],
            cv_steps=[self.ps],
            include_indices=False,
        )
        self.pipe_reg = SequentialCVPipeline(
            [("rf", RandomForestRegressor(**self.rf_params))],
            cv_steps=[self.ps],
        )

        self.gs_params = {"rf__n_estimators": [5, 10, 20]}

    def _mean_test_scores(self, results, metric="score"):
        key = f"mean_test_{metric}"
        return np.asarray(results[key])

    def test_parallel_search(self):
        gs = GridSearch(self.pipe_clf, param_grid=self.gs_params, n_jobs=-1)
        gs.fit(self.X, self.y_binary)

    def test_refit_true_creates_best_estimator_and_predict_methods(self):
        gs = GridSearch(
            self.pipe_clf, param_grid=self.gs_params, refit=True, scoring="roc_auc"
        )
        gs.fit(self.X, self.y_binary)

        # best_estimator_ must be present when refit=True
        self.assertTrue(hasattr(gs, "best_estimator_"))
        # predict and predict_proba should be available (RF classifier)
        gs.predict(self.X)

        probs = gs.predict_proba(self.X)
        # probs should have shape (n_samples, n_classes)
        self.assertEqual(probs.shape[1], self.y_binary.nunique())

        # score should be callable and produce an ROC AUC between 0 and 1
        sc = gs.score(self.X, self.y_binary)
        self.assertIsInstance(sc, list)
        [self.assertGreaterEqual(val, 0.0) for val in sc]
        [self.assertLessEqual(val, 1.0) for val in sc]

    def test_refit_false_does_not_create_best_estimator_but_has_cv_results(self):
        gs = GridSearch(
            self.pipe_clf, param_grid=self.gs_params, refit=False, scoring="roc_auc"
        )
        gs.fit(self.X, self.y_binary)

        # best_estimator_ should NOT be present when refit=False
        self.assertFalse(hasattr(gs, "best_estimator_"))
        # cv_results_ must exist
        self.assertTrue(hasattr(gs, "cv_results_"))
        # check number of param candidates match grid size
        self.assertEqual(
            len(gs.cv_results_["params"]), len(self.gs_params["rf__n_estimators"])
        )

    def test_multimetric_scoring_with_dict_and_refit_by_name(self):
        scoring = {
            "roc_auc": "roc_auc",
            "ap": make_SequentialCV_scorer(average_precision_score),
        }
        # refit by name 'ap' (average precision)
        gs = GridSearch(
            self.pipe_clf, param_grid=self.gs_params, scoring=scoring, refit="ap"
        )
        gs.fit(self.X, self.y_binary)

        # ensure multi-metric results exist
        self.assertTrue("mean_test_roc_auc" in gs.cv_results_)
        self.assertTrue("mean_test_ap" in gs.cv_results_)

        # best_score_ should correspond to mean_test_ap at best_index_
        chosen_mean_ap = gs.cv_results_["mean_test_ap"][gs.best_index_]
        # best_score_ was set from mean_test_ap for refit='ap'
        self.assertAlmostEqual(chosen_mean_ap, gs.best_score_, places=10)

    def test_refit_callable_chooses_index_returned_by_callable(self):
        # Create a refit callable that always chooses index 0
        def choose_zero(results):
            return 0

        gs = GridSearch(
            self.pipe_clf,
            param_grid=self.gs_params,
            scoring="roc_auc",
            refit=choose_zero,
        )
        gs.fit(self.X, self.y_binary)
        # best_index_ must equal 0 (as forced by callable)
        self.assertEqual(gs.best_index_, 0)

    def test_randomized_search_respects_n_iter(self):
        # RandomizedSearch should sample exactly n_iter candidates
        param_distributions = {"rf__n_estimators": [5, 10, 20]}
        rs = RandomizedSearch(
            self.pipe_clf,
            param_distributions=param_distributions,
            n_iter=2,
            refit=False,
        )
        rs.fit(self.X, self.y_binary)

        # number of tested parameter sets should equal n_iter
        self.assertEqual(len(rs.cv_results_["params"]), 2)

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.FitFailedWarning")
    @pytest.mark.filterwarnings("ignore:One or more of the test scores")
    def test_error_score_raise_vs_nan(self):
        # create a grid with an invalid candidate that will raise during fit
        bad_grid = {"rf__n_estimators": [5, "bad_value"]}

        # With error_score='raise', fitting should raise an exception
        gs_raise = GridSearch(
            self.pipe_clf,
            param_grid=bad_grid,
            scoring=None,
            error_score="raise",
            refit=False,
        )
        with self.assertRaises(InvalidParameterError):
            gs_raise.fit(self.X, self.y_binary)

        # With error_score=np.nan, fitting should complete and the bad candidate will produce NaN
        gs_nan = GridSearch(
            self.pipe_clf,
            param_grid=bad_grid,
            scoring=None,
            error_score=np.nan,
            refit=False,
        )
        gs_nan.fit(self.X, self.y_binary)
        # cv_results_ must exist
        self.assertTrue(hasattr(gs_nan, "cv_results_"))
        # mean_test_score should contain at least one NaN (the failing candidate)
        mean_scores = np.asarray(gs_nan.cv_results_["mean_test_score"])
        self.assertTrue(np.any(np.isnan(mean_scores)))

    def test_callable_scoring_returning_multimetric_dict(self):
        # A callable scoring function that returns a dict of metrics to simulate
        # a callable that produces multi-metric outputs (BaseSearch supports this)
        def callable_multi_scorer(estimator, X, y):
            rc_scorer = make_SequentialCV_scorer(roc_auc_score)

            def acc_score(y, y_pred):
                return np.mean(y == y_pred)

            acc_scorer = make_SequentialCV_scorer(acc_score)

            return {
                "m_roc_auc": rc_scorer(estimator, X, y),
                "m_acc": acc_scorer(estimator, X, y),
            }

        gs = GridSearch(
            self.pipe_clf,
            param_grid=self.gs_params,
            scoring=callable_multi_scorer,
            refit=False,
        )
        gs.fit(self.X, self.y_binary)

        # after fit, cv_results_ should contain mean_test_m_roc_auc and mean_test_m_acc
        self.assertIn("mean_test_m_roc_auc", gs.cv_results_)
        self.assertIn("mean_test_m_acc", gs.cv_results_)
        # Ensure multimetric_ flag has been set
        self.assertTrue(gs.multimetric_)

    def test_score_method_uses_refit_metric(self):
        # If the search is refit=True with scoring string, score should call that scorer
        gs = GridSearch(
            self.pipe_clf, param_grid=self.gs_params, refit=True, scoring="roc_auc"
        )
        gs.fit(self.X, self.y_binary)
        # score should return a float equal-ish to roc_auc_score of predict_proba on the best_estimator_
        score_from_method = gs.score(self.X, self.y_binary)

        reference = get_scorer("roc_auc")(gs.best_estimator_, self.X, self.y_binary)
        self.assertAlmostEqual(score_from_method, reference, places=7)


if __name__ == "__main__":
    unittest.main()
