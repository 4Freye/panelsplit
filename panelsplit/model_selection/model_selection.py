from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone
from collections import defaultdict
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import _check_method_params, check_is_fitted, indexable
from sklearn.model_selection import ParameterSampler, ParameterGrid
from sklearn.model_selection._search import (
    _search_estimator_has,
    _yield_masked_array_for_each_param,
)
from sklearn.utils.metadata_routing import (
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.parallel import Parallel, delayed


from panelsplit.metrics import check_scoring, _check_multimetric_scoring
from sklearn.metrics._scorer import (
    _MultimetricScorer,
    get_scorer_names,
)
from scipy.stats import rankdata
import time
from sklearn.utils import Bunch
from ._validation import (
    _aggregate_score_dicts,
    _fit_and_score,
    _insert_error_scores,
    _normalize_score_results,
    _warn_or_raise_about_fit_failures,
)
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions


import warnings
from inspect import signature

import numbers
from typing import (
    Tuple,
    Union,
    Callable,
    List,
    Dict,
    Optional,
    Any,
    Type,
    Literal,
)

from ..pipeline import SequentialCVPipeline
import numpy as np
from numpy.typing import NDArray
from ..utils.typing import ArrayLike, EstimatorLike
from typing_extensions import Self


__all__ = ["GridSearch", "RandomizedSearch"]


class BaseSearch(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Abstract base class for hyperparameter search with cross-validation.

    Parameters
    ----------
    estimator : SequentialCVPipeline
        The estimator (pipeline) to be optimized. Must have include_indices=False.
    scoring : Optional[Union[str, Callable, list, tuple, dict]], default=None
        A string, callable, list, tuple, or dict to evaluate the predictions on the test set.
    n_jobs : Optional[int], default=None
        Number of jobs to run in parallel.
    refit : bool, default=True
        If True, refit an estimator on the whole dataset with the best found parameters.
    verbose : int, default=0
        Controls the verbosity of the output.
    pre_dispatch : Union[int, str], default="2*n_jobs"
        Controls the number of jobs that get dispatched during parallel execution.
    error_score : Union[Literal["raise"], float], default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    return_train_score : Union[str, bool], default=False
        If True, include train scores in the results.

    Examples
    --------
    >>> from panelsplit.model_selection import BaseSearch
    >>> from panelsplit.pipeline import SequentialCVPipeline
    >>> pipeline = SequentialCVPipeline(steps=[...], include_indices=False)
    >>> search = BaseSearch(estimator=pipeline, n_jobs=2, verbose=1)
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "scoring": [
            StrOptions(set(get_scorer_names())),
            callable,
            list,
            tuple,
            dict,
            None,
        ],
        "n_jobs": [numbers.Integral, None],
        "refit": ["boolean", str, callable],
        # "cv": ["cv_object"],
        "verbose": ["verbose"],
        "pre_dispatch": [numbers.Integral, str],
        "error_score": [StrOptions({"raise"}), numbers.Real],
        "return_train_score": ["boolean"],
    }

    @abstractmethod
    def __init__(
        self,
        estimator: SequentialCVPipeline,
        *,
        scoring: Optional[Union[str, Callable, List, Tuple, Dict]] = None,
        n_jobs: Optional[int] = None,
        refit: bool = True,
        verbose: int = 0,
        pre_dispatch: Union[int, str] = "2*n_jobs",
        error_score: Union[Literal["raise"], float] = np.nan,
        return_train_score: Union[str, bool] = False,
    ):
        assert not estimator.include_indices, (
            "include_indices must be set to False when using SequentialCVPipeline with a Search object"
        )
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.error_score = error_score
        self.pre_dispatch = pre_dispatch
        self.return_train_score = return_train_score

    def score(
        self, X: ArrayLike, y: Optional[ArrayLike] = None, **params: Dict
    ) -> float:
        """
        Return the score on the given data if the estimator has been refit.

        This uses the score defined by ``scoring`` if provided, otherwise
        uses the ``best_estimator_.score`` method.

        Parameters
        ----------
        X : ArrayLike
            Array-like of shape (n_samples, n_features). Input data, where `n_samples` is the number
            of samples and `n_features` is the number of features.
        y : Optional[ArrayLike], default=None
            Target relative to X for classification or regression. None for unsupervised learning.
        **params : Dict
            Parameters to be passed to the underlying scorer(s).

        Returns
        -------
        float
            The score defined by ``scoring`` if provided, otherwise the score computed
            by ``best_estimator_.score``.

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4]])
        >>> y = np.array([0, 1])
        >>> search.score(X, y)
        0.85
        """
        _check_refit(self, "score")
        check_is_fitted(self)

        _raise_for_params(params, self, "score")

        if _routing_enabled():
            score_params = process_routing(self, "score", **params).scorer["score"]
        else:
            score_params = dict()

        if self.scorer_ is None:
            raise ValueError(
                "No score function explicitly defined, "
                "and the estimator doesn't provide one %s" % self.best_estimator_
            )
        if isinstance(self.scorer_, dict):
            if self.multimetric_:
                scorer = self.scorer_[self.refit]
            else:
                scorer = self.scorer_
            return scorer(self.best_estimator_, X, y, **score_params)

        # callable
        score = self.scorer_(self.best_estimator_, X, y, **score_params)
        if self.multimetric_:
            score = score[self.refit]
        return score

    @available_if(_search_estimator_has("score_samples"))
    def score_samples(self, X: ArrayLike) -> NDArray:
        """
        Call score_samples on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports ``score_samples``.

        Parameters
        ----------
        X : ArrayLike
            Data to predict on. Must fulfill input requirements of the underlying estimator.

        Returns
        -------
        NDArray
            Shape (n_samples,). The output of ``best_estimator_.score_samples``.

        Examples
        --------
        >>> scores = search.score_samples(X)
        >>> scores.shape
        (n_samples,)
        """
        check_is_fitted(self)
        return self.best_estimator_.score_samples(X)

    @available_if(_search_estimator_has("predict"))
    def predict(self, X: ArrayLike) -> NDArray:
        """
        Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports ``predict``.

        Parameters
        ----------
        X : ArrayLike
            Must fulfill the input assumptions of the underlying estimator.

        Returns
        -------
        NDArray
            Shape (n_samples,). Predicted labels or values for `X`.

        Examples
        --------
        >>> y_pred = search.predict(X)
        >>> y_pred.shape
        (n_samples,)
        """
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    @available_if(_search_estimator_has("predict_proba"))
    def predict_proba(self, X: ArrayLike) -> NDArray:
        """
        Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports ``predict_proba``.

        Parameters
        ----------
        X : ArrayLike
            Must fulfill the input assumptions of the underlying estimator.

        Returns
        -------
        NDArray
            Shape (n_samples,) or (n_samples, n_classes). Predicted class probabilities.

        Examples
        --------
        >>> probs = search.predict_proba(X)
        >>> probs.shape
        (n_samples, n_classes)
        """
        check_is_fitted(self)
        return self.best_estimator_.predict_proba(X)

    @available_if(_search_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X: ArrayLike) -> NDArray:
        """
        Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports ``predict_log_proba``.

        Parameters
        ----------
        X : ArrayLike
            Must fulfill the input assumptions of the underlying estimator.

        Returns
        -------
        NDArray
            Shape (n_samples,) or (n_samples, n_classes). Predicted log-probabilities.

        Examples
        --------
        >>> log_probs = search.predict_log_proba(X)
        >>> log_probs.shape
        (n_samples, n_classes)
        """
        check_is_fitted(self)
        return self.best_estimator_.predict_log_proba(X)

    @available_if(_search_estimator_has("decision_function"))
    def decision_function(self, X: ArrayLike) -> NDArray:
        """
        Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports ``decision_function``.

        Parameters
        ----------
        X : ArrayLike
            Must fulfill the input assumptions of the underlying estimator.

        Returns
        -------
        NDArray
            Shape (n_samples,) or (n_samples, n_classes) or (n_samples, n_classes * (n_classes-1)/2).
            Result of the decision function.

        Examples
        --------
        >>> scores = search.decision_function(X)
        >>> scores.shape
        (n_samples, n_classes)
        """
        check_is_fitted(self)
        return self.best_estimator_.decision_function(X)

    @available_if(_search_estimator_has("transform"))
    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and ``refit=True``.

        Parameters
        ----------
        X : ArrayLike
            Must fulfill the input assumptions of the underlying estimator.

        Returns
        -------
        ArrayLike
            Shape (n_samples, n_features). Transformed data.

        Examples
        --------
        >>> Xt = search.transform(X)
        >>> Xt.shape
        (n_samples, n_features)
        """
        check_is_fitted(self)
        return self.best_estimator_.transform(X)

    @property
    def classes_(self) -> NDArray:
        """
        Class labels.

        Only available when ``refit=True`` and the estimator is a classifier.

        Returns
        -------
        NDArray
            Array of class labels.

        Examples
        --------
        >>> labels = search.classes_
        >>> labels
        array([0, 1])
        """
        _search_estimator_has("classes_")(self)
        return self.best_estimator_.classes_

    def _run_search(self, evaluate_candidates: Callable) -> None:
        """Repeatedly calls `evaluate_candidates` to conduct a search.

        This method, implemented in sub-classes, makes it possible to
        customize the scheduling of evaluations: GridSearch and
        RandomizedSearch schedule evaluations for their whole parameter
        search space at once but other more sequential approaches are also
        possible: for instance is possible to iteratively schedule evaluations
        for new regions of the parameter search space based on previously
        collected evaluation results. This makes it possible to implement
        Bayesian optimization or more generally sequential model-based
        optimization by deriving from the BaseSearch abstract base class.
        For example, Successive Halving is implemented by calling
        `evaluate_candidates` multiples times (once per iteration of the SH
        process), each time passing a different set of candidates with `X`
        and `y` of increasing sizes.

        Parameters
        ----------
        evaluate_candidates : callable
            This callback accepts:
                - a list of candidates, where each candidate is a dict of
                  parameter settings.
                - an optional `cv` parameter which can be used to e.g.
                  evaluate candidates on different dataset splits, or
                  evaluate candidates on subsampled data (as done in the
                  Successive Halving estimators). By default, the original
                  `cv` parameter is used, and it is available as a private
                  `_checked_cv_orig` attribute.
                - an optional `more_results` dict. Each key will be added to
                  the `cv_results_` attribute. Values should be lists of
                  length `n_candidates`

            It returns a dict of all results so far, formatted like
            ``cv_results_``.

            Important note (relevant whether the default cv is used or not):
            in randomized splitters, and unless the random_state parameter of
            cv was set to an int, calling cv.split() multiple times will
            yield different splits. Since cv.split() is called in
            evaluate_candidates, this means that candidates will be evaluated
            on different splits each time evaluate_candidates is called. This
            might be a methodological issue depending on the search strategy
            that you're implementing. To prevent randomized splitters from
            being used, you may use _split._yields_constant_splits()

        Examples
        --------

        ::

            def _run_search(self, evaluate_candidates):
                "Try C=0.1 only if C=1 is better than C=10"
                all_results = evaluate_candidates([{"C": 1}, {"C": 10}])
                score = all_results["mean_test_score"]
                if score[0] < score[1]:
                    evaluate_candidates([{"C": 0.1}])
        """
        raise NotImplementedError("_run_search not implemented.")

    def _check_refit_for_multimetric(self, scores: List[Callable]) -> None:
        """Check `refit` is compatible with `scores` is valid"""
        multimetric_refit_msg = (
            "For multi-metric scoring, the parameter refit must be set to a "
            "scorer key or a callable to refit an estimator with the best "
            "parameter setting on the whole data and make the best_* "
            "attributes available for that metric. If this is not needed, "
            f"refit should be set to False explicitly. {self.refit!r} was "
            "passed."
        )

        valid_refit_dict = isinstance(self.refit, str) and self.refit in scores

        if (
            self.refit is not False
            and not valid_refit_dict
            and not callable(self.refit)
        ):
            raise ValueError(multimetric_refit_msg)

    @staticmethod
    def _select_best_index(
        refit: Union[Callable, bool], refit_metric: str, results: Dict
    ) -> int:
        """Select index of the best combination of hyperparemeters."""
        if callable(refit):
            # If callable, refit is expected to return the index of the best
            # parameter set.
            best_index = refit(results)
            if not isinstance(best_index, int):
                raise TypeError("best_index_ returned is not an integer")
            if best_index < 0 or best_index >= len(results["params"]):
                raise IndexError("best_index_ index out of range")
        else:
            best_index = results[f"rank_test_{refit_metric}"].argmin()
        return best_index

    def _get_scorers(self) -> Tuple[Union[Callable, List[Callable]], Union[str, bool]]:
        """Get the scorer(s) to be used.

        This is used in ``fit`` and ``get_metadata_routing``.

        Returns
        -------
        scorers, refit_metric
        """
        refit_metric: Union[str, bool] = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit
            scorers = _MultimetricScorer(
                scorers=scorers, raise_exc=(self.error_score == "raise")
            )

        return scorers, refit_metric

    def _check_scorers_accept_sample_weight(self) -> bool:
        # TODO(slep006): remove when metadata routing is the only way
        scorers, _ = self._get_scorers()
        # In the multimetric case, warn the user for each scorer separately
        if isinstance(scorers, _MultimetricScorer):
            for name, scorer in scorers._scorers.items():
                if not scorer._accept_sample_weight():
                    warnings.warn(
                        f"The scoring {name}={scorer} does not support sample_weight, "
                        "which may lead to statistically incorrect results when "
                        f"fitting {self} with sample_weight. "
                    )
            return scorers._accept_sample_weight()
        # In most cases, scorers is a Scorer object
        # But it's a function when user passes scoring=function
        if hasattr(scorers, "_accept_sample_weight"):
            accept = scorers._accept_sample_weight()
        elif callable(scorers):
            accept = "sample_weight" in signature(scorers).parameters
        else:
            # mypy-safe fallback: if scorers is a list of functions
            accept = all(
                "sample_weight" in signature(s).parameters
                for s in scorers  # type: ignore
            )

        if not accept:
            warnings.warn(
                f"The scoring {scorers} does not support sample_weight, "
                f"which may lead to statistically incorrect results when fitting {self} with sample_weight."
            )
        return accept

    def _get_routed_params_for_fit(self, params: Dict) -> Bunch:
        """Get the parameters to be used for routing.

        This is a method instead of a snippet in ``fit`` since it's used twice,
        here in ``fit``, and in ``HalvingRandomSearch.fit``.
        """
        if _routing_enabled():
            routed_params = process_routing(self, "fit", **params)
        else:
            params = params.copy()
            groups = params.pop("groups", None)
            routed_params = Bunch(
                estimator=Bunch(fit=params),
                splitter=Bunch(split={"groups": groups}),
                scorer=Bunch(score={}),
            )
            # NOTE: sample_weight is forwarded to the scorer if sample_weight
            # is not None and scorers accept sample_weight. For _MultimetricScorer,
            # sample_weight is forwarded if any scorer accepts sample_weight
            if (
                params.get("sample_weight") is not None
                and self._check_scorers_accept_sample_weight()
            ):
                routed_params.scorer.score["sample_weight"] = params["sample_weight"]
        return routed_params

    @_fit_context(
        # *Search.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None, **params: Any) -> Self:
        """
        Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features. For precomputed kernel or
            distance matrix, the expected shape of X is (n_samples, n_samples).

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator, the scorer,
            and the CV splitter.

            If a fit parameter is an array-like whose length is equal to
            `num_samples` then it will be split by cross-validation along with
            `X` and `y`. For example, the :term:`sample_weight` parameter is
            split because `len(sample_weights) = len(X)`. However, this behavior
            does not apply to `groups` which is passed to the splitter configured
            via the `cv` parameter of the constructor. Thus, `groups` is used
            *to perform the split* and determines which samples are
            assigned to the each side of the a split.

        Returns
        -------
        object
            Instance of fitted estimator.

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4]])
        >>> y = np.array([0, 1])
        >>> search = BaseSearch(estimator=SomeEstimator())
        >>> search.fit(X, y)
        """

        scorers, refit_metric = self._get_scorers()

        X, y = indexable(X, y)
        params = _check_method_params(X, params=params)

        routed_params = self._get_routed_params_for_fit(params)

        def get_n_splits(estimator: EstimatorLike) -> int:
            last_cv = estimator.cv_steps[-1]

            # Case 1: None
            if last_cv is None:
                return 1

            # Case 2: Has n_splits attribute
            if hasattr(last_cv, "n_splits"):
                return last_cv.n_splits

            # Case 3: Try length
            try:
                return len(last_cv)
            except TypeError:
                raise ValueError(
                    "The final cv object must be None, have an n_splits attribute, or support len()."
                )

        n_splits = get_n_splits(self.estimator)

        base_estimator = clone(self.estimator)

        # pass verbose to joblib.Parallel so it can display its own progress
        parallel = Parallel(
            n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch, verbose=self.verbose
        )

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=routed_params.estimator.fit,
            score_params=routed_params.scorer.score,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
            n_splits=n_splits,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(
                candidate_params: List, more_results: Optional[Dict] = None
            ) -> Dict:
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting SequentialCVPipeline for each of {0} candidates".format(
                            n_candidates
                        )
                    )

                # build tasks: one job per (candidate, split)
                tasks = []
                for cand_idx, parameters in enumerate(candidate_params):
                    tasks.append(
                        delayed(_fit_and_score)(
                            clone(base_estimator),
                            X,
                            y,
                            parameters=parameters,
                            candidate_progress=(cand_idx, n_candidates),
                            **fit_and_score_kwargs,
                        )
                    )

                out = parallel(tasks)

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. Were there no candidates?"
                    )

                if len(out) != n_candidates:
                    raise ValueError(
                        "_fit_and_score returned inconsistent results. "
                        f"Expected {n_candidates} results, got {len(out)}"
                    )

                _warn_or_raise_about_fit_failures(
                    out,
                    float(np.nan) if self.error_score == "raise" else self.error_score,
                )

                # For callable self.scoring, the return type is only know after calling.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score, n_splits)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results

                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callable scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            refit_metric_for_select: str = (
                refit_metric if isinstance(refit_metric, str) else "score"
            )
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric_for_select, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # here we clone the estimator as well as the parameters, since
            # sometimes the parameters themselves might be estimators, e.g.
            # when we search over different estimators in a pipeline.
            # ref: https://github.com/scikit-learn/scikit-learn/pull/26786
            self.best_estimator_ = clone(base_estimator).set_params(
                **clone(self.best_params_, safe=False)
            )

            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **routed_params.estimator.fit)
            else:
                self.best_estimator_.fit(X, **routed_params.estimator.fit)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        if isinstance(scorers, _MultimetricScorer):
            self.scorer_ = scorers._scorers
        else:
            self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def _format_results(
        self,
        candidate_params: List[Dict],
        n_splits: Union[int, Callable],
        out: Union[Dict[str, Any], List[Dict[Any, Any]]],
        more_results: Optional[Dict] = None,
    ) -> Dict:
        """
        Robust formatting of results supporting two styles of `_fit_and_score`
        outputs:
        - flattened per-split style (len(out) == n_candidates * n_splits after
            aggregation), and
        - per-candidate style where each candidate's output contains arrays of
            length n_splits.

        Parameters
        ----------
        candidate_params : list of dict
        n_splits : int
        out : list or dict (after _aggregate_score_dicts)
        more_results : dict (optional)
        """
        n_candidates = len(candidate_params)

        # If caller accidentally passed a callable as n_splits (defensive),
        # attempt to resolve it by calling with estimator if signature matches.
        if callable(n_splits):
            raise TypeError("n_splits should be an int, got a Callable instead.")

        # aggregate if needed (keep original invocation semantics)
        if isinstance(out, list):
            # mypy now knows out is list[dict], ok to call aggregator
            out = _aggregate_score_dicts(out)  # type: ignore[arg-type]  # if you want to silence a stubborn checker
        elif isinstance(out, dict):
            # already aggregated, do nothing
            pass
        else:
            raise TypeError(
                "_format_results expected 'out' to be either a list of dicts or a dict, "
                f"got {type(out)!r}"
            )
        results = dict(more_results or {})
        for key, val in results.items():
            results[key] = np.asarray(val)

        def _store(
            key_name: str,
            array: NDArray,
            weights: Optional[NDArray] = None,
            splits: bool = False,
            rank: bool = False,
        ) -> None:
            """Helper to store metrics/times into results dict.

            array may come in as 1D/2D; this will reshape to (n_candidates, n_splits).
            """

            array = np.asarray(array, dtype=np.float64).reshape(
                n_candidates, n_splits if splits else 1
            )

            # Precompute means
            array_means = np.average(array, axis=1, weights=weights)

            if splits:
                # Store split columns
                for split_idx in range(n_splits):
                    results[f"split{split_idx}_{key_name}"] = array[:, split_idx]

                # Store means
                results[f"mean_{key_name}"] = array_means

                # Weighted std (numpy has no direct function)
                diffs = array - array_means[:, None]
                array_stds = np.sqrt(np.average(diffs**2, axis=1, weights=weights))
                results[f"std_{key_name}"] = array_stds

            else:
                # Only one column: just store the means
                results[key_name] = array_means

            if key_name.startswith(("train_", "test_")) and np.any(
                ~np.isfinite(array_means)
            ):
                warnings.warn(
                    (
                        f"One or more of the {key_name.split('_')[0]} scores "
                        f"are non-finite: {array_means}"
                    ),
                    category=UserWarning,
                )

            if rank:
                # ranking: higher is better. Exclude NaNs (treat as worst).
                if np.isnan(array_means).all():
                    rank_result = np.ones_like(array_means, dtype=np.int32)
                else:
                    min_array_means = np.nanmin(array_means) - 1
                    safe_means = np.nan_to_num(array_means, nan=min_array_means)
                    rank_result = rankdata(-safe_means, method="min").astype(
                        np.int32, copy=False
                    )
                results[f"rank_{key_name}"] = rank_result

        _store("fit_time", out["fit_time"], splits=False, rank=True)
        _store("score_time", out["score_time"], splits=False, rank=True)

        # store params
        for param, ma in _yield_masked_array_for_each_param(candidate_params):
            results[param] = ma
        results["params"] = candidate_params

        # Normalize `test_scores` into dict: scorer_name -> matrix (n_candidates, n_splits)
        test_scores_entry = out.get("test_scores", None)
        if test_scores_entry is None:
            # No scoring happened; return minimal results
            return results

        test_scores_dict = _normalize_score_results(out["test_scores"])
        for scorer_name in test_scores_dict:
            # Computed the (weighted) mean and std for test scores alone
            _store(
                "test_%s" % scorer_name,
                test_scores_dict[scorer_name],
                splits=True,
                rank=True,
                weights=None,
            )

        if self.return_train_score:
            train_scores_dict = _normalize_score_results(out["train_scores"])
            for scorer_name in train_scores_dict:
                # Computed the (weighted) mean and std for test scores alone
                _store(
                    "train_%s" % scorer_name,
                    train_scores_dict[scorer_name],
                    splits=True,
                    rank=True,
                    weights=None,
                )
        return results


def _check_refit(search_cv: Type[BaseSearch], attr: str) -> None:
    if not search_cv.refit:
        raise AttributeError(
            f"This {type(search_cv).__name__} instance was initialized with "
            f"`refit=False`. {attr} is available only after refitting on the best "
            "parameters. You can refit an estimator manually using the "
            "`best_params_` attribute"
        )


class GridSearch(BaseSearch):
    """
    Exhaustive search over specified parameter values for an estimator.

    GridSearch has been adapted from sklearn's GridSearchCV to work with SequentialCVPipeline.
    For additional examples, please check out `sklearn.model_selection.GridSearchCV`.

    Important members are fit, predict.

    GridSearch implements a "fit" and a "score" method.
    It also implements "score_samples", "predict", "predict_proba",
    "decision_function", "transform" and "inverse_transform" if they are
    implemented in the estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_string_names`);
        - a callable (see :ref:`scoring_callable`) that returns a single value;
        - `None`, the `estimator`'s default evaluation criterion is used.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables as values.

        See :ref:`multimetric_grid_search` for an example.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearch`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

        - None, in which case all the jobs are immediately created and spawned. Use
          this for lightweight and fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs
        - An int, giving the exact number of total jobs that are spawned
        - A str, giving an expression as a function of n_jobs, as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

        .. versionadded:: 0.19

        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |       0.80      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |       0.70      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
            'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
            'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
            'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        For an example of visualization and interpretation of GridSearch results,
        see :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_stats.py`.

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

        .. versionadded:: 0.20

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.

        .. versionadded:: 1.0

    See Also
    --------
    ParameterGrid : Generates all the combinations of a hyperparameter grid.
    train_test_split : Utility function to split the data into a development
        set usable for fitting a GridSearch instance and an evaluation set
        for its final evaluation.
    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.

    Notes
    -----
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    Examples
    --------
    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import GridSearch
    >>> iris = datasets.load_iris()
    >>> parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    >>> svc = svm.SVC()
    >>> clf = GridSearch(svc, parameters)
    >>> clf.fit(iris.data, iris.target)
    GridSearch(estimator=SVC(),
                 param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
    >>> sorted(clf.cv_results_.keys())
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split2_test_score', ...
     'std_fit_time', 'std_score_time', 'std_test_score']
    """

    _parameter_constraints: dict = {
        **BaseSearch._parameter_constraints,
        "param_grid": [dict, list],
    }

    def __init__(
        self,
        estimator: EstimatorLike,
        param_grid: Dict,
        *,
        scoring: Optional[Union[str, Callable, List, Tuple, Dict]] = None,
        n_jobs: Optional[int] = None,
        refit: bool = True,
        verbose: int = 0,
        pre_dispatch: Union[str, int] = "2*n_jobs",
        error_score: Union[Literal["raise"], float] = np.nan,
        return_train_score: bool = False,
    ) -> None:
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates: Callable) -> None:
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))


class RandomizedSearch(BaseSearch):
    """
    Randomized search on hyper parameters.

    RandomizedSearch has been adapted from sklearn's GridSearchCV to work with SequentialCVPipeline.
    For additional examples, please check out `sklearn.model_selection.RandomizedSearchCV`.

    RandomizedSearch implements a "fit" and a "score" method.
    It also implements "score_samples", "predict", "predict_proba",
    "decision_function", "transform" and "inverse_transform" if they are
    implemented in the estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearch, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Parameters
    ----------
    estimator : estimator object
        An object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict or list of dicts
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_string_names`);
        - a callable (see :ref:`scoring_callable`) that returns a single value;
        - `None`, the `estimator`'s default evaluation criterion is used.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables as values.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's score method is used.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given the ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearch`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

        - None, in which case all the jobs are immediately created and spawned. Use
          this for lightweight and fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs
        - An int, giving the exact number of total jobs that are spawned
        - A str, giving an expression as a function of n_jobs, as in '2*n_jobs'

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

        .. versionadded:: 0.19

        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.84        |...|       3       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.84, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.67, 0.70],
            'std_test_score'     : [0.01, 0.24, 0.00],
            'rank_test_score'    : [1, 3, 2],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        For an example of analysing ``cv_results_``,
        see :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_stats.py`.

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

        .. versionadded:: 0.20

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.

        .. versionadded:: 1.0

    See Also
    --------
    GridSearch : Does exhaustive search over a grid of parameters.
    ParameterSampler : A generator over parameter settings, constructed from
        param_distributions.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import RandomizedSearch
    >>> from scipy.stats import uniform
    >>> iris = load_iris()
    >>> logistic = LogisticRegression(
    ...     solver="saga", tol=1e-2, max_iter=200, random_state=0
    ... )
    >>> distributions = dict(C=uniform(loc=0, scale=4), penalty=["l2", "l1"])
    >>> clf = RandomizedSearch(logistic, distributions, random_state=0)
    >>> search = clf.fit(iris.data, iris.target)
    >>> search.best_params_
    {'C': np.float64(2.195...), 'penalty': 'l1'}
    """

    _parameter_constraints: dict = {
        **BaseSearch._parameter_constraints,
        "param_distributions": [dict, list],
        "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator: EstimatorLike,
        param_distributions: Dict,
        *,
        n_iter: int = 10,
        scoring: Optional[Union[str, Callable, List, Tuple, Dict]] = None,
        n_jobs: Optional[int] = None,
        refit: bool = True,
        verbose: int = 0,
        pre_dispatch: Union[int, str] = "2*n_jobs",
        random_state: Optional[int] = None,
        error_score: Union[Literal["raise"], float] = np.nan,
        return_train_score: bool = False,
    ):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _run_search(self, evaluate_candidates: Callable) -> None:
        """Search n_iter candidates from param_distributions"""
        evaluate_candidates(
            ParameterSampler(
                self.param_distributions, self.n_iter, random_state=self.random_state
            )
        )
