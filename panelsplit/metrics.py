from .utils.validation import _safe_indexing
from inspect import signature
from collections.abc import Iterable
from functools import partial
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.utils._param_validation import (
    validate_params,
)
from sklearn.metrics._scorer import _PassthroughScorer, _get_response_method_name
from copy import deepcopy
from sklearn.utils.validation import _check_response_method
import warnings
from sklearn.base import is_regressor
from panelsplit.utils._response import _get_response_values
from sklearn.utils.metadata_routing import (
    _MetadataRequester,
    _raise_for_params,
    _routing_enabled,
    MetadataRequest,
)
from .utils.typing import EstimatorLike, ArrayLike
from numpy.typing import NDArray
from typing import Callable, Optional, List, Union, Any, Dict
from typing_extensions import Self

# all the error scores:
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    class_likelihood_ratios,
    d2_absolute_error_score,
    explained_variance_score,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
    top_k_accuracy_score,
)
from sklearn.metrics.cluster import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
    v_measure_score,
)


def _get_idx_from_last_cv(estimator: EstimatorLike) -> Union[None, List[NDArray]]:
    """
    Return test indices for estimator.cv_steps[-1],
    or None if last_cv is None or cannot be interpreted.
    """
    last_cv = estimator.cv_steps[-1]
    rg = 1 if estimator.return_group == "test" else 0

    if last_cv is None:
        return None
    if hasattr(last_cv, "split"):
        return [split[rg] for split in last_cv.split()]
    if isinstance(last_cv, Iterable):
        return [split[rg] for split in last_cv]
    else:
        raise ValueError(
            "the final cv object is not None, an Iterable, and does not have a split attribute"
        )


def make_SequentialCV_scorer(
    score_func: Callable[..., float],
    *,
    response_method: Union[str, Iterable[str]] = "default",
    greater_is_better: bool = True,
    **kwargs: Any,
) -> Callable[..., float]:
    sign = 1 if greater_is_better else -1

    if response_method is None:
        warnings.warn(
            "response_method=None is deprecated in version 1.6 and will be removed "
            "in version 1.8. Leave it to its default value to avoid this warning.",
            FutureWarning,
        )
        response_method = "predict"
    elif response_method == "default":
        response_method = "predict"

    return _Scorer(score_func, sign, kwargs, response_method)


def _cached_call(
    cache: Optional[Dict[str, Any]],
    estimator: EstimatorLike,
    response_method: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call estimator with method and args and kwargs."""
    if cache is not None and response_method in cache:
        return cache[response_method]

    result, _ = _get_response_values(
        estimator,
        response_method,  # =response_method,
        *args,
        **kwargs,
    )

    if cache is not None:
        cache[response_method] = result

    return result


class _BaseScorer(_MetadataRequester):
    """Base scorer that is used as `scorer(estimator, X, y_true)`.

    Parameters
    ----------
    score_func : Callable
        The score function to use. It will be called as
        `score_func(y_true, y_pred, **kwargs)`.

    sign : int
        Either 1 or -1 to returns the score with `sign * score_func(estimator, X, y)`.
        Thus, `sign` defined if higher scores are better or worse.

    kwargs : Dict
        Additional parameters to pass to the score function.

    response_method : Union[str, Iterable[str]]
        The method to call on the estimator to get the response values.
    """

    def __init__(
        self,
        score_func: Callable,
        sign: int,
        kwargs: Dict,
        response_method: Union[str, Iterable[str]] = "predict",
    ):
        self._score_func = score_func
        self._sign = sign
        self._kwargs = kwargs
        self._response_method = response_method
        # TODO (1.8): remove in 1.8 (scoring="max_error" has been deprecated in 1.6)
        self._deprecation_msg = None

    def _get_pos_label(self) -> Optional[Any]:
        if "pos_label" in self._kwargs:
            return self._kwargs["pos_label"]
        score_func_params = signature(self._score_func).parameters
        if "pos_label" in score_func_params:
            return score_func_params["pos_label"].default
        return None

    def _accept_sample_weight(self) -> bool:
        # TODO(slep006): remove when metadata routing is the only way
        return "sample_weight" in signature(self._score_func).parameters

    def __repr__(self) -> str:
        sign_string = "" if self._sign > 0 else ", greater_is_better=False"
        response_method_string = f", response_method={self._response_method!r}"
        kwargs_string = "".join([f", {k}={v}" for k, v in self._kwargs.items()])

        return (
            f"make_SequentialCV_scorer({self._score_func.__name__}{sign_string}"
            f"{response_method_string}{kwargs_string})"
        )

    def __call__(
        self,
        estimator: EstimatorLike,
        X: ArrayLike,
        y_true: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> float:
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : EstimatorLike
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : ArrayLike
            Test data that will be fed to estimator.predict.

        y_true : ArrayLike
            Gold standard target values for X.

        sample_weight : Optional[ArrayLike]
            Sample weights. Default is None.

        **kwargs : Any
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

        Returns
        -------
        float
            Score function applied to prediction of estimator on X.
        """
        # TODO (1.8): remove in 1.8 (scoring="max_error" has been deprecated in 1.6)
        if self._deprecation_msg is not None:
            warnings.warn(
                self._deprecation_msg, category=DeprecationWarning, stacklevel=2
            )

        _raise_for_params(kwargs, self, None)

        _kwargs = deepcopy(kwargs)
        if sample_weight is not None:
            _kwargs["sample_weight"] = sample_weight

        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)

    def _warn_overlap(self, message: str, kwargs: Dict[str, Any]) -> None:
        """Warn if there is any overlap between ``self._kwargs`` and ``kwargs``.

        This method is intended to be used to check for overlap between
        ``self._kwargs`` and ``kwargs`` passed as metadata.
        """
        _kwargs = set() if self._kwargs is None else set(self._kwargs.keys())
        overlap = _kwargs.intersection(kwargs.keys())
        if overlap:
            warnings.warn(
                f"{message} Overlapping parameters are: {overlap}", UserWarning
            )

    def set_score_request(self, **kwargs: Dict[str, Any]) -> Self:
        """Set requested parameters by the scorer.

        Parameters
        ----------
        **kwargs : Dict[str, Any]
            Arguments should be of the form ``param_name=alias``, and `alias`
            can be one of ``{True, False, None, str}``.

        Returns
        -------
        Self
            The scorer instance (self) with updated parameters.

        Raises
        ------
        RuntimeError
            If the requested parameter is invalid or cannot be set.
        """
        if not _routing_enabled():
            raise RuntimeError(
                "This method is only available when metadata routing is enabled."
                " You can enable it using"
                " sklearn.set_config(enable_metadata_routing=True)."
            )

        self._warn_overlap(
            message=(
                "You are setting metadata request for parameters which are "
                "already set as kwargs for this metric. These set values will be "
                "overridden by passed metadata if provided. Please pass them either "
                "as metadata or kwargs to `make_scorer`."
            ),
            kwargs=kwargs,
        )
        self._metadata_request = MetadataRequest(owner=self.__class__.__name__)
        for param, alias in kwargs.items():
            self._metadata_request.score.add_request(param=param, alias=alias)
        return self


class _Scorer(_BaseScorer):
    def _score(
        self,
        method_caller: Callable,
        estimator: EstimatorLike,
        X: NDArray,
        y_true: NDArray,
        response_method: Union[str, Iterable[str]] = "predict",
        **kwargs: Dict[str, Any],
    ) -> List:
        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer instance and"
                " passed metadata. Please pass them either as kwargs to `make_scorer`"
                " or metadata, but not both."
            ),
            kwargs=kwargs,
        )

        pos_label = None if is_regressor(estimator) else self._get_pos_label()
        response_method = _check_response_method(estimator, self._response_method)
        estimator = deepcopy(estimator)
        estimator.set_params(**{"include_indices": True})
        idx, y_pred = method_caller(
            estimator,
            _get_response_method_name(response_method),
            X,
            pos_label=pos_label,
        )
        # make lookup dict for fast matching
        pred_dict = dict(zip(idx, y_pred))

        scoring_kwargs = {**self._kwargs, **kwargs}
        cv_test_idx = _get_idx_from_last_cv(estimator)
        if cv_test_idx is None:
            return [self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)]
        else:
            return [
                self._sign
                * self._score_func(
                    _safe_indexing(y_true, test_idx),
                    [pred_dict[idx] for idx in test_idx],
                    **scoring_kwargs,
                )
                for test_idx in cv_test_idx
            ]


@validate_params(
    {
        "scoring": [str, callable, None],
    },
    prefer_skip_nested_validation=True,
)
def get_scorer(scoring: Union[str, Callable]) -> Any:
    if isinstance(scoring, str):
        try:
            scorer = deepcopy(_SCORERS[scoring])
        except KeyError:
            raise ValueError(
                "%r is not a valid scoring value. "
                "Use sklearn.metrics.get_scorer_names() "
                "to get valid options." % scoring
            )
    else:
        scorer = scoring
    return scorer


def check_scoring(
    estimator: Optional[EstimatorLike] = None,
    scoring: Optional[Union[str, Callable, Iterable]] = None,
    *,
    allow_none: bool = False,
    raise_exc: bool = True,
) -> Any:
    if isinstance(scoring, str):
        return get_scorer(scoring)
    if callable(scoring):
        # Heuristic to ensure user has not passed a metric
        module = getattr(scoring, "__module__", None)
        if (
            isinstance(module, str)
            and module.startswith("sklearn.metrics.")
            and not module.startswith("sklearn.metrics._scorer")
            and not module.startswith("sklearn.metrics.tests.")
        ):
            raise ValueError(
                "scoring value %r looks like it is a metric "
                "function rather than a scorer. A scorer should "
                "require an estimator as its first parameter. "
                "Please use `make_SequentialCV_scorer` to convert a metric "
                "to a scorer." % scoring
            )
        return get_scorer(scoring)
    if isinstance(scoring, (list, tuple, set, dict)):
        scorers = _check_multimetric_scoring(estimator, scoring=scoring)
        return _MultimetricScorer(scorers=scorers, raise_exc=raise_exc)
    if scoring is None:
        if hasattr(estimator, "score"):
            return _PassthroughScorer(estimator)
        elif allow_none:
            return None
        else:
            raise TypeError(
                "If no scoring is specified, the estimator passed should "
                "have a 'score' method. The estimator %r does not." % estimator
            )


def _check_multimetric_scoring(estimator: EstimatorLike, scoring: Iterable) -> Any:
    err_msg_generic = (
        f"scoring is invalid (got {scoring!r}). Refer to the "
        "scoring glossary for details: "
        "https://scikit-learn.org/stable/glossary.html#term-scoring"
    )

    if isinstance(scoring, (list, tuple, set)):
        err_msg = (
            "The list/tuple elements must be unique strings of predefined scorers. "
        )
        try:
            keys = set(scoring)
        except TypeError as e:
            raise ValueError(err_msg) from e

        if len(keys) != len(scoring):
            raise ValueError(
                f"{err_msg} Duplicate elements were found in"
                f" the given list. {scoring!r}"
            )
        elif len(keys) > 0:
            if not all(isinstance(k, str) for k in keys):
                if any(callable(k) for k in keys):
                    raise ValueError(
                        f"{err_msg} One or more of the elements "
                        "were callables. Use a dict of score "
                        "name mapped to the scorer callable. "
                        f"Got {scoring!r}"
                    )
                else:
                    raise ValueError(
                        f"{err_msg} Non-string types were found "
                        f"in the given list. Got {scoring!r}"
                    )
            scorers = {
                scorer: check_scoring(estimator, scoring=scorer) for scorer in scoring
            }
        else:
            raise ValueError(f"{err_msg} Empty list was given. {scoring!r}")

    elif isinstance(scoring, dict):
        keys = set(scoring)
        if not all(isinstance(k, str) for k in keys):
            raise ValueError(
                "Non-string types were found in the keys of "
                f"the given dict. scoring={scoring!r}"
            )
        if len(keys) == 0:
            raise ValueError(f"An empty dict was passed. {scoring!r}")
        scorers = {
            key: check_scoring(estimator, scoring=scorer)
            for key, scorer in scoring.items()
        }
    else:
        raise ValueError(err_msg_generic)

    return scorers


# Standard regression scores
explained_variance_scorer = make_SequentialCV_scorer(explained_variance_score)
r2_scorer = make_SequentialCV_scorer(r2_score)
neg_max_error_scorer = make_SequentialCV_scorer(max_error, greater_is_better=False)
max_error_scorer = make_SequentialCV_scorer(max_error, greater_is_better=False)

neg_mean_squared_error_scorer = make_SequentialCV_scorer(
    mean_squared_error, greater_is_better=False
)
neg_mean_squared_log_error_scorer = make_SequentialCV_scorer(
    mean_squared_log_error, greater_is_better=False
)
neg_mean_absolute_error_scorer = make_SequentialCV_scorer(
    mean_absolute_error, greater_is_better=False
)
neg_mean_absolute_percentage_error_scorer = make_SequentialCV_scorer(
    mean_absolute_percentage_error, greater_is_better=False
)
neg_median_absolute_error_scorer = make_SequentialCV_scorer(
    median_absolute_error, greater_is_better=False
)
neg_root_mean_squared_error_scorer = make_SequentialCV_scorer(
    root_mean_squared_error, greater_is_better=False
)
neg_root_mean_squared_log_error_scorer = make_SequentialCV_scorer(
    root_mean_squared_log_error, greater_is_better=False
)
neg_mean_poisson_deviance_scorer = make_SequentialCV_scorer(
    mean_poisson_deviance, greater_is_better=False
)

neg_mean_gamma_deviance_scorer = make_SequentialCV_scorer(
    mean_gamma_deviance, greater_is_better=False
)
d2_absolute_error_scorer = make_SequentialCV_scorer(d2_absolute_error_score)

# Standard Classification Scores
accuracy_scorer = make_SequentialCV_scorer(accuracy_score)
balanced_accuracy_scorer = make_SequentialCV_scorer(balanced_accuracy_score)
matthews_corrcoef_scorer = make_SequentialCV_scorer(matthews_corrcoef)


def positive_likelihood_ratio(y_true: NDArray, y_pred: NDArray) -> float:
    return class_likelihood_ratios(y_true, y_pred, replace_undefined_by=1.0)[0]


def negative_likelihood_ratio(y_true: NDArray, y_pred: NDArray) -> float:
    return class_likelihood_ratios(y_true, y_pred, replace_undefined_by=1.0)[1]


positive_likelihood_ratio_scorer = make_SequentialCV_scorer(positive_likelihood_ratio)
neg_negative_likelihood_ratio_scorer = make_SequentialCV_scorer(
    negative_likelihood_ratio, greater_is_better=False
)

# Score functions that need decision values
top_k_accuracy_scorer = make_SequentialCV_scorer(
    top_k_accuracy_score,
    greater_is_better=True,
    response_method=("decision_function", "predict_proba"),
)
roc_auc_scorer = make_SequentialCV_scorer(
    roc_auc_score,
    greater_is_better=True,
    response_method=("decision_function", "predict_proba"),
)
average_precision_scorer = make_SequentialCV_scorer(
    average_precision_score,
    response_method=("decision_function", "predict_proba"),
)
roc_auc_ovo_scorer = make_SequentialCV_scorer(
    roc_auc_score, response_method="predict_proba", multi_class="ovo"
)
roc_auc_ovo_weighted_scorer = make_SequentialCV_scorer(
    roc_auc_score,
    response_method="predict_proba",
    multi_class="ovo",
    average="weighted",
)
roc_auc_ovr_scorer = make_SequentialCV_scorer(
    roc_auc_score, response_method="predict_proba", multi_class="ovr"
)
roc_auc_ovr_weighted_scorer = make_SequentialCV_scorer(
    roc_auc_score,
    response_method="predict_proba",
    multi_class="ovr",
    average="weighted",
)

# Score function for probabilistic classification
neg_log_loss_scorer = make_SequentialCV_scorer(
    log_loss, greater_is_better=False, response_method="predict_proba"
)
neg_brier_score_scorer = make_SequentialCV_scorer(
    brier_score_loss, greater_is_better=False, response_method="predict_proba"
)
brier_score_loss_scorer = make_SequentialCV_scorer(
    brier_score_loss, greater_is_better=False, response_method="predict_proba"
)


# Clustering scores
adjusted_rand_scorer = make_SequentialCV_scorer(adjusted_rand_score)
rand_scorer = make_SequentialCV_scorer(rand_score)
homogeneity_scorer = make_SequentialCV_scorer(homogeneity_score)
completeness_scorer = make_SequentialCV_scorer(completeness_score)
v_measure_scorer = make_SequentialCV_scorer(v_measure_score)
mutual_info_scorer = make_SequentialCV_scorer(mutual_info_score)
adjusted_mutual_info_scorer = make_SequentialCV_scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = make_SequentialCV_scorer(normalized_mutual_info_score)
fowlkes_mallows_scorer = make_SequentialCV_scorer(fowlkes_mallows_score)


_SCORERS = dict(
    explained_variance=explained_variance_scorer,
    r2=r2_scorer,
    neg_max_error=neg_max_error_scorer,
    matthews_corrcoef=matthews_corrcoef_scorer,
    neg_median_absolute_error=neg_median_absolute_error_scorer,
    neg_mean_absolute_error=neg_mean_absolute_error_scorer,
    neg_mean_absolute_percentage_error=neg_mean_absolute_percentage_error_scorer,
    neg_mean_squared_error=neg_mean_squared_error_scorer,
    neg_mean_squared_log_error=neg_mean_squared_log_error_scorer,
    neg_root_mean_squared_error=neg_root_mean_squared_error_scorer,
    neg_root_mean_squared_log_error=neg_root_mean_squared_log_error_scorer,
    neg_mean_poisson_deviance=neg_mean_poisson_deviance_scorer,
    neg_mean_gamma_deviance=neg_mean_gamma_deviance_scorer,
    d2_absolute_error_score=d2_absolute_error_scorer,
    accuracy=accuracy_scorer,
    top_k_accuracy=top_k_accuracy_scorer,
    roc_auc=roc_auc_scorer,
    roc_auc_ovr=roc_auc_ovr_scorer,
    roc_auc_ovo=roc_auc_ovo_scorer,
    roc_auc_ovr_weighted=roc_auc_ovr_weighted_scorer,
    roc_auc_ovo_weighted=roc_auc_ovo_weighted_scorer,
    balanced_accuracy=balanced_accuracy_scorer,
    average_precision=average_precision_scorer,
    neg_log_loss=neg_log_loss_scorer,
    neg_brier_score=neg_brier_score_scorer,
    positive_likelihood_ratio=positive_likelihood_ratio_scorer,
    neg_negative_likelihood_ratio=neg_negative_likelihood_ratio_scorer,
    # Cluster metrics that use supervised evaluation
    adjusted_rand_score=adjusted_rand_scorer,
    rand_score=rand_scorer,
    homogeneity_score=homogeneity_scorer,
    completeness_score=completeness_scorer,
    v_measure_score=v_measure_scorer,
    mutual_info_score=mutual_info_scorer,
    adjusted_mutual_info_score=adjusted_mutual_info_scorer,
    normalized_mutual_info_score=normalized_mutual_info_scorer,
    fowlkes_mallows_score=fowlkes_mallows_scorer,
)
