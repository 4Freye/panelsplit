"""
The :mod:`sklearn.model_selection._validation` module includes classes and
functions to validate the model.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
import time
import warnings
from collections import Counter
from contextlib import suppress
from numbers import Real
from traceback import format_exc

import numpy as np

# import scipy.sparse as sp
from joblib import logger

from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import get_scorer_names
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.utils._array_api import (
    get_namespace,
)
from sklearn.utils._param_validation import (
    HasMethods,
    Integral,
    StrOptions,
    validate_params,
)
from sklearn.utils.metadata_routing import (
    _routing_enabled,
)
from sklearn.utils.validation import _check_method_params, _num_samples
from typing import Any, Dict, SupportsFloat, Literal

from ..utils.typing import EstimatorLike, ArrayLike
from typing import Union, Callable, Optional, List, Tuple


ScoreDict = Dict[str, np.ndarray]
ScoreArray = type[np.ndarray]
ScoreType = Union[ScoreDict, ScoreArray]


# TODO(SLEP6): To be removed when set_config(enable_metadata_routing=False) is not
# possible.
def _check_groups_routing_disabled(groups: Any) -> None:
    if groups is not None and _routing_enabled():
        raise ValueError(
            "`groups` can only be passed if metadata routing is not enabled via"
            " `sklearn.set_config(enable_metadata_routing=True)`. When routing is"
            " enabled, pass `groups` alongside other metadata via the `params` argument"
            " instead."
        )


@validate_params(
    {
        "estimator": [HasMethods("fit")],
        "X": ["array-like", "sparse matrix"],
        "y": ["array-like", None],
        "groups": ["array-like", None],
        "scoring": [
            StrOptions(set(get_scorer_names())),
            callable,
            list,
            tuple,
            dict,
            None,
        ],
        # "cv": ["cv_object"],
        "n_jobs": [Integral, None],
        "verbose": ["verbose"],
        "params": [dict, None],
        "pre_dispatch": [Integral, str],
        "return_train_score": ["boolean"],
        "return_estimator": ["boolean"],
        "return_indices": ["boolean"],
        "error_score": [StrOptions({"raise"}), Real],
    },
    prefer_skip_nested_validation=False,  # estimator is not validated yet
)
def _insert_error_scores(
    results: Dict, error_score: numbers.Number, n_splits: int
) -> None:
    """Insert error in `results` by replacing them inplace with `error_score`.

    This only applies to multimetric scores because `_fit_and_score` will
    handle the single metric case.
    """
    successful_score = None
    failed_indices = []
    for i, result in enumerate(results):
        if result["fit_error"] is not None:
            failed_indices.append(i)
        elif successful_score is None:
            successful_score = result["test_scores"]

    if isinstance(successful_score, dict):
        formatted_error = {
            name: np.repeat(np.asarray(error_score), n_splits)
            for name in successful_score
            if name not in ["fit_time", "score_time"]
        }
        formatted_error.update(
            {
                name: error_score
                for name in successful_score
                if name in ["fit_time", "score_time"]
            }
        )
        for i in failed_indices:
            results[i]["test_scores"] = formatted_error.copy()
            if "train_scores" in results[i]:
                results[i]["train_scores"] = formatted_error.copy()


def _normalize_score_results(scores: list, scaler_score_key: str = "score") -> Dict:
    """Creates a scoring dictionary based on the type of `scores`"""
    if isinstance(scores[0], dict):
        # multimetric scoring
        return _aggregate_score_dicts(scores)
    # scaler
    return {scaler_score_key: scores}


def _warn_or_raise_about_fit_failures(
    results: Dict, error_score: SupportsFloat
) -> None:
    fit_errors = [
        result["fit_error"] for result in results if result["fit_error"] is not None
    ]
    if fit_errors:
        num_failed_fits = len(fit_errors)
        num_fits = len(results)
        fit_errors_counter = Counter(fit_errors)
        delimiter = "-" * 80 + "\n"
        fit_errors_summary = "\n".join(
            f"{delimiter}{n} fits failed with the following error:\n{error}"
            for error, n in fit_errors_counter.items()
        )

        if num_failed_fits == num_fits:
            all_fits_failed_message = (
                f"\nAll the {num_fits} fits failed.\n"
                "It is very likely that your model is misconfigured.\n"
                "You can try to debug the error by setting error_score='raise'.\n\n"
                f"Below are more details about the failures:\n{fit_errors_summary}"
            )
            raise ValueError(all_fits_failed_message)

        else:
            some_fits_failed_message = (
                f"\n{num_failed_fits} fits failed out of a total of {num_fits}.\n"
                "The score on these train-test partitions for these parameters"
                f" will be set to {error_score}.\n"
                "If these failures are not expected, you can try to debug them "
                "by setting error_score='raise'.\n\n"
                f"Below are more details about the failures:\n{fit_errors_summary}"
            )
            warnings.warn(some_fits_failed_message, FitFailedWarning)


@validate_params(
    {
        "estimator": [HasMethods("fit")],
        "X": ["array-like", "sparse matrix"],
        "y": ["array-like", None],
        "groups": ["array-like", None],
        "scoring": [StrOptions(set(get_scorer_names())), callable, None],
        "cv": ["cv_object"],
        "n_jobs": [Integral, None],
        "verbose": ["verbose"],
        "params": [dict, None],
        "pre_dispatch": [Integral, str, None],
        "error_score": [StrOptions({"raise"}), Real],
    },
    prefer_skip_nested_validation=False,  # estimator is not validated yet
)
def _fit_and_score(
    estimator: EstimatorLike,
    X: ArrayLike,
    y: ArrayLike,
    *,
    scorer: Callable,
    verbose: int,
    parameters: Dict,
    fit_params: Dict,
    score_params: Dict,
    return_train_score: bool = False,
    return_parameters: bool = False,
    return_n_test_samples: bool = False,
    return_times: bool = False,
    return_estimator: bool = False,
    candidate_progress: Optional[Union[List, Tuple]] = None,
    error_score: Union[Literal["raise"], float] = np.nan,
    n_splits: int,
) -> Dict:
    """
    Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : EstimatorLike
        The object to use to fit the data.

    X : ArrayLike
        The data to fit. Shape (n_samples, n_features)

    y : ArrayLike
        The target variable to try to predict in supervised learning. Shape (n_samples, n_classes)

    scorer : Callable
        Callable with signature ``scorer(estimator, X, y)`` or a dict mapping
        scorer name to callable.

    verbose : int
        The verbosity level.

    parameters : Dict
        Parameters to be set on the estimator.

    fit_params : Dict
        Parameters passed to ``estimator.fit``.

    score_params : Dict
        Parameters passed to the scorer.

    return_train_score : bool
        Compute and return score on training set. default is False.

    return_parameters : bool
        Return parameters used for the estimator. Default is False.

    return_n_test_samples : bool
        Whether to return the ``n_test_samples``. Default is False.

    return_times : bool
        Whether to return fit/score times. Default is False

    return_estimator : bool
        Whether to return the fitted estimator. Default is False.

    candidate_progress : Optional[Union[List, Tuple]]
        Format: (<current_candidate_id>, <total_number_of_candidates>). Default is None.

    error_score : Union[Literal['raise'], float]
        Value to assign to the score if an error occurs in estimator fitting.
        If 'raise', the error is raised. Otherwise, FitFailedWarning is raised. Default is np.nan.

    n_splits : int
        Number of CV splits.

    Returns
    -------
    Dict
        Keys (optional based on flags):
        - 'train_scores' : dict of scorer name -> float
        - 'test_scores' : dict of scorer name -> float
        - 'n_test_samples' : int
        - 'fit_time' : float
        - 'score_time' : float
        - 'parameters' : dict or None
        - 'estimator' : estimator object
        - 'fit_error' : str or None

    Raises
    ------
    ValueError
        If ``error_score`` is neither 'raise' nor numeric, or if estimator parameters or data are invalid.
    Exception
        If an unexpected error occurs during fit or scoring.
    """

    xp, _ = get_namespace(X)
    # X_device = device(X)

    # Make sure that we can fancy index X even if train and test are provided
    # as NumPy arrays by NumPy only cross-validation splitters.
    # train, test = xp.asarray(train, device=X_device), xp.asarray(test, device=X_device)

    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        #     if split_progress is not None:
        #         progress_msg = f" {split_progress[0] + 1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0] + 1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_method_params(X, params=fit_params)
    score_params = score_params if score_params is not None else {}
    score_params_train = _check_method_params(X, params=score_params)
    score_params_test = _check_method_params(X, params=score_params)

    if parameters is not None:
        # here we clone the parameters, since sometimes the parameters
        # themselves might be estimators, e.g. when we search over different
        # estimators in a pipeline.
        # ref: https://github.com/scikit-learn/scikit-learn/pull/26786
        estimator = estimator.set_params(**clone(parameters, safe=False))

    start_time = time.time()

    # X_train, y_train = _safe_split(estimator, X, y, train)
    # X_test, y_test = _safe_split(estimator, X, y, test, train)

    result: Dict[str, Any] = {}
    try:
        if y is None:
            estimator.fit(X, **fit_params)
        else:
            estimator.fit(X, y, **fit_params)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            test_scores: ScoreType
            train_scores: ScoreType

            if isinstance(scorer, _MultimetricScorer):
                test_scores = {
                    name: np.repeat(error_score, n_splits) for name in scorer._scorers
                }
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = np.repeat(np.asarray(error_score), n_splits)
                if return_train_score:
                    train_scores = np.repeat(np.asarray(error_score), n_splits)
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores = _score(
            estimator=estimator,
            X=X,
            y=y,
            scorer=scorer,
            score_params=score_params_test,
            n_splits=n_splits,
            error_score=error_score,
        )
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            estimator.set_params(**{"return_group": "train"})
            train_scores = _score(
                estimator=estimator,
                X=X,
                y=y,
                scorer=scorer,
                score_params=score_params_train,
                n_splits=n_splits,
                error_score=error_score,
            )
            estimator.set_params(**{"return_group": "test"})

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f"mean {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]  # type: ignore
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={np.nanmean(test_scores[scorer_name]):.3f})"
            else:
                result_msg += ", mean score="
                if return_train_score:
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    result_msg += f"{np.nanmean(test_scores):.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


def _score(
    estimator: EstimatorLike,
    X: ArrayLike,
    y: ArrayLike,
    scorer: Callable,
    score_params: Dict,
    n_splits: int,
    error_score: Union[str, float] = "raise",
) -> Dict:
    """Compute the score(s) of an estimator on a given test set.

    Will return a dict of floats if `scorer` is a _MultiMetricScorer, otherwise a single
    float is returned.
    """
    score_params = {} if score_params is None else score_params

    try:
        if y is None:
            scores = scorer(estimator, X, **score_params)
        else:
            scores = scorer(estimator, X, y, **score_params)
    except Exception:
        if isinstance(scorer, _MultimetricScorer):
            # If `_MultimetricScorer` raises exception, the `error_score`
            # parameter is equal to "raise".
            raise
        else:
            if error_score == "raise":
                raise
            else:
                scores = np.repeat(error_score, n_splits)
                warnings.warn(
                    (
                        "Scoring failed. The score on this train-test partition for "
                        f"these parameters will be set to {error_score}. Details: \n"
                        f"{format_exc()}"
                    ),
                    UserWarning,
                )

    # Check non-raised error messages in `_MultimetricScorer`
    if isinstance(scorer, _MultimetricScorer):
        exception_messages = [
            (name, str_e) for name, str_e in scores.items() if isinstance(str_e, str)
        ]
        if exception_messages:
            # error_score != "raise"
            for name, str_e in exception_messages:
                scores[name] = error_score
                warnings.warn(
                    (
                        "Scoring failed. The score on this train-test partition for "
                        f"these parameters will be set to {error_score}. Details: \n"
                        f"{str_e}"
                    ),
                    UserWarning,
                )

    # error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"
    if isinstance(scores, dict):
        for name, score in scores.items():
            if hasattr(score, "item"):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score = score.item()
            # if not isinstance(score, numbers.Number):
            #     raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        if hasattr(scores, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        # if not isinstance(scores, numbers.Number):
        #     raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores


def _aggregate_score_dicts(scores: List[Dict]) -> Dict:
    """Aggregate the list of dict to dict of np ndarray"""
    return {
        key: (
            np.asarray([score[key] for score in scores])
            if isinstance(scores[0][key], numbers.Number)
            else [score[key] for score in scores]
        )
        for key in scores[0]
    }
