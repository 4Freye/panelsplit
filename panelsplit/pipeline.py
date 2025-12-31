"""
Pipeline operations in panelsplit.

Includes SequentialCVPipeline.
"""

import copy
import inspect
import time
from typing import Union, Optional, Any, Tuple, List, Dict, Iterable
from narwhals.typing import IntoDataFrame, IntoSeries
from numpy.typing import NDArray
from .utils.typing import ArrayLike, EstimatorLike
from collections.abc import Callable
from .cross_validation import PanelSplit
import functools
import narwhals as nw
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.base import is_classifier
from .utils.validation import _check_X_y, _safe_indexing, check_cv
from sklearn.utils._tags import get_tags
from sklearn.utils import Tags, Bunch
from copy import deepcopy
from typing import Literal


def _log_message(
    message: str,
    verbose: bool,
    step_idx: int,
    total: int,
    elapsed_time: Optional[Any] = None,
) -> None:
    """
    Log messages similar to scikit-learn's Pipeline verbose output.
    """
    if verbose:
        if elapsed_time is not None:
            message += " (elapsed time: %5.1fs)" % elapsed_time
        print("[SequentialCVPipeline] ({}/{}) {}".format(step_idx, total, message))


def _sort_and_combine(
    predictions_with_idx: List[Tuple[int, Any]],
    include_indices: bool = False,
    return_group: Literal["train", "test"] = "test",
) -> Any:
    """
    Sort and combine predictions from (index, prediction) pairs.

    If all predictions are numpy arrays -> returns an ndarray (vstack).
    If all predictions are narwhals-series-like -> returns a narwhals object (concat).
    Otherwise -> returns a numpy array constructed from the list.
    """
    predictions_with_idx.sort(key=lambda pair: pair[0])
    indices, _predictions = zip(*predictions_with_idx)
    predictions: Any = list(_predictions)

    # All-ndarray branch
    if predictions and all(isinstance(p, np.ndarray) for p in predictions):
        predictions = np.vstack(predictions)

    # All-narwhals (or series-like) branch: ensure every item looks like a narwhals series/frame
    elif predictions and all(
        hasattr(p, "pipe") or hasattr(p, "_compliant_series") for p in predictions
    ):
        # Cast so nw.concat's generics line up with what we pass
        predictions = nw.concat(predictions)
    else:
        # Fallback: coerce to numpy array
        predictions = np.array(predictions)
    if include_indices:
        return np.array(indices), predictions
    else:
        return predictions


# Cache for method signature inspection to avoid repeated reflection
_METHOD_SIGNATURE_CACHE = {}


def _call_method_with_correct_args(
    model: EstimatorLike,
    method_name: str,
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
) -> Any:
    """
    Call a method on a model with the correct arguments.

    Dynamically inspects the method signature (with caching) to determine
    if y parameter is required. This solves issue #59 where methods like
    score() require y but other methods like predict() don't.

    Parameters
    ----------
    model : EstimatorLike
        The fitted model
    method_name : str
        The name of the method to call (e.g., 'predict', 'score', 'transform')
    X : ArrayLike
        Input features
    y : Optional[ArrayLike]
        Target values

    Returns
    -------
    output : Any
        Result of calling the method

    Raises
    ------
    ValueError
        If the method requires y but y is None

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression().fit(X_train, y_train)
    >>> # For predict (doesn't require y)
    >>> predictions = _call_method_with_correct_args(model, "predict", X_test)
    >>> # For score (requires y)
    >>> score = _call_method_with_correct_args(model, "score", X_test, y_test)
    """

    method = getattr(model, method_name)

    # Cache the signature inspection to avoid performance overhead
    cache_key = (type(model).__name__, method_name)
    if cache_key not in _METHOD_SIGNATURE_CACHE:
        try:
            signature = inspect.signature(method)
            params = signature.parameters

            # Check if 'y' is a parameter and whether it's required
            has_y = "y" in params
            if has_y:
                y_param = params["y"]
                y_required = y_param.default is inspect.Parameter.empty
            else:
                y_required = False

            _METHOD_SIGNATURE_CACHE[cache_key] = (has_y, y_required)
        except Exception:
            # Fallback: assume y is optional
            _METHOD_SIGNATURE_CACHE[cache_key] = (False, False)

    has_y, y_required = _METHOD_SIGNATURE_CACHE[cache_key]

    # Call the method with the appropriate arguments
    if has_y:
        if y_required and y is None:
            raise ValueError(
                f"Method '{method_name}' requires y parameter but y is None"
            )
        return method(X, y) if y is not None else method(X)

    return method(X)


def _make_method(method_name: str, fit: bool = True) -> Callable:
    """
    Create a pipeline method dynamically for fitting or predicting.

    Parameters
    ----------
    method_name : str
        The name of the method to create (e.g., 'predict', 'transform').
    fit : bool
        Whether the method should perform fitting of the pipeline. Default is True.

    Returns
    -------
    function: Callable
        A method that applies the specified method to all pipeline steps.
    """

    def _method(
        self: Any, X: ArrayLike, y: Optional[ArrayLike] = None, **kwargs: Any
    ) -> Any:
        _check_X_y(X, y)
        # Fit all steps except the final one.
        current_output = X
        total_steps = len(self.steps)

        # Optimize: Move fitted state checks outside the loop
        if fit:
            # Initialize fitted_steps_ if not already present (before first step of fitting)
            if not hasattr(self, "fitted_steps_"):
                self.fitted_steps_ = {}
        else:
            # Check if pipeline is fitted before using fitted_steps_ (once before loop)
            try:
                check_is_fitted(self)
            except NotFittedError:
                raise NotFittedError(
                    "This SequentialCVPipeline instance is not fitted yet. "
                    "Call 'fit' with appropriate arguments before using this method."
                )

        for step_idx, (name, transformer) in enumerate(self.steps, start=1):
            t_start = time.time()
            not_final_step = total_steps != step_idx
            # Use 'transform' for intermediate steps and the provided method for the final step.
            method = "transform" if not_final_step else method_name
            if fit:
                # Initialize fitted_steps_ if not already done (for dynamic methods)
                if not hasattr(self, "fitted_steps_"):
                    self.fitted_steps_ = {}

                if transformer is None or transformer == "passthrough":
                    self.fitted_steps_[name] = None
                    continue

                current_output, fitted_model = self._fit_method_step(
                    transformer,
                    current_output,
                    y,
                    self.cv_steps[step_idx - 1],
                    return_output=True,
                    method=method,
                )

                self.fitted_steps_[name] = tuple(fitted_model)
            else:
                # Check if pipeline has been fitted
                if not hasattr(self, "fitted_steps_"):
                    raise NotFittedError(
                        f"This {type(self).__name__} instance is not fitted yet. "
                        "Call 'fit' with appropriate arguments before using this estimator."
                    )

                fitted_model = self.fitted_steps_.get(name)
                if fitted_model is None:
                    continue
                # Pass y along for the final step (needed for methods like score)
                current_output = self._method_step(
                    fitted_model,
                    method,
                    current_output,
                    y if not not_final_step else None,
                )

            _log_message(
                "Step '{}' completed".format(name),
                self.verbose,
                step_idx,
                total_steps,
                time.time() - t_start,
            )

        return current_output

    # Make the wrapper look like the real method to sklearn introspection:
    # set __name__ and __qualname__ and copy basic attributes.
    try:
        _method.__name__ = method_name
        _method.__qualname__ = method_name
        functools.update_wrapper(_method, _method)  # noop to ensure attributes exist
    except Exception:
        # Non-fatal: continue even if we can't set metadata
        pass

    return _method


class SequentialCVPipeline(_BaseComposition, BaseEstimator):
    """
    A pipeline that applies a series of transformers/estimators with optional cross-validation (CV).

    Each step is a tuple of (name, transformer).

    Parameters
    ----------
    steps : List[Tuple[str, BaseEstimator]]
        List where each tuple is (name, transformer, cv). 'name' is a string identifier and
        'transformer' is an estimator or transformer.
    cv_steps : List[Union[PanelSplit, Iterable, None]]
        A list of PanelSplit or None. Must be the same length as steps.
    verbose : bool, default = False
        Whether to print verbose output during fitting and transformation.
    include_indices : bool, default = False
        If True, include the indices in the output.
    return_group : {"test", "train"}, default = "test"
        Which group to return e.g. when calling predict().

    Attributes
    ----------
    steps : List[Tuple[str, BaseEstimator]]
        The list of pipeline steps.
    cv_steps : List[Union[PanelSplit, Iterable, None]]
        A list of PanelSplit or None.
    verbose : bool
        Verbosity flag.
    fitted_steps_ : dict
        Dictionary storing fitted transformers for each step.

    Raises
    ------
    ValueError
        If any of the tuples provided in ``steps`` are not of length 3.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import KFold
    >>> pipeline = SequentialCVPipeline(
    ...     steps=[("scaler", StandardScaler()), ("classifier", LogisticRegression())],
    ...     cv_steps=[None, KFold(n_splits=3)],
    ...     verbose=False,
    ... )
    >>> print(pipeline.steps)
    [('scaler', StandardScaler()), ('classifier', LogisticRegression())]
    """

    def __init__(
        self,
        steps: List[Tuple[str, BaseEstimator]],
        cv_steps: List[Union[PanelSplit, Iterable, None]],
        verbose: bool = False,
        include_indices: bool = False,
        return_group: Literal["test", "train"] = "test",
    ):
        # Each step must be a tuple: (name, transformer)
        if not len(steps) == len(cv_steps):
            raise ValueError("The length of steps and cv_steps must match")

        for step in steps:
            if not (isinstance(step, tuple) and len(step) == 2):
                raise ValueError("Each step must be a tuple of (name, transformer)")

        assert return_group in ["test", "train"], (
            "return_group must be either test or train."
        )

        self.return_group = return_group

        # just the cv of each step
        self.cv_steps = cv_steps

        # create sklearn-compatible steps attribute (list of (name, estimator))
        self.steps = steps

        self.verbose = verbose
        self.include_indices = include_indices

        final_tr = self.steps[-1][1]

        # Add _estimator_type attribute
        if hasattr(final_tr, "_estimator_type"):
            # copy string like "classifier" or "regressor"
            self._estimator_type = getattr(final_tr, "_estimator_type")
        else:
            # No final estimator_type available — remove attribute if present
            if hasattr(self, "_estimator_type"):
                delattr(self, "_estimator_type")

        # Note: fitted_steps_ is NOT initialized here to comply with sklearn conventions
        # It will be created during fit() to properly indicate fitted state
        self._inject_dynamic_methods()

    def _inject_dynamic_methods(self) -> None:
        """Inject dynamic pipeline methods based on the final step's transformer."""
        method_names = [
            "transform",
            "predict",
            "predict_proba",
            "predict_log_proba",
            "score",
        ]
        final_transformer = self.steps[-1][-1]
        for method_name in method_names:
            fit_method_name = f"fit_{method_name}"
            if hasattr(final_transformer, method_name):
                # Attach the methods to the instance.
                setattr(
                    self,
                    fit_method_name,
                    _make_method(method_name, fit=True).__get__(self, type(self)),
                )
                setattr(
                    self,
                    method_name,
                    _make_method(method_name, fit=False).__get__(self, type(self)),
                )
            else:
                # Remove these methods if they exist on the instance.
                if fit_method_name in self.__dict__:
                    del self.__dict__[fit_method_name]
                if method_name in self.__dict__:
                    del self.__dict__[method_name]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            Ignored.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        Examples
        --------
        >>> pipeline.get_params()
        {'include_indices': True, 'verbose': , ...}
        """
        params = self._get_params("steps", deep=deep)
        # Ensure top-level params are present as well so set_params can set them
        params.update(
            {"include_indices": self.include_indices, "verbose": self.verbose}
        )
        return params

    def set_params(self, **params: Any) -> "SequentialCVPipeline":
        """
        Set parameters for this estimator and delegated steps.

        Parameters
        ----------
        **params : dict
            Pipeline-level and nested step parameters.

        Returns
        -------
        SequentialCVPipeline
            The estimator instance.

        Examples
        --------
        >>> pipeline.set_params(verbose=True)
        SequentialCVPipeline(...)
        """
        # handle pipeline-level params first
        if "include_indices" in params:
            self.include_indices = params.pop("include_indices")
        if "verbose" in params:
            self.verbose = params.pop("verbose")

        # delegate the rest (nested step params) to helper
        self._set_params("steps", **params)
        return self

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union["SequentialCVPipeline", Tuple]:
        if isinstance(key, slice):
            # Create a deep copy to include the fitted state.
            new_pipe = copy.deepcopy(self)
            new_pipe.steps = self.steps[key]
            new_pipe.cv_steps = self.cv_steps[key]
            new_pipe._inject_dynamic_methods()
            return new_pipe
        elif isinstance(key, int):
            return self.steps[key][1]
        else:
            raise TypeError("Invalid index type. Expected int or slice.")

    def _subset(
        self,
        X: Union[IntoDataFrame, IntoSeries, np.ndarray],
        indices: Union[int, NDArray[np.int64]],
    ) -> Union[IntoDataFrame, IntoSeries, np.ndarray]:
        """
        Subset the input X based on provided indices.
        """
        # Use narwhals for dataframe-agnostic operations
        X_nw = nw.from_native(X, pass_through=True)

        return _safe_indexing(X_nw, indices, to_native=True)

    def _append_indexed_output(
        self, output_list: List, test_idx: Union[List, NDArray], output: Any
    ) -> None:
        """
        Helper to append outputs with their corresponding indices.

        Handles both scalar outputs (e.g., from score()) and array-like outputs.
        """
        if np.isscalar(output):
            output_list.append((test_idx[0], output))
        else:
            for i, idx in enumerate(test_idx):
                output_list.append((idx, self._subset(output, i)))

    def _combine(
        transformed_list: List,
    ) -> Union[ArrayLike, List]:
        """
        Combine a list of transformed outputs into a single output.
        """
        if not transformed_list:
            return transformed_list
        first_item = transformed_list[0]
        if isinstance(first_item, np.ndarray):
            return np.vstack(transformed_list)
        elif hasattr(first_item, "pipe"):
            # Use narwhals for dataframe-agnostic concatenation
            return nw.concat(transformed_list)
        else:
            return transformed_list

    def _apply_method_to_indices(
        self,
        model: EstimatorLike,
        method_name: str,
        X: ArrayLike,
        indices: Union[int, NDArray[np.int64]],
        y: Optional[ArrayLike] = None,
    ) -> List:
        """
        Apply a method of the given model to a subset of X based on indices.

        Parameters
        ----------
        model : EstimatorLike
            Fitted model or transformer.
        method_name : str
            The name of the method to apply (e.g., 'predict' or 'transform').
        X : ArrayLike
            Input data.
        indices : Union[int, NDArray[np.int64]]
            Indices specifying the subset of X.
        y : Optional[ArrayLike]
            Target values (required for some methods like 'score'). Default is None.

        Returns
        -------
        List
            List of predictions or transformed values.
        """
        X_subset = self._subset(X, indices)
        y_subset = None if y is None else self._subset(y, indices)
        if isinstance(X_subset, np.ndarray) and X_subset.ndim == 1:
            expected = getattr(model, "n_features_in_", None)
            if expected is not None and X_subset.shape[0] == expected:
                X_subset = X_subset.reshape(1, -1)
            else:
                X_subset = X_subset.reshape(-1, 1)

        output = _call_method_with_correct_args(model, method_name, X_subset, y_subset)
        return output

    def _fit_method_step(  # type: ignore
        self,
        transformer: EstimatorLike,
        X: ArrayLike,
        y: Union[ArrayLike, None],
        cv: Optional[Union[PanelSplit, Iterable]] = None,
        return_output: bool = True,
        method: str = "transform",
    ):
        """
        Fit one pipeline step and optionally transform the data.

        If `cv` is None, the transformer is cloned, fit on X (and y), and optionally used
        to transform X. If `cv` is provided, the transformer is cloned and fit on each fold,
        and out-of-fold predictions are reassembled.

        Parameters
        ----------
        transformer : EstimatorLike
            The transformer or estimator to be fitted.
        X : ArrayLike
            Input data for fitting.
        y : Union[ArrayLike, None]
            Target values.
        cv : Optional[Union[PanelSplit, Iterable]]
            Cross-validation splitting strategy. Default is None.
        return_output : bool
            Whether to return the transformed output. Default is True.
        method : str
            Method to call on the transformer (e.g., 'predict' or 'transform'). Default is "transform"

        Returns
        -------
        Tuple
            If `return_output` is True, returns (transformed_output, fitted_model);
            otherwise returns (None, fitted_model).
        """
        if cv is None:
            model = clone(transformer)
            model.fit(X, y)
            fitted = (None, None, model)
            if return_output:
                # return_group is not considered here as cv == None.
                if self.include_indices:
                    output = (
                        np.arange(len(X)),
                        _call_method_with_correct_args(model, method, X, y),
                    )
                else:
                    output = _call_method_with_correct_args(model, method, X, y)
        else:
            splits = check_cv(cv, X=X, y=y)
            idx_trans: List = []
            folds_models = []
            for train_idx, test_idx in splits:
                model_fold = clone(transformer)
                X_train = self._subset(X, train_idx)
                y_train = None if y is None else self._subset(y, train_idx)
                X_test = self._subset(X, test_idx)
                y_test = None if y is None else self._subset(y, test_idx)
                model_fold.fit(X_train, y_train)

                folds_models.append((train_idx, test_idx, model_fold))

                if return_output:
                    if self.return_group == "test":
                        # Use _call_method_with_correct_args to handle methods like score that need y
                        output_trans = _call_method_with_correct_args(
                            model_fold, method, X_test, y_test
                        )
                        # Pair each output with its original index
                        self._append_indexed_output(idx_trans, test_idx, output_trans)
                    else:
                        output_trans = _call_method_with_correct_args(
                            model_fold, method, X_train, y_train
                        )
                        self._append_indexed_output(idx_trans, train_idx, output_trans)
            if return_output:
                output = _sort_and_combine(
                    idx_trans, include_indices=self.include_indices
                )

            fitted = folds_models  # type: ignore
        if return_output:
            return output, fitted
        else:
            return None, fitted

    def _method_step(
        self,
        fitted_model: EstimatorLike,
        method_name: str,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
    ) -> Union[NDArray, IntoDataFrame, IntoSeries, Tuple[NDArray, Any]]:
        """
        Apply a method of a fitted model (or models) to input X.

        For steps with CV, this method applies the given method to each fold's model and
        reassembles the predictions.
        """
        if not isinstance(fitted_model, list):
            if hasattr(fitted_model[-1], method_name):
                output = _call_method_with_correct_args(
                    fitted_model[-1], method_name, X, y
                )
                if self.include_indices:
                    return np.arange(len(X)), output
                else:
                    return output
            else:
                raise ValueError(f"Fitted model does not have a {method_name} method.")
        predictions_with_idx: List = []
        for train_idx, test_idx, model in fitted_model:
            if self.return_group == "test":
                output_list = self._apply_method_to_indices(
                    model, method_name, X, test_idx, y
                )

                self._append_indexed_output(predictions_with_idx, test_idx, output_list)
            else:
                output_list = self._apply_method_to_indices(
                    model, method_name, X, train_idx, y
                )

                self._append_indexed_output(
                    predictions_with_idx, train_idx, output_list
                )

        return _sort_and_combine(
            predictions_with_idx, include_indices=self.include_indices
        )

    def _fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> ArrayLike:
        """
        Fit the pipeline sequentially, passing the transformed output of each step to the next.
        """
        _check_X_y(X, y)
        # Initialize fitted_steps_ here to comply with sklearn conventions
        # This ensures the attribute only exists after fitting
        self.fitted_steps_: Dict = {}

        X_current = X
        total_steps = len(self.steps)
        for step_idx, (name, transformer) in enumerate(self.steps, start=1):
            if transformer is None or transformer == "passthrough":
                self.fitted_steps_[name] = None
                continue
            not_final_step = not step_idx == total_steps
            X_current, fitted_model = self._fit_method_step(
                transformer,
                X_current,
                y,
                self.cv_steps[step_idx - 1],
                return_output=not_final_step,
            )

            self.fitted_steps_[name] = fitted_model
            _log_message(
                "Step '{}' completed".format(name),
                self.verbose,
                step_idx,
                total_steps,
                time.time() - 0,
            )  # time delta omitted for brevity
        return X_current

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> "SequentialCVPipeline":
        """
        Fit the entire pipeline to the data.

        Parameters
        ----------
        X : ArrayLike
            Input data.
        y : Optional[ArrayLike]
            Target values.

        Returns
        -------
        "SequentialCVPipeline"
            The fitted pipeline instance.

        Examples
        --------
        >>> import numpy as np
        >>> from panelsplit.cross_validation import PanelSplit
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.linear_model import LogisticRegression
        >>> # For demonstration, we use a simple dataset
        >>> period = np.array([1, 2, 3, 4])
        >>> X = np.array([[4, 1], [1, 3], [5, 7], [6, 7]])
        >>> y = np.array([0, 1, 1, 0])
        >>> ps_1 = PanelSplit(
        ...     periods=period, n_splits=2, include_first_train_in_test=True
        ... )
        >>> ps_2 = PanelSplit(periods=period, n_splits=2)
        >>> pipeline = SequentialCVPipeline(
        ...     [
        ...         ("scaler", StandardScaler()),
        ...         ("classifier", LogisticRegression()),
        ...     ],
        ...     cv_steps=[ps_1, ps_2],
        ... )
        >>> pipeline.fit(X, y)
        """
        _ = self._fit(X, y)
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:  # type: ignore[empty-body, return]
        """
        Transform the input data using the fitted pipeline.

        This method applies the transformation steps on the data,
        returning the transformed output.

        Parameters
        ----------
        X : ArrayLike
            Input data.

        Returns
        -------
        ArrayLike
            Transformed output.

        Examples
        --------
        >>> pipeline.transform(X)
        array([[...], [...]])
        """
        ...
        pass

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> ArrayLike:  # type: ignore[empty-body, return]
        """
        Fit the pipeline and transform the input data.

        This method fits the pipeline on the provided data and then applies the transformation steps,
        returning the transformed output.

        Parameters
        ----------
        X : ArrayLike
            Input data.
        y : Optional[ArrayLike]
            Target values (used during fitting).

        Returns
        -------
        ArrayLike
            Transformed output, in the same format as X.

        Examples
        --------
        >>> pipeline.fit_transform(X, y)
        array([[...], [...]])
        """
        pass

    def predict(self, X: ArrayLike) -> ArrayLike:  # type: ignore[empty-body, return]
        """
        Transform the data and predict the class.

        This method fits the pipeline on the provided data and then applies the final estimator's
        predict method to predict the class.

        Parameters
        ----------
        X : ArrayLike
            Input data.

        Returns
        -------
        ArrayLike
            Predicted class.

        Notes
        -----
            This method is dynamically injected based on the final estimator's capabilities.

        Examples
        --------
        >>> pipeline.predict(X)
        array([0, 1, 1, 0])
        """
        pass

    def fit_predict(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> ArrayLike:  # type: ignore[empty-body, return]
        """
        Fit the pipeline and predict target values.

        This method dynamically fits the pipeline on the provided data and then applies the final estimator's
        predict method to generate predictions.
        This method dynamically fits the pipeline on the provided data and then applies the final estimator's
        predict method to generate predictions.

        Parameters
        ----------
        X : ArrayLike
            Input data.
        y : Optional[ArrayLike]
            Target values (used during fitting).

        Returns
        -------
        ArrayLike
            Predicted target values.

        Notes
        ------
        This method is dynamically injected based on the final step of the pipeline.

        Examples
        --------
        >>> pipeline.fit_predict(X, y)
        array([0, 1, 1, 0])
        """
        pass

    def predict_proba(self, X: ArrayLike) -> ArrayLike:  # type: ignore[empty-body, return]
        """
        Predict class probabilities using the fitted pipeline.

        This method applies the final estimator's predict_proba method on the transformed data to predict class probabilities.

        Parameters
        ----------
        X : ArrayLike
            Input data.

        Returns
        -------
        ArrayLike
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.

        Notes
        -----
        This method is dynamically generated based on the final estimator's capabilities.

        Examples
        --------
        >>> pipeline.predict_proba(X)
        array([[0.1, 0.9], [0.8, 0.2]])
        """
        pass

    def fit_predict_proba(  # type: ignore[empty-body, return]
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        Fit the pipeline and predict class probabilities.

        This method fits the pipeline on the provided data and then applies the final estimator's
        predict_proba method to compute class probabilities.

        Parameters
        ----------
        X : ArrayLike
            Input data.
        y : Optional[ArrayLike]
            Target data.

        Returns
        -------
        ArrayLike
            Predicted class probabilities.

        Notes
        -----
        This method is dynamically injected based on the final estimator's capabilities.

        Examples
        --------
        >>> pipeline.fit_predict_proba(X, y)
        array([[0.1, 0.9], [0.8, 0.2]])
        """
        pass

    def score(self, X: ArrayLike, y: ArrayLike) -> float:  # type: ignore[empty-body, return]
        """
        Compute the score.

        This method applies the final estimator's score method on the transformed data to compute a performance score.

        Parameters
        ----------
        X : ArrayLike
            Input data.
        y : ArrayLike
            True target values.

        Returns
        -------
        float
            Computed performance score.

        Notes
        -----
        This method is dynamically injected based on the final estimator's capabilities.

        Examples
        --------
        >>> pipeline.score(X, y)
        0.85
        """
        pass

    def fit_score(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> float:  # type: ignore[empty-body, return]
        """
        Fit the pipeline and compute the score.

        This method fits the pipeline on the provided data and then computes a performance score using the final estimator's
        score method.

        Parameters
        ----------
        X : ArrayLike
            Input data.
        y : ArrayLike, default = None
            Target values (used during fitting and scoring).

        Returns
        -------
        float
            Computed performance score.

        Notes
        -----
        This method is dynamically injected based on the final estimator's capabilities.

        Examples
        --------
        >>> pipeline.fit_score(X, y)
        0.87
        """
        pass

    @property
    def named_steps(self) -> Bunch:
        """
        Access the steps by name.

        Read-only attribute to access any step by given name.
        Keys are steps names and values are the steps objects.

        Returns
        -------
        Bunch
            A Bunch object containing steps keyed by name.

        Examples
        --------
        >>> pipeline.named_steps["scaler"]
        StandardScaler(...)
        >>> pipeline.named_steps["classifier"]
        LogisticRegression(...)
        """
        # Use Bunch object to improve autocomplete
        return Bunch(**dict(self.steps))

    @property
    def _final_estimator(self) -> Optional[Union[str, EstimatorLike]]:
        try:
            estimator = self.steps[-1][1]
            return "passthrough" if estimator is None else estimator
        except (ValueError, AttributeError, TypeError):
            # This condition happens when a call to a method is first calling
            # `_available_if` and `fit` did not validate `steps` yet. We
            # return `None` and an `InvalidParameterError` will be raised
            # right after.
            return None

    @property
    def classes_(self) -> Optional[NDArray]:
        """
        The class labels of the final step.

        This property is available only if the last step of the pipeline is a classifier.
        Raises a `NotFittedError` if the pipeline has not been fitted yet.

        Returns
        -------
        Optional[NDArray]
            Array of class labels from the final classifier step, or None if unavailable.

        Examples
        --------
        >>> pipeline.fit(X, y)
        >>> pipeline.classes_
        array([0, 1])
        """
        if not hasattr(self, "fitted_steps_") or not self.fitted_steps_:
            raise NotFittedError(f"{type(self).__name__} instance is not fitted yet.")

        last_step_name = self.steps[-1][0]
        fitted_step = self.fitted_steps_.get(last_step_name, None)

        if fitted_step is None:
            raise NotFittedError(
                f"No fitted model found for final step '{last_step_name}'."
            )
        # Case A: cross-validated storage: dict with 'splits': [(test_idx, model), ...]
        elif fitted_step[0] is not None:
            [check_is_fitted(model) for _, _, model in fitted_step]
            assert all([is_classifier(model) for _, _, model in fitted_step])
            return np.unique(
                np.concatenate([model[-1].classes_ for model in fitted_step])
            )
        # Case B: direct estimator instance (non-CV)
        else:
            check_is_fitted(fitted_step[-1])
            assert is_classifier(fitted_step[-1])
            return getattr(fitted_step[-1], "classes_", None)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()

        if not self.steps:
            return tags

        try:
            if self.steps[0][1] is not None and self.steps[0][1] != "passthrough":
                tags.input_tags.pairwise = get_tags(
                    self.steps[0][1]
                ).input_tags.pairwise
            # WARNING: the sparse tag can be incorrect.
            # Some Pipelines accepting sparse data are wrongly tagged sparse=False.
            # For example Pipeline([PCA(), estimator]) accepts sparse data
            # even if the estimator doesn't as PCA outputs a dense array.
            tags.input_tags.sparse = all(
                get_tags(step).input_tags.sparse
                for name, step in self.steps
                if step is not None and step != "passthrough"
            )
        except (ValueError, AttributeError, TypeError):
            # This happens when the `steps` is not a list of (name, estimator)
            # tuples and `fit` is not called yet to validate the steps.
            pass

        try:
            if self.steps[-1][1] is not None and self.steps[-1][1] != "passthrough":
                last_step_tags = get_tags(self.steps[-1][1])
                tags.estimator_type = last_step_tags.estimator_type
                tags.target_tags.multi_output = last_step_tags.target_tags.multi_output
                tags.classifier_tags = deepcopy(last_step_tags.classifier_tags)
                tags.regressor_tags = deepcopy(last_step_tags.regressor_tags)
                tags.transformer_tags = deepcopy(last_step_tags.transformer_tags)
        except (ValueError, AttributeError, TypeError):
            # This happens when the `steps` is not a list of (name, estimator)
            # tuples and `fit` is not called yet to validate the steps.
            pass

        return tags

    def __getstate__(self) -> dict:
        """
        Prepare instance dictionary for pickling.

        Removes dynamically injected method attributes (bound methods and fit_* wrappers)
        that may not be reliably picklable across process boundaries.

        Returns
        -------
        dict
            The instance dictionary suitable for pickling.

        Examples
        --------
        >>> state = pipeline.__getstate__()
        >>> isinstance(state, dict)
        True
        """
        state = self.__dict__.copy()

        # Names of methods injected by _inject_dynamic_methods
        injected_method_names = [
            "transform",
            "predict",
            "predict_proba",
            "predict_log_proba",
            "score",
            "fit_transform",
            "fit_predict",
            "fit_predict_proba",
            "fit_score",
            "fit_transform",
            "fit_predict",
            "fit_predict_proba",
            "fit_score",
        ]
        # Also remove corresponding 'fit_<method>' names if present.
        injected_method_names += [
            f"fit_{name}"
            for name in [
                "transform",
                "predict",
                "predict_proba",
                "predict_log_proba",
                "score",
            ]
        ]

        for name in injected_method_names:
            if name in state:
                state.pop(name, None)

        # Defensive: remove any other attributes that are callables and likely to be bound methods
        # but keep key estimator state like fitted_steps_ etc.
        for k, v in list(state.items()):
            # don't remove core attributes (heuristic)
            if k in {
                "steps",
                "cv_steps",
                "fitted_steps_",
                "include_indices",
                "verbose",
                "return_group",
                "_estimator_type",
            }:
                continue
            if callable(v):
                state.pop(k, None)

        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restore state after unpickling.

        Re-injects dynamic methods based on the (restored) final estimator.

        Parameters
        ----------
        state : dict
            The instance dictionary previously returned by `__getstate__`.

        Examples
        --------
        >>> pipeline.__setstate__(state)
        """
        self.__dict__.update(state)
        # Re-create the injected convenience methods (only if final estimator supports them)
        try:
            # _inject_dynamic_methods checks the final estimator and only adds methods that exist.
            self._inject_dynamic_methods()
        except Exception:
            # If anything goes wrong here, we don't want to raise during unpickling.
            # Swallow but log to stderr so developer can notice during debugging.
            import warnings

            warnings.warn(
                "Could not re-inject dynamic methods on SequentialCVPipeline after unpickling.",
                RuntimeWarning,
            )
