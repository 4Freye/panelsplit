import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin, clone
from .utils.validation import check_cv, _check_X_y
import pandas as pd  # added import for pandas
import copy

def _log_message(message, verbose, step_idx, total, elapsed_time=None):
    """
    Log messages similar to scikit-learn's Pipeline verbose output.

    Parameters
    ----------
    message : str
        The message to log.
    verbose : bool
        Whether to output verbose logging.
    step_idx : int
        The current step index (zero-indexed).
    total : int
        The total number of steps.
    elapsed_time : float, optional
        Elapsed time for the step in seconds.

    Returns
    -------
    None
    """
    if verbose:
        if elapsed_time is not None:
            message += " (elapsed time: %5.1fs)" % elapsed_time
        print("[SequentialCVPipeline] ({}/{}) {}".format(step_idx, total, message))

def _to_int_indices(indices):
    """
    Convert indices to integer indices.

    Parameters
    ----------
    indices : array-like
        Array of indices, which can be boolean or integer.

    Returns
    -------
    np.ndarray
        Array of integer indices.
    """
    indices = np.atleast_1d(indices)
    if indices.dtype == bool:
        indices = np.where(indices)[0]
    return indices

def _sort_and_combine(predictions_with_idx):
    """
    Sort and combine predictions from (index, prediction) pairs.

    Parameters
    ----------
    predictions_with_idx : list of tuples
        Each tuple contains (index, prediction) where prediction can be a numpy array
        or a pandas object.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame or pandas.Series
        Combined predictions. Numpy arrays are stacked vertically, and pandas objects
        are concatenated.
    """
    predictions_with_idx.sort(key=lambda pair: pair[0])
    predictions = [pred for idx, pred in predictions_with_idx]
    if predictions and isinstance(predictions[0], np.ndarray):
        predictions = np.vstack(predictions)
    elif predictions and isinstance(predictions[0], (pd.DataFrame, pd.Series)):
        predictions = pd.concat(predictions, axis=0)
    else:
        predictions = np.array(predictions)
    return predictions

def _make_method(method_name, fit=True):
    """
    Create a pipeline method dynamically for fitting or predicting.

    Parameters
    ----------
    method_name : str
        The name of the method to create (e.g., 'predict', 'transform').
    fit : bool, default=True
        Whether the method should perform fitting of the pipeline.

    Returns
    -------
    function
        A method that applies the specified method to all pipeline steps.
    """
    def _method(self, X, y=None, **kwargs):
        _check_X_y(X, y)
        # Fit all steps except the final one.
        current_output = X
        total_steps = len(self.steps)

        for step_idx, (name, transformer, cv) in enumerate(self.steps, start=1):
            t_start = time.time()
            not_final_step = total_steps != step_idx
            # Use 'transform' for intermediate steps and the provided method for the final step.
            method = 'transform' if not_final_step else method_name
            if fit:
                if transformer is None or transformer == "passthrough":
                    self.fitted_steps_[name] = None
                    continue

                current_output, fitted_model = self._fit_method_step(
                    transformer, current_output, y, cv, return_output=True, method=method
                )

                self.fitted_steps_[name] = fitted_model
            else:
                fitted_model = self.fitted_steps_.get(name)
                if fitted_model is None:
                    continue
                current_output = self._method_step(fitted_model, method, current_output)

            _log_message("Step '{}' completed".format(name),
                         self.verbose, step_idx, total_steps, time.time() - t_start)

        return current_output

    return _method



class SequentialCVPipeline(BaseEstimator):
    """
    A sequential pipeline that applies a series of transformers/estimators with
    optional cross-validation (CV).

    Each step is a tuple of (name, transformer, cv). If cv is provided, out-of-fold
    outputs are reassembled.

    Parameters
    ----------
    steps : list of tuples
        List where each tuple is (name, transformer, cv). 'name' is a string identifier,
        'transformer' is an estimator or transformer, and 'cv' is a cross-validation splitter
        or None.
    verbose : bool, default=False
        Whether to print verbose output during fitting and transformation.

    Attributes
    ----------
    steps : list
        The list of pipeline steps.
    verbose : bool
        Verbosity flag.
    fitted_steps_ : dict
        Dictionary storing fitted transformers for each step.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import KFold
    >>> pipeline = SequentialCVPipeline([
    ...     ('scaler', StandardScaler(), None),
    ...     ('classifier', LogisticRegression(), None))
    ... ], verbose = False)
    >>> print(pipeline.steps)
    [('scaler', StandardScaler(), None), ('classifier', LogisticRegression(), KFold(n_splits=3))]
    """
    def _inject_dynamic_methods(self):
        """Inject dynamic pipeline methods based on the final step's transformer."""
        method_names = ['transform', "predict", "predict_proba", "predict_log_proba", "score"]
        final_transformer = self.steps[-1][1]
        for method_name in method_names:
            fit_method_name = f"fit_{method_name}"
            if hasattr(final_transformer, method_name):
                # Attach the methods to the instance.
                setattr(self, fit_method_name, _make_method(method_name, fit=True).__get__(self, type(self)))
                setattr(self, method_name, _make_method(method_name, fit=False).__get__(self, type(self)))
            else:
                # Remove these methods if they exist on the instance.
                if fit_method_name in self.__dict__:
                    del self.__dict__[fit_method_name]
                if method_name in self.__dict__:
                    del self.__dict__[method_name]

    def __init__(self, steps, verbose=False):
        # Each step must be a tuple: (name, transformer, cv)
        for step in steps:
            if not (isinstance(step, tuple) and len(step) == 3):
                raise ValueError("Each step must be a tuple of (name, transformer, cv)")
        self.steps = steps
        self.verbose = verbose
        self.fitted_steps_ = {}
        self._inject_dynamic_methods()

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Create a deep copy to include the fitted state.
            new_pipe = copy.deepcopy(self)
            new_pipe.steps = self.steps[key]
            new_pipe._inject_dynamic_methods()
            return new_pipe
        elif isinstance(key, int):
            return self.steps[key]
        else:
            raise TypeError("Invalid index type. Expected int or slice.")

    def _subset(self, X, indices):
        """
        Subset the input X based on provided indices.

        Parameters
        ----------
        X : array-like or pandas.DataFrame or pandas.Series
            The data to subset.
        indices : array-like
            Indices used to select a subset of X.

        Returns
        -------
        array-like or pandas.DataFrame or pandas.Series
            The subset of X corresponding to the given indices.
        """
        try:
            if isinstance(X, (pd.DataFrame, pd.Series)):
                return X.iloc[indices]
            else:
                return X[indices]
        except Exception:
            return np.array([X[i] for i in indices])

    def _combine(self, transformed_list):
        """
        Combine a list of transformed outputs into a single output.

        Parameters
        ----------
        transformed_list : list
            List of transformed outputs from different pipeline steps.

        Returns
        -------
        numpy.ndarray or pandas.DataFrame or pandas.Series or list
            Combined output. Numpy arrays are vertically stacked; pandas objects are concatenated.
        """
        if not transformed_list:
            return transformed_list
        first_item = transformed_list[0]
        if isinstance(first_item, np.ndarray):
            return np.vstack(transformed_list)
        elif isinstance(first_item, (pd.DataFrame, pd.Series)):
            return pd.concat(transformed_list, axis=0)
        else:
            return transformed_list

    def _apply_method_to_indices(self, model, method_name, X, indices):
        """
        Apply a method of the given model to a subset of X based on indices.

        Parameters
        ----------
        model : estimator
            Fitted model or transformer.
        method_name : str
            The name of the method to apply (e.g., 'predict' or 'transform').
        X : array-like or pandas.DataFrame or pandas.Series
            Input data.
        indices : array-like
            Indices specifying the subset of X.

        Returns
        -------
        list
            List of predictions or transformed values.
        """
        indices = _to_int_indices(indices)
        X_subset = self._subset(X, indices)
        if isinstance(X_subset, np.ndarray) and X_subset.ndim == 1:
            expected = getattr(model, "n_features_in_", None)
            if expected is not None and X_subset.shape[0] == expected:
                X_subset = X_subset.reshape(1, -1)
            else:
                X_subset = X_subset.reshape(-1, 1)
        output = getattr(model, method_name)(X_subset)
        return output

    def _fit_method_step(self, transformer, X, y, cv, return_output=True, method='transform'):
        """
        Fit one pipeline step and optionally transform the data.

        If `cv` is None, the transformer is cloned, fit on X (and y), and optionally used
        to transform X. If `cv` is provided, the transformer is cloned and fit on each fold,
        and out-of-fold predictions are reassembled.

        Parameters
        ----------
        transformer : estimator
            The transformer or estimator to be fitted.
        X : array-like or pandas.DataFrame or pandas.Series
            Input data for fitting.
        y : array-like, optional
            Target values.
        cv : cross-validation splitter or None
            Cross-validation splitting strategy.
        return_output : bool, default=True
            Whether to return the transformed output.
        method : str, default='transform'
            Method to call on the transformer (e.g., 'predict' or 'transform').

        Returns
        -------
        tuple
            If `return_output` is True, returns (transformed_output, fitted_model);
            otherwise returns (None, fitted_model).
        """
        if cv is None:
            model = clone(transformer)
            model.fit(X, y)
            fitted = model
            if return_output:
                output = getattr(model, method)(X)
        else:
            splits = check_cv(cv, X=X, y=y)
            idx_trans = []
            folds_models = []
            for train_idx, test_idx in splits:
                train_idx = _to_int_indices(train_idx)
                test_idx = _to_int_indices(test_idx)
                model_fold = clone(transformer)
                X_train = self._subset(X, train_idx)
                y_train = None if y is None else self._subset(y, train_idx)
                X_test = self._subset(X, test_idx)
                model_fold.fit(X_train, y_train)
                folds_models.append((test_idx, model_fold))
                if return_output:
                    output_trans = getattr(model_fold, method)(X_test)
                    # Pair each output with its original index.
                    for i, idx in enumerate(test_idx):
                        idx_trans.append((idx, output_trans[i]))
            if return_output:
                output = _sort_and_combine(idx_trans)
            fitted = {"splits": folds_models}
        if return_output:
            return output, fitted
        else:
            return None, fitted

    def _method_step(self, fitted_model, method_name, X):
        """
        Apply a method of a fitted model (or models) to input X.

        For steps with CV, this method applies the given method to each fold's model and
        reassembles the predictions.

        Parameters
        ----------
        fitted_model : estimator or dict
            Fitted model or a dictionary containing fitted models from cross-validation.
        method_name : str
            The name of the method to apply (e.g., 'predict' or 'transform').
        X : array-like or pandas.DataFrame or pandas.Series
            Input data.

        Returns
        -------
        numpy.ndarray or pandas.DataFrame or pandas.Series
            Combined output after applying the method.
        """
        if not (isinstance(fitted_model, dict) and "splits" in fitted_model):
            if hasattr(fitted_model, method_name):
                return getattr(fitted_model, method_name)(X)
            else:
                raise ValueError(f"Fitted model does not have a {method_name} method.")
        predictions_with_idx = []
        for test_idx, model in fitted_model["splits"]:
            output_list = self._apply_method_to_indices(model, method_name, X, test_idx)
            for i, idx in enumerate(test_idx):
                predictions_with_idx.append((idx, self._subset(output_list, i)))
        return _sort_and_combine(predictions_with_idx)

    def _fit(self, X, y=None):
        """
        Fit the pipeline sequentially, passing the transformed output of each step to the next.

        Parameters
        ----------
        X : array-like or pandas.DataFrame or pandas.Series
            Input data.
        y : array-like, optional
            Target values.

        Returns
        -------
        array-like or pandas.DataFrame or pandas.Series
            Transformed output after applying all pipeline steps.
        """
        _check_X_y(X, y)
        X_current = X
        total_steps = len(self.steps)
        for step_idx, (name, transformer, cv) in enumerate(self.steps, start=1):
            if transformer is None or transformer == "passthrough":
                self.fitted_steps_[name] = None
                continue
            not_final_step = not step_idx == total_steps
            X_current, fitted_model = self._fit_method_step(transformer, X_current, y, cv, return_output=not_final_step)
            
            self.fitted_steps_[name] = fitted_model
            _log_message("Step '{}' completed".format(name),
                         self.verbose, step_idx, total_steps, time.time() - 0)  # time delta omitted for brevity
        return X_current

    def fit(self, X, y=None):
        """
        Fit the entire pipeline to the data.

        Parameters
        ----------
        X : array-like or pandas.DataFrame or pandas.Series
            Input data.
        y : array-like, optional
            Target values.

        Returns
        -------
        SequentialCVPipeline
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
        >>> ps_1 = PanelSplit(periods=period, n_splits=2, include_first_train_in_test = True)
        >>> ps_2 = PanelSplit(periods=period, n_splits=2)
        >>> pipeline = SequentialCVPipeline([
        ...     ('scaler', StandardScaler(), ps_1),
        ...     ('classifier', LogisticRegression(), ps_2)
        ... ])
        >>> pipeline.fit(X, y)
        """
        _ = self._fit(X, y)
        return self
    
    def transform(self, X):
        """
        Transform the input data using the fitted pipeline.

        This method applies the transformation steps on the data,
        returning the transformed output.

        Parameters
        ----------
        X : array-like or pandas.DataFrame or pandas.Series
            Input data.
        y : array-like, optional
            Target values (used during fitting).

        Returns
        -------
        array-like or pandas.DataFrame or pandas.Series
            Transformed output.
        """
        pass

    def fit_transform(self, X, y=None):
        """
        Fit the pipeline and transform the input data.

        This method fits the pipeline on the provided data and then applies the transformation steps,
        returning the transformed output.

        Parameters
        ----------
        X : array-like or pandas.DataFrame or pandas.Series
            Input data.
        y : array-like, optional
            Target values (used during fitting).

        Returns
        -------
        array-like or pandas.DataFrame or pandas.Series
            Transformed output.

        Note
        ----
        This method is dynamically injected based on the final estimator's capabilities.
        """
        pass

    def predict(self, X):
        """
        Transform the data and predict the class.

        This method fits the pipeline on the provided data and then applies the final estimator's
        predict method to predict the class.

        Parameters
        ----------
        X : array-like or pandas.DataFrame or pandas.Series
            Input data.
        y : array-like, optional
            Target values (used during fitting).

        Returns
        -------
        array-like or pandas.DataFrame or pandas.Series
            Predicted class.

        Note
        ----
        This method is dynamically injected based on the final estimator's capabilities.
        """
        pass

    def fit_predict(self, X, y=None):
       """
       Fit the pipeline and predict target values.

        This method dynamically fits the pipeline on the provided data and then applies the final estimator's
        predict method to generate predictions.

        Parameters
        ----------
        X : array-like or pandas.DataFrame or pandas.Series
            Input data.
        y : array-like, optional
            Target values (used during fitting).

        Returns
        -------
        array-like or pandas.DataFrame or pandas.Series
            Predicted target values.

        Note
        ----
        This method is dynamically injected based on the final step of the pipeline.
        """
       pass

    def predict_proba(self, X):
        """
        Predict class probabilities using the fitted pipeline.
        
        This method applies the final estimator's predict_proba method on the transformed data to predict class probabilities.

        Parameters
        ----------
        X : array-like or pandas.DataFrame or pandas.Series
            Input data.
        y : array-like
            True target values.

        Returns
        -------
        float
            Predicted class probability.
        
        Note:
        -----
        This is a placeholder method for documentation purposes only.
        The actual method is dynamically generated based on the final estimator's capabilities.
        """
        pass

    def fit_predict_proba(self, X, y=None):
        """
        Fit the pipeline and predict class probabilities.

        This method fits the pipeline on the provided data and then applies the final estimator's
        predict_proba method to compute class probabilities.

        Parameters
        ----------
        X : array-like or pandas.DataFrame or pandas.Series
            Input data.
        y : array-like, optional
            Target values (used during fitting).

        Returns
        -------
        array-like or pandas.DataFrame or pandas.Series
            Predicted class probabilities.

        Note
        ----
        This method is dynamically injected based on the final estimator's capabilities.
        """
        pass

    def score(self, X, y):
        """
        Compute the score.

        This method applies the final estimator's score method on the transformed data to compute a performance score.

        Parameters
        ----------
        X : array-like or pandas.DataFrame or pandas.Series
            Input data.
        y : array-like
            True target values.

        Returns
        -------
        float
            Computed performance score.

        Note
        ----
        This method is dynamically injected based on the final estimator's capabilities.
        """
        pass

    def fit_score(self, X, y=None):
        """
        Fit the pipeline and compute the score.

        This method fits the pipeline on the provided data and then computes a performance score using the final estimator's
        score method.

        Parameters
        ----------
        X : array-like or pandas.DataFrame or pandas.Series
            Input data.
        y : array-like, optional
            Target values (used during fitting and scoring).

        Returns
        -------
        float
            Computed performance score.

        Note
        ----
        This method is dynamically injected based on the final estimator's capabilities.
        """
        pass