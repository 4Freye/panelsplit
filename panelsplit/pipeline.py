import numpy as np
import pandas as pd
import time
from sklearn.base import BaseEstimator, TransformerMixin, clone
from .utils.validation import check_cv

def _log_message(message, verbose, step_idx, total, elapsed_time=None):
    """Log messages similar to scikit-learn's Pipeline verbose output."""
    if verbose:
        if elapsed_time is not None:
            message += " (elapsed time: %5.1fs)" % elapsed_time
        print("[SequentialCVPipeline] ({}/{}) {}".format(step_idx + 1, total, message))

def _make_method(method_name):
    """
    Generate a pipeline method for methods such as:
      - fit_transform, fit_predict, fit_predict_proba, fit_score, etc.
      - transform, predict, predict_proba, score, etc.
    
    If the method name contains "fit", then the pipeline will first fit all steps 
    (including the final one) on the input X (and y) and then call the corresponding 
    method on the final estimator (with the "fit_" prefix removed).
    
    If "fit" is not in the method name, then the fitted estimators are used to 
    sequentially transform (or predict) the input data.
    """
    def method(self, X, y=None, **kwargs):
        # Case 1: "fit" is in the method name: do full fitting and then call the method.
        if "fit" in method_name:
            X_current = X
            total_steps = len(self.steps)
            # Fit all steps except the final one.
            for step_idx, (name, transformer, cv) in enumerate(self.steps[:-1]):
                if transformer is None or transformer == "passthrough":
                    self.fitted_steps_[name] = None
                    continue
                t_start = time.time()
                X_current, fitted_model = self._fit_step(transformer, X_current, y, cv)
                self.fitted_steps_[name] = fitted_model
                _log_message("Step '{}' completed".format(name),
                             self.verbose, step_idx, total_steps, time.time() - t_start)
            # Fit the final estimator.
            final_name, final_transformer, final_cv = self.steps[-1]
            final_model = clone(final_transformer)
            t_start = time.time()
            final_model.fit(X_current, y)
            _log_message("Final step '{}' fitted".format(final_name),
                         self.verbose, total_steps - 1, total_steps, time.time() - t_start)
            self.fitted_steps_[final_name] = final_model

            # Determine which method to call on the final estimator.
            # For example, "fit_predict" becomes "predict", "fit_transform" becomes "transform", etc.
            if method_name.startswith("fit_"):
                final_method_name = method_name[len("fit_"):]
            else:
                final_method_name = method_name

            if final_method_name == "score":
                return getattr(final_model, final_method_name)(X_current, y, **kwargs)
            else:
                return getattr(final_model, final_method_name)(X_current, **kwargs)

        # Case 2: No "fit" in method name: assume the pipeline is already fitted.
        else:
            if method_name == "transform":
                X_current = X
                for name, transformer, cv in self.steps:
                    if transformer is None or transformer == "passthrough":
                        continue
                    fitted_model = self.fitted_steps_.get(name)
                    if fitted_model is None:
                        continue
                    X_current = self._method_step(fitted_model, "transform", X_current)
                return X_current
            elif method_name in ["predict", "predict_proba", "predict_log_proba"]:
                X_current = X
                for name, transformer, cv in self.steps[:-1]:
                    if transformer is None or transformer == "passthrough":
                        continue
                    fitted_model = self.fitted_steps_.get(name)
                    if fitted_model is None:
                        continue
                    X_current = self._method_step(fitted_model, "transform", X_current)
                final_name, final_transformer, final_cv = self.steps[-1]
                fitted_final = self.fitted_steps_.get(final_name)
                if fitted_final is None:
                    raise ValueError("Final estimator is not fitted.")
                return self._method_step(fitted_final, method_name, X_current)
            elif method_name == "score":
                X_current = X
                for name, transformer, cv in self.steps[:-1]:
                    if transformer is None or transformer == "passthrough":
                        continue
                    fitted_model = self.fitted_steps_.get(name)
                    if fitted_model is None:
                        continue
                    X_current = self._method_step(fitted_model, "transform", X_current)
                final_name, final_transformer, final_cv = self.steps[-1]
                fitted_final = self.fitted_steps_.get(final_name)
                if fitted_final is None:
                    raise ValueError("Final estimator is not fitted.")
                return getattr(fitted_final, "score")(X_current, y, **kwargs)
            else:
                raise ValueError(f"Method {method_name} not supported.")
    return method

class SequentialCVPipeline(BaseEstimator):
    """
    A sequential pipeline that applies a series of transformers/estimators,
    each with an optional cross-validation (CV) strategy.
    
    Each step is a tuple of (name, transformer, cv), where:
      - name: a string identifier.
      - transformer: an estimator/transformer with fit/transform (or predict) methods.
      - cv: either a CV splitter (an object with a split method), an iterable of train/test splits,
            or None. If cv is None, the transformer is fit on the entire data.
    
    For steps with cv specified, the pipeline:
      1. For each fold in cv, clones and fits the transformer on the training set,
         then uses it to transform (or predict) the held‐out test set.
      2. Reassembles the out‐of‐fold results so that each row is produced
         by a model that was not “seen” during its fit.
    
    Example usage:
    
      pipeline = SequentialCVPipeline([
          ('imputer', IterativeImputer(), cv=panel_cv),  # fit iteratively using CV
          ('scaler', StandardScaler(), cv=None),         # fit on full data
          ('estimator', RFRegressor(), cv=fold_cv)         # fit estimator using CV
      ], verbose=True)
      
      pipeline.fit(X, y)
      Xt = pipeline.transform(X_new)
      y_pred = pipeline.predict(X_new)
      # Or:
      y_pred = pipeline.fit_predict(X, y)
    """
    def __init__(self, steps, verbose=False):
        # Each step must be a tuple: (name, transformer, cv)
        for step in steps:
            if not (isinstance(step, tuple) and len(step) == 3):
                raise ValueError("Each step must be a tuple of (name, transformer, cv)")
        self.steps = steps
        self.verbose = verbose
        self.fitted_steps_ = {}  # stores fitted model(s) for each step

        # Dynamically inject methods.
        # Methods that perform a full fit and then call the final method:
        self.fit_predict = _make_method("fit_predict").__get__(self, type(self))
        self.fit_transform = _make_method("fit_transform").__get__(self, type(self))
        self.fit_score = _make_method("fit_score").__get__(self, type(self))
        if hasattr(self.steps[-1][1], "predict_proba"):
            self.fit_predict_proba = _make_method("fit_predict_proba").__get__(self, type(self))
        if hasattr(self.steps[-1][1], "predict_log_proba"):
            self.fit_predict_log_proba = _make_method("fit_predict_log_proba").__get__(self, type(self))
        
        # Methods that use already-fitted estimators:
        self.transform = _make_method("transform").__get__(self, type(self))
        self.predict = _make_method("predict").__get__(self, type(self))
        self.score = _make_method("score").__get__(self, type(self))

    def _subset(self, X, indices):
        """Helper function to index X.
        
        If X is a pandas DataFrame or Series, return X.iloc[indices];
        otherwise, use normal indexing.
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
        Combine a list of transformed outputs.
        - If the outputs are numpy arrays, stack them using np.vstack.
        - If they are pandas DataFrames or Series, concatenate them along axis 0.
        - Otherwise, return the list as-is.
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
        # Ensure indices is array-like
        indices = np.atleast_1d(indices)
        # Convert boolean masks to integer indices if necessary
        if isinstance(indices, np.ndarray) and indices.dtype == bool:
            indices = np.where(indices)[0]
        X_subset = self._subset(X, indices)
        # Ensure X_subset is 2D if it is a numpy array
        if isinstance(X_subset, np.ndarray) and X_subset.ndim == 1:
            # Try to obtain the model's expected number of features
            try:
                expected = model.n_features_in_
            except AttributeError:
                expected = None
            # If the length matches the expected number of features,
            # treat it as a single sample (row vector).
            if expected is not None and X_subset.shape[0] == expected:
                X_subset = X_subset.reshape(1, -1)
            else:
                # Otherwise, assume it should be a column vector.
                X_subset = X_subset.reshape(-1, 1)
        output = getattr(model, method_name)(X_subset)
        try:
            output_list = output.tolist()
        except Exception:
            output_list = list(output)
        return output_list

    def _fit_step(self, transformer, X, y, cv):
        """
        Fit one step. If cv is None, simply clone, fit, and transform X.
        If cv is provided, then for each fold:
          - clone and fit the transformer on training data,
          - apply the method (transform or predict) on test data.
        The out‐of‐fold results are reassembled into X_trans.
        Returns (X_trans, fitted_model) where fitted_model is either the fitted transformer
        or (if cv was used) a dict with key "splits" that contains a list of (test_indices, model) tuples.
        """
        if cv is None:
            model = clone(transformer)
            t_start = time.time()
            print(X)
            model.fit(X, y)
            elapsed = time.time() - t_start
            _log_message("Fitted on full data", self.verbose, 0, 1, elapsed)
            if hasattr(model, "transform"):
                X_trans = model.transform(X)
            elif hasattr(model, "predict"):
                X_trans = model.predict(X)
            else:
                X_trans = X  # if no transformation, pass through
            return X_trans, model
        else:
            splits = check_cv(cv, X=X, y=y)
            n_samples = len(X)
            out_trans = [None] * n_samples
            folds_models = []
            # Loop through folds without verbose logging per fold
            for train_idx, test_idx in splits:
                # Convert boolean masks to integer indices if necessary
                if isinstance(train_idx, np.ndarray) and train_idx.dtype == bool:
                    train_idx = np.where(train_idx)[0]
                if isinstance(test_idx, np.ndarray) and test_idx.dtype == bool:
                    test_idx = np.where(test_idx)[0]
                model_fold = clone(transformer)
                X_train = self._subset(X, train_idx)
                y_train = None if y is None else self._subset(y, train_idx)
                X_test = self._subset(X, test_idx)
                model_fold.fit(X_train, y_train)
                if hasattr(model_fold, "transform"):
                    X_test_trans = model_fold.transform(X_test)
                elif hasattr(model_fold, "predict"):
                    X_test_trans = model_fold.predict(X_test)
                else:
                    raise ValueError("Transformer/estimator must have transform or predict")
                try:
                    X_test_trans_list = X_test_trans.tolist()
                except Exception:
                    X_test_trans_list = [row for row in X_test_trans]
                for i, idx in enumerate(test_idx):
                    out_trans[idx] = X_test_trans_list[i]
                folds_models.append((test_idx, model_fold))
            X_trans = self._combine(out_trans)
            fitted = {"splits": folds_models}
            return X_trans, fitted

    def _method_step(self, fitted_model, method_name, X):
        """
        A helper method to call a given method (e.g. 'predict' or 'transform')
        on a fitted model. For steps fitted with CV, it applies the method on the subset
        of X corresponding to each fold's test indices. The predictions are then reassembled
        into the original order.
        """
        # Non-CV case: fitted_model is the estimator/transformer.
        if not (isinstance(fitted_model, dict) and "splits" in fitted_model):
            if hasattr(fitted_model, method_name):
                return getattr(fitted_model, method_name)(X)
            else:
                raise ValueError(f"Fitted model does not have a {method_name} method.")

        # CV case: fitted_model is a dict with a "splits" key.
        # Collect predictions along with their original indices.
        predictions_with_idx = []
        for test_idx, model in fitted_model["splits"]:
            if hasattr(model, method_name):
                output_list = self._apply_method_to_indices(model, method_name, X, test_idx)
            else:
                raise ValueError(f"Model in CV fold does not have a {method_name} method.")
            for i, idx in enumerate(test_idx):
                predictions_with_idx.append((idx, output_list[i]))

        # Sort the predictions by the original index to preserve order.
        predictions_with_idx.sort(key=lambda pair: pair[0])
        predictions = [pred for idx, pred in predictions_with_idx]

        # If predictions are numpy arrays, stack them.
        if predictions and isinstance(predictions[0], np.ndarray):
            predictions = np.vstack(predictions)
        return predictions
    
    def _fit(self, X, y=None):
        """
        Fit the pipeline sequentially. For each step, use its cv setting to fit and transform
        the data. The transformed output from one step is passed to the next.
        """
        X_current = X
        total_steps = len(self.steps)
        for step_idx, (name, transformer, cv) in enumerate(self.steps):
            if transformer is None or transformer == "passthrough":
                self.fitted_steps_[name] = None
                continue
            t_start = time.time()
            X_current, fitted_model = self._fit_step(transformer, X_current, y, cv)
            self.fitted_steps_[name] = fitted_model
            _log_message("Step '{}' completed".format(name), self.verbose, step_idx, total_steps, time.time() - t_start)
        return X_current

    def fit(self, X, y=None):
        _ = self._fit(X, y)
        return self
    