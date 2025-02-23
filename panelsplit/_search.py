import time
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.metrics import check_scoring

class BaseSearchSequentialCV:
    """
    Base class to perform hyperparameter search on an estimator (e.g. a SequentialCVPipeline)
    that already encapsulates cross-validation (via its internal CV settings on each step).
    
    Parameters
    ----------
    estimator : object
        A scikit-learn estimator (or pipeline) implementing fit and predict.
        In our context this is expected to be a SequentialCVPipeline.
        
    param_grid : dict or list of dict
        Dictionary or list of dictionaries with parameters names (str) as keys and lists
        of parameter settings to try as values. The keys must match those of the estimator.
        
    scoring : string or callable, default=None
        A string (see scikit-learn model evaluation documentation) or a scorer callable.
        If None, the estimator's default score method is used.
        
    refit : bool, default=True
        If True, refit an estimator using the best found parameters on the whole dataset.
        
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
        
    n_jobs : int, default=1
        Number of jobs to run in parallel. (Not implemented in this basic version.)
    
    Attributes
    ----------
    best_score_ : float
        Mean cross-validated score of the best_estimator.
        
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
        
    best_estimator_ : object
        Estimator that was chosen by the search, i.e. estimator which gave highest score
        (if refit=True).
        
    cv_results_ : list of dict
        A list where each entry corresponds to one candidate parameters set, containing
        keys "params", "mean_test_score", and "fit_time".
    """
    def __init__(self, estimator, param_grid, scoring=None, 
                 refit=True, error_score=np.nan, verbose=0, n_jobs=1):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.refit = refit
        self.error_score = error_score
        self.verbose = verbose
        self.n_jobs = n_jobs  # Not used in this basic implementation.
    
    def fit(self, X, y=None, **fit_params):
        # Generate all parameter combinations
        grid = list(ParameterGrid(self.param_grid))
        best_score = -np.inf
        best_params = None
        best_estimator = None
        results = []
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        
        if self.verbose:
            print(f"Fitting {len(grid)} candidate parameter combinations...")
            
        for params in grid:
            if self.verbose:
                print(f"Trying parameters: {params}")
            # Clone the estimator to not contaminate results between iterations
            est = clone(self.estimator)
            est.set_params(**params)
            try:
                start_time = time.time()
                est.fit(X, y, **fit_params)
                score = scorer(est, X, y)
                fit_time = time.time() - start_time
            except Exception as e:
                if self.verbose:
                    print("Error while fitting parameters:", params)
                    print("Error:", e)
                score = self.error_score
                fit_time = np.nan
            results.append({
                "params": params,
                "mean_test_score": score,
                "fit_time": fit_time
            })
            if score > best_score:
                best_score = score
                best_params = params
                best_estimator = est
        
        self.best_score_ = best_score
        self.best_params_ = best_params
        if self.refit:
            self.best_estimator_ = best_estimator
        else:
            self.best_estimator_ = None
        self.cv_results_ = results
        if self.verbose:
            print("Best parameters:", best_params)
            print("Best score:", best_score)
        return self

    def predict(self, X):
        if self.best_estimator_ is None:
            raise ValueError("No estimator found. Please call fit() before predict().")
        return self.best_estimator_.predict(X)
    
    def score(self, X, y):
        if self.best_estimator_ is None:
            raise ValueError("No estimator found. Please call fit() before score().")
        return self.best_estimator_.score(X, y)
