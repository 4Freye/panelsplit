# import numpy as np
# from sklearn.base import BaseEstimator
# from sklearn.utils.validation import check_is_fitted
# from panelsplit.model_selection import BaseSearchSequentialCV

# # --- Dummy Estimators for Testing ---

# class DummyEstimator(BaseEstimator):
#     """
#     A simple estimator with a single hyperparameter 'value'.
#     It simply "predicts" an array of constant 'value',
#     and its score method returns the value.
#     """
#     def __init__(self, value=0):
#         self.value = value

#     def fit(self, X, y=None):
#         # No actual fitting; just return self.
#         return self

#     def predict(self, X):
#         return np.full((len(X),), self.value)

#     def score(self, X, y):
#         # For testing, use the parameter value as the "score"
#         return self.value


# class FailingEstimator(BaseEstimator):
#     """
#     Similar to DummyEstimator, but raises an exception in fit
#     if value == -1.
#     """
#     def __init__(self, value=0):
#         self.value = value

#     def fit(self, X, y=None):
#         if self.value == -1:
#             raise ValueError("Deliberate failure for value == -1")
#         return self

#     def predict(self, X):
#         return np.full((len(X),), self.value)

#     def score(self, X, y):
#         return self.value

# # --- Test Functions ---

# def test_search_success():
#     """
#     Test BaseSearchSequentialCV using DummyEstimator.
#     We provide a parameter grid where the candidate with the highest
#     'value' should be chosen.
#     """
#     # Create a small toy dataset.
#     X = np.array([[1], [2], [3], [4]])
#     y = np.array([10, 10, 10, 10])
    
#     # Parameter grid: higher "value" gives a higher score.
#     param_grid = {"value": [100, 10, 5000]}
    
#     # Create a DummyEstimator instance.
#     dummy = DummyEstimator()
    
#     # Create the search object. Here we use the estimator's own score method.
#     search = BaseSearchSequentialCV(
#         estimator=dummy, 
#         param_grid=param_grid, 
#         scoring="neg_mean_absolute_error", 
#         verbose=0
#     )
    
#     # Fit the search object.
#     search.fit(X, y)
    
#     # Check that the best parameters are chosen as the candidate with value 10.
#     assert search.best_params_ == {"value": 10}, f"Expected best_params {{'value': 10}}, got {search.best_params_}"
#     assert search.best_score_ == 0, f"Expected best_score 0, got {search.best_score_}"
    
#     # Test predict: All predictions should equal 10.
#     y_pred = search.predict(X)
#     expected = np.full((len(X),), 10)
#     assert np.allclose(y_pred, expected), "Predict did not return the expected constant array."
    
#     print("test_search_success passed.")


# def test_search_failure():
#     """
#     Test BaseSearchSequentialCV with one candidate that fails.
#     In our parameter grid, the candidate with value=-1 (which triggers failure)
#     should yield the error_score, and the best candidate should be selected among the rest.
#     """
#     X = np.array([[1], [2], [3], [4]])
#     y = np.array([10, 10, 10, 10])
    
#     # Parameter grid with one candidate that will fail.
#     param_grid = {"value": [1, -1, 10]}
    
#     # Create a FailingEstimator instance.
#     dummy = FailingEstimator()
    
#     # Set error_score to a low value (e.g. -100) so that the failing candidate is not chosen.
#     search = BaseSearchSequentialCV(
#         estimator=dummy, 
#         param_grid=param_grid, 
#         scoring="neg_mean_absolute_error", 
#         error_score=-100, 
#         verbose=0
#     )
    
#     search.fit(X, y)
    
#     # Expected: Candidate with value=-1 should yield -100,
#     # and the best candidate among [1, -100, 5] is 5.
#     assert search.best_params_ == {"value": 10}, f"Expected best_params {{'value': 10}}, got {search.best_params_}"
#     assert search.best_score_ == 0, f"Expected best_score 0, got {search.best_score_}"
    
#     print("test_search_failure passed.")


# def test_predict_and_score():
#     """
#     Test that the predict() and score() methods of BaseSearchSequentialCV delegate
#     to the best_estimator_ correctly.
#     """
#     X = np.array([[10], [20], [30]])
#     y = np.array([10, 10, 10])
    
#     param_grid = {"value": [2, 10, 3]}
#     dummy = DummyEstimator()
    
#     search = BaseSearchSequentialCV(
#         estimator=dummy,
#         param_grid=param_grid,
#         scoring="neg_mean_absolute_error",
#         verbose=0
#     )
#     search.fit(X, y)
    
#     y_pred = search.predict(X)
#     expected = np.full((len(X),), search.best_params_["value"])
#     assert np.allclose(y_pred, expected), "predict() did not return expected output."
    
#     score = search.score(X, y)
#     assert score == search.best_params_["value"], "score() did not return expected best score."
    
#     print("test_predict_and_score passed.")


# # if __name__ == "__main__":
# #     test_search_success()
# #     test_search_failure()
# #     test_predict_and_score()
