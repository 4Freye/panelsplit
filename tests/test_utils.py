import pytest
from panelsplit.utils.validation import _supports_sample_weights
from sklearn.base import BaseEstimator
import numpy as np


class NoSampleWeightEstimator(BaseEstimator):
    def fit(self, X, y):  # No sample_weight parameter
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.ones(len(X))


def test_warning_estimator_lacks_sample_weights():
    """Test that a warning is raised when estimator does not support sample_weight."""
    with pytest.warns(UserWarning, match="does not support sample_weight"):
        result = _supports_sample_weights(
            NoSampleWeightEstimator, sample_weight=[1, 2, 3]
        )
        assert result is False
