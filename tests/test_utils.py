import pytest
from sklearn.ensemble import StackingClassifier
from panelsplit.utils.validation import _supports_sample_weights


def test_warning_estimator_lacks_sample_weights():
    """Test that a warning is raised when estimator does not support sample_weight."""
    with pytest.warns(UserWarning, match="does not support sample_weight"):
        result = _supports_sample_weights(StackingClassifier, sample_weight=[1, 2, 3])
        assert result is False
