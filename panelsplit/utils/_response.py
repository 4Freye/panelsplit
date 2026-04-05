from sklearn.utils._response import (
    _process_decision_function,
    _process_predict_proba,
)
from sklearn.base import is_classifier, is_outlier_detector
from sklearn.utils.validation import _check_response_method
from sklearn.utils.multiclass import type_of_target
from .typing import EstimatorLike, ArrayLike
from typing import Optional, Tuple
from numpy.typing import NDArray


def _get_response_values(
    estimator: EstimatorLike,
    response_method: str,
    X: ArrayLike,
    pos_label: Optional[int] = None,
    return_response_method_used: bool = False,
) -> Tuple[Tuple[NDArray, NDArray], str]:
    if is_classifier(estimator):
        prediction_method = _check_response_method(estimator, response_method)
        classes = estimator.classes_
        target_type = type_of_target(classes)

        if target_type in ("binary", "multiclass"):
            if pos_label is not None and pos_label not in classes.tolist():
                raise ValueError(
                    f"pos_label={pos_label} is not a valid label: It should be "
                    f"one of {classes}"
                )
            elif pos_label is None and target_type == "binary":
                pos_label = classes[-1]

        test_idx, y_pred = prediction_method(X)

        if prediction_method.__name__ in ("predict_proba", "predict_log_proba"):
            y_pred = _process_predict_proba(
                y_pred=y_pred,
                target_type=target_type,
                classes=classes,
                pos_label=pos_label,
            )
        elif prediction_method.__name__ == "decision_function":
            y_pred = _process_decision_function(
                y_pred=y_pred,
                target_type=target_type,
                classes=classes,
                pos_label=pos_label,
            )
    elif is_outlier_detector(estimator):
        prediction_method = _check_response_method(estimator, response_method)
        test_idx, y_pred = prediction_method(X)

    else:  # estimator is a regressor
        if response_method != "predict":
            raise ValueError(
                f"{estimator.__class__.__name__} should either be a classifier to be "
                f"used with response_method={response_method} or the response_method "
                "should be 'predict'. Got a regressor with response_method="
                f"{response_method} instead."
            )
        prediction_method = estimator.predict
        test_idx, y_pred = prediction_method(X)

    # # Build returned 'result' as first element (either 2-tuple or 3-tuple)
    # if return_response_method_used:
    #     # return (test_idx, y_pred, pos_label) as the first element
    #     result = (test_idx, y_pred, pos_label)
    # else:
    #     result = (test_idx, y_pred)

    # Always return (result, method_name) so callers can unpack consistently
    return (test_idx, y_pred), prediction_method.__name__
