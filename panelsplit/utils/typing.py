from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from narwhals.typing import IntoDataFrame, IntoSeries
from typing import Union, List, Tuple
import numpy as np


ArrayLike = Union[IntoDataFrame, IntoSeries, NDArray]

CVIndices = List[Tuple[NDArray[np.int64], NDArray[np.int64]]]

EstimatorLike = BaseEstimator
