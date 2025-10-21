import warnings
from typing import List, Union

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.model_selection import TimeSeriesSplit

from .utils.validation import (
    _safe_indexing,
    _to_numpy_array,
    check_labels,
    check_periods,
    get_index_or_col_from_df,
)

# Keep pandas import for type annotations and fallback compatibility
try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    _PANDAS_AVAILABLE = False


def _nunique_subset(data, indices):
    """
    Helper function to get number of unique values in a subset of data.
    Works with both pandas and narwhals-compatible data.
    """
    data_nw = nw.from_native(data, pass_through=True)
    if hasattr(data_nw, "filter") and hasattr(data_nw, "n_unique"):
        return data_nw.filter(indices).n_unique()
    else:
        # Fallback to numpy operations
        subset_data = (
            data[indices] if hasattr(data, "__getitem__") else data.loc[indices]
        )
        return len(np.unique(subset_data))


class PanelSplit:
    """
    PanelSplit implements panel data cross-validation using time series splits.

    Parameters
    ----------
    periods : IntoSeries, pd.Index, or np.ndarray
        A collection of periods for each observation.
    unique_periods : array-like, optional
        An array-like object containing unique periods. If None, the unique periods will be
        computed from `periods`. Default is None.
    snapshots : IntoSeries, optional
        A Series defining the snapshot for each observation. Default is None.
    n_splits : int, optional
        Number of splits for TimeSeriesSplit. Default is 2.
    gap : int, optional
        Gap between the training and testing sets. Default is 0.
    test_size : int, optional
        Size of the test set. Default is 1.
    max_train_size : int, optional
        Maximum size for a single training set. Default is None.
    include_first_train_in_test : bool, optional
        Whether to include the first split's training set in the test set. Useful in the context of transforming data. Default is False.
    include_train_in_test : bool, optional
        Whether to include the all splits' training sets in their respective test sets. Useful in the context of transforming data that has snapshots. Default is False. If set to True, overrides include_first_train_in_test.

    Attributes
    ----------
    n_splits : int
        The number of splits for cross-validation.
    train_test_splits : list of tuples
        A list of train/test splits for the panel data. Each tuple has the form
        (train_indices, test_indices), representing the indices for the training and testing sets
        for that split.
        for that split.

    Notes
    -----
    This class is designed for panel data where cross-validation splits must respect
    temporal ordering.
    """

    def __init__(
        self,
        periods: Union[IntoSeries, np.ndarray],
        unique_periods=None,
        snapshots: IntoSeries = None,
        n_splits: int = 2,
        gap: int = 0,
        test_size: int = 1,
        max_train_size: int = None,
        include_first_train_in_test: bool = False,
        include_train_in_test: bool = False,
    ) -> None:
        periods = check_periods(periods)

        if unique_periods is None:
            periods_nw = nw.from_native(periods, pass_through=True)
            if hasattr(periods_nw, "unique"):
                unique_result = periods_nw.unique()
                unique_array = _to_numpy_array(unique_result)
                unique_vals = np.sort(unique_array)
            else:
                unique_vals = np.unique(periods)
            unique_periods = unique_vals
        else:
            unique_periods = check_periods(unique_periods, obj_name="unique_periods")

        self._tss = TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap,
            test_size=test_size,
            max_train_size=max_train_size,
        )
        # Convert to numpy array for TimeSeriesSplit
        unique_periods_array = _to_numpy_array(unique_periods)
        indices = self._tss.split(unique_periods_array)
        self._include_train_in_test = include_train_in_test
        if not self._include_train_in_test:
            self._include_first_train_in_test = include_first_train_in_test
        else:
            self._include_first_train_in_test = True
        self._u_periods_cv = self._split_unique_periods(indices, unique_periods)
        self._periods = periods
        self._snapshots = snapshots
        self.n_splits = n_splits
        self.train_test_splits = self._gen_splits()

    def _split_unique_periods(self, indices, unique_periods):
        """
        Split unique periods into training and testing sets based on TimeSeriesSplit indices.

        Parameters
        ----------
        indices : iterator
            An iterator yielding pairs of train and test indices from TimeSeriesSplit.
        unique_periods : array-like
            The collection of unique periods.

        Returns
        -------
        list of tuple of np.ndarray
            A list where each tuple contains two numpy arrays:
            (unique_train_periods, unique_test_periods) for each split.
        """
        unique_periods_array = _to_numpy_array(unique_periods)
        u_periods_cv = []
        for i, (train_index, test_index) in enumerate(indices):
            unique_train_periods = unique_periods_array[train_index]
            unique_test_periods = unique_periods_array[test_index]
            if (i == 0) & self._include_first_train_in_test:
                unique_test_periods = np.concatenate(
                    [unique_train_periods, unique_test_periods]
                )
            elif (i > 0) & self._include_train_in_test:
                unique_test_periods = np.concatenate(
                    [unique_train_periods, unique_test_periods]
                )
            u_periods_cv.append((unique_train_periods, unique_test_periods))
        return u_periods_cv

    def _gen_splits(self):
        """
        Generate boolean index arrays for training and testing sets for each split.

        Returns
        -------
        list of tuple of np.ndarray
            A list where each tuple contains two boolean arrays:
            (train_indices, test_indices) corresponding to each split.
        """
        train_test_splits = []

        # Convert snapshots once if they exist
        snapshots_array = (
            _to_numpy_array(self._snapshots) if self._snapshots is not None else None
        )

        for i, (train_periods, test_periods) in enumerate(self._u_periods_cv):
            if self._snapshots is not None:
                if test_periods.max() >= self._snapshots.min():
                    snapshot_val = test_periods.max()
                else:
                    snapshot_val = self._snapshots.min()
                    warnings.warn(
                        (
                            f"The maximum period value {test_periods.max()} for split {i} is less than "
                            f"the minimum snapshot value {self._snapshots.min()}. Defaulting to minimum "
                            f"snapshot value for split {i}."
                        ),
                        stacklevel=2,
                    )
                # Use numpy operations for boolean masking - simpler and more reliable
                train_period_mask = np.isin(self._periods, train_periods)
                test_period_mask = np.isin(self._periods, test_periods)

                # Handle snapshots
                snapshot_mask = snapshots_array == snapshot_val

                train_indices = train_period_mask & snapshot_mask
                test_indices = test_period_mask & snapshot_mask
            else:
                # Use numpy operations for boolean masking - simpler and more reliable
                train_indices = np.isin(self._periods, train_periods)
                test_indices = np.isin(self._periods, test_periods)

            train_test_splits.append((train_indices, test_indices))

        return train_test_splits

    def split(self, X=None, y=None, groups=None) -> List[List[bool]]:
        """
        Generate training and testing indices for each cross-validation split.

        Parameters
        ----------
        X : object, optional
            Ignored; included for compatibility.
        y : object, optional
            Ignored; included for compatibility.
        groups : object, optional
            Ignored; included for compatibility.

        Returns
        -------
        list of tuple of np.ndarray
            A list of tuples, where each tuple contains:
            (train_indices, test_indices) as boolean arrays.

        Examples
        --------
        >>> period = pd.Series([1, 2, 3])
        >>> y = pd.Series([0, np.nan, 1])
        >>> ps = PanelSplit(periods=period, n_splits=2)
        >>> splits = ps.split()
        >>> for train, test in splits:
        ...     print("Train:", train, "Test:", test)
        Train: [ True False False] Test: [False  True False]
        Train: [ True  True False] Test: [False False  True]
        >>> ps_modified = drop_splits(ps, y)
        >>> splits_modified = ps_modified.split()
        Dropping split 0 as either the test or train set is either empty or contains only one unique value.
        >>> for train, test in splits_modified:
        ...    print("Train:", train, "Test:", test)
        Train: [ True  True False] Test: [False False  True]
        """
        return self.train_test_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Get the number of splits.

        Parameters
        ----------
        X : object, optional
            Ignored; included for compatibility.
        y : object, optional
            Ignored; included for compatibility.
        groups : object, optional
            Ignored; included for compatibility.

        Returns
        -------
        int
            The number of splits.

        Examples
        --------
        >>> import pandas as pd
        >>> periods = pd.Series([1, 2, 3, 4, 5, 6])
        >>> ps = PanelSplit(periods=periods, n_splits=3)
        >>> ps.get_n_splits()
        3
        """
        return self.n_splits

    def _gen_labels(self, labels, fold_idx):
        """
        Generate labels for either the training or testing set based on the cross-validation splits.

        Parameters
        ----------
        labels : IntoSeries, IntoDataFrame, pd.Index, or np.ndarray
            The labels corresponding to the observations.
        fold_idx : int
            Indicator for the fold to generate labels for (0 for training, 1 for testing).

        Returns
        -------
        Same type as `labels`
            The labels corresponding to the specified fold.
        """
        check_labels(labels)
        indices = np.stack([split[fold_idx] for split in self.split()], axis=1).any(
            axis=1
        )

        # Use narwhals slice operation for clean dataframe-agnostic subsetting
        labels_nw = nw.from_native(labels, pass_through=True)
        # Convert boolean indices to row numbers
        row_indices = np.where(indices)[0]

        return _safe_indexing(labels_nw, row_indices, to_native=True)

    def gen_train_labels(
        self, labels: Union["pd.Index", IntoSeries, IntoDataFrame, np.ndarray]
    ) -> Union["pd.Index", IntoSeries, IntoDataFrame, np.ndarray]:
        """
        Generate training set labels based on the provided labels.

        Parameters
        ----------
        labels : pd.Index, IntoSeries, IntoDataFrame, or np.ndarray
            The labels corresponding to the observations.

        Returns
        -------
        Same type as `labels`
            The labels for the training set.

        Examples
        --------
        >>> import pandas as pd
        >>> periods = pd.Series([1, 2, 3])
        >>> labels = np.array(['a', 'b', 'c'])
        >>> ps = PanelSplit(periods=periods, n_splits=2)
        >>> train_labels = ps.gen_train_labels(labels)
        >>> train_labels
        array(['a', 'b'], dtype='<U1')
        """
        return self._gen_labels(labels=labels, fold_idx=0)

    def gen_test_labels(
        self, labels: Union["pd.Index", IntoSeries, IntoDataFrame, np.ndarray]
    ) -> Union["pd.Index", IntoSeries, IntoDataFrame, np.ndarray]:
        """
        Generate testing set labels based on the provided labels.

        Parameters
        ----------
        labels : pd.Index, IntoSeries, IntoDataFrame, or np.ndarray
            The labels corresponding to the observations.

        Returns
        -------
        Same type as `labels`
            The labels for the testing set.

        Examples
        --------
        >>> import pandas as pd
        >>> periods = pd.Series([1, 2, 3])
        >>> labels = np.array(['a', 'b', 'c'])
        >>> ps = PanelSplit(periods=periods, n_splits=2)
        >>> test_labels = ps.gen_test_labels(labels)
        >>> test_labels
        array(['b', 'c'], dtype='<U1')
        """
        return self._gen_labels(labels=labels, fold_idx=1)

    def gen_snapshots(self, data, period_col=None):
        """
        Generate snapshots for each cross-validation split.

        Parameters
        ----------
        data : IntoDataFrame
            The DataFrame from which snapshots are generated.
        period_col : str, optional
            The column name (or index) in `data` to derive the snapshot period. If both an index
            and a column in `data` are named `period_col`, the index is used by default.

        Returns
        -------
        IntoDataFrame
            A concatenated DataFrame containing snapshots for each split. Each snapshot includes
            a 'split' column (indicating the split number) and, if `period_col` is provided, a
            'snapshot_period' column.

        Examples
        --------
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ... 'value': [10, pd.NA, 30],
        ... 'period': [1, 2, 3],
        ... }).astype('Int32')
        >>> ps = PanelSplit(periods=data['period'], n_splits=2)
        >>> snapshots = ps.gen_snapshots(data, period_col='period')
        >>> print(snapshots)
           value  period  split  snapshot_period
        0     10       1      0                2
        1   <NA>       2      0                2
        0     10       1      1                3
        1   <NA>       2      1                3
        2     30       3      1                3
        """
        periods = get_index_or_col_from_df(data, period_col)
        periods = check_periods(periods)

        # Use narwhals for dataframe-agnostic operations
        data_nw = nw.from_native(data, pass_through=True)
        periods_nw = nw.from_native(periods, pass_through=True)

        splits = self.split()
        snapshots = []
        for i, split in enumerate(splits):
            split_indices = np.array([split[0], split[1]]).any(axis=0)

            # Convert boolean indices to row numbers for narwhals indexing
            row_indices = np.where(split_indices)[0]

            split_data = _safe_indexing(data_nw, row_indices)

            if period_col is not None:
                # Get periods for this split and find max
                split_periods = _safe_indexing(periods_nw, row_indices, to_native=True)
                last_period = np.max(np.unique(split_periods))

                # Add columns using narwhals
                split_data = split_data.with_columns(
                    [
                        nw.lit(i).alias("split"),
                        nw.lit(last_period).alias("snapshot_period"),
                    ]
                )
            else:
                split_data = split_data.with_columns([nw.lit(i).alias("split")])

            snapshots.append(split_data)

        return _safe_indexing(nw.concat(snapshots), to_native=True)


def drop_splits(cv, y):
    """
    Drop cross-validation splits if either the training or testing set is empty or contains only one unique value.

    Parameters
    ----------
    cv : list
        A list of tuples, where each tuple contains (train_indices, test_indices) as boolean arrays.
        The object is expected to have an attribute `n_splits` (an integer) and support the `pop` method.
    y : IntoSeries
        Series of shape (n_samples,) containing target values.

    Returns
    -------
    list
        The modified list of splits with the problematic splits removed.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ... 'value': [10, pd.NA, 30],
    ... 'period': [1, 2, 3],
    ... }).astype('Int32')
    >>> ps = PanelSplit(periods=periods, n_splits=2)
    >>> drop_splits(ps, data['value'])
    Dropping split 0 as either the test or train set is either empty or contains only one unique value.
    """
    for i, (train_indices, test_indices) in enumerate(cv.split()):
        if (len(train_indices) == 0 or len(test_indices) == 0) or (
            _nunique_subset(y, train_indices) == 1
            or _nunique_subset(y, test_indices) == 1
        ):
            cv.n_splits -= 1
            cv.train_test_splits.pop(i)
            print(
                f"Dropping split {i} as either the test or train set is either empty or contains only one unique value."
            )
    return cv
