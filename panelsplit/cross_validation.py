import warnings
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

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

if TYPE_CHECKING:
    import pandas as pd


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
        unique_periods: Optional[Union[IntoSeries, np.ndarray]] = None,
        snapshots: Optional[Union[IntoSeries, np.ndarray]] = None,
        n_splits: int = 2,
        gap: int = 0,
        test_size: int = 1,
        max_train_size: Optional[int] = None,
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
                unique_vals = np.unique(_to_numpy_array(periods))
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
        self._periods = _to_numpy_array(periods)
        self._snapshots = _to_numpy_array(snapshots) if snapshots is not None else None
        self.n_splits = n_splits
        self.train_test_splits = self._gen_splits()

    def _split_unique_periods(
        self, indices, unique_periods
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
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

    def _gen_splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate integer index arrays for training and testing sets for each split.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            A list where each tuple contains two integer arrays:
            (train_indices, test_indices) corresponding to each split.
        """
        train_test_splits = []

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
                # Create boolean masks then convert to integer indices
                train_period_mask = np.isin(self._periods, train_periods)
                test_period_mask = np.isin(self._periods, test_periods)

                # Handle snapshots
                snapshot_mask = self._snapshots == snapshot_val

                train_indices = np.where(train_period_mask & snapshot_mask)[0]
                test_indices = np.where(test_period_mask & snapshot_mask)[0]
            else:
                # Create boolean masks then convert to integer indices
                train_indices = np.where(np.isin(self._periods, train_periods))[0]
                test_indices = np.where(np.isin(self._periods, test_periods))[0]

            train_test_splits.append((train_indices, test_indices))

        return train_test_splits

    def split(
        self,
        X: Optional[IntoDataFrame] = None,
        y: Optional[Union[IntoSeries, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate training and testing indices for each cross-validation split.

        Parameters
        ----------
        X : IntoDataFrame, optional
            Ignored; included for compatibility.
        y : IntoSeries or np.ndarray, optional
            Ignored; included for compatibility.
        groups : np.ndarray, optional
            Ignored; included for compatibility.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            A list of tuples, where each tuple contains:
            (train_indices, test_indices) as integer arrays.

        Examples
        --------
        >>> period = pd.Series([1, 2, 3])
        >>> y = pd.Series([0, np.nan, 1])
        >>> ps = PanelSplit(periods=period, n_splits=2)
        >>> splits = ps.split()
        >>> for train, test in splits:
        ...     print("Train:", train, "Test:", test)
        Train: [0] Test: [1]
        Train: [0 1] Test: [2]
        >>> ps_modified = drop_splits(ps, y)
        >>> splits_modified = ps_modified.split()
        Dropping split 0 as either the test or train set is either empty or contains only one unique value.
        >>> for train, test in splits_modified:
        ...    print("Train:", train, "Test:", test)
        Train: [0 1] Test: [2]
        """
        return self.train_test_splits

    def get_n_splits(
        self,
        X: Optional[IntoDataFrame] = None,
        y: Optional[Union[IntoSeries, np.ndarray]] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """
        Get the number of splits.

        Parameters
        ----------
        X : IntoDataFrame, optional
            Ignored; included for compatibility.
        y : IntoSeries or np.ndarray, optional
            Ignored; included for compatibility.
        groups : np.ndarray, optional
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

    def _gen_labels(
        self,
        labels: Union["pd.Index", IntoSeries, IntoDataFrame, np.ndarray],
        fold_idx: int,
    ) -> Union["pd.Index", IntoSeries, IntoDataFrame, np.ndarray]:
        """
        Generate labels for either the training or testing set based on the cross-validation splits.

        Parameters
        ----------
        labels : pd.Index, IntoSeries, IntoDataFrame, or np.ndarray
            The labels corresponding to the observations.
        fold_idx : int
            Indicator for the fold to generate labels for (0 for training, 1 for testing).

        Returns
        -------
        pd.Index, IntoSeries, IntoDataFrame, or np.ndarray
            The labels corresponding to the specified fold.
        """
        check_labels(labels)
        # Collect all indices from all splits for the specified fold
        all_indices = np.concatenate([split[fold_idx] for split in self.split()])
        # Get unique indices to avoid duplicates
        row_indices = np.unique(all_indices)

        # Use narwhals slice operation for clean dataframe-agnostic subsetting
        labels_nw = nw.from_native(labels, pass_through=True)

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

    def gen_snapshots(
        self, data: IntoDataFrame, period_col: Optional[str] = None
    ) -> IntoDataFrame:
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
            # Combine train and test indices (both are already integer arrays)
            row_indices = np.unique(np.concatenate([split[0], split[1]]))

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
