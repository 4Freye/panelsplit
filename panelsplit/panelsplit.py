from typing import List, Union
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import warnings
from .utils.validation import check_periods, check_labels, get_index_or_col_from_df

class PanelSplit:
    def __init__(
            self,
            periods:Union[pd.Series, pd.Index, np.ndarray],
            unique_periods = None,
            snapshots:pd.Series = None,
            n_splits:int = 2,
            gap:int = 0,
            test_size:int = 1,
            max_train_size:int=None) -> None:
        """A class for performing panel data cross-validation.

        Parameters
        ----------
        
        periods : A pandas series containing all periods.
        unique_periods : array-like, optional
            Array-like object containing unique periods, by default None
        snapshots : pd.Series, optional
            Series defining the snapshot for each observation, by default None
        n_splits : int, optional
            Number of splits for TimeSeriesSplit, by default 2
        gap : int, optional
            Gap between train and test sets, by default 0
        test_size : int, optional
            Size of the test set, by default 1
        max_train_size : int, optional
            Maximum size for a single training set, by default None

        Notes
        -----
        This class implements panel data specific cross-validation with time series splits.
        """

        periods = check_periods(periods)

        if unique_periods is None:
            unique_periods = pd.Series(periods.unique()).sort_values()
            
        self.tss = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size, max_train_size = max_train_size)
        indices = self.tss.split(unique_periods.reset_index(drop=True))
        self.u_periods_cv = self._split_unique_periods(indices, unique_periods)
        self.all_periods = periods
        self.snapshots = snapshots
        self.n_splits = n_splits
        self.train_test_splits = self._gen_splits()
        
    def _split_unique_periods(
            self,
            indices,
            unique_periods):
        """
        Split unique periods into train/test sets based on TimeSeriesSplit indices.

        Parameters:
        - indices: TimeSeriesSplit indices.
        - unique_periods: Pandas DataFrame or Series containing unique periods.

        Returns: 
        List of tuples containing train and test periods.
        """

        u_periods_cv = []
        for i, (train_index, test_index) in enumerate(indices):
            unique_train_periods = unique_periods.iloc[train_index].values
            unique_test_periods = unique_periods.iloc[test_index].values
            u_periods_cv.append((unique_train_periods, unique_test_periods))
        return u_periods_cv
    
    def _gen_splits(self):
        """
        Generate train/test indices based on unique periods.
        
        """
        train_test_splits = []
        
        for i, (train_periods, test_periods) in enumerate(self.u_periods_cv):
            if self.snapshots is not None:
                if test_periods.max() >= self.snapshots.min():
                    snapshot_val = test_periods.max()
                else:
                    snapshot_val = self.snapshots.min()
                    warnings.warn(
                        f'The maximum period value {test_periods.max()} for split {i} is less than the minimum snapshot value {self.snapshots.min()}. Defaulting to minimum snapshot value for split {i}.',
                        stacklevel=2
                    )
                train_indices = self.all_periods.isin(train_periods).values & (self.snapshots == snapshot_val)
                test_indices = self.all_periods.isin(test_periods).values & (self.snapshots == snapshot_val)
            else:
                train_indices = self.all_periods.isin(train_periods).values
                test_indices = self.all_periods.isin(test_periods).values 
            
            train_test_splits.append((train_indices, test_indices))
        
        return train_test_splits

    def split(self, X = None, y = None, groups=None) -> list[list[bool]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Always ignored, exists for compatibility.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        return self.train_test_splits

    def get_n_splits(self, X=None, y =None, groups=None):
        """
        Returns: Number of splits
        """
        return self.n_splits

    def _gen_labels(self, labels, fold_idx):
        """
        General function to generate labels for either train or test sets.

        Parameters:
        - labels: Pandas Series, DataFrame, Index, or NumPy array. The labels used to identify observations.
        - fold_idx: int (0 for train, 1 for test) specifying which part of the split to use.
        
        Returns:
        The labels of the corresponding fold set as the same type as the input.
        """
        check_labels(labels)
        indices = np.stack([split[fold_idx] for split in self.split()], axis=1).any(axis=1)
        
        if isinstance(labels, (pd.Series, pd.DataFrame)):
            return labels.loc[indices].copy()
        elif isinstance(labels, pd.Index):
            return labels[indices]  # Index slicing works directly
        else:
            return labels[indices].copy()
        
    def gen_train_labels(self, labels: Union[pd.Index, pd.Series, pd.DataFrame, np.ndarray]) -> Union[pd.Index, pd.Series, pd.DataFrame, np.ndarray]:  
        """Generate train labels using the provided labels."""
        return self._gen_labels(labels= labels, fold_idx=0)

    def gen_test_labels(self, labels: Union[pd.Index, pd.Series, pd.DataFrame, np.ndarray]) -> Union[pd.Index, pd.Series, pd.DataFrame, np.ndarray]:  
        """Generate test labels using the provided labels.
        
        Examples
        --------
        >>> import pandas as pd
        >>> from panelsplit import PanelSplit
        >>> periods = pd.Series(list(range(3)) * 2, name = 'period') 
        >>> print(periods)
        0    0
        1    1
        2    2
        3    0
        4    1
        5    2
        Name: period, dtype: int64

        Generation of a series of test labels:
        >>> ps = PanelSplit(periods = periods)
        >>> print(ps.gen_test_labels(periods))
        1    1
        2    2
        4    1
        5    2
        Name: period, dtype: int64

        Generation of an index of test labels:
        >>> periods = pd.Index(list(range(3)) * 2)
        >>> print(ps.gen_test_labels(periods))
        Index([1, 2, 1, 2], dtype='int64', name='period')
        """
        return self._gen_labels(labels = labels, fold_idx=1)
    
    def gen_snapshots(self, data, period_col=None):
        """
        Generate snapshots for each split.

        Parameters:
        - data: A pandas DataFrame from which snapshots are generated.
        - period_col: Optional. A str, an index or column in data from which the column snapshot_period is created. If both an index and column are in data named period_col, the default is the index.

        Returns: 
        A pandas DataFrame where each split has its own set of observations.
        """
        periods = get_index_or_col_from_df(data, period_col)

        periods = check_periods(periods)

        _data = data.copy()
        splits = self.split()
        snapshots = []
        for i, split in enumerate(splits):
            split_indices = np.array([split[0], split[1]]).any(axis = 0)
            if period_col is not None:
                last_period = periods.loc[split_indices].unique().max()
                snapshots.append(_data.loc[split_indices].assign(split = i, snapshot_period = last_period))
            else:
                snapshots.append(data.loc[split_indices].assign(split = i))
        return pd.concat(snapshots)
    
def drop_splits(cv, y):
    """Drop splits based on y if the test or train set is either empty or contains only one unique value.

    Parameters
    ----------
    {_cv_docstring}

    y : pd.Series of shape (n_samples,)

    Returns
    -------
    cv
        A list constisting of train and test indices.
    """


    for i, (train_indices, test_indices) in enumerate(cv.split):
        if ((len(train_indices) == 0 or len(test_indices) == 0) or (y.loc[train_indices].nunique() == 1 or y.loc[test_indices].nunique() == 1)):
            cv.n_splits -= 1
            cv.pop(i)
            print(f'Dropping split {i} as either the test or train set is either empty or contains only one unique value.')
        else:
            continue
    return cv