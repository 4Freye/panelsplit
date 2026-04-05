import matplotlib.pyplot as plt
from .cross_validation import PanelSplit
from typing import Tuple, Optional


def plot_splits(
    panel_split: PanelSplit, show: bool = True
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Visualize time series cross-validation splits using a scatter plot.

    Each split is plotted on a separate horizontal line: blue markers represent training indices
    and red markers represent test indices.

    Parameters
    ----------
    panel_split : PanelSplit
        An instance of PanelSplit containing the cross-validation splits.
        It must have an attribute `_u_periods_cv`, which should be an iterable of tuples,
        each in the form `(train_index, test_index)`. Both `train_index` and `test_index`
        are array-like collections of period indices.
    show : bool, default=True
        If True, the plot is immediately displayed using `plt.show()`.
        If False, the function returns the matplotlib Figure and Axes objects for further customization.

    Returns
    -------
    Optional[Tuple[plt.Figure, plt.Axes]]
        If `show` is False, returns a tuple `(fig, ax)` where `fig` is the matplotlib Figure
        and `ax` is the Axes object. If `show` is True, the plot is displayed and the function returns None.

    Examples
    --------
    >>> from panelsplit.cross_validation import PanelSplit
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> periods = np.array([1, 2, 3, 4, 5, 6])
    >>> ps = PanelSplit(periods, n_splits=3)
    >>> # Display the plot immediately
    >>> plot_splits(ps)
    >>> # Or return the Figure and Axes for customization
    >>> fig, ax = plot_splits(ps, show=False)
    >>> ax.set_title("A custom plot of cross-validation splits")
    >>> plt.show()
    """
    split_output = panel_split._u_periods_cv
    splits = len(split_output)
    fig, ax = plt.subplots()

    for i, (train_index, test_index) in enumerate(split_output):
        ax.scatter(train_index, [i] * len(train_index), color="blue", marker=".", s=50)
        ax.scatter(test_index, [i] * len(test_index), color="red", marker=".", s=50)

    ax.set_xlabel("Periods")
    ax.set_ylabel("Split")
    ax.set_title("Cross-validation splits")
    ax.set_yticks(range(splits))  # Set the number of ticks on the y-axis
    ax.set_yticklabels(
        [f"{i}" for i in range(splits)]
    )  # Set custom labels for the y-axis

    if show:
        plt.show()
        return None
    else:
        return fig, ax
