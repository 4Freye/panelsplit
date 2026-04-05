from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .cross_validation import PanelSplit


def plot_splits(
    panel_split: PanelSplit, n_groups: int = 2, show: bool = True
) -> Optional[Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]]:
    """
    Visualize time series cross-validation splits using a scatter plot.

    Each split is plotted on a separate horizontal line: blue markers represent training indices
    and red markers represent test indices.

    If the PanelSplit instance uses groups for spatio-temporal holdouts, this will create
    `n_groups` subplots, each visualizing the train/test periods for an individual, randomly
    sampled or sequential subset of the unique groups.

    Parameters
    ----------
    panel_split : PanelSplit
        An instance of PanelSplit containing the cross-validation splits.
    n_groups : int, default=2
        The number of subgroups to plot side-by-side if groups are used.
    show : bool, default=True
        If True, the plot is immediately displayed using `plt.show()`.
        If False, the function returns the matplotlib Figure and Axes objects.

    Returns
    -------
    Optional[Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]]
        If `show` is False, returns a tuple `(fig, ax)` where `fig` is the matplotlib Figure
        and `ax` is the Axes object (or an array of Axes objects if groups are used).
        If `show` is True, the plot is displayed and the function returns None.
    """

    if panel_split._groups is not None:
        return _plot_group_subplots(panel_split, n_groups=n_groups, show=show)

    split_output = panel_split._u_periods_cv
    splits = len(split_output)
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (train_index, test_index) in enumerate(split_output):
        ax.scatter(train_index, [i] * len(train_index), color="blue", marker=".", s=50)
        ax.scatter(test_index, [i] * len(test_index), color="red", marker=".", s=50)

    ax.set_xlabel("Periods")
    ax.set_ylabel("Split")
    ax.set_title("Cross-validation splits")
    ax.set_yticks(range(splits))
    ax.set_yticklabels([f"{i}" for i in range(splits)])

    if show:
        plt.show()
        return None
    else:
        return fig, ax


def _plot_group_subplots(panel_split: PanelSplit, n_groups: int, show: bool):
    unique_groups = np.unique(np.asarray(panel_split._groups))
    selected_groups = unique_groups[:n_groups]
    actual_groups = len(selected_groups)

    fig, axes = plt.subplots(
        ncols=actual_groups,
        sharey=True,
        sharex=True,
        figsize=(min(16, 6 * actual_groups), 6),
    )

    if actual_groups == 1:
        axes = [axes]

    splits = panel_split.split()
    n_total_splits = len(splits)
    n_temporal_splits = (
        len(panel_split._temporal_splits)
        if hasattr(panel_split, "_temporal_splits")
        else n_total_splits
    )
    n_spatial_splits = (
        n_total_splits // n_temporal_splits if n_temporal_splits > 0 else 1
    )

    for ax_idx, group in enumerate(selected_groups):
        ax = axes[ax_idx]
        group_mask = panel_split._groups == group

        for i, (train_indices, test_indices) in enumerate(splits):
            group_train_indices = np.intersect1d(train_indices, np.where(group_mask)[0])
            group_test_indices = np.intersect1d(test_indices, np.where(group_mask)[0])

            train_periods = panel_split._periods[group_train_indices]
            test_periods = panel_split._periods[group_test_indices]

            temporal_idx = i // n_spatial_splits

            if len(train_periods) > 0:
                ax.scatter(
                    train_periods,
                    [temporal_idx] * len(train_periods),
                    color="blue",
                    marker=".",
                    s=50,
                )
            if len(test_periods) > 0:
                ax.scatter(
                    test_periods,
                    [temporal_idx] * len(test_periods),
                    color="red",
                    marker=".",
                    s=50,
                )

        ax.set_xlabel("Periods")
        if ax_idx == 0:
            ax.set_ylabel("Split")
        ax.set_title(f"Cross-validation splits: {group}")
        ax.set_yticks(range(n_temporal_splits))
        ax.set_yticklabels([f"{j}" for j in range(n_temporal_splits)])

    total_groups = len(unique_groups)
    remaining_groups = total_groups - actual_groups
    if remaining_groups > 0:
        fig.suptitle(
            f"Cross-validation splits: {remaining_groups} other groups not plotted"
        )

    plt.tight_layout()
    if show:
        plt.show()
        return None
    else:
        return fig, (axes if actual_groups > 1 else axes[0])
