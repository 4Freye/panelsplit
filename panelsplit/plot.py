import matplotlib.pyplot as plt

def plot_splits(PanelSplit, show=True):
    """
    Visualize time series cross-validation splits using a scatter plot.

    This function generates a scatter plot that displays training and test indices 
    for each cross-validation split contained in a PanelSplit instance. Each split 
    is plotted on a separate horizontal line: blue markers represent training indices 
    and red markers represent test indices.

    Parameters:
        PanelSplit (object): An instance of PanelSplit containing the cross-validation 
            splits. It must have an attribute `u_periods_cv`, which should be an iterable 
            of tuples, each in the form (train_index, test_index). Here, both `train_index` 
            and `test_index` are array-like collections of period indices.
        show (bool, optional): If True (default), the plot is immediately displayed 
            using `plt.show()`. If False, the function returns the matplotlib Figure and Axes 
            objects for further customization.

    Returns:
        tuple or None:
            - If show is False, returns a tuple (fig, ax) where fig is the matplotlib Figure 
              and ax is the Axes object.
            - If show is True, the plot is displayed and the function returns None.

    Example:
        >>> from panelsplit.cross_validation import PanelSplit  
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> periods = np.array([1,2,3,4,5,6])
        >>> # Create a PanelSplit instance with cross-validation splits
        >>> ps = PanelSplit(periods, n_splits = 3)  
        >>> # To display the plot immediately:
        >>> plot_splits(ps)
        >>> # Or, to further customize the plot before displaying:
        >>> fig, ax = plot_splits(ps, show=False)
        >>> ax.set_title("A custom plot of cross-validation splits")
        >>> plt.show()
    """
    split_output = PanelSplit._u_periods_cv
    splits = len(split_output)
    fig, ax = plt.subplots()
    
    for i, (train_index, test_index) in enumerate(split_output):
        ax.scatter(train_index, [i] * len(train_index), color='blue', marker='.', s=50)
        ax.scatter(test_index, [i] * len(test_index), color='red', marker='.', s=50)

    ax.set_xlabel('Periods')
    ax.set_ylabel('Split')
    ax.set_title('Cross-validation splits')
    ax.set_yticks(range(splits))  # Set the number of ticks on the y-axis
    ax.set_yticklabels([f'{i}' for i in range(splits)])  # Set custom labels for the y-axis
    
    if show:
        plt.show()
    else:
        return fig, ax
