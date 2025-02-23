import matplotlib.pyplot as plt

def plot_splits(PanelSplit, show=True):
    """
    Visualize time series splits using a scatter plot.

    Parameters:
    - PanelSplit: Instance of PanelSplit.
    """
    split_output = PanelSplit.u_periods_cv
    splits = len(split_output)
    fig, ax = plt.subplots()
    
    for i, (train_index, test_index) in enumerate(split_output):
        ax.scatter(train_index, [i] * len(train_index), color='blue', marker='.', s=50)
        ax.scatter(test_index, [i] * len(test_index), color='red', marker='.', s=50)

    ax.set_xlabel('Periods')
    ax.set_ylabel('Split')
    ax.set_title('Cross-validation splits')
    ax.set_yticks(range(splits))  # Set the number of ticks on y-axis
    ax.set_yticklabels([f'{i}' for i in range(splits)])  # Set custom labels for y-axi
    if show:
        plt.show()