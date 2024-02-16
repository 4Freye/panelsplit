---

# PanelSplit: A Tool for Panel Data Analysis

PanelSplit is a Python package designed to facilitate time series cross-validation with custom train/test splits based on unique periods. This tool is particularly useful for handling panel data in various stages throughout the data pipeline, including feature engineering, hyper-parameter tuning, and model estimation (fitting and predicting).

## Features

- **Custom Train/Test Splits**: Perform time series cross-validation with flexible train/test splits based on unique periods.
- **Visualization**: Visualize time series splits to understand the distribution of training and testing data.
- **Imputation Support**: Perform cross-validated imputation using a specified imputer object.
- **Parallelization**: Utilize parallel processing for faster cross-validation when dealing with large datasets.

## Installation

You can install PanelSplit using pip:

```bash
pip install panelsplit
```

## Usage

Here's a basic example demonstrating how to use PanelSplit for time series cross-validation:

```python
from panelsplit import PanelSplit
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

# Define your train periods and unique periods
train_periods = pd.date_range(start='2022-01-01', end='2022-12-31', freq='M')
unique_periods = pd.Series(train_periods.unique())

# Initialize PanelSplit object
panel_split = PanelSplit(train_periods=train_periods, unique_periods=unique_periods)

# Perform cross-validated predictions
# (Example code for using cross_val_predict)

# Perform cross-validated imputation
# (Example code for using cross_val_impute)
```

## Examples

For more examples and detailed usage instructions, refer to the [examples](examples) directory in this repository.

## Contributing

Contributions to PanelSplit are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize the README further to include additional information or usage examples specific to your project.

PanelSplit is an extension of sklearn's [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), but applied to panel data.

It can be installed by running ```pip install git+https://github.com/4Freye/panelsplit.git```

After installation, the PanelSplit class can be imported by running ```from panelsplit import PanelSplit```

This repo is a work in progress. Stay tuned!
