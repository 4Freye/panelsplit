# PanelSplit: a tool for panel data analysis

PanelSplit is a Python package designed to facilitate time series cross-validation when working with multiple entities (aka panel data). This tool is useful for handling panel data in various stages throughout the data pipeline, including feature engineering, hyper-parameter tuning, and model estimation.

## Features

- **Custom Time Series Cross-Validation:** Perform time series cross-validation in a panel data setting with flexible train/test splits using the same parameters as [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- **Visualization Support:** Includes features for visualizing time series splits, aiding in understanding the distribution of train and test data across periods.
- **Cross-Validated Transformation:** Enables cross-validated transformation using a given transformer, ensuring consistent preprocessing across different folds of time series data.
- **Cross-Validated Prediction:** Allows for cross-validated predictions using a given machine learning model, facilitating robust model evaluation on time series data.

## Installation

You can install PanelSplit using pip:

```bash
pip install git+https://github.com/4Freye/panelsplit.git
```

## Usage

Here's a basic example demonstrating how to use PanelSplit for time series cross-validation:

```python
# coming soon
```

## Examples

For more examples and detailed usage instructions, refer to the [examples](examples) directory in this repository. (Coming soon)

## Contributing

Contributions to PanelSplit are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This repo is a work in progress. Stay tuned!
