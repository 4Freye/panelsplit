# PanelSplit: a tool for panel data analysis

PanelSplit is a Python package designed to facilitate time series cross-validation when working with multiple entities (aka panel data). This tool is useful for handling panel data in various stages throughout the data pipeline, including feature engineering, hyper-parameter tuning, and model estimation.

## Features

- **Custom Train/Test Splits**: Perform time series cross-validation in a panel data setting with flexible train/test splits using the same parameters as [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- **Visualization**: Visualize time series splits to understand the distribution of training and testing data.
- **Imputation Support**: Perform cross-validated imputation using a specified imputer object.
- **Parallelization**: Utilize parallel processing for faster cross-validation when dealing with large datasets.

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
