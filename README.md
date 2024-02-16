# PanelSplit: a tool for panel data analysis

PanelSplit is a Python package designed to facilitate time series cross-validation when working with multiple entities (aka panel data). This tool is useful for handling panel data in various stages throughout the data pipeline, including feature engineering, hyper-parameter tuning, and model estimation.

## Installation

You can install PanelSplit using pip:

```bash
pip install git+https://github.com/4Freye/panelsplit.git
```
---

## PanelSplit Documentation

### Initialization Parameters
- **train_periods**: All available training periods.
- **unique_periods**: Pandas DataFrame or Series containing unique periods. Default is `None`, in which case unique periods are derived from `train_periods`.
- **n_splits**: Number of splits for the underlying `TimeSeriesSplit`.
- **gap**: Gap between train and test sets in the `TimeSeriesSplit`.
- **test_size**: Size of the test set in the `TimeSeriesSplit`.
- **max_train_size**: Maximum size for a single training set.
- **plot**: Flag to visualize time series splits. Default is `False`.
- **drop_folds**: Flag to drop folds with either empty or single unique values in train or test sets.
- **y**: Target variable. Required if `drop_folds` is set to `True`.

### Methods

#### `split_unique_periods(indices, unique_periods)`
Split unique periods into train/test sets based on `TimeSeriesSplit` indices.

##### Parameters
- **indices**: TimeSeriesSplit indices.
- **unique_periods**: Pandas DataFrame or Series containing unique periods.

##### Returns
List of tuples containing train and test periods.

#### `split(X=None, y=None, groups=None, init=False)`
Generate train/test indices based on unique periods.

##### Parameters
- **X**: Features.
- **y**: Target variable.
- **groups**: Group labels for the samples.
- **init**: Flag indicating initialization phase.

##### Returns
List of train/test indices.

#### `get_n_splits(X=None, y=None, groups=None)`
Returns the number of splits.

##### Parameters
- **X**: Features.
- **y**: Target variable.
- **groups**: Group labels for the samples.

##### Returns
Number of splits.

#### `_predict_fold(estimator, X_train, y_train, X_test, prediction_method='predict', sample_weight=None)`
Perform predictions for a single fold.

##### Parameters
- **estimator**: Machine learning model.
- **X_train**: Features for training.
- **y_train**: Target variable for training.
- **X_test**: Features for testing.
- **prediction_method**: Prediction method. Default is `'predict'`.
- **sample_weight**: Sample weights for the training data.

##### Returns
Series containing predicted values.

#### `cross_val_predict(estimator, X, y, indices, prediction_method='predict', y_pred_col=None, return_fitted_models=False, sample_weight=None)`
Perform cross-validated predictions using a given predictor model.

##### Parameters
- **estimator**: Machine learning model.
- **X**: Features.
- **y**: Target variable.
- **indices**: Indices corresponding to the dataset.
- **prediction_method**: Prediction method. Default is `'predict'`.
- **y_pred_col**: Column name for the predicted values.
- **return_fitted_models**: Whether to return fitted models. Default is `False`.
- **sample_weight**: Sample weights for the training data.

##### Returns
Concatenated DataFrame containing predictions made by the model during cross-validation.

#### `plot_time_series_splits(split_output)`
Visualize time series splits using a scatter plot.

##### Parameters
- **split_output**: Output of time series splits.

#### `cross_val_transform(transformer, X, return_fitted_transformers=False, include_test_in_fit=False)`
Perform cross-validated transformation using a given transformer.

##### Parameters
- **transformer**: Transformer object.
- **X**: Features.
- **return_fitted_transformers**: Whether to return fitted transformers. Default is `False`.
- **include_test_in_fit**: Whether to include test data in fitting. Default is `False`.

##### Returns
DataFrame containing transformed values during cross-validation.

---

## Examples

For more examples and detailed usage instructions, refer to the [examples](examples) directory in this repository. (Coming soon)

## Contributing

Contributions to PanelSplit are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This repo is a work in progress. Stay tuned!
