[![PyPI version](https://badge.fury.io/py/panelsplit.svg)](https://badge.fury.io/py/panelsplit) [![DOI](https://zenodo.org/badge/742040227.svg)](https://zenodo.org/doi/10.5281/zenodo.10777259)

# PanelSplit: a tool for panel data analysis

PanelSplit is a Python package designed to facilitate time series cross-validation when working with multiple entities (aka panel data). This tool is useful for handling panel data in various stages throughout the data pipeline, including feature engineering, hyper-parameter tuning, and model estimation.

## Installation

You can install PanelSplit using pip:

```bash
pip install panelsplit
```
---

## Documentation

### Initialization Parameters
- **periods**: *Pandas Series*. Represents the time series of the DataFrame.
- **unique_periods**: *Pandas Series*. Contains unique periods. Default is `None`, in which case unique periods are derived from `periods` and then sorted.
- **snapshots**: *Pandas Series, default=None*. Defines the snapshot for the observation, i.e. when the observation was updated.
- **n_splits**: *int, default=5*. Number of splits for the underlying `TimeSeriesSplit`.
- **gap**: *int, default=0*. Gap between train and test sets in `TimeSeriesSplit`.
- **test_size**: *int, default=1*. Size of the test set in `TimeSeriesSplit`.
- **max_train_size**: *int, default=None*. Maximum size for a single training set in `TimeSeriesSplit`.
- **plot**: *bool, default=False*. Flag to visualize time series splits.
- **drop_splits**: *bool, default=False*. Flag to drop splits with either empty or single unique values in train or test sets.
- **y**: *Pandas Series, default=None* Target variable. Required if `drop_splits` is set to `True`.

### Methods

#### `split(X=None, y=None, groups=None, init=False)`
Generate train/test indices based on unique periods.

  > ##### Parameters
  > - **X, y, groups**: Always ignored, exist for compatibility.
  > - **init**: *bool, default=False*. Flag indicating initialization phase, when n_splits is modified depending on whether or not drop_splits is True. When split is called apart from initialization, this should be set to False.

  > ##### Returns
  > List of train/test indices.

#### `get_n_splits(X=None, y=None, groups=None)`
Returns the number of splitting iterations in the cross-validator.

  > ##### Parameters
  > - **X, y, groups**: Always ignored, exist for compatibility.
  
  > ##### Returns
  > Number of splits.

#### `gen_snapshots(data, period_col = None)`
Generate snapshots for each split.

  > ##### Parameters
  > - **data**:  *Pandas DataFrame*. DataFrame from which snapshots are generated.
  > - **period_col**: *str, default=None*. The column in data from which the column snapshot_period is created.

  > ##### Returns
  > A pandas DataFrame where each split has its own set of observations.

#### `gen_train_labels(labels)`
Generate train labels for each split.

  > ##### Parameters
  > - **labels**:  *Pandas DataFrame or Series*. The labels used to identify observations.

  > ##### Returns
  > The labels of each fold's train set as a single DataFrame.

#### `gen_test_labels(labels)`
Generate test labels for each split.

  > ##### Parameters
  > - **labels**:  *Pandas DataFrame or Series*. The labels used to identify observations.

  > ##### Returns
  > The labels of each fold's test set as a single DataFrame.

#### `cross_val_fit(estimator, X, y, sample_weight=None, n_jobs=1)`
Perform cross-validated predictions using a given predictor model.
  
  > ##### Parameters
  > - **estimator**: *estimator object implementing ‘fit’*. The object to use to fit the data.
  > - **X**: *Pandas DataFrame*. Features.
  > - **y**: *Pandas Series*. Target variable.
  > - **sample_weight**: *Pandas Series*. Sample weights for the training data.
  > - **n_jobs**: *Optional int (default=1)*. The number of jobs to run in parallel. See the [n_jobs](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html) argument for the Parallel class in the joblib package for further details.
  
  > ##### Returns
  > **fitted_estimators:** A list containing fitted estimators for each split.

#### `cross_val_predict(fitted_estimators, X, prediction_method='predict', return_train_preds=False, n_jobs=1, )`
Perform cross-validated predictions using a list of fitted estimators.
  
  > ##### Parameters
  > - **fitted_estimators**: A list of fitted estimators, one for each split.
  > - **X**: *Pandas DataFrame*. Features.
  > - **prediction_method**: The prediction method to use. It can be 'predict', 'predict_proba', or 'predict_log_proba'. Default is `'predict'`.
  > - **return_train_preds**: **Optional bool (default=False)*. If True, return predictions for the training set as well.
  > - **n_jobs**: *Optional int (default=1)*. The number of jobs to run in parallel. See the [n_jobs](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html) argument for the Parallel class in the joblib package for further details.

  > ##### Returns
  > **y:** *ndarray of shape (n_samples,) or (n_samples, n_outputs)*. The predicted values concatenated across folds. If return_train_preds is True, the output will be **y_test, y_train**. 

#### `cross_val_fit_predict(estimator, X, y, prediction_method='predict', sample_weight=None, n_jobs=1)`
Perform cross-validated predictions using a given predictor model.
  
  > ##### Parameters
  > - **estimator**: estimator object.
  > - **X**: *Pandas DataFrame*. Features.
  > - **y**: *Pandas Series*. Target variable.
  > - **prediction_method**: The prediction method to use. It can be 'predict', 'predict_proba', or 'predict_log_proba'. Default is `'predict'`.
  > - **sample_weight**: *Pandas Series*. Sample weights for the training data.
  > - **n_jobs**: *Optional int (default=1)*. The number of jobs to run in parallel. See the [n_jobs](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html) argument for the Parallel class in the joblib package for further details.
  
  > ##### Returns
  > **y, fitted_estimators:** The predicted values concatenated across folds as well as a list containing fitted estimators for each split. If return_train_preds is True, the output will be **y_test, y_train, fitted_estimators**. 

#### `cross_val_fit_transform(transformer, X, include_test_in_fit=False, transform_train=False)`
Perform cross-validated transformation using a given transformer.

> ##### Parameters
> - **transformer**: Transformer object.
> - **X**: Features.
> - **include_test_in_fit**: *bool (default=False)*. Whether to include test data in fitting for each split.
> - **transform_train**: *bool (default=False)*. Whether to transform train set as well as the test set.

> ##### Returns
> **X, fitted_transformers:** DataFrame containing transformed values during cross-validation as well as a list containing fitted transformers for each split.

---

## Examples

For more examples and detailed usage instructions, refer to the [examples](examples) directory in this repository. Also feel free to check out [an article I wrote about PanelSplit](https://towardsdatascience.com/how-to-cross-validate-your-panel-data-in-python-9ad981ddd043).

## Background
Work on panelsplit started at [EconAI](https://www.linkedin.com/company/econ-ai/) in December 2023 and has been under active development since then.

## Contributing

Contributions to PanelSplit are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
