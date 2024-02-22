# PanelSplit: a tool for panel data analysis

PanelSplit is a Python package designed to facilitate time series cross-validation when working with multiple entities (aka panel data). This tool is useful for handling panel data in various stages throughout the data pipeline, including feature engineering, hyper-parameter tuning, and model estimation.

## Installation

You can install PanelSplit using pip:

```bash
pip install git+https://github.com/4Freye/panelsplit.git
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
- **drop_folds**: *bool, default=False*. Flag to drop folds with either empty or single unique values in train or test sets.
- **y**: *Pandas Series, default=None* Target variable. Required if `drop_folds` is set to `True`.

### Methods

#### `split(X=None, y=None, groups=None, init=False)`
Generate train/test indices based on unique periods.

  > ##### Parameters
  > - **X, y, groups**: Always ignored, exist for compatibility.
  > - **init**: Flag indicating initialization phase, when n_splits is modified depending on whether or not drop_folds is True. When split is called apart from initialization, this should be set to False.

  > ##### Returns
  > List of train/test indices.

#### `get_n_splits(X=None, y=None, groups=None)`
Returns the number of splitting iterations in the cross-validator.

  > ##### Parameters
  > - **X, y, groups**: Always ignored, exist for compatibility.
  
  > ##### Returns
  > Number of splits.

#### `gen_snapshots(self, data, period_col = None)`
Generate snapshots for each split.

  > ##### Parameters
  > - **data**: A pandas DataFrame from which snapshots are generated.
  > - **period_col**: Optional. A str, the column in data from which the column snapshot_period is created.

  > ##### Returns
  > A pandas DataFrame where each split has its own set of observations.

#### `cross_val_fit(estimator, X, y, sample_weight=None, n_jobs=1)`
Perform cross-validated predictions using a given predictor model.
  
  > ##### Parameters
  > - **estimator**: estimator object.
  > - **X**: Features.
  > - **y**: Target variable.
  > - **sample_weight**: Sample weights for the training data.
  > - **n_jobs**: *Optional int (default=1)*. The number of jobs to run in parallel.
  
  > ##### Returns
  > **fitted_models:** A list containing fitted models for each fold.

#### `cross_val_predict(fitted_models, X, labels, prediction_method='predict', y_pred_col=None, n_jobs=1)`
Perform cross-validated predictions using a given predictor model.
  
  > ##### Parameters
  > - **fitted_models**: A list of fitted models, one for each fold.
  > - **X**: Features.
  > - **labels**: pandas DataFrame containing labels for the target variable predicted by the model. The predicted target will be a new column added to this DataFrame.
  > - **prediction_method**: The prediction method to use. It can be 'predict', 'predict_proba', or 'predict_log_proba'. Default is `'predict'`.
  > - **y_pred_col**: Column name for the predicted values. Default is `None`, in which case the name will be the name of `y.name + 'pred'`. If y does not have a name attribute, the name will be `'y_pred'`.
  > - **n_jobs**: *Optional int (default=1)*. The number of jobs to run in parallel.

  > ##### Returns
  > **result_df:** Concatenated DataFrame containing predictions made by the model during cross-validation.


#### `cross_val_fit_predict(estimator, X, y, labels, prediction_method='predict', y_pred_col=None, sample_weight=None, n_jobs=1)`
Perform cross-validated predictions using a given predictor model.
  
  > ##### Parameters
  > - **estimator**: estimator object.
  > - **X**: Features.
  > - **y**: Target variable.
  > - **labels**: pandas DataFrame containing labels for the target variable predicted by the model. The predicted target will be a new column added to this DataFrame.
  > - **prediction_method**: The prediction method to use. It can be 'predict', 'predict_proba', or 'predict_log_proba'. Default is `'predict'`.
  > - **y_pred_col**: Column name for the predicted values. Default is `None`, in which case the name will be the name of `y.name + 'pred'`. If y does not have a name attribute, the name will be `'y_pred'`.
  > - **sample_weight**: Sample weights for the training data.
  > - **n_jobs**: *Optional int (default=1)*. The number of jobs to run in parallel.
  
  > ##### Returns
  > **result_df, fitted_models:** Concatenated DataFrame containing predictions made by the model during cross-validation as well as a list containing fitted models for each fold.

#### `cross_val_fit_transform(transformer, X, include_test_in_fit=False, transform_train=False)`
Perform cross-validated transformation using a given transformer.

> ##### Parameters
> - **transformer**: Transformer object.
> - **X**: Features.
> - **include_test_in_fit**: *bool (default=False)*. Whether to include test data in fitting for each fold.
> - **transform_train**: *bool (default=False)*. Whether to transform train set as well as the test set.

> ##### Returns
> DataFrame containing transformed values during cross-validation.
> **result_df, fitted_transformers:** DataFrame containing transformed values during cross-validation as well as a list containing fitted transformers for each fold.

---

## Examples

For more examples and detailed usage instructions, refer to the [examples](examples) directory in this repository.

## Contributing

Contributions to PanelSplit are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This repo is a work in progress. Stay tuned!
