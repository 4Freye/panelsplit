"""

# panelsplit: A tool for panel data analysis

panelsplit is a Python package designed to facilitate time series cross-validation for panel (multi-entity) data. Whether you're in feature engineering, hyper-parameter tuning, or model estimation, panelsplit provides robust modules that make working with panel data both easier and more efficient.

**Key Features:**
- **Panel data cross-validation:** Split up your panel dataset, respecting its temporal structure.
- **Data compatibility:** Works effortlessly with both numpy arrays and pandas DataFrames.
- **Flexible pipelines:** Easily build pipelines that integrate with popular libraries such as scikit-learn and feature-engine.
- **Parallel Processing:** Leverages parallel computing to speed up cross-validation and prediction tasks.


---

## Why choose panelsplit?

panelsplit is built with the practical needs of data scientists working with panel data in mind:
- **Robust & flexible:** Whether experimenting with models or deploying production pipelines, its modular design lets you focus on analysis rather than plumbing.
- **User-friendly:** Clear API design and comprehensive documentation make it easy to integrate into your existing workflows.
- **Efficient:** Parallel processing and tailored cross-validation ensure that your computations are both fast and accurate.

Explore the modules in detail by clicking on the links below to see full documentation and examples.

---
## Modules


### `panelsplit.cross_validation`
- **PanelSplit class:** Automatically generates train/test splits while preserving temporal order and handling edge cases.
- **Label generation:** Provides helper functions to create training and testing labels.
- **Snapshot generation:** Generates snapshots of data in cases where transformations aren't comparable across time.

### `panelsplit.application`
- **Model fitting & prediction:** Fits models on each training split using cloned estimators and supports multiple prediction methods (e.g., `predict`, `predict_proba`).
- **Parallel execution:** Leverages parallel processing for efficient handling of cross-validation splits.
- **Data integrity:** Restores predictions to the original data order for consistency.

### `panelsplit.pipeline`
- **Sequential processing:** Chains multiple transformers and estimators into a streamlined workflow.
- **Dynamic method injection:** Automatically creates methods (like `predict` and `score`) based on the final estimator’s capabilities.
- **Out-of-fold predictions:** Supports cross-validation based predictions with reassembled outputs.

### `panelsplit.plot`
- Visualize time series splits easily.
"""

__all__ = ["application", "cross_validation", "pipeline", "plot"]


