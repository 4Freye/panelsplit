![PyPI - Version](https://img.shields.io/pypi/v/panelsplit)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.114933814.svg)](https://doi.org/10.5281/zenodo.14933814)

# PanelSplit: a tool for panel data analysis

PanelSplit is a Python package designed to facilitate time series cross-validation when working with multiple entities (aka panel data). This tool is useful for handling panel data in various stages throughout the data pipeline, including feature engineering, hyper-parameter tuning, and model estimation.

## Installation

You can install PanelSplit using pip:

```bash
pip install panelsplit
```
---
## Documentation

To read the documentation, visit [here](https://4freye.github.io/panelsplit/panelsplit.html).


### Example Usage

```python
import pandas as pd
from panelsplit import PanelSplit

# Generate example data
num_countries = 2
years = range(2001, 2004)
num_years = len(years)

data_dict = {
    'country_id': [c for c in range(1, num_countries + 1) for _ in years],
    'year': [year for _ in range(num_countries) for year in years],
    'y': np.random.normal(0, 1, num_countries * num_years),
    'x1': np.random.normal(0, 1, num_countries * num_years),
    'x2': np.random.normal(0, 1, num_countries * num_years)
}

panel_data = pd.DataFrame(data_dict)
panel_split = PanelSplit(periods = panel_data.year, n_splits =2)

splits = panel_split.split()

for train_idx, test_idx in splits:
    print("Train:"); display(panel_data.loc[train_idx]) 
    print("Test:"); display(panel_data.loc[test_idx])
```

For more examples and detailed usage instructions, refer to the [examples](examples) directory in this repository. Also feel free to check out [an article I wrote about PanelSplit](https://towardsdatascience.com/how-to-cross-validate-your-panel-data-in-python-9ad981ddd043).

## Background
Work on panelsplit started at [EconAI](https://www.linkedin.com/company/econ-ai/) in December 2023 and has been under active development since then.

## Contributing

Contributions to PanelSplit are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
