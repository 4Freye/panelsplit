import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from panelsplit.cross_validation import PanelSplit


@pytest.fixture
def mock_panel_data():
    entities = 10
    years = 5
    data = []
    for e in range(entities):
        # State mapping: first 5 in State A, next 5 in State B
        state = "A" if e < 5 else "B"
        y_val = 0 if e < 5 else 1
        for y in range(years):
            data.append(
                {
                    "entity_id": e,
                    "state": state,
                    "year": y,
                    "value": np.random.randn(),
                    "y": y_val,
                }
            )
    return pd.DataFrame(data)


def test_normal_time_split(mock_panel_data):
    # Base functionality
    ps = PanelSplit(periods=mock_panel_data["year"], n_splits=2)
    assert ps.n_splits == 2
    assert len(ps.split()) == 2


def test_grouped_spatial_splits(mock_panel_data):
    # Single group logic safely executed with group_splitter
    ps = PanelSplit(
        periods=mock_panel_data["year"],
        n_splits=2,
        groups=mock_panel_data["state"],
        group_splitter=GroupKFold(n_splits=2),
    )
    # Expected total splits = n_splits * group_splitter folds
    assert ps.n_splits == 4

    splits = ps.split()  # GroupKFold computes implicitly fine without X or y given
    assert len(splits) == 4

    groups_array = ps._groups
    for train_idx, test_idx in splits:
        train_groups = set(groups_array[train_idx])
        test_groups = set(groups_array[test_idx])
        # Intersection between train and test groups should be fundamentally empty
        assert len(train_groups.intersection(test_groups)) == 0, (
            "Spatial leakage detected!"
        )


def test_stratified_grouped_spatial_splits(mock_panel_data):
    # StratifiedGroupKFold forces Lazy Eval, requiring X/y
    ps = PanelSplit(
        periods=mock_panel_data["year"],
        n_splits=2,
        groups=mock_panel_data["state"],
        group_splitter=StratifiedGroupKFold(n_splits=2),
    )

    # Needs X and y to compute splits because StratifiedGroupKFold parses `y`
    with pytest.raises(ValueError):
        ps.split()

    splits = ps.split(X=mock_panel_data, y=mock_panel_data["y"])
    assert len(splits) == 4


def test_multi_grouped_spatial_splits(mock_panel_data):
    # Multi group / flattening behavior
    ps = PanelSplit(
        periods=mock_panel_data["year"],
        n_splits=2,
        groups=mock_panel_data[["state", "entity_id"]],
        group_splitter=GroupKFold(n_splits=3),
    )
    assert ps.n_splits == 6
    splits = ps.split()
    assert len(splits) == 6

    groups_array = ps._groups
    for train_idx, test_idx in splits:
        train_groups = set(groups_array[train_idx])
        test_groups = set(groups_array[test_idx])
        assert len(train_groups.intersection(test_groups)) == 0, (
            "Spatial leakage detected in multi-groups!"
        )

        # Test also temporal integrity
        tr_periods = set(mock_panel_data["year"].iloc[train_idx])
        ts_periods = set(mock_panel_data["year"].iloc[test_idx])
        if len(tr_periods) > 0 and len(ts_periods) > 0:
            assert max(tr_periods) < min(ts_periods), "Temporal anomaly encountered!"


def test_plot_splits_with_stratified_group_kfold(mock_panel_data):
    from panelsplit.plot import plot_splits

    ps = PanelSplit(
        periods=mock_panel_data["year"],
        n_splits=2,
        groups=mock_panel_data["state"],
        group_splitter=StratifiedGroupKFold(n_splits=2),
    )

    # Calling plot_splits without previously generating splits shouldn't fail
    result = plot_splits(ps, X=mock_panel_data, y=mock_panel_data["y"], show=False)

    assert result is not None


def test_spatial_splitter_caching(mock_panel_data):
    # 1. Independent splitter (GroupKFold)
    ps_ind = PanelSplit(
        periods=mock_panel_data["year"],
        n_splits=2,
        groups=mock_panel_data["state"],
        group_splitter=GroupKFold(n_splits=2),
    )

    # Pre-generated in __init__ should be cached
    assert ps_ind._cached_splits is not None
    splits_ind_1 = ps_ind.split()
    splits_ind_2 = ps_ind.split(X=mock_panel_data, y=mock_panel_data["y"])
    
    # Check that it returns the exact same cached object reference
    assert splits_ind_1 is splits_ind_2
    assert len(splits_ind_1) == 4

    # 2. Dependent splitter (StratifiedGroupKFold)
    ps_dep = PanelSplit(
        periods=mock_panel_data["year"],
        n_splits=2,
        groups=mock_panel_data["state"],
        group_splitter=StratifiedGroupKFold(n_splits=2),
    )

    # Cannot pre-generate in __init__ due to StratifiedGroupKFold requiring y
    assert ps_dep._cached_splits is None

    # First call with X and y computes and caches
    splits_dep_1 = ps_dep.split(X=mock_panel_data, y=mock_panel_data["y"])
    assert ps_dep._cached_splits is not None

    # Second call with the same X and y should hit the cache
    splits_dep_2 = ps_dep.split(X=mock_panel_data, y=mock_panel_data["y"])
    
    assert splits_dep_1 is splits_dep_2
    assert len(splits_dep_1) == 4

    # Call with a different X/y should miss cache and recompute
    new_X = mock_panel_data.copy()
    splits_dep_3 = ps_dep.split(X=new_X, y=mock_panel_data["y"])
    assert splits_dep_1 is not splits_dep_3
    # Check content is still identical
    for (tr1, ts1), (tr3, ts3) in zip(splits_dep_1, splits_dep_3):
        np.testing.assert_array_equal(tr1, tr3)
        np.testing.assert_array_equal(ts1, ts3)


def test_custom_independent_splitter_caching(mock_panel_data):
    from sklearn.model_selection import GroupShuffleSplit

    # 1. Custom subclass of GroupKFold
    class MyCustomGroupKFold(GroupKFold):
        pass

    ps_custom = PanelSplit(
        periods=mock_panel_data["year"],
        n_splits=2,
        groups=mock_panel_data["state"],
        group_splitter=MyCustomGroupKFold(n_splits=2),
    )

    assert ps_custom._cached_splits is not None
    splits_custom_1 = ps_custom.split()
    splits_custom_2 = ps_custom.split(X=mock_panel_data, y=mock_panel_data["y"])
    assert splits_custom_1 is splits_custom_2

    # 2. GroupShuffleSplit (which is independent of X/y)
    ps_gss = PanelSplit(
        periods=mock_panel_data["year"],
        n_splits=2,
        groups=mock_panel_data["state"],
        group_splitter=GroupShuffleSplit(n_splits=2, random_state=42),
    )

    assert ps_gss._cached_splits is not None
    splits_gss_1 = ps_gss.split()
    splits_gss_2 = ps_gss.split(X=mock_panel_data, y=mock_panel_data["y"])
    assert splits_gss_1 is splits_gss_2

