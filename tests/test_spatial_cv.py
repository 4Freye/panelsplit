import pytest
import pandas as pd
import numpy as np

from panelsplit.cross_validation import PanelSplit

@pytest.fixture
def mock_panel_data():
    entities = 10
    years = 5
    data = []
    for e in range(entities):
        # State mapping: first 5 in State A, next 5 in State B
        state = "A" if e < 5 else "B"
        for y in range(years):
            data.append({"entity_id": e, "state": state, "year": y, "value": np.random.randn()})
    return pd.DataFrame(data)


def test_normal_time_split(mock_panel_data):
    # Base functionality
    ps = PanelSplit(periods=mock_panel_data["year"], n_splits=2)
    assert ps.n_splits == 2
    assert len(ps.split()) == 2


def test_grouped_spatial_splits(mock_panel_data):
    # Single group logic
    ps = PanelSplit(
        periods=mock_panel_data["year"],
        n_splits=2,
        groups=mock_panel_data["state"],
        n_group_splits=2
    )
    # Expected total splits = n_splits * n_group_splits
    assert ps.n_splits == 4
    splits = ps.split()
    assert len(splits) == 4

    groups_array = ps._groups
    for train_idx, test_idx in splits:
        train_groups = set(groups_array[train_idx])
        test_groups = set(groups_array[test_idx])
        # Intersection between train and test groups should be fundamentally empty
        assert len(train_groups.intersection(test_groups)) == 0, "Spatial leakage detected!"


def test_multi_grouped_spatial_splits(mock_panel_data):
    # Multi group / flattening behavior
    ps = PanelSplit(
        periods=mock_panel_data["year"],
        n_splits=2,
        groups=mock_panel_data[["state", "entity_id"]],
        n_group_splits=3
    )
    assert ps.n_splits == 6
    splits = ps.split()
    assert len(splits) == 6

    groups_array = ps._groups
    for train_idx, test_idx in splits:
        train_groups = set(groups_array[train_idx])
        test_groups = set(groups_array[test_idx])
        assert len(train_groups.intersection(test_groups)) == 0, "Spatial leakage detected in multi-groups!"
        
        # Test also temporal integrity
        tr_periods = set(mock_panel_data["year"].iloc[train_idx])
        ts_periods = set(mock_panel_data["year"].iloc[test_idx])
        if len(tr_periods) > 0 and len(ts_periods) > 0:
            assert max(tr_periods) <= max(ts_periods), "Temporal anomaly encountered!"
