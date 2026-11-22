"""Tests for the time split, metrics, and median baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from house_prices.modeling import (
    predict_global_median,
    predict_hierarchical_median,
    regression_metrics,
    time_split,
)


def _sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transfer_date": pd.to_datetime(
                [
                    "2024-03-01",
                    "2024-06-01",
                    "2025-11-01",
                    "2025-12-15",
                    "2026-02-01",
                    "2026-06-01",
                    "2026-09-01",
                ]
            ),
            "price": [100_000, 300_000, 200_000, 400_000, 150_000, 250_000, 900_000],
            "postcode_district": ["M1", "M1", "SW1", "SW1", "M1", "ZZ9", "M1"],
            "property_type": ["Flat", "Detached", "Flat", "Detached", "Flat", "Flat", "Flat"],
            "region": ["North West", "North West", "London", "London", "North West", "London", "North West"],
        }
    )


def test_time_split_holds_out_2026_and_drops_provisional_months():
    train, test = time_split(_sales())
    assert train["transfer_date"].max() <= pd.Timestamp("2025-12-31")
    assert test["transfer_date"].min() >= pd.Timestamp("2026-01-01")
    assert test["transfer_date"].max() <= pd.Timestamp("2026-07-31")
    assert len(train) == 4
    assert len(test) == 2


def test_regression_metrics_match_sklearn():
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 180.0, 360.0])
    metrics = regression_metrics(actual, predicted)
    assert metrics["mae"] == mean_absolute_error(actual, predicted)
    assert metrics["rmse"] == root_mean_squared_error(actual, predicted)
    assert metrics["r2"] == r2_score(actual, predicted)
    assert metrics["median_ae"] == 20.0


def test_hierarchical_median_falls_back_when_the_group_is_unseen():
    frame = _sales()
    train, test = time_split(frame)
    global_median = float(train["price"].median())
    assert predict_global_median(train, test).tolist() == [global_median, global_median]

    predicted = predict_hierarchical_median(train, test)
    # M1 flats in training: 100,000 only, so the February 2026 M1 flat uses that median.
    assert predicted[0] == 100_000
    # ZZ9 never appears in training. London flats do: the training London flat median is 200,000.
    assert predicted[1] == 200_000


def test_unseen_postcode_district_is_missing():
    from house_prices.modeling import FeatureBuilder

    train = pd.DataFrame(
        {
            "property_type": ["Flat", "Detached"],
            "new_build": ["Existing", "Existing"],
            "tenure": ["Leasehold", "Freehold"],
            "region": ["London", "North West"],
            "postcode_district": ["SW1A", "M1"],
            "month": [1, 6],
            "time_index": [0, 5],
        }
    )
    holdout = train.copy()
    holdout["postcode_district"] = ["SW1A", "ZZ9"]
    builder = FeatureBuilder()
    builder.fit_transform(train)
    transformed = builder.transform(holdout)
    district = builder.feature_names.index("postcode_district")
    assert not np.isnan(transformed[0, district])
    assert np.isnan(transformed[1, district])


def test_download_helpers(tmp_path):
    from house_prices.download import destination_is_current, yearly_file_url

    assert yearly_file_url(2024).endswith("/pp-2024.csv")
    path = tmp_path / "pp-2024.csv"
    assert destination_is_current(path, 10) is False
    path.write_bytes(b"abc")
    assert destination_is_current(path, 3) is True
    assert destination_is_current(path, 4) is False
    assert destination_is_current(path, None) is True
