"""Baselines and a gradient-boosting price model.

The target is the sale price in pounds. The booster is trained on the log
price, because sale prices are right-skewed and a given percentage miss should
cost about the same in a cheap and an expensive market. Every reported metric
is converted back to pounds.

The split is by transfer date. Random splits leak future sales from the same
street into training and overstate how well a model would have priced the
next year.

Postcode district has a few thousand levels. scikit-learn's histogram gradient
boosting only accepts categorical features with at most 255 levels, so the
booster is LightGBM, which splits high-cardinality categories natively.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import OrdinalEncoder

from house_prices.schema import TEST_END, TEST_START, TRAIN_END, VALIDATION_START

CATEGORICAL_FEATURES = [
    "property_type",
    "new_build",
    "tenure",
    "region",
    "postcode_district",
    "month",
]
NUMERIC_FEATURES = ["time_index"]
FEATURE_LABELS = {
    "property_type": "Property type",
    "new_build": "New build",
    "tenure": "Tenure",
    "region": "Region",
    "postcode_district": "Postcode district",
    "month": "Calendar month",
    "time_index": "Months since Jan 2024",
}

# Fallback chain for the strong baseline, from the most specific median that
# is estimated on the training window to the global training median.
BASELINE_LEVELS = [
    ["postcode_district", "property_type"],
    ["postcode_district"],
    ["region", "property_type"],
    ["region"],
]

MODEL_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 4,
    "verbosity": -1,
}
MAX_TREES = 800
EARLY_STOPPING_ROUNDS = 40


@dataclass
class FeatureBuilder:
    """Ordinal-encode categoricals on the training window only.

    Unseen categories and missing values are coded as -1. Histogram gradient
    boosting treats negative categorical codes as missing, so a postcode
    district that never sold in training does not crash prediction.
    """

    categorical: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)
    numeric: tuple[str, ...] = tuple(NUMERIC_FEATURES)
    encoder: OrdinalEncoder | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self.categorical) + list(self.numeric)

    @property
    def categorical_indices(self) -> list[int]:
        return list(range(len(self.categorical)))

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.int32,
        )
        categories = self._categorical_frame(frame)
        encoded = self.encoder.fit_transform(categories)
        return self._stack(encoded, frame)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("FeatureBuilder must be fit before transform.")
        encoded = self.encoder.transform(self._categorical_frame(frame))
        return self._stack(encoded, frame)

    def _categorical_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for column in self.categorical:
            if column == "month":
                out[column] = frame[column].astype("int64").astype("string").str.zfill(2)
            else:
                out[column] = frame[column].astype("string").fillna("Unknown")
        return out

    def _stack(self, encoded: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        # Negative codes are unseen or missing categories. LightGBM treats NaN
        # as its own missing bin; it does not accept negative category codes.
        codes = encoded.astype(np.float64)
        codes[codes < 0] = np.nan
        numeric = frame.loc[:, list(self.numeric)].to_numpy(dtype=np.float64)
        return np.hstack([codes, numeric])


def time_split(
    frame: pd.DataFrame,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on the transfer date. Rows outside both windows are excluded."""
    dates = frame["transfer_date"]
    train = frame.loc[dates <= pd.Timestamp(train_end)].copy()
    test = frame.loc[(dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test set.")
    return train, test


def predict_global_median(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict the training-window median for every test sale."""
    return np.full(len(test), float(train["price"].median()), dtype=np.float64)


def predict_hierarchical_median(
    train: pd.DataFrame,
    test: pd.DataFrame,
    levels: list[list[str]] | None = None,
) -> np.ndarray:
    """Predict training medians, falling back from specific groups to broader ones."""
    levels = levels or BASELINE_LEVELS
    prediction = pd.Series(np.nan, index=test.index, dtype="float64")
    for keys in levels:
        medians = train.groupby(keys, observed=True)["price"].median()
        if len(keys) == 1:
            looked_up = test[keys[0]].map(medians)
        else:
            index = pd.MultiIndex.from_frame(test.loc[:, keys])
            looked_up = pd.Series(medians.reindex(index).to_numpy(), index=test.index)
        prediction = prediction.fillna(looked_up)
    return prediction.fillna(float(train["price"].median())).to_numpy(dtype=np.float64)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, R², and median absolute error, all in pounds."""
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    error = predicted - actual
    residual_sum = float(np.sum(error**2))
    total_sum = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float("nan") if total_sum == 0 else 1.0 - residual_sum / total_sum
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(r2),
        "median_ae": float(np.median(np.abs(error))),
    }


def fit_price_model(
    train: pd.DataFrame,
    validation_start: str = VALIDATION_START,
) -> tuple[LGBMRegressor, FeatureBuilder, dict[str, int]]:
    """Fit a LightGBM model on log price.

    Early stopping uses the last months of the training window. The model is
    then refit on the whole training window for that number of trees, so the
    reported model has seen every pre-test sale and the test set is never used
    to choose a hyperparameter.
    """
    boundary = pd.Timestamp(validation_start)
    fit_frame = train.loc[train["transfer_date"] < boundary]
    validation = train.loc[train["transfer_date"] >= boundary]
    if fit_frame.empty or validation.empty:
        raise ValueError("Training window does not cover the validation boundary.")

    search_builder = FeatureBuilder()
    x_fit = search_builder.fit_transform(fit_frame)
    y_fit = np.log(fit_frame["price"].to_numpy(dtype=np.float64))
    x_val = search_builder.transform(validation)
    y_val = np.log(validation["price"].to_numpy(dtype=np.float64))

    search = LGBMRegressor(**MODEL_PARAMS, n_estimators=MAX_TREES)
    search.fit(
        x_fit,
        y_fit,
        eval_set=[(x_val, y_val)],
        eval_metric="l2",
        categorical_feature=search_builder.categorical_indices,
        callbacks=[
            early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            log_evaluation(period=0),
        ],
    )
    n_iter = int(search.best_iteration_ or MAX_TREES)
    n_iter = max(n_iter, 1)

    builder = FeatureBuilder()
    x_train = builder.fit_transform(train)
    y_train = np.log(train["price"].to_numpy(dtype=np.float64))
    model = LGBMRegressor(**MODEL_PARAMS, n_estimators=n_iter)
    model.fit(x_train, y_train, categorical_feature=builder.categorical_indices)
    info = {
        "n_iter": n_iter,
        "n_fit": int(len(fit_frame)),
        "n_validation": int(len(validation)),
        "n_train": int(len(train)),
    }
    return model, builder, info


def predict_prices(model: LGBMRegressor, builder: FeatureBuilder, frame: pd.DataFrame) -> np.ndarray:
    """Predict sale prices in pounds."""
    log_pred = model.predict(builder.transform(frame))
    return np.exp(log_pred)


def permutation_importance_pounds(
    model: LGBMRegressor,
    builder: FeatureBuilder,
    frame: pd.DataFrame,
    n_repeats: int = 4,
    sample_size: int = 12_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Increase in pound MAE when each feature is shuffled.

    Scored on a fixed sample of the evaluation frame so the cost stays modest.
    Importance is measured in pounds, not on the log scale the model was fit on.
    """
    from sklearn.inspection import permutation_importance

    if len(frame) > sample_size:
        sample = frame.sample(n=sample_size, random_state=random_state)
    else:
        sample = frame
    features = builder.transform(sample)
    y_log = np.log(sample["price"].to_numpy(dtype=np.float64))

    def neg_mae_pounds(estimator, x_values, y_values):
        predicted = np.exp(estimator.predict(x_values))
        actual = np.exp(np.asarray(y_values, dtype=np.float64))
        return -float(np.mean(np.abs(predicted - actual)))

    result = permutation_importance(
        model,
        features,
        y_log,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=neg_mae_pounds,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": builder.feature_names,
            "label": [FEATURE_LABELS.get(name, name) for name in builder.feature_names],
            "mae_increase": result.importances_mean,
            "mae_increase_std": result.importances_std,
        }
    )
    return table.sort_values("mae_increase", ascending=False).reset_index(drop=True)
