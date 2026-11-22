"""Turn raw Price Paid rows into an analysis table.

The cleaning rules are deliberate and conservative:

* Keep standard residential sales only (PPD category A). Category B mixes
  repossessions, identified buy-to-lets, corporate buyers, and non-standard
  property, so it is not a like-for-like market price.
* Drop deleted monthly records if any appear.
* Keep the four dwelling types Land Registry uses for standard stock.
* Drop prices below £10,000 or above £5,000,000.
* Derive postcode area and district, and map county to region.
"""

from __future__ import annotations

import re
from typing import Mapping

import numpy as np
import pandas as pd

from house_prices.regions import county_to_region
from house_prices.schema import (
    ANALYSIS_COLUMNS,
    MAX_PRICE,
    MIN_PRICE,
    PROPERTY_TYPE_MAP,
    PROVISIONAL_FROM,
    RAW_COLUMNS,
    READ_COLUMNS,
)

_POSTCODE_RE = re.compile(r"^([A-Z]{1,2})(\d{1,2}[A-Z]?)")
_SMALL_WORDS = {"of", "and", "upon", "the", "with", "under"}


def split_postcode(postcode: object) -> tuple[str, str]:
    """Return ``(area, district)`` for a UK postcode.

    ``SW1A 1AA`` becomes ``("SW", "SW1A")``. Blank or unparseable values
    become ``("Unknown", "Unknown")``.
    """
    if postcode is None or pd.isna(postcode):
        return "Unknown", "Unknown"
    text = str(postcode).strip().upper()
    match = _POSTCODE_RE.match(text)
    if not match:
        return "Unknown", "Unknown"
    area = match.group(1)
    return area, f"{area}{match.group(2)}"


def pretty_place(name: object) -> str:
    """Title-case a place name, keeping small words lower case."""
    if name is None or pd.isna(name):
        return "Unknown"
    text = " ".join(str(name).strip().split())
    if not text:
        return "Unknown"
    parts = text.title().split(" ")
    fixed = [parts[0]]
    for part in parts[1:]:
        fixed.append(part.lower() if part.lower() in _SMALL_WORDS else part)
    return " ".join(fixed)


def read_price_paid_file(path: str | pd.PathLike) -> pd.DataFrame:
    """Read one headerless Price Paid CSV, keeping the columns this project uses."""
    frame = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        usecols=READ_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )
    return frame.loc[:, READ_COLUMNS]


def _normalise_codes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text_cols = [
        "transaction_id",
        "postcode",
        "property_type",
        "old_new",
        "duration",
        "town",
        "district",
        "county",
        "ppd_category",
        "record_status",
    ]
    for column in text_cols:
        out[column] = out[column].fillna("").astype("string").str.strip()
    for column in ("property_type", "old_new", "duration", "county", "ppd_category", "record_status"):
        out[column] = out[column].str.upper()
    out["postcode"] = out["postcode"].str.upper()
    return out


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean one raw frame.

    Returns the analysis table and a count of rows removed at each rule.
    The counts are sequential, so each row is attributed to the first rule
    that rejects it.
    """
    missing = [column for column in READ_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Raw frame is missing columns: {missing}")

    df = _normalise_codes(raw)
    stats: dict[str, int] = {"rows_read": int(len(df))}

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce",
        format="%Y-%m-%d %H:%M",
    )

    df, removed = _drop(df, df["price"].isna() | df["transfer_date"].isna() | df["transaction_id"].eq(""))
    stats["dropped_invalid_price_or_date"] = removed

    df, removed = _drop(df, df["record_status"].eq("D"))
    stats["dropped_deleted"] = removed

    df, removed = _drop(df, df["ppd_category"].ne("A"))
    stats["dropped_non_standard"] = removed

    df, removed = _drop(df, ~df["property_type"].isin(PROPERTY_TYPE_MAP))
    stats["dropped_property_type"] = removed

    df, removed = _drop(df, ~df["duration"].isin(["F", "L"]))
    stats["dropped_tenure"] = removed

    df, removed = _drop(df, ~df["old_new"].isin(["Y", "N"]))
    stats["dropped_new_build_flag"] = removed

    df, removed = _drop(df, (df["price"] < MIN_PRICE) | (df["price"] > MAX_PRICE))
    stats["dropped_price_bounds"] = removed

    if df.empty:
        stats["rows_kept"] = 0
        stats["rows_unknown_region"] = 0
        return pd.DataFrame(columns=ANALYSIS_COLUMNS), stats

    extracted = df["postcode"].str.extract(_POSTCODE_RE.pattern)
    area = extracted[0]
    district_number = extracted[1]
    df["postcode_area"] = area.fillna("Unknown")
    df["postcode_district"] = np.where(area.notna() & district_number.notna(), area + district_number, "Unknown")
    df["property_type"] = df["property_type"].map(PROPERTY_TYPE_MAP)
    df["new_build"] = np.where(df["old_new"].eq("Y"), "New build", "Existing")
    df["tenure"] = np.where(df["duration"].eq("F"), "Freehold", "Leasehold")
    df["region"] = df["county"].map(county_to_region).fillna("Unknown")
    df["nation"] = np.where(
        df["region"].eq("Wales"),
        "Wales",
        np.where(df["region"].eq("Unknown"), "Unknown", "England"),
    )
    df["town"] = df["town"].map(pretty_place)
    df["local_authority"] = df["district"].map(pretty_place)
    df["county"] = df["county"].map(pretty_place)
    df["year"] = df["transfer_date"].dt.year.astype("int16")
    df["month"] = df["transfer_date"].dt.month.astype("int8")
    df["quarter"] = df["transfer_date"].dt.quarter.astype("int8")
    df["time_index"] = ((df["year"].astype("int16") - 2024) * 12 + df["month"].astype("int16") - 1).astype(
        "int16"
    )
    df["provisional"] = df["transfer_date"] >= pd.Timestamp(PROVISIONAL_FROM)
    df["price"] = df["price"].astype("int64")

    cleaned = df.loc[:, ANALYSIS_COLUMNS].reset_index(drop=True)
    stats["rows_kept"] = int(len(cleaned))
    stats["rows_unknown_region"] = int((cleaned["region"] == "Unknown").sum())
    return cleaned, stats


def combine_clean_stats(parts: list[Mapping[str, int]]) -> dict[str, int]:
    """Sum per-file cleaning counts."""
    totals: dict[str, int] = {}
    for part in parts:
        for key, value in part.items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def _drop(df: pd.DataFrame, mask: pd.Series) -> tuple[pd.DataFrame, int]:
    removed = int(mask.sum())
    if removed:
        df = df.loc[~mask]
    return df, removed
