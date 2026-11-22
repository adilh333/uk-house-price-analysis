"""Tests for Price Paid parsing and the cleaning rules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from house_prices.clean import (
    clean_transactions,
    pretty_place,
    read_price_paid_file,
    split_postcode,
)
from house_prices.schema import READ_COLUMNS


def _row(**overrides) -> dict[str, str]:
    base = {
        "transaction_id": "{AAA}",
        "price": "250000",
        "transfer_date": "2024-06-15 00:00",
        "postcode": "SW1A 1AA",
        "property_type": "F",
        "old_new": "N",
        "duration": "L",
        "town": "LONDON",
        "district": "CITY OF WESTMINSTER",
        "county": "GREATER LONDON",
        "ppd_category": "A",
        "record_status": "A",
    }
    base.update(overrides)
    return base


def _frame(*rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=READ_COLUMNS)


def test_split_postcode_area_and_district():
    assert split_postcode("SW1A 1AA") == ("SW", "SW1A")
    assert split_postcode("ec1a 1bb") == ("EC", "EC1A")
    assert split_postcode("M1 1AE") == ("M", "M1")
    assert split_postcode("W1K 7AA") == ("W", "W1K")
    assert split_postcode("WS13 6US") == ("WS", "WS13")
    assert split_postcode("") == ("Unknown", "Unknown")
    assert split_postcode("not a postcode") == ("Unknown", "Unknown")
    assert split_postcode(None) == ("Unknown", "Unknown")


def test_pretty_place_keeps_small_words():
    assert pretty_place("CITY OF WESTMINSTER") == "City of Westminster"
    assert pretty_place("KINGSTON UPON HULL") == "Kingston upon Hull"
    assert pretty_place("THE VALE OF GLAMORGAN") == "The Vale of Glamorgan"
    assert pretty_place("  ") == "Unknown"


def test_clean_keeps_a_standard_leasehold_flat():
    cleaned, stats = clean_transactions(_frame(_row()))
    assert stats["rows_kept"] == 1
    row = cleaned.iloc[0]
    assert row["price"] == 250000
    assert row["property_type"] == "Flat"
    assert row["new_build"] == "Existing"
    assert row["tenure"] == "Leasehold"
    assert row["region"] == "London"
    assert row["nation"] == "England"
    assert row["postcode_area"] == "SW"
    assert row["postcode_district"] == "SW1A"
    assert row["local_authority"] == "City of Westminster"
    assert row["month"] == 6
    assert row["time_index"] == 5
    assert bool(row["provisional"]) is False


def test_clean_drops_non_standard_deleted_and_out_of_range_prices():
    rows = [
        _row(transaction_id="{KEEP}", county="west yorkshire", postcode="LS1 1UR", property_type="s", duration="f"),
        _row(transaction_id="{CHEAP}", price="500"),
        _row(transaction_id="{PRIME}", price="9000000"),
        _row(transaction_id="{CATB}", ppd_category="B"),
        _row(transaction_id="{DELETED}", record_status="D"),
        _row(transaction_id="{OTHER}", property_type="O"),
        _row(transaction_id="{BADDATE}", transfer_date="not-a-date"),
        _row(transaction_id="{WALES}", postcode="CF10 1AA", county="CARDIFF", property_type="T", duration="F"),
    ]
    cleaned, stats = clean_transactions(_frame(*rows))
    kept = set(cleaned["transaction_id"])
    assert kept == {"{KEEP}", "{WALES}"}
    assert stats["dropped_non_standard"] == 1
    assert stats["dropped_deleted"] == 1
    assert stats["dropped_price_bounds"] == 2
    assert stats["dropped_property_type"] == 1
    assert stats["dropped_invalid_price_or_date"] == 1
    wales = cleaned.loc[cleaned["transaction_id"].eq("{WALES}")].iloc[0]
    assert wales["region"] == "Wales"
    assert wales["nation"] == "Wales"
    yorkshire = cleaned.loc[cleaned["transaction_id"].eq("{KEEP}")].iloc[0]
    assert yorkshire["region"] == "Yorkshire and The Humber"
    assert yorkshire["property_type"] == "Semi-detached"
    assert yorkshire["tenure"] == "Freehold"
    assert yorkshire["postcode_district"] == "LS1"


def test_clean_marks_provisional_months_and_new_builds():
    cleaned, _stats = clean_transactions(
        _frame(
            _row(
                transaction_id="{NEW}",
                transfer_date="2026-08-20 00:00",
                old_new="Y",
                property_type="D",
                duration="F",
                postcode="",
                county="DEVON",
            )
        )
    )
    row = cleaned.iloc[0]
    assert row["new_build"] == "New build"
    assert bool(row["provisional"]) is True
    assert row["postcode_area"] == "Unknown"
    assert row["region"] == "South West"
    assert row["time_index"] == (2026 - 2024) * 12 + 8 - 1


def test_read_price_paid_file_uses_documented_column_order(tmp_path: Path):
    line = (
        '"{ID}","405000","2024-09-20 00:00","WS13 6US","S","N","F","24","","TAYLOR WAY","","LICHFIELD",'
        '"LICHFIELD","STAFFORDSHIRE","A","A"\n'
    )
    path = tmp_path / "pp-snippet.csv"
    path.write_text(line, encoding="utf-8")
    raw = read_price_paid_file(path)
    assert list(raw.columns) == READ_COLUMNS
    cleaned, stats = clean_transactions(raw)
    assert stats["rows_kept"] == 1
    assert cleaned.loc[0, "postcode_district"] == "WS13"
    assert cleaned.loc[0, "region"] == "West Midlands"
    assert cleaned.loc[0, "town"] == "Lichfield"
