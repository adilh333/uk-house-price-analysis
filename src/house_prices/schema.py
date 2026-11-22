"""Paths, column names, and analysis constants.

Yearly Price Paid files are published without a header row. Column order follows
HM Land Registry's documented layout. The final column (record status) is
described as monthly-file-only, but the yearly files used here still include it.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
FIGURE_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

PROCESSED_PATH = PROCESSED_DIR / "transactions.parquet"
SAMPLE_PATH = SAMPLE_DIR / "price_paid_sample.csv"
EDA_SUMMARY_PATH = REPORT_DIR / "eda_summary.json"
MODEL_METRICS_PATH = REPORT_DIR / "model_metrics.json"

# Deed-dated yearly files. 2026 is partial: the file used for this project was
# published on 28 October 2026 and runs through September.
DEFAULT_YEARS = (2024, 2025, 2026)
YEARLY_URL = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-{year}.csv"

RAW_COLUMNS = [
    "transaction_id",
    "price",
    "transfer_date",
    "postcode",
    "property_type",
    "old_new",
    "duration",
    "paon",
    "saon",
    "street",
    "locality",
    "town",
    "district",
    "county",
    "ppd_category",
    "record_status",
]

# Address lines are not needed for the analysis and are not stored.
READ_COLUMNS = [
    "transaction_id",
    "price",
    "transfer_date",
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

PROPERTY_TYPE_MAP = {
    "D": "Detached",
    "S": "Semi-detached",
    "T": "Terraced",
    "F": "Flat",
}

# Category A is a standard residential sale for value. The handful of prices
# below £10,000 are not credible sale prices, and the cap drops the extreme
# right tail (about 0.05% of standard sales) so squared error is not dominated
# by a few super-prime transactions.
MIN_PRICE = 10_000
MAX_PRICE = 5_000_000

# HM Land Registry warns that the latest two months are incomplete because of
# the lag between completion and registration. The 2026 yearly file was last
# modified on 28 October 2026, so August and September 2026 are provisional.
PROVISIONAL_FROM = "2026-08-01"

# Time-based evaluation. The test window stops before the provisional months.
TRAIN_END = "2025-12-31"
VALIDATION_START = "2025-10-01"
TEST_START = "2026-01-01"
TEST_END = "2026-07-31"

SAMPLE_ROWS = 4_000
SAMPLE_SEED = 42

ANALYSIS_COLUMNS = [
    "transaction_id",
    "price",
    "transfer_date",
    "postcode",
    "postcode_area",
    "postcode_district",
    "property_type",
    "new_build",
    "tenure",
    "town",
    "local_authority",
    "county",
    "region",
    "nation",
    "year",
    "month",
    "quarter",
    "time_index",
    "provisional",
]
