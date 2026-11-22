"""Build the analysis table from yearly Price Paid files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from house_prices.clean import clean_transactions, combine_clean_stats, read_price_paid_file
from house_prices.schema import (
    DEFAULT_YEARS,
    PROCESSED_PATH,
    RAW_DIR,
    REPORT_DIR,
    SAMPLE_PATH,
    SAMPLE_ROWS,
    SAMPLE_SEED,
)

logger = logging.getLogger(__name__)


def build_analysis_table(
    raw_dir: Path = RAW_DIR,
    years: tuple[int, ...] | list[int] = DEFAULT_YEARS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read, clean, and stack the requested yearly files."""
    frames: list[pd.DataFrame] = []
    reports: list[dict[str, int]] = []
    for year in years:
        path = raw_dir / f"pp-{year}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download it with `python -m house_prices.download`."
            )
        logger.info("Cleaning %s", path.name)
        cleaned, stats = clean_transactions(read_price_paid_file(path))
        frames.append(cleaned)
        reports.append(stats)
        logger.info("  kept %s of %s rows", f"{stats['rows_kept']:,}", f"{stats['rows_read']:,}")

    combined = pd.concat(frames, ignore_index=True)
    stats = combine_clean_stats(reports)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["transaction_id"], keep="first")
    stats["dropped_duplicate_ids"] = before - len(combined)
    combined = combined.sort_values(["transfer_date", "transaction_id"]).reset_index(drop=True)
    stats["rows_kept"] = int(len(combined))
    return combined, stats


def write_outputs(
    frame: pd.DataFrame,
    processed_path: Path = PROCESSED_PATH,
    sample_path: Path = SAMPLE_PATH,
    sample_rows: int = SAMPLE_ROWS,
) -> None:
    """Write the full parquet (gitignored) and a small committed sample."""
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed_path, index=False)
    sample = frame.sample(n=min(sample_rows, len(frame)), random_state=SAMPLE_SEED)
    sample = sample.sort_values(["transfer_date", "transaction_id"])
    sample.to_csv(sample_path, index=False)
    logger.info("Wrote %s (%s rows)", processed_path, f"{len(frame):,}")
    logger.info("Wrote %s (%s rows)", sample_path, f"{len(sample):,}")


def write_prepare_stats(stats: dict[str, int], path: Path | None = None) -> None:
    """Save the row-level cleaning counts next to the other report files."""
    destination = path or (REPORT_DIR / "prepare_stats.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def load_analysis_table(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the prepared parquet, parsing dates back to datetimes."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build it with `python -m house_prices.prepare`."
        )
    frame = pd.read_parquet(path)
    frame["transfer_date"] = pd.to_datetime(frame["transfer_date"])
    return frame


def load_or_build(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the prepared table, building it from raw files when it is absent."""
    if path.exists():
        return load_analysis_table(path)
    frame, stats = build_analysis_table()
    write_outputs(frame, processed_path=path)
    write_prepare_stats(stats)
    return frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Clean yearly Price Paid files into an analysis table.")
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed", type=Path, default=PROCESSED_PATH)
    args = parser.parse_args()
    frame, stats = build_analysis_table(args.raw_dir, args.years)
    write_outputs(frame, processed_path=args.processed)
    write_prepare_stats(stats)
    for key, value in stats.items():
        logger.info("%s: %s", key, f"{value:,}")


if __name__ == "__main__":
    main()
