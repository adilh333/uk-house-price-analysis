"""Download HM Land Registry yearly Price Paid files.

The yearly files are collated on the transfer (deed) date, unlike the monthly
change files. They are large enough to be useful and small enough to avoid the
multi-gigabyte complete extract. Nothing downloaded here is committed.
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

from house_prices.schema import DEFAULT_YEARS, RAW_DIR, YEARLY_URL

logger = logging.getLogger(__name__)

USER_AGENT = "uk-house-price-analysis/1.0 (research; Open Government Licence reuse)"


def yearly_file_url(year: int) -> str:
    """Public URL of the deed-dated yearly Price Paid CSV."""
    if year < 1995 or year > 2100:
        raise ValueError(f"Unexpected Price Paid year: {year}")
    return YEARLY_URL.format(year=year)


def content_length(url: str, timeout: int = 60) -> int | None:
    """Return the remote Content-Length, or None if the server does not send it."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        header = response.headers.get("Content-Length")
    if not header:
        return None
    return int(header)


def destination_is_current(path: Path, expected_bytes: int | None) -> bool:
    """True when a local file already matches the published size."""
    if not path.exists():
        return False
    size = path.stat().st_size
    if size <= 0:
        return False
    if expected_bytes is None:
        return True
    return size == expected_bytes


def download_file(url: str, dest: Path, timeout: int = 180) -> None:
    """Stream ``url`` to ``dest`` via a temporary file, then rename it into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
            if total and written != total:
                raise IOError(f"Incomplete download for {url}: got {written} of {total} bytes")
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.replace(dest)


def download_years(
    years: tuple[int, ...] | list[int] = DEFAULT_YEARS,
    dest_dir: Path = RAW_DIR,
    force: bool = False,
) -> list[Path]:
    """Download each yearly file, skipping files that already match the remote size."""
    saved: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        url = yearly_file_url(year)
        dest = dest_dir / f"pp-{year}.csv"
        expected = None
        try:
            expected = content_length(url)
        except Exception as exc:
            logger.warning("Could not read Content-Length for %s (%s); downloading if needed.", url, exc)
        if not force and destination_is_current(dest, expected):
            logger.info("Already present: %s", dest.name)
            saved.append(dest)
            continue
        logger.info("Downloading %s", url)
        download_file(url, dest)
        saved.append(dest)
    return saved


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Download HM Land Registry yearly Price Paid files.")
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true", help="Download again even if the file size matches.")
    args = parser.parse_args()
    paths = download_years(args.years, args.dest, force=args.force)
    for path in paths:
        logger.info("%s  (%.1f MB)", path, path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
