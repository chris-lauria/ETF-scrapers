#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 15:43:52 2025

@author: christopher
"""
#!/usr/bin/env python3
import sys
import csv
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

URL = "https://www.simplify.us/etfs/pink-simplify-health-care-etf"


# -------------------------------------------------------------------
# Scraping + snapshot saving
# -------------------------------------------------------------------

def fetch_pink_holdings(debug=False):
    """
    Scrape PINK holdings from Simplify's website.

    Returns:
        as_of_date (datetime.date): the 'As of' date from the page
        holdings (list[dict]): list with keys:
            - ticker
            - name
            - quantity
            - weight_pct
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    resp = requests.get(URL, headers=headers, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines()]

    # 1. Find the "Portfolio Holdings" section
    start_idx = None
    for i, line in enumerate(lines):
        if line.replace("#", "").strip() == "Portfolio Holdings":
            start_idx = i
            break

    if start_idx is None:
        raise RuntimeError("Could not find 'Portfolio Holdings' section in page.")

    # 2. Find "As of ..." date line shortly after that
    as_of_date = None
    for i in range(start_idx + 1, min(start_idx + 6, len(lines))):
        if lines[i].startswith("As of "):
            # Example: "As of 11/19/2025"
            date_str = lines[i].replace("As of", "").strip()
            try:
                as_of_date = datetime.strptime(date_str, "%m/%d/%Y").date()
            except ValueError:
                pass
            break

    if as_of_date is None:
        raise RuntimeError("Could not parse 'As of' date from page.")

    # 3. Find the header block: Ticker / Name / Quantity / Weight
    header_idx = None
    for i in range(start_idx + 1, len(lines) - 3):
        if (
            lines[i] == "Ticker"
            and lines[i + 1] == "Name"
            and lines[i + 2] == "Quantity"
            and lines[i + 3] == "Weight"
        ):
            header_idx = i
            break

    if header_idx is None:
        if debug:
            print("DEBUG: Could not find Ticker/Name/Quantity/Weight header.")
            for l in lines[start_idx:start_idx + 40]:
                print(repr(l))
        raise RuntimeError("Could not find holdings header block.")

    holdings = []

    # 4. Parse groups of 4 lines: ticker, name, quantity, weight
    data_start = header_idx + 4
    i = data_start

    while i + 3 < len(lines):
        ticker = lines[i].strip()
        name = lines[i + 1].strip()
        qty_line = lines[i + 2].strip()
        wt_line = lines[i + 3].strip()

        # Stop if we hit the end of the holdings section
        if (
            ticker.startswith("Holdings are subject to change")
            or ticker.startswith("Distribution History")
            or ticker.startswith("### Distribution History")
        ):
            break

        # Stop if header repeats or we hit something obviously not a holding
        if ticker in ("Ticker", "Name", "Quantity", "Weight") or not ticker:
            break

        try:
            quantity = float(qty_line.replace(",", ""))
            weight_pct = float(wt_line.replace("%", "").replace(",", ""))
        except ValueError:
            # Likely reached a non-data block
            break

        holdings.append(
            {
                "ticker": ticker,
                "name": name,
                "quantity": quantity,
                "weight_pct": weight_pct,
            }
        )

        i += 4  # move to next holding (next 4 lines)

    if not holdings:
        if debug:
            print("DEBUG: No holdings parsed. Lines after header:")
            for l in lines[start_idx:start_idx + 60]:
                print(repr(l))
        raise RuntimeError("Found holdings header but could not parse any rows.")

    return as_of_date, holdings


def get_base_dir():
    # Folder where this script lives
    return Path(__file__).resolve().parent


def save_snapshot_csv(as_of_date, holdings, base_dir=None):
    """
    Save holdings snapshot to CSV, sorted by weight descending.

    CSV columns: ticker, name, quantity, weight_pct
    """
    if base_dir is None:
        base_dir = get_base_dir()

    output_dir = base_dir / "data" / "pink"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"pink_holdings_{as_of_date.isoformat()}.csv"
    filepath = output_dir / filename

    # Don't overwrite existing snapshot for same as_of_date
    if filepath.exists():
        print(f"Snapshot already exists for {as_of_date} at:\n  {filepath}")
        return filepath

    # Sort by weight descending
    holdings_sorted = sorted(
        holdings, key=lambda h: h["weight_pct"], reverse=True
    )

    fieldnames = ["ticker", "name", "quantity", "weight_pct"]

    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in holdings_sorted:
            writer.writerow(row)

    print(f"Saved snapshot for {as_of_date} to:\n  {filepath}")
    return filepath


# -------------------------------------------------------------------
# Snapshot discovery & loading
# -------------------------------------------------------------------

def find_snapshot_files(base_dir=None):
    """
    Find all pink_holdings_YYYY-MM-DD.csv files and return a dict:
        {date: path}
    """
    if base_dir is None:
        base_dir = get_base_dir()

    data_dir = base_dir / "data" / "pink"
    if not data_dir.exists():
        return {}

    snapshot_map = {}
    for path in data_dir.glob("pink_holdings_*.csv"):
        stem = path.stem  # e.g. 'pink_holdings_2025-11-20'
        prefix = "pink_holdings_"
        if not stem.startswith(prefix):
            continue
        date_str = stem[len(prefix):]
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        snapshot_map[d] = path

    return snapshot_map


def load_holdings_csv(path):
    """
    Load a snapshot CSV into a list of dicts with proper types.
    """
    holdings = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            holdings.append(
                {
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "quantity": float(row["quantity"]),
                    "weight_pct": float(row["weight_pct"]),
                }
            )
    return holdings


# -------------------------------------------------------------------
# Comparison logic
# -------------------------------------------------------------------

def choose_dates_for_compare(snapshot_map, curr_date_str=None, prev_date_str=None):
    """
    Determine which two dates to compare.

    Cases:
      - No strings provided:
          compare latest snapshot vs the immediately previous snapshot.
      - Only curr_date_str provided:
          compare that date vs the previous available snapshot before it.
      - Both provided:
          use them exactly.
    """
    if not snapshot_map:
        raise RuntimeError("No snapshot CSVs found in data/pink.")

    all_dates = sorted(snapshot_map.keys())

    # Helper to parse YYYY-MM-DD
    def parse_date_str(s):
        return datetime.strptime(s, "%Y-%m-%d").date()

    # 1) No dates → latest vs previous latest
    if curr_date_str is None and prev_date_str is None:
        if len(all_dates) < 2:
            raise RuntimeError("Need at least two snapshot dates to compare.")
        curr_date = all_dates[-1]
        prev_date = all_dates[-2]

    # 2) Only curr date given → compare that vs snapshot just before it
    elif curr_date_str is not None and prev_date_str is None:
        curr_date = parse_date_str(curr_date_str)
        if curr_date not in snapshot_map:
            raise RuntimeError(f"No snapshot found for current date {curr_date}")

        # Find previous date in sorted list
        idx = all_dates.index(curr_date)
        if idx == 0:
            raise RuntimeError(
                f"{curr_date} is the earliest snapshot; no previous snapshot to compare."
            )
        prev_date = all_dates[idx - 1]

    # 3) Both dates given → use them directly
    else:
        curr_date = parse_date_str(curr_date_str)
        prev_date = parse_date_str(prev_date_str)

    if curr_date not in snapshot_map:
        raise RuntimeError(f"No snapshot found for current date {curr_date}")
    if prev_date not in snapshot_map:
        raise RuntimeError(f"No snapshot found for previous date {prev_date}")

    return curr_date, prev_date, snapshot_map[curr_date], snapshot_map[prev_date]


def compare_snapshots(curr_date, prev_date, curr_path, prev_path, base_dir=None):
    """
    Compare two snapshots and produce a diff structure.

    Returns:
        diff_rows (list[dict]), where each row has:
            ticker
            name_prev
            name_curr
            weight_prev
            weight_curr
            weight_change
            is_new       (True if not in previous snapshot)
            is_removed   (True if not in current snapshot)
            big_move     (abs(change) >= 0.5)
    """
    prev_holdings = load_holdings_csv(prev_path)
    curr_holdings = load_holdings_csv(curr_path)

    prev_map = {h["ticker"]: h for h in prev_holdings}
    curr_map = {h["ticker"]: h for h in curr_holdings}

    all_tickers = set(prev_map.keys()) | set(curr_map.keys())
    diff_rows = []

    for ticker in all_tickers:
        prev_row = prev_map.get(ticker)
        curr_row = curr_map.get(ticker)

        weight_prev = prev_row["weight_pct"] if prev_row else 0.0
        weight_curr = curr_row["weight_pct"] if curr_row else 0.0
        name_prev = prev_row["name"] if prev_row else ""
        name_curr = curr_row["name"] if curr_row else ""

        weight_change = weight_curr - weight_prev
        is_new = prev_row is None and curr_row is not None
        is_removed = prev_row is not None and curr_row is None
        big_move = abs(weight_change) >= 0.5  # 0.5% threshold

        diff_rows.append(
            {
                "ticker": ticker,
                "name_prev": name_prev,
                "name_curr": name_curr,
                "weight_prev": weight_prev,
                "weight_curr": weight_curr,
                "weight_change": weight_change,
                "is_new": is_new,
                "is_removed": is_removed,
                "big_move": big_move,
            }
        )

    # Sort by absolute change in weight, descending
    diff_rows_sorted = sorted(
        diff_rows, key=lambda r: abs(r["weight_change"]), reverse=True
    )
    return diff_rows_sorted


def save_diff_csv(curr_date, prev_date, diff_rows, base_dir=None):
    """
    Save diff to CSV, including is_new, is_removed and big_move flags.
    """
    if base_dir is None:
        base_dir = get_base_dir()

    output_dir = base_dir / "data" / "pink"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"pink_diff_{curr_date.isoformat()}_vs_{prev_date.isoformat()}.csv"
    filepath = output_dir / filename

    fieldnames = [
        "ticker",
        "name_prev",
        "name_curr",
        "weight_prev",
        "weight_curr",
        "weight_change",
        "is_new",
        "is_removed",
        "big_move",
    ]

    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in diff_rows:
            writer.writerow(row)

    print(f"Saved diff CSV to:\n  {filepath}")
    return filepath


# -------------------------------------------------------------------
# Command wrappers
# -------------------------------------------------------------------

def cmd_scrape():
    """Fetch today's holdings and save snapshot only."""
    print(f"Scraping PINK holdings from:\n  {URL}")
    as_of_date, holdings = fetch_pink_holdings(debug=False)
    print(f"Successfully scraped {len(holdings)} holdings (As of {as_of_date})")

    # Preview top 5 by weight
    holdings_sorted = sorted(
        holdings, key=lambda h: h["weight_pct"], reverse=True
    )
    print("\nTop 5 holdings by weight:")
    print(f"{'Ticker':<8} {'Name':<40} {'Quantity':>15} {'Weight %':>9}")
    print("-" * 80)
    for h in holdings_sorted[:5]:
        print(
            f"{h['ticker']:<8} "
            f"{h['name']:<40.40} "
            f"{h['quantity']:>15,.2f} "
            f"{h['weight_pct']:>8.2f}"
        )

    save_snapshot_csv(as_of_date, holdings)


def cmd_compare(curr_date_str=None, prev_date_str=None):
    """
    Compare two snapshots.

    Default behaviour (no dates passed):
        compare latest snapshot vs previous snapshot.
    """
    base_dir = get_base_dir()
    snapshot_map = find_snapshot_files(base_dir=base_dir)

    if len(snapshot_map) < 2:
        print("Need at least two snapshot CSVs in data/pink to run a comparison.")
        return

    curr_date, prev_date, curr_path, prev_path = choose_dates_for_compare(
        snapshot_map, curr_date_str, prev_date_str
    )

    print(
        f"Comparing snapshots:\n"
        f"  Current:  {curr_date}  ({curr_path.name})\n"
        f"  Previous: {prev_date}  ({prev_path.name})"
    )

    diff_rows = compare_snapshots(curr_date, prev_date, curr_path, prev_path, base_dir)

    # Simple summary
    new_count = sum(1 for r in diff_rows if r["is_new"])
    removed_count = sum(1 for r in diff_rows if r["is_removed"])
    big_moves_count = sum(1 for r in diff_rows if r["big_move"])

    print(
        f"\nSummary:\n"
        f"  New positions:      {new_count}\n"
        f"  Removed positions:  {removed_count}\n"
        f"  Big moves (>=0.5%): {big_moves_count}"
    )

    # Preview top 15 biggest moves
    print("\nTop 15 by absolute weight change:")
    print(
        f"{'Ticker':<8} {'w_prev%':>9} {'w_curr%':>9} "
        f"{'Δ%':>9} {'NEW':>5} {'REM':>5} {'BIG':>5}"
    )
    print("-" * 70)
    for r in diff_rows[:15]:
        print(
            f"{r['ticker']:<8} "
            f"{r['weight_prev']:>9.2f} "
            f"{r['weight_curr']:>9.2f} "
            f"{r['weight_change']:>9.2f} "
            f"{('Y' if r['is_new'] else ''):>5} "
            f"{('Y' if r['is_removed'] else ''):>5} "
            f"{('Y' if r['big_move'] else ''):>5}"
        )

    # Save full diff CSV
    save_diff_csv(curr_date, prev_date, diff_rows, base_dir=base_dir)


# -------------------------------------------------------------------
# Main dispatcher
# -------------------------------------------------------------------

def main():
    """
    Command-line dispatcher.

    Behaviour:
      - No arguments (e.g. in Spyder):
            1) scrape latest snapshot
            2) compare latest snapshot vs previous snapshot
      - 'scrape':
            only scrape & save snapshot
      - 'compare':
            compare latest vs previous snapshot
      - 'compare YYYY-MM-DD':
            compare that date vs previous snapshot before it
      - 'compare YYYY-MM-DD YYYY-MM-DD':
            compare explicit pair
    """
    args = sys.argv[1:]

    # If run without arguments (e.g. from Spyder's %runfile),
    # scrape today's data and then run a comparison (latest vs previous).
    if not args:
        print("No command-line arguments detected. Defaulting to:")
        print("  1) scrape latest snapshot")
        print("  2) compare latest snapshot vs previous snapshot\n")

        cmd_scrape()
        print("\nNow comparing latest snapshot vs previous snapshot...\n")
        cmd_compare()
        return

    cmd = args[0].lower()

    if cmd == "scrape":
        cmd_scrape()
    elif cmd == "compare":
        curr = args[1] if len(args) >= 2 else None
        prev = args[2] if len(args) >= 3 else None
        cmd_compare(curr, prev)
    else:
        print(f"Unknown command: {cmd}")
        print("Usage:")
        print("  python Pinkscraper1.py scrape")
        print("  python Pinkscraper1.py compare")
        print("  python Pinkscraper1.py compare YYYY-MM-DD")
        print("  python Pinkscraper1.py compare YYYY-MM-DD YYYY-MM-DD")


if __name__ == "__main__":
    main()
