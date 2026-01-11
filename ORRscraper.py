#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 13:34:02 2026

@author: christopher
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import re
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

URL = "https://militiaetf.com/"

DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
CUSIP_RE = re.compile(r"^[0-9A-Z]{9}$")


def base_dir() -> Path:
    return Path(__file__).resolve().parent


def norm_header(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def parse_float_cell(s: str) -> float:
    s = s.strip()
    s = s.replace("$", "").replace("%", "")
    s = s.replace(",", "")
    if s == "":
        return 0.0
    return float(s)


def parse_date_mmddyyyy(s: str):
    return datetime.strptime(s.strip(), "%m/%d/%Y").date()


# ----------------------------
# Robust holdings scrape
# ----------------------------

def fetch_orr_holdings(debug: bool = False):
    """
    Returns:
      as_of_date (date): max EFFECTIVE_DATE found
      holdings (list[dict])
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    r = requests.get(URL, headers=headers, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # --- A) Prefer parsing a real HTML table (most stable) ---
    tables = soup.find_all("table")
    for t in tables:
        rows = t.find_all("tr")
        if not rows:
            continue

        header_cells = rows[0].find_all(["th", "td"])
        header_texts = [c.get_text(" ", strip=True) for c in header_cells]
        hnorm = [norm_header(x) for x in header_texts]

        # Identify holdings table by presence of these columns
        if not any("ticker" == h for h in hnorm):
            continue
        if not any("effective" in h and "date" in h for h in hnorm):
            continue

        def idx_of(predicate):
            for i, h in enumerate(hnorm):
                if predicate(h):
                    return i
            return None

        i_ticker = idx_of(lambda h: h == "ticker")
        i_name = idx_of(lambda h: h == "name")
        i_cusip = idx_of(lambda h: "cusip" in h)
        i_shares = idx_of(lambda h: "shares" in h)
        i_price = idx_of(lambda h: h == "price")
        i_mv = idx_of(lambda h: "market" in h and "value" in h)
        i_pct = idx_of(lambda h: ("net" in h and "assets" in h) or ("of_net_assets" in h))
        i_eff = idx_of(lambda h: "effective" in h and "date" in h)

        needed = [i_ticker, i_name, i_shares, i_price, i_mv, i_pct, i_eff]
        if any(x is None for x in needed):
            continue

        holdings = []
        for tr in rows[1:]:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if not cells:
                continue
            # pad to header length if needed
            if len(cells) < len(hnorm):
                cells = cells + [""] * (len(hnorm) - len(cells))

            eff_raw = cells[i_eff].strip()
            if not DATE_RE.match(eff_raw):
                continue

            try:
                eff = parse_date_mmddyyyy(eff_raw)
                shares = parse_float_cell(cells[i_shares])
                price = parse_float_cell(cells[i_price])
                mv = parse_float_cell(cells[i_mv])
                pct = parse_float_cell(cells[i_pct])
            except ValueError:
                continue

            ticker = cells[i_ticker].strip()
            name = cells[i_name].strip()
            cusip = cells[i_cusip].strip() if i_cusip is not None else ""

            holdings.append({
                "ticker": ticker,
                "name": name,
                "cusip": cusip,
                "shares": shares,
                "price": price,
                "market_value_mm": mv,
                "pct_net_assets": pct,
                "effective_date": eff,
            })

        if holdings:
            as_of = max(h["effective_date"] for h in holdings)
            return as_of, holdings

    # --- B) Fallback: text-based parsing (handles header split across lines) ---
    text = soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Find "Fund Holdings"
    start = None
    for i, ln in enumerate(lines):
        if ln in ("Fund Holdings", "## Fund Holdings"):
            start = i
            break
    if start is None:
        if debug:
            print("DEBUG: Couldn't find 'Fund Holdings'. First 120 lines:")
            for ln in lines[:120]:
                print(repr(ln))
        raise RuntimeError("Could not find Fund Holdings section.")

    # Detect header either as one line OR as a multi-line header block
    header_line_idx = None
    header_block_idx = None

    for i in range(start, min(start + 80, len(lines))):
        ln = lines[i]
        if ("EFFECTIVE_DATE" in ln) and ("Ticker" in ln) and ("Shares" in ln):
            header_line_idx = i
            break

    if header_line_idx is None:
        # Try multiline header block
        expected = ["Ticker", "Name", "CUSIP", "Shares", "Price", "Market Value ($mm)", "% of Net Assets", "EFFECTIVE_DATE"]
        for i in range(start, len(lines) - len(expected)):
            window = lines[i:i + len(expected)]
            if window == expected:
                header_block_idx = i
                break

    if header_line_idx is None and header_block_idx is None:
        if debug:
            print("DEBUG: Could not find holdings header. Lines near holdings:")
            for ln in lines[start:start + 120]:
                print(repr(ln))
        raise RuntimeError("Could not find holdings header (line or block).")

    holdings = []

    def stop_line(ln: str) -> bool:
        return (
            ln.startswith("Fund holdings and allocations are subject to change")
            or ln.startswith("#### Important Information")
            or ln.startswith("Important Information")
            or ln.startswith("## ")
        )

    # Case 1: header is a single line and each holding is usually a single line
    if header_line_idx is not None:
        for ln in lines[header_line_idx + 1:]:
            if stop_line(ln):
                break
            parts = ln.split()
            if len(parts) < 6:
                continue
            if not DATE_RE.match(parts[-1]):
                continue

            eff = parse_date_mmddyyyy(parts[-1])
            pct = parse_float_cell(parts[-2])
            mv = parse_float_cell(parts[-3])
            price = parse_float_cell(parts[-4])
            shares = parse_float_cell(parts[-5])

            rest = parts[:-5]
            if not rest:
                continue
            ticker = rest[0]
            mid = rest[1:]
            cusip = ""
            if mid and CUSIP_RE.match(mid[-1]):
                cusip = mid[-1]
                mid = mid[:-1]
            name = " ".join(mid).strip()

            holdings.append({
                "ticker": ticker,
                "name": name,
                "cusip": cusip,
                "shares": shares,
                "price": price,
                "market_value_mm": mv,
                "pct_net_assets": pct,
                "effective_date": eff,
            })

    # Case 2: header is a multi-line block and rows may be split across lines.
    else:
        i = header_block_idx + 8
        buf = []
        while i < len(lines):
            ln = lines[i]
            i += 1
            if stop_line(ln):
                break
            if ln in ("Ticker", "Name", "CUSIP", "Shares", "Price", "Market Value ($mm)", "% of Net Assets", "EFFECTIVE_DATE"):
                continue

            buf.append(ln)
            if DATE_RE.match(ln):
                # row ends at the effective date line
                if len(buf) >= 6:
                    eff = parse_date_mmddyyyy(buf[-1])
                    pct = parse_float_cell(buf[-2])
                    mv = parse_float_cell(buf[-3])
                    price = parse_float_cell(buf[-4])
                    shares = parse_float_cell(buf[-5])

                    rest = buf[:-5]
                    ticker = rest[0] if rest else ""
                    mid = rest[1:] if len(rest) >= 2 else []
                    cusip = ""
                    if mid and CUSIP_RE.match(mid[-1]):
                        cusip = mid[-1]
                        mid = mid[:-1]
                    name = " ".join(mid).strip()

                    if ticker:
                        holdings.append({
                            "ticker": ticker,
                            "name": name,
                            "cusip": cusip,
                            "shares": shares,
                            "price": price,
                            "market_value_mm": mv,
                            "pct_net_assets": pct,
                            "effective_date": eff,
                        })
                buf = []

    if not holdings:
        if debug:
            print("DEBUG: Parsed zero holdings. Showing lines near holdings start:")
            for ln in lines[start:start + 150]:
                print(repr(ln))
        raise RuntimeError("Could not parse any holdings rows.")

    as_of = max(h["effective_date"] for h in holdings)
    return as_of, holdings


# ----------------------------
# Snapshot + diff
# ----------------------------

def save_snapshot(as_of_date, holdings):
    outdir = base_dir() / "data" / "orr"
    outdir.mkdir(parents=True, exist_ok=True)

    path = outdir / f"orr_holdings_{as_of_date.isoformat()}.csv"
    if path.exists():
        print(f"Snapshot already exists: {path}")
        return path

    holdings_sorted = sorted(holdings, key=lambda h: h["pct_net_assets"], reverse=True)

    fields = ["ticker", "name", "cusip", "shares", "price", "market_value_mm", "pct_net_assets", "effective_date"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for h in holdings_sorted:
            row = dict(h)
            row["effective_date"] = h["effective_date"].isoformat()
            w.writerow(row)

    print(f"Saved snapshot: {path}")
    return path


def list_snapshots():
    d = base_dir() / "data" / "orr"
    if not d.exists():
        return {}
    m = {}
    for p in d.glob("orr_holdings_*.csv"):
        s = p.stem.replace("orr_holdings_", "")
        try:
            dt = datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            continue
        m[dt] = p
    return m


def load_snapshot(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "ticker": row["ticker"],
                "name": row["name"],
                "cusip": row.get("cusip", ""),
                "shares": float(row["shares"]),
                "price": float(row["price"]),
                "market_value_mm": float(row["market_value_mm"]),
                "pct_net_assets": float(row["pct_net_assets"]),
            })
    return rows


def pick_compare_dates(snap_map, curr=None, prev=None):
    if not snap_map:
        raise RuntimeError("No snapshots found in data/orr.")
    dates = sorted(snap_map.keys())

    def pd(s): return datetime.strptime(s, "%Y-%m-%d").date()

    if curr is None and prev is None:
        if len(dates) < 2:
            raise RuntimeError("Need at least 2 snapshots to compare.")
        curr_d, prev_d = dates[-1], dates[-2]
    elif curr is not None and prev is None:
        curr_d = pd(curr)
        if curr_d not in snap_map:
            raise RuntimeError(f"No snapshot for {curr_d}")
        idx = dates.index(curr_d)
        if idx == 0:
            raise RuntimeError("That snapshot has no earlier snapshot to compare.")
        prev_d = dates[idx - 1]
    else:
        curr_d, prev_d = pd(curr), pd(prev)

    return curr_d, prev_d, snap_map[curr_d], snap_map[prev_d]


def compare(curr_path, prev_path, big_move=0.5):
    prev = {r["ticker"]: r for r in load_snapshot(prev_path)}
    curr = {r["ticker"]: r for r in load_snapshot(curr_path)}
    all_tickers = set(prev) | set(curr)

    diff = []
    for t in all_tickers:
        p = prev.get(t)
        c = curr.get(t)

        pct_prev = p["pct_net_assets"] if p else 0.0
        pct_curr = c["pct_net_assets"] if c else 0.0
        pct_chg = pct_curr - pct_prev

        diff.append({
            "ticker": t,
            "name_prev": p["name"] if p else "",
            "name_curr": c["name"] if c else "",
            "pct_prev": pct_prev,
            "pct_curr": pct_curr,
            "pct_change": pct_chg,
            "is_new": p is None and c is not None,
            "is_removed": p is not None and c is None,
            "big_move": abs(pct_chg) >= big_move,
        })

    return sorted(diff, key=lambda r: abs(r["pct_change"]), reverse=True)


def save_diff(curr_date, prev_date, diff_rows):
    outdir = base_dir() / "data" / "orr"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"orr_diff_{curr_date.isoformat()}_vs_{prev_date.isoformat()}.csv"

    fields = ["ticker", "name_prev", "name_curr", "pct_prev", "pct_curr", "pct_change", "is_new", "is_removed", "big_move"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in diff_rows:
            w.writerow(r)

    print(f"Saved diff: {path}")
    return path


# ----------------------------
# CLI
# ----------------------------

def cmd_scrape():
    print(f"Scraping ORR holdings from:\n  {URL}")
    as_of, holdings = fetch_orr_holdings(debug=False)
    print(f"Scraped {len(holdings)} rows (EFFECTIVE_DATE {as_of})")

    top = sorted(holdings, key=lambda h: h["pct_net_assets"], reverse=True)[:10]
    print("\nTop 10 by % net assets:")
    print(f"{'Ticker':<10} {'%Net':>8} {'Shares':>14} {'Price':>10}  Name")
    print("-" * 90)
    for h in top:
        print(f"{h['ticker']:<10} {h['pct_net_assets']:>8.2f} {h['shares']:>14,.0f} {h['price']:>10.2f}  {h['name'][:45]}")

    save_snapshot(as_of, holdings)


def cmd_compare(curr=None, prev=None):
    snaps = list_snapshots()
    if len(snaps) < 2:
        print("Need at least two snapshots in data/orr to compare.")
        return

    curr_d, prev_d, curr_p, prev_p = pick_compare_dates(snaps, curr, prev)
    print(f"Comparing:\n  Current:  {curr_d} ({curr_p.name})\n  Previous: {prev_d} ({prev_p.name})")

    diff = compare(curr_p, prev_p, big_move=0.5)

    new_n = sum(1 for r in diff if r["is_new"])
    rem_n = sum(1 for r in diff if r["is_removed"])
    big_n = sum(1 for r in diff if r["big_move"])
    print(f"\nSummary:\n  New: {new_n}\n  Removed: {rem_n}\n  Big moves (>=0.5%): {big_n}")

    print("\nTop 20 by abs % change:")
    print(f"{'Ticker':<10} {'%Prev':>8} {'%Curr':>8} {'Δ%':>8} {'NEW':>5} {'REM':>5}")
    print("-" * 55)
    for r in diff[:20]:
        print(f"{r['ticker']:<10} {r['pct_prev']:>8.2f} {r['pct_curr']:>8.2f} {r['pct_change']:>8.2f} {('Y' if r['is_new'] else ''):>5} {('Y' if r['is_removed'] else ''):>5}")

    save_diff(curr_d, prev_d, diff)


def main():
    args = sys.argv[1:]
    if not args:
        print("No args → scrape then compare latest vs previous\n")
        cmd_scrape()
        print("\nNow comparing latest vs previous...\n")
        cmd_compare()
        return

    if args[0].lower() == "scrape":
        cmd_scrape()
    elif args[0].lower() == "compare":
        curr = args[1] if len(args) >= 2 else None
        prev = args[2] if len(args) >= 3 else None
        cmd_compare(curr, prev)
    else:
        print("Usage:")
        print("  python orr_tracker.py scrape")
        print("  python orr_tracker.py compare")
        print("  python orr_tracker.py compare YYYY-MM-DD")
        print("  python orr_tracker.py compare YYYY-MM-DD YYYY-MM-DD")


if __name__ == "__main__":
    main()
