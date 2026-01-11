#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 15:41:22 2025

@author: christopher
"""

import asyncio
import re
from pathlib import Path
from datetime import datetime
import nest_asyncio
import pandas as pd

from playwright.async_api import async_playwright


# =========================
# CONFIG
# =========================
URL = "https://tweedybrowne.filepoint.live/allocation-of-investments"
DOWNLOADS_DIR = Path.home() / "Downloads"
COPY_RE = re.compile(r"copy\s+holdings", re.I)


# =========================
# Spyder-friendly coroutine runner (NO un-awaited coroutine warnings)
# =========================
def run_coro(coro):
    try:
        asyncio.get_running_loop()  # raises if no running loop
        # Spyder/IPython: loop is already running
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # Normal script execution: no loop is running
        return asyncio.run(coro)


# =========================
# 1) DOWNLOAD STEP (Playwright)
#   - Succeeds if either:
#       A) browser download event happens, OR
#       B) we detect a CSV response and save it
#   - NO clipboard fallback
# =========================
def _pick_filename(preferred: str, clicked_text: str) -> str:
    # Keep the site-provided filename if it looks right
    if preferred and preferred.lower().endswith(".csv"):
        return preferred

    # Otherwise make a decent filename from the date in the button text
    # Example clicked_text: "COPY HOLDINGS 12/27/2024"
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", clicked_text)
    if m:
        mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
        try:
            d = datetime(int(yyyy), int(mm), int(dd))
            return f"TweedyBrowne_Holdings_{d.strftime('%Y%m%d')}.csv"
        except Exception:
            pass

    return "TweedyBrowne_Holdings_download.csv"


async def download_holdings_csv(downloads_dir: Path = DOWNLOADS_DIR, headless: bool = True) -> Path:
    downloads_dir.mkdir(parents=True, exist_ok=True)

    csv_bytes_holder = {"body": None, "suggested": None, "clicked_text": "COPY HOLDINGS"}
    csv_found = asyncio.Event()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        async def maybe_capture_csv(response):
            try:
                if not response.ok:
                    return
                headers = {k.lower(): v for k, v in (response.headers or {}).items()}
                ct = headers.get("content-type", "").lower()
                cd = headers.get("content-disposition", "").lower()
                url = response.url.lower()

                looks_like_csv = (
                    "text/csv" in ct
                    or "application/csv" in ct
                    or (".csv" in url)
                    or ("attachment" in cd and "csv" in cd)
                )
                if not looks_like_csv:
                    return

                body = await response.body()
                if not body or len(body) < 20:
                    return

                # sanity-check: CSV-like header
                prefix = body[:300].decode("utf-8", errors="ignore").lower()
                if "ticker" not in prefix:
                    return

                # Try filename from Content-Disposition if present
                m = re.search(r'filename="?([^"]+)"?', headers.get("content-disposition", ""), re.I)
                suggested = m.group(1) if m else None

                csv_bytes_holder["body"] = body
                csv_bytes_holder["suggested"] = suggested
                csv_found.set()
            except Exception:
                # ignore noisy parsing errors
                return

        # Hook response listener
        page.on("response", lambda resp: asyncio.create_task(maybe_capture_csv(resp)))

        try:
            await page.goto(URL, wait_until="networkidle")

            locator = page.get_by_text(COPY_RE).first
            await locator.wait_for(state="visible", timeout=30000)
            await locator.scroll_into_view_if_needed()

            clicked_text = (await locator.inner_text()).strip()
            csv_bytes_holder["clicked_text"] = clicked_text
            print(f"Clicking: {clicked_text}")

            # Wait for either a real download OR a captured CSV response
            download_task = asyncio.create_task(page.wait_for_event("download", timeout=60000))
            csv_task = asyncio.create_task(csv_found.wait())

            # Click after tasks are set up
            await locator.click(force=True)

            done, pending = await asyncio.wait(
                {download_task, csv_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=65,
            )

            for t in pending:
                t.cancel()

            # Case A: real download event
            if download_task in done and not download_task.cancelled():
                download = download_task.result()
                suggested = download.suggested_filename or ""
                filename = _pick_filename(suggested, clicked_text)
                out_path = downloads_dir / filename

                # avoid overwrite
                if out_path.exists():
                    stem, suffix = out_path.stem, out_path.suffix
                    i = 1
                    while (downloads_dir / f"{stem}_{i}{suffix}").exists():
                        i += 1
                    out_path = downloads_dir / f"{stem}_{i}{suffix}"

                await download.save_as(out_path)
                print(f"Saved download to: {out_path}")
                return out_path

            # Case B: CSV response captured (no "download" event)
            if csv_task in done and csv_bytes_holder["body"] is not None:
                filename = _pick_filename(csv_bytes_holder["suggested"] or "", clicked_text)
                out_path = downloads_dir / filename

                # avoid overwrite
                if out_path.exists():
                    stem, suffix = out_path.stem, out_path.suffix
                    i = 1
                    while (downloads_dir / f"{stem}_{i}{suffix}").exists():
                        i += 1
                    out_path = downloads_dir / f"{stem}_{i}{suffix}"

                out_path.write_bytes(csv_bytes_holder["body"])
                print(f"Saved CSV response to: {out_path}")
                return out_path

            raise RuntimeError(
                "No file download was detected, and no CSV response was captured. "
                "If this control only copies text to clipboard, then there is no downloadable file to catch."
            )

        finally:
            await browser.close()


# =========================
# 2) YOUR COMPARISON CODE (unchanged except we pass downloads_dir)
# =========================
def find_latest_two_holdings_files(downloads_dir: Path | None = None):
    if downloads_dir is None:
        downloads_dir = Path.home() / "Downloads"

    files = list(downloads_dir.glob("TweedyBrowne_Holdings_*.csv"))
    if len(files) < 2:
        raise FileNotFoundError(
            f"Need at least two TweedyBrowne_Holdings_*.csv files in {downloads_dir}, "
            f"but found {len(files)}."
        )

    files_sorted = sorted(files, key=lambda p: p.stat().st_mtime)
    old_file = files_sorted[-2]
    new_file = files_sorted[-1]
    return old_file, new_file


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def find_column(df: pd.DataFrame, candidate_names: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidate_names:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def add_weight_pct_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    weight_col = find_column(df, ["% portfolio", "weight", "pct portfolio", "weighting"])
    if weight_col is None:
        raise RuntimeError("Could not find a weight column. " f"Available columns: {list(df.columns)}")

    df["weight_pct"] = (
        df[weight_col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    )
    df["weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce")

    max_w = df["weight_pct"].max()
    if pd.notna(max_w) and max_w <= 1.0 + 1e-6:
        df["weight_pct"] = df["weight_pct"] * 100.0

    df = df.dropna(subset=["weight_pct"])
    return df


def drop_cash_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ticker_col = find_column(df, ["ticker", "symbol"])
    name_col = find_column(df, ["company name", "name"])

    mask_cash = pd.Series(False, index=df.index)
    if ticker_col is not None:
        mask_cash |= df[ticker_col].astype(str).str.lower().str.contains("cash")
    if name_col is not None:
        mask_cash |= df[name_col].astype(str).str.lower().str.contains("cash")

    return df[~mask_cash]


def standardize_security_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    id_col = find_column(df, ["cusip/identifier", "ticker", "symbol", "company name", "name"])
    if id_col is None:
        raise RuntimeError(
            "Could not find a suitable identifier column "
            "(CUSIP/Identifier, Ticker, Symbol, Company Name, Name). "
            f"Columns: {list(df.columns)}"
        )

    df["sec_id"] = df[id_col].astype(str).str.strip()

    ticker_col = find_column(df, ["ticker", "symbol"])
    if ticker_col is not None:
        df["display_ticker"] = df[ticker_col].astype(str).str.strip()
    else:
        df["display_ticker"] = df["sec_id"]

    name_col = find_column(df, ["company name", "name"])
    if name_col is not None:
        df["display_name"] = df[name_col].astype(str).str.strip()
    else:
        df["display_name"] = df["sec_id"]

    return df[["sec_id", "display_ticker", "display_name", "weight_pct"]]


def prepare_holdings(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    df = add_weight_pct_column(df)
    df = drop_cash_rows(df)
    df = standardize_security_ids(df)
    return df


def compare_holdings(old_df: pd.DataFrame, new_df: pd.DataFrame, top_n_moves: int = 15):
    old_df = old_df.set_index("sec_id")
    new_df = new_df.set_index("sec_id")

    all_ids = old_df.index.union(new_df.index)
    old_w = old_df["weight_pct"].reindex(all_ids).fillna(0.0)
    new_w = new_df["weight_pct"].reindex(all_ids).fillna(0.0)
    delta = new_w - old_w
    abs_delta = delta.abs()

    new_ticker = new_df["display_ticker"].reindex(all_ids)
    old_ticker = old_df["display_ticker"].reindex(all_ids)
    ticker = new_ticker.combine_first(old_ticker)

    new_name = new_df["display_name"].reindex(all_ids)
    old_name = old_df["display_name"].reindex(all_ids)
    name = new_name.combine_first(old_name)

    summary = pd.DataFrame(
        {
            "ticker": ticker,
            "name": name,
            "old_weight_pct": old_w,
            "new_weight_pct": new_w,
            "delta_pct": delta,
            "abs_delta": abs_delta,
        }
    )

    top_moves = summary.sort_values("abs_delta", ascending=False).head(top_n_moves).copy()
    new_positions = summary[(summary["old_weight_pct"] == 0) & (summary["new_weight_pct"] > 0)].sort_values(
        "new_weight_pct", ascending=False
    )

    for col in ["old_weight_pct", "new_weight_pct", "delta_pct"]:
        top_moves[col] = top_moves[col].round(2)
    new_positions["new_weight_pct"] = new_positions["new_weight_pct"].round(2)

    print("\nTop 15 largest moves in portfolio weights (percentage points):\n")
    print(top_moves[["ticker", "name", "old_weight_pct", "new_weight_pct", "delta_pct"]].to_string(index=False))

    if not new_positions.empty:
        print("\nNew positions opened (not in previous file):\n")
        print(new_positions[["ticker", "name", "new_weight_pct"]].to_string(index=False))
    else:
        print("\nNo new positions were opened between these two dates.\n")


# =========================
# MAIN: download, then compare latest two
# =========================
def main():
    downloaded = run_coro(download_holdings_csv(DOWNLOADS_DIR, headless=True))
    print(f"\nDownloaded: {downloaded}\n")

    old_file, new_file = find_latest_two_holdings_files(DOWNLOADS_DIR)

    print("Comparing Tweedy Browne holdings CSVs:")
    print(f"  Older file: {old_file}")
    print(f"  Newer file: {new_file}\n")

    old_df = prepare_holdings(old_file)
    new_df = prepare_holdings(new_file)
    compare_holdings(old_df, new_df, top_n_moves=15)


if __name__ == "__main__":
    main()
