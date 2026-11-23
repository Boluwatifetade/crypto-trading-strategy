"""
Threaded OHLCV downloader for 5-minute candles on Bybit.

Downloads 1 year of 5m data for BTC, ETH, SOL, DOGE, XRP in parallel
threads, each with its own CCXT session and persistent HTTPS connection.
Auto-resumes from existing CSV files if interrupted.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "DOGE/USDT:USDT",
    "XRP/USDT:USDT",
]


def csv_path(symbol: str, timeframe: str = "5m", exchange: str = "bybit") -> Path:
    clean = symbol.split(":")[0].replace("/", "")
    return DATA_DIR / f"{exchange}_{clean}_{timeframe}.csv"


def fetch_symbol_thread(
    symbol: str,
    timeframe: str = "5m",
    since_date: str = "2025-10-20",
    limit: int = 1000,
) -> int:
    DATA_DIR.mkdir(exist_ok=True)
    out_file = csv_path(symbol, timeframe)
    exchange = ccxt.bybit({"enableRateLimit": True, "timeout": 20000})

    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    now_ms = exchange.milliseconds()
    target_since_ms = exchange.parse8601(f"{since_date}T00:00:00Z")

    existing_rows = []
    start_ms = target_since_ms

    if out_file.exists():
        try:
            df_existing = pd.read_csv(out_file, parse_dates=["timestamp"])
            if not df_existing.empty:
                last_ts = int(df_existing["timestamp"].iloc[-1].timestamp() * 1000)
                if last_ts > target_since_ms:
                    start_ms = last_ts + tf_ms
                    print(f"[{symbol}] Resuming from {exchange.iso8601(start_ms)} ({len(df_existing)} rows already saved)", flush=True)
                    existing_rows = df_existing.to_dict("records")
        except Exception as e:
            print(f"[{symbol}] Could not read existing file: {e}", flush=True)

    rows: list = []
    cursor = start_ms
    batch_count = 0

    while cursor + tf_ms < now_ms:
        batch = None
        for attempt in range(5):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
                break
            except Exception as e:
                wait_sec = 2 ** attempt
                print(f"[{symbol}] Network error ({e}), retrying in {wait_sec}s...", flush=True)
                time.sleep(wait_sec)

        if not batch:
            break

        rows.extend(batch)
        last_batch_ts = batch[-1][0]
        if last_batch_ts < cursor:
            break
        cursor = last_batch_ts + tf_ms
        batch_count += 1

        if batch_count % 10 == 0:
            total_so_far = len(existing_rows) + len(rows)
            print(f"[{symbol}] {total_so_far} candles (reached {exchange.iso8601(last_batch_ts)})", flush=True)

        time.sleep(exchange.rateLimit / 1000.0)

    if rows:
        df_new = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True)
        if existing_rows:
            df_old = pd.DataFrame(existing_rows)
            df_old["timestamp"] = pd.to_datetime(df_old["timestamp"], utc=True)
            df = pd.concat([df_old, df_new]).drop_duplicates("timestamp").sort_values("timestamp")
        else:
            df = df_new.drop_duplicates("timestamp").sort_values("timestamp")
    elif existing_rows:
        df = pd.DataFrame(existing_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        print(f"[{symbol}] No data returned.", flush=True)
        return 0

    df = df.set_index("timestamp")
    df.to_csv(out_file)
    print(f"[{symbol}] COMPLETE: {len(df)} candles saved to {out_file.name}", flush=True)
    return len(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--since", default="2025-10-20", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    print(f"Fetching 1 year of {args.timeframe} data for {len(args.symbols)} pairs using {args.workers} threads...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_symbol_thread, sym, args.timeframe, args.since): sym
            for sym in args.symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                count = future.result()
                print(f"--> {sym} finished ({count} bars)", flush=True)
            except Exception as e:
                print(f"--> {sym} failed: {e}", flush=True)

    print(f"All downloads finished in {round((time.time() - t0) / 60.0, 2)} minutes.", flush=True)


if __name__ == "__main__":
    main()
