"""
Download historical OHLCV candles via CCXT (no API keys needed).

Defaults to Bybit USDT perpetuals, which is what Tadex trades.

Examples:
    python fetch_data.py
    python fetch_data.py --symbol BTC/USDT:USDT --timeframe 1h --since 2025-01-01
    python fetch_data.py --symbol DOGE/USDT --exchange binance   # spot fallback
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def csv_path(symbol: str, timeframe: str, exchange: str = "bybit") -> Path:
    """data/bybit_DOGEUSDT_1h.csv  (':USDT' perp suffix is dropped)."""
    clean = symbol.split(":")[0].replace("/", "")
    return DATA_DIR / f"{exchange}_{clean}_{timeframe}.csv"


def fetch_historical_data(
    symbol: str = "DOGE/USDT:USDT",
    timeframe: str = "1h",
    since: str = "2025-01-01",
    exchange_id: str = "bybit",
    limit: int = 1000,
) -> pd.DataFrame:
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()
    if symbol not in exchange.markets:
        sys.exit(f"Symbol {symbol!r} not found on {exchange_id}. "
                 f"Perps use 'BASE/USDT:USDT', spot uses 'BASE/USDT'.")

    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")
    now_ms = exchange.milliseconds()

    print(f"Fetching {symbol} {timeframe} from {exchange_id} since {since} ...")
    rows: list = []
    cursor = since_ms
    while cursor < now_ms:
        for attempt in range(5):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  error ({e}); retrying in {wait}s")
                time.sleep(wait)
        else:
            sys.exit("Giving up after repeated network errors.")

        if not batch:
            break
        rows.extend(batch)
        time.sleep(exchange.rateLimit / 1000.0)
        last_ts = batch[-1][0]
        print(f"  {len(rows):>6} candles (up to {exchange.iso8601(last_ts)})")
        if last_ts < cursor:  # exchange returned stale data - avoid infinite loop
            break
        cursor = last_ts + tf_ms

    if not rows:
        sys.exit("No data returned.")

    df = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    # Drop the still-forming candle so backtests only see closed bars
    df = df[df["timestamp"] + tf_ms <= now_ms]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    # Gap check (exchange maintenance etc.) - informational only
    expected = pd.Timedelta(milliseconds=tf_ms)
    gaps = (df.index.to_series().diff() > expected).sum()
    if gaps:
        print(f"  note: {gaps} gap(s) in the data (exchange downtime / delisting)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="DOGE/USDT:USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--since", default="2025-01-01", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--exchange", default="bybit")
    args = ap.parse_args()

    df = fetch_historical_data(args.symbol, args.timeframe, args.since, args.exchange)
    DATA_DIR.mkdir(exist_ok=True)
    out = csv_path(args.symbol, args.timeframe, args.exchange)
    df.to_csv(out)
    print(f"Saved {len(df)} rows ({df.index[0]} -> {df.index[-1]}) to {out}")


if __name__ == "__main__":
    main()
