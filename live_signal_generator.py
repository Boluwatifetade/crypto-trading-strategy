"""
Live strategy signal generator -> Tadex (TESTNET forward-testing).

Every candle close it:
  1. fetches recent CLOSED candles from Bybit (public data, no keys),
  2. runs the exact same rules as the backtest (strategy_core.py),
  3. tracks a "virtual position" per symbol so - like the backtest - only one
     trade is open at a time (resolved when price touches SL or TP),
  4. formats a Cosmas-style signal, verifies it through Tadex's real parser,
  5. (only with --insert) inserts it into public.signals for the Tadex dispatcher.

Safety gates for --insert:
  - TADEX_TARGET_ENV=testnet must be set in .env
  - a PASSING backtest verdict (results/*_verdict.json) with identical params
    must exist for each symbol (override: --allow-unvalidated, testnet only)
  - the provider id must exist in public.providers

Examples:
    python live_signal_generator.py --once                 # dry run, one evaluation
    python live_signal_generator.py                        # dry run, loop forever
    python live_signal_generator.py --insert               # live inserts (testnet)
    python live_signal_generator.py --symbols DOGE/USDT:USDT BTC/USDT:USDT --mode ema_cross
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

from fetch_data import csv_path
from strategy_core import LONG, MODES, SHORT, StrategyParams, compute_signals, stop_and_target
from tadex_bridge import TadexSignalSink, format_signal, verify_with_tadex

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).parent
STATE_DIR = ROOT / "state"
STATE_FILE = STATE_DIR / "live_state.json"
TRADE_LOG = STATE_DIR / "virtual_trades.csv"
RESULTS_DIR = ROOT / "results"
CLOSE_DELAY_SEC = 20  # wait a bit after candle close so the exchange has finalised it

log = logging.getLogger("live_signal_generator")


# ---------------------------------------------------------------------------
# State (survives restarts -> no duplicate signals)
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict):
    STATE_DIR.mkdir(exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def log_virtual_trade(row: dict):
    STATE_DIR.mkdir(exist_ok=True)
    new = not TRADE_LOG.exists()
    with TRADE_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def fetch_closed_candles(exchange, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=min(bars + 1, 1000))
    df = pd.DataFrame(ohlcv, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df = df[df["ts"] + tf_ms <= exchange.milliseconds()]  # closed candles only
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Verdict gate
# ---------------------------------------------------------------------------
def check_verdict(symbol: str, timeframe: str, exchange_id: str, params: StrategyParams) -> tuple[bool, str]:
    vfile = RESULTS_DIR / f"{csv_path(symbol, timeframe, exchange_id).stem}_{params.mode}_verdict.json"
    if not vfile.exists():
        return False, f"no backtest verdict at {vfile.name} - run strategy.py first"
    doc = json.loads(vfile.read_text(encoding="utf-8"))
    if doc.get("params") != params.to_dict():
        return False, f"{vfile.name} was produced with different params: {doc.get('params')}"
    if not doc.get("passed"):
        return False, f"{vfile.name} FAILED: {doc.get('reasons')}"
    return True, f"{vfile.name} PASSED"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def resolve_virtual_position(pos: dict, df: pd.DataFrame) -> Optional[dict]:
    """Return a closed-trade record if SL/TP was touched after entry, else None."""
    after = df[df["ts"] > pos["signal_bar_ts"]]
    for _, bar in after.iterrows():
        if pos["side"] == LONG:
            hit_sl, hit_tp = bar["Low"] <= pos["sl"], bar["High"] >= pos["tp"]
        else:
            hit_sl, hit_tp = bar["High"] >= pos["sl"], bar["Low"] <= pos["tp"]
        if hit_sl or hit_tp:
            # Both inside one candle: assume the worst (SL) - conservative
            outcome = "SL" if hit_sl else "TP"
            return {**pos, "outcome": outcome, "closed_bar_ts": int(bar["ts"]),
                    "closed_at": pd.to_datetime(bar["ts"], unit="ms", utc=True).isoformat()}
    return None


def evaluate_symbol(exchange, symbol: str, args, params: StrategyParams, state: dict,
                    sink: Optional[TadexSignalSink], tadex_repo: Optional[Path]):
    key = f"{symbol}|{args.timeframe}|{params.mode}"
    st = state.setdefault(key, {})
    df = fetch_closed_candles(exchange, symbol, args.timeframe, bars=params.warmup_bars * 3 + 50)
    if len(df) < params.warmup_bars * 2:
        log.warning("%s: only %d candles, skipping", symbol, len(df))
        return
    last = df.iloc[-1]
    last_ts = int(last["ts"])
    bar_time = pd.to_datetime(last_ts, unit="ms", utc=True)

    # 1) Resolve an open virtual position first
    pos = st.get("open_position")
    if pos:
        closed = resolve_virtual_position(pos, df)
        if closed:
            log.info("%s: virtual %s trade closed at %s (%s)", symbol,
                     "LONG" if pos["side"] == LONG else "SHORT", closed["outcome"], closed["closed_at"])
            log_virtual_trade(closed)
            st["open_position"] = None
            # Same as the backtest: no new entry on the bar that closed the trade
            st["last_evaluated_ts"] = last_ts
            return
        log.info("%s: virtual position still open (SL %s / TP %s) - no new signal", symbol, pos["sl"], pos["tp"])
        st["last_evaluated_ts"] = last_ts
        return

    # 2) Don't evaluate the same bar twice (restarts / overlapping runs)
    if st.get("last_evaluated_ts") == last_ts:
        log.info("%s: bar %s already evaluated", symbol, bar_time)
        return

    sig = compute_signals(df[["Open", "High", "Low", "Close", "Volume"]], params)
    side = int(sig[-1])
    st["last_evaluated_ts"] = last_ts
    if side == 0:
        log.info("%s: no signal on bar %s (close %s)", symbol, bar_time, last["Close"])
        return

    # 3) Build the signal
    ref_price = float(last["Close"])
    sl_raw, tp_raw = stop_and_target(ref_price, side, params)
    sl = exchange.price_to_precision(symbol, sl_raw)
    tp = exchange.price_to_precision(symbol, tp_raw)
    symbol_id = exchange.market(symbol)["id"]
    text = format_signal(symbol_id, side, tp, sl, args.leverage)
    log.info("%s: %s SIGNAL on bar %s\n%s", symbol, "LONG" if side == LONG else "SHORT", bar_time, text)

    problems = verify_with_tadex(text, symbol_id, side, tp, sl, args.leverage, tadex_repo)
    if problems:
        if args.insert:
            log.error("%s: Tadex parser check FAILED - not inserting: %s", symbol, problems)
            return
        log.warning("%s: Tadex parser check: %s", symbol, problems)

    # 4) Insert (or simulate)
    signal_id = None
    if args.insert and sink is not None:
        meta = {"strategy": params.to_dict(), "symbol": symbol_id, "timeframe": args.timeframe,
                "signal_bar_ts": last_ts, "reference_price": ref_price}
        try:
            signal_id = sink.insert(text, meta)
            log.info("%s: inserted into public.signals id=%s", symbol, signal_id)
        except Exception as e:
            log.error("%s: insert failed: %s", symbol, e)
            return  # don't open a virtual position for a signal Tadex never received
    else:
        log.info("%s: DRY RUN - not inserted (use --insert)", symbol)

    st["open_position"] = {
        "symbol": symbol_id, "side": side, "reference_price": ref_price,
        "sl": float(sl), "tp": float(tp), "signal_bar_ts": last_ts,
        "opened_at": bar_time.isoformat(), "signal_id": signal_id, "inserted": bool(signal_id),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def seconds_until_next_close(exchange, timeframe: str) -> float:
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    now = exchange.milliseconds()
    next_close = (now // tf_ms + 1) * tf_ms
    return (next_close - now) / 1000 + CLOSE_DELAY_SEC


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["DOGE/USDT:USDT"])
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--exchange", default="bybit", help="market-data source (must match the backtest data)")
    ap.add_argument("--mode", choices=MODES, default="rsi_reversal")
    ap.add_argument("--ema-fast", type=int, default=50)
    ap.add_argument("--ema-slow", type=int, default=200)
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--rsi-oversold", type=float, default=30.0)
    ap.add_argument("--rsi-overbought", type=float, default=70.0)
    ap.add_argument("--sl-pct", type=float, default=0.02)
    ap.add_argument("--risk-reward", type=float, default=2.0)
    ap.add_argument("--use-adx-filter", action="store_true", help="enable ADX and DI directional filter")
    ap.add_argument("--adx-period", type=int, default=14)
    ap.add_argument("--adx-thresh", type=float, default=20.0)
    ap.add_argument("--leverage", type=int, default=10, help="leverage written into the signal text")
    ap.add_argument("--once", action="store_true", help="evaluate once and exit (for Task Scheduler/cron)")
    ap.add_argument("--insert", action="store_true", help="actually insert into Tadex public.signals")
    ap.add_argument("--allow-unvalidated", action="store_true",
                    help="skip the backtest-verdict gate (testnet only, not recommended)")
    ap.add_argument("--env-file", default=str(ROOT / ".env"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(ROOT / "live_signal_generator.log", encoding="utf-8")])
    if load_dotenv and Path(args.env_file).exists():
        load_dotenv(args.env_file)

    params = StrategyParams(
        mode=args.mode,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        rsi_period=args.rsi_period,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        sl_pct=args.sl_pct,
        risk_reward=args.risk_reward,
        adx_period=args.adx_period,
        adx_threshold=args.adx_thresh,
        use_adx_filter=args.use_adx_filter,
    )
    repo = os.getenv("TADEX_REPO_PATH")
    tadex_repo = Path(repo) if repo else None

    sink = None
    if args.insert:
        if os.getenv("TADEX_TARGET_ENV", "").lower() != "testnet":
            sys.exit("Refusing to insert: set TADEX_TARGET_ENV=testnet in .env. "
                     "This bot is for testnet forward-testing only.")
        if tadex_repo is None:
            sys.exit("Refusing to insert: TADEX_REPO_PATH must be set so signals are verified by Tadex's parser.")
        for sym in args.symbols:
            ok, msg = check_verdict(sym, args.timeframe, args.exchange, params)
            if not ok and not args.allow_unvalidated:
                sys.exit(f"Refusing to insert for {sym}: {msg}")
            log.info("Verdict gate %s: %s", sym, msg if ok else f"BYPASSED ({msg})")
        sink = TadexSignalSink.from_env()
        provider = sink.check_provider()
        log.info("Inserting as provider '%s' (%s, status=%s)", provider.get("name"), provider["id"], provider.get("status"))
    else:
        log.info("DRY RUN mode - signals are printed, not inserted.")

    exchange = getattr(ccxt, args.exchange)({"enableRateLimit": True})
    exchange.load_markets()
    for sym in args.symbols:
        if sym not in exchange.markets:
            sys.exit(f"{sym} not found on {args.exchange}")

    state = load_state()
    while True:
        for sym in args.symbols:
            try:
                evaluate_symbol(exchange, sym, args, params, state, sink, tadex_repo)
            except ccxt.BaseError as e:
                log.error("%s: exchange error: %s", sym, e)
            except Exception:
                log.exception("%s: unexpected error", sym)
            finally:
                save_state(state)
        if args.once:
            break
        wait = seconds_until_next_close(exchange, args.timeframe)
        log.info("Sleeping %.0fs until next %s candle close", wait, args.timeframe)
        time.sleep(wait)


if __name__ == "__main__":
    main()
