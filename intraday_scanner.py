"""
Intraday Momentum Live Scanner (Top 20 Bybit USDT Perpetuals on 5-Minute Candles).

Runs every 5 minutes:
1. Dynamically discovers top 20 Bybit USDT perpetuals by 24h volume.
2. Evaluates ATR(14)/Close > 0.5%, Volume > 1.5x SMA(20), and 3-candle breakout.
3. Manages active positions with 0.8% SL, 2.4% TP (1:3 R:R), and 48-bar (4h) time stop.
4. Formats signals in canonical Tadex/Cosmas format and validates against Tadex parser.
5. Strictly enforces the Safety Gate:
   Refuses to insert to Tadex unless Profit Factor > 1.2 and Max Drawdown < 20%.
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
from typing import Dict, List, Optional

import ccxt
import pandas as pd

from intraday_momentum_core import (
    LONG,
    SHORT,
    IntradayMomentumParams,
    compute_indicators,
    compute_signals,
)
from tadex_bridge import TadexSignalSink, format_signal, verify_with_tadex

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT = Path(__file__).parent
STATE_DIR = ROOT / "state"
POSITIONS_FILE = STATE_DIR / "intraday_positions.json"
TRADES_FILE = STATE_DIR / "intraday_virtual_trades.csv"
RESULTS_DIR = ROOT / "results"
REPORT_FILE = RESULTS_DIR / "intraday_momentum_report.json"

log = logging.getLogger("intraday_scanner")


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------
def load_positions() -> Dict:
    if POSITIONS_FILE.exists():
        try:
            return json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_positions(positions: Dict):
    STATE_DIR.mkdir(exist_ok=True)
    tmp = POSITIONS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(positions, indent=2), encoding="utf-8")
    tmp.replace(POSITIONS_FILE)


def log_trade(trade: Dict):
    STATE_DIR.mkdir(exist_ok=True)
    new_file = not TRADES_FILE.exists()
    with TRADES_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trade.keys()))
        if new_file:
            w.writeheader()
        w.writerow(trade)


# ---------------------------------------------------------------------------
# Top 20 Pairs Discovery
# ---------------------------------------------------------------------------
def fetch_top_20_pairs(exchange: ccxt.bybit) -> List[str]:
    """Fetch top 20 active USDT perpetual pairs by 24h quote volume."""
    try:
        tickers = exchange.fetch_tickers()
        perps = [
            t for s, t in tickers.items()
            if s.endswith(":USDT") and t.get("quoteVolume") and t.get("active", True)
        ]
        top20 = sorted(perps, key=lambda x: x["quoteVolume"], reverse=True)[:20]
        symbols = [t["symbol"] for t in top20]
        return symbols
    except Exception as e:
        log.error("Failed to fetch top 20 pairs by volume: %s", e)
        return [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
            "DOGE/USDT:USDT", "XRP/USDT:USDT"
        ]


# ---------------------------------------------------------------------------
# Safety Gate Enforcement
# ---------------------------------------------------------------------------
def check_safety_gate() -> Tuple[bool, str]:
    """
    Safety Rule: Do not connect to Tadex unless Profit Factor > 1.2 and Max Drawdown < 20%.
    """
    if not REPORT_FILE.exists():
        return False, "No intraday backtest report found at results/intraday_momentum_report.json. Run intraday_backtest.py first."

    try:
        doc = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
        metrics = doc.get("portfolio_metrics", {})
        pf = metrics.get("profit_factor", 0.0)
        max_dd = metrics.get("max_drawdown_pct", 100.0)
        trades = metrics.get("total_trades", 0)

        if trades < 30:
            return False, f"Insufficient sample size: {trades} trades (< 30 trades)."
        if pf <= 1.2:
            return False, f"Profit Factor {pf} <= 1.2 threshold (Criteria failed)."
        if max_dd >= 20.0:
            return False, f"Max Drawdown {max_dd}% >= 20.0% threshold (Criteria failed)."

        return True, f"SAFETY GATE PASSED (Profit Factor: {pf}, Max Drawdown: {max_dd}%, Trades: {trades})"
    except Exception as e:
        return False, f"Could not verify safety gate: {e}"


# ---------------------------------------------------------------------------
# Position Management & Resolution
# ---------------------------------------------------------------------------
def resolve_open_position(
    symbol: str,
    pos: Dict,
    df: pd.DataFrame,
    p: IntradayMomentumParams,
) -> Optional[Dict]:
    """Check if open position hit SL, TP, or 48-bar (4h) time stop."""
    after = df[df["ts"] > pos["entry_bar_ts"]]
    if after.empty:
        return None

    pos_side = pos["side"]
    sl = pos["sl"]
    tp = pos["tp"]
    entry_bar_ts = pos["entry_bar_ts"]

    for idx, (_, bar) in enumerate(after.iterrows(), start=1):
        hit_sl = False
        hit_tp = False

        if pos_side == LONG:
            hit_sl = bar["Low"] <= sl
            hit_tp = bar["High"] >= tp
        else:
            hit_sl = bar["High"] >= sl
            hit_tp = bar["Low"] <= tp

        exit_reason = None
        exit_price = 0.0

        if hit_sl and hit_tp:
            exit_reason = "SL"
            exit_price = sl
        elif hit_sl:
            exit_reason = "SL"
            exit_price = sl
        elif hit_tp:
            exit_reason = "TP"
            exit_price = tp
        elif idx >= p.time_stop_bars:
            exit_reason = "TIME_STOP"
            exit_price = bar["Close"]

        if exit_reason:
            if pos_side == LONG:
                gross = (exit_price - pos["entry_price"]) / pos["entry_price"]
            else:
                gross = (pos["entry_price"] - exit_price) / pos["entry_price"]

            net = gross - (p.taker_fee * 2.0)
            return {
                "symbol": symbol,
                "side": "LONG" if pos_side == LONG else "SHORT",
                "entry_time": pos["opened_at"],
                "entry_price": pos["entry_price"],
                "exit_time": pd.to_datetime(bar["ts"], unit="ms", utc=True).isoformat(),
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "bars_held": idx,
                "gross_return_pct": round(gross * 100, 3),
                "net_return_pct": round(net * 100, 3),
            }

    return None


# ---------------------------------------------------------------------------
# Scanner Evaluation
# ---------------------------------------------------------------------------
def evaluate_symbol(
    exchange: ccxt.bybit,
    symbol: str,
    p: IntradayMomentumParams,
    positions: Dict,
    sink: Optional[TadexSignalSink],
    tadex_repo: Optional[Path],
    args,
):
    # Fetch last 40 closed candles
    tf_ms = exchange.parse_timeframe("5m") * 1000
    ohlcv = exchange.fetch_ohlcv(symbol, "5m", limit=45)
    now_ms = exchange.milliseconds()
    closed = [c for c in ohlcv if c[0] + tf_ms <= now_ms]
    if len(closed) < 30:
        return

    df = pd.DataFrame(closed, columns=["ts", "Open", "High", "Low", "Close", "Volume"])

    # 1. Check open position first
    pos = positions.get(symbol)
    if pos:
        closed_trade = resolve_open_position(symbol, pos, df, p)
        if closed_trade:
            log.info("[%s] Virtual trade CLOSED: %s at %s (%s, Net: %s%%)",
                     symbol, closed_trade["side"], closed_trade["exit_reason"],
                     closed_trade["exit_price"], closed_trade["net_return_pct"])
            log_trade(closed_trade)
            del positions[symbol]
        else:
            log.debug("[%s] Position active (SL %s / TP %s) - skipping entry check",
                      symbol, pos["sl"], pos["tp"])
            return

    # 2. Check entry rules on latest closed candle
    signals = compute_signals(df, p)
    sig = signals[-1]
    if sig == 0:
        return

    last_bar = df.iloc[-1]
    last_ts = int(last_bar["ts"])
    last_close = float(last_bar["Close"])
    bar_time = pd.to_datetime(last_ts, unit="ms", utc=True)

    side_str = "LONG" if sig == LONG else "SHORT"
    direction = "Long" if sig == LONG else "Short"

    # Compute entry price (CMP / current market price) and SL/TP
    if sig == LONG:
        sl_raw = last_close * (1.0 - p.sl_pct)
        tp_raw = last_close * (1.0 + p.tp_pct)
    else:
        sl_raw = last_close * (1.0 + p.sl_pct)
        tp_raw = last_close * (1.0 - p.tp_pct)

    sl_str = exchange.price_to_precision(symbol, sl_raw)
    tp_str = exchange.price_to_precision(symbol, tp_raw)
    symbol_id = exchange.market(symbol)["id"]

    signal_text = format_signal(symbol_id, sig, tp_str, sl_str, args.leverage)
    log.info("\n>>> [%s] %s MOMENTUM BREAKOUT DETECTED at %s:\n%s\n",
             symbol_id, side_str, bar_time, signal_text)

    # 3. Verify against Tadex's signal parser
    problems = verify_with_tadex(signal_text, symbol_id, sig, tp_str, sl_str, args.leverage, tadex_repo)
    if problems:
        log.warning("[%s] Tadex parser warning: %s", symbol_id, problems)
        if args.insert:
            log.error("[%s] Aborting insert due to parser validation failure", symbol_id)
            return

    # 4. Insert into Tadex if safety gate passed and --insert enabled
    signal_id = None
    if args.insert and sink is not None:
        metadata = {
            "strategy": "intraday_momentum",
            "symbol": symbol_id,
            "timeframe": "5m",
            "bar_time": bar_time.isoformat(),
            "reference_price": last_close,
        }
        try:
            signal_id = sink.insert(signal_text, metadata)
            log.info("[%s] INSERTED into public.signals id=%s", symbol_id, signal_id)
        except Exception as e:
            log.error("[%s] Insert failed: %s", symbol_id, e)
            return

    # 5. Record open position
    positions[symbol] = {
        "symbol": symbol_id,
        "side": int(sig),
        "entry_price": float(last_close),
        "sl": float(sl_str),
        "tp": float(tp_str),
        "entry_bar_ts": int(last_ts),
        "opened_at": bar_time.isoformat(),
        "signal_id": signal_id,
    }


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leverage", type=int, default=10)
    ap.add_argument("--once", action="store_true", help="Run scan once and exit")
    ap.add_argument("--symbols", nargs="+", default=["BTC/USDT:USDT"], help="Symbols to scan (default: BTC/USDT:USDT)")
    ap.add_argument("--top20", action="store_true", help="Dynamically discover and scan top 20 Bybit volume pairs")
    ap.add_argument("--insert", action="store_true", help="Insert signals into Tadex public.signals")
    ap.add_argument("--env-file", default=str(ROOT / ".env"))
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "intraday_scanner.log", encoding="utf-8"),
        ],
    )

    if load_dotenv and Path(args.env_file).exists():
        load_dotenv(args.env_file)

    p = IntradayMomentumParams()
    exchange = ccxt.bybit({"enableRateLimit": True, "timeout": 15000})
    exchange.load_markets()

    repo = os.getenv("TADEX_REPO_PATH")
    tadex_repo = Path(repo) if repo else None

    # Safety Gate Check
    gate_ok, gate_msg = check_safety_gate()
    log.info("SAFETY GATE STATUS: %s", gate_msg)

    sink = None
    if args.insert:
        if not gate_ok:
            log.error("SAFETY GATE BLOCKED: %s", gate_msg)
            sys.exit(f"Refusing to insert signals: Safety criteria not met ({gate_msg})")

        if os.getenv("TADEX_TARGET_ENV", "").lower() != "testnet":
            sys.exit("Refusing to insert: TADEX_TARGET_ENV must be 'testnet' in .env.")

        sink = TadexSignalSink.from_env()
        provider = sink.check_provider()
        log.info("Connected to Tadex as provider: %s (%s)", provider.get("name"), provider.get("id"))
    else:
        log.info("DRY RUN MODE: Signals will be printed and tracked virtually (not inserted).")

    positions = load_positions()

    while True:
        try:
            if args.top20:
                log.info("Fetching top 20 Bybit pairs by 24h volume...")
                target_pairs = fetch_top_20_pairs(exchange)
            else:
                target_pairs = args.symbols

            log.info("Scanning %d pair(s) on 5m candles: %s", len(target_pairs), ", ".join(target_pairs))

            for sym in target_pairs:
                try:
                    evaluate_symbol(exchange, sym, p, positions, sink, tadex_repo, args)
                except Exception as e:
                    log.error("Error evaluating %s: %s", sym, e)
                finally:
                    save_positions(positions)

        except Exception as e:
            log.error("Scanner cycle error: %s", e)

        if args.once:
            break

        # Sleep until 15s after next 5-minute candle close
        now = time.time()
        sleep_sec = 300 - (now % 300) + 15
        log.info("Scan complete. Sleeping %.0fs until next 5m close...", sleep_sec)
        time.sleep(sleep_sec)


if __name__ == "__main__":
    main()
