"""
Backtest the Tadex strategy with backtesting.py.

Examples:
    # Default: DOGE perp, 1h, RSI-reversal
    python strategy.py

    # Trend pullback on BTC 4h
    python strategy.py --symbol BTC/USDT:USDT --timeframe 4h --strategy trend_pullback

    # Donchian breakout on BTC 4h
    python strategy.py --symbol BTC/USDT:USDT --timeframe 4h --strategy donchian_breakout

    # Optimise on the first 70% of data, then test the winner on the unseen 30%
    python strategy.py --symbol BTC/USDT:USDT --timeframe 4h --strategy trend_pullback --optimize

    # Open the interactive chart in the browser
    python strategy.py --plot

Outputs (in results/):
    <name>_stats.txt     full statistics (in-sample, out-of-sample, full period)
    <name>_trades.csv    every trade of the full-period run
    <name>_chart.html    interactive chart
    <name>_verdict.json  PASS/FAIL gate read by live_signal_generator.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from fetch_data import csv_path
from strategy_core import LONG, MODES, SHORT, StrategyParams, compute_indicators, compute_signals, stop_and_target

try:  # backtesting >= 0.6 - lets us trade fractions of a coin (needed for BTC/ETH)
    from backtesting.lib import FractionalBacktest
except ImportError:  # pragma: no cover
    FractionalBacktest = None

RESULTS_DIR = Path(__file__).parent / "results"

# Bybit USDT-perp taker fee is 0.055% per side. Market orders = taker.
DEFAULT_COMMISSION = 0.00055

# Go / no-go thresholds for the verdict gate
MIN_TRADES = 30
MIN_SHARPE = 0.0
MAX_DRAWDOWN_PCT = 25.0  # absolute value


class TadexStrategy(Strategy):
    # --- strategy parameters (class attributes so bt.optimize() can tune them) ---
    mode = "rsi_reversal"
    ema_fast = 50
    ema_slow = 200
    rsi_period = 14
    rsi_oversold = 30.0
    rsi_overbought = 70.0
    sl_pct = 0.02
    risk_reward = 2.0
    donchian_period = 20
    atr_period = 14
    atr_mult = 2.0
    trailing_stop_pct = 0.03

    # --- execution parameters ---
    margin_fraction = 0.10  # fraction of equity posted as margin per trade (x leverage = notional)
    trade_start = 0         # bar index before which no trades are opened (warm-up for OOS runs)

    def params(self) -> StrategyParams:
        return StrategyParams(
            mode=self.mode,
            ema_fast=int(self.ema_fast),
            ema_slow=int(self.ema_slow),
            rsi_period=int(self.rsi_period),
            rsi_oversold=float(self.rsi_oversold),
            rsi_overbought=float(self.rsi_overbought),
            sl_pct=float(self.sl_pct),
            risk_reward=float(self.risk_reward),
            donchian_period=int(self.donchian_period),
            atr_period=int(self.atr_period),
            atr_mult=float(self.atr_mult),
            trailing_stop_pct=float(self.trailing_stop_pct),
        )

    def init(self):
        p = self.params()
        df = self.data.df
        ind = compute_indicators(df, p)
        sig = compute_signals(df, p)
        self.ema_f = self.I(lambda: ind["ema_fast"].to_numpy(), name=f"EMA{p.ema_fast}")
        self.ema_s = self.I(lambda: ind["ema_slow"].to_numpy(), name=f"EMA{p.ema_slow}")
        self.rsi = self.I(lambda: ind["rsi"].to_numpy(), name="RSI", overlay=False)
        self.atr_ind = self.I(lambda: ind["atr"].to_numpy(), name="ATR", overlay=False)
        self.d_high = self.I(lambda: ind["donchian_high"].to_numpy(), name=f"DonchianHigh{p.donchian_period}")
        self.d_low = self.I(lambda: ind["donchian_low"].to_numpy(), name=f"DonchianLow{p.donchian_period}")
        self.signal = self.I(lambda: sig.astype(float), name="signal", plot=False)

    def next(self):
        p = self.params()
        price = float(self.data.Close[-1])

        # If in position, manage trailing stop for breakout strategy
        if self.position:
            if p.mode == "donchian_breakout":
                for trade in self.trades:
                    if trade.is_long:
                        trail_sl = price * (1.0 - p.trailing_stop_pct)
                        if trade.sl is None or trail_sl > trade.sl:
                            trade.sl = trail_sl
                    elif trade.is_short:
                        trail_sl = price * (1.0 + p.trailing_stop_pct)
                        if trade.sl is None or trail_sl < trade.sl:
                            trade.sl = trail_sl
            return

        if self.orders:
            return
        if len(self.data) - 1 < self.trade_start:
            return

        side = int(self.signal[-1])
        if side == 0:
            return

        atr_val = float(self.atr_ind[-1]) if hasattr(self, "atr_ind") else None
        sl, tp = stop_and_target(price, side, p, atr_val=atr_val)

        if side == LONG:
            self.buy(size=self.margin_fraction, sl=sl, tp=tp)
        elif side == SHORT:
            self.sell(size=self.margin_fraction, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"{path} not found. Run fetch_data.py first.")
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)  # backtesting.py prefers naive timestamps
    return df


def make_backtest(df: pd.DataFrame, args, plot_ready: bool = False) -> Backtest:
    kwargs = dict(
        cash=args.cash,
        commission=args.commission,
        margin=1.0 / args.leverage,
        trade_on_close=False,      # fill at next bar's open (realistic)
        hedging=False,
        exclusive_orders=False,
        finalize_trades=True,
    )
    cls = FractionalBacktest if FractionalBacktest is not None else Backtest
    try:
        return cls(df, TadexStrategy, **kwargs)
    except TypeError:  # older backtesting.py without finalize_trades
        kwargs.pop("finalize_trades")
        return cls(df, TadexStrategy, **kwargs)


def run_params(args) -> dict:
    return dict(
        mode=args.mode,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        rsi_period=args.rsi_period,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        sl_pct=args.sl_pct,
        risk_reward=args.risk_reward,
        donchian_period=args.donchian_period,
        atr_period=args.atr_period,
        atr_mult=args.atr_mult,
        trailing_stop_pct=args.trailing_stop_pct,
        margin_fraction=args.margin_fraction,
    )


def summarize(stats, label: str) -> dict:
    keys = ["Start", "End", "Return [%]", "Buy & Hold Return [%]", "Max. Drawdown [%]",
            "Sharpe Ratio", "Sortino Ratio", "# Trades", "Win Rate [%]", "Profit Factor",
            "Expectancy [%]", "Avg. Trade Duration"]
    out = {"segment": label}
    for k in keys:
        v = stats.get(k)
        if isinstance(v, (float, np.floating)):
            v = None if np.isnan(v) else round(float(v), 3)
        elif v is not None and not isinstance(v, (int, np.integer)):
            v = str(v)
        out[k] = int(v) if isinstance(v, np.integer) else v
    return out


def verdict(seg: dict) -> tuple[bool, list[str]]:
    reasons = []
    trades = seg.get("# Trades") or 0
    sharpe = seg.get("Sharpe Ratio")
    ret = seg.get("Return [%]") or 0.0
    dd = abs(seg.get("Max. Drawdown [%]") or 0.0)
    if trades < MIN_TRADES:
        reasons.append(f"only {trades} trades (< {MIN_TRADES}) - not statistically meaningful")
    if sharpe is None or sharpe <= MIN_SHARPE:
        reasons.append(f"Sharpe {sharpe} <= {MIN_SHARPE}")
    if ret <= 0:
        reasons.append(f"return {ret}% <= 0")
    if dd > MAX_DRAWDOWN_PCT:
        reasons.append(f"max drawdown {dd}% > {MAX_DRAWDOWN_PCT}%")
    return (not reasons), reasons


def print_segment(seg: dict):
    print(f"\n=== {seg['segment']} ===")
    for k, v in seg.items():
        if k != "segment":
            print(f"  {k:<24} {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="DOGE/USDT:USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--exchange", default="bybit")
    ap.add_argument("--csv", help="explicit CSV path (overrides symbol/timeframe/exchange)")
    ap.add_argument("--strategy", "--mode", dest="mode", choices=MODES, default="rsi_reversal",
                    help="Strategy choice: rsi_reversal, ema_cross, trend_pullback, donchian_breakout")
    ap.add_argument("--ema-fast", type=int, default=50)
    ap.add_argument("--ema-slow", type=int, default=200)
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--rsi-oversold", type=float, default=30.0)
    ap.add_argument("--rsi-overbought", type=float, default=70.0)
    ap.add_argument("--sl-pct", type=float, default=0.02)
    ap.add_argument("--risk-reward", type=float, default=2.0)
    ap.add_argument("--donchian-period", type=int, default=20)
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--atr-mult", type=float, default=2.0)
    ap.add_argument("--trailing-stop-pct", type=float, default=0.03)
    ap.add_argument("--cash", type=float, default=10_000)
    ap.add_argument("--leverage", type=float, default=10.0)
    ap.add_argument("--margin-fraction", type=float, default=0.10,
                    help="equity fraction used as margin per trade (0.10 x 10x = 1x notional)")
    ap.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    ap.add_argument("--split", type=float, default=0.7, help="in-sample fraction (0 < split < 1)")
    ap.add_argument("--optimize", action="store_true", help="grid-search on in-sample only")
    ap.add_argument("--plot", action="store_true", help="open the chart in the browser")
    args = ap.parse_args()

    # Validate params early (raises on bad combos)
    StrategyParams(
        mode=args.mode,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        rsi_period=args.rsi_period,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        sl_pct=args.sl_pct,
        risk_reward=args.risk_reward,
        donchian_period=args.donchian_period,
        atr_period=args.atr_period,
        atr_mult=args.atr_mult,
        trailing_stop_pct=args.trailing_stop_pct,
    )

    path = Path(args.csv) if args.csv else csv_path(args.symbol, args.timeframe, args.exchange)
    df = load_data(path)
    name = f"{path.stem}_{args.mode}"
    RESULTS_DIR.mkdir(exist_ok=True)

    split_idx = int(len(df) * args.split)
    warmup = (args.ema_slow * 3) if args.mode in ("rsi_reversal", "ema_cross", "trend_pullback") else (args.donchian_period * 3)
    if split_idx <= warmup or len(df) - split_idx < 100:
        sys.exit(f"Not enough data ({len(df)} bars) for a {args.split:.0%} split with warm-up ({warmup} bars). Fetch more history.")

    train = df.iloc[:split_idx]
    # Out-of-sample: prepend warm-up bars but forbid trading on them
    test_start = split_idx - warmup
    test = df.iloc[test_start:]
    print(f"Data: {path.name}  {len(df)} bars  {df.index[0]} -> {df.index[-1]}")
    print(f"In-sample : {train.index[0]} -> {train.index[-1]}  ({len(train)} bars)")
    print(f"Out-sample: {df.index[split_idx]} -> {df.index[-1]}  ({len(df) - split_idx} bars)")

    params = run_params(args)
    warnings.filterwarnings("ignore", category=UserWarning)

    # ---- In-sample (optionally optimise) ----
    bt_train = make_backtest(train, args)
    if args.optimize:
        print("\nOptimising on IN-SAMPLE data only (this can take a minute)...")
        if args.mode in ("rsi_reversal", "ema_cross"):
            grid = dict(
                sl_pct=[0.015, 0.02, 0.03],
                risk_reward=[1.5, 2.0, 2.5, 3.0],
                rsi_oversold=[25, 30, 35],
                rsi_overbought=[65, 70, 75],
            )
            constraint = lambda s: s.rsi_oversold < s.rsi_overbought
        elif args.mode == "trend_pullback":
            grid = dict(
                sl_pct=[0.015, 0.02, 0.03],
                risk_reward=[1.5, 2.0, 2.5, 3.0],
                ema_fast=[30, 50, 70],
            )
            constraint = None
        elif args.mode == "donchian_breakout":
            grid = dict(
                donchian_period=[15, 20, 30],
                atr_mult=[1.5, 2.0, 2.5],
                trailing_stop_pct=[0.02, 0.03, 0.04, 0.05],
            )
            constraint = None

        fixed = {k: v for k, v in params.items() if k not in grid}
        opt_kwargs = dict(
            maximize="Sharpe Ratio",
            **grid,
            **{k: [v] for k, v in fixed.items()},
        )
        if constraint:
            opt_kwargs["constraint"] = constraint

        stats_train = bt_train.optimize(**opt_kwargs)
        best = stats_train._strategy
        for k in grid:
            params[k] = getattr(best, k)
        print("Best in-sample params:", {k: params[k] for k in grid})
    else:
        stats_train = bt_train.run(**params)

    # ---- Out-of-sample with the (possibly optimised) params ----
    bt_test = make_backtest(test, args)
    stats_test = bt_test.run(**params, trade_start=warmup)

    # ---- Full period (for the chart and trade list) ----
    bt_full = make_backtest(df, args)
    stats_full = bt_full.run(**params)

    segs = [summarize(stats_train, "IN-SAMPLE" + (" (optimised - optimistic!)" if args.optimize else "")),
            summarize(stats_test, "OUT-OF-SAMPLE (the number that matters)"),
            summarize(stats_full, "FULL PERIOD")]
    for s in segs:
        print_segment(s)

    passed_oos, reasons_oos = verdict(segs[1])
    passed_full, reasons_full = verdict(segs[2])
    passed = passed_oos and passed_full
    reasons = [f"OOS: {r}" for r in reasons_oos] + [f"FULL: {r}" for r in reasons_full]

    print("\n" + "=" * 60)
    if passed:
        print("VERDICT: PASS - eligible for TESTNET forward-testing only.")
    else:
        print("VERDICT: FAIL - do NOT connect this to Tadex.")
        for r in reasons:
            print("   -", r)
    print("=" * 60)

    # ---- Save artefacts ----
    (RESULTS_DIR / f"{name}_stats.txt").write_text(
        "\n\n".join(str(s) for s in (stats_train, stats_test, stats_full)), encoding="utf-8")
    trades = stats_full["_trades"]
    trades.to_csv(RESULTS_DIR / f"{name}_trades.csv", index=False)

    sp_dict = {k: v for k, v in params.items() if k != "margin_fraction"}
    strategy_params = StrategyParams(
        mode=sp_dict["mode"],
        ema_fast=int(sp_dict["ema_fast"]),
        ema_slow=int(sp_dict["ema_slow"]),
        rsi_period=int(sp_dict["rsi_period"]),
        rsi_oversold=float(sp_dict["rsi_oversold"]),
        rsi_overbought=float(sp_dict["rsi_overbought"]),
        sl_pct=float(sp_dict["sl_pct"]),
        risk_reward=float(sp_dict["risk_reward"]),
        donchian_period=int(sp_dict["donchian_period"]),
        atr_period=int(sp_dict["atr_period"]),
        atr_mult=float(sp_dict["atr_mult"]),
        trailing_stop_pct=float(sp_dict["trailing_stop_pct"]),
    ).to_dict()

    verdict_doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_file": path.name,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "exchange": args.exchange,
        "params": strategy_params,
        "execution": {"cash": args.cash, "leverage": args.leverage,
                      "margin_fraction": args.margin_fraction, "commission": args.commission},
        "segments": segs,
        "passed": passed,
        "reasons": reasons,
    }
    vpath = RESULTS_DIR / f"{name}_verdict.json"
    vpath.write_text(json.dumps(verdict_doc, indent=2, default=str), encoding="utf-8")

    chart = RESULTS_DIR / f"{name}_chart.html"
    try:
        bt_full.plot(filename=str(chart), open_browser=args.plot, resample=False)
    except Exception as e:  # plotting must never break the run
        print(f"(chart skipped: {e})")
    print(f"\nSaved: {vpath.name}, {name}_stats.txt, {name}_trades.csv, {chart.name} in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
