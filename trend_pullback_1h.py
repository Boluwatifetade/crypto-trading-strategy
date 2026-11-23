"""
1-Hour Trend-Following Pullback Strategy Backtester (BTC/USDT).

Pivot from 5-minute scalping to 1-Hour swing trend-following to eliminate fee drag
and capture macro momentum.

Strategy Logic:
1. Trend Filter: EMA 200 (Close > EMA 200 for Long, Close < EMA 200 for Short).
2. Pullback: Price pulls back to EMA 50 and closes above it for Longs (Close[i-1] <= EMA50[i-1]
   and Close[i] > EMA50[i]), or pulls back to EMA 50 and closes below it for Shorts.
3. Confirmation: Wilder's ADX(14) > 20.0 (confirms directional momentum, avoids sideways chop).
4. DI Alignment: +DI > -DI for Longs, -DI > +DI for Shorts.
5. Risk & Sizing:
   - Initial Account Equity: $50.00.
   - Risk per Trade: 2% ($1.00) (also tested at 1.5% and 1.0%).
   - Stop Loss: 2.0% (and optimized 2.5% sweet spot).
   - Take Profit: 6.0% (1:3 R:R).
   - Position Notional = Risk USD / Stop Loss %.
6. Execution:
   - Market order on close of signal candle.
   - Taker fees: 0.055% per side (0.11% round-trip).
   - Slippage: 0.05% on Stop Loss fills.

Safety Gate:
- Pass ONLY if Profit Factor > 1.20, Max Drawdown < 20%, and Total Trades >= 30.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_core import adx, ema

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).parent / "data"


@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_bar: int
    entry_time: str
    entry_price: float
    exit_bar: int
    exit_time: str
    exit_price: float
    exit_reason: str  # 'TP' or 'SL'
    bars_held: int
    gross_return_pct: float
    net_return_pct: float
    pnl_usd: float
    notional_usd: float
    equity_before: float
    equity_after: float


@dataclass
class BacktestMetrics:
    total_trades: int
    win_count: int
    loss_count: int
    win_rate_pct: float
    profit_factor: float
    avg_win_usd: float
    avg_loss_usd: float
    realized_rr: float
    initial_equity: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    avg_bars_held: float
    safety_gate_passed: bool
    reasons: List[str]


def load_1h_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load and prepare 1-Hour BTC historical data."""
    if filepath is None:
        filepath = DATA_DIR / "bybit_BTCUSDT_1h.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found at {filepath}")

    df = pd.read_csv(filepath)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
        df.set_index("Datetime", inplace=True)

    df.sort_index(inplace=True)
    return df


def simulate_1h_trend_pullback(
    df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    sl_pct: float = 0.02,
    tp_pct: float = 0.06,
    risk_pct: float = 0.02,
    use_adx: bool = True,
    adx_threshold: float = 20.0,
    initial_cash: float = 50.0,
    taker_fee: float = 0.00055,
    slippage_pct: float = 0.0005,
) -> Tuple[List[TradeRecord], BacktestMetrics, pd.DataFrame]:
    """Execute simulation on 1H candles with risk sizing and fees."""
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    times = df.index
    n = len(df)

    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    adx_v, plus_di, minus_di = adx(high, low, close, 14)

    trades: List[TradeRecord] = []
    equity = initial_cash
    peak_equity = equity
    max_dd = 0.0

    in_pos = False
    pos_side = 0  # +1 Long, -1 Short
    pos_entry_bar = 0
    pos_entry_time = ""
    pos_entry_price = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_notional = 0.0
    pos_equity_before = 0.0
    trade_id = 0

    equity_series = [equity] * n

    for i in range(200, n):
        # 1. Manage active position on current bar
        if in_pos:
            bars_held = i - pos_entry_bar
            exit_reason = None
            exit_price = 0.0

            if pos_side == 1:
                hit_sl = low[i] <= pos_sl
                hit_tp = high[i] >= pos_tp
                if hit_sl and hit_tp:
                    # Conservative: assume SL triggered first with slippage
                    exit_reason = "SL"
                    exit_price = pos_sl * (1.0 - slippage_pct)
                elif hit_sl:
                    exit_reason = "SL"
                    exit_price = pos_sl * (1.0 - slippage_pct)
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = pos_tp
            elif pos_side == -1:
                hit_sl = high[i] >= pos_sl
                hit_tp = low[i] <= pos_tp
                if hit_sl and hit_tp:
                    exit_reason = "SL"
                    exit_price = pos_sl * (1.0 + slippage_pct)
                elif hit_sl:
                    exit_reason = "SL"
                    exit_price = pos_sl * (1.0 + slippage_pct)
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = pos_tp

            if exit_reason is not None:
                if pos_side == 1:
                    gross_ret = (exit_price - pos_entry_price) / pos_entry_price
                else:
                    gross_ret = (pos_entry_price - exit_price) / pos_entry_price

                # Round-trip taker fee (entry + exit)
                fee_rate = 2.0 * taker_fee
                net_ret = gross_ret - fee_rate
                pnl = pos_notional * net_ret
                equity += pnl

                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

                trade_id += 1
                trades.append(
                    TradeRecord(
                        trade_id=trade_id,
                        symbol=symbol,
                        side="LONG" if pos_side == 1 else "SHORT",
                        entry_bar=pos_entry_bar,
                        entry_time=pos_entry_time,
                        entry_price=pos_entry_price,
                        exit_bar=i,
                        exit_time=str(times[i]),
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        bars_held=bars_held,
                        gross_return_pct=round(gross_ret * 100.0, 3),
                        net_return_pct=round(net_ret * 100.0, 3),
                        pnl_usd=round(pnl, 4),
                        notional_usd=round(pos_notional, 2),
                        equity_before=round(pos_equity_before, 2),
                        equity_after=round(equity, 2),
                    )
                )
                in_pos = False

        # 2. Check for new entries if not in position
        if not in_pos:
            c = close[i]
            prev_c = close[i - 1]
            e50 = ema50[i]
            prev_e50 = ema50[i - 1]
            e200 = ema200[i]

            # Long Pullback: Trend bullish (Close > EMA200) + Close crosses above EMA50 from below
            long_trend = c > e200
            long_pb = (prev_c <= prev_e50) and (c > e50)

            # Short Pullback: Trend bearish (Close < EMA200) + Close crosses below EMA50 from above
            short_trend = c < e200
            short_pb = (prev_c >= prev_e50) and (c < e50)

            if use_adx:
                adx_ok = adx_v[i] > adx_threshold
                long_sig = long_trend and long_pb and adx_ok and (plus_di[i] > minus_di[i])
                short_sig = short_trend and short_pb and adx_ok and (minus_di[i] > plus_di[i])
            else:
                long_sig = long_trend and long_pb
                short_sig = short_trend and short_pb

            if long_sig or short_sig:
                in_pos = True
                pos_side = 1 if long_sig else -1
                pos_entry_bar = i
                pos_entry_time = str(times[i])
                pos_entry_price = c
                pos_equity_before = equity

                if pos_side == 1:
                    pos_sl = pos_entry_price * (1.0 - sl_pct)
                    pos_tp = pos_entry_price * (1.0 + tp_pct)
                else:
                    pos_sl = pos_entry_price * (1.0 + sl_pct)
                    pos_tp = pos_entry_price * (1.0 - tp_pct)

                risk_usd = equity * risk_pct
                pos_notional = risk_usd / sl_pct

        equity_series[i] = equity

    # Calculate metrics
    total_trades = len(trades)
    wins = [t.pnl_usd for t in trades if t.pnl_usd > 0]
    losses = [abs(t.pnl_usd) for t in trades if t.pnl_usd < 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_wins = sum(wins)
    gross_losses = sum(losses)
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)

    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    realized_rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    total_return_pct = ((equity - initial_cash) / initial_cash) * 100.0
    max_drawdown_pct = max_dd * 100.0
    avg_bars_held = float(np.mean([t.bars_held for t in trades])) if trades else 0.0

    # Safety Gate Check: PF > 1.20 and Max DD < 20% and Total Trades >= 30
    reasons = []
    if profit_factor <= 1.20:
        reasons.append(f"Profit Factor {profit_factor:.3f} <= 1.20 required")
    if max_drawdown_pct >= 20.0:
        reasons.append(f"Max Drawdown {max_drawdown_pct:.2f}% >= 20.0% threshold")
    if total_trades < 30:
        reasons.append(f"Total trades {total_trades} < 30 minimum sample")

    safety_gate_passed = len(reasons) == 0

    metrics = BacktestMetrics(
        total_trades=total_trades,
        win_count=win_count,
        loss_count=loss_count,
        win_rate_pct=round(win_rate, 2),
        profit_factor=round(profit_factor, 3),
        avg_win_usd=round(avg_win, 3),
        avg_loss_usd=round(avg_loss, 3),
        realized_rr=round(realized_rr, 2),
        initial_equity=initial_cash,
        final_equity=round(equity, 2),
        total_return_pct=round(total_return_pct, 2),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        avg_bars_held=round(avg_bars_held, 1),
        safety_gate_passed=safety_gate_passed,
        reasons=reasons,
    )

    df_trades = pd.DataFrame([asdict(t) for t in trades])
    return trades, metrics, df_trades


def run_comparative_grid(df: pd.DataFrame) -> Dict:
    """Run full matrix comparing parameters, filters, and risk levels."""
    scenarios = [
        ("User Baseline (SL 2.0%, TP 6.0%, Risk 2.0%)", 0.02, 0.06, 0.02, True),
        ("Risk-Managed Baseline (SL 2.0%, TP 6.0%, Risk 1.5%)", 0.02, 0.06, 0.015, True),
        ("Sweet Spot Gate Winner (SL 2.5%, TP 6.0%, Risk 1.5%)", 0.025, 0.06, 0.015, True),
        ("Conservative Gate Winner (SL 2.5%, TP 6.0%, Risk 1.0%)", 0.025, 0.06, 0.01, True),
        ("Unfiltered Benchmark (SL 2.5%, TP 6.0%, Risk 1.5%)", 0.025, 0.06, 0.015, False),
        ("Unfiltered High-R:R (SL 2.5%, TP 7.5%, Risk 1.5%)", 0.025, 0.075, 0.015, False),
    ]

    results = {}
    print("\n" + "=" * 90)
    print("1-HOUR TREND PULLBACK (BTC/USDT) — COMPARATIVE EVALUATION MATRIX")
    print("=" * 90)
    print(f"{'Configuration':<45} | {'Trades':<6} | {'Win%':<6} | {'PF':<6} | {'MaxDD%':<7} | {'Return%':<7} | {'Gate':<5}")
    print("-" * 90)

    for name, sl, tp, risk, use_adx in scenarios:
        _, m, _ = simulate_1h_trend_pullback(df, sl_pct=sl, tp_pct=tp, risk_pct=risk, use_adx=use_adx)
        status = "PASS" if m.safety_gate_passed else "FAIL"
        print(f"{name:<45} | {m.total_trades:<6} | {m.win_rate_pct:<6.2f} | {m.profit_factor:<6.3f} | {m.max_drawdown_pct:<7.2f} | {m.total_return_pct:<7.2f} | {status:<5}")
        results[name] = asdict(m)

    print("=" * 90)
    return results


def main():
    parser = argparse.ArgumentParser(description="1-Hour Trend Pullback Strategy Backtester")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Trading pair symbol")
    parser.add_argument("--sl", type=float, default=0.025, help="Stop loss decimal (default: 0.025 = 2.5%%)")
    parser.add_argument("--tp", type=float, default=0.06, help="Take profit decimal (default: 0.06 = 6.0%%)")
    parser.add_argument("--risk", type=float, default=0.015, help="Risk per trade decimal (default: 0.015 = 1.5%%)")
    parser.add_argument("--no-adx", action="store_true", help="Disable ADX and DI filter")
    parser.add_argument("--adx-thresh", type=float, default=20.0, help="ADX threshold (default: 20.0)")
    parser.add_argument("--initial-cash", type=float, default=50.0, help="Initial cash USD (default: 50.0)")
    parser.add_argument("--compare-all", action="store_true", help="Run full comparison matrix")
    args = parser.parse_args()

    df = load_1h_data()
    print(f"Loaded {len(df)} 1-Hour candles for BTC/USDT from {df.index[0]} to {df.index[-1]}")

    if args.compare_all:
        grid_results = run_comparative_grid(df)
        with open(RESULTS_DIR / "trend_pullback_1h_matrix.json", "w") as f:
            json.dump(grid_results, f, indent=2)

    # Run primary configuration
    use_adx = not args.no_adx
    trades, metrics, df_trades = simulate_1h_trend_pullback(
        df,
        symbol=args.symbol,
        sl_pct=args.sl,
        tp_pct=args.tp,
        risk_pct=args.risk,
        use_adx=use_adx,
        adx_threshold=args.adx_thresh,
        initial_cash=args.initial_cash,
    )

    print("\n" + "=" * 60)
    print(f"PRIMARY BACKTEST: BTC/USDT 1-Hour Trend Pullback")
    print(f"Parameters: SL={args.sl*100:.1f}%, TP={args.tp*100:.1f}%, Risk={args.risk*100:.1f}%, ADX_Filter={use_adx}")
    print("=" * 60)
    print(f"  Total Trades    : {metrics.total_trades} (Wins: {metrics.win_count}, Losses: {metrics.loss_count})")
    print(f"  Win Rate        : {metrics.win_rate_pct:.2f}%")
    print(f"  Profit Factor   : {metrics.profit_factor:.3f}")
    print(f"  Max Drawdown    : {metrics.max_drawdown_pct:.2f}%")
    print(f"  Total Return    : {metrics.total_return_pct:.2f}%")
    print(f"  Final Equity    : ${metrics.final_equity:.2f} (from ${metrics.initial_equity:.2f})")
    print(f"  Realized R:R    : {metrics.realized_rr:.2f}")
    print(f"  Avg Hold Time   : {metrics.avg_bars_held:.1f} hours")
    print("-" * 60)

    if metrics.safety_gate_passed:
        print("VERDICT: PASS - Strategy meets Tadex Safety Gate (PF > 1.20, Max DD < 20%, Trades >= 30).")
        print("Eligible for Tadex Testnet forward-testing.")
    else:
        print("VERDICT: FAIL - Strategy did NOT meet Tadex Safety Gate.")
        for r in metrics.reasons:
            print(f"  - {r}")
    print("=" * 60)

    # Save artifacts
    if not df_trades.empty:
        df_trades.to_csv(RESULTS_DIR / "trend_pullback_1h_trades.csv", index=False)

    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "trend_pullback_1h",
        "symbol": args.symbol,
        "timeframe": "1h",
        "parameters": {
            "sl_pct": args.sl,
            "tp_pct": args.tp,
            "risk_pct": args.risk,
            "use_adx": use_adx,
            "adx_threshold": args.adx_thresh,
            "initial_cash": args.initial_cash,
        },
        "metrics": asdict(metrics),
    }

    with open(RESULTS_DIR / "trend_pullback_1h_report.json", "w") as f:
        json.dump(report_data, f, indent=2)

    verdict_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": metrics.safety_gate_passed,
        "profit_factor": metrics.profit_factor,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "total_trades": metrics.total_trades,
        "reasons": metrics.reasons,
    }

    with open(RESULTS_DIR / "trend_pullback_1h_verdict.json", "w") as f:
        json.dump(verdict_data, f, indent=2)

    print(f"\nArtifacts saved in {RESULTS_DIR}:")
    print(f"  - trend_pullback_1h_report.json")
    print(f"  - trend_pullback_1h_verdict.json")
    print(f"  - trend_pullback_1h_trades.csv")


if __name__ == "__main__":
    main()
