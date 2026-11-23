"""
Intraday Momentum Backtest Runner — BTC/USDT 5-Minute Candles (1 Year).

Evaluates the Final Optimized Configuration:
1. Single Pair: BTC/USDT:USDT (105,287 5m candles).
2. Risk & Sizing: 2% risk ($1.00 on $50 equity). Notional = $1.00 / SL%.
   Minimum size check: Skip trade if Notional < $1.00 (Bybit minimum).
3. Maker Execution Simulation:
   - Post-Only Limit Order placed at the breakout level (or 0.05% below for longs).
   - Order cancelled if not retested and filled within 2 candles.
   - Entry fee: Maker fee 0.020%.
   - Exit fee: Taker fee 0.055% on SL / TP / Time Stop (market orders).
   - Slippage: 0.050% on Stop Loss execution.
4. ADX Filter: ADX(14) > 25.0 vs. No ADX filter.
5. Volume Filter: Breakout volume > 2.0x SMA(20) vs. 1.5x SMA(20).
6. Safety Gate: Profit Factor > 1.2 and Max Drawdown < 20%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fetch_5m_data import DEFAULT_SYMBOLS, csv_path
from intraday_momentum_core import (
    IntradayMomentumParams,
    TradeRecord,
    simulate_trades,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BTC_ONLY_SYMBOLS = ["BTC/USDT:USDT"]


def calculate_metrics(trades: List[TradeRecord], initial_cash: float = 50.0) -> Dict:
    """Calculate comprehensive performance metrics from trade records."""
    if not trades:
        return {
            "total_trades": 0,
            "win_count": 0,
            "loss_count": 0,
            "scratch_count": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "avg_rr": 0.0,
            "initial_equity": initial_cash,
            "final_equity": initial_cash,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "tp_count": 0,
            "sl_count": 0,
            "time_stop_count": 0,
        }

    net_returns = [t.net_return for t in trades]
    wins = [r for r in net_returns if r > 0]
    losses = [r for r in net_returns if r < 0]
    scratches = [r for r in net_returns if r == 0]

    win_count = len(wins)
    loss_count = len(losses)
    total_trades = len(trades)

    win_rate = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0

    gross_wins = sum(wins)
    gross_losses = abs(sum(losses))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)

    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = abs(float(np.mean(losses))) if losses else 0.0
    avg_rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    # Calculate equity curve from sequential trades
    equity = initial_cash
    peak = equity
    max_dd = 0.0

    for t in trades:
        equity += t.pnl_usd
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    total_return_pct = ((equity - initial_cash) / initial_cash) * 100.0
    max_drawdown_pct = max_dd * 100.0

    tp_count = sum(1 for t in trades if t.exit_reason == "TP")
    sl_count = sum(1 for t in trades if t.exit_reason == "SL")
    time_stop_count = sum(1 for t in trades if t.exit_reason == "TIME_STOP")

    return {
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "scratch_count": len(scratches),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "avg_win_pct": round(avg_win * 100, 3),
        "avg_loss_pct": round(avg_loss * 100, 3),
        "avg_rr": round(avg_rr, 2),
        "initial_equity": round(initial_cash, 2),
        "final_equity": round(equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "tp_count": tp_count,
        "sl_count": sl_count,
        "time_stop_count": time_stop_count,
    }


def run_single_simulation(
    symbols: List[str] = BTC_ONLY_SYMBOLS,
    timeframe: str = "5m",
    p: Optional[IntradayMomentumParams] = None,
    use_maker_fee: bool = True,
    initial_cash_per_pair: float = 50.0,
) -> Tuple[List[TradeRecord], Dict, Dict]:
    """Execute backtest across given symbols."""
    if p is None:
        p = IntradayMomentumParams()

    all_trades: List[TradeRecord] = []
    symbol_results = {}
    valid_symbols = []

    for sym in symbols:
        file_path = csv_path(sym, timeframe)
        if not file_path.exists():
            continue
        valid_symbols.append(sym)

        df = pd.read_csv(file_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        trades, _ = simulate_trades(
            df,
            symbol=sym,
            p=p,
            initial_cash=initial_cash_per_pair,
            use_maker_fee=use_maker_fee,
        )
        metrics = calculate_metrics(trades, initial_cash=initial_cash_per_pair)
        symbol_results[sym] = {"metrics": metrics, "bars": len(df)}
        all_trades.extend(trades)

    all_trades.sort(key=lambda t: t.entry_time)
    portfolio_initial_cash = initial_cash_per_pair * max(len(valid_symbols), 1)
    portfolio_metrics = calculate_metrics(all_trades, initial_cash=portfolio_initial_cash)
    return all_trades, portfolio_metrics, symbol_results


def run_backtest(
    symbols: List[str] = BTC_ONLY_SYMBOLS,
    timeframe: str = "5m",
    compare: bool = True,
):
    p = IntradayMomentumParams()

    print("\n" + "=" * 80)
    print("INTRADAY MOMENTUM STRATEGY — FINAL OPTIMIZATION & POST-ONLY BACKTEST")
    print("=" * 80)
    print(f"Asset:          {', '.join(symbols)} (BTCUSDT Only)")
    print(f"Account:        $50.00 base equity | Risk: 2% ($1.00) | Min Size: $1.00")
    print(f"Entry Mode:     Post-Only Limit Retest at Breakout Price (2-candle expiry)")
    print(f"Fee Model:      Maker Entry (0.020%) + Taker Exit (0.055%) = 0.075% round-trip")
    print(f"Filters:        ADX(14) > {p.adx_threshold} | Volume > {p.vol_mult}x SMA(20)")
    print(f"Risk Limits:    SL {p.sl_pct*100}% | TP {p.tp_pct*100}% | 4h Time Stop | Slippage {p.slippage_pct*100}%")
    print("-" * 80)

    all_trades, portfolio_metrics, symbol_results = run_single_simulation(
        symbols, timeframe, p=p, use_maker_fee=True
    )

    for sym, res in symbol_results.items():
        m = res["metrics"]
        print(f"\n[{sym}] ({res['bars']} candles)")
        print(f"  Trades:        {m['total_trades']:<6} | Win Rate: {m['win_rate_pct']}% ({m['win_count']}W / {m['loss_count']}L)")
        print(f"  Avg R:R:       {m['avg_rr']:<6} | Avg Win: +{m['avg_win_pct']}% | Avg Loss: -{m['avg_loss_pct']}%")
        print(f"  Profit Factor: {m['profit_factor']:<6} | Total Return: {m['total_return_pct']:+6.2f}% (${m['initial_equity']} -> ${m['final_equity']})")
        print(f"  Max Drawdown:  {m['max_drawdown_pct']}%")
        print(f"  Exit Breakdown: TP={m['tp_count']}, SL={m['sl_count']}, TimeStop={m['time_stop_count']}")

    print("\n" + "=" * 80)
    print("PRIMARY CONFIGURATION SUMMARY (ADX FILTER + POST-ONLY LIMIT RETEST)")
    print("=" * 80)
    print(f"  Total Trades:        {portfolio_metrics['total_trades']}")
    print(f"  Win Rate:            {portfolio_metrics['win_rate_pct']}% ({portfolio_metrics['win_count']} wins / {portfolio_metrics['loss_count']} losses)")
    print(f"  Avg R:R:             {portfolio_metrics['avg_rr']}")
    print(f"  Profit Factor:       {portfolio_metrics['profit_factor']}")
    print(f"  Account Equity:      ${portfolio_metrics['initial_equity']} -> ${portfolio_metrics['final_equity']}")
    print(f"  Total Return:        {portfolio_metrics['total_return_pct']}%")
    print(f"  Max Drawdown:        {portfolio_metrics['max_drawdown_pct']}%")
    print(f"  Exit Breakdown:      TP={portfolio_metrics['tp_count']}, SL={portfolio_metrics['sl_count']}, TimeStop={portfolio_metrics['time_stop_count']}")

    # Comparison: "No Filter" vs "ADX Filter" (both Post-Only limit and Market Fill)
    comparison_data = {}
    if compare:
        # 1. No Filter (Post-Only Limit Retest): Vol 1.5x, no ADX
        p_no_filter_limit = IntradayMomentumParams(
            vol_mult=1.5, use_adx_filter=False, use_post_only_limit=True, risk_per_trade_pct=0.02
        )
        _, m_nf_limit, _ = run_single_simulation(symbols, timeframe, p=p_no_filter_limit, use_maker_fee=True)

        # 2. ADX Filter (Post-Only Limit Retest): Vol 2.0x, ADX > 25 (User Setup)
        m_adx_limit = portfolio_metrics

        # 3. No Filter (Market Order Fill): Vol 1.5x, no ADX
        p_no_filter_mkt = IntradayMomentumParams(
            vol_mult=1.5, use_adx_filter=False, use_post_only_limit=False, risk_per_trade_pct=0.02
        )
        _, m_nf_mkt, _ = run_single_simulation(symbols, timeframe, p=p_no_filter_mkt, use_maker_fee=True)

        # 4. ADX Filter (Market Order Fill): Vol 2.0x, ADX > 25
        p_adx_mkt = IntradayMomentumParams(
            vol_mult=2.0, use_adx_filter=True, adx_threshold=25.0, use_post_only_limit=False, risk_per_trade_pct=0.02
        )
        _, m_adx_mkt, _ = run_single_simulation(symbols, timeframe, p=p_adx_mkt, use_maker_fee=True)

        print("\n" + "=" * 80)
        print("EMPIRICAL COMPARISON: 'NO FILTER' vs. 'ADX FILTER' (BTC/USDT 5M)")
        print("=" * 80)
        print(f"{'Metric':<20} | {'No Filter (Limit)':<20} | {'ADX Filter (Limit)':<20} | {'No Filter (Market)':<20} | {'ADX Filter (Market)':<20}")
        print("-" * 110)
        def f_pct(v): return f"{v}%"
        def f_ret(v): return f"{v:+.2f}%"
        def f_usd(v): return f"${v}"

        print(f"{'Total Trades':<20} | {str(m_nf_limit['total_trades']):<20} | {str(m_adx_limit['total_trades']):<20} | {str(m_nf_mkt['total_trades']):<20} | {str(m_adx_mkt['total_trades']):<20}")
        print(f"{'Win Rate':<20} | {f_pct(m_nf_limit['win_rate_pct']):<20} | {f_pct(m_adx_limit['win_rate_pct']):<20} | {f_pct(m_nf_mkt['win_rate_pct']):<20} | {f_pct(m_adx_mkt['win_rate_pct']):<20}")
        print(f"{'Profit Factor':<20} | {str(m_nf_limit['profit_factor']):<20} | {str(m_adx_limit['profit_factor']):<20} | {str(m_nf_mkt['profit_factor']):<20} | {str(m_adx_mkt['profit_factor']):<20}")
        print(f"{'Avg R:R Ratio':<20} | {str(m_nf_limit['avg_rr']):<20} | {str(m_adx_limit['avg_rr']):<20} | {str(m_nf_mkt['avg_rr']):<20} | {str(m_adx_mkt['avg_rr']):<20}")
        print(f"{'Total Return':<20} | {f_ret(m_nf_limit['total_return_pct']):<20} | {f_ret(m_adx_limit['total_return_pct']):<20} | {f_ret(m_nf_mkt['total_return_pct']):<20} | {f_ret(m_adx_mkt['total_return_pct']):<20}")
        print(f"{'Max Drawdown':<20} | {f_pct(m_nf_limit['max_drawdown_pct']):<20} | {f_pct(m_adx_limit['max_drawdown_pct']):<20} | {f_pct(m_nf_mkt['max_drawdown_pct']):<20} | {f_pct(m_adx_mkt['max_drawdown_pct']):<20}")
        print(f"{'Final Equity ($50)':<20} | {f_usd(m_nf_limit['final_equity']):<20} | {f_usd(m_adx_limit['final_equity']):<20} | {f_usd(m_nf_mkt['final_equity']):<20} | {f_usd(m_adx_mkt['final_equity']):<20}")
        print("=" * 110)

        comparison_data = {
            "no_filter_limit": m_nf_limit,
            "adx_filter_limit": m_adx_limit,
            "no_filter_market": m_nf_mkt,
            "adx_filter_market": m_adx_mkt,
        }

    # Safety Gate Check
    pf_pass = portfolio_metrics["profit_factor"] > 1.2
    dd_pass = portfolio_metrics["max_drawdown_pct"] < 20.0
    passed = pf_pass and dd_pass and portfolio_metrics["total_trades"] >= 30

    print("-" * 80)
    print(f"SAFETY GATE CHECK (Profit Factor > 1.2 and Max Drawdown < 20%):")
    print(f"  - Profit Factor > 1.2:  {'PASS' if pf_pass else 'FAIL'} ({portfolio_metrics['profit_factor']})")
    print(f"  - Max Drawdown < 20%:   {'PASS' if dd_pass else 'FAIL'} ({portfolio_metrics['max_drawdown_pct']}%)")
    print(f"  - Total Trades >= 30:   {'PASS' if portfolio_metrics['total_trades'] >= 30 else 'FAIL'} ({portfolio_metrics['total_trades']})")
    print("=" * 80)

    if passed:
        print("VERDICT: PASS - Strategy meets strict criteria for Tadex Testnet forward-testing.")
    else:
        print("VERDICT: FAIL - Do NOT connect to Tadex. Safety criteria not met.")
    print("=" * 80)

    # Save report
    report_data = {
        "parameters": p.to_dict(),
        "portfolio_metrics": portfolio_metrics,
        "symbol_metrics": symbol_results,
        "comparison": comparison_data,
        "safety_gate": {
            "passed": passed,
            "min_profit_factor": 1.2,
            "max_drawdown_allowed": 20.0,
            "actual_profit_factor": portfolio_metrics["profit_factor"],
            "actual_max_drawdown": portfolio_metrics["max_drawdown_pct"],
        },
    }

    report_path = RESULTS_DIR / "intraday_momentum_report.json"
    report_path.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")

    if all_trades:
        df_trades = pd.DataFrame([
            {
                "symbol": t.symbol,
                "side": "LONG" if t.side == 1 else "SHORT",
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
                "gross_return_pct": round(t.gross_return * 100, 3),
                "net_return_pct": round(t.net_return * 100, 3),
                "pnl_usd": round(t.pnl_usd, 2),
                "notional": round(t.notional, 2),
                "account_equity_before": round(t.account_equity_before, 2),
                "account_equity_after": round(t.account_equity_after, 2),
            }
            for t in all_trades
        ])
        df_trades.to_csv(RESULTS_DIR / "intraday_momentum_trades.csv", index=False)

    print(f"\nArtifacts saved to {RESULTS_DIR}:")
    print(f"  - intraday_momentum_report.json")
    print(f"  - intraday_momentum_trades.csv")
    return passed, portfolio_metrics, symbol_results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=BTC_ONLY_SYMBOLS)
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--no-compare", action="store_true", help="Disable comparative summary table")
    args = ap.parse_args()

    run_backtest(
        symbols=args.symbols,
        timeframe=args.timeframe,
        compare=not args.no_compare,
    )


if __name__ == "__main__":
    main()
