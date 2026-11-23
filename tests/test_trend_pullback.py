"""
Unit and regression tests for 1-Hour Trend-Following Pullback Strategy.
Run: python -m pytest tests/test_trend_pullback.py -v
"""

import numpy as np
import pandas as pd
import pytest

from strategy_core import (
    FLAT,
    LONG,
    SHORT,
    StrategyParams,
    adx,
    compute_indicators,
    compute_signals,
    ema,
)
from trend_pullback_1h import simulate_1h_trend_pullback


def synthetic_1h_data(n: int = 500, seed: int = 123) -> pd.DataFrame:
    """Generate synthetic 1H OHLCV series for deterministic unit testing."""
    rng = np.random.default_rng(seed)
    # Generate random walk with drift
    returns = rng.normal(0.0005, 0.01, n)
    close = 50000.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + rng.uniform(0.002, 0.015, n))
    low = close * (1.0 - rng.uniform(0.002, 0.015, n))
    open_p = (high + low) / 2.0
    vol = rng.uniform(50.0, 500.0, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    }, index=idx)


def test_adx_bounds_and_validity():
    df = synthetic_1h_data(250)
    adx_v, p_di, m_di = adx(df["High"], df["Low"], df["Close"], 14)

    assert len(adx_v) == 250
    assert len(p_di) == 250
    assert len(m_di) == 250

    # Past warm-up (30 bars), check bounds
    valid_adx = adx_v[30:]
    valid_p = p_di[30:]
    valid_m = m_di[30:]

    assert not np.isnan(valid_adx).any()
    assert (valid_adx >= 0.0).all() and (valid_adx <= 100.0).all()
    assert (valid_p >= 0.0).all() and (valid_p <= 100.0).all()
    assert (valid_m >= 0.0).all() and (valid_m <= 100.0).all()


def test_no_lookahead_adx_pullback():
    """Signals on bar i must not change when subsequent bars are appended."""
    df = synthetic_1h_data(400)
    p = StrategyParams(mode="trend_pullback", use_adx_filter=True, adx_threshold=20.0)

    full_sig = compute_signals(df, p)
    for cut in (250, 300, 350):
        cut_sig = compute_signals(df.iloc[:cut], p)
        assert np.array_equal(cut_sig, full_sig[:cut])


def test_trend_pullback_with_adx_rules():
    """Verify that every Long and Short signal strictly obeys the mathematical definitions."""
    df = synthetic_1h_data(600)
    p = StrategyParams(mode="trend_pullback", use_adx_filter=True, adx_threshold=20.0)
    ind = compute_indicators(df, p)
    sig = compute_signals(df, p)

    long_indices = np.where(sig == LONG)[0]
    for i in long_indices:
        # 1. Bullish Trend: Close > EMA 200
        assert ind["Close"].iat[i] > ind["ema_slow"].iat[i]
        # 2. Pullback & Bounce: Prev Close <= Prev EMA 50, Current Close > EMA 50
        assert ind["Close"].iat[i - 1] <= ind["ema_fast"].iat[i - 1]
        assert ind["Close"].iat[i] > ind["ema_fast"].iat[i]
        # 3. ADX Confirmation: ADX > 20
        assert ind["adx"].iat[i] > 20.0
        # 4. DI Directional Alignment: +DI > -DI
        assert ind["plus_di"].iat[i] > ind["minus_di"].iat[i]

    short_indices = np.where(sig == SHORT)[0]
    for i in short_indices:
        # 1. Bearish Trend: Close < EMA 200
        assert ind["Close"].iat[i] < ind["ema_slow"].iat[i]
        # 2. Pullback & Bounce: Prev Close >= Prev EMA 50, Current Close < EMA 50
        assert ind["Close"].iat[i - 1] >= ind["ema_fast"].iat[i - 1]
        assert ind["Close"].iat[i] < ind["ema_fast"].iat[i]
        # 3. ADX Confirmation: ADX > 20
        assert ind["adx"].iat[i] > 20.0
        # 4. DI Directional Alignment: -DI > +DI
        assert ind["minus_di"].iat[i] > ind["plus_di"].iat[i]


def test_warmup_bars_are_flat():
    df = synthetic_1h_data(300)
    p = StrategyParams(mode="trend_pullback", use_adx_filter=True)
    sig = compute_signals(df, p)
    assert (sig[: p.warmup_bars] == FLAT).all()


def test_simulation_accounting_and_risk_sizing():
    """Verify position sizing and fee deduction in simulation engine."""
    df = synthetic_1h_data(500)
    trades, metrics, df_trades = simulate_1h_trend_pullback(
        df,
        sl_pct=0.025,
        tp_pct=0.06,
        risk_pct=0.015,
        use_adx=True,
        initial_cash=50.0,
    )

    assert metrics.initial_equity == 50.0
    if trades:
        # Check first trade sizing: notional = (equity * 0.015) / 0.025
        t0 = trades[0]
        expected_notional = (t0.equity_before * 0.015) / 0.025
        assert t0.notional_usd == pytest.approx(expected_notional, rel=1e-2)
        # Check that exit reason is valid
        assert t0.exit_reason in ("TP", "SL")
        # Check that net return accounts for round-trip taker fees (2 * 0.055% = 0.11%)
        fee_rate = 2.0 * 0.00055
        assert t0.net_return_pct == pytest.approx((t0.gross_return_pct - fee_rate * 100.0), abs=1e-3)
