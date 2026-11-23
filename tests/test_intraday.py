"""Unit and integration tests for Intraday Momentum Strategy."""

import numpy as np
import pandas as pd
import pytest

from intraday_momentum_core import (
    FLAT,
    LONG,
    SHORT,
    IntradayMomentumParams,
    adx,
    atr,
    compute_indicators,
    compute_signals,
    recent_extremes,
    simulate_trades,
    volume_sma,
)


def synthetic_5m_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    high = close * (1.0 + rng.uniform(0.001, 0.008, n))
    low = close * (1.0 - rng.uniform(0.001, 0.008, n))
    open_p = (high + low) / 2.0
    vol = rng.uniform(100.0, 1000.0, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    }, index=idx)


def test_indicators_shape_and_no_nan_propagation():
    df = synthetic_5m_data(100)
    a = atr(df["High"], df["Low"], df["Close"], 14)
    v_sma = volume_sma(df["Volume"], 20)
    hi, lo = recent_extremes(df["High"], df["Low"], 3)

    assert len(a) == 100
    assert len(v_sma) == 100
    assert len(hi) == 100
    assert len(lo) == 100

    # Past warm-up, all values must be valid numbers
    assert not np.isnan(a[25:]).any()
    assert not np.isnan(v_sma[25:]).any()
    assert not np.isnan(hi[5:]).any()
    assert not np.isnan(lo[5:]).any()


def test_no_lookahead_intraday():
    """Signals on bar i must not change when future bars are appended."""
    df = synthetic_5m_data(400)
    p = IntradayMomentumParams()
    full_signals = compute_signals(df, p)

    for cutoff in (100, 200, 300):
        sliced_signals = compute_signals(df.iloc[:cutoff], p)
        assert np.array_equal(sliced_signals, full_signals[:cutoff])


def test_signal_rule_semantics():
    df = synthetic_5m_data(500)
    p = IntradayMomentumParams()
    ind = compute_indicators(df, p)
    signals = compute_signals(df, p)

    for i in np.where(signals == LONG)[0]:
        assert ind["Close"].iat[i] > ind["high_recent"].iat[i]
        assert ind["Volume"].iat[i] > (p.vol_mult * ind["vol_sma"].iat[i])
        assert ind["atr_ratio"].iat[i] > p.volatility_threshold

    for i in np.where(signals == SHORT)[0]:
        assert ind["Close"].iat[i] < ind["low_recent"].iat[i]
        assert ind["Volume"].iat[i] > (p.vol_mult * ind["vol_sma"].iat[i])
        assert ind["atr_ratio"].iat[i] > p.volatility_threshold


def test_trade_simulation_execution():
    df = synthetic_5m_data(500)
    p = IntradayMomentumParams(use_post_only_limit=False)
    trades, df_eq = simulate_trades(df, symbol="BTCUSDT", p=p, use_maker_fee=False)

    assert isinstance(trades, list)
    assert not df_eq.empty

    for t in trades:
        assert t.exit_reason in ("TP", "SL", "TIME_STOP")
        assert t.bars_held <= p.time_stop_bars
        # Net return must account for 2x taker fee when use_post_only_limit is False
        assert t.net_return == pytest.approx(t.gross_return - 2 * p.taker_fee, abs=1e-6)


def test_maker_fee_simulation():
    df = synthetic_5m_data(500)
    # Default is Post-Only: maker fee on entry + taker fee on exit
    p = IntradayMomentumParams(use_post_only_limit=True)
    trades, _ = simulate_trades(df, symbol="BTCUSDT", p=p)

    for t in trades:
        assert t.net_return == pytest.approx(t.gross_return - (p.maker_fee + p.taker_fee), abs=1e-6)


def test_min_notional_skip():
    df = synthetic_5m_data(500)
    # Set min_notional impossibly high ($10,000) so all trades are skipped
    p = IntradayMomentumParams(min_notional=10000.0)
    trades, _ = simulate_trades(df, symbol="BTCUSDT", p=p, initial_cash=50.0)

    assert len(trades) == 0


def test_risk_based_position_sizing():
    df = synthetic_5m_data(500)
    p = IntradayMomentumParams()
    trades, _ = simulate_trades(df, symbol="BTCUSDT", p=p, initial_cash=50.0)

    for t in trades:
        # Notional must equal (equity_before * risk_pct) / sl_pct
        expected_notional = (t.account_equity_before * p.risk_per_trade_pct) / p.sl_pct
        assert t.notional == pytest.approx(expected_notional, rel=1e-5)


def test_calculate_metrics_structure():
    from intraday_backtest import calculate_metrics

    df = synthetic_5m_data(500)
    p = IntradayMomentumParams()
    trades, _ = simulate_trades(df, symbol="BTCUSDT", p=p, initial_cash=50.0)

    metrics = calculate_metrics(trades, initial_cash=50.0)
    assert "profit_factor" in metrics
    assert "win_rate_pct" in metrics
    assert "max_drawdown_pct" in metrics
    assert "initial_equity" in metrics
    assert "final_equity" in metrics
    assert metrics["initial_equity"] == 50.0


def test_adx_indicator_calculation():
    df = synthetic_5m_data(300)
    adx_arr, plus_di, minus_di = adx(df["High"], df["Low"], df["Close"], 14)

    assert len(adx_arr) == 300
    assert len(plus_di) == 300
    assert len(minus_di) == 300

    # Past warm-up period, values must be finite and within 0..100
    valid_adx = adx_arr[30:]
    assert not np.isnan(valid_adx).any()
    assert (valid_adx >= 0.0).all() and (valid_adx <= 100.0).all()


def test_adx_regime_filtering():
    df = synthetic_5m_data(500)
    p = IntradayMomentumParams(use_adx_filter=True, adx_threshold=25.0)
    ind = compute_indicators(df, p)
    sig = compute_signals(df, p)

    for i in np.where(sig != FLAT)[0]:
        assert ind["adx"].iat[i] > p.adx_threshold
        assert ind["Volume"].iat[i] > (p.vol_mult * ind["vol_sma"].iat[i])


