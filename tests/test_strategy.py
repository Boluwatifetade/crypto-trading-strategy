"""Tests for the strategy bot. Run: python -m pytest -q"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy_core import (
    LONG,
    MODES,
    SHORT,
    StrategyParams,
    atr,
    compute_indicators,
    compute_signals,
    donchian_channels,
    rsi,
    stop_and_target,
)
from tadex_bridge import format_signal, verify_with_tadex

TADEX_REPO = Path(os.getenv("TADEX_REPO_PATH", r"C:\Users\user\Documents\Tadex"))
HAS_TADEX = (TADEX_REPO / "bybit-client" / "bybit_client" / "signal_parser.py").exists()


def synthetic_ohlc(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    close = 0.1 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "Open": close,
        "High": close * 1.008,
        "Low": close * 0.992,
        "Close": close,
        "Volume": 1.0,
    }, index=idx)


def test_rsi_bounds_and_extremes():
    up = np.arange(1, 100, dtype=float)
    assert np.nanmax(rsi(up)) == pytest.approx(100.0)
    vals = rsi(synthetic_ohlc()["Close"].to_numpy())
    finite = vals[~np.isnan(vals)]
    assert finite.min() >= 0 and finite.max() <= 100


def test_atr_and_donchian_channels():
    df = synthetic_ohlc()
    a = atr(df["High"], df["Low"], df["Close"], 14)
    finite_a = a[~np.isnan(a)]
    assert len(finite_a) > 0
    assert (finite_a > 0).all()

    up, dn = donchian_channels(df["High"], df["Low"], 20)
    valid_mask = ~np.isnan(up) & ~np.isnan(dn)
    assert (up[valid_mask] >= dn[valid_mask]).all()


@pytest.mark.parametrize("mode", MODES)
def test_no_lookahead(mode):
    """Signals at bar i must not change when future bars are added."""
    df = synthetic_ohlc()
    p = StrategyParams(mode=mode)
    full = compute_signals(df, p)
    for cut in (600, 900, 1200):
        assert np.array_equal(compute_signals(df.iloc[:cut], p), full[:cut])


@pytest.mark.parametrize("mode", MODES)
def test_warmup_is_flat_and_signals_exist(mode):
    p = StrategyParams(mode=mode)
    sig = compute_signals(synthetic_ohlc(), p)
    assert not sig[: p.warmup_bars].any()
    assert set(np.unique(sig)) <= {-1, 0, 1}


def test_rsi_reversal_rule_semantics():
    df = synthetic_ohlc()
    p = StrategyParams(mode="rsi_reversal")
    ind = compute_indicators(df, p)
    sig = compute_signals(df, p)
    for i in np.where(sig == LONG)[0]:
        assert ind["Close"].iat[i] > ind["ema_slow"].iat[i]
        assert ind["rsi"].iat[i - 1] <= 30 < ind["rsi"].iat[i]
    for i in np.where(sig == SHORT)[0]:
        assert ind["Close"].iat[i] < ind["ema_slow"].iat[i]
        assert ind["rsi"].iat[i - 1] >= 70 > ind["rsi"].iat[i]


def test_trend_pullback_rule_semantics():
    df = synthetic_ohlc()
    p = StrategyParams(mode="trend_pullback")
    ind = compute_indicators(df, p)
    sig = compute_signals(df, p)
    for i in np.where(sig == LONG)[0]:
        assert ind["Close"].iat[i] > ind["ema_slow"].iat[i]
        assert ind["Close"].iat[i - 1] <= ind["ema_fast"].iat[i - 1]
        assert ind["Close"].iat[i] > ind["ema_fast"].iat[i]
    for i in np.where(sig == SHORT)[0]:
        assert ind["Close"].iat[i] < ind["ema_slow"].iat[i]
        assert ind["Close"].iat[i - 1] >= ind["ema_fast"].iat[i - 1]
        assert ind["Close"].iat[i] < ind["ema_fast"].iat[i]


def test_donchian_breakout_rule_semantics():
    df = synthetic_ohlc()
    p = StrategyParams(mode="donchian_breakout")
    ind = compute_indicators(df, p)
    sig = compute_signals(df, p)
    for i in np.where(sig == LONG)[0]:
        assert ind["Close"].iat[i - 1] <= ind["donchian_high"].iat[i - 1]
        assert ind["Close"].iat[i] > ind["donchian_high"].iat[i]
    for i in np.where(sig == SHORT)[0]:
        assert ind["Close"].iat[i - 1] >= ind["donchian_low"].iat[i - 1]
        assert ind["Close"].iat[i] < ind["donchian_low"].iat[i]


def test_stop_and_target_1_to_2():
    p = StrategyParams(mode="trend_pullback")
    assert stop_and_target(100, LONG, p) == pytest.approx((98, 104))
    assert stop_and_target(100, SHORT, p) == pytest.approx((102, 96))


def test_donchian_stop_and_target():
    p = StrategyParams(mode="donchian_breakout", atr_mult=2.0)
    sl_long, tp_long = stop_and_target(100, LONG, p, atr_val=2.5)
    assert sl_long == pytest.approx(95.0)  # 100 - 2 * 2.5
    assert tp_long is None

    sl_short, tp_short = stop_and_target(100, SHORT, p, atr_val=2.5)
    assert sl_short == pytest.approx(105.0)  # 100 + 2 * 2.5
    assert tp_short is None


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        StrategyParams(mode="nope")
    with pytest.raises(ValueError):
        StrategyParams(ema_fast=200, ema_slow=50)
    with pytest.raises(ValueError):
        StrategyParams(donchian_period=-5)


@pytest.mark.skipif(not HAS_TADEX, reason="Tadex repo not available")
@pytest.mark.parametrize("symbol,side,tp,sl", [
    ("DOGEUSDT", LONG, "0.08500", "0.08000"),
    ("DOGEUSDT", SHORT, "0.07680", "0.08160"),
    ("BTCUSDT", LONG, "104000.0", "98000.0"),
    ("1000PEPEUSDT", SHORT, "0.009216", "0.009792"),
    ("ETHUSDT", SHORT, "3360.00", "3570.00"),
])
def test_signal_text_roundtrips_through_tadex_parser(symbol, side, tp, sl):
    text = format_signal(symbol, side, tp, sl, 10)
    assert verify_with_tadex(text, symbol, side, tp, sl, 10, TADEX_REPO) == []
