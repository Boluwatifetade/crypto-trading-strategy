"""
Shared strategy logic for the Tadex strategy bot.

This module is the SINGLE source of truth for indicators and entry rules.
Both the backtest (strategy.py) and the live signal generator
(live_signal_generator.py) import from here, so what you backtest is exactly
what runs live.

Rule sets ("modes" / "strategies"):

rsi_reversal
    Long : Close > EMA(slow)  AND  RSI crosses ABOVE `rsi_oversold`  (oversold bounce)
    Short: Close < EMA(slow)  AND  RSI crosses BELOW `rsi_overbought` (overbought rejection)
    Exit : Fixed % SL (2%) and fixed % TP (4%, 1:2 R:R)

ema_cross
    Long : EMA(fast) crosses ABOVE EMA(slow) AND rsi_oversold < RSI < rsi_overbought
    Short: EMA(fast) crosses BELOW EMA(slow) AND rsi_oversold < RSI < rsi_overbought
    Exit : Fixed % SL (2%) and fixed % TP (4%, 1:2 R:R)

trend_pullback
    Long : Close > EMA 200 AND prev Close <= EMA 50 AND current Close > EMA 50
    Short: Close < EMA 200 AND prev Close >= EMA 50 AND current Close < EMA 50
    Exit : Fixed % SL (2%) and fixed % TP (4%, 1:2 R:R)

donchian_breakout
    Long : Close breaks above 20-period High (highest high of previous 20 bars)
    Short: Close breaks below 20-period Low (lowest low of previous 20 bars)
    Exit : Stop Loss at 2x ATR(14) from entry; Trailing Stop of 3% (no fixed TP)

All signals are evaluated on CLOSED candles only. A signal on bar i uses data
up to and including bar i, and is executed at the open of bar i+1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

Mode = Literal["rsi_reversal", "ema_cross", "trend_pullback", "donchian_breakout"]
MODES: Tuple[str, ...] = ("rsi_reversal", "ema_cross", "trend_pullback", "donchian_breakout")

LONG = 1
SHORT = -1
FLAT = 0


# ---------------------------------------------------------------------------
# Indicators (pure pandas / numpy, no TA-Lib)
# ---------------------------------------------------------------------------
def ema(values, n: int) -> np.ndarray:
    """Exponential moving average (same as TradingView's ta.ema)."""
    return pd.Series(values, dtype=float).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(values, n: int = 14) -> np.ndarray:
    """Wilder's RSI (same smoothing as TradingView / Bybit charts)."""
    close = pd.Series(values, dtype=float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 -> RSI 100 (or 50 if there was no movement at all)
    out = out.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
    out[avg_gain.isna()] = np.nan
    return out.to_numpy()


def atr(high, low, close, n: int = 14) -> np.ndarray:
    """Wilder's Average True Range (ATR)."""
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    prev_c = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return atr_series.to_numpy()


def donchian_channels(high, low, n: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Donchian Channel upper and lower bounds over the PREVIOUS n bars.
    Excludes the current bar so breakouts are evaluated against past highs/lows.
    """
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    upper = h.shift(1).rolling(n).max()
    lower = l.shift(1).rolling(n).min()
    return upper.to_numpy(), lower.to_numpy()


def adx(high, low, close, n: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wilder's Average Directional Index (ADX) and Directional Movement (+DI, -DI).
    Measures trend strength independently of direction (0-100).
    """
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_c = c.shift(1)

    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    up = h - prev_h
    down = prev_l - l

    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=l.index)

    tr_s = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    p_dm_s = plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    m_dm_s = minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    plus_di = 100.0 * (p_dm_s / tr_s.replace(0, np.nan))
    minus_di = 100.0 * (m_dm_s / tr_s.replace(0, np.nan))

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * (di_diff / di_sum.replace(0, np.nan)).fillna(0.0)

    adx_series = dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return adx_series.to_numpy(), plus_di.to_numpy(), minus_di.to_numpy()


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyParams:
    mode: str = "rsi_reversal"
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    sl_pct: float = 0.02          # 2% stop loss
    risk_reward: float = 2.0      # TP = 4% with defaults (1:2)
    donchian_period: int = 20     # Donchian channel breakout period
    atr_period: int = 14          # ATR period for volatility stops
    atr_mult: float = 2.0         # Multiple of ATR for Donchian SL (2x ATR)
    trailing_stop_pct: float = 0.03  # 3% trailing stop for Donchian TP
    adx_period: int = 14          # ADX period
    adx_threshold: float = 20.0   # ADX threshold for trend strength confirmation
    use_adx_filter: bool = False  # Enable ADX > threshold and DI directional alignment filter

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not (0 < self.sl_pct < 0.5):
            raise ValueError("sl_pct must be between 0 and 0.5")
        if self.risk_reward <= 0:
            raise ValueError("risk_reward must be > 0")
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be smaller than ema_slow")
        if not (0 < self.rsi_oversold < self.rsi_overbought < 100):
            raise ValueError("Need 0 < rsi_oversold < rsi_overbought < 100")
        if self.donchian_period <= 0:
            raise ValueError("donchian_period must be > 0")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be > 0")
        if self.atr_mult <= 0:
            raise ValueError("atr_mult must be > 0")
        if not (0 < self.trailing_stop_pct < 0.5):
            raise ValueError("trailing_stop_pct must be between 0 and 0.5")
        if self.adx_period <= 0:
            raise ValueError("adx_period must be > 0")
        if not (0 <= self.adx_threshold <= 100):
            raise ValueError("adx_threshold must be between 0 and 100")

    @property
    def tp_pct(self) -> float:
        return self.sl_pct * self.risk_reward

    @property
    def warmup_bars(self) -> int:
        """Bars needed before indicators are trustworthy."""
        if self.mode in ("rsi_reversal", "ema_cross", "trend_pullback"):
            return self.ema_slow
        elif self.mode == "donchian_breakout":
            return max(self.donchian_period, self.atr_period)
        return max(self.ema_slow, self.donchian_period, self.atr_period)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Indicators & Signals
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Return a copy of df with indicator columns needed for the strategies."""
    out = df.copy()
    close = out["Close"].to_numpy(dtype=float)
    high = out["High"].to_numpy(dtype=float)
    low = out["Low"].to_numpy(dtype=float)

    out["ema_fast"] = ema(close, p.ema_fast)
    out["ema_slow"] = ema(close, p.ema_slow)
    out["rsi"] = rsi(close, p.rsi_period)
    out["atr"] = atr(high, low, close, p.atr_period)
    up, dn = donchian_channels(high, low, p.donchian_period)
    out["donchian_high"] = up
    out["donchian_low"] = dn
    adx_v, p_di, m_di = adx(high, low, close, p.adx_period)
    out["adx"] = adx_v
    out["plus_di"] = p_di
    out["minus_di"] = m_di
    return out


def compute_signals(df: pd.DataFrame, p: StrategyParams) -> np.ndarray:
    """
    Vectorised entry signals: +1 long, -1 short, 0 nothing.

    Value at index i only depends on rows <= i (no look-ahead).
    The first `p.warmup_bars` rows are always 0.
    """
    ind = compute_indicators(df, p)
    close = ind["Close"].to_numpy(dtype=float)
    fast = ind["ema_fast"].to_numpy()
    slow = ind["ema_slow"].to_numpy()
    r = ind["rsi"].to_numpy()
    up = ind["donchian_high"].to_numpy()
    dn = ind["donchian_low"].to_numpy()
    adx_v = ind["adx"].to_numpy()
    p_di = ind["plus_di"].to_numpy()
    m_di = ind["minus_di"].to_numpy()

    prev_close = np.roll(close, 1)
    prev_fast = np.roll(fast, 1)
    prev_slow = np.roll(slow, 1)
    prev_r = np.roll(r, 1)
    prev_up = np.roll(up, 1)
    prev_dn = np.roll(dn, 1)

    sig = np.zeros(len(df), dtype=int)

    with np.errstate(invalid="ignore"):
        if p.mode == "rsi_reversal":
            long_ = (close > slow) & (prev_r <= p.rsi_oversold) & (r > p.rsi_oversold)
            short = (close < slow) & (prev_r >= p.rsi_overbought) & (r < p.rsi_overbought)

        elif p.mode == "ema_cross":
            rsi_ok = (r > p.rsi_oversold) & (r < p.rsi_overbought)
            long_ = (prev_fast <= prev_slow) & (fast > slow) & rsi_ok
            short = (prev_fast >= prev_slow) & (fast < slow) & rsi_ok

        elif p.mode == "trend_pullback":
            # 1. Trend Filter: EMA 200. Long only if Close > EMA 200. Short only if Close < EMA 200.
            # 2. Pullback & Crossover back:
            #    Long: Close was below or at EMA 50, now closes back above EMA 50.
            #    Short: Close was above or at EMA 50, now closes back below EMA 50.
            long_ = (close > slow) & (prev_close <= prev_fast) & (close > fast)
            short = (close < slow) & (prev_close >= prev_fast) & (close < fast)

            if p.use_adx_filter:
                long_ = long_ & (adx_v > p.adx_threshold) & (p_di > m_di)
                short = short & (adx_v > p.adx_threshold) & (m_di > p_di)

        elif p.mode == "donchian_breakout":
            # Breakout: Close breaks above 20-period High or below 20-period Low
            long_ = (prev_close <= prev_up) & (close > up)
            short = (prev_close >= prev_dn) & (close < dn)

        else:
            raise ValueError(f"Unknown mode {p.mode}")

    sig[long_] = LONG
    sig[short] = SHORT

    # Warm-up + first row (np.roll wraps around) + any NaN indicator rows
    sig[: max(p.warmup_bars, 1)] = FLAT
    if p.mode in ("rsi_reversal", "ema_cross"):
        sig[np.isnan(r) | np.isnan(prev_r) | np.isnan(slow)] = FLAT
    elif p.mode == "trend_pullback":
        sig[np.isnan(fast) | np.isnan(prev_fast) | np.isnan(slow)] = FLAT
        if p.use_adx_filter:
            sig[np.isnan(adx_v) | np.isnan(p_di) | np.isnan(m_di)] = FLAT
    elif p.mode == "donchian_breakout":
        sig[np.isnan(up) | np.isnan(prev_up) | np.isnan(dn) | np.isnan(prev_dn)] = FLAT

    return sig


def stop_and_target(
    entry: float,
    side: int,
    p: StrategyParams,
    atr_val: Optional[float] = None
) -> Tuple[float, Optional[float]]:
    """
    Return (stop_loss, take_profit) prices for an entry.
    Take profit is None when trailing stop is used (e.g. donchian_breakout).
    """
    if p.mode == "donchian_breakout":
        if atr_val is not None and not np.isnan(atr_val) and atr_val > 0:
            sl_dist = p.atr_mult * atr_val
        else:
            sl_dist = entry * p.sl_pct
        if side == LONG:
            return entry - sl_dist, None
        elif side == SHORT:
            return entry + sl_dist, None
        raise ValueError("side must be LONG (1) or SHORT (-1)")
    else:
        if side == LONG:
            return entry * (1 - p.sl_pct), entry * (1 + p.tp_pct)
        elif side == SHORT:
            return entry * (1 + p.sl_pct), entry * (1 - p.tp_pct)
        raise ValueError("side must be LONG (1) or SHORT (-1)")
