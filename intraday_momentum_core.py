"""
Intraday Momentum Strategy Core (5-minute Volatility Expansion + Volume Breakout).

Single source of truth for:
- 5m ATR(14) volatility filter (ATR / Close > 0.005)
- 20-period Volume SMA (Breakout Volume > 1.5 * Volume SMA)
- 3-candle High/Low breakout detection
- Fixed 0.8% Stop Loss, 2.4% Take Profit (1:3 R:R)
- 48-candle (4-hour) Time Stop
- 0.05% Slippage on Stop Loss execution
- Risk-Based Position Sizing: Notional = (Equity * Risk%) / StopLoss%
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LONG = 1
SHORT = -1
FLAT = 0


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
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


def volume_sma(volume, n: int = 20) -> np.ndarray:
    """SMA of volume over previous n bars (excludes current bar)."""
    return pd.Series(volume, dtype=float).shift(1).rolling(n).mean().to_numpy()


def recent_extremes(high, low, n: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Highest high and lowest low of the previous n bars."""
    h = pd.Series(high, dtype=float).shift(1).rolling(n).max()
    l = pd.Series(low, dtype=float).shift(1).rolling(n).min()
    return h.to_numpy(), l.to_numpy()


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
class IntradayMomentumParams:
    atr_period: int = 14
    vol_sma_period: int = 20
    vol_mult: float = 2.0                 # 2.0x Volume SMA (user request)
    volatility_threshold: float = 0.005   # 0.5% ATR / Close
    breakout_bars: int = 3
    adx_period: int = 14                  # ADX period
    adx_threshold: float = 25.0           # Regime filter: ADX > 25 (strong trend)
    use_adx_filter: bool = True           # Toggle for ADX regime filter
    use_post_only_limit: bool = True      # Simulate Post-Only Limit entry at breakout level
    limit_retest_bars: int = 2            # Retest window: cancel order if unfilled within 2 candles
    limit_offset_pct: float = 0.0         # Offset below breakout price (0.0 = at breakout price)
    min_notional: float = 1.0             # Bybit minimum order size ($1.00)
    sl_pct: float = 0.008                 # 0.8% Stop Loss (widened to breathe above 5m noise)
    tp_pct: float = 0.024                 # 2.4% Take Profit (1:3 R:R)
    time_stop_bars: int = 48              # 4 hours (48 x 5m candles)
    slippage_pct: float = 0.0005          # 0.05% slippage on Stop Loss fills
    taker_fee: float = 0.00055            # 0.055% Bybit taker fee per side (used on market exit)
    maker_fee: float = 0.00020            # 0.020% Bybit maker fee per side (used on limit entry)
    risk_per_trade_pct: float = 0.02      # 2% account risk per trade ($1.00 on $50 account)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame, p: IntradayMomentumParams) -> pd.DataFrame:
    """Calculate and return DataFrame with all strategy indicators."""
    out = df.copy()
    c = out["Close"].to_numpy(dtype=float)
    h = out["High"].to_numpy(dtype=float)
    l = out["Low"].to_numpy(dtype=float)
    v = out["Volume"].to_numpy(dtype=float)

    out["atr"] = atr(h, l, c, p.atr_period)
    out["atr_ratio"] = out["atr"] / c
    out["vol_sma"] = volume_sma(v, p.vol_sma_period)
    hi3, lo3 = recent_extremes(h, l, p.breakout_bars)
    out["high_recent"] = hi3
    out["low_recent"] = lo3
    adx_vals, p_di, m_di = adx(h, l, c, p.adx_period)
    out["adx"] = adx_vals
    out["plus_di"] = p_di
    out["minus_di"] = m_di
    return out


def compute_signals(df: pd.DataFrame, p: Optional[IntradayMomentumParams] = None) -> np.ndarray:
    """
    Vectorised entry signals: +1 (Long), -1 (Short), 0 (Flat).
    Evaluated on closed candles only (strictly no look-ahead).
    """
    if p is None:
        p = IntradayMomentumParams()

    ind = compute_indicators(df, p)
    c = ind["Close"].to_numpy(dtype=float)
    v = ind["Volume"].to_numpy(dtype=float)
    atr_r = ind["atr_ratio"].to_numpy()
    vol_sma = ind["vol_sma"].to_numpy()
    hi = ind["high_recent"].to_numpy()
    lo = ind["low_recent"].to_numpy()
    adx_arr = ind["adx"].to_numpy()

    # Rule checks
    volatility_ok = atr_r > p.volatility_threshold
    volume_ok = v > (p.vol_mult * vol_sma)
    adx_ok = (adx_arr > p.adx_threshold) if p.use_adx_filter else np.ones(len(df), dtype=bool)
    long_breakout = c > hi
    short_breakout = c < lo

    sig = np.zeros(len(df), dtype=int)
    long_mask = volatility_ok & volume_ok & adx_ok & long_breakout
    short_mask = volatility_ok & volume_ok & adx_ok & short_breakout

    sig[long_mask] = LONG
    sig[short_mask] = SHORT

    # Filter out warm-up period and any NaN rows
    warmup = max(p.atr_period * 2, p.vol_sma_period, p.breakout_bars) + 1
    sig[:warmup] = FLAT
    sig[np.isnan(atr_r) | np.isnan(vol_sma) | np.isnan(hi) | np.isnan(lo) | np.isnan(adx_arr)] = FLAT
    return sig


# ---------------------------------------------------------------------------
# Trade Simulation (Risk-Based Position Sizing)
# ---------------------------------------------------------------------------
@dataclass
class TradeRecord:
    symbol: str
    side: int
    entry_bar: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_bar: int
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str  # 'TP', 'SL', 'TIME_STOP'
    bars_held: int
    gross_return: float
    net_return: float
    pnl_usd: float
    notional: float
    account_equity_before: float
    account_equity_after: float


def simulate_trades(
    df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    p: Optional[IntradayMomentumParams] = None,
    initial_cash: float = 50.0,
    use_maker_fee: bool = False,
) -> Tuple[List[TradeRecord], pd.DataFrame]:
    """
    Simulate execution with Risk-Based Position Sizing:
    - Account starts at initial_cash ($50.0)
    - Position Notional = (Equity * Risk_Pct) / Stop_Loss_Pct (e.g. $1 / 0.008 = $125)
    - Fills on open of the bar immediately after signal
    - Enforces 1 open position per pair
    - Checks intra-bar SL (0.8%), TP (2.4%), and 4-hour (48 bars) Time Stop
    - Simulates 0.05% slippage on Stop Loss fills
    - Applies Bybit taker (0.055%/side) or maker (0.020%/side) fees
    """
    if p is None:
        p = IntradayMomentumParams()

    ind = compute_indicators(df, p)
    signals = compute_signals(df, p)
    n = len(df)
    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    times = df.index
    hi3 = ind["high_recent"].to_numpy(dtype=float)
    lo3 = ind["low_recent"].to_numpy(dtype=float)

    trades: List[TradeRecord] = []
    equity = initial_cash
    equity_curve = [equity] * n

    in_pos = False
    pos_side = FLAT
    pos_entry_bar = 0
    pos_entry_time = None
    pos_entry_price = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_notional = 0.0
    pos_equity_before = 0.0

    # Pending Post-Only limit order: (side, limit_price, bar_placed, expiry_bar)
    pending_order: Optional[Tuple[int, float, int, int]] = None

    if p.use_post_only_limit:
        # Maker fee (0.02%) on entry + Taker fee (0.055%) on exit
        fee_rate = p.maker_fee + p.taker_fee
    else:
        fee_per_side = p.maker_fee if use_maker_fee else p.taker_fee
        fee_rate = fee_per_side * 2.0

    for i in range(1, n):
        # 1. If in position, check exits first on current bar i
        if in_pos:
            bars_held = i - pos_entry_bar
            exit_reason = None
            exit_price = 0.0

            if pos_side == LONG:
                hit_sl = lows[i] <= pos_sl
                hit_tp = highs[i] >= pos_tp
                if hit_sl and hit_tp:
                    # Conservative: assume SL triggered first with slippage
                    exit_reason = "SL"
                    exit_price = pos_sl * (1.0 - p.slippage_pct)
                elif hit_sl:
                    exit_reason = "SL"
                    exit_price = pos_sl * (1.0 - p.slippage_pct)
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = pos_tp
                elif bars_held >= p.time_stop_bars:
                    exit_reason = "TIME_STOP"
                    exit_price = closes[i]

            elif pos_side == SHORT:
                hit_sl = highs[i] >= pos_sl
                hit_tp = lows[i] <= pos_tp
                if hit_sl and hit_tp:
                    exit_reason = "SL"
                    exit_price = pos_sl * (1.0 + p.slippage_pct)
                elif hit_sl:
                    exit_reason = "SL"
                    exit_price = pos_sl * (1.0 + p.slippage_pct)
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = pos_tp
                elif bars_held >= p.time_stop_bars:
                    exit_reason = "TIME_STOP"
                    exit_price = closes[i]

            if exit_reason is not None:
                if pos_side == LONG:
                    gross_ret = (exit_price - pos_entry_price) / pos_entry_price
                else:
                    gross_ret = (pos_entry_price - exit_price) / pos_entry_price

                net_ret = gross_ret - fee_rate
                pnl = pos_notional * net_ret
                equity_after = max(equity + pnl, 0.0)

                trades.append(
                    TradeRecord(
                        symbol=symbol,
                        side=pos_side,
                        entry_bar=pos_entry_bar,
                        entry_time=pos_entry_time,
                        entry_price=pos_entry_price,
                        exit_bar=i,
                        exit_time=times[i],
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        bars_held=bars_held,
                        gross_return=gross_ret,
                        net_return=net_ret,
                        pnl_usd=pnl,
                        notional=pos_notional,
                        account_equity_before=pos_equity_before,
                        account_equity_after=equity_after,
                    )
                )

                equity = equity_after
                in_pos = False
                pos_side = FLAT

        # 2. Check pending Post-Only limit order if flat
        if not in_pos and pending_order is not None:
            p_side, p_limit, p_placed, p_expiry = pending_order
            filled = False
            if p_side == LONG and lows[i] <= p_limit:
                filled = True
            elif p_side == SHORT and highs[i] >= p_limit:
                filled = True

            if filled:
                risk_amount = equity * p.risk_per_trade_pct
                notional = risk_amount / p.sl_pct
                # Bybit Minimum Size Check ($1.00 minimum notional)
                if notional >= p.min_notional and equity > 1.0:
                    in_pos = True
                    pos_side = p_side
                    pos_entry_bar = i
                    pos_entry_time = times[i]
                    pos_entry_price = p_limit
                    pos_notional = notional
                    pos_equity_before = equity
                    if pos_side == LONG:
                        pos_sl = pos_entry_price * (1.0 - p.sl_pct)
                        pos_tp = pos_entry_price * (1.0 + p.tp_pct)
                    else:
                        pos_sl = pos_entry_price * (1.0 + p.sl_pct)
                        pos_tp = pos_entry_price * (1.0 - p.tp_pct)
                pending_order = None
            elif i >= p_expiry:
                # Cancel order after limit_retest_bars without fill
                pending_order = None

        # 3. Check for new breakout signal from bar i-1 if flat and no pending order
        if not in_pos and pending_order is None and equity > 1.0 and i > 0 and signals[i - 1] != FLAT:
            sig_side = signals[i - 1]
            if p.use_post_only_limit:
                limit_p = hi3[i - 1] * (1.0 - p.limit_offset_pct) if sig_side == LONG else lo3[i - 1] * (1.0 + p.limit_offset_pct)
                p_expiry = i + p.limit_retest_bars - 1
                pending_order = (sig_side, limit_p, i, p_expiry)

                # Check if bar i immediately fills the limit order
                filled = False
                if sig_side == LONG and lows[i] <= limit_p:
                    filled = True
                elif sig_side == SHORT and highs[i] >= limit_p:
                    filled = True

                if filled:
                    risk_amount = equity * p.risk_per_trade_pct
                    notional = risk_amount / p.sl_pct
                    # Bybit Minimum Size Check ($1.00 minimum notional)
                    if notional >= p.min_notional and equity > 1.0:
                        in_pos = True
                        pos_side = sig_side
                        pos_entry_bar = i
                        pos_entry_time = times[i]
                        pos_entry_price = limit_p
                        pos_notional = notional
                        pos_equity_before = equity
                        if pos_side == LONG:
                            pos_sl = pos_entry_price * (1.0 - p.sl_pct)
                            pos_tp = pos_entry_price * (1.0 + p.tp_pct)
                        else:
                            pos_sl = pos_entry_price * (1.0 + p.sl_pct)
                            pos_tp = pos_entry_price * (1.0 - p.tp_pct)
                    pending_order = None
            else:
                # Market order fill on open of bar i
                risk_amount = equity * p.risk_per_trade_pct
                notional = risk_amount / p.sl_pct
                if notional >= p.min_notional:
                    pos_side = sig_side
                    pos_entry_bar = i
                    pos_entry_time = times[i]
                    pos_entry_price = opens[i]
                    pos_equity_before = equity
                    pos_notional = notional
                    if pos_side == LONG:
                        pos_sl = pos_entry_price * (1.0 - p.sl_pct)
                        pos_tp = pos_entry_price * (1.0 + p.tp_pct)
                    else:
                        pos_sl = pos_entry_price * (1.0 + p.sl_pct)
                        pos_tp = pos_entry_price * (1.0 - p.tp_pct)
                    in_pos = True

        equity_curve[i] = equity

    df_equity = pd.DataFrame({"equity": equity_curve}, index=times)
    return trades, df_equity
