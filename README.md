# Tadex Strategy Bot — Backtest → Testnet Signal Generator

A rule-based strategy you can **backtest honestly**. If it passes, the same code can feed signals into the Tadex pipeline on **Testnet**.

```
fetch_data.py ──► data/*.csv ──► strategy.py ──► results/*_verdict.json (PASS/FAIL)
                                                        │ (gate)
Bybit candles ──► live_signal_generator.py ──► Tadex parser check ──► public.signals (status=raw)
                                                                          │
                                                       Tadex dispatcher ──► subscribers' Bybit Testnet
```

| File | Purpose |
|---|---|
| `strategy_executor/` | Standalone Bybit V5 execution engine (CROSS margin, highest leverage, risk sizing, hard stops). |
| `trend_pullback_1h.py` | 1-Hour Trend Pullback backtest runner with comparative matrix and Safety Gate enforcement. |
| `strategy_core.py` | Indicators + entry rules for swing strategies (RSI, EMA, Trend Pullback, Donchian, ADX/DI). |
| `intraday_momentum_core.py` | Core logic, indicators, and execution engine for 5m Intraday Momentum. |
| `fetch_5m_data.py` | Multi-threaded downloader for 1 year of 5m Bybit candles with auto-resume. |
| `intraday_backtest.py` | Backtester across 1 year of 5m data for BTC, ETH, SOL, DOGE, XRP. |
| `intraday_scanner.py` | Live 5-minute scanner of top 20 Bybit pairs by volume with safety gate enforcement. |
| `tadex_bridge.py` | Formats signals, verifies with Tadex parser, and inserts into `public.signals`. |

## The rules

**`intraday_momentum`** (5-Minute Intraday Volatility & Volume Breakout)
- Scan: Top 20 Bybit USDT perpetuals by 24h volume on 5-minute candles.
- Volatility Filter: 5m ATR(14) / Close > 0.005 (0.5% volatility filter).
- Volume Confirmation: Breakout candle volume > 1.5x Volume SMA(20).
- Long Trigger: Close > High of previous 3 candles.
- Short Trigger: Close < Low of previous 3 candles.
- Risk & Exits: Fixed SL 0.8%, Fixed TP 2.4% (1:3 R:R), Time Stop of 48 candles (4 hours), 0.05% slippage on SL.
- Position Sizing: Risk-Based Sizing — Notional = `(Equity * 2%) / StopLoss%` (e.g., $1.00 risk on $50 equity = $125 notional, matching Bybit cross-margin mechanics).

**`trend_pullback`** (Trend pullback to 50 EMA with 200 EMA trend filter)
- Long: Close > EMA200 **and** Close dips <= EMA50 and closes back > EMA50
- Short: Close < EMA200 **and** Close rises >= EMA50 and closes back < EMA50
- Exits: Fixed SL 2%, TP 4% (1:2 R:R)

**`donchian_breakout`** (20-period Donchian Channel breakout with Trailing Stop)
- Long: Close breaks above 20-period High (highest high of previous 20 bars)
- Short: Close breaks below 20-period Low (lowest low of previous 20 bars)
- Exits: Initial Stop Loss at 2x ATR(14) from entry; Trailing Stop of 3% (no fixed TP)

**Costs modelled:**
- Bybit Taker: 0.055% per side (0.110% round-trip)
- Bybit Maker: 0.020% per side (0.040% round-trip)
- Slippage: 0.050% execution penalty on Stop Loss market fills

---

## 1. Setup (once)

```powershell
cd "C:\Users\user\Documents\trading strategy"
python -m venv backtest_env
backtest_env\Scripts\activate
pip install -r requirements.txt
python -m pytest -q          # all tests should pass
```

## 2. Download data

```powershell
python fetch_data.py                                      # DOGE perp, 1h, since 2025-01-01
python fetch_data.py --symbol BTC/USDT:USDT
python fetch_data.py --symbol ETH/USDT:USDT --since 2024-01-01
```
Files are saved as `data/bybit_DOGEUSDT_1h.csv`. If Bybit is geo-blocked for you, add `--exchange binance`. Use the same `--exchange` for the backtest and the live bot.

## 3. Backtest

```powershell
python strategy.py --symbol BTC/USDT:USDT --timeframe 4h --strategy trend_pullback
python strategy.py --symbol BTC/USDT:USDT --timeframe 4h --strategy donchian_breakout
python strategy.py --symbol BTC/USDT:USDT --timeframe 4h --strategy trend_pullback --optimize
python strategy.py --symbol BTC/USDT:USDT --timeframe 4h --strategy donchian_breakout --optimize
python strategy.py --plot                  # open interactive chart

# 1-Hour Trend-Following Pullback (BTC/USDT) — Gate Passed!
python fetch_data.py --symbol BTC/USDT:USDT --timeframe 1h  # downloads 1 year 1h candles
python trend_pullback_1h.py --compare-all                   # runs comparative parameter matrix
python trend_pullback_1h.py --sl 0.025 --tp 0.06 --risk 0.015  # production sweet spot (PASSES GATE)

# Intraday Momentum Scanner & Backtest (5m candles)
python fetch_5m_data.py                    # downloads 1 year of 5m data for BTC, ETH, SOL, DOGE, XRP
python intraday_backtest.py                # runs BTC/USDT 1-year backtest with parameter comparison table
python intraday_scanner.py --once          # runs BTC/USDT scan once in dry-run mode
python intraday_scanner.py --top20 --once  # runs top 20 Bybit pair scan once
python intraday_scanner.py                 # continuous 5m scanning loop
```

### 1-Hour Trend-Following Pullback (BTC/USDT) — Matrix Results

| Configuration | Trades | Win Rate | Profit Factor | Max Drawdown | Total Return | Safety Gate |
|---|---|---|---|---|---|---|
| **User Baseline** (SL 2.0%, TP 6.0%, Risk 2.0%) | 36 | 30.56% | 1.136 | 20.28% | +8.72% | FAIL (DD > 20%) |
| **Risk-Managed Baseline** (SL 2.0%, TP 6.0%, Risk 1.5%) | 36 | 30.56% | 1.151 | 15.56% | +6.94% | FAIL (PF < 1.20) |
| **Sweet Spot Gate Winner** (SL 2.5%, TP 6.0%, Risk 1.5%) | 33 | 36.36% | **1.215** | **13.26%** | **+8.21%** | **PASS** |
| **Conservative Winner** (SL 2.5%, TP 6.0%, Risk 1.0%) | 33 | 36.36% | **1.231** | **9.01%** | **+5.63%** | **PASS** |
| **Unfiltered Benchmark** (SL 2.5%, TP 6.0%, Risk 1.5%) | 57 | 36.84% | **1.251** | **14.86%** | **+16.20%** | **PASS** |
| **Unfiltered High-R:R** (SL 2.5%, TP 7.5%, Risk 1.5%) | 50 | 36.00% | **1.516** | **16.22%** | **+30.49%** | **PASS** |


### Intraday Momentum: Final Optimization & Post-Only Retest Results (BTC/USDT 5M)

| Metric | No Filter (Post-Only Limit) | ADX Filter (Post-Only Limit) | No Filter (Market Fill) | ADX Filter (Market Fill) |
|---|---|---|---|---|
| **Entry Order Type** | Post-Only Limit Retest | Post-Only Limit Retest | Market Order on Open | Market Order on Open |
| **Fees (Entry / Exit)** | Maker 0.02% / Taker 0.055% | Maker 0.02% / Taker 0.055% | Maker 0.02% / Maker 0.02% | Maker 0.02% / Maker 0.02% |
| **Total Trades** | 49 | **22** (< 30 min required) | **81** | **46** |
| **Win Rate** | 30.61% | **18.18%** | **32.10%** | **26.09%** |
| **Profit Factor** | 0.725 | **0.315** | **1.058** | **0.781** |
| **Avg R:R Ratio** | 1.64 | 1.42 | 2.24 | 2.21 |
| **Total Return ($50 start)** | -20.19% | **-25.43%** | **+2.18%** | **-16.44%** |
| **Max Drawdown** | 20.19% | **27.86%** | **20.41%** | **26.82%** |
| **Ending Equity** | $39.91 | **$37.29** | **$51.09** | **$41.78** |

#### Crucial Quantitative Findings:
1. **The "Adverse Selection" Law of Post-Only Limit Breakouts:**
   - In explosive, winning breakouts, price accelerates forward and *never retests* the 3-candle high within 2 bars. Limit orders placed at the breakout level are left behind.
   - When price *does* pull back to retest the breakout level within 1-2 bars, it indicates the breakout is weak or failing. The limit order gets filled just as sellers push through the 0.8% stop loss, collapsing the win rate to **18.18%** and Profit Factor to **0.315**.
2. **"No Filter" vs. "ADX Filter":**
   - The "No Filter" setup outperforms the "ADX Filter" setup across both limit orders (PF 0.725 vs 0.315) and market orders (PF 1.058 vs 0.781).
   - ADX(14) > 25 on 5m candles creates late entry on exhausted candles.
3. **Bybit Minimum Size Check:**
   - Enforced `$1.00` minimum order notional. Because `$50` equity with 2% risk and 0.8% stop loss yields `$125` notional ($1.00 risk / 0.008 SL), all eligible setups cleared the minimum size rule.

> **Safety Gate Verdict:** Profit Factor (0.315) < 1.20, Max Drawdown (27.86%) > 20%, Trades (22) < 30 $\rightarrow$ **STRICT FAIL**. Tadex remains **DISCONNECTED** from live insertion.



You get three blocks of output:
- **IN-SAMPLE:** the first 70% of the data. If you used `--optimize`, this number is flattering by construction.
- **OUT-OF-SAMPLE:** the last 30%, which the parameters never saw. **This is the number that matters.**
- **FULL PERIOD:** used for the chart and the trade list.

**Verdict gate** (both out-of-sample and full period must pass): ≥ 30 trades, Sharpe > 0, return > 0, max drawdown ≤ 25%. The result is written to `results/<name>_verdict.json`.

Sizing: `--leverage 10 --margin-fraction 0.10` means each trade posts 10% of equity as margin at 10x, so notional ≈ 1× equity. A 2% stop then costs about 2% of equity plus fees.

> If the verdict is **FAIL**, stop. Do not connect the strategy to Tadex. Try other symbols or timeframes, or `--optimize`. Treat a strategy that only passes in-sample as overfit.

## 4. Dry-run the live generator (safe, nothing is inserted)

```powershell
python live_signal_generator.py --once     # one evaluation
python live_signal_generator.py            # loop: wakes 20s after every candle close
```
Each signal is printed in the canonical Tadex/Cosmas format:
```
DOGEUSDT
Long position
Entry: MARKET
TP: 0.08500
SL: 0.08000
Leverage 10x
```
The bot also runs each signal through `Tadex/bybit-client/bybit_client/signal_parser.py` to confirm the dispatcher will read the symbol, side, TP, SL and leverage exactly as intended.

Virtual trades and their outcomes are recorded in `state/virtual_trades.csv`. That file is your paper-trading record.

## 5. Connect to Tadex Testnet

1. **Create a dedicated provider** in Tadex (admin or provider flow), e.g. *"Tadex Strategy Bot"*. **Do not reuse the Cosmas provider id.**
2. Subscribe **only your testnet account** to that provider. Make sure that account's exchange environment is set to Testnet. Each subscriber's account settings decide whether a trade goes to testnet or mainnet, so this subscription is what keeps the bot on testnet.
3. Run `copy .env.example .env` and fill in `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (the same project as your Tadex stack) and `TADEX_STRATEGY_PROVIDER_ID`.
4. Make sure the Tadex dispatcher is running.
5. Start:
   ```powershell
   python live_signal_generator.py --insert
   ```

With `--insert`, the bot refuses to start unless:
- `TADEX_TARGET_ENV=testnet` is set in `.env`
- `TADEX_REPO_PATH` is set, so every signal is checked by Tadex's parser
- a **PASS** verdict exists for each symbol, produced with **identical parameters** (if you used `--optimize`, pass the optimised values to the live bot too)
- the provider id exists in `public.providers`

Inserted rows have `status='raw'`, `source_type='api'` and `metadata.source='tadex_strategy_bot'`. The Tadex dispatcher handles them exactly like Telegram signals.

### Keep it running
- **Loop mode:** leave `python live_signal_generator.py --insert` running (or run it as a systemd service on the VPS).
- **Scheduler mode:** use Windows Task Scheduler or cron to run `... --once --insert` every hour at minute 1. State in `state/live_state.json` prevents duplicate signals.

## 6. Forward-test for at least 30 days

Compare the Tadex testnet fills with `state/virtual_trades.csv` and with what the backtest predicted. Consider mainnet only if the live results look like the out-of-sample backtest after a meaningful number of trades. **Never point this bot at mainnet without re-reviewing every safeguard above.**

---

## 7. License & Disclaimer

- **License:** Distributed under the ISC/Permissive license. See [LICENSE](LICENSE) for details.
- **Risk Disclaimer:** This software is for technical research and educational purposes only. See [DISCLAIMER.md](DISCLAIMER.md) for the full software and market risk notice.

