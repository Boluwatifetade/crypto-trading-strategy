# Strategy Executor: Standalone Bybit V5 Execution Engine

A dedicated, isolated execution engine built directly in `trading strategy/strategy_executor/` for live forward-testing of the **1-Hour Trend-Following Pullback Strategy** on Bybit Testnet.

This engine connects directly to the Bybit V5 REST API without importing or modifying the main Tadex execution engine, eliminating risk to existing infrastructure.

---

## 1. Margin & Risk Model

The executor strictly enforces the specified production margin and risk rules:

| Rule | Specification | Bybit V5 Endpoint |
|---|---|---|
| **Margin Mode** | **CROSS Margin** (`tradeMode = 0`) | `POST /v5/position/switch-isolated` |
| **Leverage** | **Highest Available Leverage** (e.g. 150x on BTCUSDT) | `POST /v5/position/set-leverage` |
| **Flat Invariant** | Leverage and margin are **only configured when FLAT**. If a position is already open for the symbol, leverage modifications and duplicate entries are skipped. | `GET /v5/position/list` |
| **Risk-Based Sizing** | Risk Amount = **1.5% of Account Balance** ($0.75 on $50 account). Position Notional = $\text{Risk} / \text{SL\%} = \$0.75 / 0.025 = \mathbf{\$30.00}$. | `GET /v5/account/wallet-balance` |
| **Lot Size Rounding** | Raw Quantity is rounded **DOWN** to the exchange's lot step (`qtyStep`). | `GET /v5/market/instruments-info` |
| **Minimum Order Check** | If actual notional $< \$1.00$ Bybit minimum, the trade is logged and **skipped** to prevent risk rule violations. | Local validation |
| **Hard Exits** | Immediately sets **Stop Loss at 2.5%** and **Take Profit at 6.0%** upon entry. | `POST /v5/position/trading-stop` |

---

## 2. Safety Guards

1. **Environment Lock:**
   - Reads `TADEX_TARGET_ENV` from `.env`.
   - If `TADEX_TARGET_ENV != "testnet"`, live execution **immediately aborts** with a `PermissionError`.
2. **Dry-Run Simulation (`--dry-run`):**
   - Logs every API payload (instrument query, margin switch, leverage set, market order, trading stop) without placing live orders.
3. **No Duplicate Trades:**
   - Detects active positions (`size > 0`) before order submission to prevent double-exposure.

---

## 3. Configuration & Environment Setup

Create or verify `.env` in the root of `trading strategy/`:

```ini
# Must be exactly "testnet" for live execution
TADEX_TARGET_ENV=testnet

# Your Bybit Testnet API credentials
BYBIT_API_KEY="your_bybit_testnet_key"
BYBIT_API_SECRET="your_bybit_testnet_secret"
BYBIT_TESTNET=true
```

---

## 4. How to Run

### A. Dry-Run (Paper Testing without Live Orders)

Test with default $50 account balance:
```powershell
.\backtest_env\Scripts\python -m strategy_executor.execution_engine --symbol BTCUSDT --side Buy --dry-run
```

Test with `--allow-min-order` (bumps to 0.001 BTC if calculated size rounds down):
```powershell
.\backtest_env\Scripts\python -m strategy_executor.execution_engine --symbol BTCUSDT --side Buy --dry-run --allow-min-order
```

### B. Live Testnet Execution ($50 Account)

Ensure `TADEX_TARGET_ENV=testnet` is set in `.env`:
```powershell
.\backtest_env\Scripts\python -m strategy_executor.execution_engine --symbol BTCUSDT --side Buy
```

### C. Programmatic Usage in Python

```python
import asyncio
from strategy_executor import BybitClient, ExecutionConfig, ExecutionEngine

async def main():
    cfg = ExecutionConfig(
        symbol="BTCUSDT",
        risk_pct=0.015,       # 1.5% risk ($0.75 on $50)
        sl_pct=0.025,         # 2.5% stop loss
        tp_pct=0.060,         # 6.0% take profit
        dry_run=True,         # Set to False for live Testnet orders
    )
    
    engine = ExecutionEngine.from_env(dry_run=cfg.dry_run, config=cfg)
    async with engine.client:
        result = await engine.execute_trade(symbol="BTCUSDT", side="Buy")
        print("Trade Success:", result.success)
        print("Entry Price  :", result.entry_price)
        print("Stop Loss    :", result.stop_loss)
        print("Take Profit  :", result.take_profit)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Running the Unit Tests

Execute the mocked Bybit API test suite:

```powershell
.\backtest_env\Scripts\python -m pytest strategy_executor/test_execution.py -v
```

Verified Test Scenarios:
- `test_bybit_signature_generation`: HMAC-SHA256 signature canonicalization for GET and POST.
- `test_position_size_rounding_down`: Position sizing respects lot size step rounding.
- `test_skip_trade_if_notional_below_minimum`: Trade is skipped if notional $< \$1.00$.
- `test_leverage_set_only_when_flat`: Leverage and margin are only configured when flat.
- `test_stop_and_tp_price_calculations`: 2.5% SL and 6.0% TP prices calculated to tick precision.
- `test_safety_guard_env_check`: Enforces `TADEX_TARGET_ENV=testnet` requirement.
