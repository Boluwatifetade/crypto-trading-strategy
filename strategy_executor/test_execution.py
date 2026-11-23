"""
Unit Tests for Standalone Strategy Executor.
Run: python -m pytest strategy_executor/test_execution.py -v
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strategy_executor.bybit_client import BybitAPIError, BybitClient
from strategy_executor.execution_engine import (
    ExecutionConfig,
    ExecutionEngine,
    OrderResult,
    get_decimals,
    round_down_step,
    round_to_tick,
)


# ---------------------------------------------------------------------------
# 1. Signature & Canonicalization Unit Tests
# ---------------------------------------------------------------------------
def test_bybit_signature_generation():
    """Verify HMAC-SHA256 signature conforms to Bybit V5 canonical rules."""
    client = BybitClient(api_key="test_api_key", api_secret="test_api_secret", testnet=True)
    ts = 1700000000000

    # GET request signature (parameters sorted alphabetically)
    get_params = {"symbol": "BTCUSDT", "category": "linear", "limit": 10}
    sig_get = client.generate_signature(ts, "GET", params=get_params)
    assert isinstance(sig_get, str)
    assert len(sig_get) == 64  # SHA256 hex length
    assert sig_get == sig_get.lower()  # MUST be lowercase

    # POST request signature (compact JSON, keys sorted alphabetically)
    post_body = {"symbol": "BTCUSDT", "side": "Buy", "qty": "0.001", "orderType": "Market"}
    sig_post = client.generate_signature(ts, "POST", body=post_body)
    assert isinstance(sig_post, str)
    assert len(sig_post) == 64
    assert sig_post == sig_post.lower()


# ---------------------------------------------------------------------------
# 2. Position Size Rounding & Math Tests
# ---------------------------------------------------------------------------
def test_position_size_rounding_down():
    """Requirement 7a: Position size calculation handles rounding down correctly."""
    client = MagicMock(spec=BybitClient)
    cfg = ExecutionConfig(risk_pct=0.015, sl_pct=0.025)
    engine = ExecutionEngine(client=client, config=cfg)

    # Balance $50.00, Risk 1.5% = $0.75. Notional = $0.75 / 0.025 = $30.00
    # Price = $68,432.10
    # Raw Qty = 30.00 / 68432.10 = 0.00043839...
    # Step = 0.001 -> rounded down should be 0.000 (since 0.000438 < 0.001)
    qty, notional, qty_str = engine.calculate_position_size(
        balance=50.0,
        current_price=68432.10,
        qty_step=0.001,
        min_qty=0.001,
    )
    assert qty == 0.0
    assert qty_str == "0.000"

    # Suppose price is $20,000.00
    # Raw Qty = 30.00 / 20000.00 = 0.0015
    # Step = 0.001 -> rounded down = 0.001
    qty2, notional2, qty_str2 = engine.calculate_position_size(
        balance=50.0,
        current_price=20000.00,
        qty_step=0.001,
        min_qty=0.001,
    )
    assert qty2 == 0.001
    assert qty_str2 == "0.001"
    assert notional2 == pytest.approx(20.00)

    # Test fine-grained step: step = 0.0001
    # Raw Qty = 30.00 / 68432.10 = 0.00043839...
    # Step = 0.0001 -> rounded down = 0.0004
    qty3, notional3, qty_str3 = engine.calculate_position_size(
        balance=50.0,
        current_price=68432.10,
        qty_step=0.0001,
        min_qty=0.0001,
    )
    assert qty3 == 0.0004
    assert qty_str3 == "0.0004"
    assert notional3 == pytest.approx(0.0004 * 68432.10)


def test_round_down_step_utility():
    assert round_down_step(0.0019, 0.001) == 0.001
    assert round_down_step(0.0020, 0.001) == 0.002
    assert round_down_step(15.789, 0.1) == 15.7
    assert round_down_step(15.789, 0.05) == 15.75


# ---------------------------------------------------------------------------
# 3. Minimum Order Size ($1.00 Notional) Check Tests
# ---------------------------------------------------------------------------
def test_skip_trade_if_notional_below_minimum():
    """Requirement 7b: The executor skips trades if notional < $1.00."""
    async def _run():
        client = AsyncMock(spec=BybitClient)
        cfg = ExecutionConfig(
            risk_pct=0.015,
            sl_pct=0.025,
            min_notional=1.00,
            dry_run=True,
        )
        engine = ExecutionEngine(client=client, config=cfg)

        # Mock flat position
        engine.is_position_active = AsyncMock(return_value=(False, 0, 0.0))
        # Mock instrument specs with BTC qty_step=0.001
        engine.get_instrument_specs = AsyncMock(return_value={
            "max_leverage": 100.0,
            "min_leverage": 1.0,
            "qty_step": 0.001,
            "min_qty": 0.001,
            "max_qty": 100.0,
            "tick_size": 0.1,
            "min_price": 0.1,
        })
        # Current BTC price = $100,000
        # $50 balance -> $0.75 risk / 0.025 = $30.00 notional
        # But 30 / 100,000 = 0.0003 BTC -> rounds down to 0.000 BTC (< 0.001 min_qty)
        # Notional = 0.00 * 100,000 = $0.00 < $1.00
        engine.get_current_price = AsyncMock(return_value=100000.0)
        engine.get_account_balance = AsyncMock(return_value=50.0)

        result = await engine.execute_trade(symbol="BTCUSDT", side="Buy")

        assert result.success is False
        assert result.skipped is True
        assert "below Bybit minimum" in result.skip_reason
        # Verify that order was NOT submitted
        client.place_order.assert_not_called()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. Leverage Configuration Only When Flat Tests
# ---------------------------------------------------------------------------
def test_leverage_set_only_when_flat():
    """Requirement 7c: Leverage is set to max only when flat."""
    async def _run():
        client = AsyncMock(spec=BybitClient)
        cfg = ExecutionConfig(dry_run=False, testnet=True)
        engine = ExecutionEngine(client=client, config=cfg)

        # Case A: When position is already ACTIVE (e.g. size = 0.015)
        engine.is_position_active = AsyncMock(return_value=(True, 0, 0.015))
        res_active = await engine.execute_trade(symbol="BTCUSDT", side="Buy")

        assert res_active.success is False
        assert res_active.skipped is True
        assert "Position already open" in res_active.skip_reason
        # Ensure neither switch_isolated nor set_leverage were called
        client.switch_isolated.assert_not_called()
        client.set_leverage.assert_not_called()

        # Case B: When FLAT (no active position)
        engine.is_position_active = AsyncMock(return_value=(False, 0, 0.0))
        engine.get_instrument_specs = AsyncMock(return_value={
            "max_leverage": 100.0,
            "min_leverage": 1.0,
            "qty_step": 0.0001,
            "min_qty": 0.0001,
            "max_qty": 100.0,
            "tick_size": 0.1,
            "min_price": 0.1,
        })
        engine.get_current_price = AsyncMock(return_value=60000.0)
        engine.get_account_balance = AsyncMock(return_value=100.0)

        # Mock order place and trading stop responses
        client.switch_isolated.return_value = {"retCode": 0, "retMsg": "OK"}
        client.set_leverage.return_value = {"retCode": 0, "retMsg": "OK"}
        client.place_order.return_value = {"retCode": 0, "result": {"orderId": "test-order-999"}}
        client.set_trading_stop.return_value = {"retCode": 0, "retMsg": "OK"}

        res_flat = await engine.execute_trade(symbol="BTCUSDT", side="Buy")

        assert res_flat.success is True
        assert res_flat.skipped is False
        # Verify switch_isolated and set_leverage WERE called with max leverage "100"
        client.switch_isolated.assert_called_once_with(
            category="linear",
            symbol="BTCUSDT",
            trade_mode=0,
            buy_leverage="100",
            sell_leverage="100",
        )
        client.set_leverage.assert_called_once_with(
            category="linear",
            symbol="BTCUSDT",
            buy_leverage="100",
            sell_leverage="100",
        )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. Stop Loss and Take Profit Price Calculation Tests
# ---------------------------------------------------------------------------
def test_stop_and_tp_price_calculations():
    """Requirement 7d: Stop and TP prices are calculated correctly."""
    client = MagicMock(spec=BybitClient)
    cfg = ExecutionConfig(sl_pct=0.025, tp_pct=0.060)
    engine = ExecutionEngine(client=client, config=cfg)

    # LONG Position from $100,000.00
    # SL = 100,000 * (1 - 0.025) = $97,500.00
    # TP = 100,000 * (1 + 0.060) = $106,000.00
    sl_long, tp_long, sl_long_str, tp_long_str = engine.calculate_tp_sl(
        entry_price=100000.0,
        side="Buy",
        tick_size=0.10,
    )
    assert sl_long == pytest.approx(97500.0)
    assert tp_long == pytest.approx(106000.0)
    assert sl_long_str == "97500.0"
    assert tp_long_str == "106000.0"

    # SHORT Position from $100,000.00
    # SL = 100,000 * (1 + 0.025) = $102,500.00
    # TP = 100,000 * (1 - 0.060) = $94,000.00
    sl_short, tp_short, sl_short_str, tp_short_str = engine.calculate_tp_sl(
        entry_price=100000.0,
        side="Sell",
        tick_size=0.10,
    )
    assert sl_short == pytest.approx(102500.0)
    assert tp_short == pytest.approx(94000.0)
    assert sl_short_str == "102500.0"
    assert tp_short_str == "94000.0"


# ---------------------------------------------------------------------------
# 6. Safety Guard Environment Lock Test
# ---------------------------------------------------------------------------
def test_safety_guard_env_check():
    """Requirement 6: Abort if TADEX_TARGET_ENV != 'testnet' unless --dry-run."""
    with patch.dict("os.environ", {"TADEX_TARGET_ENV": "mainnet", "BYBIT_TESTNET": "false"}):
        with pytest.raises(PermissionError) as exc_info:
            ExecutionEngine.from_env(dry_run=False)
        assert "SAFETY GUARD ACTIVATED" in str(exc_info.value)

    # Should succeed without raising when dry_run=True even on mainnet env
    with patch.dict("os.environ", {"TADEX_TARGET_ENV": "mainnet"}):
        engine_dry = ExecutionEngine.from_env(dry_run=True)
        assert engine_dry.config.dry_run is True
