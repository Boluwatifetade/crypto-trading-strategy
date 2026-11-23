"""
Dedicated Execution Engine for BTC 1-Hour Trend Pullback Strategy.

Implements the required Bybit V5 margin and risk execution model:
1. Cross Margin Mode + Highest Available Leverage.
2. Flat Check: Only configure leverage and open new orders when flat.
3. Risk-Based Sizing:
   - Risk Amount = 1.5% of Account Balance ($0.75 on $50 account).
   - Position Notional = Risk Amount / 2.5% Stop Loss = $30.00.
   - Quantity = math.floor(Notional / Price / qtyStep) * qtyStep.
   - Minimum Notional Check: Skip if < $1.00 Bybit minimum.
4. Hard Stops: Immediately set Stop Loss at 2.5% and Take Profit at 6.0%.
5. Safety Guards:
   - Enforce TADEX_TARGET_ENV == "testnet" (unless --dry-run).
   - --dry-run mode for paper testing without sending live orders.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from .bybit_client import BybitAPIError, BybitClient

logger = logging.getLogger("strategy_executor.engine")


@dataclass
class ExecutionConfig:
    symbol: str = "BTCUSDT"
    category: str = "linear"
    risk_pct: float = 0.015         # 1.5% risk per trade
    sl_pct: float = 0.025           # 2.5% hard stop loss
    tp_pct: float = 0.060           # 6.0% take profit
    default_balance: float = 50.0   # $50 test account
    dry_run: bool = False
    testnet: bool = True
    min_notional: float = 1.0       # Bybit minimum notional USD
    allow_min_order: bool = False   # If True, allows minimum order qty if notional >= $1.00


@dataclass
class OrderResult:
    success: bool
    symbol: str
    side: str
    entry_price: float
    qty: float
    notional: float
    stop_loss: float
    take_profit: float
    leverage: str
    market_order_id: Optional[str] = None
    trading_stop_set: bool = False
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    dry_run: bool = False


def round_down_step(val: float, step: float) -> float:
    """Round value DOWN to the nearest exchange lot size step."""
    if step <= 0:
        return val
    # Use Decimal to avoid floating point precision artifacts
    d_val = Decimal(str(val))
    d_step = Decimal(str(step))
    multiplier = (d_val / d_step).quantize(Decimal("1"), rounding=ROUND_DOWN)
    result = float(multiplier * d_step)
    return result


def round_to_tick(price: float, tick_size: float) -> float:
    """Round price to the nearest tick size."""
    if tick_size <= 0:
        return price
    d_price = Decimal(str(price))
    d_tick = Decimal(str(tick_size))
    # Round to nearest tick
    rounded = (d_price / d_tick).quantize(Decimal("1")) * d_tick
    return float(rounded)


def get_decimals(step: float) -> int:
    """Calculate number of decimal places in a float step."""
    s = f"{step:.10f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 0


class ExecutionEngine:
    """
    Dedicated Bybit V5 Trade Execution Engine.
    """

    def __init__(self, client: BybitClient, config: Optional[ExecutionConfig] = None):
        self.client = client
        self.config = config or ExecutionConfig()

    @classmethod
    def from_env(
        cls,
        dry_run: bool = False,
        env_path: Optional[Path] = None,
        config: Optional[ExecutionConfig] = None,
    ) -> ExecutionEngine:
        """
        Instantiate execution engine from environment variables.
        Reads TADEX_TARGET_ENV and ensures testnet safety.
        """
        if env_path and env_path.exists():
            load_dotenv(env_path)
        else:
            # Check strategy .env or Tadex .env
            strat_env = Path(__file__).resolve().parent.parent / ".env"
            tadex_env = Path(r"C:\Users\user\Documents\Tadex\.env")
            if strat_env.exists():
                load_dotenv(strat_env)
            elif tadex_env.exists():
                load_dotenv(tadex_env)

        target_env = os.getenv("TADEX_TARGET_ENV", "").strip().lower()
        # Fall back to BYBIT_TESTNET if TADEX_TARGET_ENV is unset
        if not target_env and os.getenv("BYBIT_TESTNET", "").lower() == "true":
            target_env = "testnet"

        # Safety Guard: refuse live execution unless target_env == "testnet"
        if not dry_run and target_env != "testnet":
            raise PermissionError(
                f"SAFETY GUARD ACTIVATED: TADEX_TARGET_ENV is '{target_env}', but must be 'testnet'. "
                f"Real money execution is locked. Use --dry-run or set TADEX_TARGET_ENV=testnet."
            )

        api_key = os.getenv("BYBIT_API_KEY", "")
        api_secret = os.getenv("BYBIT_API_SECRET", "")

        cfg = config or ExecutionConfig()
        cfg.dry_run = dry_run
        cfg.testnet = True  # Always testnet for this runner

        client = BybitClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=cfg.testnet,
        )
        return cls(client=client, config=cfg)

    async def get_instrument_specs(self, symbol: str) -> Dict[str, Any]:
        """Fetch lot size, tick size, and leverage filter from Bybit."""
        res = await self.client.get_instruments_info(category=self.config.category, symbol=symbol)
        item_list = res.get("result", {}).get("list", [])
        if not item_list:
            raise ValueError(f"No instrument info returned from Bybit for {symbol}")

        info = item_list[0]
        lev_filter = info.get("leverageFilter", {})
        lot_filter = info.get("lotSizeFilter", {})
        price_filter = info.get("priceFilter", {})

        return {
            "max_leverage": float(lev_filter.get("maxLeverage", "100.0")),
            "min_leverage": float(lev_filter.get("minLeverage", "1.0")),
            "qty_step": float(lot_filter.get("qtyStep", "0.001")),
            "min_qty": float(lot_filter.get("minOrderQty", "0.001")),
            "max_qty": float(lot_filter.get("maxOrderQty", "1000.0")),
            "tick_size": float(price_filter.get("tickSize", "0.1")),
            "min_price": float(price_filter.get("minPrice", "0.1")),
        }

    async def get_current_price(self, symbol: str) -> float:
        """Fetch latest market price from tickers."""
        res = await self.client.get_ticker(category=self.config.category, symbol=symbol)
        item_list = res.get("result", {}).get("list", [])
        if not item_list:
            raise ValueError(f"No ticker data available for {symbol}")
        last_price = float(item_list[0].get("lastPrice", "0.0"))
        if last_price <= 0:
            raise ValueError(f"Invalid last price for {symbol}: {last_price}")
        return last_price

    async def is_position_active(self, symbol: str) -> Tuple[bool, int, float]:
        """
        Check if a position is currently open for symbol.
        Returns: (is_active, position_idx, current_size)
        """
        if self.config.dry_run:
            if not self.client.api_key:
                return False, 0, 0.0
            try:
                positions = await self.client.get_positions(category=self.config.category, symbol=symbol)
            except Exception as e:
                logger.info("[DRY-RUN] Live position query (%s). Simulating FLAT position.", e)
                return False, 0, 0.0
        else:
            positions = await self.client.get_positions(category=self.config.category, symbol=symbol)

        for pos in positions:
            size = float(pos.get("size", "0.0"))
            if size > 0:
                pos_idx = int(pos.get("positionIdx", 0))
                return True, pos_idx, size
        return False, 0, 0.0

    async def get_account_balance(self) -> float:
        """Fetch Unified account equity or return default balance."""
        if self.config.dry_run and not self.client.api_key:
            return self.config.default_balance

        try:
            res = await self.client.get_wallet_balance(account_type="UNIFIED")
            item_list = res.get("result", {}).get("list", [])
            if item_list:
                total_equity = float(item_list[0].get("totalEquity", "0.0"))
                if total_equity > 0:
                    return total_equity
        except Exception as e:
            if self.config.dry_run:
                logger.info("[DRY-RUN] Balance query (%s). Using test balance $%.2f", e, self.config.default_balance)
            else:
                logger.warning("Could not fetch live balance (%s). Falling back to default $%.2f", e, self.config.default_balance)

        return self.config.default_balance

    def calculate_position_size(
        self,
        balance: float,
        current_price: float,
        qty_step: float,
        min_qty: float,
    ) -> Tuple[float, float, str]:
        """
        Calculate risk-based position size and format to step size.
        Returns: (rounded_qty, notional_usd, qty_string)
        """
        risk_usd = balance * self.config.risk_pct
        notional_usd = risk_usd / self.config.sl_pct
        raw_qty = notional_usd / current_price

        # Round down to exchange lot size step
        rounded_qty = round_down_step(raw_qty, qty_step)
        if rounded_qty < min_qty and self.config.allow_min_order and raw_qty > 0:
            rounded_qty = min_qty
            logger.info("Allowing minimum exchange order size: %s", min_qty)

        actual_notional = rounded_qty * current_price

        # Format with proper decimal precision
        decimals = get_decimals(qty_step)
        qty_str = f"{rounded_qty:.{decimals}f}"

        return rounded_qty, actual_notional, qty_str

    def calculate_tp_sl(
        self,
        entry_price: float,
        side: str,
        tick_size: float,
    ) -> Tuple[float, float, str, str]:
        """
        Calculate hard Stop Loss (2.5%) and Take Profit (6.0%).
        Returns: (sl_price, tp_price, sl_str, tp_str)
        """
        is_buy = side.strip().lower() in ("buy", "long")
        if is_buy:
            raw_sl = entry_price * (1.0 - self.config.sl_pct)
            raw_tp = entry_price * (1.0 + self.config.tp_pct)
        else:
            raw_sl = entry_price * (1.0 + self.config.sl_pct)
            raw_tp = entry_price * (1.0 - self.config.tp_pct)

        sl_price = round_to_tick(raw_sl, tick_size)
        tp_price = round_to_tick(raw_tp, tick_size)

        decimals = get_decimals(tick_size)
        sl_str = f"{sl_price:.{decimals}f}"
        tp_str = f"{tp_price:.{decimals}f}"

        return sl_price, tp_price, sl_str, tp_str

    async def prepare_margin_and_leverage(self, symbol: str, max_leverage: float) -> str:
        """
        Configure CROSS margin mode and set highest available leverage when flat.
        """
        lev_str = str(int(max_leverage))
        if self.config.dry_run:
            logger.info("[DRY-RUN] Setting Margin: CROSS mode (tradeMode=0), Leverage: %sx", lev_str)
            return lev_str

        # 1. Set CROSS Margin Mode (tradeMode = 0)
        logger.info("Setting %s margin mode to CROSS with %sx leverage", symbol, lev_str)
        try:
            await self.client.switch_isolated(
                category=self.config.category,
                symbol=symbol,
                trade_mode=0,
                buy_leverage=lev_str,
                sell_leverage=lev_str,
            )
        except BybitAPIError as e:
            # 110026: Cross/isolated margin mode not modified
            if e.ret_code != 110026:
                logger.error("Failed to set CROSS margin mode: %s", e)
                raise

        # 2. Set Leverage to Maximum Available
        try:
            await self.client.set_leverage(
                category=self.config.category,
                symbol=symbol,
                buy_leverage=lev_str,
                sell_leverage=lev_str,
            )
        except BybitAPIError as e:
            # 110043: Leverage not modified
            if e.ret_code != 110043:
                logger.error("Failed to set leverage to %sx: %s", lev_str, e)
                raise

        return lev_str

    async def execute_trade(
        self,
        symbol: Optional[str] = None,
        side: str = "Buy",
        reference_entry: Optional[float] = None,
    ) -> OrderResult:
        """
        Execute full trade pipeline:
        1. Check active position -> abort if already in position (do NOT change leverage).
        2. Query instrument specs for max leverage, tick size, and lot step.
        3. Set CROSS margin and max leverage.
        4. Calculate risk-based sizing and verify >= $1.00 notional.
        5. Place Market Order.
        6. Immediately set 2.5% Stop Loss and 6.0% Take Profit via trading-stop.
        """
        sym = (symbol or self.config.symbol).strip().upper()
        norm_side = "Buy" if side.strip().lower() in ("buy", "long") else "Sell"

        logger.info("=" * 60)
        logger.info("START TRADE EXECUTION: %s %s (Dry-Run: %s)", norm_side, sym, self.config.dry_run)
        logger.info("=" * 60)

        # Step 1: Check existing position
        is_active, pos_idx, active_size = await self.is_position_active(sym)
        if is_active:
            skip_msg = f"Position already open for {sym} (size: {active_size}). Skipping entry to prevent double-exposure and leverage modification."
            logger.warning(skip_msg)
            return OrderResult(
                success=False,
                symbol=sym,
                side=norm_side,
                entry_price=0.0,
                qty=0.0,
                notional=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                leverage="",
                skipped=True,
                skip_reason=skip_msg,
                dry_run=self.config.dry_run,
            )

        # Step 2: Fetch instrument specs and market price
        specs = await self.get_instrument_specs(sym)
        current_price = reference_entry if (reference_entry and reference_entry > 0) else await self.get_current_price(sym)
        logger.info("%s Specs: Max Lev=%sx, QtyStep=%s, TickSize=%s, CurrentPrice=$%.2f",
                    sym, specs["max_leverage"], specs["qty_step"], specs["tick_size"], current_price)

        # Step 3: Configure Margin & Max Leverage (Only when flat!)
        active_lev = await self.prepare_margin_and_leverage(sym, specs["max_leverage"])

        # Step 4: Risk-Based Position Sizing
        balance = await self.get_account_balance()
        qty, notional, qty_str = self.calculate_position_size(
            balance=balance,
            current_price=current_price,
            qty_step=specs["qty_step"],
            min_qty=specs["min_qty"],
        )

        logger.info("Risk Sizing: Balance=$%.2f, Risk 1.5%%=$%.2f, SL 2.5%% -> Desired Notional=$%.2f",
                    balance, balance * self.config.risk_pct, (balance * self.config.risk_pct) / self.config.sl_pct)
        logger.info("Calculated Lot: %s %s (Actual Notional: $%.2f)", qty_str, sym, notional)

        # Step 4b: Minimum Notional Check ($1.00 Bybit Minimum)
        if notional < self.config.min_notional or qty < specs["min_qty"]:
            skip_msg = (
                f"Trade SKIPPED: Calculated notional ${notional:.2f} is below Bybit minimum "
                f"${self.config.min_notional:.2f} (qty: {qty_str} < min {specs['min_qty']}). "
                f"Aborting trade to prevent risk rule violations."
            )
            logger.error(skip_msg)
            return OrderResult(
                success=False,
                symbol=sym,
                side=norm_side,
                entry_price=current_price,
                qty=qty,
                notional=notional,
                stop_loss=0.0,
                take_profit=0.0,
                leverage=active_lev,
                skipped=True,
                skip_reason=skip_msg,
                dry_run=self.config.dry_run,
            )

        # Step 5: Calculate Hard Stops (SL: 2.5%, TP: 6.0%)
        sl_price, tp_price, sl_str, tp_str = self.calculate_tp_sl(
            entry_price=current_price,
            side=norm_side,
            tick_size=specs["tick_size"],
        )
        logger.info("Hard Exits: Entry=$%.2f -> SL (2.5%%)=$%s, TP (6.0%%)=$%s", current_price, sl_str, tp_str)

        # Step 6: Execute Market Order
        market_order_id = None
        if self.config.dry_run:
            logger.info("[DRY-RUN] Market Order Payload: category=%s, symbol=%s, side=%s, qty=%s, positionIdx=%s",
                        self.config.category, sym, norm_side, qty_str, pos_idx)
            market_order_id = "dry-run-order-id-12345"
            filled_entry = current_price
        else:
            logger.info("Submitting live Market Order: %s %s %s...", norm_side, qty_str, sym)
            order_res = await self.client.place_order(
                category=self.config.category,
                symbol=sym,
                side=norm_side,
                order_type="Market",
                qty=qty_str,
                position_idx=pos_idx,
            )
            market_order_id = order_res.get("result", {}).get("orderId", "")
            logger.info("Market Order FILLED! Order ID: %s", market_order_id)
            # Re-read price from ticker to confirm fill level
            filled_entry = await self.get_current_price(sym)
            # Recalculate exact SL and TP from filled entry
            sl_price, tp_price, sl_str, tp_str = self.calculate_tp_sl(
                entry_price=filled_entry,
                side=norm_side,
                tick_size=specs["tick_size"],
            )

        # Step 7: Immediately set Stop Loss and Take Profit
        trading_stop_ok = False
        if self.config.dry_run:
            logger.info("[DRY-RUN] Trading Stop Payload: category=%s, symbol=%s, SL=%s, TP=%s, positionIdx=%s, tpslMode=Full",
                        self.config.category, sym, sl_str, tp_str, pos_idx)
            trading_stop_ok = True
        else:
            logger.info("Setting Trading Stop: SL=%s, TP=%s on %s...", sl_str, tp_str, sym)
            stop_res = await self.client.set_trading_stop(
                category=self.config.category,
                symbol=sym,
                stop_loss=sl_str,
                take_profit=tp_str,
                position_idx=pos_idx,
                tp_trigger_by="MarkPrice",
                sl_trigger_by="MarkPrice",
                tpsl_mode="Full",
            )
            logger.info("Trading Stop ACTIVE: %s", stop_res.get("retMsg", "OK"))
            trading_stop_ok = True

        logger.info("=" * 60)
        logger.info("EXECUTION COMPLETE: %s %s @ $%.2f | SL: $%s | TP: $%s", norm_side, sym, filled_entry, sl_str, tp_str)
        logger.info("=" * 60)

        return OrderResult(
            success=True,
            symbol=sym,
            side=norm_side,
            entry_price=filled_entry,
            qty=qty,
            notional=notional,
            stop_loss=sl_price,
            take_profit=tp_price,
            leverage=active_lev,
            market_order_id=market_order_id,
            trading_stop_set=trading_stop_ok,
            dry_run=self.config.dry_run,
        )


async def main():
    parser = argparse.ArgumentParser(description="Standalone Strategy Executor for Bybit V5")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (e.g. BTCUSDT)")
    parser.add_argument("--side", choices=["Buy", "Sell", "buy", "sell"], default="Buy", help="Order side")
    parser.add_argument("--dry-run", action="store_true", help="Log API payloads without executing live orders")
    parser.add_argument("--risk", type=float, default=0.015, help="Risk fraction (default: 0.015 / 1.5%%)")
    parser.add_argument("--sl", type=float, default=0.025, help="Stop loss fraction (default: 0.025 / 2.5%%)")
    parser.add_argument("--tp", type=float, default=0.060, help="Take profit fraction (default: 0.060 / 6.0%%)")
    parser.add_argument("--balance", type=float, default=50.0, help="Account balance USD for testing (default: $50)")
    parser.add_argument("--allow-min-order", action="store_true", help="Allow bumping to exchange min_qty (e.g. 0.001 BTC) for small accounts")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = ExecutionConfig(
        symbol=args.symbol,
        risk_pct=args.risk,
        sl_pct=args.sl,
        tp_pct=args.tp,
        default_balance=args.balance,
        dry_run=args.dry_run,
        testnet=True,
        allow_min_order=args.allow_min_order,
    )

    engine = ExecutionEngine.from_env(dry_run=args.dry_run, config=cfg)
    async with engine.client:
        res = await engine.execute_trade(symbol=args.symbol, side=args.side)
        print("\nExecution Summary:")
        print(json.dumps(asdict(res), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
