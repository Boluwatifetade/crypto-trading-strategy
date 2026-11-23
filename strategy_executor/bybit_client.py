"""
Standalone Minimal Bybit V5 API Client.

Handles:
- Cryptographic HMAC-SHA256 authentication with strict canonicalization.
- Automatic server time synchronization to prevent timestamp skew errors (retCode: 10002).
- Asynchronous execution over aiohttp with connection pooling.
- Cross margin, leverage, order placement, and trading stop endpoints.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urljoin

import aiohttp

logger = logging.getLogger("strategy_executor.bybit_client")


class BybitAPIError(Exception):
    """Raised when Bybit API returns an error."""

    def __init__(self, ret_code: int, ret_msg: str, http_status: int = 200, raw_data: Optional[Dict] = None):
        super().__init__(f"Bybit API Error [code={ret_code}]: {ret_msg} (HTTP {http_status})")
        self.ret_code = ret_code
        self.ret_msg = ret_msg
        self.http_status = http_status
        self.raw_data = raw_data or {}


class BybitClient:
    """
    Minimal asynchronous client for Bybit V5 REST API.
    """

    MAINNET_BASE = "https://api.bybit.com"
    TESTNET_BASE = "https://api-testnet.bybit.com"
    API_VERSION = "/v5"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        recv_window: int = 5000,
        session: Optional[aiohttp.ClientSession] = None,
        timeout_seconds: float = 10.0,
    ):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.testnet = bool(testnet)
        self.recv_window = int(recv_window)
        self.base_url = self.TESTNET_BASE if self.testnet else self.MAINNET_BASE
        self.timeout_seconds = timeout_seconds

        self._session = session
        self._owns_session = session is None
        self._time_offset_ms: int = 0
        self._is_synced: bool = False

    async def __aenter__(self) -> BybitClient:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def initialize(self) -> None:
        """Initialize HTTP session and synchronize time with Bybit."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True

        await self.sync_time()

    async def close(self) -> None:
        """Close the underlying HTTP session if owned."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    # -----------------------------------------------------------------------
    # Time Synchronization
    # -----------------------------------------------------------------------
    async def sync_time(self) -> int:
        """
        Query Bybit public server time and calculate local clock offset.
        offset_ms = server_time - local_time
        """
        url = urljoin(self.base_url, f"{self.API_VERSION}/market/time")
        t_start = time.time()
        async with self._session.get(url) as resp:
            data = await resp.json()
            t_end = time.time()

        if data.get("retCode") != 0:
            raise BybitAPIError(data.get("retCode", -1), data.get("retMsg", "Failed to fetch server time"))

        server_time_ms = int(data["result"]["timeNano"]) // 1_000_000
        # Estimate local time at server processing instant: midpoint of request
        local_time_ms = int((t_start + t_end) / 2.0 * 1000.0)
        self._time_offset_ms = server_time_ms - local_time_ms
        self._is_synced = True
        logger.debug("Synchronized with Bybit server. Clock offset: %d ms", self._time_offset_ms)
        return self._time_offset_ms

    def get_synced_timestamp(self) -> int:
        """Return current UTC timestamp in ms, corrected by server offset."""
        local_ms = int(time.time() * 1000.0)
        return local_ms + self._time_offset_ms

    # -----------------------------------------------------------------------
    # Cryptographic Authentication & Canonicalization
    # -----------------------------------------------------------------------
    def generate_signature(
        self,
        timestamp: int,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate HMAC-SHA256 signature adhering to Bybit V5 canonicalization rules:
        - GET: parameters sorted alphabetically by key, URL-encoded.
        - POST/DELETE: JSON body sorted alphabetically, compact (no whitespace).
        - Pre-signing string: timestamp + api_key + recv_window + payload
        - Digest: lowercase hex string.
        """
        if not self.api_secret:
            raise ValueError("API secret is required for authenticated requests")

        method = method.upper()
        if method == "GET":
            if params:
                clean_params = {k: v for k, v in params.items() if v is not None}
                sorted_params = sorted(clean_params.items(), key=lambda x: x[0])
                payload_str = urlencode(sorted_params)
            else:
                payload_str = ""
        elif method in ("POST", "DELETE"):
            if body:
                clean_body = {k: v for k, v in body.items() if v is not None}
                payload_str = json.dumps(clean_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            else:
                payload_str = ""
        else:
            raise ValueError(f"Unsupported HTTP method for signature: {method}")

        pre_hash = f"{timestamp}{self.api_key}{self.recv_window}{payload_str}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            pre_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest().lower()
        return signature

    # -----------------------------------------------------------------------
    # Core Request Execution
    # -----------------------------------------------------------------------
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        authenticated: bool = True,
    ) -> Dict[str, Any]:
        """Execute request with headers, signing, and automatic error handling."""
        if self._session is None or self._session.closed:
            await self.initialize()

        method = method.upper()
        url = urljoin(self.base_url, f"{self.API_VERSION}{endpoint}")

        headers = {}
        if authenticated:
            if not self.api_key or not self.api_secret:
                raise ValueError("API credentials (key and secret) are required for authenticated endpoints")
            ts = self.get_synced_timestamp()
            sig = self.generate_signature(ts, method, params=params, body=body)
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": str(ts),
                "X-BAPI-SIGN": sig,
                "X-BAPI-RECV-WINDOW": str(self.recv_window),
            }

        if method in ("POST", "DELETE") and body is not None:
            headers["Content-Type"] = "application/json"
            clean_body = {k: v for k, v in body.items() if v is not None}
            data_payload = json.dumps(clean_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        else:
            data_payload = None

        req_url = url
        if method == "GET" and params:
            clean_params = {k: v for k, v in params.items() if v is not None}
            sorted_params = sorted(clean_params.items(), key=lambda x: x[0])
            req_url = f"{url}?{urlencode(sorted_params)}"

        logger.debug("Executing %s %s", method, req_url)
        async with self._session.request(
            method=method,
            url=req_url,
            headers=headers,
            data=data_payload,
        ) as response:
            text = await response.text()
            try:
                res_json = json.loads(text)
            except Exception:
                raise BybitAPIError(
                    ret_code=-1,
                    ret_msg=f"Invalid JSON returned: {text[:200]}",
                    http_status=response.status,
                )

            ret_code = res_json.get("retCode", 0)
            ret_msg = res_json.get("retMsg", "")

            # If time drift error (10002), resync time once and retry
            if ret_code == 10002 and authenticated:
                logger.warning("Timestamp out of recv_window (10002). Re-synchronizing clock and retrying...")
                await self.sync_time()
                ts = self.get_synced_timestamp()
                sig = self.generate_signature(ts, method, params=params, body=body)
                headers["X-BAPI-TIMESTAMP"] = str(ts)
                headers["X-BAPI-SIGN"] = sig
                async with self._session.request(
                    method=method,
                    url=req_url,
                    headers=headers,
                    data=data_payload,
                ) as retry_resp:
                    retry_text = await retry_resp.text()
                    res_json = json.loads(retry_text)
                    ret_code = res_json.get("retCode", 0)
                    ret_msg = res_json.get("retMsg", "")

            if ret_code != 0:
                # RetCodes that indicate no-op / already configured state:
                # 110026: Cross/isolated margin mode not modified
                # 110043: Leverage not modified
                if ret_code in (110026, 110043):
                    logger.info("Bybit setting unchanged (retCode %d: %s). Treating as OK.", ret_code, ret_msg)
                    return res_json
                raise BybitAPIError(ret_code, ret_msg, http_status=response.status, raw_data=res_json)

            return res_json

    # -----------------------------------------------------------------------
    # Public Endpoints
    # -----------------------------------------------------------------------
    async def get_instruments_info(self, category: str = "linear", symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Query contract specifications, lot size, tick size, and leverage limits.
        Endpoint: GET /v5/market/instruments-info
        """
        params = {"category": category, "symbol": symbol}
        res = await self._request("GET", "/market/instruments-info", params=params, authenticated=False)
        return res

    async def get_ticker(self, category: str = "linear", symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Query latest market ticker for current market price.
        Endpoint: GET /v5/market/tickers
        """
        params = {"category": category, "symbol": symbol}
        res = await self._request("GET", "/market/tickers", params=params, authenticated=False)
        return res

    # -----------------------------------------------------------------------
    # Account & Position Endpoints
    # -----------------------------------------------------------------------
    async def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict[str, Any]:
        """
        Query wallet balance and equity.
        Endpoint: GET /v5/account/wallet-balance
        """
        params = {"accountType": account_type}
        return await self._request("GET", "/account/wallet-balance", params=params, authenticated=True)

    async def get_positions(self, category: str = "linear", symbol: str = "BTCUSDT") -> List[Dict[str, Any]]:
        """
        Query current positions for symbol.
        Endpoint: GET /v5/position/list
        """
        params = {"category": category, "symbol": symbol}
        res = await self._request("GET", "/position/list", params=params, authenticated=True)
        return res.get("result", {}).get("list", [])

    async def switch_isolated(
        self,
        category: str,
        symbol: str,
        trade_mode: int,
        buy_leverage: str,
        sell_leverage: str,
    ) -> Dict[str, Any]:
        """
        Switch margin mode (0 = Cross Margin, 1 = Isolated Margin).
        Endpoint: POST /v5/position/switch-isolated
        """
        body = {
            "category": category,
            "symbol": symbol,
            "tradeMode": int(trade_mode),
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage),
        }
        return await self._request("POST", "/position/switch-isolated", body=body, authenticated=True)

    async def set_leverage(
        self,
        category: str,
        symbol: str,
        buy_leverage: str,
        sell_leverage: str,
    ) -> Dict[str, Any]:
        """
        Set position leverage.
        Endpoint: POST /v5/position/set-leverage
        """
        body = {
            "category": category,
            "symbol": symbol,
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage),
        }
        return await self._request("POST", "/position/set-leverage", body=body, authenticated=True)

    # -----------------------------------------------------------------------
    # Trading Endpoints
    # -----------------------------------------------------------------------
    async def place_order(
        self,
        category: str,
        symbol: str,
        side: str,
        order_type: str,
        qty: str,
        price: Optional[str] = None,
        time_in_force: str = "GTC",
        position_idx: int = 0,
        reduce_only: bool = False,
        order_link_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Place order on Bybit V5.
        Endpoint: POST /v5/order/create
        """
        body: Dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "side": side.capitalize(),
            "orderType": order_type.capitalize(),
            "qty": str(qty),
            "timeInForce": time_in_force,
            "positionIdx": int(position_idx),
        }
        if price is not None:
            body["price"] = str(price)
        if reduce_only:
            body["reduceOnly"] = True
        if order_link_id:
            body["orderLinkId"] = str(order_link_id)
        body.update(kwargs)
        return await self._request("POST", "/order/create", body=body, authenticated=True)

    async def set_trading_stop(
        self,
        category: str,
        symbol: str,
        stop_loss: Optional[str] = None,
        take_profit: Optional[str] = None,
        position_idx: int = 0,
        tp_trigger_by: str = "MarkPrice",
        sl_trigger_by: str = "MarkPrice",
        tpsl_mode: str = "Full",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Set or amend position Stop Loss and Take Profit.
        Endpoint: POST /v5/position/trading-stop
        """
        body: Dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "positionIdx": int(position_idx),
            "tpslMode": tpsl_mode,
        }
        if stop_loss is not None:
            body["stopLoss"] = str(stop_loss)
            body["slTriggerBy"] = sl_trigger_by
        if take_profit is not None:
            body["takeProfit"] = str(take_profit)
            body["tpTriggerBy"] = tp_trigger_by
        body.update(kwargs)
        return await self._request("POST", "/position/trading-stop", body=body, authenticated=True)

    async def cancel_all_orders(self, category: str = "linear", symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Cancel all active orders for symbol.
        Endpoint: POST /v5/order/cancel-all
        """
        body = {"category": category, "symbol": symbol}
        return await self._request("POST", "/order/cancel-all", body=body, authenticated=True)
