"""
Bridge between the strategy bot and Tadex.

- format_signal():      builds signal text in the canonical shape Tadex's
                        telegram_listener preprocessor produces (Cosmas style).
- verify_with_tadex():  round-trips the text through Tadex's REAL parser
                        (bybit_client/signal_parser.py) so we never insert a
                        signal the dispatcher would mis-read.
- TadexSignalSink:      inserts into public.signals (status='raw') via the
                        Supabase REST API. The Tadex dispatcher picks up the
                        INSERT exactly as it does for Telegram signals.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Optional
from uuid import UUID

import requests

logger = logging.getLogger("tadex_bridge")

SOURCE_TAG = "tadex_strategy_bot"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_signal(symbol_id: str, side: int, tp: str, sl: str, leverage: int) -> str:
    """
    symbol_id: exchange id, e.g. 'DOGEUSDT'
    side:      1 long / -1 short
    tp, sl:    prices already rounded to the exchange tick size (strings)
    """
    direction = "Long" if side == 1 else "Short"
    return (
        f"{symbol_id}\n"
        f"{direction} position\n"
        f"Entry: MARKET\n"
        f"TP: {tp}\n"
        f"SL: {sl}\n"
        f"Leverage {int(leverage)}x"
    )


# ---------------------------------------------------------------------------
# Verification against Tadex's own parser
# ---------------------------------------------------------------------------
_parser = None


def _load_tadex_parser(tadex_repo: Path):
    global _parser
    if _parser is not None:
        return _parser
    path = tadex_repo / "bybit-client" / "bybit_client" / "signal_parser.py"
    if not path.exists():
        raise FileNotFoundError(f"Tadex signal parser not found at {path}")
    # Load the module file directly so we don't import the whole bybit_client package
    spec = importlib.util.spec_from_file_location("tadex_signal_parser", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logging.getLogger(mod.__name__).setLevel(logging.WARNING)
    _parser = mod.UltimateSignalParser(logger=logging.getLogger("tadex_signal_parser"))
    return _parser


def verify_with_tadex(
    text: str, symbol_id: str, side: int, tp: str, sl: str, leverage: int, tadex_repo: Optional[Path]
) -> list[str]:
    """Return a list of problems (empty list == Tadex will read it exactly as intended)."""
    if tadex_repo is None:
        return ["TADEX_REPO_PATH not set - cannot verify against the Tadex parser"]
    parser = _load_tadex_parser(tadex_repo)
    parsed = parser.parse_signal(text)
    if parsed is None:
        return ["Tadex parser returned None"]

    problems = []
    if parsed.symbol != symbol_id:
        problems.append(f"symbol {parsed.symbol} != {symbol_id}")
    if parsed.side.value != ("Buy" if side == 1 else "Sell"):
        problems.append(f"side {parsed.side.value} wrong")
    if parsed.order_type.value != "Market":
        problems.append(f"order type {parsed.order_type.value} != Market")
    if parsed.stop_loss != Decimal(sl):
        problems.append(f"SL {parsed.stop_loss} != {sl}")
    if [Decimal(str(t)) for t in parsed.take_profits] != [Decimal(tp)]:
        problems.append(f"TPs {parsed.take_profits} != [{tp}]")
    if parsed.leverage != int(leverage):
        problems.append(f"leverage {parsed.leverage} != {leverage}")
    if parsed.confidence < 0.5:  # dispatcher discards below 0.5
        problems.append(f"confidence {parsed.confidence} < 0.5")
    ok, errors = parser.validate_signal(parsed)
    if not ok:
        problems.extend(errors)
    return problems


# ---------------------------------------------------------------------------
# Supabase sink
# ---------------------------------------------------------------------------
class TadexSignalSink:
    def __init__(self, supabase_url: str, service_key: str, provider_id: str, timeout: float = 15.0):
        if not supabase_url or not service_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required to insert signals")
        UUID(provider_id)  # raises if malformed
        self.base = supabase_url.rstrip("/") + "/rest/v1"
        self.provider_id = provider_id
        self.timeout = timeout
        self.headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
        }

    @classmethod
    def from_env(cls) -> "TadexSignalSink":
        return cls(
            os.getenv("SUPABASE_URL", ""),
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
            os.getenv("TADEX_STRATEGY_PROVIDER_ID", ""),
        )

    def check_provider(self) -> dict:
        """Confirm the provider row exists (and is not the Cosmas provider by accident)."""
        r = requests.get(
            f"{self.base}/providers",
            params={"id": f"eq.{self.provider_id}", "select": "id,name,status"},
            headers=self.headers, timeout=self.timeout,
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            raise RuntimeError(f"Provider {self.provider_id} does not exist in public.providers")
        return rows[0]

    def insert(self, raw_text: str, metadata: dict) -> Optional[str]:
        payload = {
            "provider_id": self.provider_id,
            "raw_text": raw_text,
            "status": "raw",
            "source_type": "api",          # allowed by signals_source_type_check
            "telegram_group_id": None,
            "telegram_message_id": None,
            "metadata": {"source": SOURCE_TAG, "ingested_by": SOURCE_TAG, **metadata},
        }
        r = requests.post(
            f"{self.base}/signals",
            json=payload,
            headers={**self.headers, "Prefer": "return=representation"},
            timeout=self.timeout,
        )
        if r.status_code >= 300:
            # Never echo headers (they contain the service key)
            raise RuntimeError(f"Supabase insert failed: HTTP {r.status_code} {r.text[:300]}")
        rows = r.json() or []
        return rows[0].get("id") if rows else None
