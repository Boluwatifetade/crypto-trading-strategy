"""
Strategy Executor: Standalone Bybit V5 Execution Engine.

Isolated from Tadex core, implementing:
- Cross Margin Mode
- Highest Available Leverage
- Risk-Based Sizing (1.5% Risk, 2.5% SL, 6.0% TP)
- Immediate Hard Stop Loss and Take Profit
"""

from .bybit_client import BybitClient, BybitAPIError
from .execution_engine import ExecutionConfig, ExecutionEngine, OrderResult

__all__ = [
    "BybitClient",
    "BybitAPIError",
    "ExecutionConfig",
    "ExecutionEngine",
    "OrderResult",
]
