"""
Mudrex Futures API client for XAUT EMA Pullback strategy.
Uses mudrex-trading-sdk (https://github.com/DecentralizedJM/mudrex-api-trading-python-sdk).
API docs: https://docs.trade.mudrex.com/docs/overview
Rate limit: 2 req/s. Bot spaces calls (e.g. 0.6s between balance and positions).
Retries on transient errors (502, 503, 504).

Resolves XAUTUSDT via paginated list_all() and uses asset_id for API calls when needed.
"""

import logging
import time
from typing import Any, Optional

# Transient server errors: retry after a short wait
RETRY_STATUSES = (502, 503, 504)
RETRY_WAIT = 5
RETRY_ATTEMPTS = 2

from mudrex import MudrexClient as _MudrexClient
from mudrex.exceptions import MudrexAPIError as _MudrexAPIError
from mudrex.models import OrderRequest, OrderType, TriggerType

logger = logging.getLogger(__name__)


class MudrexAPIError(Exception):
    """Raised on Mudrex API errors. Compatible with bot's exception handling."""

    def __init__(self, status_code: int, response: dict):
        self.status_code = status_code
        self.response = response
        errors = response.get("errors", [])
        msg = errors[0].get("text", str(response)) if errors else str(response)
        super().__init__(f"Mudrex API error ({status_code}): {msg}")


def _wrap_sdk_error(e: Exception) -> MudrexAPIError:
    """Convert SDK exception to MudrexAPIError."""
    if isinstance(e, _MudrexAPIError):
        status = getattr(e, "status_code", 0) or 0
        resp = getattr(e, "response", None) or {"message": str(e)}
        if not isinstance(resp, dict):
            resp = {"message": str(e)}
        return MudrexAPIError(status, resp)
    return MudrexAPIError(0, {"message": str(e)})


class MudrexClient:
    """
    Mudrex Futures API client using the official SDK.
    Resolves symbol to asset_id via paginated list_all() for XAUTUSDT.
    """

    def __init__(self, api_secret: str):
        self._client = _MudrexClient(api_secret=api_secret)
        self._asset_id_cache: dict[str, str] = {}  # symbol -> asset_id

    def _resolve_asset(self, symbol: str) -> Optional[str]:
        """
        Resolve symbol to asset_id via paginated list_all().
        XAUTUSDT may require asset_id for some endpoints.
        """
        if symbol in self._asset_id_cache:
            return self._asset_id_cache[symbol]
        try:
            assets = self._client.assets.list_all(refresh=True)
            sym_upper = symbol.upper()
            for a in assets:
                if (getattr(a, "symbol", "") or "").upper() == sym_upper:
                    aid = getattr(a, "asset_id", None) or getattr(a, "id", None)
                    if aid:
                        self._asset_id_cache[symbol] = str(aid)
                        logger.debug("Resolved %s -> asset_id=%s", symbol, aid)
                        return str(aid)
        except Exception as e:
            logger.warning("Asset resolve failed for %s: %s", symbol, e)
        return None

    def get_futures_balance(self) -> float:
        """Get USDT balance in futures wallet. Retries on 502/503/504."""
        last_err = None
        for attempt in range(RETRY_ATTEMPTS + 1):
            try:
                balance = self._client.wallet.get_futures_balance()
                val = float(getattr(balance, "balance", 0) or getattr(balance, "available", 0) or 0)
                return val
            except Exception as e:
                last_err = e
                status = getattr(e, "status_code", None)
                msg = str(e).lower()
                is_transient = status in RETRY_STATUSES or "502" in msg or "503" in msg or "504" in msg or "bad gateway" in msg
                if is_transient and attempt < RETRY_ATTEMPTS:
                    logger.warning("Mudrex transient error (attempt %d/%d), retrying in %ds", attempt + 1, RETRY_ATTEMPTS + 1, RETRY_WAIT)
                    time.sleep(RETRY_WAIT)
                    continue
                raise _wrap_sdk_error(e)
        raise _wrap_sdk_error(last_err)

    def set_leverage(
        self,
        symbol: str,
        leverage: int,
        margin_type: str = "ISOLATED",
    ) -> dict:
        """Set leverage for symbol. Uses asset_id when available (from paginated list)."""
        try:
            asset_id = self._resolve_asset(symbol)
            if asset_id:
                # Use asset_id (no is_symbol) - required for some assets like XAUTUSDT
                resp = self._client.post(
                    f"/futures/{asset_id}/leverage",
                    {"margin_type": margin_type, "leverage": str(leverage)},
                )
                return {"success": True, "data": resp.get("data", resp)}
            result = self._client.leverage.set(
                symbol=symbol,
                leverage=str(leverage),
                margin_type=margin_type,
            )
            return {"success": True, "data": result}
        except Exception as e:
            raise _wrap_sdk_error(e)

    def place_market_order(
        self,
        symbol: str,
        order_type: str,
        quantity: float,
        leverage: int,
        order_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reduce_only: bool = False,
    ) -> dict:
        """
        Place market order. order_type: LONG | SHORT
        order_price: current/reference price (required even for MARKET)
        Uses asset_id when available (from paginated list) for XAUTUSDT.
        """
        try:
            side = OrderType.LONG if str(order_type).upper() == "LONG" else OrderType.SHORT
            request = OrderRequest(
                quantity=str(round(quantity, 4)),
                order_type=side,
                trigger_type=TriggerType.MARKET,
                leverage=str(leverage),
                order_price=str(round(order_price, 2)),
                is_stoploss=stop_loss is not None,
                stoploss_price=str(round(stop_loss, 2)) if stop_loss is not None else None,
                is_takeprofit=take_profit is not None,
                takeprofit_price=str(round(take_profit, 2)) if take_profit is not None else None,
                reduce_only=reduce_only,
            )
            asset_id = self._resolve_asset(symbol)
            if asset_id:
                order = self._client.orders.create(asset_id=asset_id, request=request)
            else:
                order = self._client.orders.create(symbol=symbol, request=request)
            logger.info(
                "Order placed: %s %s qty=%s order_id=%s",
                order_type,
                symbol,
                quantity,
                getattr(order, "order_id", order),
            )
            return {"success": True, "data": order}
        except Exception as e:
            raise _wrap_sdk_error(e)

    def get_open_positions(self, symbol: Optional[str] = None) -> list:
        """Get open positions, optionally filtered by symbol."""
        try:
            positions = self._client.positions.list_open()
            # Convert to dict format expected by get_current_position
            out = []
            for p in positions:
                pos_symbol = getattr(p, "symbol", None) or getattr(p, "asset_id", "")
                if symbol and symbol.upper() not in (pos_symbol or "").upper():
                    continue
                out.append({
                    "symbol": pos_symbol,
                    "asset_id": getattr(p, "asset_id", pos_symbol),
                    "side": getattr(getattr(p, "side", None), "value", "") or getattr(p, "order_type", ""),
                    "order_type": getattr(getattr(p, "side", None), "value", "") or getattr(p, "order_type", ""),
                })
            return out
        except Exception as e:
            raise _wrap_sdk_error(e)
