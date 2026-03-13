"""
Mudrex Futures API client for XAUT EMA Pullback strategy.

API docs: https://docs.trade.mudrex.com/docs/overview
Rate limit: 2 requests/second
"""

import logging
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

MUDREX_BASE = "https://trade.mudrex.com/fapi/v1"


class RateLimiter:
    """Enforce 2 req/sec for Mudrex API."""

    def __init__(self, min_interval: float = 0.55):
        self.min_interval = min_interval
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


class MudrexClient:
    """
    Mudrex Futures API client.
    Uses symbol-first trading (XAUTUSDT) via is_symbol query param.
    """

    def __init__(self, api_secret: str):
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-Authentication": api_secret,
                "Content-Type": "application/json",
            }
        )
        self.rate_limiter = RateLimiter(min_interval=0.55)

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> dict:
        self.rate_limiter.wait()
        url = f"{MUDREX_BASE}{path}"
        try:
            resp = self.session.request(
                method, url, params=params, json=json, timeout=15
            )
            data = resp.json() if resp.content else {}
            if not resp.ok:
                logger.error("Mudrex API error: %s %s", resp.status_code, data)
                raise MudrexAPIError(resp.status_code, data)
            return data
        except requests.RequestException as e:
            logger.exception("Mudrex request failed: %s", e)
            raise

    def get_futures_balance(self) -> float:
        """
        Get USDT balance in futures wallet.
        Tries common Mudrex endpoint patterns.
        """
        for path in ["/wallet/futures", "/futures/balance", "/futures/funds"]:
            try:
                data = self._request("GET", path)
                if data.get("success"):
                    balance_data = data.get("data", {})
                    if isinstance(balance_data, dict):
                        val = balance_data.get("available_balance") or balance_data.get("balance") or balance_data.get("availableBalance") or 0
                        return float(val)
                    if isinstance(balance_data, (int, float)):
                        return float(balance_data)
            except MudrexAPIError:
                continue
        logger.warning("Could not fetch futures balance; use config.initial_equity")
        return 0.0

    def get_leverage(self, symbol: str) -> dict:
        """Get current leverage for symbol."""
        return self._request(
            "GET",
            f"/futures/{symbol}/leverage",
            params={"is_symbol": ""},
        )

    def set_leverage(
        self,
        symbol: str,
        leverage: int,
        margin_type: str = "ISOLATED",
    ) -> dict:
        """Set leverage for symbol."""
        return self._request(
            "POST",
            f"/futures/{symbol}/leverage",
            params={"is_symbol": ""},
            json={"leverage": leverage, "margin_type": margin_type},
        )

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
        """
        payload = {
            "leverage": leverage,
            "quantity": round(quantity, 4),
            "order_price": round(order_price, 2),
            "order_type": order_type,
            "trigger_type": "MARKET",
            "is_takeprofit": take_profit is not None,
            "is_stoploss": stop_loss is not None,
            "reduce_only": reduce_only,
        }
        if stop_loss is not None:
            payload["stoploss_price"] = round(stop_loss, 2)
        if take_profit is not None:
            payload["takeprofit_price"] = round(take_profit, 2)

        data = self._request(
            "POST",
            f"/futures/{symbol}/order",
            params={"is_symbol": ""},
            json=payload,
        )
        if data.get("success") and data.get("data"):
            logger.info(
                "Order placed: %s %s qty=%s order_id=%s",
                order_type,
                symbol,
                quantity,
                data["data"].get("order_id"),
            )
        return data

    def get_open_positions(self, symbol: Optional[str] = None) -> list:
        """Get open positions, optionally filtered by symbol."""
        for path in ["/positions", "/futures/positions", "/positions/open"]:
            try:
                params = {"symbol": symbol} if symbol else None
                data = self._request("GET", path, params=params)
                if data.get("success"):
                    return data.get("data", []) or []
            except MudrexAPIError:
                continue
        return []


class MudrexAPIError(Exception):
    def __init__(self, status_code: int, response: dict):
        self.status_code = status_code
        self.response = response
        errors = response.get("errors", [])
        msg = errors[0].get("text", str(response)) if errors else str(response)
        super().__init__(f"Mudrex API error ({status_code}): {msg}")
