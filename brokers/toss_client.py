import os
import time
from typing import Any, Dict, Optional
import requests


class TossSecuritiesClient:
    """토스증권 Open API 클라이언트 (로컬 허용 IP 환경 전용)"""

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self.client_id = (client_id or os.environ.get("TOSS_CLIENT_ID", "")).strip()
        self.client_secret = (client_secret or os.environ.get("TOSS_CLIENT_SECRET", "")).strip()
        self.base_url = os.environ.get("TOSS_API_BASE_URL", "https://openapi.tossinvest.com").rstrip("/")
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _get_token(self) -> str:
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token
        resp = requests.post(
            f"{self.base_url}/v1/oauth2/token",
            json={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            headers={"Content-Type": "application/json"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        self._token_expires_at = time.time() + int(data.get("expires_in", 3600))
        return self._access_token

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }

    def get_us_quote(self, ticker: str) -> Dict[str, Any]:
        """미국주식 실시간 체결가·시가·고가·누적거래량 조회"""
        resp = requests.get(
            f"{self.base_url}/v1/overseas/stocks/{ticker.upper()}/quote",
            headers=self._headers(),
            timeout=6,
        )
        resp.raise_for_status()
        raw = resp.json()
        data = raw.get("data", raw)
        return {
            "ticker": ticker.upper(),
            "price": float(data.get("lastPrice") or data.get("price") or 0.0),
            "open": float(data.get("openPrice") or data.get("open") or 0.0),
            "high": float(data.get("highPrice") or data.get("high") or 0.0),
            "prev_close": float(data.get("prevClosePrice") or data.get("prev_close") or 0.0),
            "volume": int(float(data.get("accumulatedVolume") or data.get("volume") or 0)),
        }

    def place_us_buy_order(self, ticker: str, qty: int, limit_price: float, dry_run: bool = True) -> Dict[str, Any]:
        """실전 전환 시 지정가 매수 주문 (기본값 dry_run=True로 안전 보호)"""
        payload = {
            "symbol": ticker.upper(),
            "side": "BUY",
            "orderType": "LIMIT",
            "quantity": qty,
            "price": round(limit_price, 2),
        }
        if dry_run:
            return {"status": "DRY_RUN_SIMULATED", "order": payload}

        resp = requests.post(
            f"{self.base_url}/v1/overseas/orders",
            json=payload,
            headers=self._headers(),
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json()
