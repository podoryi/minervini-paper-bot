import os
import time
from typing import Optional
import pandas as pd
import requests
import yfinance as yf


class KiwoomUSClient:
    """키움증권 REST API 미국주식 일봉 수집기 (GitHub Actions IP 무제한 환경용)"""

    def __init__(self, app_key: Optional[str] = None, app_secret: Optional[str] = None, is_mock: bool = False):
        self.app_key = (app_key or os.environ.get("KIWOOM_APP_KEY", "")).strip()
        self.app_secret = (app_secret or os.environ.get("KIWOOM_APP_SECRET", "")).strip()
        self.base_url = "https://mockapi.kiwoom.com" if is_mock else "https://api.kiwoom.com"
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    @property
    def is_configured(self) -> bool:
        return bool(self.app_key and self.app_secret)

    def _get_token(self) -> str:
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token
        resp = requests.post(
            f"{self.base_url}/oauth2/token",
            json={
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "secretkey": self.app_secret,
            },
            headers={"Content-Type": "application/json;charset=UTF-8"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["token"]
        self._token_expires_at = time.time() + int(data.get("expires_in", 86400))
        return self._access_token

    def get_daily_ohlcv(self, ticker: str, exchange: str = "NAS", count: int = 260) -> pd.DataFrame:
        """미국주식 일봉 조회 (키움 API 미설정 또는 응답 지연 시 yfinance 폴백)"""
        if not self.is_configured:
            return self._fallback_yfinance(ticker)

        try:
            token = self._get_token()
            headers = {
                "Content-Type": "application/json;charset=UTF-8",
                "authorization": f"Bearer {token}",
                "api-id": "ka10081",
            }
            payload = {"stk_cd": ticker.upper(), "excd": exchange, "upd_stkpc_tp": "1"}
            resp = requests.post(f"{self.base_url}/api/dostk/chart", json=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            rows = resp.json().get("stk_dt_pole_chart_qry", [])
            if not rows:
                return self._fallback_yfinance(ticker)

            records = []
            for r in rows[:count]:
                records.append({
                    "Date": pd.to_datetime(r["dt"]),
                    "Open": abs(float(r["open_pric"])),
                    "High": abs(float(r["high_pric"])),
                    "Low": abs(float(r["low_pric"])),
                    "Close": abs(float(r["cur_prc"])),
                    "Volume": int(abs(float(r["trde_qty"]))),
                })
            df = pd.DataFrame(records).sort_values("Date").set_index("Date")
            return df
        except Exception:
            return self._fallback_yfinance(ticker)

    @staticmethod
    def _fallback_yfinance(ticker: str) -> pd.DataFrame:
        df = yf.download(ticker, period="1y", progress=False)
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)
        return df
