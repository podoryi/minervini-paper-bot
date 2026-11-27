import os
import time
from typing import Optional
import pandas as pd
import requests
import yfinance as yf


class KiwoomUSClient:
    """키움증권 REST API 미국주식 일봉 수집기 (NAS/NYS/AMS 거래소 자동 판별)"""

    def __init__(self, app_key: Optional[str] = None, app_secret: Optional[str] = None, is_mock: Optional[bool] = None):
        self.app_key = (app_key or os.environ.get("KIWOOM_APP_KEY", "")).strip()
        self.app_secret = (app_secret or os.environ.get("KIWOOM_APP_SECRET", "")).strip()
        self.is_mock = is_mock
        self.base_url = "https://mockapi.kiwoom.com" if is_mock else "https://api.kiwoom.com"
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._excd_cache: dict[str, str] = {"SPY": "AMS", "QQQ": "NAS", "IWM": "AMS"}

    @property
    def is_configured(self) -> bool:
        return bool(self.app_key and self.app_secret)

    def _get_token(self) -> str:
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        candidates = (
            [self.base_url]
            if self.is_mock is not None
            else ["https://api.kiwoom.com", "https://mockapi.kiwoom.com"]
        )
        last_err = ""
        for url in candidates:
            resp = requests.post(
                f"{url}/oauth2/token",
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
            if "token" in data:
                self.base_url = url
                self._access_token = data["token"]
                self._token_expires_at = time.time() + int(data.get("expires_in", 86400))
                return self._access_token
            last_err = f"[{url}] {data.get('return_msg', data)}"

        raise RuntimeError(f"키움 토큰 발급 실패 -> {last_err}")

    def get_daily_ohlcv(self, ticker: str, exchange: Optional[str] = None, count: int = 260) -> pd.DataFrame:
        sym = ticker.upper()
        if not self.is_configured:
            return self._fallback_yfinance(sym)

        try:
            token = self._get_token()
            headers = {
                "Content-Type": "application/json;charset=UTF-8",
                "authorization": f"Bearer {token}",
                "api-id": "ka10081",
            }
            exchanges = [exchange] if exchange else (
                [self._excd_cache[sym]] if sym in self._excd_cache else ["NAS", "NYS", "AMS"]
            )
            for excd in exchanges:
                payload = {"stk_cd": sym, "excd": excd, "upd_stkpc_tp": "1"}
                resp = requests.post(f"{self.base_url}/api/dostk/chart", json=payload, headers=headers, timeout=10)
                resp.raise_for_status()
                rows = resp.json().get("stk_dt_pole_chart_qry", [])
                if rows:
                    self._excd_cache[sym] = excd
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
                    return pd.DataFrame(records).sort_values("Date").set_index("Date")
            return self._fallback_yfinance(sym)
        except Exception:
            return self._fallback_yfinance(sym)

    @staticmethod
    def _fallback_yfinance(ticker: str) -> pd.DataFrame:
        df = yf.download(ticker, period="1y", progress=False)
        if df is not None and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
