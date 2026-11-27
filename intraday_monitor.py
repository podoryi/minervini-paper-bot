import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List
import yfinance as yf
from brokers.toss_client import TossSecuritiesClient

WATCHLIST_PATH = os.path.join("paper_logs", "active_watchlist.json")
INTRADAY_ALERTS_PATH = os.path.join("paper_logs", "intraday_breakouts.json")
LIVE_TRADING_ENABLED = False  # 가상매매 검증 완료 후 True로 전환


def load_dotenv_local(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_watchlist() -> List[Dict]:
    if not os.path.exists(WATCHLIST_PATH):
        return []
    with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def calc_session_elapsed_ratio(is_half_day: bool = False) -> float:
    """미국 정규장(09:30~16:00 ET, 조기폐장일 13:00 ET) 경과 비율 계산"""
    now_utc = datetime.now(timezone.utc)
    # EST (UTC-5) 기준 정규장 시작 14:30 UTC
    open_min = 14 * 60 + 30
    curr_min = now_utc.hour * 60 + now_utc.minute
    total_min = 210 if is_half_day else 390
    elapsed = max(1, min(total_min, curr_min - open_min))
    return elapsed / total_min


def run_monitor_loop(poll_sec: int = 60, is_half_day: bool = False) -> None:
    load_dotenv_local()
    toss = TossSecuritiesClient()
    watchlist = load_watchlist()
    triggered = set()

    print(f"[INIT] 감시 종목 {len(watchlist)}개 로드 | 토스 API 연결: {toss.is_configured} | 실주문 모드: {LIVE_TRADING_ENABLED}")

    while True:
        elapsed_ratio = calc_session_elapsed_ratio(is_half_day=is_half_day)
        now_str = datetime.now().strftime("%H:%M:%S")

        for item in watchlist:
            ticker = item.get("ticker")
            pivot = float(item.get("pivot_price") or item.get("pivot") or 0)
            stop_loss = float(item.get("stop_loss") or 0)
            avg_vol_50 = float(item.get("avg_vol_50") or item.get("vol_sma50") or 0)
            if not ticker or pivot <= 0 or ticker in triggered:
                continue

            try:
                if toss.is_configured:
                    q = toss.get_us_quote(ticker)
                else:
                    tk = yf.Ticker(ticker)
                    fi = tk.fast_info
                    q = {
                        "ticker": ticker,
                        "price": float(fi.last_price),
                        "open": float(fi.open),
                        "prev_close": float(fi.previous_close),
                        "volume": int(fi.last_volume or 0),
                    }

                price = q["price"]
                open_p = q["open"] or price
                prev_c = q["prev_close"] or open_p
                gap_pct = (open_p - prev_c) / prev_c if prev_c > 0 else 0.0
                projected_vol = q["volume"] / elapsed_ratio if elapsed_ratio > 0 else q["volume"]
                vol_pace = (projected_vol / avg_vol_50) if avg_vol_50 > 0 else 1.5

                # 미너비니 실시간 돌파 조건: 피벗 돌파 + 시가갭 +2.5% 이내 + 예상 거래량 1.35배 이상
                if price >= pivot and gap_pct <= 0.025 and vol_pace >= 1.35:
                    triggered.add(ticker)
                    order_res = toss.place_us_buy_order(
                        ticker=ticker,
                        qty=max(1, int(337.5 // price)),
                        limit_price=price * 1.003,
                        dry_run=(not LIVE_TRADING_ENABLED),
                    )
                    print(f"[{now_str}] BREAKOUT {ticker}: 현재가 ${price:.2f} >= 피벗 ${pivot:.2f} | 거래량페이스 {vol_pace:.2f}x | 손절 ${stop_loss:.2f}")
                    print(f"   -> 주문 처리 결과: {order_res}")
            except Exception as e:
                print(f"[{now_str}] {ticker} 조회 오류: {e}")

        time.sleep(poll_sec)


if __name__ == "__main__":
    # 오늘(11/27 블랙프라이데이)은 조기 폐장일이므로 is_half_day=True
    run_monitor_loop(poll_sec=60, is_half_day=True)
