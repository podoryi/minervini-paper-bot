import glob
import json
import math
import os
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from brokers.kiwoom_client import KiwoomUSClient

US_TZ = ZoneInfo("America/New_York")
KST_TZ = ZoneInfo("Asia/Seoul")
ALERT_LOG_PATH = "paper_logs/intraday_breakouts.txt"
VOL_MULTIPLIER_REQ = 1.50  # 진입 기준: 50일 평균 거래량 대비 150% 이상
ALLOC_PCT = 0.10           # 종목당 계좌 비중 10%
STOP_LOSS_PCT = 0.07       # 손절 기준 -7%


def load_dotenv_local(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def load_account_equity(default_equity: float = 1350.0) -> float:
    """paper_logs 내 계좌 JSON에서 현재 총자산을 읽어 반환"""
    equity = default_equity
    for p in sorted(glob.glob("paper_logs/*.json")):
        if "watchlist" in p or "breakout" in p:
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k in ("equity", "total_equity", "account_value", "nav", "cash"):
                    if k in data and isinstance(data[k], (int, float)) and data[k] > 0:
                        return float(data[k])
        except Exception:
            pass
    return equity


def calc_position_sizing(price: float, equity: float) -> tuple[float, float, float]:
    """계좌 10% 기준 (매수주수, 실투자금, -7% 손절가) 반환"""
    alloc = equity * ALLOC_PCT
    shares = math.floor((alloc / price) * 10000) / 10000.0 if price > 0 else 0.0
    invested = round(shares * price, 2)
    stop_price = round(price * (1.0 - STOP_LOSS_PCT), 2)
    return shares, invested, stop_price


def show_silent_toast(title: str, message: str) -> None:
    """소리 없이 Windows 우측 하단 배너 알림 표시"""
    ps_script = f"""
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $xml.LoadXml('<toast><audio silent="true"/><visual><binding template="ToastGeneric"><text>{title}</text><text>{message}</text></binding></visual></toast>')
    $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Minervini Bot").Show($toast)
    """
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", ps_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def append_breakout_log(line: str) -> None:
    os.makedirs(os.path.dirname(ALERT_LOG_PATH), exist_ok=True)
    with open(ALERT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_elapsed_session_ratio() -> tuple[float, bool]:
    """미국 정규장 경과 비율 계산 (일반 390분, 조기폐장 210분)"""
    now_us = datetime.now(US_TZ)
    is_early = (now_us.month == 11 and now_us.weekday() == 4 and 23 <= now_us.day <= 29) or (
        now_us.month in (7, 12) and now_us.day in (3, 24)
    )
    session_len = 210.0 if is_early else 390.0
    open_min = 9 * 60 + 30
    curr_min = now_us.hour * 60 + now_us.minute
    elapsed = max(curr_min - open_min, 5)
    return min(elapsed / session_len, 1.0), is_early


def run_monitor(interval_sec: int = 60) -> None:
    load_dotenv_local()
    kw = KiwoomUSClient()

    with open("paper_logs/active_watchlist.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    watchlist = raw if isinstance(raw, list) else next((v for v in raw.values() if isinstance(v, list)), [])

    avg_vol_cache: dict[str, float] = {}
    for item in watchlist:
        t = item.get("ticker")
        v_preset = float(item.get("avg_volume") or item.get("vol_ma50") or 0)
        if t and v_preset > 0:
            avg_vol_cache[t] = v_preset

    entry_alerted: set[str] = set()
    price_alerted: set[str] = set()
    equity = load_account_equity()
    print(f"[INIT] 감시 {len(watchlist)}종목 | 계좌자산: ${equity:,.2f} (10%: ${equity*ALLOC_PCT:,.2f}) | 거래량기준: {int(VOL_MULTIPLIER_REQ*100)}%")

    while True:
        now_kst = datetime.now(KST_TZ).strftime("%Y-%m-%d %H:%M:%S")
        short_time = now_kst.split(" ")[1]
        elapsed_ratio, is_early = get_elapsed_session_ratio()
        equity = load_account_equity(equity)

        ready_entries = []
        price_only = []
        near_pivots = []

        for item in watchlist:
            ticker = item.get("ticker")
            pivot = float(item.get("pivot_price") or item.get("pivot") or 0)
            if not ticker or pivot <= 0:
                continue

            try:
                need_hist = ticker not in avg_vol_cache
                df = kw.get_daily_ohlcv(ticker, count=60 if need_hist else 2)
                if df is None or df.empty:
                    continue

                if need_hist and len(df) >= 2:
                    avg_vol_cache[ticker] = float(df.iloc[:-1]["Volume"].tail(50).mean())

                c = df.iloc[-1]
                price = float(c["Close"])
                vol = int(c["Volume"])
                avg_vol = avg_vol_cache.get(ticker, 0.0)

                diff_pct = (price / pivot - 1.0) * 100.0
                raw_vol_pct = (vol / avg_vol * 100.0) if avg_vol > 0 else 0.0
                est_vol_pct = (raw_vol_pct / elapsed_ratio) if elapsed_ratio > 0 else raw_vol_pct

                if price >= pivot:
                    if est_vol_pct >= VOL_MULTIPLIER_REQ * 100.0:
                        shares, invested, stop_px = calc_position_sizing(price, equity)
                        ready_entries.append((ticker, price, pivot, diff_pct, vol, raw_vol_pct, est_vol_pct, shares, invested, stop_px))
                        if ticker not in entry_alerted:
                            entry_alerted.add(ticker)
                            msg = (
                                f"{ticker} ${price:.2f} ({diff_pct:+.2f}%) | 예상거래량 {est_vol_pct:.0f}% | "
                                f"매수 {shares:.4f}주(${invested:.2f}) | 손절 ${stop_px:.2f}"
                            )
                            show_silent_toast("★ [진입 신호] 피벗+거래량 충족", msg)
                            append_breakout_log(f"[{now_kst} KST] ★ [진입가능] {msg}")
                    else:
                        price_only.append((ticker, price, pivot, diff_pct, vol, raw_vol_pct, est_vol_pct))
                        if ticker not in price_alerted and ticker not in entry_alerted:
                            price_alerted.add(ticker)
                            msg = f"{ticker} 가격만 돌파 ${price:.2f} ({diff_pct:+.2f}%) | 예상거래량 {est_vol_pct:.0f}% (대기)"
                            append_breakout_log(f"[{now_kst} KST] ☆ [가격돌파] {msg}")
                elif diff_pct >= -1.5:
                    near_pivots.append((ticker, price, pivot, diff_pct, est_vol_pct))
            except Exception as e:
                print(f"[{short_time}] {ticker} 조회 오류: {e}")

        session_tag = "조기폐장" if is_early else "정규장"
        print(f"\n[{short_time} KST | {session_tag} {elapsed_ratio*100:.0f}% 경과] 진입가능 {len(ready_entries)} | 거래대기 {len(price_only)} | 임박 {len(near_pivots)}")
        for t, p, pv, dp, v, rvp, evp, sh, inv, stp in sorted(ready_entries, key=lambda x: x[6], reverse=True):
            print(
                f"  ★ [진입가능] {t:<5} ${p:7.2f} / 피벗 ${pv:7.2f} ({dp:+.2f}%) | "
                f"예상거래량 {evp:4.0f}% | 10%매수: {sh}주(${inv:.2f}) | 손절(-7%): ${stp:.2f}"
            )
        for t, p, pv, dp, v, rvp, evp in sorted(price_only, key=lambda x: x[6], reverse=True):
            print(f"  ☆ [거래대기] {t:<5} ${p:7.2f} / 피벗 ${pv:7.2f} ({dp:+.2f}%) | 예상거래량 {evp:4.0f}% (현재 {rvp:3.0f}%, {v:,}주)")
        for t, p, pv, dp, evp in sorted(near_pivots, key=lambda x: x[3], reverse=True):
            print(f"  · [돌파임박] {t:<5} ${p:7.2f} / 피벗 ${pv:7.2f} ({dp:+.2f}%) | 예상거래량 {evp:4.0f}%")

        time.sleep(interval_sec)


if __name__ == "__main__":
    run_monitor(interval_sec=60)
