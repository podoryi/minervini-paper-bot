import json
import os
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from config import TradingConfig
from screener import MinerviniScreener, ScreenerSignal
from risk_manager import RiskManager
from execution_engine import ExecutionEngine
from broker_api import MockUSBroker
from main import get_full_us_universe, load_us_market_data_fast
from review_paper import summarize_paper_trading

US_TZ = ZoneInfo("America/New_York")
LOG_DIR = "paper_logs"
ACTIVE_WL_PATH = os.path.join(LOG_DIR, "active_watchlist.json")
LAST_DATE_PATH = os.path.join(LOG_DIR, "last_processed_date.txt")
INITIAL_CASH_USD = 1_350.0


def load_saved_watchlist() -> list[ScreenerSignal]:
    if not os.path.exists(ACTIVE_WL_PATH):
        return []
    try:
        with open(ACTIVE_WL_PATH, "r", encoding="utf-8") as f:
            items = json.load(f)
        return [ScreenerSignal(**item) for item in items]
    except Exception:
        return []


def save_watchlist(signals: list[ScreenerSignal]) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(ACTIVE_WL_PATH, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in signals], f, ensure_ascii=False, indent=2)


def get_last_processed_date() -> str:
    if not os.path.exists(LAST_DATE_PATH):
        return ""
    with open(LAST_DATE_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def set_last_processed_date(date_str: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LAST_DATE_PATH, "w", encoding="utf-8") as f:
        f.write(date_str)


def run_daily_batch() -> None:
    cfg = TradingConfig()
    broker = MockUSBroker(initial_cash=INITIAL_CASH_USD)
    screener = MinerviniScreener(cfg)
    risk_mgr = RiskManager(cfg)
    engine = ExecutionEngine(cfg, risk_mgr, broker)

    broker.sync_engine_positions(engine)
    prev_signals = load_saved_watchlist()
    engine.load_watchlist(prev_signals)

    tickers, meta_map = get_full_us_universe()
    universe_data, index_df = load_us_market_data_fast(
        tickers, cfg, held_tickers=list(engine.positions.keys())
    )

    latest_trading_date = index_df.index[-1].strftime("%Y-%m-%d")
    last_done_date = get_last_processed_date()
    us_now = datetime.now(US_TZ)

    spy_close = float(index_df["Close"].iloc[-1])
    spy_sma50 = float(index_df["Close"].rolling(50).mean().iloc[-1])
    spy_sma200 = float(index_df["Close"].rolling(200).mean().iloc[-1])

    if spy_close < spy_sma200:
        cfg.market_regime = "BEAR"
    elif spy_close < spy_sma50:
        cfg.market_regime = "CAUTION"
    else:
        cfg.market_regime = "BULL"

    print(f"[기준 거래일: {latest_trading_date} | 이전 처리일: {last_done_date or '없음'} | 시장국면: {cfg.market_regime}]")

    if latest_trading_date != last_done_date and prev_signals:
        print(f"전일 감시 피벗({len(prev_signals)}개) 및 보유 종목({len(engine.positions)}개) 일봉 정산 중...")

        for ticker in list(engine.positions.keys()):
            if ticker not in universe_data:
                continue
            df = universe_data[ticker]
            bar = df.iloc[-1]
            o_px, h_px, l_px, c_px, vol = (
                float(bar["Open"]), float(bar["High"]), float(bar["Low"]),
                float(bar["Close"]), float(bar["Volume"])
            )
            vol_sma50 = float(df["Volume"].rolling(50).mean().iloc[-1])
            sma20 = float(df["Close"].rolling(20).mean().iloc[-1])
            prev_high5 = float(df["High"].iloc[:-1].rolling(5).max().iloc[-1]) if len(df) > 6 else h_px

            pos = engine.positions[ticker]
            if l_px <= pos.stop_price:
                fill_stop = round(min(o_px, pos.stop_price), 2)
                broker.update_price(ticker, fill_stop)
                engine.on_intraday_tick(ticker, fill_stop, vol, vol, vol_sma50, sma20, prev_high5, us_now)
                continue

            broker.update_price(ticker, h_px)
            engine.on_intraday_tick(ticker, h_px, vol, vol, vol_sma50, sma20, prev_high5, us_now)

            if ticker in engine.positions:
                broker.update_price(ticker, c_px)
                engine.on_market_close_check(ticker, df, us_now)

        sector_counts = {}
        for held_t in engine.positions.keys():
            sec = getattr(engine.watchlist.get(held_t), "sector", "Unknown")
            if sec != "Unknown":
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

        for sig in prev_signals:
            ticker = sig.ticker
            if ticker in engine.positions or ticker not in universe_data:
                continue
            if sig.sector != "Unknown" and sector_counts.get(sig.sector, 0) >= 2:
                continue

            df = universe_data[ticker]
            bar = df.iloc[-1]
            o_px, h_px, l_px, c_px, vol = (
                float(bar["Open"]), float(bar["High"]), float(bar["Low"]),
                float(bar["Close"]), float(bar["Volume"])
            )
            vol_sma50 = float(df["Volume"].rolling(50).mean().iloc[-1])
            sma20 = float(df["Close"].rolling(20).mean().iloc[-1])
            prev_high5 = float(df["High"].iloc[:-1].rolling(5).max().iloc[-1]) if len(df) > 6 else h_px

            if (
                o_px <= sig.pivot_price * 1.025
                and h_px >= sig.pivot_price
                and c_px > o_px
                and vol >= vol_sma50 * 1.35
            ):
                fill_px = round(max(o_px, sig.pivot_price), 2)
                broker.update_price(ticker, fill_px)
                engine.on_intraday_tick(ticker, fill_px, vol, vol, vol_sma50, sma20, prev_high5, us_now)

                if ticker in engine.positions:
                    if sig.sector != "Unknown":
                        sector_counts[sig.sector] = sector_counts.get(sig.sector, 0) + 1
                    if l_px <= engine.positions[ticker].stop_price:
                        stop_fill = round(engine.positions[ticker].stop_price, 2)
                        broker.update_price(ticker, stop_fill)
                        engine.on_intraday_tick(ticker, stop_fill, vol, vol, vol_sma50, sma20, prev_high5, us_now)
                    else:
                        broker.update_price(ticker, c_px)

        broker.save_engine_positions(engine)
        broker.log_daily_equity(cfg.market_regime)
        set_last_processed_date(latest_trading_date)
    else:
        if latest_trading_date == last_done_date:
            print("휴장일 또는 이미 정산 완료된 거래일입니다. 중복 매매를 건너뜁니다.")
        else:
            set_last_processed_date(latest_trading_date)

    for t, df in universe_data.items():
        broker.update_price(t, float(df["Close"].iloc[-1]))

    new_signals = screener.scan_universe(universe_data, index_df, meta_map)
    save_watchlist(new_signals)
    broker.log_watchlist_snapshot(new_signals, cfg.market_regime)
    broker.save_engine_positions(engine)

    print(f"\n[다음 거래일 피벗 감시 목록 갱신 완료: {len(new_signals)}종목]")
    for idx, s in enumerate(new_signals[:5], start=1):
        print(
            f" {idx}위. {s.ticker} | RS:{s.rs_score} | 피벗:${s.pivot_price:,.2f} "
            f"(손절:${s.suggested_stop:,.2f}) | {s.fund_summary}"
        )

    summarize_paper_trading()


if __name__ == "__main__":
    run_daily_batch()
