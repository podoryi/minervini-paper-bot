
def _get_position_ohlcv_kiwoom_first(ticker: str, count: int = 60):
    """?? ??? ????? ???: ?? REST API ?? ?? ? ?? ? yfinance ??"""
    try:
        kw = KiwoomUSClient()
        if getattr(kw, "is_configured", False):
            df_kw = kw.get_daily_ohlcv(ticker, count=count)
            if df_kw is not None and not df_kw.empty:
                return df_kw
    except Exception as e:
        print(f"[WARN] {ticker} ???? ?? ?? ?? ??, yfinance ??: {e}")
    return None

from brokers.kiwoom_client import KiwoomUSClient
import os
import pickle
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import FinanceDataReader as fdr
import pandas as pd
import yfinance as yf

from config import TradingConfig
from screener import MinerviniScreener
from risk_manager import RiskManager
from execution_engine import ExecutionEngine
from broker_api import MockUSBroker

CACHE_DIR = "data_cache"
US_TZ = ZoneInfo("America/New_York")
KR_TZ = ZoneInfo("Asia/Seoul")

# 가상매매 초기 자본금: $1,350.00
INITIAL_CASH_USD = 1_350.0

# 미국 증시 휴장일 및 조기 폐장일 (13:00 ET 마감)
US_HOLIDAYS = {"2026-11-26", "2026-12-25"}
EARLY_CLOSE_DATES = {"2026-11-27", "2026-12-24"}

EXCLUDE_NAME_KEYWORDS = (
    "preferred", "pref", "depositary", "fund", "trust", "etf", "etn",
    "notes", "debenture", "bond", "warrant", "right", "unit",
    "acquisition", "income", "municipal", "portfolio", "%"
)


def get_full_us_universe():
    meta_map = {}
    for market in ["NASDAQ", "NYSE"]:
        ex_code = "NASD" if market == "NASDAQ" else "NYSE"
        try:
            listing = fdr.StockListing(market)
            for _, row in listing.iterrows():
                sym = str(row["Symbol"]).strip().replace(".", "-")
                name = str(row.get("Name", sym))
                name_lower = name.lower()

                if not sym.replace("-", "").isalpha():
                    continue
                if len(sym) > 5 or (len(sym) == 5 and sym[-1] in {"M", "N", "O", "P", "Q", "R", "U", "V", "W", "Z"}):
                    continue
                if any(kw in name_lower for kw in EXCLUDE_NAME_KEYWORDS):
                    continue

                meta_map[sym] = {"name": name, "exchange": ex_code}
        except Exception as e:
            print(f"{market} 목록 로드 경고: {e}")

    if not meta_map:
        fallback = ["NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "PLTR", "LLY", "TGTX"]
        meta_map = {t: {"name": t, "exchange": "NASD"} for t in fallback}

    return sorted(list(meta_map.keys())), meta_map


def load_us_market_data_fast(tickers: list, cfg: TradingConfig, held_tickers: list = None, days: int = 400):
    os.makedirs(CACHE_DIR, exist_ok=True)
    held_tickers = set(held_tickers or [])
    us_today = datetime.now(US_TZ).strftime("%Y%m%d")
    cache_path = os.path.join(CACHE_DIR, f"us_ohlcv_inst_{us_today}.pkl")

    if os.path.exists(cache_path):
        print(f"당일 기관 주도주 캐시 데이터 로드 중 ({cache_path})...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    start_date = (datetime.now(US_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")

    spy_raw = yf.download("SPY", start=start_date, progress=False)
    if spy_raw is not None and isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    index_df = spy_raw.dropna(subset=["Close"]).copy() if spy_raw is not None else pd.DataFrame()

    if index_df.empty:
        raise RuntimeError("SPY 벤치마크 데이터 수신 실패. 1분 후 재시도하세요.")

    universe_data = {}
    chunk_size = 150
    total_chunks = (len(tickers) + chunk_size - 1) // chunk_size
    print(f"미국 보통주 {len(tickers)}개 ({total_chunks}개 배치) 분할 수집 및 기관 유동성($50M/일+) 필터링 시작...")

    for idx in range(total_chunks):
        batch = tickers[idx * chunk_size : (idx + 1) * chunk_size]
        for attempt in range(2):
            try:
                raw = yf.download(
                    tickers=batch,
                    start=start_date,
                    group_by="ticker",
                    threads=4,
                    progress=False,
                )
                if raw is None or raw.empty:
                    raise ValueError("Empty batch")

                for t in batch:
                    try:
                        df = raw[t].dropna(subset=["Close"]).copy() if len(batch) > 1 else raw.dropna(subset=["Close"]).copy()
                        if len(df) >= 225:
                            avg_dollar_vol = float((df["Close"] * df["Volume"]).tail(50).mean())
                            is_liquid = float(df["Close"].iloc[-1]) >= cfg.min_price and avg_dollar_vol >= cfg.min_dollar_volume
                            if is_liquid or (t in held_tickers):
                                universe_data[t] = df
                    except Exception:
                        continue
                break
            except Exception:
                if attempt == 0:
                    time.sleep(8)

        if (idx + 1) % 10 == 0 or idx == total_chunks - 1:
            print(f" -> 진행률 {idx + 1}/{total_chunks} 배치 완료 (유효 기관주도주: {len(universe_data)}개)")
        if idx < total_chunks - 1:
            time.sleep(1.5)

    with open(cache_path, "wb") as f:
        pickle.dump((universe_data, index_df), f)

    return universe_data, index_df


def run_daily_screening(
    screener: MinerviniScreener,
    engine: ExecutionEngine,
    broker: MockUSBroker,
    tickers: list,
    meta_map: dict,
):
    t0 = time.time()
    kr_now = datetime.now(KR_TZ).strftime("%Y-%m-%d %H:%M:%S")
    us_now = datetime.now(US_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[KST {kr_now} | ET {us_now}] 미국장 4대 원칙 + 정량대용지표 주도주 스크리닝 시작...")

    universe_data, index_df = load_us_market_data_fast(
        tickers, screener.cfg, held_tickers=list(engine.positions.keys())
    )

    for t, df in universe_data.items():
        broker.update_price(t, float(df["Close"].iloc[-1]))

    spy_close = float(index_df["Close"].iloc[-1])
    spy_sma50 = float(index_df["Close"].rolling(50).mean().iloc[-1])
    spy_sma200 = float(index_df["Close"].rolling(200).mean().iloc[-1])

    if spy_close < spy_sma200:
        screener.cfg.market_regime = "BEAR"
    elif spy_close < spy_sma50:
        screener.cfg.market_regime = "CAUTION"
    else:
        screener.cfg.market_regime = "BULL"

    print(
        f"SPY 종가: ${spy_close:,.2f} (50일선: ${spy_sma50:,.2f} | 200일선: ${spy_sma200:,.2f}) -> "
        f"[{screener.cfg.market_regime}] 모드 (최대 {screener.cfg.max_positions}종목 | "
        f"기본비중 {screener.cfg.initial_position_pct*100:.0f}% | 손절 {screener.cfg.hard_stop_pct*100:.1f}%)"
    )

    signals = screener.scan_universe(universe_data, index_df, meta_map)
    engine.load_watchlist(signals)
    broker.log_watchlist_snapshot(signals, screener.cfg.market_regime)

    print(f"\n스크리닝 완료 ({time.time() - t0:.1f}초 소요) | 최종 주도주 돌파 감시 목록: {len(signals)}개")
    print("=" * 78)
    print(" [사람 최종 확인용 TOP 5 핵심 주도주 브리핑]")
    for idx, s in enumerate(signals[:5], start=1):
        print(
            f" {idx}위. {s.ticker} ({s.name}) | {s.pattern_type} | RS:{s.rs_score} | "
            f"피벗:${s.pivot_price:,.2f} (손절:${s.suggested_stop:,.2f}) | 수축:{s.contractions}\n"
            f"      ├ 정량지표: {s.fund_summary}\n"
            f"      ├ LLM판정: {s.llm_note}\n"
            f"      └ 최근뉴스: {s.catalyst_headline[:75]}"
        )
    print("=" * 78)

    if len(signals) > 5:
        print(" [나머지 돌파 감시 후보군]")
        for s in signals[5:]:
            print(
                f" -> {s.ticker} | RS:{s.rs_score} | 피벗:${s.pivot_price:,.2f} | "
                f"{s.fund_summary} | 뉴스:{s.catalyst_headline[:55]}"
            )

    return universe_data


def run_intraday_monitor(engine: ExecutionEngine, broker: MockUSBroker, universe_data: dict):
    us_now = datetime.now(US_TZ)
    today_str = us_now.strftime("%Y-%m-%d")

    if us_now.weekday() >= 5 or today_str in US_HOLIDAYS:
        return

    is_early = today_str in EARLY_CLOSE_DATES
    minutes_of_day = us_now.hour * 60 + us_now.minute
    market_open_min = 9 * 60 + 30
    market_close_min = (13 * 60) if is_early else (16 * 60)
    session_length = float(market_close_min - market_open_min)

    if not (market_open_min <= minutes_of_day < market_close_min):
        return

    active_tickers = list(dict.fromkeys(list(engine.watchlist.keys()) + list(engine.positions.keys())))
    if not active_tickers:
        return

    elapsed_min = minutes_of_day - market_open_min
    elapsed_ratio = max(elapsed_min / session_length, 0.05)
    live_data = yf.download(tickers=active_tickers, period="1d", interval="1m", group_by="ticker", progress=False)
    if live_data is None or live_data.empty:
        return

    # 동일 섹터 최대 2종목 쏠림 제한을 위한 현재 보유 섹터 집계
    sector_counts = {}
    for held_t in engine.positions.keys():
        sig_obj = engine.watchlist.get(held_t)
        sec = getattr(sig_obj, "sector", "Unknown") if sig_obj else "Unknown"
        if sec != "Unknown":
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

    for ticker in active_tickers:
        if ticker not in universe_data:
            continue
        try:
            df_daily = universe_data[ticker]
            vol_sma50 = float(df_daily["Volume"].rolling(50).mean().iloc[-1])
            sma20 = float(df_daily["Close"].rolling(20).mean().iloc[-1])
            prev_high5 = float(df_daily["High"].rolling(5).max().iloc[-1])

            if len(active_tickers) == 1:
                t_live = live_data.copy()
                if isinstance(t_live.columns, pd.MultiIndex):
                    t_live.columns = t_live.columns.get_level_values(0)
                t_live = t_live.dropna(subset=["Close"])
            else:
                t_live = live_data[ticker].dropna(subset=["Close"])

            if t_live.empty:
                continue

            open_price = float(t_live["Open"].iloc[0])
            curr_price = float(t_live["Close"].iloc[-1])
            curr_vol = float(t_live["Volume"].sum())
            projected_vol = curr_vol / elapsed_ratio

            broker.update_price(ticker, curr_price)

            # 신규 진입 종목 방어: (1) 개장 첫 10분간 시가 위 양봉 필수 (2) 동일 섹터 최대 2종목 제한
            if ticker not in engine.positions:
                if elapsed_min < 10 and curr_price <= open_price:
                    continue
                sig_obj = engine.watchlist.get(ticker)
                sec = getattr(sig_obj, "sector", "Unknown") if sig_obj else "Unknown"
                if sec != "Unknown" and sector_counts.get(sec, 0) >= 2:
                    continue

            before_cnt = len(engine.positions)
            engine.on_intraday_tick(
                ticker=ticker,
                current_price=curr_price,
                current_vol=curr_vol,
                projected_vol=projected_vol,
                vol_sma50=vol_sma50,
                sma20=sma20,
                prev_high5=prev_high5,
                now=us_now,
            )
            if len(engine.positions) > before_cnt and ticker in engine.positions:
                sig_obj = engine.watchlist.get(ticker)
                sec = getattr(sig_obj, "sector", "Unknown") if sig_obj else "Unknown"
                if sec != "Unknown":
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
        except Exception:
            continue

    broker.save_engine_positions(engine)


def run_closing_check(engine: ExecutionEngine, broker: MockUSBroker, regime: str):
    us_now = datetime.now(US_TZ)
    today_str = us_now.strftime("%Y-%m-%d")
    is_early = today_str in EARLY_CLOSE_DATES
    start_date = (us_now - timedelta(days=120)).strftime("%Y-%m-%d")
    pos_tickers = list(engine.positions.keys())

    # 마감 10분 전 기준 경과 시간 비율(일반 380/390, 조기폐장 200/210)로 당일 최종 거래량 보정
    vol_scale = (210.0 / 200.0) if is_early else (390.0 / 380.0)

    if pos_tickers:
        raw = _get_position_ohlcv_kiwoom_first(tickers=pos_tickers, count=60)
        if raw is None or raw.empty:
            raw = yf.download(tickers=pos_tickers, start=start_date, group_by="ticker", progress=False)
        if raw is not None and not raw.empty:
            for ticker in pos_tickers:
                try:
                    if len(pos_tickers) == 1:
                        df = raw.copy()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        df = df.dropna(subset=["Close"])
                    else:
                        df = raw[ticker].dropna(subset=["Close"])

                    if not df.empty and df.index[-1].strftime("%Y-%m-%d") == today_str:
                        df = df.copy()
                        df.iloc[-1, df.columns.get_loc("Volume")] *= vol_scale

                    broker.update_price(ticker, float(df["Close"].iloc[-1]))
                    engine.on_market_close_check(ticker, df, us_now)
                except Exception:
                    continue

    broker.save_engine_positions(engine)
    broker.log_daily_equity(regime)


if __name__ == "__main__":
    cfg = TradingConfig()
    broker = MockUSBroker(initial_cash=INITIAL_CASH_USD)
    screener = MinerviniScreener(cfg)
    risk_mgr = RiskManager(cfg)
    engine = ExecutionEngine(cfg, risk_mgr, broker)

    broker.sync_engine_positions(engine)
    target_tickers, meta_map = get_full_us_universe()

    cached_data = run_daily_screening(screener, engine, broker, target_tickers, meta_map)

    last_screen_date = datetime.now(US_TZ).strftime("%Y-%m-%d")
    last_close_check_date = None

    print("\n[$1,350.00 가상매매 엔진 가동 중] 모든 기록은 paper_logs/ 폴더에 자동 저장됩니다. (종료: Ctrl + C)")
    while True:
        us_now = datetime.now(US_TZ)
        today_str = us_now.strftime("%Y-%m-%d")

        if us_now.weekday() >= 5 or today_str in US_HOLIDAYS:
            time.sleep(60)
            continue

        is_early = today_str in EARLY_CLOSE_DATES

        # 1. 미국 장전 08:30 ET 일일 스크리닝
        if us_now.hour == 8 and us_now.minute >= 30 and last_screen_date != today_str:
            cached_data = run_daily_screening(screener, engine, broker, target_tickers, meta_map)
            last_screen_date = today_str

        # 2. 미국 장 마감 10분 전 (조기폐장일 12:50 ET / 일반일 15:50 ET) 청산 체크 및 일별 계좌 스냅샷 기록
        close_check_reached = (
            (is_early and us_now.hour == 12 and us_now.minute >= 50)
            or (not is_early and us_now.hour == 15 and us_now.minute >= 50)
        )
        if close_check_reached and last_close_check_date != today_str:
            run_closing_check(engine, broker, cfg.market_regime)
            last_close_check_date = today_str

        # 3. 미국 정규장 실시간 틱 감시 (15초 주기)
        run_intraday_monitor(engine, broker, cached_data)
        time.sleep(15)
