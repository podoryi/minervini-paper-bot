import os
import pickle
import time
from datetime import date, datetime
import numpy as np
import pandas as pd
import yfinance as yf

from config import TradingConfig
from risk_manager import RiskManager

CACHE_DIR = "data_cache"
MAX_PORTFOLIO_HEAT = 0.06  # 계좌 전체 총 위험(Heat) 6% 상한


def detect_swing_vcp(sub_df: pd.DataFrame, max_last: float = 0.12) -> dict | None:
    """minervini.py의 스윙 고점·저점 파동 기반 VCP 수축 탐지"""
    h = sub_df["High"].values
    l = sub_df["Low"].values
    w = 5
    pts = []
    for i in range(w, len(sub_df) - w):
        if h[i] == h[i - w : i + w + 1].max():
            pts.append((i, "H", h[i]))
        elif l[i] == l[i - w : i + w + 1].min():
            pts.append((i, "L", l[i]))

    sw = []
    for p in pts:
        if sw and sw[-1][1] == p[1]:
            if (p[1] == "H" and p[2] > sw[-1][2]) or (p[1] == "L" and p[2] < sw[-1][2]):
                sw[-1] = p
        else:
            sw.append(p)

    cons = []
    for a, b in zip(sw, sw[1:]):
        if a[1] == "H" and b[1] == "L" and a[2] > 0:
            cons.append(((a[2] - b[2]) / a[2], a[0], b[0], b[2]))
    cons = cons[-4:]
    if len(cons) < 2:
        return None

    depths = [x[0] for x in cons]
    strict = all(depths[i + 1] <= depths[i] * 1.05 for i in range(len(depths) - 1))
    relaxed = depths[-1] <= depths[0] and depths[-1] <= max(depths) * 0.8

    if not ((strict or relaxed) and min(depths) >= 0.015 and depths[-1] <= max_last and depths[0] <= 0.42):
        return None

    last = cons[-1]
    # 마지막 수축 시작 고점 이후 최고가를 피벗으로 설정
    slice_hi = sub_df["High"].iloc[last[1]:]
    if slice_hi.empty:
        return None
    pivot = float(slice_hi.max())
    low_px = float(last[3])
    return {
        "pivot": round(pivot, 2),
        "low": low_px,
        "strict": strict,
        "depths": [round(x * 100, 1) for x in depths],
    }


def build_nongaap_eps_cache(tickers: list) -> dict:
    """yfinance Reported EPS(Non-GAAP) 히스토리 로컬 캐싱"""
    cache_path = os.path.join(CACHE_DIR, "yf_reported_eps_bt.pkl")
    eps_db = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            eps_db = pickle.load(f)

    missing = [t for t in tickers if t not in eps_db]
    if not missing:
        return eps_db

    print(f"Non-GAAP Reported EPS 히스토리 수집 중: {len(missing)}개 종목 (1회 캐싱)...")
    for idx, t in enumerate(missing):
        try:
            time.sleep(0.08)
            ed = yf.Ticker(t).get_earnings_dates(limit=16)
            if ed is not None and "Reported EPS" in ed.columns:
                s = ed["Reported EPS"].dropna().sort_index(ascending=False)
                records = [(idx_dt.date(), float(val)) for idx_dt, val in s.items()]
                eps_db[t] = records
            else:
                eps_db[t] = []
        except Exception:
            eps_db[t] = []

        if (idx + 1) % 100 == 0 or idx == len(missing) - 1:
            print(f" -> Reported EPS 진행률: {idx + 1}/{len(missing)}")

    with open(cache_path, "wb") as f:
        pickle.dump(eps_db, f)
    return eps_db


def check_hybrid_fundamentals(sec_entry: dict, yf_eps_list: list, as_of: date) -> bool:
    """SEC GAAP 공시 또는 Non-GAAP Reported EPS 중 하나라도 실적 성장(흑자·YoY 20%+·가속) 충족 시 통과"""
    # 1) Non-GAAP Reported EPS 우선 확인
    avail_yf = [(d, v) for d, v in (yf_eps_list or []) if d <= as_of]
    if len(avail_yf) >= 5:
        q0, q1, q4, q5 = avail_yf[0][1], avail_yf[1][1], avail_yf[4][1], (avail_yf[5][1] if len(avail_yf) >= 6 else None)
        if q0 > 0:
            if q4 < 0:
                return True  # 흑자 전환
            if q4 > 0:
                yoy0 = (q0 - q4) / abs(q4)
                if yoy0 >= 0.20:
                    if q5 is not None and q5 != 0:
                        yoy1 = (q1 - q5) / abs(q5)
                        if yoy0 >= yoy1 * 0.9:
                            return True
                    else:
                        return True

    # 2) SEC EDGAR 매출·순이익 보완 확인
    if not sec_entry:
        return False

    def eval_sec(series: list):
        avail = [(d_e, val) for d_f, d_e, val in series if d_f <= as_of]
        if not avail:
            return False, None, False
        q0_e, q0_v = avail[0]
        is_prof = q0_v > 0
        g0 = None
        for j in range(1, len(avail)):
            pe, pv = avail[j]
            if 345 <= (q0_e - pe).days <= 385 and pv != 0:
                g0 = 999.0 if (pv < 0 and q0_v > 0) else ((q0_v - pv) / abs(pv) * 100.0 if pv > 0 else None)
                break
        return is_prof, g0, (len(avail) >= 3 and avail[0][1] > avail[1][1] > avail[2][1] > 0)

    p_prof, p_yoy, _ = eval_sec(sec_entry.get("profit", []))
    _, r_yoy, r_acc = eval_sec(sec_entry.get("rev", []))
    return bool((p_prof and p_yoy is not None and p_yoy >= 20.0) or (p_prof and r_yoy is not None and r_yoy >= 20.0) or (r_yoy is not None and r_yoy >= 25.0 and r_acc))


def run_hybrid_backtest(years: int = 2, initial_cash: float = 1_350.0, slippage_pct: float = 0.001):
    t0 = time.time()
    cfg = TradingConfig()
    risk_mgr = RiskManager(cfg)

    cache_files = sorted([f for f in os.listdir(CACHE_DIR) if f.startswith("us_clean_bt_") and f.endswith(f"_{years}y.pkl")])
    with open(os.path.join(CACHE_DIR, cache_files[-1]), "rb") as f:
        raw_universe, index_df = pickle.load(f)

    sec_cache_path = os.path.join(CACHE_DIR, "sec_bt_pit_history.pkl")
    sec_hist = pickle.load(open(sec_cache_path, "rb")) if os.path.exists(sec_cache_path) else {}

    filtered_universe = {}
    for ticker, df in raw_universe.items():
        avg_dvol = float((df["Close"] * df["Volume"]).tail(100).mean())
        if float(df["Close"].iloc[-1]) >= cfg.min_price and avg_dvol >= cfg.min_dollar_volume:
            filtered_universe[ticker] = df

    prepared = {}
    rs_frames = {}
    idx_ret63 = index_df["Close"].pct_change(63)
    idx_ret126 = index_df["Close"].pct_change(126)
    idx_ret252 = index_df["Close"].pct_change(252)

    for ticker, raw_df in filtered_universe.items():
        df = raw_df.copy()
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA150"] = df["Close"].rolling(150).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["VOL_SMA50"] = df["Volume"].rolling(50).mean()
        df["HIGH_52W"] = df["High"].rolling(252, min_periods=200).max()
        df["LOW_52W"] = df["Low"].rolling(252, min_periods=200).min()
        df["HIGH_5D"] = df["High"].rolling(5).max()

        surge_hi = df["High"].shift(12).rolling(40).max()
        surge_lo = df["Low"].shift(12).rolling(40).min()
        base_hi = df["High"].rolling(12).max()
        base_lo = df["Low"].rolling(12).min()
        peak = np.maximum(surge_hi, base_hi)
        r5 = (df["High"].rolling(5).max() - df["Low"].rolling(5).min()) / df["High"].rolling(5).max()
        df["IS_POWER_PLAY"] = ((surge_hi - surge_lo) / surge_lo >= 0.90) & ((peak - base_lo) / peak <= 0.18) & (r5 <= 0.10)
        df["PP_PIVOT"] = base_hi
        df["PP_LOW"] = base_lo

        cond_ma = (df["Close"] > df["SMA50"]) & (df["SMA50"] > df["SMA150"]) & (df["SMA150"] > df["SMA200"])
        cond_200up = df["SMA200"] > df["SMA200"].shift(cfg.sma200_uptrend_days)
        cond_low = df["Close"] >= df["LOW_52W"] * (1 + cfg.min_above_52w_low)
        cond_high = df["Close"] >= df["HIGH_52W"] * (1 - cfg.max_below_52w_high)
        df["STAGE2"] = cond_ma & cond_200up & cond_low & cond_high

        s_ret63 = df["Close"].pct_change(63)
        s_ret126 = df["Close"].pct_change(126)
        s_ret252 = df["Close"].pct_change(252)
        rs_frames[ticker] = (s_ret63 - idx_ret63) * 0.4 + (s_ret126 - idx_ret126) * 0.3 + (s_ret252 - idx_ret252) * 0.3
        prepared[ticker] = df

    rs_df = pd.DataFrame(rs_frames).rank(axis=1, pct=True) * 100.0
    for ticker in prepared:
        prepared[ticker]["RS_PCT"] = rs_df[ticker]

    index_df = index_df.copy()
    index_df["SMA50"] = index_df["Close"].rolling(50).mean()
    index_df["SMA200"] = index_df["Close"].rolling(200).mean()

    bt_start = index_df.index[-1] - pd.DateOffset(years=years)
    trading_days = index_df.loc[index_df.index >= bt_start].index

    ever_s2 = [
        t for t, df in prepared.items()
        if (df.loc[df.index >= bt_start, "STAGE2"] & (df.loc[df.index >= bt_start, "RS_PCT"] >= cfg.min_rs_percentile)).any()
    ]
    yf_eps_db = build_nongaap_eps_cache(ever_s2)

    cash = initial_cash
    positions = {}
    watchlist = []
    trade_logs = []
    equity_curve = []
    invested_ratios = []

    print(f"\n[이식 후보 하이브리드 v2 백테스트] Swing VCP + Non-GAAP EPS + Portfolio Heat 6% ({years}년)...")

    for i in range(1, len(trading_days)):
        prev_date = trading_days[i - 1]
        curr_date = trading_days[i]

        spy_c = float(index_df.loc[prev_date, "Close"])
        spy_50 = float(index_df.loc[prev_date, "SMA50"])
        spy_200 = float(index_df.loc[prev_date, "SMA200"])

        if spy_c < spy_200:
            cfg.market_regime = "BEAR"
        elif spy_c < spy_50:
            cfg.market_regime = "CAUTION"
        else:
            cfg.market_regime = "BULL"

        prev_holdings_val = sum(
            pos["shares"] * (float(prepared[t].loc[prev_date, "Close"]) if prev_date in prepared[t].index else pos["avg_price"])
            for t, pos in positions.items()
        )
        total_equity = cash + prev_holdings_val

        # [A] 보유 포지션 관리
        for ticker in list(positions.keys()):
            df_t = prepared[ticker]
            if curr_date not in df_t.index or prev_date not in df_t.index:
                continue

            row = df_t.loc[curr_date]
            prev_row = df_t.loc[prev_date]
            pos = positions[ticker]
            pos["days_held"] += 1

            o_px, h_px, l_px, c_px, vol = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), float(row["Volume"])
            vol_sma50_prev = float(prev_row["VOL_SMA50"])
            sma20 = float(row["SMA20"])
            sma50 = float(row["SMA50"])

            if l_px <= pos["stop_price"]:
                exit_px = min(o_px, pos["stop_price"]) * (1 - slippage_pct)
                cash += exit_px * pos["shares"]
                pnl_usd = (exit_px - pos["avg_price"]) * pos["shares"]
                pnl_pct = (exit_px - pos["avg_price"]) / pos["avg_price"] * 100
                reason = "TRAILING_STOP_WIN" if pnl_usd > 0 else "HARD_STOP_LOSS"
                trade_logs.append({
                    "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": curr_date,
                    "entry_price": round(pos["avg_price"], 2), "exit_price": round(exit_px, 2),
                    "shares": pos["shares"], "pnl_usd": round(pnl_usd, 2),
                    "pnl_pct": round(pnl_pct, 2), "reason": reason,
                })
                if pnl_usd <= 0:
                    risk_mgr.record_stop_out(ticker, curr_date.to_pydatetime())
                del positions[ticker]
                continue

            if pos["tranche"] == 1 and c_px >= pos["avg_price"] * 1.02 and vol >= vol_sma50_prev * 1.3:
                add_px = c_px * (1 + slippage_pct)
                add_qty = risk_mgr.get_confirmation_shares(total_equity, pos["shares"], add_px, pos["stop_price"])
                if add_qty > 0 and cash >= add_px * add_qty:
                    cash -= add_px * add_qty
                    total_cost = pos["avg_price"] * pos["shares"] + add_px * add_qty
                    pos["shares"] = round(pos["shares"] + add_qty, 4)
                    pos["avg_price"] = total_cost / pos["shares"]
                    pos["tranche"] = 2
                    pos["stop_price"] = max(pos["stop_price"], pos["pivot_price"] * 0.985)

            gain_pct = (c_px - pos["avg_price"]) / pos["avg_price"]
            prev_high5 = float(prev_row["HIGH_5D"])
            if (
                cfg.market_regime == "BULL"
                and pos["tranche"] == 2
                and gain_pct >= 0.10
                and c_px > sma20
                and c_px > prev_high5
                and vol >= vol_sma50_prev * 1.2
            ):
                lead_px = c_px * (1 + slippage_pct)
                lead_qty = risk_mgr.get_leader_add_shares(total_equity, pos["shares"], lead_px)
                if lead_qty > 0 and cash >= lead_px * lead_qty:
                    cash -= lead_px * lead_qty
                    total_cost = pos["avg_price"] * pos["shares"] + lead_px * lead_qty
                    pos["shares"] = round(pos["shares"] + lead_qty, 4)
                    pos["avg_price"] = total_cost / pos["shares"]
                    pos["tranche"] = 3
                    pos["stop_price"] = max(pos["stop_price"], pos["avg_price"])

            gain_pct = (c_px - pos["avg_price"]) / pos["avg_price"]
            cond_sma20_heavy = (c_px < sma20) and (vol > vol_sma50_prev) and (gain_pct < 0.03)
            cond_failed_bo = (c_px < pos["pivot_price"] * 0.985) and (c_px < o_px) and (vol > vol_sma50_prev * 1.3)

            recent3 = df_t.loc[:curr_date].tail(3)
            cond_3lows = (
                len(recent3) == 3
                and recent3["Low"].iloc[0] > recent3["Low"].iloc[1] > recent3["Low"].iloc[2]
                and recent3["Volume"].mean() > vol_sma50_prev
                and c_px < sma20
                and gain_pct < 0.03
            )

            if cond_sma20_heavy or cond_failed_bo or cond_3lows:
                exit_px = c_px * (1 - slippage_pct)
                cash += exit_px * pos["shares"]
                pnl_usd = (exit_px - pos["avg_price"]) * pos["shares"]
                pnl_pct = (exit_px - pos["avg_price"]) / pos["avg_price"] * 100
                reason = "ANOMALY_SMA20_VOL" if cond_sma20_heavy else "ANOMALY_3LOWS" if cond_3lows else "ANOMALY_FAILED_BO"
                trade_logs.append({
                    "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": curr_date,
                    "entry_price": round(pos["avg_price"], 2), "exit_price": round(exit_px, 2),
                    "shares": pos["shares"], "pnl_usd": round(pnl_usd, 2),
                    "pnl_pct": round(pnl_pct, 2), "reason": reason,
                })
                if pnl_usd <= 0:
                    risk_mgr.record_stop_out(ticker, curr_date.to_pydatetime())
                del positions[ticker]
                continue

            if pos["days_held"] >= cfg.stagnant_days_limit and gain_pct < 0.03:
                exit_px = c_px * (1 - slippage_pct)
                cash += exit_px * pos["shares"]
                pnl_usd = (exit_px - pos["avg_price"]) * pos["shares"]
                pnl_pct = (exit_px - pos["avg_price"]) / pos["avg_price"] * 100
                trade_logs.append({
                    "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": curr_date,
                    "entry_price": round(pos["avg_price"], 2), "exit_price": round(exit_px, 2),
                    "shares": pos["shares"], "pnl_usd": round(pnl_usd, 2),
                    "pnl_pct": round(pnl_pct, 2), "reason": "DEAD_MONEY_ROTATION",
                })
                del positions[ticker]
                continue

            daily_ret = (c_px - float(prev_row["Close"])) / float(prev_row["Close"])
            pos["min_daily_ret"] = min(pos["min_daily_ret"], daily_ret)

            if (c_px < sma50 and vol > vol_sma50_prev) or (daily_ret <= -0.08 and daily_ret <= pos["min_daily_ret"]):
                exit_px = c_px * (1 - slippage_pct)
                cash += exit_px * pos["shares"]
                pnl_pct = (exit_px - pos["avg_price"]) / pos["avg_price"] * 100
                trade_logs.append({
                    "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": curr_date,
                    "entry_price": round(pos["avg_price"], 2), "exit_price": round(exit_px, 2),
                    "shares": pos["shares"], "pnl_usd": round((exit_px - pos["avg_price"]) * pos["shares"], 2),
                    "pnl_pct": round(pnl_pct, 2), "reason": "TP_STAGE3_OVERT_SELL",
                })
                del positions[ticker]
                continue

            if gain_pct >= 0.12 and (c_px < float(prev_row["Close"])) and (vol >= vol_sma50_prev * 2.2):
                sell_qty = round(pos["shares"] * 0.5, 4)
                if sell_qty > 0:
                    exit_px = c_px * (1 - slippage_pct)
                    cash += exit_px * sell_qty
                    pnl_pct = (exit_px - pos["avg_price"]) / pos["avg_price"] * 100
                    trade_logs.append({
                        "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": curr_date,
                        "entry_price": round(pos["avg_price"], 2), "exit_price": round(exit_px, 2),
                        "shares": sell_qty, "pnl_usd": round((exit_px - pos["avg_price"]) * sell_qty, 2),
                        "pnl_pct": round(pnl_pct, 2), "reason": "TP_STAGE2_REVERSAL_TRIM",
                    })
                    pos["shares"] = round(pos["shares"] - sell_qty, 4)
                    pos["stop_price"] = max(pos["stop_price"], pos["avg_price"] * 1.02)

            if not pos["partial_sold"] and gain_pct >= cfg.stage1_trim_gain_pct:
                sell_qty = round(pos["shares"] / 3.0, 4)
                if sell_qty > 0:
                    exit_px = c_px * (1 - slippage_pct)
                    cash += exit_px * sell_qty
                    pnl_pct = (exit_px - pos["avg_price"]) / pos["avg_price"] * 100
                    trade_logs.append({
                        "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": curr_date,
                        "entry_price": round(pos["avg_price"], 2), "exit_price": round(exit_px, 2),
                        "shares": sell_qty, "pnl_usd": round((exit_px - pos["avg_price"]) * sell_qty, 2),
                        "pnl_pct": round(pnl_pct, 2), "reason": "TP_STAGE1_STRENGTH_TRIM",
                    })
                    pos["shares"] = round(pos["shares"] - sell_qty, 4)
                    pos["partial_sold"] = True
                    pos["stop_price"] = max(pos["stop_price"], pos["avg_price"] * 1.03)

        # [B] 피벗 돌파 진입 + Portfolio Heat 6% 통제
        for sig in watchlist:
            ticker = sig["ticker"]
            if ticker in positions or ticker not in prepared:
                continue
            df_t = prepared[ticker]
            if curr_date not in df_t.index or prev_date not in df_t.index:
                continue
            if not risk_mgr.can_enter(ticker, curr_date.to_pydatetime(), len(positions)):
                continue

            row = df_t.loc[curr_date]
            o_px, h_px, l_px, c_px, vol = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), float(row["Volume"])
            vol_sma50_prev = float(df_t.loc[prev_date, "VOL_SMA50"])
            pivot = sig["pivot_price"]
            max_chase = pivot * (1 + cfg.max_chase_pct)
            close_strength = (c_px - l_px) / max(h_px - l_px, 0.01)

            if (
                o_px <= max_chase
                and c_px > pivot
                and vol >= vol_sma50_prev * cfg.min_breakout_vol_ratio
                and close_strength >= 0.55
            ):
                entry_px = max(o_px, (pivot + c_px) / 2.0) * (1 + slippage_pct)
                stop_px = sig["stop_price"]
                trial_qty = risk_mgr.get_trial_shares(total_equity, entry_px, stop_px)

                # Portfolio Heat 6% 체크
                curr_heat = sum(max(p["avg_price"] - p["stop_price"], 0.0) * p["shares"] for p in positions.values()) / total_equity
                new_heat = max(entry_px - stop_px, 0.0) * trial_qty / total_equity
                if curr_heat + new_heat > MAX_PORTFOLIO_HEAT:
                    continue

                if trial_qty > 0 and cash >= entry_px * trial_qty:
                    cash -= entry_px * trial_qty
                    positions[ticker] = {
                        "shares": trial_qty,
                        "avg_price": entry_px,
                        "pivot_price": pivot,
                        "stop_price": stop_px,
                        "entry_date": curr_date,
                        "days_held": 0,
                        "tranche": 1,
                        "partial_sold": False,
                        "min_daily_ret": 0.0,
                    }

        # [C] 내일 감시할 후보 갱신: Swing VCP + Non-GAAP/SEC 하이브리드 실적
        as_of_d = curr_date.date()
        next_watchlist = []
        for ticker in ever_s2:
            df_t = prepared[ticker]
            if curr_date not in df_t.index:
                continue
            r = df_t.loc[curr_date]
            if pd.isna(r["RS_PCT"]) or r["RS_PCT"] < cfg.min_rs_percentile or not bool(r["STAGE2"]):
                continue

            if not check_hybrid_fundamentals(sec_hist.get(ticker), yf_eps_db.get(ticker), as_of_d):
                continue

            is_pp = bool(r["IS_POWER_PLAY"])
            if is_pp:
                pivot_px = round(float(r["PP_PIVOT"]), 2)
                stop_px = round(max(float(r["PP_LOW"]) * 0.99, pivot_px * (1 - cfg.hard_stop_pct)), 2)
                next_watchlist.append({"ticker": ticker, "rs": float(r["RS_PCT"]), "pivot_price": pivot_px, "stop_price": stop_px})
            else:
                sub_hist = df_t.loc[:curr_date].tail(120)
                v = detect_swing_vcp(sub_hist)
                if v:
                    pivot_px = v["pivot"]
                    stop_px = round(max(v["low"] * 0.99, pivot_px * (1 - cfg.hard_stop_pct)), 2)
                    next_watchlist.append({"ticker": ticker, "rs": float(r["RS_PCT"]), "pivot_price": pivot_px, "stop_price": stop_px})

        next_watchlist.sort(key=lambda x: x["rs"], reverse=True)
        watchlist = next_watchlist

        curr_holdings_val = sum(
            pos["shares"] * (float(prepared[t].loc[curr_date, "Close"]) if curr_date in prepared[t].index else pos["avg_price"])
            for t, pos in positions.items()
        )
        tot = cash + curr_holdings_val
        equity_curve.append({"date": curr_date, "equity": tot})
        invested_ratios.append(curr_holdings_val / tot * 100)

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trade_logs)

    final_equity = float(eq_df["equity"].iloc[-1])
    total_ret_pct = (final_equity - initial_cash) / initial_cash * 100
    cagr_pct = ((final_equity / initial_cash) ** (1 / years) - 1) * 100
    spy_ret_pct = (float(index_df.loc[trading_days[-1], "Close"]) - float(index_df.loc[trading_days[0], "Close"])) / float(index_df.loc[trading_days[0], "Close"]) * 100
    rolling_max = eq_df["equity"].cummax()
    mdd_pct = float(((eq_df["equity"] - rolling_max) / rolling_max).min() * 100)
    avg_invested = float(np.mean(invested_ratios))

    print("\n" + "=" * 68)
    print(f" 하이브리드 v2 백테스트 결과 ({years}년 | {time.time() - t0:.1f}초)")
    print("=" * 68)
    print(f" 초기 자본금       : ${initial_cash:,.2f}")
    print(f" 최종 평가자산     : ${final_equity:,.2f}")
    print(f" 총 수익률         : {total_ret_pct:+.2f}%  (동기간 SPY 수익률: {spy_ret_pct:+.2f}%)")
    print(f" 연평균 수익률     : {cagr_pct:+.2f}% (CAGR)")
    print(f" 최대 낙폭 (MDD)   : {mdd_pct:.2f}%  |  평균 주식 투입비중: {avg_invested:.1f}%")

    if not trades_df.empty:
        wins = trades_df[trades_df["pnl_usd"] > 0]
        losses = trades_df[trades_df["pnl_usd"] <= 0]
        win_rate = len(wins) / len(trades_df) * 100
        avg_win = wins["pnl_pct"].mean() if not wins.empty else 0.0
        avg_loss = losses["pnl_pct"].mean() if not losses.empty else 0.0
        rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        profit_factor = abs(wins["pnl_usd"].sum() / losses["pnl_usd"].sum()) if not losses.empty and losses["pnl_usd"].sum() != 0 else float("inf")

        print("-" * 68)
        print(f" 총 청산 거래수    : {len(trades_df)}회 (익절 {len(wins)}회 / 손절 {len(losses)}회)")
        print(f" 승률 (Win Rate)   : {win_rate:.1f}%")
        print(f" 평균 수익률       : +{avg_win:.2f}%  |  평균 손실률: {avg_loss:.2f}%")
        print(f" 손익비 (R:R)      : {rr_ratio:.2f} : 1  |  수익지수(PF): {profit_factor:.2f}")
        print("-" * 68)
        print("[청산 사유별 분포]")
        print(trades_df["reason"].value_counts().to_string())

    # 자동 승격 판정 기준: 기존 4대원칙(+7.26%, PF 1.11) 및 기술적 기준(+25.83%, PF 1.35)과 비교
    is_superior = (total_ret_pct >= 25.83) and (profit_factor >= 1.35) and (mdd_pct >= -17.39)
    print("\n" + "=" * 68)
    if is_superior:
        print(" [판정: 우월함] 기존 최고 기록(+25.83%, PF 1.35, MDD -17.39%)을 상회했습니다 -> 실전 코드 이식 진행!")
    else:
        print(" [판정: 기존 로직 유지] 기존 최고 기준을 넘지 못했습니다 -> 실전 코드는 현재 상태 그대로 유지합니다.")
    print("=" * 68)


if __name__ == "__main__":
    run_hybrid_backtest(years=2, initial_cash=1_350.0)