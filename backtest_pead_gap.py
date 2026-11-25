import os
import pickle
import time
import numpy as np
import pandas as pd

from config import TradingConfig
from risk_manager import RiskManager

CACHE_DIR = "data_cache"

# 1순위 정량 대용 지표 파라미터: 최근 60거래일(1분기) 내 기관 매집 어닝 갭/급등 흔적
GAP_LOOKBACK_DAYS = 60
MIN_OPEN_GAP_PCT = 0.08     # 전일 종가 대비 시가 갭 +8.0% 이상
MIN_DAY_SURGE_PCT = 0.10    # 또는 당일 종가 급등 +10.0% 이상 (장중 어닝 반응 포함)
MIN_VOL_SURGE_RATIO = 3.0   # 거래량 50일 평균 대비 3.0배 이상 폭발


def run_pead_gap_backtest(years: int = 2, initial_cash: float = 1_350.0, slippage_pct: float = 0.001):
    t0 = time.time()
    cfg = TradingConfig()
    risk_mgr = RiskManager(cfg)

    cache_files = sorted([f for f in os.listdir(CACHE_DIR) if f.startswith("us_clean_bt_") and f.endswith(f"_{years}y.pkl")])
    if not cache_files:
        raise FileNotFoundError("data_cache 폴더에 us_clean_bt_*_2y.pkl 캐시 파일이 없습니다.")

    with open(os.path.join(CACHE_DIR, cache_files[-1]), "rb") as f:
        raw_universe, index_df = pickle.load(f)

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
        df["DAILY_RANGE_PCT"] = (df["High"] - df["Low"]) / df["Low"]

        prev_close = df["Close"].shift(1)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["ATR20_PCT"] = tr.rolling(20).mean() / df["Close"]

        # [핵심] 정량 대용 지표: 최근 60거래일 내 대량거래(3배+) 어닝 갭상승(+8% 갭 또는 +10% 급등)
        open_gap = (df["Open"] - prev_close) / prev_close
        close_surge = (df["Close"] - prev_close) / prev_close
        vol_ratio = df["Volume"] / df["VOL_SMA50"].shift(1)
        pead_event = ((open_gap >= MIN_OPEN_GAP_PCT) | (close_surge >= MIN_DAY_SURGE_PCT)) & (vol_ratio >= MIN_VOL_SURGE_RATIO)
        df["HAS_PEAD_CATALYST"] = pead_event.rolling(GAP_LOOKBACK_DAYS, min_periods=1).max() > 0

        df["R50"] = (df["High"].rolling(50).max() - df["Low"].rolling(50).min()) / df["High"].rolling(50).max()
        df["R25"] = (df["High"].rolling(25).max() - df["Low"].rolling(25).min()) / df["High"].rolling(25).max()
        df["R10"] = (df["High"].rolling(10).max() - df["Low"].rolling(10).min()) / df["High"].rolling(10).max()
        df["R5"] = (df["High"].rolling(5).max() - df["Low"].rolling(5).min()) / df["High"].rolling(5).max()

        df["PIVOT_HIGH"] = df["High"].rolling(cfg.pivot_lookback_days).max()
        df["PIVOT_LOW"] = df["Low"].rolling(cfg.pivot_lookback_days).min()
        df["DRYUP_CNT"] = (df["Volume"] <= df["VOL_SMA50"] * 0.75).rolling(cfg.pivot_lookback_days).sum()
        df["TIGHT_CNT"] = (df["DAILY_RANGE_PCT"] <= 0.065).rolling(cfg.pivot_lookback_days).sum()
        df["VOL3_AVG"] = df["Volume"].rolling(3).mean()

        surge_hi = df["High"].shift(12).rolling(40).max()
        surge_lo = df["Low"].shift(12).rolling(40).min()
        base_hi = df["High"].rolling(12).max()
        base_lo = df["Low"].rolling(12).min()
        peak = np.maximum(surge_hi, base_hi)
        df["IS_POWER_PLAY"] = (
            ((surge_hi - surge_lo) / surge_lo >= 0.90)
            & ((peak - base_lo) / peak <= 0.18)
            & (df["R5"] <= 0.10)
        )
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

    cash = initial_cash
    positions = {}
    watchlist = []
    trade_logs = []
    equity_curve = []
    invested_ratios = []

    print(f"\n[정량 대용 지표(어닝 갭상승 +8% & 거래량 3배) 백테스트] 초기자본 ${initial_cash:,.2f} ({years}년)...")

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

        # [A] 보유 포지션 관리 (기존 1번 엔진과 100% 동일)
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

        # [B] 피벗 돌파 진입
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

                    if vol >= vol_sma50_prev * 2.0:
                        add_px = c_px * (1 + slippage_pct)
                        add_qty = risk_mgr.get_confirmation_shares(total_equity, trial_qty, add_px, stop_px)
                        if add_qty > 0 and cash >= add_px * add_qty:
                            cash -= add_px * add_qty
                            total_cost = entry_px * trial_qty + add_px * add_qty
                            positions[ticker]["shares"] = round(positions[ticker]["shares"] + add_qty, 4)
                            positions[ticker]["avg_price"] = total_cost / positions[ticker]["shares"]
                            positions[ticker]["tranche"] = 2

        # [C] 내일 감시 후보: Stage2 + RS 92+ + VCP/PowerPlay + PEAD 어닝 갭/거래량 폭발 이력
        next_watchlist = []
        for ticker, df_t in prepared.items():
            if curr_date not in df_t.index:
                continue
            r = df_t.loc[curr_date]
            if pd.isna(r["RS_PCT"]) or r["RS_PCT"] < cfg.min_rs_percentile or not bool(r["STAGE2"]):
                continue
            if not bool(r["HAS_PEAD_CATALYST"]):
                continue

            dynamic_r5_cap = min(0.13, max(0.095, float(r["ATR20_PCT"]) * 3.0))
            is_vcp = (
                r["R10"] <= r["R50"] * 0.58
                and r["R5"] <= r["R25"] * 0.75
                and 0.012 <= r["R5"] <= dynamic_r5_cap
                and (r["DRYUP_CNT"] >= 1 or r["VOL3_AVG"] <= r["VOL_SMA50"] * 0.85)
                and r["TIGHT_CNT"] >= 2
            )
            is_pp = bool(r["IS_POWER_PLAY"])

            if is_pp or is_vcp:
                pivot_px = round(float(r["PP_PIVOT"] if is_pp else r["PIVOT_HIGH"]), 2)
                pivot_lo = float(r["PP_LOW"] if is_pp else r["PIVOT_LOW"])
                stop_px = round(max(pivot_lo * 0.99, pivot_px * (1 - cfg.hard_stop_pct)), 2)
                next_watchlist.append({
                    "ticker": ticker,
                    "rs": float(r["RS_PCT"]),
                    "pivot_price": pivot_px,
                    "stop_price": stop_px,
                })

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
    print(f" PEAD 어닝 갭상승(+8% & 거래량 3배) 백테스트 결과 ({years}년 | {time.time() - t0:.1f}초)")
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

    is_superior = (total_ret_pct > 25.83) and (profit_factor >= 1.35)
    print("\n" + "=" * 68)
    if is_superior:
        print(" [판정: 우월함] 기존 최고 기록(+25.83%, PF 1.35)을 상회했습니다 -> 실전 코드에 1순위(PEAD 갭) + 2순위(컨센서스·LLM) 이식 진행!")
    else:
        print(" [판정: 기존 로직 유지] 기존 최고 기록(+25.83%, PF 1.35)을 넘지 못했습니다 -> 실전 진입 필터는 현행 유지하고 LLM 브리핑만 보조로 연결합니다.")
    print("=" * 68)


if __name__ == "__main__":
    run_pead_gap_backtest(years=2, initial_cash=1_350.0)