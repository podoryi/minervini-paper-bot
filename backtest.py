import os
import pickle
import time
from datetime import datetime
import numpy as np
import pandas as pd

from config import TradingConfig
from risk_manager import RiskManager

CACHE_DIR = "data_cache"


def load_filtered_cache(cfg: TradingConfig, years: int = 2):
    """기존 로컬 캐시에서 기관 유동성($50M/일, $15+) 통과 종목만 즉시 필터링"""
    today_str = datetime.now().strftime("%Y%m%d")
    cache_path = os.path.join(CACHE_DIR, f"us_clean_bt_{today_str}_{years}y.pkl")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"캐시 파일을 찾을 수 없습니다: {cache_path}")

    with open(cache_path, "rb") as f:
        raw_universe, index_df = pickle.load(f)

    filtered_universe = {}
    for ticker, df in raw_universe.items():
        avg_dollar_vol = float((df["Close"] * df["Volume"]).tail(100).mean())
        if float(df["Close"].iloc[-1]) >= cfg.min_price and avg_dollar_vol >= cfg.min_dollar_volume:
            filtered_universe[ticker] = df

    print(
        f"캐시 필터링 완료: 전체 {len(raw_universe)}개 -> 기관 유동성($50M/일+) {len(filtered_universe)}개 "
        f"(TGTX 포함: {'TGTX' in filtered_universe})"
    )
    return filtered_universe, index_df


def prepare_indicators(universe_data: dict, index_df: pd.DataFrame, cfg: TradingConfig):
    prepared = {}
    rs_frames = {}

    idx_ret63 = index_df["Close"].pct_change(63)
    idx_ret126 = index_df["Close"].pct_change(126)
    idx_ret252 = index_df["Close"].pct_change(252)

    for ticker, raw_df in universe_data.items():
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
    return prepared, index_df


def run_backtest(years: int = 2, initial_cash: float = 100_000.0, slippage_pct: float = 0.001):
    t0 = time.time()
    cfg = TradingConfig()
    risk_mgr = RiskManager(cfg)

    universe_data, index_df = load_filtered_cache(cfg, years=years)
    prepared, index_df = prepare_indicators(universe_data, index_df, cfg)

    bt_start = index_df.index[-1] - pd.DateOffset(years=years)
    trading_days = index_df.loc[index_df.index >= bt_start].index

    cash = initial_cash
    positions = {}
    watchlist = []
    trade_logs = []
    equity_curve = []
    invested_ratios = []

    print(f"\n기관 주도주 + 순환매 백테스트 실행 ({trading_days[0].strftime('%Y-%m-%d')} ~ {trading_days[-1].strftime('%Y-%m-%d')})...")

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

        prev_holdings_val = 0.0
        for t, pos in positions.items():
            df_t = prepared[t]
            p_close = float(df_t.loc[prev_date, "Close"]) if prev_date in df_t.index else pos["avg_price"]
            prev_holdings_val += pos["shares"] * p_close
        total_equity = cash + prev_holdings_val

        # ================= [A] 보유 포지션 관리 =================
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

            # 1) 장중 하드 스톱 / 트레일링 스톱
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

            # 2) 피라미딩 2단계 (기본 비중 완성)
            if pos["tranche"] == 1 and c_px >= pos["avg_price"] * 1.02 and vol >= vol_sma50_prev * 1.3:
                add_px = c_px * (1 + slippage_pct)
                add_qty = risk_mgr.get_confirmation_shares(total_equity, pos["shares"], add_px, pos["stop_price"])
                if add_qty > 0 and cash >= add_px * add_qty:
                    cash -= add_px * add_qty
                    total_cost = pos["avg_price"] * pos["shares"] + add_px * add_qty
                    pos["shares"] += add_qty
                    pos["avg_price"] = total_cost / pos["shares"]
                    pos["tranche"] = 2
                    pos["stop_price"] = max(pos["stop_price"], pos["pivot_price"] * 0.985)

            # 3) 피라미딩 3단계 (강세장 전용 주도주 24% 집중)
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
                    pos["shares"] += lead_qty
                    pos["avg_price"] = total_cost / pos["shares"]
                    pos["tranche"] = 3
                    pos["stop_price"] = max(pos["stop_price"], pos["avg_price"])

            # 4) 종가 기준 3대 이상징후 손절
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

            # 5) 정체 종목(Dead Money) 순환매: 12거래일 경과 후 +3% 미달 시 자본 회수
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

            # 6) 3단계 분할 익절
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

            if gain_pct >= 0.12 and (c_px < float(prev_row["Close"])) and (vol >= vol_sma50_prev * 2.2) and pos["shares"] >= 2:
                sell_qty = pos["shares"] // 2
                exit_px = c_px * (1 - slippage_pct)
                cash += exit_px * sell_qty
                pnl_pct = (exit_px - pos["avg_price"]) / pos["avg_price"] * 100
                trade_logs.append({
                    "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": curr_date,
                    "entry_price": round(pos["avg_price"], 2), "exit_price": round(exit_px, 2),
                    "shares": sell_qty, "pnl_usd": round((exit_px - pos["avg_price"]) * sell_qty, 2),
                    "pnl_pct": round(pnl_pct, 2), "reason": "TP_STAGE2_REVERSAL_TRIM",
                })
                pos["shares"] -= sell_qty
                pos["stop_price"] = max(pos["stop_price"], pos["avg_price"] * 1.02)

            if not pos["partial_sold"] and gain_pct >= cfg.stage1_trim_gain_pct and pos["shares"] >= 3:
                sell_qty = pos["shares"] // 3
                exit_px = c_px * (1 - slippage_pct)
                cash += exit_px * sell_qty
                pnl_pct = (exit_px - pos["avg_price"]) / pos["avg_price"] * 100
                trade_logs.append({
                    "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": curr_date,
                    "entry_price": round(pos["avg_price"], 2), "exit_price": round(exit_px, 2),
                    "shares": sell_qty, "pnl_usd": round((exit_px - pos["avg_price"]) * sell_qty, 2),
                    "pnl_pct": round(pnl_pct, 2), "reason": "TP_STAGE1_STRENGTH_TRIM",
                })
                pos["shares"] -= sell_qty
                pos["partial_sold"] = True
                pos["stop_price"] = max(pos["stop_price"], pos["avg_price"] * 1.03)

        # ================= [B] 피벗 돌파 진입 =================
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

            candle_range = max(h_px - l_px, 0.01)
            close_strength = (c_px - l_px) / candle_range

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
                            positions[ticker]["shares"] += add_qty
                            positions[ticker]["avg_price"] = total_cost / positions[ticker]["shares"]
                            positions[ticker]["tranche"] = 2

        # ================= [C] 내일 감시할 정밀 VCP + 파워플레이 후보 갱신 =================
        next_watchlist = []
        for ticker, df_t in prepared.items():
            if curr_date not in df_t.index:
                continue
            r = df_t.loc[curr_date]
            if pd.isna(r["RS_PCT"]) or r["RS_PCT"] < cfg.min_rs_percentile:
                continue
            if not bool(r["STAGE2"]):
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

        # ================= [D] 일별 계좌 평가액 및 주식 투입 비중 기록 =================
        curr_holdings_val = 0.0
        for t, pos in positions.items():
            df_t = prepared[t]
            c_val = float(df_t.loc[curr_date, "Close"]) if curr_date in df_t.index else pos["avg_price"]
            curr_holdings_val += pos["shares"] * c_val
        tot = cash + curr_holdings_val
        equity_curve.append({"date": curr_date, "equity": tot})
        invested_ratios.append(curr_holdings_val / tot * 100)

    # ================= [E] 백테스트 성과 리포트 출력 =================
    eq_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trade_logs)

    final_equity = float(eq_df["equity"].iloc[-1])
    total_ret_pct = (final_equity - initial_cash) / initial_cash * 100
    cagr_pct = ((final_equity / initial_cash) ** (1 / years) - 1) * 100

    spy_start = float(index_df.loc[trading_days[0], "Close"])
    spy_end = float(index_df.loc[trading_days[-1], "Close"])
    spy_ret_pct = (spy_end - spy_start) / spy_start * 100

    rolling_max = eq_df["equity"].cummax()
    mdd_pct = float(((eq_df["equity"] - rolling_max) / rolling_max).min() * 100)
    avg_invested = float(np.mean(invested_ratios))

    print("\n" + "=" * 68)
    print(f" 마크 미너비니 전략 기관 주도주 백테스트 결과 ({years}년 | 소요: {time.time() - t0:.1f}초)")
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

    if positions:
        print("-" * 68)
        print("[백테스트 종료일(2026-11-24) 현재 보유 중 포지션]")
        last_dt = trading_days[-1]
        for t, pos in positions.items():
            curr_c = float(prepared[t].loc[last_dt, "Close"])
            unreal_pct = (curr_c - pos["avg_price"]) / pos["avg_price"] * 100
            print(
                f" -> {t} | 진입일: {pos['entry_date'].strftime('%Y-%m-%d')} | "
                f"평단: ${pos['avg_price']:.2f} | 현재가: ${curr_c:.2f} ({unreal_pct:+.2f}%) | 손절가: ${pos['stop_price']:.2f}"
            )

    if not trades_df.empty:
        trades_df.to_csv("backtest_trades.csv", index=False, encoding="utf-8-sig")
        eq_df.to_csv("backtest_equity.csv", encoding="utf-8-sig")
    print("=" * 68)


if __name__ == "__main__":
    run_backtest(years=2, initial_cash=100_000.0)