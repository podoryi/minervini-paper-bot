import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict
from config import TradingConfig
from fundamental_analyzer import analyze_fundamentals_and_catalyst


@dataclass
class ScreenerSignal:
    ticker: str
    name: str
    exchange: str
    pattern_type: str
    close: float
    pivot_price: float
    suggested_stop: float
    rs_score: float
    contractions: List[float]
    vol_dryup_days: int
    sector: str = "Unknown"
    fund_grade: str = "N/A"
    priority_score: int = 0
    fund_summary: str = ""
    catalyst_headline: str = ""
    llm_note: str = ""
    has_catalyst: bool = False


class MinerviniScreener:
    def __init__(self, config: TradingConfig):
        self.cfg = config

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA150"] = df["Close"].rolling(150).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["VOL_SMA50"] = df["Volume"].rolling(50).mean()
        df["HIGH_52W"] = df["High"].rolling(252, min_periods=200).max()
        df["LOW_52W"] = df["Low"].rolling(252, min_periods=200).min()

        prev_close = df["Close"].shift(1)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["ATR20_PCT"] = tr.rolling(20).mean() / df["Close"]
        df["DAILY_RANGE_PCT"] = (df["High"] - df["Low"]) / df["Low"]
        return df

    def compute_relative_strength(self, stock_df: pd.DataFrame, index_df: pd.DataFrame) -> float:
        periods = [(63, 0.4), (126, 0.3), (252, 0.3)]
        rs_total = 0.0
        for p, weight in periods:
            if len(stock_df) > p and len(index_df) > p:
                s_ret = float(stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-p] - 1.0)
                i_ret = float(index_df["Close"].iloc[-1] / index_df["Close"].iloc[-p] - 1.0)
                rs_total += (s_ret - i_ret) * weight
        return rs_total

    def check_stage2_template(self, df: pd.DataFrame) -> bool:
        if len(df) < 225:
            return False

        curr = df.iloc[-1]
        sma200_prev = df["SMA200"].iloc[-self.cfg.sma200_uptrend_days]
        if pd.isna(curr["SMA200"]) or pd.isna(sma200_prev):
            return False

        cond_ma_align = curr["Close"] > curr["SMA50"] > curr["SMA150"] > curr["SMA200"]
        cond_sma200_up = curr["SMA200"] > sma200_prev
        cond_above_low = curr["Close"] >= curr["LOW_52W"] * (1 + self.cfg.min_above_52w_low)
        cond_near_high = curr["Close"] >= curr["HIGH_52W"] * (1 - self.cfg.max_below_52w_high)

        return all([cond_ma_align, cond_sma200_up, cond_above_low, cond_near_high])

    def detect_vcp_and_pivot(self, df: pd.DataFrame) -> Optional[dict]:
        def range_pct(sub_df: pd.DataFrame) -> float:
            hi = float(sub_df["High"].max())
            lo = float(sub_df["Low"].min())
            return (hi - lo) / hi if hi > 0 else 1.0

        r50 = range_pct(df.iloc[-50:])
        r25 = range_pct(df.iloc[-25:])
        r10 = range_pct(df.iloc[-10:])
        r5 = range_pct(df.iloc[-5:])

        atr_pct = float(df["ATR20_PCT"].iloc[-1])
        dynamic_r5_cap = min(0.13, max(0.095, atr_pct * 3.0))

        is_contracting = (
            r10 <= r50 * 0.58
            and r5 <= r25 * 0.75
            and 0.012 <= r5 <= dynamic_r5_cap
        )
        if not is_contracting:
            return None

        recent_pivot = df.iloc[-self.cfg.pivot_lookback_days:]
        vol_sma50_now = float(df["VOL_SMA50"].iloc[-1])

        dryup_days = int((recent_pivot["Volume"] <= recent_pivot["VOL_SMA50"] * 0.75).sum())
        vol3_contracted = float(df["Volume"].iloc[-3:].mean()) <= vol_sma50_now * 0.85
        tight_days = int((recent_pivot["DAILY_RANGE_PCT"] <= 0.065).sum())

        if (dryup_days >= 1 or vol3_contracted) and tight_days >= 2:
            pivot_price = round(float(recent_pivot["High"].max()), 2)
            pivot_low = float(recent_pivot["Low"].min())
            tech_stop = round(max(pivot_low * 0.99, pivot_price * (1 - self.cfg.hard_stop_pct)), 2)
            return {
                "pattern_type": "VCP_PIVOT",
                "pivot_price": pivot_price,
                "suggested_stop": tech_stop,
                "contractions": [round(r50 * 100, 1), round(r25 * 100, 1), round(r10 * 100, 1), round(r5 * 100, 1)],
                "vol_dryup_days": dryup_days,
            }
        return None

    def detect_power_play(self, df: pd.DataFrame) -> Optional[dict]:
        if len(df) < 65:
            return None

        recent_base = df.iloc[-12:]
        prior_surge = df.iloc[-52:-12]

        surge_low = float(prior_surge["Low"].min())
        surge_high = float(prior_surge["High"].max())
        if surge_low <= 0 or (surge_high - surge_low) / surge_low < 0.90:
            return None

        base_high = float(recent_base["High"].max())
        base_low = float(recent_base["Low"].min())
        peak = max(surge_high, base_high)
        pullback_from_peak = (peak - base_low) / peak

        if pullback_from_peak <= 0.18:
            pivot_price = round(base_high, 2)
            tech_stop = round(max(base_low * 0.99, pivot_price * (1 - self.cfg.hard_stop_pct)), 2)
            return {
                "pattern_type": "POWER_PLAY",
                "pivot_price": pivot_price,
                "suggested_stop": tech_stop,
                "contractions": [round(pullback_from_peak * 100, 1)],
                "vol_dryup_days": int((recent_base["Volume"] < recent_base["VOL_SMA50"]).sum()),
            }
        return None

    def scan_universe(
        self,
        universe_data: Dict[str, pd.DataFrame],
        index_df: pd.DataFrame,
        meta_map: Dict[str, dict] = None,
    ) -> List[ScreenerSignal]:
        meta_map = meta_map or {}
        rs_map = {}
        processed_dfs = {}

        for ticker, raw_df in universe_data.items():
            df = self.add_indicators(raw_df)
            processed_dfs[ticker] = df
            rs_map[ticker] = self.compute_relative_strength(df, index_df)

        if not rs_map:
            return []

        rs_series = pd.Series(rs_map)
        rs_percentiles = rs_series.rank(pct=True) * 100.0

        rs_passed = []
        stage2_passed = []
        tech_candidates = []

        for ticker, df in processed_dfs.items():
            if rs_percentiles[ticker] < self.cfg.min_rs_percentile:
                continue
            rs_passed.append(ticker)

            if not self.check_stage2_template(df):
                continue
            stage2_passed.append((ticker, round(float(rs_percentiles[ticker]), 1)))

            pattern = self.detect_power_play(df) or self.detect_vcp_and_pivot(df)
            if pattern:
                tech_candidates.append((ticker, df, pattern, round(float(rs_percentiles[ticker]), 1)))

        print(f"[실적·정량대용지표·어닝일정·재료 검증] 기술적 VCP 통과 {len(tech_candidates)}개 종목 심층 분석 중...")
        signals: List[ScreenerSignal] = []

        for ticker, df, pattern, rs_val in tech_candidates:
            rep = analyze_fundamentals_and_catalyst(ticker, df)
            if not rep.passed:
                reason_txt = rep.llm_note if rep.llm_note else rep.summary
                print(f"   (제외) {ticker}: {reason_txt}")
                continue

            info = meta_map.get(ticker, {})
            signals.append(
                ScreenerSignal(
                    ticker=ticker,
                    name=info.get("name", ticker),
                    exchange=info.get("exchange", "NASD"),
                    pattern_type=pattern["pattern_type"],
                    close=round(float(df["Close"].iloc[-1]), 2),
                    pivot_price=pattern["pivot_price"],
                    suggested_stop=pattern["suggested_stop"],
                    rs_score=rs_val,
                    contractions=pattern["contractions"],
                    vol_dryup_days=pattern["vol_dryup_days"],
                    sector=rep.sector,
                    fund_grade=rep.grade,
                    priority_score=rep.priority_score,
                    fund_summary=rep.summary,
                    catalyst_headline=rep.headline,
                    llm_note=rep.llm_note,
                    has_catalyst=rep.has_catalyst,
                )
            )

        print(
            f"[필터링 요약] 전체 {len(processed_dfs)}개 -> "
            f"RS상위({self.cfg.min_rs_percentile:.0f}+) {len(rs_passed)}개 -> "
            f"2단계 {len(stage2_passed)}개 -> "
            f"기술적VCP {len(tech_candidates)}개 -> "
            f"최종 통과 {len(signals)}개"
        )

        signals.sort(key=lambda x: (-x.priority_score, -x.rs_score))
        return signals