from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import pandas as pd

from config import TradingConfig
from screener import ScreenerSignal
from risk_manager import RiskManager


@dataclass
class Position:
    ticker: str
    shares: float
    avg_price: float
    pivot_price: float
    stop_price: float
    entry_date: datetime
    days_held: int = 0
    tranche: int = 1
    highest_price: float = 0.0
    partial_sold: bool = False
    last_counted_date: str = ""


class ExecutionEngine:
    def __init__(self, config: TradingConfig, risk_mgr: RiskManager, broker):
        self.cfg = config
        self.risk_mgr = risk_mgr
        self.broker = broker
        self.watchlist: Dict[str, ScreenerSignal] = {}
        self.positions: Dict[str, Position] = {}

    def load_watchlist(self, signals: List[ScreenerSignal]):
        self.watchlist = {s.ticker: s for s in signals if s.ticker not in self.positions}

    def on_intraday_tick(
        self,
        ticker: str,
        current_price: float,
        current_vol: float,
        projected_vol: float,
        vol_sma50: float,
        sma20: float,
        prev_high5: float,
        now: datetime,
    ):
        equity = self.broker.get_total_equity()

        # ================= [A] 보유 포지션 장중 관리 =================
        if ticker in self.positions:
            pos = self.positions[ticker]
            pos.highest_price = max(pos.highest_price, current_price)

            if current_price <= pos.stop_price:
                reason = "TRAILING_STOP_WIN" if current_price > pos.avg_price else "HARD_STOP_LOSS"
                self.broker.sell_market(ticker, pos.shares, reason=reason)
                if current_price <= pos.avg_price:
                    self.risk_mgr.record_stop_out(ticker, now)
                del self.positions[ticker]
                return

            if (
                pos.tranche == 1
                and current_price >= pos.avg_price * 1.02
                and (current_vol >= vol_sma50 * 1.3 or projected_vol >= vol_sma50 * 1.35)
            ):
                add_qty = self.risk_mgr.get_confirmation_shares(equity, pos.shares, current_price, pos.stop_price)
                if add_qty > 0:
                    fill_price = self.broker.buy_market(ticker, add_qty, reason="PYRAMID_CONFIRM_100PCT")
                    if fill_price > 0:
                        total_cost = (pos.avg_price * pos.shares) + (fill_price * add_qty)
                        pos.shares = round(pos.shares + add_qty, 4)
                        pos.avg_price = total_cost / pos.shares
                        pos.tranche = 2
                        pos.stop_price = max(pos.stop_price, pos.pivot_price * 0.985)
                return

            gain_pct = (current_price - pos.avg_price) / pos.avg_price
            if (
                self.cfg.market_regime == "BULL"
                and pos.tranche == 2
                and gain_pct >= 0.10
                and current_price > sma20
                and current_price > prev_high5
                and (current_vol >= vol_sma50 * 1.2 or projected_vol >= vol_sma50 * 1.25)
            ):
                lead_qty = self.risk_mgr.get_leader_add_shares(equity, pos.shares, current_price)
                if lead_qty > 0:
                    fill_price = self.broker.buy_market(ticker, lead_qty, reason="PYRAMID_LEADER_24PCT")
                    if fill_price > 0:
                        total_cost = (pos.avg_price * pos.shares) + (fill_price * lead_qty)
                        pos.shares = round(pos.shares + lead_qty, 4)
                        pos.avg_price = total_cost / pos.shares
                        pos.tranche = 3
                        pos.stop_price = max(pos.stop_price, pos.avg_price)
            return

        # ================= [B] 감시 종목 피벗 포인트 돌파 진입 =================
        if ticker in self.watchlist:
            sig = self.watchlist[ticker]
            max_buy_price = sig.pivot_price * (1 + self.cfg.max_chase_pct)

            if (
                sig.pivot_price < current_price <= max_buy_price
                and projected_vol >= vol_sma50 * self.cfg.min_breakout_vol_ratio
            ):
                if self.risk_mgr.can_enter(ticker, now, len(self.positions)):
                    trial_qty = self.risk_mgr.get_trial_shares(equity, current_price, sig.suggested_stop)
                    if trial_qty > 0:
                        fill_price = self.broker.buy_market(ticker, trial_qty, reason=f"PIVOT_TRIAL_{sig.pattern_type}")
                        if fill_price > 0:
                            self.positions[ticker] = Position(
                                ticker=ticker,
                                shares=trial_qty,
                                avg_price=fill_price,
                                pivot_price=sig.pivot_price,
                                stop_price=sig.suggested_stop,
                                entry_date=now,
                                days_held=0,
                                tranche=1,
                                highest_price=fill_price,
                            )
                            del self.watchlist[ticker]

    def on_market_close_check(self, ticker: str, daily_df: pd.DataFrame, now: datetime):
        if ticker not in self.positions or len(daily_df) < 60:
            return

        pos = self.positions[ticker]
        today_str = now.strftime("%Y-%m-%d")
        if pos.last_counted_date != today_str:
            pos.days_held += 1
            pos.last_counted_date = today_str

        curr = daily_df.iloc[-1]
        prev = daily_df.iloc[-2]
        sma20 = float(daily_df["Close"].rolling(20).mean().iloc[-1])
        sma50 = float(daily_df["Close"].rolling(50).mean().iloc[-1])
        vol_sma50 = float(daily_df["Volume"].rolling(50).mean().iloc[-1])

        c_px = float(curr["Close"])
        o_px = float(curr["Open"])
        vol = float(curr["Volume"])
        gain_pct = (c_px - pos.avg_price) / pos.avg_price

        # 1. 3대 이상징후 손절
        cond_sma20_heavy = (c_px < sma20) and (vol > vol_sma50) and (gain_pct < 0.03)
        cond_failed_bo = (c_px < pos.pivot_price * 0.985) and (c_px < o_px) and (vol > vol_sma50 * 1.3)

        recent3 = daily_df.iloc[-3:]
        cond_3lows = (
            recent3["Low"].iloc[0] > recent3["Low"].iloc[1] > recent3["Low"].iloc[2]
            and float(recent3["Volume"].mean()) > vol_sma50
            and c_px < sma20
            and gain_pct < 0.03
        )

        if cond_sma20_heavy or cond_failed_bo or cond_3lows:
            reason = (
                "ANOMALY_SMA20_VOL" if cond_sma20_heavy
                else "ANOMALY_3LOWS" if cond_3lows
                else "ANOMALY_FAILED_BO"
            )
            self.broker.sell_market(ticker, pos.shares, reason=reason)
            if c_px <= pos.avg_price:
                self.risk_mgr.record_stop_out(ticker, now)
            del self.positions[ticker]
            return

        # 2. 12거래일 정체 종목 순환매
        if pos.days_held >= self.cfg.stagnant_days_limit and gain_pct < 0.03:
            self.broker.sell_market(ticker, pos.shares, reason="DEAD_MONEY_ROTATION")
            del self.positions[ticker]
            return

        # 3. 3단계 분할 익절
        daily_drop_pct = (c_px - float(prev["Close"])) / float(prev["Close"])
        since_entry = daily_df.loc[daily_df.index >= pos.entry_date.strftime("%Y-%m-%d"), "Close"].pct_change()
        max_drop_since_entry = float(since_entry.min()) if not since_entry.empty else daily_drop_pct

        if (c_px < sma50 and vol > vol_sma50) or (daily_drop_pct <= -0.08 and daily_drop_pct <= max_drop_since_entry):
            self.broker.sell_market(ticker, pos.shares, reason="TP_STAGE3_OVERT_SELL")
            del self.positions[ticker]
            return

        if gain_pct >= 0.12 and (c_px < float(prev["Close"])) and (vol >= vol_sma50 * 2.2):
            sell_qty = round(pos.shares * 0.5, 4) if self.cfg.use_fractional_shares else float(int(pos.shares // 2))
            if sell_qty > 0:
                self.broker.sell_market(ticker, sell_qty, reason="TP_STAGE2_REVERSAL_TRIM")
                pos.shares = round(pos.shares - sell_qty, 4)
                pos.stop_price = max(pos.stop_price, pos.avg_price * 1.02)
            return

        if not pos.partial_sold and gain_pct >= self.cfg.stage1_trim_gain_pct:
            sell_qty = round(pos.shares / 3.0, 4) if self.cfg.use_fractional_shares else float(int(pos.shares // 3))
            if sell_qty > 0:
                self.broker.sell_market(ticker, sell_qty, reason="TP_STAGE1_STRENGTH_TRIM")
                pos.shares = round(pos.shares - sell_qty, 4)
                pos.partial_sold = True
                pos.stop_price = max(pos.stop_price, pos.avg_price * 1.03)