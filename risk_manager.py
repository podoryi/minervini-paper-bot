from datetime import datetime, date
from typing import Dict
from config import TradingConfig


class RiskManager:
    def __init__(self, config: TradingConfig):
        self.cfg = config
        self.stop_out_dates: Dict[str, date] = {}

    def _format_shares(self, raw_shares: float) -> float:
        if raw_shares <= 0:
            return 0.0
        if self.cfg.use_fractional_shares:
            return round(raw_shares, 4)
        return float(int(raw_shares))

    def record_stop_out(self, ticker: str, dt: datetime):
        self.stop_out_dates[ticker] = dt.date()

    def can_enter(self, ticker: str, dt: datetime, open_positions_count: int) -> bool:
        if open_positions_count >= self.cfg.max_positions:
            return False
        if ticker in self.stop_out_dates:
            elapsed_days = (dt.date() - self.stop_out_dates[ticker]).days
            if elapsed_days < self.cfg.reentry_cooldown_days:
                return False
        return True

    def get_trial_shares(self, total_equity: float, entry_price: float, stop_price: float) -> float:
        if entry_price <= 0 or total_equity <= 0:
            return 0.0

        target_pos_usd = total_equity * self.cfg.initial_position_pct * self.cfg.trial_entry_ratio
        risk_per_share = max(entry_price - stop_price, entry_price * 0.01)
        max_risk_usd = total_equity * self.cfg.max_account_risk_pct * self.cfg.trial_entry_ratio

        raw_shares = min(target_pos_usd / entry_price, max_risk_usd / risk_per_share)
        return self._format_shares(raw_shares)

    def get_confirmation_shares(
        self, total_equity: float, current_shares: float, entry_price: float, stop_price: float
    ) -> float:
        if entry_price <= 0 or total_equity <= 0:
            return 0.0

        full_pos_usd = total_equity * self.cfg.initial_position_pct
        current_usd = current_shares * entry_price
        rem_pos_usd = max(0.0, full_pos_usd - current_usd)

        risk_per_share = max(entry_price - stop_price, entry_price * 0.01)
        total_risk_usd = total_equity * self.cfg.max_account_risk_pct
        rem_risk_usd = max(0.0, total_risk_usd - (current_shares * risk_per_share))

        raw_add = min(rem_pos_usd / entry_price, rem_risk_usd / risk_per_share)
        return self._format_shares(raw_add)

    def get_leader_add_shares(self, total_equity: float, current_shares: float, entry_price: float) -> float:
        if entry_price <= 0 or total_equity <= 0:
            return 0.0

        leader_target_usd = total_equity * self.cfg.max_leader_pct
        current_usd = current_shares * entry_price
        rem_usd = max(0.0, leader_target_usd - current_usd)
        return self._format_shares(rem_usd / entry_price)