from dataclasses import dataclass

@dataclass
class TradingConfig:
    market_regime: str = "BULL"           # "BULL", "CAUTION", "BEAR"
    use_fractional_shares: bool = True    # 소액 계좌($1,350) 소수점 4자리 주문 허용

    # 1. 기관 수급 유동성 및 2단계(Stage 2) 트렌드 템플릿
    min_price: float = 15.0
    min_dollar_volume: float = 50_000_000
    min_above_52w_low: float = 0.25
    max_below_52w_high: float = 0.25
    sma200_uptrend_days: int = 22
    min_rs_percentile: float = 92.0

    # 2. VCP · 피벗 · 돌파 검증 조건
    pivot_lookback_days: int = 5
    max_chase_pct: float = 0.025
    min_breakout_vol_ratio: float = 1.35

    # 3. 분할 진입 및 정체 종목 순환매 통제
    trial_entry_ratio: float = 0.50
    reentry_cooldown_days: int = 1
    stagnant_days_limit: int = 12

    @property
    def bear_market_mode(self) -> bool:
        return self.market_regime == "BEAR"

    @bear_market_mode.setter
    def bear_market_mode(self, val: bool):
        self.market_regime = "BEAR" if val else "BULL"

    @property
    def max_positions(self) -> int:
        return {"BULL": 6, "CAUTION": 5, "BEAR": 5}.get(self.market_regime, 5)

    @property
    def initial_position_pct(self) -> float:
        return {"BULL": 0.16, "CAUTION": 0.11, "BEAR": 0.08}.get(self.market_regime, 0.10)

    @property
    def max_leader_pct(self) -> float:
        return {"BULL": 0.24, "CAUTION": 0.11, "BEAR": 0.08}.get(self.market_regime, 0.10)

    @property
    def max_account_risk_pct(self) -> float:
        return {"BULL": 0.020, "CAUTION": 0.015, "BEAR": 0.010}.get(self.market_regime, 0.015)

    @property
    def hard_stop_pct(self) -> float:
        return {"BULL": 0.080, "CAUTION": 0.065, "BEAR": 0.055}.get(self.market_regime, 0.065)

    @property
    def stage1_trim_gain_pct(self) -> float:
        return {"BULL": 0.22, "CAUTION": 0.15, "BEAR": 0.12}.get(self.market_regime, 0.15)