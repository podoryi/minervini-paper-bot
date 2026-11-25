import csv
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

US_TZ = ZoneInfo("America/New_York")
KR_TZ = ZoneInfo("Asia/Seoul")
LOG_DIR = "paper_logs"


class MockUSBroker:
    """자본 $1,350.00 영구 저장(JSON) 및 매매·자산·관심종목 일지(CSV) 자동 누적 가상 브로커"""

    def __init__(self, initial_cash: float = 1_350.0, slippage_pct: float = 0.001):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.state_path = os.path.join(LOG_DIR, "paper_state.json")
        self.trades_csv = os.path.join(LOG_DIR, "paper_trades.csv")
        self.equity_csv = os.path.join(LOG_DIR, "paper_equity.csv")
        self.watchlist_csv = os.path.join(LOG_DIR, "paper_watchlist.csv")

        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.slippage_pct = slippage_pct
        self.last_prices: dict = {}
        self.holdings: dict = {}

        self._init_csv_files()
        self._load_state()

    def _init_csv_files(self):
        if not os.path.exists(self.trades_csv):
            with open(self.trades_csv, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([
                    "kst_time", "et_time", "action", "ticker", "shares",
                    "fill_price_usd", "amount_usd", "pnl_usd", "pnl_pct",
                    "cash_after_usd", "total_equity_usd", "reason"
                ])

        if not os.path.exists(self.equity_csv):
            with open(self.equity_csv, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([
                    "et_date", "kst_time", "regime", "cash_usd", "holdings_usd",
                    "total_equity_usd", "total_return_pct", "invested_pct", "open_positions"
                ])

        if not os.path.exists(self.watchlist_csv):
            with open(self.watchlist_csv, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([
                    "et_date", "kst_time", "regime", "rank", "priority_score", "grade",
                    "ticker", "name", "pattern", "rs_score", "close_usd", "pivot_usd",
                    "stop_usd", "contractions", "fund_summary", "note", "headline"
                ])

    def _load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_init = float(data.get("initial_cash", self.initial_cash))
            saved_cash = float(data.get("cash", self.cash))
            saved_holdings = data.get("holdings", {})

            if not saved_holdings and abs(saved_cash - saved_init) < 0.01 and abs(saved_init - self.initial_cash) >= 0.01:
                self._save_state()
                print(f"[가상계좌 초기자본 재설정] 초기 자본금: ${self.initial_cash:,.2f}")
                return

            self.initial_cash = saved_init
            self.cash = saved_cash
            self.holdings = saved_holdings
            self.last_prices = data.get("last_prices", {})
            print(
                f"[가상계좌 복원 완료] 현금: ${self.cash:,.2f} | "
                f"보유종목: {list(self.holdings.keys())} | 총 평가액: ${self.get_total_equity():,.2f}"
            )
        else:
            self._save_state()
            print(f"[신규 가상계좌 개설] 초기 자본금: ${self.initial_cash:,.2f}")

    def _save_state(self):
        data = {
            "initial_cash": round(self.initial_cash, 2),
            "cash": round(self.cash, 2),
            "holdings": self.holdings,
            "last_prices": self.last_prices,
            "updated_kst": datetime.now(KR_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def sync_engine_positions(self, engine):
        from execution_engine import Position
        for ticker, info in self.holdings.items():
            engine.positions[ticker] = Position(
                ticker=ticker,
                shares=float(info["shares"]),
                avg_price=float(info["avg_price"]),
                pivot_price=float(info["pivot_price"]),
                stop_price=float(info["stop_price"]),
                entry_date=datetime.fromisoformat(info["entry_date"]),
                days_held=int(info.get("days_held", 0)),
                tranche=int(info.get("tranche", 1)),
                highest_price=float(info.get("highest_price", info["avg_price"])),
                partial_sold=bool(info.get("partial_sold", False)),
                last_counted_date=info.get("last_counted_date", ""),
            )

    def save_engine_positions(self, engine):
        for ticker, pos in engine.positions.items():
            self.holdings[ticker] = {
                "shares": round(float(pos.shares), 4),
                "avg_price": round(float(pos.avg_price), 4),
                "pivot_price": pos.pivot_price,
                "stop_price": pos.stop_price,
                "entry_date": pos.entry_date.isoformat(),
                "days_held": pos.days_held,
                "tranche": pos.tranche,
                "highest_price": pos.highest_price,
                "partial_sold": pos.partial_sold,
                "last_counted_date": pos.last_counted_date,
            }
        for ticker in list(self.holdings.keys()):
            if ticker not in engine.positions:
                del self.holdings[ticker]
        self._save_state()

    def update_price(self, ticker: str, price: float):
        if price > 0:
            self.last_prices[ticker] = float(price)

    def get_total_equity(self) -> float:
        holdings_val = 0.0
        for ticker, info in self.holdings.items():
            px = self.last_prices.get(ticker, info["avg_price"])
            holdings_val += float(info["shares"]) * px
        return self.cash + holdings_val

    def buy_market(self, ticker: str, shares: float, reason: str = "", price: float = None) -> float:
        ref_px = price or self.last_prices.get(ticker)
        shares = round(float(shares), 4)
        if not ref_px or shares <= 0:
            return 0.0

        fill_price = round(ref_px * (1 + self.slippage_pct), 4)
        cost = round(fill_price * shares, 4)
        if cost > self.cash:
            shares = round(self.cash / fill_price, 4)
            if shares <= 0:
                return 0.0
            cost = round(fill_price * shares, 4)

        self.cash = round(self.cash - cost, 4)
        if ticker in self.holdings:
            old = self.holdings[ticker]
            new_shares = round(float(old["shares"]) + shares, 4)
            new_avg = (old["avg_price"] * float(old["shares"]) + cost) / new_shares
            old["shares"] = new_shares
            old["avg_price"] = round(new_avg, 4)
        else:
            self.holdings[ticker] = {
                "shares": shares,
                "avg_price": fill_price,
                "pivot_price": fill_price,
                "stop_price": round(fill_price * 0.945, 2),
                "entry_date": datetime.now(US_TZ).isoformat(),
            }

        tot_eq = self.get_total_equity()
        kst_str = datetime.now(KR_TZ).strftime("%Y-%m-%d %H:%M:%S")
        et_str = datetime.now(US_TZ).strftime("%Y-%m-%d %H:%M:%S")

        with open(self.trades_csv, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([
                kst_str, et_str, "BUY", ticker, shares,
                round(fill_price, 2), round(cost, 2), 0.0, 0.0,
                round(self.cash, 2), round(tot_eq, 2), reason
            ])

        self._save_state()
        print(f"[가상 매수 체결 | {kst_str}] {ticker} {shares:.4f}주 @ ${fill_price:,.2f} (${cost:,.2f}) | 사유: {reason}")
        return fill_price

    def sell_market(self, ticker: str, shares: float, reason: str = "", price: float = None) -> float:
        if ticker not in self.holdings or shares <= 0:
            return 0.0

        old = self.holdings[ticker]
        sell_qty = round(min(float(shares), float(old["shares"])), 4)
        ref_px = price or self.last_prices.get(ticker, old["avg_price"])
        fill_price = round(ref_px * (1 - self.slippage_pct), 4)

        proceeds = round(fill_price * sell_qty, 4)
        pnl_usd = (fill_price - old["avg_price"]) * sell_qty
        pnl_pct = (fill_price - old["avg_price"]) / old["avg_price"] * 100.0

        self.cash = round(self.cash + proceeds, 4)
        old["shares"] = round(float(old["shares"]) - sell_qty, 4)
        if old["shares"] <= 0.0001:
            del self.holdings[ticker]

        tot_eq = self.get_total_equity()
        kst_str = datetime.now(KR_TZ).strftime("%Y-%m-%d %H:%M:%S")
        et_str = datetime.now(US_TZ).strftime("%Y-%m-%d %H:%M:%S")

        with open(self.trades_csv, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([
                kst_str, et_str, "SELL", ticker, sell_qty,
                round(fill_price, 2), round(proceeds, 2), round(pnl_usd, 2), round(pnl_pct, 2),
                round(self.cash, 2), round(tot_eq, 2), reason
            ])

        self._save_state()
        print(
            f"[가상 매도 체결 | {kst_str}] {ticker} {sell_qty:.4f}주 @ ${fill_price:,.2f} | "
            f"손익: ${pnl_usd:+,.2f} ({pnl_pct:+.2f}%) | 사유: {reason}"
        )
        return fill_price

    def log_watchlist_snapshot(self, signals: list, regime: str):
        et_date = datetime.now(US_TZ).strftime("%Y-%m-%d")
        kst_str = datetime.now(KR_TZ).strftime("%Y-%m-%d %H:%M:%S")
        with open(self.watchlist_csv, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            for idx, s in enumerate(signals, start=1):
                writer.writerow([
                    et_date, kst_str, regime, idx, s.priority_score, s.fund_grade,
                    s.ticker, s.name, s.pattern_type, s.rs_score, s.close, s.pivot_price,
                    s.suggested_stop, str(s.contractions), s.fund_summary, s.llm_note, s.catalyst_headline
                ])

    def log_daily_equity(self, regime: str):
        et_date = datetime.now(US_TZ).strftime("%Y-%m-%d")
        kst_str = datetime.now(KR_TZ).strftime("%Y-%m-%d %H:%M:%S")
        tot_eq = self.get_total_equity()
        holdings_usd = tot_eq - self.cash
        ret_pct = (tot_eq - self.initial_cash) / self.initial_cash * 100.0
        invested_pct = (holdings_usd / tot_eq * 100.0) if tot_eq > 0 else 0.0

        with open(self.equity_csv, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([
                et_date, kst_str, regime, round(self.cash, 2), round(holdings_usd, 2),
                round(tot_eq, 2), round(ret_pct, 2), round(invested_pct, 1),
                ",".join(self.holdings.keys())
            ])
        print(
            f"[일별 계좌 스냅샷 저장 | {et_date}] 총 평가액: ${tot_eq:,.2f} ({ret_pct:+.2f}%) | "
            f"주식비중: {invested_pct:.1f}% | 보유: {list(self.holdings.keys())}"
        )