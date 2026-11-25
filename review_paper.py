import json
import os
import pandas as pd

LOG_DIR = "paper_logs"


def summarize_paper_trading():
    state_path = os.path.join(LOG_DIR, "paper_state.json")
    trades_path = os.path.join(LOG_DIR, "paper_trades.csv")
    equity_path = os.path.join(LOG_DIR, "paper_equity.csv")

    print("=" * 68)
    print(" 마크 미너비니 가상매매 실전 운용 성과 리포트 ($1,350 시작)")
    print("=" * 68)

    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        init_cash = float(st.get("initial_cash", 1350.0))
        cash = float(st.get("cash", 1350.0))
        holdings = st.get("holdings", {})
        last_px = st.get("last_prices", {})
        hold_val = sum(float(v["shares"]) * float(last_px.get(t, v["avg_price"])) for t, v in holdings.items())
        tot_eq = cash + hold_val
        ret_pct = (tot_eq - init_cash) / init_cash * 100.0
        print(f" 초기 자본금   : ${init_cash:,.2f}")
        print(f" 현재 평가자산 : ${tot_eq:,.2f} ({ret_pct:+.2f}%) | 현금: ${cash:,.2f} | 주식: ${hold_val:,.2f}")
        print(f" 현재 보유종목 : {list(holdings.keys()) if holdings else '없음'}")
        for t, v in holdings.items():
            cur = float(last_px.get(t, v["avg_price"]))
            u_pct = (cur - v["avg_price"]) / v["avg_price"] * 100.0
            print(f"   -> {t}: {v['shares']}주 | 평단 ${v['avg_price']:.2f} -> 현재 ${cur:.2f} ({u_pct:+.2f}%) | 손절 ${v['stop_price']:.2f}")

    if os.path.exists(trades_path):
        df = pd.read_csv(trades_path)
        sells = df[df["action"] == "SELL"]
        print("-" * 68)
        print(f" 총 매수 횟수  : {len(df[df['action'] == 'BUY'])}회 | 총 청산 횟수: {len(sells)}회")
        if not sells.empty:
            wins = sells[sells["pnl_usd"] > 0]
            losses = sells[sells["pnl_usd"] <= 0]
            win_rate = len(wins) / len(sells) * 100.0
            avg_win = wins["pnl_pct"].mean() if not wins.empty else 0.0
            avg_loss = losses["pnl_pct"].mean() if not losses.empty else 0.0
            rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
            print(f" 실전 승률     : {win_rate:.1f}% ({len(wins)}승 {len(losses)}패)")
            print(f" 평균 수익률   : +{avg_win:.2f}% | 평균 손실률: {avg_loss:.2f}% | 손익비: {rr:.2f} : 1")
            print("\n [청산 사유별 분포]")
            print(sells["reason"].value_counts().to_string())

    if os.path.exists(equity_path):
        eq = pd.read_csv(equity_path)
        if not eq.empty:
            print("-" * 68)
            print(" [최근 일별 계좌 기록]")
            print(eq.tail(10).to_string(index=False))
    print("=" * 68)


if __name__ == "__main__":
    summarize_paper_trading()