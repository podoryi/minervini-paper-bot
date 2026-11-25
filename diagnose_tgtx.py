import glob
import pickle
import pandas as pd

cache_files = sorted(glob.glob("data_cache/us_clean_bt_*y.pkl"))
if not cache_files:
    raise FileNotFoundError("data_cache 폴더에 캐시 파일이 없습니다.")

with open(cache_files[-1], "rb") as f:
    universe_data, index_df = pickle.load(f)

df = universe_data["TGTX"].copy()
df["SMA20"] = df["Close"].rolling(20).mean()
df["SMA50"] = df["Close"].rolling(50).mean()
df["SMA150"] = df["Close"].rolling(150).mean()
df["SMA200"] = df["Close"].rolling(200).mean()
df["VOL_SMA50"] = df["Volume"].rolling(50).mean()
df["VOL_RATIO"] = df["Volume"] / df["VOL_SMA50"]
df["HIGH_52W"] = df["High"].rolling(252, min_periods=200).max()
df["LOW_52W"] = df["Low"].rolling(252, min_periods=200).min()
df["RANGE_PCT"] = (df["High"] - df["Low"]) / df["Low"] * 100

df["R50"] = (df["High"].rolling(50).max() - df["Low"].rolling(50).min()) / df["High"].rolling(50).max() * 100
df["R25"] = (df["High"].rolling(25).max() - df["Low"].rolling(25).min()) / df["High"].rolling(25).max() * 100
df["R10"] = (df["High"].rolling(10).max() - df["Low"].rolling(10).min()) / df["High"].rolling(10).max() * 100
df["R5"] = (df["High"].rolling(5).max() - df["Low"].rolling(5).min()) / df["High"].rolling(5).max() * 100

# 전체 유니버스 대비 RS 백분위 계산 (전일 기준)
idx_ret63 = index_df["Close"].iloc[-2] / index_df["Close"].iloc[-65] - 1
idx_ret126 = index_df["Close"].iloc[-2] / index_df["Close"].iloc[-128] - 1
idx_ret252 = index_df["Close"].iloc[-2] / index_df["Close"].iloc[-254] - 1

rs_scores = {}
for t, d in universe_data.items():
    if len(d) > 255:
        r63 = d["Close"].iloc[-2] / d["Close"].iloc[-65] - 1
        r126 = d["Close"].iloc[-2] / d["Close"].iloc[-128] - 1
        r252 = d["Close"].iloc[-2] / d["Close"].iloc[-254] - 1
        rs_scores[t] = (r63 - idx_ret63) * 0.4 + (r126 - idx_ret126) * 0.3 + (r252 - idx_ret252) * 0.3

rs_pct = pd.Series(rs_scores).rank(pct=True) * 100.0
tgtx_rs = float(rs_pct.get("TGTX", 0.0))

print("=== [1] TGTX 최근 7거래일 시세 및 거래량 비율 ===")
cols = ["Open", "High", "Low", "Close", "VOL_RATIO", "RANGE_PCT"]
print(df[cols].tail(7).round(2).to_string())

setup_day = df.iloc[-2]  # 2026-11-23 (진입 전일)
entry_day = df.iloc[-1]  # 2026-11-24 (진입 당일)
pivot_5d = float(df["High"].iloc[-6:-1].max())

print(f"\n=== [2] 진입 전일({setup_day.name.strftime('%Y-%m-%d')}) 스크리닝 조건 점검 ===")
print(f" - RS 백분위 점수      : {tgtx_rs:.1f}점 (기준: 90점 이상)")
print(f" - 이평선 정배열       : Close({setup_day['Close']:.2f}) > SMA50({setup_day['SMA50']:.2f}) > SMA150({setup_day['SMA150']:.2f}) > SMA200({setup_day['SMA200']:.2f})")
print(f" - 52주 고점 대비 위치 : {(setup_day['Close']/setup_day['HIGH_52W'] - 1)*100:.1f}% (기준: -25% 이내)")
print(f" - 52주 저점 대비 상승 : {(setup_day['Close']/setup_day['LOW_52W'] - 1)*100:.1f}% (기준: +25% 이상)")
print(f" - VCP 수축(50->25->10->5일): {setup_day['R50']:.1f}% -> {setup_day['R25']:.1f}% -> {setup_day['R10']:.1f}% -> {setup_day['R5']:.1f}%")
print(f" - 직전 5일 피벗 돌파가: ${pivot_5d:.2f}")

print(f"\n=== [3] 진입 당일({entry_day.name.strftime('%Y-%m-%d')}) 돌파 조건 점검 ===")
print(f" - 시가 / 고가 / 종가  : ${entry_day['Open']:.2f} / ${entry_day['High']:.2f} / ${entry_day['Close']:.2f}")
print(f" - 피벗 돌파 여부      : {'돌파 성공' if entry_day['High'] > pivot_5d else '미돌파'} (피벗 ${pivot_5d:.2f})")
print(f" - 당일 거래량 배수    : {entry_day['VOL_RATIO']:.2f}배 (기준: 1.35배 이상)")