import re
import json
import os
import pickle
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf

CACHE_DIR = "data_cache"
SEC_HEADERS = {"User-Agent": "MinerviniQuantResearch/1.0 (research@quantlocal.org)"}
RSS_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

CATALYST_KEYWORDS = (
    "fda", "approval", "approved", "clearance", "cleared", "ce mark", "certification",
    "trial", "phase", "sbla", "nda", "bla", "breakthrough", "patent", "milestone",
    "earnings", "quarter", "q1", "q2", "q3", "q4", "results",
    "beat", "beats", "surge", "surges", "record", "guidance", "raise", "raises",
    "outlook", "contract", "deal", "partnership", "award", "order", "backlog",
    "upgrade", "outperform", "buy rating", "price target", "ai", "demand",
    "launch", "commercial", "expansion", "growth", "profit", "revenue", "soar", "rally"
)

NEGATIVE_KEYWORDS = (
    "public offering", "registered direct", "dilution", "lawsuit", "investigation",
    "sec probe", "miss", "misses", "cut guidance", "lowers guidance",
    "downgrade", "recall", "halt", "bankruptcy", "delist", "plunge", "slump"
)

_CIK_MAP_CACHE: Optional[Dict[str, str]] = None


@dataclass
class FundamentalReport:
    ticker: str
    passed: bool
    grade: str
    sector: str
    days_to_earnings: Optional[int]
    is_profitable: bool
    eps_yoy_pct: Optional[float]
    rev_yoy_pct: Optional[float]
    is_accelerating: bool
    has_catalyst: bool
    has_bad_news: bool
    has_pead_gap: bool
    consensus_up: bool
    forward_accel: bool
    priority_score: int
    headline: str
    llm_note: str
    summary: str


def _get_sec_cik_map() -> Dict[str, str]:
    global _CIK_MAP_CACHE
    if _CIK_MAP_CACHE is not None:
        return _CIK_MAP_CACHE

    os.makedirs(CACHE_DIR, exist_ok=True)
    cik_cache_path = os.path.join(CACHE_DIR, "sec_company_tickers.json")

    raw_bytes = None
    if os.path.exists(cik_cache_path):
        if (time.time() - os.path.getmtime(cik_cache_path)) / 86400.0 <= 7:
            with open(cik_cache_path, "rb") as f:
                raw_bytes = f.read()

    if raw_bytes is None:
        req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_bytes = resp.read()
        with open(cik_cache_path, "wb") as f:
            f.write(raw_bytes)

    data = json.loads(raw_bytes.decode("utf-8"))
    cik_map = {}
    for item in data.values():
        t = str(item.get("ticker", "")).upper().strip()
        cik = str(item.get("cik_str", "")).zfill(10)
        if t:
            cik_map[t] = cik
            cik_map[t.replace("-", ".")] = cik
            cik_map[t.replace(".", "-")] = cik

    _CIK_MAP_CACHE = cik_map
    return cik_map


def _extract_quarterly_series(facts: dict, taxonomy: str, tag_list: List[str], unit_key: str) -> List[Tuple[date, float]]:
    tax_block = facts.get("facts", {}).get(taxonomy, {})
    for tag in tag_list:
        if tag not in tax_block:
            continue
        entries = tax_block[tag].get("units", {}).get(unit_key, [])
        by_end: Dict[date, Tuple[str, float]] = {}

        for e in entries:
            start_s, end_s, val = e.get("start"), e.get("end"), e.get("val")
            form, filed = str(e.get("form", "")), str(e.get("filed", ""))
            if not start_s or not end_s or val is None:
                continue
            if form not in ("10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "6-K"):
                continue
            try:
                d_start, d_end = date.fromisoformat(start_s), date.fromisoformat(end_s)
            except ValueError:
                continue

            if 75 <= (d_end - d_start).days <= 105:
                if d_end not in by_end or filed >= by_end[d_end][0]:
                    by_end[d_end] = (filed, float(val))

        if len(by_end) >= 2:
            return [(d_end, val_tuple[1]) for d_end, val_tuple in sorted(by_end.items(), reverse=True)]
    return []


def _calc_yoy_and_accel(q_series: List[Tuple[date, float]]) -> Tuple[bool, Optional[float], bool]:
    if not q_series:
        return False, None, False
    q0_date, q0_val = q_series[0]
    is_prof = q0_val > 0

    def find_yoy_base(target_date: date, start_idx: int) -> Optional[float]:
        for idx in range(start_idx, len(q_series)):
            d, v = q_series[idx]
            if 345 <= (target_date - d).days <= 385:
                return v
        return None

    def growth(curr: float, prev: Optional[float]) -> Optional[float]:
        if prev is None or prev == 0:
            return None
        if prev < 0 and curr > 0:
            return 999.0
        if prev < 0 and curr <= 0:
            return None
        return (curr - prev) / abs(prev) * 100.0

    g0 = growth(q0_val, find_yoy_base(q0_date, 1))
    is_accel = False
    if len(q_series) >= 2 and g0 is not None:
        g1 = growth(q_series[1][1], find_yoy_base(q_series[1][0], 2))
        if g1 is not None and g0 > g1:
            is_accel = True
    if not is_accel and len(q_series) >= 3 and q_series[0][1] > q_series[1][1] > q_series[2][1] > 0:
        is_accel = True

    return is_prof, g0, is_accel


def _fetch_sec_fundamentals(ticker: str) -> Tuple[Optional[bool], Optional[float], Optional[float], bool]:
    cik = _get_sec_cik_map().get(ticker.upper())
    if not cik:
        return None, None, None, False

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    req = urllib.request.Request(url, headers=SEC_HEADERS)
    with urllib.request.urlopen(req, timeout=12) as resp:
        facts = json.loads(resp.read().decode("utf-8"))

    eps_q = _extract_quarterly_series(facts, "us-gaap", ["EarningsPerShareDiluted", "EarningsPerShareBasic"], "USD/shares")
    ni_q = _extract_quarterly_series(facts, "us-gaap", ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"], "USD")
    rev_q = _extract_quarterly_series(
        facts, "us-gaap", [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "RevenuesNetOfInterestExpense",
        ], "USD"
    )

    profit_series = eps_q if len(eps_q) >= 4 else ni_q
    if not profit_series and not rev_q:
        return None, None, None, False

    is_prof, eps_yoy, eps_accel = _calc_yoy_and_accel(profit_series)
    _, rev_yoy, rev_accel = _calc_yoy_and_accel(rev_q)
    return is_prof, eps_yoy, rev_yoy, (eps_accel or rev_accel)


def _fetch_ticker_meta_and_consensus(ticker: str) -> Tuple[str, Optional[int], bool, bool]:
    """섹터, 어닝 발표 잔여일, 30일 EPS 컨센서스 상향, 향후 2분기 가속 조회"""
    sector = "Unknown"
    days_to_earnings = None
    consensus_up = False
    forward_accel = False
    today_d = datetime.now(timezone.utc).date()

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        sector = str(info.get("sector") or "Unknown")

        # 어닝 발표 예정일 확인 (yfinance calendar 또는 earningsTimestampStart)
        cal = getattr(tk, "calendar", None)
        if isinstance(cal, dict) and cal.get("Earnings Date"):
            ed_list = cal["Earnings Date"]
            if isinstance(ed_list, list) and ed_list:
                ed = ed_list[0]
                if hasattr(ed, "date"):
                    ed = ed.date()
                if isinstance(ed, date):
                    days_to_earnings = (ed - today_d).days
        elif info.get("earningsTimestampStart"):
            ts = int(info["earningsTimestampStart"])
            ed = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            days_to_earnings = (ed - today_d).days

        rev_df = getattr(tk, "eps_revisions", None)
        if rev_df is not None and not rev_df.empty and "0q" in rev_df.index:
            up30 = float(rev_df.loc["0q"].get("upLast30days", 0) or 0)
            dn30 = float(rev_df.loc["0q"].get("downLast30days", 0) or 0)
            if up30 > 0 and up30 >= dn30:
                consensus_up = True

        est_df = getattr(tk, "earnings_estimate", None)
        if est_df is not None and not est_df.empty and "0q" in est_df.index and "+1q" in est_df.index:
            g0 = est_df.loc["0q"].get("growth")
            g1 = est_df.loc["+1q"].get("growth")
            if pd.notna(g0) and pd.notna(g1) and float(g1) > float(g0) and float(g1) > 0:
                forward_accel = True
    except Exception:
        pass

    return sector, days_to_earnings, consensus_up, forward_accel



BUYOUT_TARGET_PHRASES = (
    "agreed to be acquired",
    "to be acquired by",
    "definitive merger agreement",
    "definitive agreement to be acquired",
    "tender offer",
    "take-private",
    "take private",
    "going-private",
    "going private",
    "buyout offer",
    "buyout deal",
    "merger arbitrage",
    "hsr waiting period",
    "refile premerger",
    "antitrust clearance",
)


def _detect_buyout_target(ticker: str, titles: List[str]) -> Optional[str]:
    t_esc = re.escape(ticker.lower())
    target_patterns = [
        re.compile(rf"\b(to acquire|acquires|acquired|acquiring|acquisition of|buyout of|takeover of)\s+{t_esc}\b"),
        re.compile(rf"\b{t_esc}\s+(to be acquired|agreed to sell|agrees to be acquired|buyout)\b"),
    ]
    for title in titles:
        t_low = title.lower()
        if any(phrase in t_low for phrase in BUYOUT_TARGET_PHRASES):
            return title
        if any(pat.search(t_low) for pat in target_patterns):
            return title
    return None


def _is_merger_arb_pinned(daily_df: Optional[pd.DataFrame]) -> bool:
    if daily_df is None or len(daily_df) < 30:
        return False
    sub = daily_df.tail(65).copy()
    prev_c = sub["Close"].shift(1)
    surge = (sub["Close"] - prev_c) / prev_c
    if not bool((surge >= 0.15).any()):
        return False

    recent = sub.tail(12)
    intraday_range = ((recent["High"] - recent["Low"]) / recent["Close"]).median()
    daily_move = ((recent["Close"] - recent["Close"].shift(1)).abs() / recent["Close"].shift(1)).dropna().median()
    peak_65 = float(sub["High"].max())
    curr_c = float(sub["Close"].iloc[-1])
    near_cap = (peak_65 - curr_c) / peak_65 <= 0.035

    return bool(near_cap and intraday_range <= 0.013 and daily_move <= 0.006)


def _fetch_rss_news(ticker: str) -> List[str]:
    now_dt = datetime.now(timezone.utc)
    titles = []
    urls = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(ticker)}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={urllib.parse.quote(ticker + ' stock')}&hl=en-US&gl=US&ceid=US:en",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=RSS_HEADERS)
            with urllib.request.urlopen(req, timeout=8) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for item in root.findall(".//item")[:10]:
                t_el, p_el = item.find("title"), item.find("pubDate")
                if t_el is None or not t_el.text:
                    continue
                title = t_el.text.strip()
                age_days = 0.0
                if p_el is not None and p_el.text:
                    try:
                        pub_dt = parsedate_to_datetime(p_el.text.strip())
                        if pub_dt.tzinfo is None:
                            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                        age_days = (now_dt - pub_dt).total_seconds() / 86400.0
                    except Exception:
                        age_days = 0.0
                if age_days <= 30.0 and title not in titles:
                    titles.append(title)
        except Exception:
            continue
    return titles


def evaluate_catalyst_with_llm(ticker: str, headlines: List[str]) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or not headlines:
        return "API키 미설정(정량지표 우선)"
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = (
            f"종목 {ticker}의 최근 뉴스 헤드라인입니다:\n"
            + "\n".join(f"- {h}" for h in headlines[:6])
            + "\n\n마크 미너비니 관점에서 다음 한 줄 형식으로만 답하세요(다른 설명 금지):\n"
            "[유형: 반복업사이클/실적서프라이즈/인허가수주/일회성 | 가이던스: 상향/유지/하향/언급없음] 핵심근거 25자 이내"
        )
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return (resp.text or "").strip().replace("\n", " ")[:110]
    except Exception:
        return "LLM 판정 생략(정량지표 우선)"


def check_pead_gap_event(daily_df: Optional[pd.DataFrame]) -> bool:
    if daily_df is None or len(daily_df) < 55:
        return False
    sub = daily_df.tail(65).copy()
    prev_c = sub["Close"].shift(1)
    vol_avg = sub["Volume"].rolling(50, min_periods=20).mean().shift(1)
    open_gap = (sub["Open"] - prev_c) / prev_c
    close_surge = (sub["Close"] - prev_c) / prev_c
    vol_ratio = sub["Volume"] / vol_avg
    events = ((open_gap >= 0.08) | (close_surge >= 0.10)) & (vol_ratio >= 3.0)
    return bool(events.tail(60).any())


def analyze_fundamentals_and_catalyst(ticker: str, daily_df: Optional[pd.DataFrame] = None) -> FundamentalReport:
    os.makedirs(CACHE_DIR, exist_ok=True)
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    cache_path = os.path.join(CACHE_DIR, f"fundamentals_v5_{today_str}.pkl")

    cache_dict = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache_dict = pickle.load(f)
            if ticker in cache_dict:
                return cache_dict[ticker]
        except Exception:
            cache_dict = {}

    has_pead = check_pead_gap_event(daily_df)
    recent_titles = _fetch_rss_news(ticker)

    buyout_headline = _detect_buyout_target(ticker, recent_titles)
    is_pinned = _is_merger_arb_pinned(daily_df)
    if buyout_headline or is_pinned:
        reason = "M&A \ud53c\uc778\uc218 \ub274\uc2a4 \uac10\uc9c0" if buyout_headline else "\uc778\uc218 \ucc28\uc775\uac70\ub798 \uac00\uaca9\uace0\uc815 \uac10\uc9c0"
        rep = FundamentalReport(
            ticker=ticker, passed=False, grade="C", sector="Unknown", days_to_earnings=None,
            is_profitable=False, eps_yoy_pct=None, rev_yoy_pct=None, is_accelerating=False,
            has_catalyst=False, has_bad_news=True, has_pead_gap=has_pead,
            consensus_up=False, forward_accel=False, priority_score=0,
            headline=buyout_headline or (recent_titles[0] if recent_titles else reason),
            llm_note="\ud53c\uc778\uc218(M&A) \uc0c1\ubc29\uc81c\ud55c \uc81c\uc678",
            summary=f"\ud53c\uc778\uc218(M&A) \ud655\uc815 \uc885\ubaa9 \uc81c\uc678 ({reason})"
        )
        cache_dict[ticker] = rep
        with open(cache_path, "wb") as f:
            pickle.dump(cache_dict, f)
        return rep
    has_catalyst = False
    has_bad_news = False
    best_headline = recent_titles[0] if recent_titles else "최근 14일 주요 뉴스 없음"

    for title in recent_titles:
        t_low = title.lower()
        if any(bad in t_low for bad in NEGATIVE_KEYWORDS):
            has_bad_news = True
        if any(cat in t_low for cat in CATALYST_KEYWORDS):
            if not has_catalyst:
                best_headline = title
            has_catalyst = True

    if has_bad_news:
        rep = FundamentalReport(
            ticker=ticker, passed=False, grade="C", sector="Unknown", days_to_earnings=None,
            is_profitable=False, eps_yoy_pct=None, rev_yoy_pct=None, is_accelerating=False,
            has_catalyst=has_catalyst, has_bad_news=True, has_pead_gap=has_pead,
            consensus_up=False, forward_accel=False, priority_score=0,
            headline=best_headline, llm_note="악재 제외", summary="악재 뉴스 감지(유상증자/가이던스 하향 등)"
        )
        cache_dict[ticker] = rep
        with open(cache_path, "wb") as f:
            pickle.dump(cache_dict, f)
        return rep

    time.sleep(0.12)
    is_prof, eps_yoy, rev_yoy, is_accel = None, None, None, False
    try:
        is_prof, eps_yoy, rev_yoy, is_accel = _fetch_sec_fundamentals(ticker)
    except Exception:
        pass

    if is_prof is None and rev_yoy is None:
        try:
            info = yf.Ticker(ticker).info or {}
            t_eps = info.get("trailingEps")
            ni = info.get("netIncomeToCommon")
            eq_g = info.get("earningsQuarterlyGrowth")
            rv_g = info.get("revenueGrowth")
            if t_eps is not None:
                is_prof = float(t_eps) > 0
            elif ni is not None:
                is_prof = float(ni) > 0
            if eq_g is not None:
                eps_yoy = float(eq_g) * 100.0
            if rv_g is not None:
                rev_yoy = float(rv_g) * 100.0
        except Exception:
            pass

    prof_flag = bool(is_prof)
    eps_ok = prof_flag and (eps_yoy is not None and eps_yoy >= 20.0)
    rev_ok = (rev_yoy is not None and rev_yoy >= 20.0)

    if eps_ok and is_accel:
        grade, fund_ok = "A+", True
    elif eps_ok:
        grade, fund_ok = "A", True
    elif (prof_flag and rev_ok) or (rev_ok and has_catalyst):
        grade, fund_ok = "B", True
    elif is_prof is None and rev_yoy is None:
        grade, fund_ok = "ADR", True
    else:
        grade, fund_ok = "C", False

    passed = bool(fund_ok and has_catalyst)

    sector, days_to_earnings, consensus_up, forward_accel = ("Unknown", None, False, False)
    llm_note = ""
    if passed:
        sector, days_to_earnings, consensus_up, forward_accel = _fetch_ticker_meta_and_consensus(ticker)
        # 어닝 룰렛 차단: 0~3일 이내 실적 발표 예정이면 신규 돌파 진입 보류
        if days_to_earnings is not None and 0 <= days_to_earnings <= 3:
            passed = False
            llm_note = f"어닝 D-{days_to_earnings} 임박 제외"
        else:
            llm_note = evaluate_catalyst_with_llm(ticker, recent_titles)

    base_pts = {"A+": 100, "A": 80, "B": 60, "ADR": 50}.get(grade, 0)
    priority_score = base_pts + (30 if has_pead else 0) + (25 if consensus_up else 0) + (20 if forward_accel else 0)

    eps_txt = "흑자전환" if eps_yoy == 999.0 else (f"{eps_yoy:+.1f}%" if eps_yoy is not None else "N/A")
    rev_txt = f"{rev_yoy:+.1f}%" if rev_yoy is not None else "N/A"
    tags = []
    if has_pead:
        tags.append("PEAD갭+")
    if consensus_up:
        tags.append("컨센서스상향")
    if forward_accel:
        tags.append("선행2Q가속")
    if days_to_earnings is not None and days_to_earnings >= 0:
        tags.append(f"어닝D-{days_to_earnings}")
    tag_str = f" [{'·'.join(tags)}]" if tags else ""

    summary = f"등급:{grade}({priority_score}점) | 섹터:{sector} | {'흑자' if prof_flag else '적자'} | 이익YoY:{eps_txt} | 매출YoY:{rev_txt}{tag_str}"

    rep = FundamentalReport(
        ticker=ticker,
        passed=passed,
        grade=grade,
        sector=sector,
        days_to_earnings=days_to_earnings,
        is_profitable=prof_flag,
        eps_yoy_pct=round(eps_yoy, 1) if eps_yoy is not None else None,
        rev_yoy_pct=round(rev_yoy, 1) if rev_yoy is not None else None,
        is_accelerating=is_accel,
        has_catalyst=has_catalyst,
        has_bad_news=False,
        has_pead_gap=has_pead,
        consensus_up=consensus_up,
        forward_accel=forward_accel,
        priority_score=priority_score,
        headline=best_headline,
        llm_note=llm_note,
        summary=summary,
    )

    cache_dict[ticker] = rep
    with open(cache_path, "wb") as f:
        pickle.dump(cache_dict, f)
    return rep