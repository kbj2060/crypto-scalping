"""«Macro Trend» 카드 -- ETF 이후 기준 매크로 추세 지표 (2026-10-28 사용자 «대시보드 카드 하나 만들어서 모두 추가»).

무엇: 연구 판정표(docs/experiments/macro_crypto_correlation_20261027/L_p3_trend_verdict.csv, 만든 스크립트
  scripts/research_macro_trend_p3_verdict_20261028.py)에서 «사용 가능»·«관찰» 지표를 읽고, 같은 계산(주간 금요일 값의 13주 변화)을
  지금 데이터로 다시 해서 보여준다.
  동행13 = 코인과 같은 13주 동안 같이 움직이는 지표 → «지금 큰 흐름이 어느 쪽인가»
  선행13 = 지표의 직전 13주 변화 → 코인 다음 13주 수익과 상관 → «다음 13주 편향»
행마다:
  chg13 = 지난 완결 주 기준 13주 변화 · z = ETF 이후(2024-01-12~) 13주 변화 분포에서의 위치
  bias  = sign(chg13) × sign(연구 상관) (|z|<0.5 면 중립) -- 동행은 «코인 강세 쪽/약세 쪽», 선행은 «다음 13주 ↑/↓»
  watch = 최근 52주 상관이 연구 부호와 같은가(«유효» |r|≥0.15 · «약화» · «이탈» 반대 부호 |r|≥0.15) -- 한 번 뒤집힌 관계가 다시
          뒤집히는지 감시하는 장치(사용자 «반전·신규도 꾸준하면 쓸 수 있는 것으로»). 선행은 미래 13주가 필요해 13주 전에 끝나는 창이다.
  stale = 최근 26주 13주 변화의 흔들림이 ETF 이후 전체의 25% 미만이면 «변동 거의 없음»(RRP 잔고 ≈0 같은 경우)
원천(1시간 캐시): FRED 정식 API(FRED_API_KEY) · DefiLlama 스테이블코인 합계 · 바이낸스 USDⓈ-M 일봉(BTC·ETH).
  🔴 연구는 Coin Metrics 4종(USDT·USDC·DAI·BUSD) 합을 썼고 여기는 DefiLlama USD 스테이블코인 전체다(13주 변화는 거의 같다).
  🔴 연구는 Yahoo BTC/ETH 일봉 Open[D+1](= D 의 00:00 UTC 종가)을 썼고 여기는 바이낸스 선물 일봉 종가(같은 시각)다.
🔴 판정은 표본내 상관(ETF 이후 ≈2.8년 · 13주 독립 관측 ≈11)이다 -- 표본외 검증 전, 매매 신호 아님(카드 머리에 고정 표시).
  python -m dashboard.macro_trend --selftest   # 네트워크 없이
"""
from __future__ import annotations

import csv
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
VERDICT_CSV = ROOT / "docs/experiments/macro_crypto_correlation_20261027/L_p3_trend_verdict.csv"
P3_START = "2024-01-12"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
LLAMA_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
Z_NEUTRAL, WATCH_MIN, STALE_RATIO, OLD_DAYS = 0.5, 0.15, 0.25, 21

# 지표 -> (원천, 단위, 실시간 등급 free|paid|none, 실시간 경로 설명). 원천 «fred:ID» · «netliq» · «stables».
# KOSPI 는 FRED 에 없어(연구는 미러) 카드에서 뺀다.
SOURCES = {
    "S&P500": ("fred:SP500", "%", "free", "XYZ100(나스닥100 무기한) 무료 실시간 · ES 선물은 유료"),
    "10Y금리": ("fred:DGS10", "bp", "paid", "ZN 선물 유료 실시간 · 무료 시세는 약 10분 지연"),
    "기대인플레": ("fred:T10YIE", "bp", "none", "하루 1회(다음 영업일) · 장중은 TIP/IEF 비율로 근사"),
    "기간프리미엄": ("fred:THREEFYTP10", "bp", "none", "하루 1회 모형 추정치(Kim-Wright)"),
    "VIX": ("fred:VIXCLS", "%", "paid", "CBOE 유료 실시간 · 무료 시세는 약 15분 지연"),
    "Baa스프레드": ("fred:BAA10Y", "bp", "none", "하루 1회 · 장중은 LQD/HYG 로 근사"),
    "HY스프레드": ("fred:BAMLH0A0HYM2", "bp", "none", "하루 1회(다음 영업일) · 장중은 HYG ETF 로 근사"),
    "STLFSI": ("fred:STLFSI4", "", "none", "주 1회 공표(주간 평균으로 만든 지수라 반응이 늦다)"),
    "연준BS": ("fred:WALCL", "%", "none", "주 1회(목요일 H.4.1)"),
    "스테이블코인": ("stables", "%", "free", "온체인 발행·소각(블록 단위) · 이 카드는 DefiLlama 일별 합계"),
    "브렌트": ("fred:DCOILBRENTEU", "%", "paid", "ICE 선물 유료 실시간 · 무료 시세는 지연"),
    "USD/JPY": ("fred:DEXJPUS", "%", "free", "FX 브로커 API 무료 실시간(주 5일 24시간)"),
    "USD/CNY": ("fred:DEXCHUS", "%", "free", "역외 위안(CNH) 무료 실시간"),
    "EUR/USD": ("fred:DEXUSEU", "%", "free", "FX 브로커 API 무료 실시간"),
    "원/달러": ("fred:DEXKOUS", "%", "free", "국내 증권사 API(한국 장중)"),
    "순유동성": ("netliq", "%", "none", "주 1회(연준 대차대조표 − 재무부 계좌 − 역레포)"),
    "지준": ("fred:WRESBAL", "%", "none", "주 1회(H.4.1)"),
    "TGA": ("fred:WTREGEN", "%", "none", "재무부 일일보고서(다음 영업일) · 이 카드는 FRED 주간"),
    "달러지수": ("fred:DTWEXBGS", "%", "none", "주 1회(H.10) · 장중은 DX 선물(유료)·EUR/USD(무료)로 근사"),
    "연방기금": ("fred:DFF", "bp", "none", "FOMC 결정 때만 바뀐다"),
    "RRP": ("fred:RRPONTSYD", "$bn", "none", "하루 1회(뉴욕 연은)"),
}
# 연구 판정 때 본 주의(2026-10-28 대화 기록). 동적 «변동 거의 없음»은 stale 로 따로 판정한다.
NOTES = {
    "연준BS": "양적긴축 기간과 BTC 상승이 겹친 관계 -- 최근 변동이 1/3 로 줄었다(보류 권장)",
    "RRP": "잔고가 이미 0 근처($0.2bn) -- 더 움직일 여지가 없다",
    "연방기금": "ETF 이후 금리 변경 7번뿐(사실상 사건 7개) · «인하 뒤 ETH 약세»라는 뜻 -- 해석 주의",
    "USD/CNY": "관리 환율이라 최근 변동이 작다(13주 변화 흔들림 1/3)",
    "스테이블코인": "BTC 를 뒤따르는 변수(BTC 상승 2~7주 뒤 발행) -- 확인용",
    "HY스프레드": "FRED 에 2023-10 이후만 있어 2020–23 과 비교 불가",
    "STLFSI": "주간 평균으로 만든 지수라 한 주 늦게 반응한다",
}


def load_spec(path: Path = VERDICT_CSV) -> list[dict]:
    """판정표에서 «사용 가능»·«관찰»만, 원천이 있는 지표만. 지표×관계 하나에 BTC·ETH 를 묶는다."""
    rows: dict[tuple, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["verdict"] not in ("사용 가능", "관찰") or r["indicator"] not in SOURCES:
                continue
            key = (r["indicator"], r["kind"])
            src, unit, tier, rt = SOURCES[r["indicator"]]
            g = rows.setdefault(key, {"key": f"{'co' if r['kind'] == '동행13' else 'lead'}:{r['indicator']}", "name": r["indicator"],
                                      "kind": "co" if r["kind"] == "동행13" else "lead", "source": src, "transform": r["transform"],
                                      "unit": unit, "realtime": tier, "realtime_note": rt, "note": NOTES.get(r["indicator"], ""),
                                      "assets": {}})
            g["assets"][r["asset"]] = {k: (r[k] if k in ("verdict", "vs_2020_23") else (float(r[k]) if r[k] not in ("", "nan") else None))
                                       for k in ("verdict", "vs_2020_23", "r_p3", "perm_p", "r_first_half", "r_second_half",
                                                 "r_excl_2024h2", "roll52_same_sign", "r_2020_23")}
    order = {"사용 가능": 0, "관찰": 1}
    return sorted(rows.values(), key=lambda g: (g["kind"] != "co", min(order[a["verdict"]] for a in g["assets"].values()),
                                                -max(abs(a["r_p3"]) for a in g["assets"].values())))


# ── 원천 ────────────────────────────────────────────────────────────────────────────────────────────
def parse_fred_json(js: dict) -> pd.Series:
    """FRED API series/observations. 결측 «.» 은 버린다."""
    ob = [o for o in js["observations"] if o["value"] not in (".", "")]
    return pd.Series([float(o["value"]) for o in ob], index=pd.to_datetime([o["date"] for o in ob]), dtype="float64").sort_index()


def fetch_fred(sid: str) -> pd.Series:
    """정식 API · 키는 서버 .env(server.py 가 시작 때 load_dotenv). 🔴2026-10-28 fredgraph.csv 는 서버에서 TLS 뒤 응답이 없어
    19개 계열이 전부 25초 시간 초과였다 -- 카드에 스테이블코인 한 줄만 남았다. 같은 서버에서 API 는 19개 모두 받았다."""
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY 없음")
    for i in range(3):
        try:
            r = requests.get(FRED_API, params={"series_id": sid, "api_key": key, "file_type": "json", "observation_start": "2022-01-01"},
                             timeout=15)
            r.raise_for_status()
            return parse_fred_json(r.json())
        except requests.RequestException as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if i == 2 or (code is not None and 400 <= code < 500 and code != 429):   # 잘못된 키·계열은 다시 해도 같다
                raise
            time.sleep(2 * (i + 1))


def fetch_stables() -> pd.Series:
    r = requests.get(LLAMA_URL, timeout=25)
    r.raise_for_status()
    d, v = [], []
    for x in r.json():
        tot = (x.get("totalCirculatingUSD") or x.get("totalCirculating") or {}).get("peggedUSD")
        if tot:
            d.append(pd.Timestamp(int(x["date"]), unit="s").normalize())
            v.append(float(tot))
    return pd.Series(v, index=pd.DatetimeIndex(d), dtype="float64").sort_index()


def fetch_klines_close(symbol: str) -> pd.Series:
    """바이낸스 선물 일봉 종가. 라벨 = 봉 시작 날짜(종가 = 다음 날 00:00 UTC) -- 연구의 Open[D+1] 라벨과 같다. 진행 중 봉 제외."""
    r = requests.get(KLINES_URL, params={"symbol": symbol, "interval": "1d", "limit": 1200}, timeout=20)
    r.raise_for_status()
    now_ms = time.time() * 1000
    k = [x for x in r.json() if float(x[6]) < now_ms]
    return pd.Series([float(x[4]) for x in k], index=pd.to_datetime([int(x[0]) for x in k], unit="ms").normalize(), dtype="float64")


def fetch_all(spec: list[dict]) -> tuple[dict[str, pd.Series], dict[str, str]]:
    need = {"BTC": ("klines", "BTCUSDT"), "ETH": ("klines", "ETHUSDT")}
    for g in spec:
        src = g["source"]
        if src.startswith("fred:"):
            need[src[5:]] = ("fred", src[5:])
        elif src == "netliq":
            for sid in ("WALCL", "WTREGEN", "RRPONTSYD"):
                need[sid] = ("fred", sid)
        elif src == "stables":
            need["stables"] = ("stables", "")
    fn = {"fred": fetch_fred, "stables": lambda _: fetch_stables(), "klines": fetch_klines_close}
    out, errors = {}, {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {k: ex.submit(fn[kind], arg) for k, (kind, arg) in need.items()}
        for k, f in futs.items():
            try:
                out[k] = f.result()
            except Exception as e:   # 원천 하나가 실패해도 카드는 나머지로 그린다 -- 실패는 payload.errors 로 보인다
                errors[k] = re.sub(r"api_key=[^&\s]+", "api_key=***", f"{type(e).__name__}: {str(e)}")[:160]   # 오류 문자열(요청 URL)이 화면으로 간다 -- 키는 지운다
    return out, errors


# ── 계산 (순수 함수) ─────────────────────────────────────────────────────────────────────────────────
def weekly(s: pd.Series, today: pd.Timestamp) -> pd.Series:
    """연구 sample('W') 와 같은 규약: 달력 일별로 펴서 ffill(최대 10일) → 금요일 마감 주의 마지막 값. 아직 안 끝난 주는 뺀다."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        return s
    cal = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).ffill(limit=10)
    w = cal.resample("W-FRI").last()
    return w[w.index <= today.normalize()]


def level(w: pd.Series, how: str) -> pd.Series:
    return 100 * np.log(w.where(w > 0)) if how == "logpct" else (100 * w if how == "bp" else w)


def indicator_level(g: dict, raw: dict[str, pd.Series], today: pd.Timestamp) -> pd.Series | None:
    src = g["source"]
    if src.startswith("fred:"):
        s = raw.get(src[5:])
        return None if s is None else level(weekly(s, today), g["transform"])
    if src == "netliq":   # WALCL[백만$] − WTREGEN[백만$] − RRP[십억$]×1000 (연구 패널과 같은 식, 수요일 격자)
        if any(raw.get(k) is None for k in ("WALCL", "WTREGEN", "RRPONTSYD")):
            return None
        a = raw["WALCL"]
        nl = a - raw["WTREGEN"].reindex(a.index) - raw["RRPONTSYD"].reindex(
            pd.date_range(raw["RRPONTSYD"].index.min(), a.index.max(), freq="D")).ffill(limit=5).reindex(a.index).fillna(0.0) * 1000.0
        return level(weekly(nl.dropna(), today), "logpct")
    if src == "stables":
        s = raw.get("stables")
        return None if s is None else level(weekly(s, today), "logpct")
    return None


def row_live(g: dict, x: pd.Series, coins: dict[str, pd.Series], today: pd.Timestamp | None = None) -> dict:
    """지표 한 줄의 현재 판독. x = 주간 지표 수준(변환 후), coins = 주간 100·ln(가격)."""
    chg = x.diff(13)
    p3 = chg.loc[P3_START:].dropna()
    last_week = chg.last_valid_index()
    out = {"week": None if last_week is None else str(last_week.date()), "chg13": None, "z": None, "stale": False, "old": False,
           "assets": {}}
    if last_week is None or len(p3) < 30:
        return out
    cur, mu, sd = float(chg[last_week]), float(p3.mean()), float(p3.std())
    z = (cur - mu) / sd if sd > 0 else 0.0
    recent = p3.iloc[-26:]
    out.update(chg13=cur, z=z, p3_mean=mu, p3_sd=sd, stale=bool(sd > 0 and recent.std() < STALE_RATIO * sd),
               old=bool(today is not None and (today - last_week).days > OLD_DAYS),
               recent_sd_ratio=float(recent.std() / sd) if sd > 0 else None,
               spark=[round(float(v), 4) for v in chg.loc[:last_week].iloc[-52:]],   # 화면 겹친 선 -- 지표의 최근 52주(빈 주는 NaN → null 로 자리 유지)
               coin13={a: [round(float(v), 4) for v in c.diff(13).reindex(chg.loc[:last_week].index).iloc[-52:]] for a, c in coins.items()})   # 같은 주에 맞춘 코인 쪽
    for asset, research in g["assets"].items():
        c = coins.get(asset)
        sign = 1.0 if research["r_p3"] >= 0 else -1.0
        bias = 0 if abs(z) < Z_NEUTRAL else int(np.sign(cur) * sign)
        rc = None
        if c is not None:
            if g["kind"] == "co":
                d = pd.concat([c.diff(13), chg], axis=1).dropna()
            else:   # 지표 13주 변화(1주 공표 시차) vs 다음 13주 수익 -- 끝이 13주 전인 창(미래가 확정된 구간만)
                d = pd.concat([chg.shift(1), c.shift(-13) - c], axis=1).dropna()
            d = d.iloc[-52:]
            if len(d) >= 26:
                rc = float(d.corr().iloc[0, 1])
        if rc is None or not math.isfinite(rc):
            watch = "모름"
        elif np.sign(rc) == sign and abs(rc) >= WATCH_MIN:
            watch = "유효"
        elif np.sign(rc) != sign and abs(rc) >= WATCH_MIN:
            watch = "이탈"
        else:
            watch = "약화"
        out["assets"][asset] = {"bias": bias, "roll52": rc, "watch": watch}
        out["assets"][asset].update(zip(("state", "why"), cell_state(research, out, out["assets"][asset], today)))
    return out


def _f(x: float, d: int = 2) -> str:
    return f"{x:+.{d}f}".replace("-", "−")


def cell_state(research: dict, live: dict, la: dict, today: pd.Timestamp | None = None) -> tuple[str, str]:
    """칸 하나(지표 × 관계 × 코인)의 상태와 이유 한 줄. summarize 가 이 상태로만 센다(규칙이 한 곳).
    count/zero = 셈(↑↓/중립) · obs = 연구 판정 «관찰»(안 셈) · stale/old/weak/off/unknown = «사용 가능»인데 지금은 안 셈."""
    r = research["r_p3"]
    if research["verdict"] != "사용 가능":   # 연구 스크립트의 «사용 가능» 다섯 조건 중 못 넘은 것
        miss = []
        if abs(r) < 0.3:
            miss.append(f"ETF 이후 |r| {abs(r):.2f} < 0.30")
        if research.get("perm_p") is not None and research["perm_p"] >= 0.05:
            miss.append(f"순열 p {research['perm_p']:.2f} ≥ 0.05")
        h = min(abs(research.get("r_first_half") or 0), abs(research.get("r_second_half") or 0))
        if h < 0.15:
            miss.append(f"전반·후반 중 약한 쪽 |r| {h:.2f} < 0.15")
        if research.get("roll52_same_sign") is not None and research["roll52_same_sign"] < 0.75:
            miss.append(f"52주 굴림 상관 같은 부호 {research['roll52_same_sign']:.0%} < 75%")
        return "obs", "관찰(안 셈) — 연구 «사용 가능» 기준 미달: " + (" · ".join(miss) or "판정표 기준")
    if live.get("stale"):
        return "stale", f"변동 없음(안 셈) — 최근 26주 13주 변화의 흔들림이 ETF 이후의 {live.get('recent_sd_ratio') or 0:.0%} (< 25%)"
    if live.get("old"):
        days = (today - pd.Timestamp(live["week"])).days if today is not None else None
        return "old", f"지연(안 셈) — 마지막 완결 주 {live['week']}{f' ({days}일 전)' if days is not None else ''} · 21일 넘음"
    rc = la.get("roll52")
    if la["watch"] == "모름":
        return "unknown", "감시 모름(안 셈) — 최근 52주 상관을 낼 겹치는 주가 26주 미만"
    if la["watch"] == "이탈":
        return "off", f"이탈(안 셈) — 최근 52주 r {_f(rc)} 가 연구 r {_f(r)} 와 반대 부호이고 |r| ≥ 0.15"
    if la["watch"] == "약화":
        side = "같은 쪽이지만" if np.sign(rc) == np.sign(r) else "반대 쪽이지만 아직"
        return "weak", f"약화(안 셈) — 최근 52주 r {_f(rc)}: 연구 r {_f(r)} 와 {side} |r| < 0.15"
    if la["bias"] == 0:
        return "zero", f"중립(셈) — 13주 변화가 ETF 이후 분포 {_f(live['z'], 1)}σ (|z| < 0.5)"
    return "count", f"셈 — 연구 r {_f(r)} · 최근 52주 r {_f(rc)} 유효 · {_f(live['z'], 1)}σ"


def summarize(rows: list[dict]) -> dict:
    """자산별: 판정 «사용 가능» · 감시 «유효» · 변동 있음 · 데이터 신선(기준 주 21일 이내)인 줄만 센다. 나머지는 excluded."""
    out = {}
    for asset in ("BTC", "ETH"):
        s = {"co_up": 0, "co_dn": 0, "lead_up": 0, "lead_dn": 0, "neutral": 0, "excluded": 0}
        for r in rows:
            live = (r.get("live") or {}).get("assets", {}).get(asset)
            st = live and live.get("state")
            if st in (None, "obs"):
                continue
            if st == "zero":
                s["neutral"] += 1
            elif st == "count":
                s[f"{r['kind']}_{'up' if live['bias'] > 0 else 'dn'}"] += 1
            else:
                s["excluded"] += 1
        out[asset] = s
    return out


def build_payload(spec: list[dict], raw: dict[str, pd.Series], errors: dict[str, str], now: float) -> dict:
    today = pd.Timestamp(now, unit="s").normalize()
    coins = {a: level(weekly(raw[a], today), "logpct") for a in ("BTC", "ETH") if raw.get(a) is not None}
    rows = []
    for g in spec:
        x = indicator_level(g, raw, today)
        live = None if x is None or x.dropna().empty else row_live(g, x, coins, today)
        rows.append({**{k: g[k] for k in ("key", "name", "kind", "unit", "realtime", "realtime_note", "note", "assets")}, "live": live})
    return {"ok": bool(coins) and any(r["live"] and r["live"]["chg13"] is not None for r in rows), "asof": int(now),
            "rows": rows, "summary": summarize(rows), "errors": errors,
            "method": "주간 금요일 값의 13주 변화 · z = ETF 이후(2024-01-12~) 분포 위치 · 편향 = 변화 부호 × 연구 상관 부호(|z|<0.5 중립) · "
                      "감시 = 최근 52주 상관이 연구 부호와 같은가 · 표본내 연구, 검증 전"}


def json_safe(x):
    """NaN·inf → None (브라우저 JSON.parse 는 NaN 을 못 읽는다), numpy 스칼라 → 파이썬 수."""
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def payload(now: float | None = None) -> dict:
    now = time.time() if now is None else now
    spec = load_spec()
    raw, errors = fetch_all(spec)
    return json_safe(build_payload(spec, raw, errors, now))


def _selftest() -> None:
    """합성 주간 데이터: 지표가 코인을 13주 선행(+), 다른 지표는 무관. 편향·감시·정지 판정과 FRED CSV 파서를 본다."""
    rng = np.random.default_rng(3)
    days = pd.date_range("2022-01-03", "2026-10-23", freq="D")
    f = np.cumsum(rng.normal(0, 1, len(days)))
    btc = pd.Series(np.exp(10 + 0.02 * np.r_[np.zeros(91), f[:-91]] + 0.002 * np.cumsum(rng.normal(0, 1, len(days)))), index=days)
    lead_lvl = pd.Series(np.exp(4 + 0.02 * f), index=days)
    walk = np.exp(5 + 0.01 * np.cumsum(rng.normal(0, 1, len(days) - 300)))
    flat = pd.Series(np.r_[walk, np.full(300, walk[-1])], index=days)   # 마지막 300일은 마지막 값 그대로(잔고 ≈0 인 RRP 꼴)
    spec = [{"key": "lead:L", "name": "L", "kind": "lead", "source": "fred:L", "transform": "logpct", "unit": "%", "realtime": "free",
             "realtime_note": "", "note": "", "assets": {"BTC": {"verdict": "사용 가능", "r_p3": 0.5}}},
            {"key": "co:F", "name": "F", "kind": "co", "source": "fred:F", "transform": "logpct", "unit": "%", "realtime": "none",
             "realtime_note": "", "note": "", "assets": {"BTC": {"verdict": "사용 가능", "r_p3": -0.4}}}]
    p = build_payload(spec, {"BTC": btc, "L": lead_lvl, "F": flat}, {}, pd.Timestamp("2026-10-25").timestamp())
    lead, co = p["rows"][0]["live"], p["rows"][1]["live"]
    assert lead["assets"]["BTC"]["watch"] == "유효", lead
    assert lead["assets"]["BTC"]["roll52"] > 0.3, lead
    assert lead["assets"]["BTC"]["bias"] == (0 if abs(lead["z"]) < Z_NEUTRAL else int(np.sign(lead["chg13"]))), lead
    assert co["stale"] is True, co                                   # 마지막 200일 값이 고정 → 변동 거의 없음
    assert p["summary"]["BTC"]["excluded"] >= 1, p["summary"]
    assert lead["week"] == "2026-10-23", lead["week"]                 # 일요일 기준: 금요일(10-23) 주가 마지막 완결 주
    w = weekly(btc, pd.Timestamp("2026-10-22"))
    assert w.index[-1] == pd.Timestamp("2026-10-16"), w.index[-1]    # 목요일 기준: 이번 주(10-23)는 아직 미완결
    s = parse_fred_json({"observations": [{"date": "2024-01-01", "value": "1.5"}, {"date": "2024-01-02", "value": "."},
                                          {"date": "2024-01-03", "value": "2"}]})
    assert list(s.values) == [1.5, 2.0] and len(s) == 2
    assert lead["assets"]["BTC"]["state"] in ("count", "zero") and co["assets"]["BTC"]["state"] == "stale", (lead, co)
    obs = cell_state({"verdict": "관찰", "r_p3": -0.31, "perm_p": 0.08, "r_first_half": -0.4, "r_second_half": -0.1, "roll52_same_sign": 0.9},
                     {}, {})
    assert obs[0] == "obs" and "순열 p 0.08" in obs[1] and "0.10 < 0.15" in obs[1] and "|r| 0.31" not in obs[1], obs
    off = cell_state({"verdict": "사용 가능", "r_p3": -0.37}, {"z": 0.6}, {"watch": "이탈", "roll52": 0.21, "bias": 1})
    assert off[0] == "off" and "+0.21" in off[1] and "−0.37" in off[1], off
    import json
    json.dumps(json_safe(p), allow_nan=False)                         # 응답에 NaN 이 섞이면 여기서 터진다
    spec_real = load_spec()
    assert spec_real and all(g["source"] for g in spec_real) and {g["kind"] for g in spec_real} == {"co", "lead"}
    print("selftest OK:", {"rows": len(spec_real), "lead_watch": lead["assets"]["BTC"]["watch"],
                           "lead_roll52": round(lead["assets"]["BTC"]["roll52"], 2), "co_stale": co["stale"]})


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        import json
        print(json.dumps(payload(), ensure_ascii=False, default=float)[:2000])
