#!/usr/bin/env python3
"""GEX (dealer gamma exposure) live collector — Stage 0 (2026-08-15 per user request, following
the "which discretionary/systematic trader fits this model" research this session: dealer-gamma-
positioning traders (Cem Karsan/Kai Volatility, SqueezeMetrics' GEX methodology) were the one
genuinely open information axis after orderflow/AMT/VSA/iFVG/trend-following/vol-targeting/
funding/DVOL-level/Fear&Greed were all already closed in this repo -- see
docs/experiments/eth_amt_vsa_footprint_ifvg_strategy_absorption_study_20260815.md and
docs/experiments/eth_h48qual_quality_new_data_source_research_20260811.md (candidate 5/9: "옵션체인
스큐/OI/GEX -- 아직 안 죽은 차별점").

WHY LIVE-FORWARD ONLY, NOT A BACKTEST: confirmed empirically 2026-08-15 that Deribit's free public
REST API cannot reconstruct historical option chains -- get_instruments?expired=true only returns
instruments that expired in roughly the last 1-2 days (tested: returned only yesterday's 38 ETH
expiries, nothing from the repo's actual VAL 2025-09..12 / OOS 2026-01..02 window), there is no
historical open-interest-by-strike or historical-greeks endpoint, and every free 3rd-party source
checked (CryptoDataDownload's options chain files, Tardis.dev's free tier) is either paywalled for
the actual chain/OI fields or only gives 1 day/month (useless for bar-level evidence-study work).
Paid providers (Tardis.dev full history, Amberdata) exist but that is a real spending decision, not
something to commit to unilaterally. So this script starts a live snapshot collector TODAY, in the
same spirit as the existing F4-C altdata collector and the Polymarket duckdb collector -- it will
take weeks to accumulate anything backtestable, and this script makes no promotion/signal claim.

WHAT IT COLLECTS (Deribit public REST, no auth, same deliberate choice as every other raw
downloader in this repo):
  - get_book_summary_by_currency(kind=option) for ETH and BTC: one bulk call each, returns
    instrument_name/open_interest/mark_iv/underlying_price/mark_price/volume for every currently
    LIVE option instrument (694 for ETH as of this writing) -- no per-instrument ticker calls
    needed, avoiding hundreds of requests per snapshot.
  - Per-instrument strike/expiry/option_type parsed from instrument_name (Deribit's own format,
    e.g. "ETH-4SEP26-2400-C"), not re-fetched.
  - Per-instrument gamma computed HERE via Black-Scholes (r=0, matching Deribit's own
    ticker.interest_rate=0.0 convention observed empirically), using mark_iv as sigma -- Deribit's
    book-summary endpoint does not return greeks directly, only per-instrument ticker calls do,
    and 700+ ticker calls per snapshot is both slow and needlessly duplicates a one-line formula.

GEX CONVENTION (disclosed simplification, not verified real dealer positioning -- same caveat the
literature itself carries): GEX_i = gamma_i * open_interest_i * contract_size * S^2 * 0.01, signed
+1 for calls / -1 for puts (the standard SqueezeMetrics-style assumption that call OI is
dealer-short and put OI is dealer-long from customer order flow). Reported in USD notional terms
treating open_interest as ETH/BTC-denominated contracts (Deribit options are inverse-settled;
this is the same simplification every retail GEX calculator makes, not a rigorous inverse-contract
adjustment -- documented here, not hidden). Two aggregates stored: total (all expiries) and
front_month (expiries within the next 30 days -- the literature's claim is specifically about
near-dated dealer hedging flow, not far OTM long-dated OI).

Schema: data/live/deribit_gex.duckdb
  option_chain_snapshot: raw per-instrument row per poll (recorded_at_utc, currency,
    instrument_name, option_type, strike, expiration_ts, days_to_expiry, open_interest, mark_iv,
    underlying_price, mark_price, volume, gamma_bs)
  gex_summary: one row per poll per currency (recorded_at_utc, currency, spot_price,
    total_gex_usd, front_month_gex_usd, n_instruments, n_front_month)

Run standalone (single poll) or loop with --interval-sec. No live trading file touched.

2026-10-02 사용자 «실현 감마 국면으로 대시보드와 데이터 저장 로직 모두 교체»: 딜러 감마(체결 기반·관행 가정) 계산을 전부 걷었다 --
  gex_summary 쓰기 · 감마 곡선/플립/DEX/charm(gamma·gamma_by) · 사다리 감마 칸 · 체결 기반 딜러 포지션(_taker_flow).
  어떤 딜러 가정도 검증되지 않았다(scripts/research_eth_dealer_assumption_matrix_20261002.py). 실현 감마는 가격(5분봉)만으로
  화면이 계산하므로 저장할 것이 없다. 원자료(option_chain_snapshot · option_trades)는 그대로 쌓는다(11-10 사전등록 판정이 쓴다).
  옛 gex_summary 표는 지우지 않고 이력으로 남긴다(새로 쓰지 않음). 위 «GEX CONVENTION» 문단은 옛 설명이다.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data/live/deribit_gex.duckdb"
# 🔴대시보드는 이 JSON 만 읽는다. duckdb 는 **단일 writer** 라 매시 cron 이 쓰는 파일을
# 대시보드가 직접 열면 `Could not set lock` 이 난다 -- 이 저장소 수집기 관례대로
# 끝에 작은 상태파일을 떨군다(tmp -> os.replace, 부분 읽기 없음).
STATE_PATH = ROOT / "data/live/deribit_gex_state.json"
PUBLIC = "https://www.deribit.com/api/v2/public/"
BASE_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
CURRENCIES = ("ETH", "BTC", "SOL", "XRP")
# 2026-09-28 SOL·XRP(사용자 지시): Deribit 에서 이 둘은 **USDC 결제 선형 옵션**이다 -- `currency=USDC` 한 목록에
#   SOL_USDC-·XRP_USDC-(와 HYPE·AVAX·TRX…)가 섞여 온다(실측: SOL 720 · XRP 440종목). 표시가(mark_price)는 USDC,
#   미결제는 기초자산 수량(ETH·BTC 역옵션과 같은 단위), 행사가 소수점은 `d`(XRP `1d55` = 1.55), DVOL 지수는 없다.
#   (API 목록 이름, 종목 접두어, 지수 이름, 역옵션 여부 = 표시가가 코인 단위인가)
SPECS = {"ETH": ("ETH", "ETH-", "eth_usd", True), "BTC": ("BTC", "BTC-", "btc_usd", True),
         "SOL": ("USDC", "SOL_USDC-", "sol_usdc", False), "XRP": ("USDC", "XRP_USDC-", "xrp_usdc", False)}
INSTRUMENT_RE = re.compile(r"^(?P<ccy>[A-Z_]+)-(?P<exp>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>\d+(?:[.d]\d+)?)-(?P<type>[CP])$")


def log(msg: str) -> None:
    print(f"[deribit_gex] {msg}", flush=True)


def connect_retry(path: Path, retries: int = 5, backoff: float = 2.0):
    last_exc = None
    for attempt in range(retries):
        try:
            return duckdb.connect(str(path))
        except duckdb.IOException as exc:
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
    raise last_exc


def ensure_tables(con) -> None:
    con.execute(
        """CREATE TABLE IF NOT EXISTS option_chain_snapshot (
            recorded_at_utc TIMESTAMPTZ, currency VARCHAR, instrument_name VARCHAR,
            option_type VARCHAR, strike DOUBLE, expiration_ts TIMESTAMPTZ, days_to_expiry DOUBLE,
            open_interest DOUBLE, mark_iv DOUBLE, underlying_price DOUBLE, mark_price DOUBLE,
            volume DOUBLE, gamma_bs DOUBLE
        )"""
    )
    # 2026-09-28 옵션 요약 이력(스큐·기간 구조·만기 규모) -- 과거분을 살 수 없어서 지금부터 쌓는다(나중에 검정).
    con.execute(
        """CREATE TABLE IF NOT EXISTS option_summary (
            recorded_at_utc TIMESTAMPTZ, currency VARCHAR, index_price DOUBLE, dvol DOUBLE, rv7 DOUBLE,
            front_atm_iv DOUBLE, front_rr25 DOUBLE, front_bf25 DOUBLE, gamma_flip DOUBLE, payload VARCHAR
        )"""
    )
    # 2026-10-03 사용자 «OKX·Bybit 옵션 수집»: 일간 만기 미결제의 70~80% 가 두 거래소에 있다(Deribit 21~29%, 10-03 실측).
    #   원자료만 쌓는다(화면 없음) -- 세 거래소 합산 max pain 을 장부(eth_maxpain_1h_oos_ledger_20261003)와 나란히 판정하려고.
    con.execute(
        """CREATE TABLE IF NOT EXISTS option_oi_other (
            recorded_at_utc TIMESTAMPTZ, venue VARCHAR, currency VARCHAR, instrument_name VARCHAR, option_type VARCHAR,
            strike DOUBLE, expiration_ts TIMESTAMPTZ, open_interest DOUBLE, mark_iv DOUBLE, volume_24h DOUBLE
        )"""
    )


def _parse_instrument(name: str) -> dict | None:
    m = INSTRUMENT_RE.match(name)
    if not m:
        return None
    exp_dt = datetime.strptime(m.group("exp"), "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
    return {
        "strike": float(m.group("strike").replace("d", ".")),
        "expiration_ts": exp_dt,
        "option_type": "call" if m.group("type") == "C" else "put",
    }


def _bs_gamma(spot: float, strike: float, iv_pct: float, years: float) -> float:
    """Black-Scholes gamma, r=0 (matches Deribit's own ticker.interest_rate=0.0 convention)."""
    sigma = iv_pct / 100.0
    if spot <= 0 or strike <= 0 or sigma <= 0 or years <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * years) / (sigma * math.sqrt(years))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return pdf / (spot * sigma * math.sqrt(years))


_LIST_CACHE: dict = {}
_RV_CACHE: dict = {}


class _Cached(Exception):
    pass


def fetch_chain(currency: str) -> pd.DataFrame:
    api, prefix, _, _ = SPECS[currency]
    # USDC 목록은 SOL·XRP 가 같이 쓴다 -- 한 폴링 안(60초)에서는 한 번만 받는다.
    hit = _LIST_CACHE.get(api)
    if hit and time.time() - hit[0] < 60:
        rows = hit[1]
    else:
        r = requests.get(BASE_URL, params={"currency": api, "kind": "option"}, timeout=20)
        r.raise_for_status()
        rows = r.json().get("result", [])
        _LIST_CACHE[api] = (time.time(), rows)
    rows = [x for x in rows if x["instrument_name"].startswith(prefix)]
    now = datetime.now(timezone.utc)
    out = []
    for row in rows:
        parsed = _parse_instrument(row["instrument_name"])
        if parsed is None:
            continue
        years = (parsed["expiration_ts"] - now).total_seconds() / (365.0 * 86400.0)
        if years <= 0:
            continue
        spot = float(row.get("underlying_price") or 0.0)
        gamma = _bs_gamma(spot, parsed["strike"], float(row.get("mark_iv") or 0.0), years)
        out.append({
            "recorded_at_utc": now, "currency": currency, "instrument_name": row["instrument_name"],
            "option_type": parsed["option_type"], "strike": parsed["strike"],
            "expiration_ts": parsed["expiration_ts"], "days_to_expiry": years * 365.0,
            "open_interest": float(row.get("open_interest") or 0.0), "mark_iv": float(row.get("mark_iv") or 0.0),
            "underlying_price": spot, "mark_price": float(row.get("mark_price") or 0.0),
            "volume": float(row.get("volume") or 0.0), "gamma_bs": gamma,
            # 2026-10-01 ATM 호가 폭(IV pt) 계산용 -- 표에는 안 넣는다(스키마 그대로, INSERT 는 열 이름으로)
            "bid_price": float(row.get("bid_price") or 0.0), "ask_price": float(row.get("ask_price") or 0.0),
        })
    return pd.DataFrame(out)


def _pub(method: str, timeout: float = 30, **params):
    r = requests.get(PUBLIC + method, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()["result"]


def _bs_delta(fwd: float, strike: float, iv_pct: float, years: float, call: bool) -> float:
    sigma = iv_pct / 100.0
    if fwd <= 0 or strike <= 0 or sigma <= 0 or years <= 0:
        return 0.0
    d1 = (math.log(fwd / strike) + 0.5 * sigma * sigma * years) / (sigma * math.sqrt(years))
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    return nd1 if call else nd1 - 1.0


def _skew_interp(g: pd.DataFrame, fwd: float, yrs: float) -> dict:
    """한 만기의 외가격 옵션만으로 델타 보간 25Δ(콜 +0.25 · 풋 −0.25) IV 와 로그머니니스 보간 ATM IV → rr25i · bf25i · atm_i · hours.
    외가격이 양쪽 2개 미만이거나 25Δ 가 행사가 범위 밖이면 None(외삽하지 않는다)."""
    import numpy as np
    out = {"hours": yrs * 8760, "rr25i": None, "bf25i": None, "atm_i": None}
    oc = g[(g["option_type"] == "call") & (g["strike"] >= fwd)].sort_values("strike")
    op = g[(g["option_type"] == "put") & (g["strike"] <= fwd)].sort_values("strike")
    if len(oc) < 2 or len(op) < 2 or yrs <= 0:
        return out
    def at(x, y, x0):
        o = np.argsort(x); x, y = np.asarray(x, float)[o], np.asarray(y, float)[o]
        return float(np.interp(x0, x, y)) if x[0] <= x0 <= x[-1] else None
    otm = pd.concat([op[op["strike"] < fwd], oc])
    atm = at(np.log(otm["strike"] / fwd), otm["mark_iv"], 0.0)
    dc = [_bs_delta(fwd, k, v, yrs, True) for k, v in zip(oc["strike"], oc["mark_iv"])]
    dp = [_bs_delta(fwd, k, v, yrs, False) for k, v in zip(op["strike"], op["mark_iv"])]
    c25, p25 = at(dc, oc["mark_iv"], 0.25), at(dp, op["mark_iv"], -0.25)
    out["atm_i"] = atm
    if c25 is not None and p25 is not None:
        out["rr25i"] = c25 - p25
        if atm is not None:
            out["bf25i"] = (c25 + p25) / 2 - atm
    return out


def _smile(g: pd.DataFrame, fwd: float):
    """한 만기의 외가격 mark IV 를 k = ln(K/F) 로 선형 보간(양 끝 밖은 끝값 고정)하는 함수. 외가격이 3개 미만이면 None."""
    import numpy as np
    otm = pd.concat([g[(g["option_type"] == "put") & (g["strike"] < fwd)], g[(g["option_type"] == "call") & (g["strike"] >= fwd)]])
    otm = otm[otm["mark_iv"] > 0].sort_values("strike")
    if len(otm) < 3:
        return None
    k, v = np.log(otm["strike"].to_numpy(float) / fwd), otm["mark_iv"].to_numpy(float) / 100.0
    return lambda kk: np.interp(kk, k, v)


def _surface_stats(g: pd.DataFrame, fwd: float, yrs: float) -> dict:
    """2026-10-01 문헌 조사 뒤 옵션 카드 «옵션 정보»: 한 만기 스마일에서
    ① 모델프리 분산(VIX 식, 외가격 mark 가격 적분 -- Britten-Jones·Neuberger, Jiang·Tian) → 연율 IV
    ② 위험중립 꼬리확률 P(만기 가격이 선도가 ±2·3·5% 밖) -- 스마일로 만든 콜·풋 가격의 행사가 미분(Breeden·Litzenberger)
    ③ 위험중립 왜도·첨도(Bakshi·Kapadia·Madan 2003). r=0 · 선도가 기준 · 가격은 USD(역옵션도 같은 식 -- 비율만 쓴다).
    🔴서술용(검정 전). 스마일 밖은 끝값 고정 외삽이라 먼 꼬리는 근사다."""
    import numpy as np
    out = {"mfiv": None, "tail": None, "rn_skew": None, "rn_kurt": None}
    sm = _smile(g, fwd)
    if sm is None or yrs <= 0:
        return out
    sig0 = float(sm(0.0))
    w = max(4.5 * sig0 * math.sqrt(yrs), 0.06)
    K = fwd * np.exp(np.linspace(-w, w, 801))
    kk = np.log(K / fwd); sig = sm(kk); sq = sig * math.sqrt(yrs)
    d1 = (-kk + 0.5 * sq * sq) / sq; d2 = d1 - sq
    N = lambda x: 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))   # noqa: E731
    call = fwd * N(d1) - K * N(d2); put = call - (fwd - K)
    q = np.where(K < fwd, put, call)                                     # 외가격 가격
    dK = np.gradient(K)
    var = 2.0 / yrs * float(np.sum(q / K ** 2 * dK))                      # VIX 식(F = K0 근처라 보정항 생략)
    out["mfiv"] = math.sqrt(var) * 100 if var > 0 else None
    cdf = np.clip(1 + np.gradient(call, K), 0, 1)                         # P(S_T < K) = 1 + dC/dK
    tail = {}
    for x in (2, 3, 5):
        lo, hi = float(np.interp(fwd * (1 - x / 100), K, cdf)), float(np.interp(fwd * (1 + x / 100), K, cdf))
        tail[str(x)] = round(min(1.0, max(0.0, lo + (1 - hi))), 4)
    out["tail"] = tail
    up, dn = K >= fwd, K < fwd
    lr = np.log(K / fwd)
    V = np.sum((2 * (1 - lr) / K ** 2 * call * dK)[up]) + np.sum((2 * (1 - lr) / K ** 2 * put * dK)[dn])
    W = np.sum(((6 * lr - 3 * lr ** 2) / K ** 2 * call * dK)[up]) + np.sum(((6 * lr - 3 * lr ** 2) / K ** 2 * put * dK)[dn])
    X = np.sum(((12 * lr ** 2 - 4 * lr ** 3) / K ** 2 * call * dK)[up]) + np.sum(((12 * lr ** 2 - 4 * lr ** 3) / K ** 2 * put * dK)[dn])
    mu = -V / 2 - W / 6 - X / 24
    den = V - mu * mu
    if den > 0:
        out["rn_skew"] = float((W - 3 * mu * V + 2 * mu ** 3) / den ** 1.5)
        out["rn_kurt"] = float((X - 4 * mu * W + 6 * mu * mu * V - 3 * mu ** 4) / den ** 2)
    return out


def _atm_spread_iv(g: pd.DataFrame, fwd: float, yrs: float, linear: bool) -> float | None:
    """ATM(선도가 ±2%) 옵션 호가 폭을 IV 포인트로: (매도호가 − 매수호가)[USD] ÷ 베가[USD/1 vol pt] 의 중앙. 호가가 한쪽뿐이면 뺀다."""
    import numpy as np
    x = g[((g["strike"] / fwd - 1).abs() <= 0.02) & (g["bid_price"] > 0) & (g["ask_price"] > g["bid_price"])] if "bid_price" in g else g.iloc[0:0]
    if x.empty or yrs <= 0:
        return None
    out = []
    for _, r in x.iterrows():
        sig = r["mark_iv"] / 100.0
        if sig <= 0:
            continue
        d1 = (math.log(fwd / r["strike"]) + 0.5 * sig * sig * yrs) / (sig * math.sqrt(yrs))
        vega = fwd * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(yrs) / 100.0
        spread = (r["ask_price"] - r["bid_price"]) * (1.0 if linear else fwd)
        if vega > 0:
            out.append(spread / vega)
    return float(np.median(out)) if out else None


def _const_maturity(exps: list[dict], days: float) -> dict | None:
    """만기 둘 사이 보간한 고정만기 값: ATM = 총분산 선형 · RR/BF = 시간 선형. 양쪽 만기가 없으면(외삽) None."""
    tgt = days * 24
    ok = [e for e in exps if e.get("rr25i") is not None and e.get("bf25i") is not None and e.get("atm_i") is not None]
    a = max((e for e in ok if e["hours"] <= tgt), key=lambda e: e["hours"], default=None)
    b = min((e for e in ok if e["hours"] >= tgt), key=lambda e: e["hours"], default=None)
    if a is None or b is None:
        return None
    w = 0.0 if b["hours"] == a["hours"] else (tgt - a["hours"]) / (b["hours"] - a["hours"])
    var = (1 - w) * a["atm_i"] ** 2 * a["hours"] + w * b["atm_i"] ** 2 * b["hours"]
    return {"atm": math.sqrt(var / tgt) if var > 0 else None, "rr": (1 - w) * a["rr25i"] + w * b["rr25i"],
            "bf": (1 - w) * a["bf25i"] + w * b["bf25i"]}


def options_summary(chain: pd.DataFrame, currency: str) -> dict:
    """화면 «옵션» 카드 한 판(2026-09-28 사용자 선택 A+C). **참고 표시 전용 -- 신호 아님.**
    예상 폭(DVOL) · VRP(DVOL − 실현 7일) · 만기별(ATM IV·25Δ RR/BF·콜/풋 미결제·P/C·max pain) ·
    보험용 OTM 옵션 가격. 🔴가격 기준은 **지수**(eth_usd)다 -- 체인의 underlying_price 는 만기마다 다른 선도가라
    먼 만기 것을 «현물»로 쓰면 1~2% 어긋난다(2026-09-28 실측 2,700 vs 지수 2,645). 부가 조회 실패는 None 으로 둔다."""
    now = datetime.now(timezone.utc)
    out: dict = {"recorded_at_utc": now.isoformat(), "currency": currency}
    _, _, index_name, inverse = SPECS[currency]
    out["dvol"] = None                          # DVOL 지수는 ETH·BTC 뿐 -- SOL·XRP 는 아래 iv30 을 쓴다
    for key, fn in (("index", lambda: _pub("get_index_price", index_name=index_name)["index_price"]),
                    ("dvol" if inverse else "_skip", lambda: _pub("get_volatility_index_data", currency=currency, resolution="60",
                                          start_timestamp=int((time.time() - 7200) * 1000),
                                          end_timestamp=int(time.time() * 1000))["data"][-1][4])):
        if key == "_skip":
            continue
        try:
            out[key] = float(fn())
        except Exception as exc:        # 부가 값 -- 없으면 None, 수집은 계속
            out[key] = None; log(f"{currency}: {key} 실패 {exc}")
    # 실현 7일은 Deribit 차트 응답이 4초~55초+ 로 들쭉날쭉하다(2026-09-28 실측, 같은 호출) -- 7일치라 천천히 변하므로
    #   성공값을 6시간 재사용하고, 실패하면 24시간 안의 직전 값을 쓴다. 이 호출만 제한 시간 90초.
    hit = _RV_CACHE.get(currency)
    try:
        if hit and time.time() - hit[0] < 6 * 3600:
            raise _Cached
        tv = _pub("get_tradingview_chart_data", timeout=90, instrument_name=f"{currency}-PERPETUAL" if inverse else f"{currency}_USDC-PERPETUAL", resolution="60",
                  start_timestamp=int((time.time() - 7 * 86400) * 1000), end_timestamp=int(time.time() * 1000))
        cl = [c for c in tv["close"] if c]
        rets = [math.log(b / a) for a, b in zip(cl, cl[1:])]
        out["rv7"] = math.sqrt(sum(x * x for x in rets) / len(rets) * 24 * 365) * 100 if len(rets) > 24 else None
        if out["rv7"] is not None:      # None 을 6시간 붙잡지 않는다(09-30 검증)
            _RV_CACHE[currency] = (time.time(), out["rv7"])
    except _Cached:
        out["rv7"] = hit[1]
    except Exception as exc:
        out["rv7"] = hit[1] if hit and time.time() - hit[0] < 24 * 3600 else None
        log(f"{currency}: rv7 실패 {exc}{' (직전 값 사용)' if out['rv7'] is not None else ''}")
    idx = out["index"] or float(chain["underlying_price"].median())
    exps = []
    for exp, g in sorted(chain.groupby("expiration_ts"), key=lambda kv: kv[0]):
        yrs = float(g["days_to_expiry"].iloc[0]) / 365.0
        fwd = float(g["underlying_price"].iloc[0])
        calls, puts = g[g["option_type"] == "call"], g[g["option_type"] == "put"]
        if calls.empty or puts.empty:
            continue
        atm_k = float(g.iloc[(g["strike"] - fwd).abs().argsort()[:1]]["strike"].iloc[0])
        atm_iv = float(g[g["strike"] == atm_k]["mark_iv"].mean())
        cd = calls.assign(d=[_bs_delta(fwd, k, v, yrs, True) for k, v in zip(calls["strike"], calls["mark_iv"])])
        pdl = puts.assign(d=[_bs_delta(fwd, k, v, yrs, False) for k, v in zip(puts["strike"], puts["mark_iv"])])
        c25 = float(cd.iloc[(cd["d"] - 0.25).abs().argsort()[:1]]["mark_iv"].iloc[0])
        p25 = float(pdl.iloc[(pdl["d"] + 0.25).abs().argsort()[:1]]["mark_iv"].iloc[0])
        ks = sorted(g["strike"].unique())
        pain = min(ks, key=lambda P: float(((P - calls["strike"]).clip(lower=0) * calls["open_interest"]).sum()
                                          + ((puts["strike"] - P).clip(lower=0) * puts["open_interest"]).sum()))
        coi, poi = float(calls["open_interest"].sum()), float(puts["open_interest"].sum())
        exps.append({"exp_ms": int(exp.timestamp() * 1000), "atm_iv": atm_iv, "rr25": c25 - p25,
                     "bf25": (c25 + p25) / 2 - atm_iv, "call_oi_usd": coi * idx, "put_oi_usd": poi * idx,
                     "pc": (poi / coi) if coi else None, "pain": float(pain), **_skew_interp(g, fwd, yrs),
                     "fwd": fwd, "atm_oi_usd": float(g.loc[(g["strike"] / fwd - 1).abs() <= 0.02, "open_interest"].sum()) * idx,
                     "_g": g, "_yrs": yrs})
    # 2026-10-01 표기: 만기별 미결제가 전체의 몇 %인가 -- max pain 선(늘 가까운 만기)이 보통 2% 남짓의 계약만 반영한다는 걸 보이려고.
    tot_oi = sum(e["call_oi_usd"] + e["put_oi_usd"] for e in exps)
    for e in exps:
        e["oi_share"] = (e["call_oi_usd"] + e["put_oi_usd"]) / tot_oi if tot_oi else None
    # 2026-10-01 «옵션 정보»(문헌 조사 ②): 가까운 만기(남은 3시간 미만이면 다음 만기)와 7·30일에 가장 가까운 만기의 표면 통계 ·
    #   ATM 호가 폭(7일 근처) · 만기 ATM 미결제. 화면은 전부 «검정 전 · 서술».
    live = [e for e in exps if e["hours"] >= 3]
    near = lambda d: min(live, key=lambda e: abs(e["hours"] - d * 24), default=None)   # noqa: E731
    pick = {"front": live[0] if live else None, "7": near(7), "30": near(30)}
    out["surface"] = {k: ({"exp_ms": e["exp_ms"], "hours": round(e["hours"], 2), "fwd": e["fwd"], **_surface_stats(e["_g"], e["fwd"], e["_yrs"])}
                          if e else None) for k, e in pick.items()}
    e7 = pick["7"]
    out["atm_spread_iv"] = _atm_spread_iv(e7["_g"], e7["fwd"], e7["_yrs"], not inverse) if e7 else None
    out["front_atm_oi_usd"] = exps[0]["atm_oi_usd"] if exps else None
    for e in exps:
        e.pop("_g", None); e.pop("_yrs", None)
    out["expiries"] = exps[:8]
    # 2026-10-01 연구(research_eth_option_metric_ambiguity_20260930): 가까운 만기(늘 24시간 미만)의 최근접 행사가 RR·BF 는
    #   실제 델타가 0.15~0.31 이고 1시간 변화 SD 3.6pt(61% 가 1pt 넘게 흔들리고 되돌림) = 잡음. 화면은 고정만기(7·30일)
    #   델타 보간 값을 쓴다(7일 SD 0.58pt). expiries[*].rr25/bf25(최근접)는 이력 호환으로 남긴다.
    out["cm"] = {str(d): _const_maturity(exps, d) for d in (1, 7, 30, 60)}   # 1일 = 기울기(1일−7일) · 60일 = 기간 구조 셋째 점
    # 30일에 가장 가까운 만기의 ATM IV -- DVOL(30일 내재 변동성 지수)이 없는 코인의 대용. 화면은 dvol ?? iv30.
    near30 = min(exps, key=lambda e: abs(e["exp_ms"] / 1000 - time.time() - 30 * 86400), default=None)
    out["iv30"] = near30["atm_iv"] if near30 else None
    # 2026-10-02 딜러 감마(곡선·플립·DEX·charm·gamma_by) 제거 -- 사다리 «가까운 만기» 정의(아직 안 끝난 가장 빠른 만기)만 남긴다.
    g_fut = chain[chain["expiration_ts"] > pd.Timestamp.now(tz="UTC")]
    g_first = g_fut["expiration_ts"].min() if len(g_fut) else None
    # 보험: 가까운 만기 둘(12시간 이상 남은 것 중)의 지수 ±10% OTM 옵션 -- 화면이 포지션·손절에 맞춰 고른다.
    near = [e for e in exps if e["exp_ms"] / 1000 - time.time() > 12 * 3600][:2]
    hedge = []
    for e in near:
        g = chain[chain["expiration_ts"] == pd.Timestamp(e["exp_ms"], unit="ms", tz="UTC")]
        for _, r in g.iterrows():
            otm = (r["option_type"] == "put" and idx * 0.9 <= r["strike"] <= idx) or \
                  (r["option_type"] == "call" and idx <= r["strike"] <= idx * 1.1)
            if otm and r["mark_price"] > 0:
                hedge.append({"exp_ms": e["exp_ms"], "k": float(r["strike"]), "type": r["option_type"][0].upper(),
                              "usd": float(r["mark_price"]) * (idx if inverse else 1.0), "iv": float(r["mark_iv"])})
    out["hedge"] = hedge
    # 2026-09-28 행사가 사다리(사용자 선택 1-B): 지수 ±8% 행사가별 콜·풋 미결제(USD). 범위 셋 -- front(가장 가까운 만기) ·
    #   week(7일 안 만기 합) · all(전 만기). 행 = [행사가, 콜$, 풋$] (2026-10-02 감마 칸 제거).
    #   🔴«행사가 자석»은 검정에서 기각 -- 화면은 «어디에 계약이 쌓였나»로만 쓴다.
    band = chain[(chain["strike"] >= idx * 0.92) & (chain["strike"] <= idx * 1.08)]
    first = g_first
    usd_oi = band["open_interest"] * idx
    band = band.assign(c=usd_oi.where(band["option_type"] == "call", 0.0), p=usd_oi.where(band["option_type"] == "put", 0.0))
    def _ladder(sel):
        agg = band[sel].groupby("strike")[["c", "p"]].sum().reset_index()
        return [[float(r.strike), round(float(r.c)), round(float(r.p))] for r in agg.itertuples()]
    out["strikes"] = {"front": _ladder(band["expiration_ts"] == first) if first is not None else [],
                      "week": _ladder(band["days_to_expiry"] <= 7), "all": _ladder(band["days_to_expiry"] > 0),
                      "front_exp_ms": int(first.timestamp() * 1000) if first is not None else None}
    return out




OTHER_VENUE_COINS = ("ETH", "BTC")
_OTHER_HDR = {"User-Agent": "Mozilla/5.0"}       # OKX 는 UA 없으면 403


def parse_other(venue: str, name: str) -> tuple[str, float, datetime] | None:
    """(콜/풋, 행사가, 만기 08:00 UTC). OKX `ETH-USD-261004-2700-C` · Bybit `ETH-4OCT26-2700-C-USDT`(접미사 있어도 됨)."""
    if venue == "OKX":
        m = re.match(r"^[A-Z]+-USD-(\d{6})-(\d+(?:\.\d+)?)-([CP])$", name)
        fmt = "%y%m%d"
    else:
        m = re.match(r"^[A-Z]+-(\d{1,2}[A-Z]{3}\d{2})-(\d+(?:\.\d+)?)-([CP])(?:-[A-Z]+)?$", name)
        fmt = "%d%b%y"
    if not m:
        return None
    exp = datetime.strptime(m.group(1), fmt).replace(hour=8, tzinfo=timezone.utc)
    return ("call" if m.group(3) == "C" else "put"), float(m.group(2)), exp


def fetch_other_venues(currency: str, now: datetime) -> list[tuple]:
    """OKX·Bybit 종목별 미결제(기초자산 수량)·mark IV(%)·24h 거래량. 실패한 거래소는 빈 채로 넘어간다."""
    out: list[tuple] = []
    try:
        oi = requests.get("https://www.okx.com/api/v5/public/open-interest", headers=_OTHER_HDR, timeout=20,
                          params={"instType": "OPTION", "instFamily": f"{currency}-USD"}).json()["data"]
        iv = {x["instId"]: float(x["markVol"]) * 100 for x in requests.get(
            "https://www.okx.com/api/v5/public/opt-summary", headers=_OTHER_HDR, timeout=20,
            params={"instFamily": f"{currency}-USD"}).json()["data"] if x.get("markVol")}
        for x in oi:
            sp = parse_other("OKX", x["instId"])
            if sp and sp[2] > now and float(x["oiCcy"]) > 0:   # OKX 는 끝난 만기(미결제 0)까지 돌려준다 -- ETH 10-03 실측 1,070종목 · 미결제 0 은 max pain 에 무관
                out.append((now, "OKX", currency, x["instId"], sp[0], sp[1], sp[2], float(x["oiCcy"]), iv.get(x["instId"]), None))
    except Exception as exc:
        log(f"{currency}: OKX 옵션 실패 {exc}")
    try:
        for x in requests.get("https://api.bybit.com/v5/market/tickers", headers=_OTHER_HDR, timeout=20,
                              params={"category": "option", "baseCoin": currency}).json()["result"]["list"]:
            sp = parse_other("Bybit", x["symbol"])
            if sp and float(x["openInterest"]) > 0:
                out.append((now, "Bybit", currency, x["symbol"], sp[0], sp[1], sp[2], float(x["openInterest"]),
                            float(x["markIv"]) * 100 if x.get("markIv") else None, float(x.get("volume24h") or 0)))
    except Exception as exc:
        log(f"{currency}: Bybit 옵션 실패 {exc}")
    return out


def poll_once(con) -> None:
    now = datetime.now(timezone.utc)
    for currency in OTHER_VENUE_COINS:
        rows = fetch_other_venues(currency, now)
        if rows:
            con.register("oth_df", pd.DataFrame(rows, columns=["recorded_at_utc", "venue", "currency", "instrument_name", "option_type",
                                                              "strike", "expiration_ts", "open_interest", "mark_iv", "volume_24h"]))
            con.execute("INSERT INTO option_oi_other SELECT * FROM oth_df")      # 한 번에(executemany 는 행마다 커밋 -- 09-23 사고)
            con.unregister("oth_df")
            log(f"{currency}: OKX·Bybit 옵션 {len(rows)}종목")
    for currency in CURRENCIES:
        try:
            chain = fetch_chain(currency)
        except Exception as exc:   # 한 코인 조회 실패가 나머지 코인·상태 파일을 막지 않게(09-30 검증)
            log(f"{currency}: 체인 조회 실패 {exc}")
            continue
        if chain.empty:
            log(f"{currency}: empty response, skipping")
            continue
        con.register("chain_df", chain)
        con.execute("INSERT INTO option_chain_snapshot SELECT recorded_at_utc, currency, instrument_name, option_type, strike, "
                    "expiration_ts, days_to_expiry, open_interest, mark_iv, underlying_price, mark_price, volume, gamma_bs FROM chain_df")
        con.unregister("chain_df")
        try:
            opt = options_summary(chain, currency)
            f0 = (opt["expiries"] or [{}])[0]
            # gamma_flip 열은 2026-10-02 부터 NULL(딜러 감마 제거 · 표 스키마는 그대로)
            con.execute("INSERT INTO option_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [chain["recorded_at_utc"].iloc[0], currency, opt.get("index"), opt.get("dvol"), opt.get("rv7"),
                         f0.get("atm_iv"), f0.get("rr25"), f0.get("bf25"), None, json.dumps(opt)])
            log(f"{currency}: index={opt.get('index')} n={len(chain)}")
        except Exception as exc:     # 옵션 요약이 깨져도 체인 스냅샷은 이미 저장됐다
            log(f"{currency}: options_summary 실패 {exc}")
    write_state(con)


def write_state(con) -> None:
    """화면용 상태파일. 🔴여기서 쓰는 이유는 **이 프로세스가 이미 연결을 쥐고 있어서**다 --
    대시보드가 제 연결을 열면 락에 걸린다. 2026-10-02 GEX 이력(gex_summary)·dex_1h_ago 제거 -- 옵션 요약만 싣는다."""
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "currencies": {}}
    for currency in CURRENCIES:
        opt_row = con.execute("SELECT recorded_at_utc, index_price, payload FROM option_summary WHERE currency = ? "
                              "ORDER BY recorded_at_utc DESC LIMIT 1", [currency]).fetchone()
        if not opt_row:
            continue
        ts, spot = opt_row[0], opt_row[1]
        opt_row = (opt_row[2],)
        # 2026-10-01 «옵션 정보»: VoV = 내재 변동성(DVOL, 없으면 iv30)의 시간별 로그 변화 표준편차(지난 24시간, %) ·
        #   만기 ATM 미결제 분위 = 지금 값이 지난 30일 매일 07:00~07:10 UTC(정산 1시간 전) 값 중 몇 분위인가(Weiss 외 2026 의 «ATM 미결제 상위 날»).
        oh: dict = {"vov24": None, "atm_oi_pct": None, "atm_oi_n": 0}
        try:
            iv = con.execute("""SELECT date_trunc('hour', recorded_at_utc) h,
                                         arg_max(coalesce(dvol, TRY_CAST(json_extract_string(payload, '$.iv30') AS DOUBLE)), recorded_at_utc)
                                  FROM option_summary WHERE currency = ? AND recorded_at_utc >= now() - INTERVAL 25 HOUR GROUP BY 1 ORDER BY 1""",
                             [currency]).fetchall()
            v = [x[1] for x in iv if x[1] and x[1] > 0]
            if len(v) >= 12:
                d = [math.log(b2 / a2) for a2, b2 in zip(v, v[1:])]
                m = sum(d) / len(d)
                oh["vov24"] = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1)) * 100
            hist = [x[0] for x in con.execute("""SELECT TRY_CAST(json_extract_string(payload, '$.front_atm_oi_usd') AS DOUBLE) FROM option_summary
                                                 WHERE currency = ? AND recorded_at_utc >= now() - INTERVAL 30 DAY
                                                   AND hour(timezone('UTC', recorded_at_utc)) = 7 AND minute(timezone('UTC', recorded_at_utc)) < 10""", [currency]).fetchall()
                    if x[0] is not None]
            cur_oi = (json.loads(opt_row[0]) or {}).get("front_atm_oi_usd") if opt_row else None
            if cur_oi is not None and len(hist) >= 5:
                oh.update(atm_oi_pct=sum(1 for x in hist if x <= cur_oi) / len(hist), atm_oi_n=len(hist))
        except Exception as exc:   # 부가 값 -- 없으면 None
            log(f"{currency}: 옵션 이력 지표 생략 {exc}")
        opts = json.loads(opt_row[0])
        out["currencies"][currency] = {
            "opt_hist": oh,
            "options": opts,
            "recorded_at_utc": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "spot_price": spot,
        }
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)
    log(f"state -> {STATE_PATH.name} ({len(out['currencies'])} currencies)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-sec", type=int, default=0, help="0 = single poll and exit (default, for cron)")
    args = ap.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = connect_retry(DB_PATH)
    ensure_tables(con)

    if args.interval_sec <= 0:
        poll_once(con)
    else:
        while True:
            poll_once(con)
            time.sleep(args.interval_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
