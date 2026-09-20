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
STATE_HISTORY = 48        # 48시간 스트립. 매시 1행이라 그대로 48개다
BASE_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
CURRENCIES = ("ETH", "BTC")
FRONT_MONTH_DAYS = 30.0
INSTRUMENT_RE = re.compile(r"^(?P<ccy>[A-Z]+)-(?P<exp>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<type>[CP])$")


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
    con.execute(
        """CREATE TABLE IF NOT EXISTS gex_summary (
            recorded_at_utc TIMESTAMPTZ, currency VARCHAR, spot_price DOUBLE,
            total_gex_usd DOUBLE, front_month_gex_usd DOUBLE, n_instruments INTEGER,
            n_front_month INTEGER
        )"""
    )


def _parse_instrument(name: str) -> dict | None:
    m = INSTRUMENT_RE.match(name)
    if not m:
        return None
    exp_dt = datetime.strptime(m.group("exp"), "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
    return {
        "strike": float(m.group("strike")),
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


def fetch_chain(currency: str) -> pd.DataFrame:
    r = requests.get(BASE_URL, params={"currency": currency, "kind": "option"}, timeout=20)
    r.raise_for_status()
    rows = r.json().get("result", [])
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
        })
    return pd.DataFrame(out)


def summarize_gex(chain: pd.DataFrame, currency: str) -> dict:
    if chain.empty:
        return {"recorded_at_utc": datetime.now(timezone.utc), "currency": currency, "spot_price": None,
                "total_gex_usd": None, "front_month_gex_usd": None, "n_instruments": 0, "n_front_month": 0}
    spot = float(chain["underlying_price"].iloc[0])
    sign = chain["option_type"].map({"call": 1.0, "put": -1.0})
    contrib = sign * chain["gamma_bs"] * chain["open_interest"] * (spot ** 2) * 0.01
    front = chain["days_to_expiry"] <= FRONT_MONTH_DAYS
    return {
        "recorded_at_utc": chain["recorded_at_utc"].iloc[0], "currency": currency, "spot_price": spot,
        "total_gex_usd": float(contrib.sum()), "front_month_gex_usd": float(contrib[front].sum()),
        "n_instruments": int(len(chain)), "n_front_month": int(front.sum()),
    }


def poll_once(con) -> None:
    for currency in CURRENCIES:
        chain = fetch_chain(currency)
        if chain.empty:
            log(f"{currency}: empty response, skipping")
            continue
        con.register("chain_df", chain)
        con.execute("INSERT INTO option_chain_snapshot SELECT * FROM chain_df")
        con.unregister("chain_df")
        summary = summarize_gex(chain, currency)
        con.execute(
            "INSERT INTO gex_summary VALUES (?, ?, ?, ?, ?, ?, ?)",
            [summary["recorded_at_utc"], summary["currency"], summary["spot_price"],
             summary["total_gex_usd"], summary["front_month_gex_usd"],
             summary["n_instruments"], summary["n_front_month"]],
        )
        log(f"{currency}: spot={summary['spot_price']:.1f} total_gex=${summary['total_gex_usd']:,.0f} "
            f"front_month_gex=${summary['front_month_gex_usd']:,.0f} n={summary['n_instruments']} "
            f"n_front={summary['n_front_month']}")
    write_state(con)


def write_state(con) -> None:
    """화면용 상태파일. 🔴여기서 쓰는 이유는 **이 프로세스가 이미 연결을 쥐고 있어서**다 --
    대시보드가 제 연결을 열면 락에 걸린다.

    두 축을 따로 낸다(eth_gamma_zomma_graphic_pinning_gex_20260916):
      level  = total_gex_usd      -- 이론과 부호가 **반대**다(rho(GEX, 전방RV) +0.44~+0.51).
                                     명목 달러라 «옵션시장 활동 = 변동성»의 결과 대리변수다.
      struct = front / total      -- total 을 통제하면 front 가 이론 부호를 회복한다(t -6.22).
    판정일(1h 2026-09-28 / 4h 10-17)까지는 **참고 표시 전용**이고 신호가 아니다.
    """
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "currencies": {}}
    for currency in CURRENCIES:
        rows = con.execute(
            "SELECT recorded_at_utc, spot_price, total_gex_usd, front_month_gex_usd "
            "FROM gex_summary WHERE currency = ? ORDER BY recorded_at_utc DESC LIMIT ?",
            [currency, STATE_HISTORY],
        ).fetchall()
        if not rows:
            continue
        rows = rows[::-1]                      # 오래된 것부터 -- 스트립이 왼쪽에서 오른쪽으로 흐른다
        ts, spot, total, front = rows[-1]
        # 0 나눗셈과 부호를 같이 막는다. total 이 0 이면 비율에 뜻이 없다(구조를 못 읽는다).
        ratio = (front / total) if total else None
        # 2026-09-20 분위와 «실제로 변하는 축»을 같이 낸다.
        # 🔴`negative_gamma`(= total<0)는 **854 스냅샷 37일 내내 한 번도 참이 아니었다**(0.0%).
        #   화면의 경고 톤이 켜진 적이 없다는 뜻이라 그 칩은 상수였다. 변하는 축은 front(6.7%)이고,
        #   연구가 이론 부호를 회복한다고 말한 축도 front 다(total 통제 후 t -6.22).
        # 🔴달러 절대값은 아무도 보정 못 한다 -- 이 저장소 규약대로 **분위**를 같이 보낸다
        #   (임계값은 달러가 아니라 분위로 선언한다, 2026-09-19).
        pct = con.execute(
            "SELECT avg(CASE WHEN total_gex_usd <= ? THEN 1.0 ELSE 0.0 END), "
            "       avg(CASE WHEN front_month_gex_usd <= ? THEN 1.0 ELSE 0.0 END), "
            "       count(*), count(DISTINCT date_trunc('day', recorded_at_utc)) "
            "FROM gex_summary WHERE currency = ?", [total, front, currency]).fetchone()
        out["currencies"][currency] = {
            "recorded_at_utc": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "spot_price": spot, "total_gex_usd": total, "front_month_gex_usd": front,
            "front_ratio": ratio,
            "total_pct": (float(pct[0]) if pct and pct[0] is not None else None),
            "front_pct": (float(pct[1]) if pct and pct[1] is not None else None),
            "history_n": (int(pct[2]) if pct else 0), "history_days": (int(pct[3]) if pct else 0),
            "front_negative": bool(front is not None and front < 0),
            "negative_gamma": bool(total is not None and total < 0),
            "history": [{"t": (r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0])),
                         "total": r[2], "front": r[3]} for r in rows],
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
