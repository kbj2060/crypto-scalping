#!/usr/bin/env python3
"""사전등록 판정: 뒤집은 관행 GEX → 다음 4h ETH 실현변동성 (2026-10-01).
정의·판정 규칙 원문 = docs/experiments/eth_reverse_conventional_gex_prereg_20261001.md (이 스크립트가 그 정의를 그대로 코드로 옮긴다).

  X  = 뒤집은 관행 GEX$ (딜러 = 콜 매도·풋 매수 → 가중치 콜 −OI · 풋 +OI) = Σ −sg·OI·γ(F,K,mark_iv,T)·F²·1%.
       범위 week(만기−t ≤ 7일, 1순위) · front(가장 가까운 만기) · all. F = underlying_price. t = 매 UTC 정시 첫 체인 스냅샷.
  y  = log 실현변동성, ETH-PERPETUAL 5분봉 open ≥ t 인 봉부터 48개(4h).
  통제 = log RV(닫힌 봉 직전 1h·24h·7d) + log DVOL(t 직전 닫힌 1h 봉) + UTC 시 더미 + 상수. X 는 표본 안 백분위 순위.
  CI = 일 블록 부트스트랩 500회 · 시드 3 · 95% 백분위.  판정 = 창 전체 β<0 ∧ CI 상한<0 (통제판).
  함수는 research_eth_dealer_gex_reconstruct_2026_20261001(R) 의 exposures·Bars·beta_test·halves·positions_at 을 그대로 쓴다.

데이터
  체인: 서버 data/live/deribit_options.duckdb option_chain_snapshot -- 이 스크립트가 handoff.sh 로 자동 추출한다
        (launch: read_only 로 열어 필요한 열·정시 첫 스냅샷만 COPY → 서버 tmp parquet · logs 로 완료 확인 · pull).
        수동: bash scripts/ops/handoff.sh launch server <job> -- python -c "<SQL>" ; ... logs server <job> ; ... pull server <path>
  가격·DVOL: www.deribit.com 공개 API(get_tradingview_chart_data ETH-PERPETUAL 5분 · get_volatility_index_data ETH 1h). 🔴바이낸스는 안 부른다.
  대조(판정 안 씀) 체결 기반 딜러 GEX: 연구 캐시 체결 + history.deribit.com 꼬리. 캐시가 없으면 «unavailable» 로 적고 넘어간다.

사용
  python scripts/prereg_eth_reverse_conventional_gex_eval_20261001.py --start 2026-10-01 --end 2026-11-10   # 표본 밖 판정(11-10 05:00 UTC 이후)
  python scripts/prereg_eth_reverse_conventional_gex_eval_20261001.py --start 2026-08-15 --end 2026-09-30T14:30   # 표본 안 재현(판정 아님)
  python scripts/prereg_eth_reverse_conventional_gex_eval_20261001.py --selftest
출력 tmp/prereg_reverse_conventional_gex_20261001/ : snap_*.parquet · hourly_*.parquet · result_*.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_eth_dealer_gex_reconstruct_2026_20261001 as R  # noqa: E402
import research_eth_dealer_position_accuracy_20260930 as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REL = "tmp/prereg_reverse_conventional_gex_20261001"
OUT = ROOT / REL
H_MS, B_MS, DAY = R.H_MS, R.B_MS, 86_400_000
OOS_START, OOS_END = pd.Timestamp("2026-10-01", tz="UTC"), pd.Timestamp("2026-11-10", tz="UTC")   # 문서 §4 -- 바꾸면 새 사전등록
COLS = "recorded_at_utc, instrument_name, option_type, strike, expiration_ts, open_interest, mark_iv, underlying_price"


def ms(t: pd.Timestamp) -> int:
    return int(t.value // 10**6)


# ───────────────────────── 받기 ─────────────────────────
def pull_snap(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    tag = f"{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}"
    rel = f"{REL}/snap_{tag}.parquet"
    if not (ROOT / rel).exists():
        sql = (f"COPY (SELECT {COLS} FROM option_chain_snapshot WHERE currency='ETH' AND instrument_name LIKE 'ETH-%' AND recorded_at_utc IN "
               f"(SELECT min(recorded_at_utc) FROM option_chain_snapshot WHERE currency='ETH' AND recorded_at_utc >= TIMESTAMPTZ '{start.isoformat()}' "
               f"AND recorded_at_utc < TIMESTAMPTZ '{end.isoformat()}' GROUP BY date_trunc('hour', recorded_at_utc))) TO '{rel}' (FORMAT parquet)")
        # 상주 수집기 파일: read_only 로 짧게. 수집기가 쓰는 중이라 열기가 막히면 3초 간격 재시도.
        code = ("import duckdb,os,time\nos.makedirs('%s',exist_ok=True)\nfor i in range(40):\n try:\n  c=duckdb.connect('data/live/deribit_options.duckdb',read_only=True);break\n"
                " except duckdb.IOException:\n  time.sleep(3)\nc.execute(\"SET TimeZone='UTC'\");c.execute(%r);c.close();print('SNAP_DONE')\n") % (REL, sql)
        job = f"rcgex_{tag}"
        ho = ["bash", str(ROOT / "scripts/ops/handoff.sh")]
        subprocess.run(ho + ["launch", "server", job, "--", "python", "-c", code], check=True)
        for _ in range(30):                      # 10초 간격(handoff 폭주 방지 상한 30회/분 안)
            time.sleep(10)
            log = subprocess.run(ho + ["logs", "server", job], capture_output=True, text=True).stdout
            if "SNAP_DONE" in log or "Traceback" in log:
                break
        if "SNAP_DONE" not in log:
            sys.exit(f"서버 추출 실패/시간초과:\n{log}")
        subprocess.run(ho + ["pull", "server", rel], check=True)
    s = pd.read_parquet(ROOT / rel)
    s["ts"] = s["recorded_at_utc"].astype("datetime64[us, UTC]").astype("int64") // 1000
    s["exp_ms"] = s["expiration_ts"].astype("datetime64[us, UTC]").astype("int64") // 1000
    return s


def fetch_bars(t0: int, t1: int):
    """ETH-PERPETUAL 5분봉 · DVOL 1h, [t0, t1) 중 지금 닫힌 봉만."""
    now = int(time.time() * 1000); t1 = min(t1, now)
    bars, a = [], t0
    while a < t1:
        b = min(a + 14 * DAY, t1)
        r = D._get(D.LIVE, "get_tradingview_chart_data", instrument_name="ETH-PERPETUAL", start_timestamp=a, end_timestamp=b, resolution="5")
        bars.append(pd.DataFrame({k: r[k] for k in ("ticks", "close", "cost")})); a = b
    bars = pd.concat(bars).drop_duplicates("ticks").sort_values("ticks")
    rows, a = [], t0
    while a < t1:
        b = min(a + 30 * DAY, t1)
        rows += D._get(D.LIVE, "get_volatility_index_data", currency="ETH", resolution="3600", start_timestamp=a, end_timestamp=b)["data"]; a = b
    dv = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"]).drop_duplicates("ts").sort_values("ts")
    return bars[bars["ticks"] + B_MS <= now], dv[dv["ts"] + H_MS <= now]


def load_trades(end_ms: int) -> pd.DataFrame:
    """연구 캐시(2026 전 종목 seq 1 부터 ~10-01) + history.deribit.com 꼬리(캐시 끝 ~ end)."""
    tr = R.load_trades(end_ms)
    t0 = int(tr["timestamp"].max())
    if end_ms <= t0:
        return tr
    p = OUT / f"trades_tail_{t0}_{end_ms}.parquet"
    raw = pd.read_parquet(p) if p.exists() else D.fetch_paged(
        p, "get_last_trades_by_currency_and_time", {"currency": "ETH", "kind": "option"}, t0, end_ms)
    raw = raw[~raw["trade_id"].isin(tr["trade_id"])]
    pr = raw["instrument_name"].map(D.parse); raw = raw[pr.notna()].copy(); pr = pr[pr.notna()]
    raw["K"] = [x[0] for x in pr]; raw["exp_ms"] = [int(x[1].timestamp() * 1000) for x in pr]; raw["sg"] = [x[2] for x in pr]
    raw["q"] = np.where(raw["direction"] == "buy", 1.0, -1.0) * raw["amount"]; raw["iv"] = raw["iv"].where(raw["iv"] > 0)
    return pd.concat([tr, raw[tr.columns]]).sort_values(["instrument_name", "timestamp", "trade_seq"]).reset_index(drop=True)


# ───────────────────────── 계산 ─────────────────────────
def hourly(snap: pd.DataFrame, B: R.Bars) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """시각 = UTC 정시별 첫 스냅샷. 행 = 그 스냅샷의 아직 안 끝난 종목. 반환 (시간별 표, 종목 행, 시각 배열)."""
    st = np.sort(snap["ts"].unique()); s_eval = pd.Series(st).groupby(st // H_MS).min().to_numpy()
    sn = snap[snap["ts"].isin(s_eval) & (snap["exp_ms"] > snap["ts"])].rename(columns={"instrument_name": "inst"}).copy()
    sn["k"] = np.searchsorted(s_eval, sn["ts"].to_numpy()); sn["K"] = sn["strike"]
    sn["sg"] = np.where(sn["option_type"] == "call", 1.0, -1.0)
    L = B.at(s_eval); S = L["S"].to_numpy()
    parts = [pd.DataFrame({"ts": s_eval}), L]
    for nm, w in (("rev", -sn["sg"] * sn["open_interest"]), ("conv", sn["sg"] * sn["open_interest"])):
        E = R.exposures(sn.assign(w=w), s_eval, S, "mark_iv", fwd_col="underlying_price")
        parts.append(E[[f"gex_{sc}" for sc in R.SCOPES]].add_prefix(f"{nm}_"))
    return pd.concat(parts, axis=1), sn, s_eval


def dealer_gex(sn: pd.DataFrame, s_eval: np.ndarray, S: np.ndarray, end_ms: int) -> tuple[pd.DataFrame, dict]:
    """대조: 체결 기반 딜러 GEX(딜러 = 테이커 반대편, seq 1 부터 끊김 없는 종목만) · IV·선도가는 스냅샷 값(연구 H3 와 같다)."""
    P = R.positions_at(load_trades(end_ms), s_eval)
    P = P.merge(sn[["k", "inst", "mark_iv", "underlying_price"]], on=["k", "inst"], how="inner")
    P["w"] = np.where(P["cov"], P["pos"], 0.0)
    E = R.exposures(P, s_eval, S, "mark_iv", fwd_col="underlying_price")
    c = sn.merge(P[["k", "inst", "cov"]], on=["k", "inst"], how="left")
    c["oc"] = np.where(c["cov"].astype("boolean").fillna(False).to_numpy(bool), c["open_interest"], 0.0)
    g = c.groupby("k")[["oc", "open_interest"]].sum(); r = g["oc"] / g["open_interest"].clip(lower=1e-9)
    return E[[f"gex_{sc}" for sc in R.SCOPES]].add_prefix("dealer_"), {"oi_weighted_coverage_min": round(float(r.min()), 4),
                                                                         "oi_weighted_coverage_mean": round(float(r.mean()), 4)}


def test(H: pd.DataFrame, col: str) -> dict:
    mid = int(np.median(H["ts"]))
    full = {"ctrl": R.beta_test(H, col, "y4", False), "dayfe": R.beta_test(H, col, "y4", True)}
    halves = R.halves(H, col, "y4", split=mid)
    return {"full": full, **halves, "split_utc": str(pd.Timestamp(mid, unit="ms", tz="UTC"))}


def main(start: pd.Timestamp, end: pd.Timestamp):
    now = pd.Timestamp.now(tz="UTC")
    if end > OOS_START and now < OOS_END + pd.Timedelta(hours=5):
        sys.exit(f"표본 밖 창({OOS_START:%m-%d}~{OOS_END:%m-%d})은 {OOS_END + pd.Timedelta(hours=5)} 전에는 보지 않는다(사전등록 §4).")
    OUT.mkdir(parents=True, exist_ok=True)
    snap = pull_snap(start, end)
    bars, dv = fetch_bars(ms(start) - 8 * DAY, ms(end) + 5 * H_MS)
    B = R.Bars(bars, dv)
    H, sn, s_eval = hourly(snap, B)
    res = {"window": [str(start), str(end)], "run_utc": str(now), "n_hours": len(H),
           "expected_hours": int((end - start) / pd.Timedelta(hours=1)), "bars_missing": B.missing}
    try:
        Ed, cov = dealer_gex(sn, s_eval, H["S"].to_numpy(), ms(end))
        H = pd.concat([H, Ed], axis=1); res["dealer_coverage"] = cov
    except Exception as e:   # 대조군 실패가 판정을 막지 않는다
        res["dealer_coverage"] = f"unavailable: {e!r}"
    H.to_parquet(OUT / f"hourly_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.parquet")
    for c in [c for c in H.columns if c.startswith(("rev_", "conv_", "dealer_"))]:
        res[c] = test(H, c)
    p = res["rev_gex_week"]["full"]["ctrl"]
    res["primary"] = {"x": "rev_gex_week", **p, "pass": "ci" in p and p["beta"] < 0 and p["ci"][1] < 0,
                      "halves_sign_agree_negative": all(res["rev_gex_week"][h]["ctrl"].get("beta", 0) < 0 for h in ("H_a", "H_b"))}
    (OUT / f"result_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for c in [c for c in res if c.startswith(("rev_", "conv_", "dealer_gex"))]:
        r = res[c]; f = r["full"]["ctrl"]
        print(f"{c:18s} 전체 β {f.get('beta')} {f.get('ci')} n={f.get('n')} 일={f.get('days')} | "
              f"전반 {r['H_a']['ctrl'].get('beta')} {r['H_a']['ctrl'].get('ci')} · 후반 {r['H_b']['ctrl'].get('beta')} {r['H_b']['ctrl'].get('ci')} | "
              f"일FE {r['full']['dayfe'].get('beta')} {r['full']['dayfe'].get('ci')}")
    print("대조 커버", res["dealer_coverage"])
    print("판정(1순위 rev_gex_week 전체 통제판):", json.dumps(res["primary"], ensure_ascii=False))


# ───────────────────────── 자체점검 ─────────────────────────
def selftest():
    h = H_MS; t = 10 * h + 2                       # 정시 +2ms 스냅샷
    mk = lambda ts, inst, typ, exp_d, oi: dict(ts=ts, instrument_name=inst, option_type=typ, strike=100.0, exp_ms=10 * h + int(exp_d * DAY),
                                               open_interest=oi, mark_iv=50.0, underlying_price=100.0)
    snap = pd.DataFrame([mk(t, "C3", "call", 3, 10.0), mk(t, "P3", "put", 3, 4.0), mk(t, "P1", "put", 1, 2.0), mk(t, "C30", "call", 30, 50.0),
                         mk(t, "OLD", "call", -1, 99.0),                       # 이미 만기 → 빠져야
                         mk(t + 1800_000, "C3", "call", 3, 1000.0)])           # 같은 시간 둘째 스냅샷 → 안 써야
    ticks = np.arange(0, 20 * h, B_MS)
    close = np.full(len(ticks), 100.0)
    close[ticks >= 10 * h] = 101.0                  # 봉 [10h, 10h+5m) 에서만 점프 = t 를 품은 봉 → 과거도 라벨도 아님
    Bx = R.Bars(pd.DataFrame({"ticks": ticks, "close": close, "cost": 1.0}), pd.DataFrame({"ts": [9 * h], "c": [50.0]}))
    H, sn, s_eval = hourly(snap, Bx)
    assert list(s_eval) == [t] and "OLD" not in set(sn["inst"]) and len(sn) == 4
    g = lambda c: H[c].iat[0]
    # 부호: 뒤집은 가중치 = 콜 −OI · 풋 +OI. 3일물 콜10·풋4 → week 음, 1일물 풋만 → front 양, 관행은 정확히 반대
    assert g("rev_gex_front") > 0 and g("rev_gex_week") < 0 and g("rev_gex_all") < g("rev_gex_week")
    for sc in R.SCOPES:
        assert abs(g(f"rev_gex_{sc}") + g(f"conv_gex_{sc}")) < 1e-9
    # 범위: week = 3일물+1일물(30일물 제외) -- front 는 1일물 하나
    one = R.exposures(sn[sn["inst"] == "P1"].assign(w=2.0), s_eval, np.array([100.0]), "mark_iv", fwd_col="underlying_price")["gex_all"][0]
    assert abs(g("rev_gex_front") - one) < 1e-9
    # 라벨: t 를 품은 봉의 점프는 y4 에도 p1 에도 없다 → 둘 다 ≈ log(1e-6)
    assert g("y4") < -13 and g("p1") < -13
    Bx2 = R.Bars(pd.DataFrame({"ticks": ticks, "close": np.where(ticks >= 10 * h + B_MS, 101.0, 100.0), "cost": 1.0}), pd.DataFrame({"ts": [9 * h], "c": [50.0]}))
    assert hourly(snap, Bx2)[0]["y4"].iat[0] > -6     # 다음 봉(open ≥ t) 점프는 라벨에 들어간다
    R.selftest()
    print("selftest OK (prereg)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else main(pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC"))
