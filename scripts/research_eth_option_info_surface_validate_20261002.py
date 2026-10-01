#!/usr/bin/env python3
"""ETH 옵션 카드 «옵션 정보 · 검정 전» 지표 — «화면이 말하는 의미가 2026 데이터에서 성립하는가» 검증(2026-10-02).
사용자: «수익률은 없어도 돼. 그 데이터가 의미하고 추구하는 방향이 맞는지만 모두 검증». 매매 수익은 재지 않는다.

데이터(옵션은 2026년만): tmp/opt_validate_20261002/
  eth_chain.parquet(Deribit 체인 08-15~10-01, 09-28 까지 1시간·이후 10분 → 정시 최근접 1개로 맞춤) · eth_summary.parquet(10분 payload 09-28~)
  ethusdt_1m.parquet(바이낸스 선물 1분봉 ~09-30) · eth_dvol_1h.parquet
  + 이 스크립트가 data.binance.vision 에서 받는 것: ETHUSDT 지수가 1분봉 · 펀딩비(08~09월)  🔴바이낸스 REST 는 부르지 않는다
  + 매크로 «높음» 일정 재구성(scripts/live_macro_calendar_20260826.compute_macro_calendar, 과거 now 로 3회) → events_high.json
지표 = 수집기(collect_deribit_option_gex_20260815) 함수 그대로(_skew_interp·_surface_stats·_const_maturity),
  만기 목록은 options_summary 376~400행과 같은 식(재현 대조로 확인 — 불일치면 중단). 이벤트 폭 = app.js optEventMove 이식.
시점 계약: 지표는 스냅샷 recorded_at 에 알려짐. 결과는 그 다음 분(분 올림, 같은 분 공유 금지) 봉 시가부터.
정산가 근사: Deribit 정산 = 07:30~08:00 UTC 지수 평균 → 같은 구간 바이낸스 선물 1분 종가 30개 평균.

사전 판정 규칙(결과 보기 전 고정, CRITERIA 에 기록): 주장 방향 부호 + 95% CI 가 0 배제 = 지지 · 반대 부호 + CI 0 배제 = 반대 ·
  CI 0 포함 + 주장 방향 = 검정력 부족(80% 검정력 필요 블록 수 추정) · CI 0 포함 + 반대 방향, 또는 필요 블록 수가 지금의 20배 초과 = 근거 없음.
CI = 블록 부트스트랩 2000회(블록은 항목마다 명시: UTC 일 · 정산 만기 · 7일).

실행: python scripts/research_eth_option_info_surface_validate_20261002.py [--selftest]
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/opt_validate_20261002"
_spec = importlib.util.spec_from_file_location("col", ROOT / "scripts/collect_deribit_option_gex_20260815.py")
col = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(col)

MIN, HR, DAY = 60_000, 3_600_000, 86_400_000
SQ2PI = math.sqrt(2 / math.pi)
NBOOT = 2000
RNG = np.random.default_rng(20261002)

CRITERIA = {
    "rule": "주장 부호 + CI 0 배제 = 지지 · 반대 부호 + CI 0 배제 = 반대 · CI 0 포함·주장 방향 = 검정력 부족(필요 블록 수) · "
            "CI 0 포함·반대 방향 또는 필요 블록 > 20배 = 근거 없음",
    "1_mfiv": "QLIKE(mfiv) − QLIKE(DVOL) < 0 → «DVOL 보다 낫다» 지지 · 크기 비 CI 가 1 포함 → «폭이 맞다»",
    "2_tail": "평균(예측 − 실제 빈도) > 0 → «실제보다 크게» 지지 · Brier 는 DVOL 로그정규 기준선과 나란히(주장 없음, 참고)",
    "3_slope": "ln RV(다음 24h) ~ (1일−7일) + ln DVOL + ln RV(지난 24h) 의 기울기 계수 > 0 → 지지",
    "4_skew_kurt": "ρ(왜도, ln 하방/상방 반분산) < 0 · ρ(왜도, 큰 하락−상승 불균형) < 0 · ρ(첨도, |r|>2σ 빈도) > 0 → 지지",
    "5_settle": "Weiss: ATM 미결제 상위 절반의 07–08 UTC 수익률 − 하위 절반 < 0, 08–09 > 0 · 카드: 상위 절반 |z| − 하위 > 0 → 지지",
    "6_event": "ρ(예상 이벤트 폭, 실제 초과 움직임) > 0 · «들어감» − «거의 안 들어감» 초과 움직임 > 0 → 지지",
    "7_fwd": "ρ(선도−지수, 7일 RR) > 0 → «콜 수요면 +» 지지 · ρ(선도−지수, 펀딩 캐리) > 0 → «캐리» 지지 · 방향 IC 는 0 이 정상",
}


# ───────── 수집기 계산 이식(options_summary 376~417행과 같은 식) ─────────
def exps_of(chain: pd.DataFrame, idx: float) -> list[dict]:
    exps = []
    for exp, g in sorted(chain.groupby("expiration_ts"), key=lambda kv: kv[0]):
        yrs = float(g["days_to_expiry"].iloc[0]) / 365.0
        fwd = float(g["underlying_price"].iloc[0])
        calls, puts = g[g["option_type"] == "call"], g[g["option_type"] == "put"]
        if calls.empty or puts.empty:
            continue
        atm_k = float(g.iloc[(g["strike"] - fwd).abs().argsort()[:1]]["strike"].iloc[0])
        coi, poi = float(calls["open_interest"].sum()), float(puts["open_interest"].sum())
        exps.append({"exp_ms": int(exp.timestamp() * 1000), "atm_iv": float(g[g["strike"] == atm_k]["mark_iv"].mean()),
                     "call_oi_usd": coi * idx, "put_oi_usd": poi * idx, **col._skew_interp(g, fwd, yrs), "fwd": fwd,
                     "atm_oi_usd": float(g.loc[(g["strike"] / fwd - 1).abs() <= 0.02, "open_interest"].sum()) * idx,
                     "_g": g, "_yrs": yrs})
    tot = sum(e["call_oi_usd"] + e["put_oi_usd"] for e in exps)
    for e in exps:
        e["oi_share"] = (e["call_oi_usd"] + e["put_oi_usd"]) / tot if tot else None
    return exps


def card_values(exps: list[dict]) -> dict:
    live = [e for e in exps if e["hours"] >= 3]
    near = lambda d: min(live, key=lambda e: abs(e["hours"] - d * 24), default=None)   # noqa: E731
    pick = {"front": live[0] if live else None, "7": near(7), "30": near(30)}
    return {"surface": {k: ({"exp_ms": e["exp_ms"], "hours": round(e["hours"], 2), "fwd": e["fwd"],
                             **col._surface_stats(e["_g"], e["fwd"], e["_yrs"])} if e else None) for k, e in pick.items()},
            "front_atm_oi_usd": exps[0]["atm_oi_usd"] if exps else None,
            "cm": {str(d): col._const_maturity(exps, d) for d in (1, 7, 30, 60)}}


def event_move(exps8: list[dict], now_ms: int, ev_ms: int):
    """app.js optEventMove 이식. None = «만기 사이에 안 걸림», 그 밖은 이벤트 1σ 폭(%)."""
    ex = []
    for e in exps8:
        a = e["atm_i"] if e.get("atm_i") is not None else e["atm_iv"]
        if e["exp_ms"] > now_ms and a and a > 0:
            ex.append({"T": (e["exp_ms"] - now_ms) / 3.156e10, "v": (a / 100) ** 2, "ms": e["exp_ms"]})
    bi = next((i for i, e in enumerate(ex) if e["ms"] > ev_ms), -1)
    if bi < 0 or bi + 1 >= len(ex):
        return None
    a = ex[bi - 1] if bi > 0 else {"T": 0.0, "v": 0.0}
    b, c = ex[bi], ex[bi + 1]
    normal = (c["v"] * c["T"] - b["v"] * b["T"]) / (c["T"] - b["T"])
    ev = b["v"] * b["T"] - a["v"] * a["T"] - normal * (b["T"] - a["T"])
    return math.sqrt(ev) * 100 if ev > 0 else 0.0


# ───────── 가격 · 시점 ─────────
def first_after(t_ms: int) -> int:
    """스냅샷이 든 분의 다음 분 시작(ms). 지표와 결과가 같은 분을 공유하지 않는다."""
    return (int(t_ms) // MIN + 1) * MIN


class Px:
    def __init__(self, k: pd.DataFrame):
        ot = k["open_time"].to_numpy(np.int64)
        self.m0 = int(ot.min()); n = int((ot.max() - self.m0) // MIN) + 1
        self.o = np.full(n, np.nan); self.c = np.full(n, np.nan)
        i = (ot - self.m0) // MIN
        self.o[i] = k["open"].to_numpy(float); self.c[i] = k["close"].to_numpy(float)
        self.last_close_ms = int(ot.max()) + MIN

    def i(self, ms):
        return int((int(ms) - self.m0) // MIN)

    def open_at(self, ms):          # 그 분 봉의 시가
        j = self.i(ms)
        return self.o[j] if 0 <= j < len(self.o) else np.nan

    def close_at(self, ms):         # ms 에 끝나는 봉의 종가
        j = self.i(ms) - 1
        return self.c[j] if 0 <= j < len(self.c) else np.nan

    def ret(self, a_ms, b_ms):      # a 분 시가 → b 에 끝나는 종가
        return math.log(self.close_at(b_ms) / self.open_at(a_ms)) if b_ms <= self.last_close_ms else np.nan

    def settle(self, exp_ms):       # Deribit 정산 근사: 07:30~08:00 1분 종가 30개 평균
        if exp_ms > self.last_close_ms:
            return np.nan
        j = self.i(exp_ms)
        return float(np.mean(self.c[j - 30:j]))

    def r5(self, start_ms, minutes):   # start 분 시가부터 5분 로그수익
        if start_ms + minutes * MIN > self.last_close_ms:
            return None
        j = self.i(start_ms)
        p = np.concatenate([[self.o[j]], self.c[j + 4:j + minutes:5]])
        return np.diff(np.log(p))

    def r5_past(self, t_ms, minutes):  # t 이전에 끝난 봉만
        end = (int(t_ms) // MIN) * MIN
        j = self.i(end) - 1
        if j - minutes < 0:
            return None
        return np.diff(np.log(self.c[j - minutes:j + 1:5]))


def ann_rv(r, minutes_per):          # 연율 %
    return math.sqrt(np.nansum(r ** 2) / (len(r) * minutes_per) * 525600) * 100


# ───────── 통계 ─────────
def qlike(r, sig):
    """분산 예측 손실(Patton 2011 의 차분 형태): ln σ² + r²/σ². 차이만 쓴다."""
    s2 = np.asarray(sig, float) ** 2
    return np.log(s2) + np.asarray(r, float) ** 2 / s2


def ln_tail(x_pct, s):
    """DVOL 로그정규 기준선: ln(S/F) ~ N(−s²/2, s²) 에서 P(S 가 F(1±x) 밖)."""
    x = x_pct / 100
    return norm.cdf((math.log(1 - x) + s * s / 2) / s) + norm.sf((math.log(1 + x) + s * s / 2) / s)


def spear(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    return float(np.corrcoef(rankdata(x[m]), rankdata(y[m]))[0, 1])


def boot(fn, blk):
    blk = np.asarray(blk)
    u, inv = np.unique(blk, return_inverse=True)
    rows = [np.flatnonzero(inv == j) for j in range(len(u))]
    est = fn(np.arange(len(blk)))
    vals = np.array([fn(np.concatenate([rows[j] for j in RNG.integers(0, len(u), len(u))])) for _ in range(NBOOT)], float)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "n": int(len(blk)), "n_blocks": int(len(u))}


def verdict(r: dict, sign: int) -> dict:
    e, lo, hi = r["est"] * sign, *sorted((r["lo"] * sign, r["hi"] * sign))
    se = (hi - lo) / 3.92
    need = int(math.ceil(r["n_blocks"] * (2.8 * se / abs(e)) ** 2)) if e and se > 0 else None
    if lo > 0:
        v = "지지"
    elif hi < 0:
        v = "반대"
    elif e > 0 and need is not None and need <= 20 * r["n_blocks"]:
        v = "검정력 부족"
    else:
        v = "근거 없음"
    return {**r, "verdict": v, "blocks_needed_80pct": need}


def fmt(r, d=3):
    return f"{r['est']:+.{d}f} [{r['lo']:+.{d}f}, {r['hi']:+.{d}f}] (n={r['n']}, 블록 {r['n_blocks']})"


# ───────── 데이터 ─────────
def _vision(path: str) -> bytes | None:
    r = requests.get(f"https://data.binance.vision/data/{path}", timeout=60)
    return r.content if r.status_code == 200 else None


def _csv(blob: bytes, names: list[str]) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(blob)); raw = z.read(z.namelist()[0]).decode()
    hdr = not raw[:1].isdigit()
    return pd.read_csv(io.StringIO(raw), header=0 if hdr else None, names=None if hdr else names, usecols=range(len(names)))


def load_aux():
    """바이낸스 vision(REST 아님): ETHUSDT 지수가 1분봉 · 펀딩비. 없으면 받는다."""
    fi, ff = D / "binance_index_1m.parquet", D / "binance_funding.parquet"
    if not fi.exists():
        parts = [_vision("futures/um/monthly/indexPriceKlines/ETHUSDT/1m/ETHUSDT-1m-2026-08.zip")]
        parts += [_vision(f"futures/um/daily/indexPriceKlines/ETHUSDT/1m/ETHUSDT-1m-{d:%Y-%m-%d}.zip")
                  for d in pd.date_range("2026-09-01", "2026-10-01")]
        k = pd.concat([_csv(p, ["open_time", "open", "high", "low", "close"]) for p in parts if p])
        k.drop_duplicates("open_time").sort_values("open_time").to_parquet(fi)
    if not ff.exists():
        f = pd.concat([_csv(_vision(f"futures/um/monthly/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2026-{m:02d}.zip"),
                            ["calc_time", "funding_interval_hours", "last_funding_rate"]) for m in (8, 9)])
        f.sort_values("calc_time").to_parquet(ff)
    return pd.read_parquet(fi), pd.read_parquet(ff)


def load_events() -> list[dict]:
    fp = D / "events_high.json"
    if not fp.exists():
        sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
        import live_macro_calendar_20260826 as mc
        key = None
        for p in [ROOT, *ROOT.parents]:
            if (p / ".env").exists():
                key = next((l.split("=", 1)[1].strip().strip('"\'') for l in (p / ".env").read_text().splitlines()
                            if l.startswith("FRED_API_KEY=")), None)
                if key:
                    break
        ev = {}
        for d in ("2026-08-15", "2026-09-03", "2026-09-22"):          # LOOKAHEAD 21일 창 셋 = 08-15~10-13 연속
            for e in mc.compute_macro_calendar(key, None, None, now=datetime.fromisoformat(d + "T12:00:00+00:00"))["events"]:
                if e["importance"] == "high":
                    ev[(e["time_utc"], e["title_ko"])] = e
        fp.write_text(json.dumps({"fred_key": bool(key), "events": sorted(ev.values(), key=lambda e: e["time_utc"])},
                                 ensure_ascii=False, indent=1))
    return json.loads(fp.read_text())["events"]


# ───────── 재현 대조 ─────────
def reproduce(chain_by_t: dict, summ: pd.DataFrame) -> dict:
    worst, n_cmp, n_key = {}, {"expiries": 0, "cm": 0, "surface": 0, "front_atm_oi_usd": 0}, {}

    def cmp(name, a, b):
        if a is None and b is None:
            return
        if (a is None) != (b is None):
            raise SystemExit(f"재현 불일치 {name}: 재계산 {a} vs payload {b}")
        d = abs(a - b) / max(1.0, abs(b))
        worst[name] = max(worst.get(name, 0.0), d)
        if d > 1e-9:
            raise SystemExit(f"재현 불일치 {name}: 재계산 {a} vs payload {b}")

    for t, pl in zip(summ["recorded_at_utc"], summ["payload"]):
        p = json.loads(pl)
        ex = exps_of(chain_by_t[t], p["index"])
        for e, q in zip(ex[:8], p["expiries"]):
            for k in ("exp_ms", "atm_iv", "hours", "atm_i", "rr25i", "bf25i", "fwd", "atm_oi_usd", "call_oi_usd", "put_oi_usd", "oi_share"):
                if k in q:                     # 옛 payload 에 없는 키(09-28 초기분은 hours·atm_i 등이 나중에 생김)
                    cmp(f"expiries.{k}", e[k], q[k])
                    n_key[k] = n_key.get(k, 0) + 1
            n_cmp["expiries"] += 1
        if len(ex[:8]) != len(p["expiries"]):
            raise SystemExit(f"재현 불일치 만기 수 {t}")
        need_s, need_c = p.get("surface") is not None, p.get("cm") is not None
        if not (need_s or need_c):
            continue
        cv = card_values(ex)
        if need_c:
            for d, v in p["cm"].items():
                for k in ("atm", "rr", "bf"):
                    cmp(f"cm.{k}", (cv["cm"][d] or {}).get(k), (v or {}).get(k))
            n_cmp["cm"] += 1
        if need_s:
            for s, v in p["surface"].items():
                w = cv["surface"][s]
                for k in ("exp_ms", "hours", "fwd", "mfiv", "rn_skew", "rn_kurt"):
                    cmp(f"surface.{k}", w[k], v[k])
                for x in ("2", "3", "5"):
                    cmp("surface.tail", w["tail"][x], v["tail"][x])
            cmp("front_atm_oi_usd", cv["front_atm_oi_usd"], p["front_atm_oi_usd"])
            n_cmp["surface"] += 1; n_cmp["front_atm_oi_usd"] += 1
    return {"n_compared": n_cmp, "n_expiry_rows_by_key": n_key, "max_rel_diff": worst}


# ───────── 본 계산 ─────────
def build(chain_by_t, times, px: Px, idxpx: Px, dvol: pd.DataFrame) -> pd.DataFrame:
    dts, dc = dvol["ts"].to_numpy(np.int64) + HR, dvol["c"].to_numpy(float)
    rows = []
    for t in times:
        t_ms = int(t.timestamp() * 1000)
        j = np.searchsorted(dts, t_ms, side="right") - 1          # 끝난 DVOL 시간봉만
        bidx = idxpx.open_at((t_ms // MIN) * MIN)
        ex = exps_of(chain_by_t[t], bidx if np.isfinite(bidx) else float(chain_by_t[t]["underlying_price"].median()))
        cv = card_values(ex)
        r = {"t": t, "t_ms": t_ms, "dvol": dc[j] if j >= 0 else np.nan, "bidx": bidx,
             "front_atm_oi_usd": cv["front_atm_oi_usd"], "exp0_ms": ex[0]["exp_ms"] if ex else None,
             "exps8": [{k: e[k] for k in ("exp_ms", "atm_i", "atm_iv")} for e in ex[:8]]}
        for s0, v in cv["surface"].items():
            s = {"front": "front", "7": "s7", "30": "s30"}[s0]       # 숫자로 시작하는 열 이름은 itertuples 가 지운다
            if v:
                r.update({f"{s}_exp": v["exp_ms"], f"{s}_h": v["hours"], f"{s}_fwd": v["fwd"], f"{s}_mfiv": v["mfiv"],
                          f"{s}_skew": v["rn_skew"], f"{s}_kurt": v["rn_kurt"],
                          **{f"{s}_tail{x}": (v["tail"] or {}).get(x) for x in ("2", "3", "5")}})
        for d, v in cv["cm"].items():
            r[f"cm{d}"] = (v or {}).get("atm"); r[f"cm{d}_rr"] = (v or {}).get("rr")
        fr = next((e for e in ex if e["hours"] >= 3), None)
        r["front_rr25i"] = fr["rr25i"] if fr else None
        rows.append(r)
    return pd.DataFrame(rows)


def hourly_times(times: list) -> list:
    s = pd.Series(times)
    hr = s.dt.round("h")
    d = (s - hr).abs()
    ok = d <= pd.Timedelta(minutes=10)
    return s[ok].groupby(hr[ok]).apply(lambda x: x.loc[d[x.index].idxmin()]).tolist()


def t1_mfiv(H, px):
    rows = []
    for r in H.itertuples():
        if not (r.front_mfiv and np.isfinite(r.dvol)):
            continue
        S = first_after(r.t_ms)
        st = px.settle(int(r.front_exp))
        if not np.isfinite(st):
            continue
        rr = math.log(st / px.open_at(S)); f = math.sqrt(r.front_h / 8760)
        rows.append({"exp": r.front_exp, "t_ms": r.t_ms, "r": rr, "sm": r.front_mfiv / 100 * f, "sd": r.dvol / 100 * f, "h": r.front_h})
    X = pd.DataFrame(rows)
    r_, sm, sd, blk = X["r"].to_numpy(), X["sm"].to_numpy(), X["sd"].to_numpy(), X["exp"].to_numpy()
    out = {"note": "결과 = ln(정산 근사 / 다음 분 시가) · 블록 = 정산 만기(일간 만기라 = UTC 일, 만기 사이 창은 안 겹침)",
           "n_snapshots": len(X), "n_expiries": int(X["exp"].nunique()),
           "mfiv_over_dvol_median": float(np.median(sm / sd))}
    for nm, s in (("mfiv", sm), ("dvol", sd)):
        out[f"in1_{nm}"] = boot(lambda i, s=s: float(np.mean(np.abs(r_[i]) <= s[i])), blk)
        out[f"size_{nm}"] = boot(lambda i, s=s: float(np.mean(np.abs(r_[i])) / np.mean(s[i] * SQ2PI)), blk)
    out["dqlike_mfiv_minus_dvol"] = verdict(boot(lambda i: float(np.mean(qlike(r_[i], sm[i]) - qlike(r_[i], sd[i]))), blk), -1)
    one = X.sort_values("t_ms").groupby("exp").head(1)        # 만기마다 가장 이른 스냅샷 1개(겹침 없음)
    r1, m1, d1 = one["r"].to_numpy(), one["sm"].to_numpy(), one["sd"].to_numpy()
    out["one_per_expiry"] = {"n": len(one), "median_h": float(one["h"].median()),
                             "dqlike": verdict(boot(lambda i: float(np.mean(qlike(r1[i], m1[i]) - qlike(r1[i], d1[i]))), one["exp"].to_numpy()), -1),
                             "size_mfiv": boot(lambda i: float(np.mean(np.abs(r1[i])) / np.mean(m1[i] * SQ2PI)), one["exp"].to_numpy()),
                             "size_dvol": boot(lambda i: float(np.mean(np.abs(r1[i])) / np.mean(d1[i] * SQ2PI)), one["exp"].to_numpy())}
    return out


def t2_tail(H, px):
    out = {"note": "실제 = |정산 근사/다음 분 시가 − 1| > x · 기준선 = DVOL σ√T 로그정규(드리프트 −σ²/2)"}
    for pre, xs, lab in (("front", ("2", "3", "5"), "front"), ("s7", ("5",), "7d")):
        rows = []
        for r in H.itertuples():
            d = r._asdict()
            ex = d.get(f"{pre}_exp")
            if ex is None or not np.isfinite(ex) or not np.isfinite(r.dvol):
                continue
            st = px.settle(int(ex))
            if not np.isfinite(st):
                continue
            mv = abs(st / px.open_at(first_after(r.t_ms)) - 1)
            s = r.dvol / 100 * math.sqrt(d[f"{pre}_h"] / 8760)
            for x in xs:
                p = d.get(f"{pre}_tail{x}")
                if p is None or not np.isfinite(p):
                    continue
                rows.append({"x": x, "exp": int(ex), "p": p, "pl": ln_tail(float(x), s), "y": float(mv > int(x) / 100)})
        X = pd.DataFrame(rows)
        for x, g in X.groupby("x"):
            p, pl, y, blk = g["p"].to_numpy(), g["pl"].to_numpy(), g["y"].to_numpy(), g["exp"].to_numpy()
            q = pd.qcut(g["p"], 5, duplicates="drop")
            cal = [{"bin": str(k), "n": int(len(v)), "n_exp": int(v["exp"].nunique()), "pred": float(v["p"].mean()),
                    "dvol_ln": float(v["pl"].mean()), "actual": float(v["y"].mean())} for k, v in g.groupby(q, observed=True)]
            out[f"{lab}_{x}"] = {"n": len(g), "n_expiries": int(g["exp"].nunique()), "mean_pred": float(p.mean()),
                                 "mean_dvol_ln": float(pl.mean()), "actual_freq": float(y.mean()),
                                 "pred_minus_actual": verdict(boot(lambda i: float(np.mean(p[i] - y[i])), blk), +1),
                                 "dvol_ln_minus_actual": boot(lambda i: float(np.mean(pl[i] - y[i])), blk),
                                 "brier_opt": float(np.mean((p - y) ** 2)), "brier_dvol": float(np.mean((pl - y) ** 2)),
                                 "dbrier_opt_minus_dvol": boot(lambda i: float(np.mean((p[i] - y[i]) ** 2 - (pl[i] - y[i]) ** 2)), blk),
                                 "calibration": cal}
    return out


def _ols_coef(Xm, y, k):
    b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    return b[k]


def t3_slope(H, px):
    rows = []
    for r in H.itertuples():
        if r.cm1 is None or r.cm7 is None or not np.isfinite(r.cm1) or not np.isfinite(r.cm7) or not np.isfinite(r.dvol):
            continue
        nx, pa = px.r5(first_after(r.t_ms), 1440), px.r5_past(r.t_ms, 1440)
        if nx is None or pa is None:
            continue
        rows.append({"t": r.t, "day": r.t.strftime("%Y-%m-%d"), "hour": r.t.hour, "s1": r.cm1 - r.cm7,
                     "s2": (r.cm7 - r.cm30) if r.cm30 is not None and np.isfinite(r.cm30) else np.nan,
                     "dvol": r.dvol, "rvn": ann_rv(nx, 5), "rvp": ann_rv(pa, 5)})
    X = pd.DataFrame(rows).dropna(subset=["s1"])
    out = {"n": len(X), "n_days": int(X["day"].nunique()), "inverted_share": float((X["s1"] > 0).mean()),
           "inverted_days": int(X.loc[X["s1"] > 0, "day"].nunique()),
           "s1_quantiles": X["s1"].quantile([0, .1, .5, .9, 1]).round(2).tolist(),
           "note": "y = ln RV(다음 24h, 5분 수익률, 연율 %) · 블록 = UTC 일 · 계수 = 기울기 1pt 당 ln RV 변화"}
    for nm, D_ in (("hourly", X), ("daily00", X[X["hour"] == 0])):
        D_ = D_.dropna(subset=["s1"])
        y = np.log(D_["rvn"].to_numpy())
        base = np.column_stack([np.ones(len(D_)), np.log(D_["dvol"]), np.log(D_["rvp"])])
        full = np.column_stack([base, D_["s1"]])
        blk = D_["day"].to_numpy()

        def r2(Xm, i):
            b, *_ = np.linalg.lstsq(Xm[i], y[i], rcond=None)
            return 1 - np.var(y[i] - Xm[i] @ b) / np.var(y[i])
        res = {"n": len(D_),
               "coef_s1_controlled": verdict(boot(lambda i: _ols_coef(full[i], y[i], 3), blk), +1),
               "coef_s1_raw": boot(lambda i: _ols_coef(np.column_stack([np.ones(len(i)), D_["s1"].to_numpy()[i]]), y[i], 1), blk),
               "dR2": boot(lambda i: r2(full, i) - r2(base, i), blk)}
        rr = np.log(D_["rvn"] / D_["dvol"]).to_numpy(); inv = (D_["s1"] > 0).to_numpy()
        if inv.sum() >= 3 and (~inv).sum() >= 3:
            res["ln_rv_over_dvol_inverted_minus_normal"] = verdict(
                boot(lambda i: float(rr[i][inv[i]].mean() - rr[i][~inv[i]].mean()) if inv[i].any() and (~inv[i]).any() else np.nan, blk), +1)
        s2 = D_["s2"].to_numpy(); m = np.isfinite(s2)
        if m.sum() > 20:
            f2 = np.column_stack([base[m], s2[m]]); y2 = y[m]
            res["coef_s2_controlled(7d-30d)"] = verdict(boot(lambda i: _ols_coef(f2[i], y2[i], 3), blk[m]), +1)
        if nm == "hourly":                     # 사후 강건성(사전 판정 아님): 1일 IV 의 시각 계절성 통제 -- UTC 시각 더미
            fe = np.column_stack([full] + [(D_["hour"].to_numpy() == h).astype(float) for h in range(1, 24)])
            res["coef_s1_controlled_hourFE_posthoc"] = boot(lambda i: _ols_coef(fe[i], y[i], 3), blk)
        out[nm] = res
    return out


def t4_skew(H, px):
    out = {}
    for hz, mins, bl in (("1d", 1440, 1), ("7d", 10080, 7)):
        rows = []
        for r in H.itertuples():
            d = r._asdict()
            sk, ku = d.get("s30_skew"), d.get("s30_kurt")
            if sk is None or not np.isfinite(sk) or not np.isfinite(r.dvol):
                continue
            x = px.r5(first_after(r.t_ms), mins)
            if x is None:
                continue
            s5 = r.dvol / 100 * math.sqrt(5 / 525600)
            dn, up = float(np.sum(x[x < 0] ** 2)), float(np.sum(x[x > 0] ** 2))
            nd, nu = int(np.sum(x < -2 * s5)), int(np.sum(x > 2 * s5))
            rows.append({"day": r.t.normalize(), "sk": sk, "ku": ku, "semi": math.log(dn / up),
                         "imb": (nd - nu) / (nd + nu) if nd + nu else np.nan,
                         "tail_dvol": float(np.mean(np.abs(x) > 2 * s5)), "tail_real": float(np.mean(np.abs(x) > 2 * np.std(x)))})
        X = pd.DataFrame(rows)
        d0 = X["day"].min()
        blk = ((X["day"] - d0).dt.days // bl).to_numpy()
        sk, ku = X["sk"].to_numpy(), X["ku"].to_numpy()
        g = lambda a, b: (lambda i: spear(a[i], b[i]))   # noqa: E731
        out[hz] = {"n": len(X), "independent_windows": int(round(X["day"].nunique() * 1440 / mins)), "block_days": bl,
                   "skew_vs_ln_down_up_semivar": verdict(boot(g(sk, X["semi"].to_numpy()), blk), -1),
                   "skew_vs_bigmove_down_minus_up": verdict(boot(g(sk, X["imb"].to_numpy()), blk), -1),
                   "kurt_vs_tailfreq_dvol_sigma": verdict(boot(g(ku, X["tail_dvol"].to_numpy()), blk), +1),
                   "kurt_vs_tailfreq_realized_sigma": verdict(boot(g(ku, X["tail_real"].to_numpy()), blk), +1)}
    s = H["s30_skew"].dropna(); k = H["s30_kurt"].dropna()
    out["rn_level"] = {"skew_q": s.quantile([0, .1, .5, .9, 1]).round(3).tolist(), "skew_neg_share": float((s < 0).mean()),
                       "kurt_q": k.quantile([0, .1, .5, .9, 1]).round(2).tolist(), "kurt_gt3_share": float((k > 3).mean())}
    return out


def realized_daily_shape(px, dvol):
    """수준 주장 점검(옵션 아닌 DVOL·바이낸스, 2026-01~09): 일간(00→00 UTC) 수익률을 그 시점 DVOL 1일 σ 로 나눈 z 의 모양."""
    dts, dc = dvol["ts"].to_numpy(np.int64) + HR, dvol["c"].to_numpy(float)
    zs, rs = [], []
    for d in pd.date_range("2026-01-02", "2026-09-30", tz="UTC"):
        a = int(d.timestamp() * 1000)
        j = np.searchsorted(dts, a, side="right") - 1
        r = px.ret(a, a + DAY)
        if j >= 0 and np.isfinite(r):
            zs.append(r / (dc[j] / 100 * math.sqrt(1 / 365))); rs.append(r)
    z, r = np.array(zs), np.array(rs)
    blk = np.arange(len(z)) // 7
    kurt = lambda a: float(np.mean((a - a.mean()) ** 4) / np.var(a) ** 2)   # noqa: E731
    skew = lambda a: float(np.mean((a - a.mean()) ** 3) / np.var(a) ** 1.5)   # noqa: E731
    return {"n_days": len(z), "z_sd": float(z.std()), "share_abs_z_gt2": float(np.mean(np.abs(z) > 2)), "normal_gt2": 0.0455,
            "kurtosis_daily": boot(lambda i: kurt(r[i]), blk), "skew_daily": boot(lambda i: skew(r[i]), blk),
            "note": "블록 = 7일. 30일 지평 실현 모양은 9개월로 못 잰다(독립 9개)"}


def t5_settle(H, px):
    rows = []
    for r in H.itertuples():
        if r.t.hour != 7 or r.front_atm_oi_usd is None or not np.isfinite(r.dvol):
            continue
        e = int(r.exp0_ms)
        if pd.Timestamp(e, unit="ms", tz="UTC").normalize() != r.t.normalize():
            continue                                                     # 그날 08:00 만기가 가까운 만기여야
        S = first_after(r.t_ms); s1 = r.dvol / 100 * math.sqrt(1 / 8760)
        pre, post = px.ret(S, e), px.ret(e, e + HR)
        if not (np.isfinite(pre) and np.isfinite(post)):
            continue
        oth = [abs(px.ret(e - HR + h * HR, e + h * HR)) for h in range(-7, 17) if h != 0]   # 같은 날 다른 1시간 창
        rows.append({"day": r.t.strftime("%Y-%m-%d"), "dow": r.t.dayofweek, "oi": r.front_atm_oi_usd, "pre": pre, "post": post,
                     "zpre": pre / s1, "zpost": post / s1, "apre": abs(pre) / s1, "aother": np.nanmean(oth) / s1})
    X = pd.DataFrame(rows)
    hi = (X["oi"] > X["oi"].median()).to_numpy()
    blk = X["day"].to_numpy()
    dif = lambda a: (lambda i: float(a[i][hi[i]].mean() - a[i][~hi[i]].mean()))   # noqa: E731
    oi = X["oi"].to_numpy()
    out = {"n_days": len(X), "oi_median_usd": float(X["oi"].median()), "oi_q": X["oi"].quantile([0, .5, 1]).round(0).tolist(),
           "friday_share_top_half": float((X["dow"][hi] == 4).mean()),
           "note": "07:00 스냅샷의 그날 08:00 만기 ATM(±2%) 미결제 × 바이낸스 지수가. z = 수익률 / DVOL 1시간 σ. 블록 = 일(일마다 1개라 iid)",
           "weiss_pre_top_minus_bottom(z)": verdict(boot(dif(X["zpre"].to_numpy()), blk), -1),
           "weiss_post_top_minus_bottom(z)": verdict(boot(dif(X["zpost"].to_numpy()), blk), +1),
           "rho_oi_zpre": verdict(boot(lambda i: spear(oi[i], X["zpre"].to_numpy()[i]), blk), -1),
           "card_abs_pre_top_minus_bottom(|z|)": verdict(boot(dif(X["apre"].to_numpy()), blk), +1),
           "rho_oi_abs_zpre": verdict(boot(lambda i: spear(oi[i], X["apre"].to_numpy()[i]), blk), +1),
           "abs_pre_over_same_day_other_hours": boot(lambda i: float(X["apre"].to_numpy()[i].mean() / X["aother"].to_numpy()[i].mean()), blk),
           "abs_pre_over_other_top_minus_bottom": verdict(boot(dif((X["apre"] / X["aother"]).to_numpy()), blk), +1)}
    return out


def unconditional_0708(px):
    """2026-01~09 바이낸스만: 07–08 UTC |수익률| ÷ 그날 전체 시간 평균 |수익률| (만기 시간대 자체가 큰가)."""
    ratios = []
    for d in pd.date_range("2026-01-01", "2026-09-30", tz="UTC"):
        a = int(d.timestamp() * 1000)
        hrs = [abs(px.ret(a + h * HR, a + (h + 1) * HR)) for h in range(24)]
        if all(np.isfinite(hrs)) and np.mean(hrs) > 0:
            ratios.append(hrs[7] / np.mean(hrs))
    r = np.array(ratios)
    return {"n_days": len(r), "ratio_07_to_day_mean": boot(lambda i: float(r[i].mean()), np.arange(len(r)) // 7)}


def t6_event(events, chain_times, chain_by_t, idxpx, px):
    ev_t = sorted({int(pd.Timestamp(e["time_utc"]).timestamp() * 1000) for e in events})
    names = {}
    for e in events:
        names.setdefault(int(pd.Timestamp(e["time_utc"]).timestamp() * 1000), []).append(e["title_ko"])
    t_ms = np.array([int(t.timestamp() * 1000) for t in chain_times])
    lo_ms = int(pd.Timestamp("2026-08-15T08:00Z").timestamp() * 1000)
    rows = []
    for E in ev_t:
        if E < lo_ms or E + HR > px.last_close_ms:
            continue
        j = np.searchsorted(t_ms, E - HR, side="right") - 1
        if j < 0:
            continue
        t = chain_times[j]; tm = t_ms[j]
        ex = exps_of(chain_by_t[t], idxpx.open_at((tm // MIN) * MIN))
        pct = event_move(ex[:8], tm, E)
        a = max(first_after(tm), E - HR)
        act = abs(px.ret(a, E + HR))
        ctrl = []                                    # 같은 요일·시각, 다른 주(08-15~09-30), ±3시간 안 «높음» 일정 없는 날
        for k in range(-7, 8):
            c = E + k * 7 * DAY
            if k == 0 or c < lo_ms or c + HR > px.last_close_ms or any(abs(c - x) <= 3 * HR for x in ev_t):
                continue
            ctrl.append(abs(px.ret(c - HR, c + HR)))
        if not ctrl:
            continue
        rows.append({"E": pd.Timestamp(E, unit="ms", tz="UTC").isoformat(), "names": "·".join(names[E]), "snap_lag_min": (E - tm) / MIN,
                     "pct": pct, "act": act * 100, "ctrl": float(np.mean(ctrl)) * 100, "n_ctrl": len(ctrl),
                     "day": pd.Timestamp(E, unit="ms", tz="UTC").strftime("%Y-%m-%d")})
    X = pd.DataFrame(rows)
    X["excess"] = X["act"] - X["ctrl"]; X["ratio"] = X["act"] / X["ctrl"]
    cls = np.where(X["pct"].isna(), "안 걸림", np.where(X["pct"] < 0.05, "거의 안 들어감", "들어감"))
    X["cls"] = cls
    out = {"n_events": len(X), "class_counts": X["cls"].value_counts().to_dict(),
           "note": "실제 = |ln P(E+60분)/P(max(E−60분, 스냅샷 다음 분))| % · 대조 = 같은 요일·시각 다른 주 평균(±3h 일정 없는 날) · 블록 = UTC 일",
           "mean_act_pct": float(X["act"].mean()), "mean_ctrl_pct": float(X["ctrl"].mean())}
    blk = X["day"].to_numpy()
    ex_, ra = X["excess"].to_numpy(), np.log(X["ratio"].to_numpy())
    out["all_events_ln_act_over_ctrl"] = boot(lambda i: float(ra[i].mean()), blk)
    m = X["pct"].notna().to_numpy()
    p = X["pct"].to_numpy(float)
    out["rho_pct_excess"] = verdict(boot(lambda i: spear(p[i][m[i]], ex_[i][m[i]]), blk), +1)
    out["rho_pct_ln_ratio"] = verdict(boot(lambda i: spear(p[i][m[i]], ra[i][m[i]]), blk), +1)
    a, b = (cls == "들어감"), (cls == "거의 안 들어감")
    if a.sum() >= 2 and b.sum() >= 2:
        out["excess_in_minus_barely"] = verdict(boot(lambda i: float(ex_[i][a[i]].mean() - ex_[i][b[i]].mean())
                                                     if a[i].any() and b[i].any() else np.nan, blk), +1)
    out["by_class"] = {k: {"n": int(len(g)), "mean_pct": float(g["pct"].mean()) if g["pct"].notna().any() else None,
                           "mean_act": float(g["act"].mean()), "mean_ctrl": float(g["ctrl"].mean()),
                           "mean_excess": float(g["excess"].mean())} for k, g in X.groupby("cls")}
    out["events"] = X.round(4).to_dict("records")
    return out


def t7_fwd(H, summ, px, idxpx, fund, chain_by_t):
    fc, fr_ = fund["calc_time"].to_numpy(np.int64), fund["last_funding_rate"].to_numpy(float)
    fh = fund["funding_interval_hours"].to_numpy(float)
    rows = []
    for r in H.itertuples():
        if r.front_fwd is None or not np.isfinite(r.bidx):
            continue
        fl = (r.t_ms // MIN) * MIN
        j = np.searchsorted(fc, r.t_ms, side="right") - 1
        S = first_after(r.t_ms)
        rows.append({"t_ms": r.t_ms, "day": r.t.strftime("%Y-%m-%d"), "exp": r.front_exp, "bp": (r.front_fwd / r.bidx - 1) * 1e4,
                     "rr7": r.cm7_rr, "rrf": r.front_rr25i, "fund_carry_bp": fr_[j] * r.front_h / fh[j] * 1e4 if j >= 0 else np.nan,
                     "prem_bp": (px.open_at(fl) / idxpx.open_at(fl) - 1) * 1e4,
                     "r1h": px.ret(S, S + HR), "r24h": px.ret(S, S + DAY)})
    X = pd.DataFrame(rows).sort_values("t_ms")
    X["d1"] = X["bp"].diff().where((X["exp"] == X["exp"].shift()) & (X["t_ms"].diff() <= 1.2 * HR))
    X["drr7"] = X["rr7"].diff().where(X["d1"].notna())
    blk = X["day"].to_numpy()
    bp = X["bp"].to_numpy()
    g = lambda col_: (lambda i: spear(bp[i], X[col_].to_numpy(float)[i]))   # noqa: E731
    out = {"n": len(X), "n_days": int(X["day"].nunique()),
           "note": "지수 대리 = 바이낸스 ETHUSDT 지수가(USDT) 같은 분 시가. Deribit 지수(USD) 와의 차는 payload 구간에서 잼",
           "level_bp": {"mean": float(np.mean(bp)), "sd": float(np.std(bp)), "q": np.percentile(bp, [5, 25, 50, 75, 95]).round(2).tolist()},
           "noise_1h_change_bp": {"sd": float(X["d1"].std()), "median_abs": float(X["d1"].abs().median()),
                                  "lag1_autocorr_level": float(pd.Series(bp).autocorr())},
           "rho_bp_rr7_level": verdict(boot(g("rr7"), blk), +1),
           "rho_bp_rr_front_level": verdict(boot(g("rrf"), blk), +1),
           "rho_d1_drr7_change": verdict(boot(lambda i: spear(X["d1"].to_numpy(float)[i], X["drr7"].to_numpy(float)[i]), blk), +1),
           "rho_bp_funding_carry": verdict(boot(g("fund_carry_bp"), blk), +1),
           "rho_bp_binance_premium": verdict(boot(g("prem_bp"), blk), +1),
           "ic_next1h": boot(g("r1h"), blk), "ic_next24h": boot(g("r24h"), blk),
           "hit_sign_next1h": float(np.mean(np.sign(bp) == np.sign(X["r1h"]))), "hit_sign_next24h": float(np.nanmean(np.where(X["r24h"].notna(), np.sign(bp) == np.sign(X["r24h"]), np.nan)))}
    # payload 구간(진짜 Deribit 지수, 10분): 같은 스냅샷 선도−지수 수준·10분 변화, 그리고 바이낸스 대리와의 차
    P = []
    for t, pl in zip(summ["recorded_at_utc"], summ["payload"]):
        p = json.loads(pl)
        g = chain_by_t[t]                     # payload 의 fwd 는 10-01 15시 이후만 있다 → 같은 스냅샷 체인의 «3시간 이상 남은 첫 만기» 선도가
        g = g[g["days_to_expiry"] * 24 >= 3]
        tm = int(t.timestamp() * 1000)
        bi = idxpx.open_at((tm // MIN) * MIN)
        if len(g) and p.get("index"):
            g = g[g["expiration_ts"] == g["expiration_ts"].min()]
            c7 = col._const_maturity(exps_of(chain_by_t[t], p["index"]), 7)
            fl = (tm // MIN) * MIN
            P.append({"t_ms": tm, "exp": int(g["expiration_ts"].iloc[0].timestamp() * 1000), "rr7": c7["rr"] if c7 else np.nan,
                      "prem_bp": (px.open_at(fl) / idxpx.open_at(fl) - 1) * 1e4, "day": t.strftime("%Y-%m-%d"),
                      "bp": (float(g["underlying_price"].iloc[0]) / p["index"] - 1) * 1e4,
                      "proxy_gap_bp": (bi / p["index"] - 1) * 1e4 if np.isfinite(bi) else np.nan})
    P = pd.DataFrame(P).sort_values("t_ms")
    d10 = P["bp"].diff().where((P["exp"] == P["exp"].shift()) & (P["t_ms"].diff() <= 11 * MIN))
    out["deribit_index_window"] = {"n": len(P), "from": pd.Timestamp(P["t_ms"].min(), unit="ms", tz="UTC").isoformat(),
                                   "level_bp_q": np.percentile(P["bp"], [5, 50, 95]).round(2).tolist(), "sd_10min_change_bp": float(d10.std()),
                                   "binance_index_minus_deribit_index_bp": {"mean": float(P["proxy_gap_bp"].mean()), "sd": float(P["proxy_gap_bp"].std())},
                                   # 사후 점검: 진짜 지수로 잰 선도−지수 -- 바이낸스 지수를 분모로 공유하지 않아 기계적 상관이 없다. 3.5일뿐(블록 = 6시간)
                                   "rho_true_bp_rr7_posthoc": boot(lambda i: spear(P["bp"].to_numpy()[i], P["rr7"].to_numpy()[i]), (P["t_ms"] // (6 * HR)).to_numpy()),
                                   "rho_true_bp_binance_premium_posthoc": boot(lambda i: spear(P["bp"].to_numpy()[i], P["prem_bp"].to_numpy()[i]), (P["t_ms"] // (6 * HR)).to_numpy()),
                                   "rho_true_bp_vs_proxy_bp_posthoc": float(spear(P["bp"].to_numpy(), (P["bp"] - P["proxy_gap_bp"]).to_numpy()))}
    return out


# ───────── 자체 점검 ─────────
def selftest():
    # ① 시점 경계: 07:00:02 스냅샷 → 07:01 봉부터 · 정각 스냅샷도 다음 분부터(같은 분 공유 금지)
    t = int(pd.Timestamp("2026-09-01T07:00:02Z").timestamp() * 1000)
    assert first_after(t) == int(pd.Timestamp("2026-09-01T07:01Z").timestamp() * 1000)
    t0 = int(pd.Timestamp("2026-09-01T07:00:00Z").timestamp() * 1000)
    assert first_after(t0) == t0 + MIN
    # ② QLIKE: 참 σ 가 과대·과소 σ 보다 기대 손실이 작다
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.02, 200_000)
    q = {k: qlike(r, np.full_like(r, 0.02 * k)).mean() for k in (0.7, 1.0, 1.5)}
    assert q[1.0] < q[0.7] and q[1.0] < q[1.5], q
    # ③ 로그정규 꼬리 기준선 = 몬테카를로
    s = 0.03
    z = np.exp(rng.normal(-s * s / 2, s, 400_000))
    assert abs(ln_tail(3, s) - np.mean(np.abs(z - 1) > 0.03)) < 0.003
    # ④ 평평한 스마일(IV 40%)의 수집기 꼬리 확률 ≈ 로그정규(수집기 함수 재사용 확인)
    K = np.arange(2000, 3600, 25.0)
    g = pd.DataFrame({"strike": np.r_[K, K], "option_type": ["call"] * len(K) + ["put"] * len(K), "mark_iv": 40.0})
    st = col._surface_stats(g, 2700.0, 7 / 365)
    s7 = 0.40 * math.sqrt(7 / 365)
    assert abs(st["tail"]["5"] - ln_tail(5, s7)) < 0.01, (st["tail"], ln_tail(5, s7))
    assert abs(st["mfiv"] - 40) < 1.0, st["mfiv"]
    # ⑤ 이벤트 폭: 평평한 분산 속도면 0, 이벤트 만기에 분산 덩어리를 얹으면 그 크기
    ex = [{"exp_ms": 10 * HR, "atm_i": 40.0, "atm_iv": 40.0}, {"exp_ms": 34 * HR, "atm_i": 40.0, "atm_iv": 40.0},
          {"exp_ms": 58 * HR, "atm_i": 40.0, "atm_iv": 40.0}]
    assert abs(event_move(ex, 0, 20 * HR)) < 1e-4          # 부동소수 잔차(√ 라 1e-9 분산 → 1e-7%)
    T2 = 34 * HR / 3.156e10
    ex[1]["atm_i"] = math.sqrt((0.4 ** 2 * T2 + 0.01 ** 2) / T2) * 100      # 이벤트 1σ 1%
    T3 = 58 * HR / 3.156e10
    ex[2]["atm_i"] = math.sqrt((0.4 ** 2 * T3 + 0.01 ** 2) / T3) * 100
    assert abs(event_move(ex, 0, 20 * HR) - 1.0) < 1e-6
    print("selftest OK")


def summary_md(res: dict) -> str:
    L = ["# 옵션 정보 칸 의미 검증 요약 (2026-10-02)", "", f"재현 대조: {res['reproduction']}", ""]
    def v(r):
        return f"{fmt(r)} → **{r.get('verdict', '-')}**" + (f" (80% 검정력 필요 블록 ≈ {r['blocks_needed_80pct']})" if r.get("verdict") == "검정력 부족" else "")
    t = res["1_mfiv"]
    L += ["## 1 다음 만기까지 MFIV vs DVOL", f"- 스냅샷 {t['n_snapshots']} · 만기 {t['n_expiries']} · mfiv/dvol 중앙 {t['mfiv_over_dvol_median']:.3f}",
          f"- ±1σ 안: mfiv {fmt(t['in1_mfiv'])} · dvol {fmt(t['in1_dvol'])}",
          f"- 크기 비: mfiv {fmt(t['size_mfiv'])} · dvol {fmt(t['size_dvol'])}",
          f"- ΔQLIKE(mfiv−dvol): {v(t['dqlike_mfiv_minus_dvol'])}",
          f"- 만기당 1개: ΔQLIKE {v(t['one_per_expiry']['dqlike'])} · 크기 비 mfiv {fmt(t['one_per_expiry']['size_mfiv'])} / dvol {fmt(t['one_per_expiry']['size_dvol'])}", ""]
    L += ["## 2 꼬리 확률"]
    for k, x in res["2_tail"].items():
        if k == "note":
            continue
        L += [f"- {k}: n {x['n']} · 만기 {x['n_expiries']} · 예측 {x['mean_pred']:.3f} · DVOL 로그정규 {x['mean_dvol_ln']:.3f} · 실제 {x['actual_freq']:.3f}",
              f"  - 예측−실제 {v(x['pred_minus_actual'])} · DVOL−실제 {fmt(x['dvol_ln_minus_actual'])}",
              f"  - Brier 옵션 {x['brier_opt']:.4f} vs DVOL {x['brier_dvol']:.4f} · Δ {fmt(x['dbrier_opt_minus_dvol'], 4)}",
              "  - 보정: " + " · ".join(f"[{c['pred']:.3f}→{c['actual']:.3f}, n{c['n']}/만기{c['n_exp']}]" for c in x["calibration"])]
    t = res["3_slope"]
    L += ["", "## 3 기간 구조 기울기", f"- n {t['n']} · 일 {t['n_days']} · 역전 비율 {t['inverted_share']:.3f} (역전 있는 날 {t['inverted_days']}) · 1일−7일 분위 {t['s1_quantiles']}"]
    for nm in ("hourly", "daily00"):
        x = t[nm]
        L += [f"- {nm} (n {x['n']}): 통제 계수 {v(x['coef_s1_controlled'])} · 원 계수 {fmt(x['coef_s1_raw'])} · ΔR² {fmt(x['dR2'], 4)}"]
        if "ln_rv_over_dvol_inverted_minus_normal" in x:
            L += [f"  - ln(RV/DVOL) 역전−정상 {v(x['ln_rv_over_dvol_inverted_minus_normal'])}"]
        if "coef_s1_controlled_hourFE_posthoc" in x:
            L += [f"  - (사후) 시각 고정효과 추가 계수 {fmt(x['coef_s1_controlled_hourFE_posthoc'])}"]
        if "coef_s2_controlled(7d-30d)" in x:
            L += [f"  - 7일−30일 통제 계수 {v(x['coef_s2_controlled(7d-30d)'])}"]
    t = res["4_skew_kurt"]
    L += ["", "## 4 위험중립 왜도·첨도(30일)", f"- 수준: {t['rn_level']}", f"- 실현 일간 모양(2026-01~09): {json.dumps(t['realized_daily_shape'], ensure_ascii=False, default=str)}"]
    for hz in ("1d", "7d"):
        x = t[hz]
        L += [f"- {hz}: n {x['n']} · 독립 창 ≈ {x['independent_windows']} · 블록 {x['block_days']}일"]
        for k in ("skew_vs_ln_down_up_semivar", "skew_vs_bigmove_down_minus_up", "kurt_vs_tailfreq_dvol_sigma", "kurt_vs_tailfreq_realized_sigma"):
            L += [f"  - {k}: {v(x[k])}"]
    t = res["5_settle"]
    L += ["", "## 5 정산 창 · ATM 미결제", f"- 일 {t['n_days']} · 미결제 중앙 ${t['oi_median_usd']:,.0f} · 상위 절반 금요일 비율 {t['friday_share_top_half']:.2f}"]
    for k in ("weiss_pre_top_minus_bottom(z)", "weiss_post_top_minus_bottom(z)", "rho_oi_zpre", "card_abs_pre_top_minus_bottom(|z|)",
              "rho_oi_abs_zpre", "abs_pre_over_other_top_minus_bottom"):
        L += [f"- {k}: {v(t[k])}"]
    L += [f"- 07–08 |r| ÷ 같은 날 다른 시간(옵션 표본): {fmt(t['abs_pre_over_same_day_other_hours'])}",
          f"- 무조건 07–08 |r| ÷ 일 평균(2026-01~09): {fmt(res['5_settle_unconditional']['ratio_07_to_day_mean'])}"]
    t = res["6_event"]
    L += ["", "## 6 이벤트 예상 폭", f"- 일정 {t['n_events']} · 분류 {t['class_counts']} · 실제 평균 {t['mean_act_pct']:.3f}% vs 대조 {t['mean_ctrl_pct']:.3f}%",
          f"- 전체 ln(실제/대조) {fmt(t['all_events_ln_act_over_ctrl'])}",
          f"- ρ(폭, 초과) {v(t['rho_pct_excess'])}", f"- ρ(폭, ln 비) {v(t['rho_pct_ln_ratio'])}"]
    if "excess_in_minus_barely" in t:
        L += [f"- 초과(들어감 − 거의 안 들어감) {v(t['excess_in_minus_barely'])}"]
    L += [f"- 분류별: {json.dumps(t['by_class'], ensure_ascii=False)}"]
    t = res["7_fwd"]
    L += ["", "## 7 옵션 선도 − 지수", f"- n {t['n']} · 일 {t['n_days']} · 수준 {t['level_bp']} · 1시간 변화 {t['noise_1h_change_bp']}",
          f"- Deribit 지수 구간: {t['deribit_index_window']}"]
    for k in ("rho_bp_rr7_level", "rho_bp_rr_front_level", "rho_d1_drr7_change", "rho_bp_funding_carry", "rho_bp_binance_premium"):
        L += [f"- {k}: {v(t[k])}"]
    L += [f"- IC 1h {fmt(t['ic_next1h'])} · 24h {fmt(t['ic_next24h'])} · 부호 적중 1h {t['hit_sign_next1h']:.3f} / 24h {t['hit_sign_next24h']:.3f}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    chain = pd.read_parquet(D / "eth_chain.parquet")
    chain = chain[chain["currency"] == "ETH"]
    chain_by_t = dict(tuple(chain.groupby("recorded_at_utc", sort=True)))
    times = sorted(chain_by_t)
    summ = pd.read_parquet(D / "eth_summary.parquet")
    summ = summ[summ["currency"] == "ETH"].sort_values("recorded_at_utc")
    rep = reproduce(chain_by_t, summ)
    print("재현 대조 통과", rep, flush=True)
    px = Px(pd.read_parquet(D / "ethusdt_1m.parquet"))
    idx_k, fund = load_aux()
    idxpx = Px(idx_k)
    dvol = pd.read_parquet(D / "eth_dvol_1h.parquet").sort_values("ts")
    ht = hourly_times(times)
    H = build(chain_by_t, ht, px, idxpx, dvol)
    print("시간 스냅샷", len(H), H["t"].min(), H["t"].max(), flush=True)
    events = load_events()
    res = {"generated_at": datetime.now(timezone.utc).isoformat(), "criteria": CRITERIA, "reproduction": rep,
           "data": {"chain": [str(times[0]), str(times[-1])], "hourly_snapshots": len(H), "klines_end": str(pd.Timestamp(px.last_close_ms, unit="ms", tz="UTC")),
                    "settle_proxy": "Deribit 07:30~08:00 지수 TWAP ≈ 바이낸스 선물 1분 종가 30개 평균",
                    "events_source": "live_macro_calendar compute_macro_calendar(now=08-15·09-03·09-22), importance=high, 미시간대는 홈페이지가 «다음 1건»만 줘서 과거분 없음"}}
    res["1_mfiv"] = t1_mfiv(H, px); print("1 done", flush=True)
    res["2_tail"] = t2_tail(H, px); print("2 done", flush=True)
    res["3_slope"] = t3_slope(H, px); print("3 done", flush=True)
    res["4_skew_kurt"] = {**t4_skew(H, px), "realized_daily_shape": realized_daily_shape(px, dvol)}; print("4 done", flush=True)
    res["5_settle"] = t5_settle(H, px); res["5_settle_unconditional"] = unconditional_0708(px); print("5 done", flush=True)
    res["6_event"] = t6_event(events, times, chain_by_t, idxpx, px); print("6 done", flush=True)
    res["7_fwd"] = t7_fwd(H, summ, px, idxpx, fund, chain_by_t); print("7 done", flush=True)
    (D / "surface_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    md = summary_md(res)
    (D / "surface_summary.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
