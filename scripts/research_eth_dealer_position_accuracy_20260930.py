#!/usr/bin/env python3
"""ETH 옵션 «딜러 포지션» 추정 정확도 연구 (2026-09-30, 사용자 «딜러 입장인지 누가 샀는지 모호 — 정확하게 할 방법?»).

질문 4개(보고는 최종 메시지):
  Q1 커버리지 -- 수집 시작(09-27) 뒤 상장 종목의 미결제 비중: 지금 실측(가까운 만기·7일 안·전 만기) + 만기 교체 예측 곡선.
  Q2 «메이커 = 딜러» 가정 -- 첫 체결부터 다 본 종목에서 |Σ테이커 순|/OI, 스냅샷 OI 경로 대비 위반, 필터(블록·청산·크기) 효과,
     테이커 순의 «방향성» z = N/sqrt(Σa²)(무작위 동전이면 |z|≈0.8).
  Q3 추정법 비교 -- 관행(콜+·풋−) · 테이커 전량 · 블록 제외 · 소형만 · OI 제약판(창 ΔOI>0 만 ±ΔOI 로 잘라 누적, ΔOI<0 은 비례 축소)
     의 DEX·GEX 부호가 관행과 얼마나 다른가(08-15~ 매시).
  Q4 간접 검증 -- (i) 옵션 고객 델타 흐름 → Deribit ETH-PERPETUAL 테이커 순매수(헤지) 선행·지연 회귀,
     (ii) 방법별 GEX 가 다음 4h 변동폭을 얼마나 설명하나(부호가 맞는 쪽이 감쇠를 설명해야).

데이터:
  서버 deribit_options.duckdb 에서 뽑아 온 tmp/dpa20260930/{snap,db_trades}.parquet (체인 스냅샷 08-15~ · 수집 체결 09-27~)
  + history.deribit.com 체결 전량(2025-12-25~, 현 상장 종목의 첫 체결까지 덮는다) + ETH-PERPETUAL 체결(최근 7일)
  + www.deribit.com get_instruments(상장 시각)·chart(5분봉). 🔴바이낸스는 부르지 않는다.
출력 tmp/dealer_position_accuracy_20260930/ (캐시 parquet + results.json).

사용:
  python scripts/research_eth_dealer_position_accuracy_20260930.py            # 받기(캐시) + 분석
  python scripts/research_eth_dealer_position_accuracy_20260930.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/dpa20260930"
OUT = ROOT / "tmp/dealer_position_accuracy_20260930"
HIST = "https://history.deribit.com/api/v2/public/"
LIVE = "https://www.deribit.com/api/v2/public/"
HIST_START_MS = int(datetime(2025, 12, 25, tzinfo=timezone.utc).timestamp() * 1000)   # 현 상장 최古 종목(12월물) 상장일
PERP_DAYS = 7
SMALL = 25.0          # 소형 체결 상한(ETH) -- 개인 대리
INST_RE = re.compile(r"^ETH-(\d{1,2}[A-Z]{3}\d{2})-(\d+(?:d\d+)?)-([CP])$")


def _get(base: str, method: str, **params):
    for i in range(6):
        try:
            r = requests.get(base + method, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (i + 1)); continue
            r.raise_for_status()
            return r.json()["result"]
        except requests.RequestException:
            if i == 5:
                raise
            time.sleep(2 * (i + 1))


def parse(name: str):
    m = INST_RE.match(name)
    if not m:
        return None
    exp = datetime.strptime(m.group(1), "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
    return float(m.group(2).replace("d", ".")), exp, 1.0 if m.group(3) == "C" else -1.0


def exp_type(exp: pd.Timestamp) -> str:
    """Deribit 만기 종류: 금요일 아니면 일간 · 마지막 금요일이면 월간(3·6·9·12월은 분기) · 나머지 금요일은 주간."""
    if exp.weekday() != 4:
        return "daily"
    if (exp + pd.Timedelta(days=7)).month != exp.month:
        return "quarterly" if exp.month in (3, 6, 9, 12) else "monthly"
    return "weekly"


# ───────────────────────── 받기(캐시) ─────────────────────────
def fetch_paged(path: Path, method: str, key: dict, start_ms: int, end_ms: int) -> pd.DataFrame:
    """asc 로 1000건씩 -- 마지막 시각부터 다시(경계 중복은 trade_id 로 버림). 중간 저장으로 이어받기."""
    part = path.with_suffix(".part.parquet")
    rows, t = [], start_ms
    if part.exists():
        old = pd.read_parquet(part); rows = old.to_dict("records"); t = int(old["timestamp"].max())
    n = 0
    while True:
        res = _get(HIST, method, **key, start_timestamp=t, end_timestamp=end_ms, count=1000,
                   sorting="asc", include_old="true")
        tr = res["trades"]
        rows += tr
        if not res.get("has_more") or not tr:
            break
        nt = int(tr[-1]["timestamp"])
        t = nt if nt > t else t + 1
        n += 1
        if n % 50 == 0:
            pd.DataFrame(rows).to_parquet(part); print(f"  {method} {key} ~{datetime.fromtimestamp(t/1000, timezone.utc):%m-%d %H:%M} {len(rows)}행", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("trade_id")
    df.to_parquet(path); part.unlink(missing_ok=True)
    return df


def load_all():
    OUT.mkdir(parents=True, exist_ok=True)
    now_ms = int(pd.Timestamp(pd.read_parquet(SRC / "snap.parquet", columns=["recorded_at_utc"])["recorded_at_utc"].max()).timestamp() * 1000)
    p = OUT / "opt_trades_hist.parquet"
    tr = pd.read_parquet(p) if p.exists() else fetch_paged(p, "get_last_trades_by_currency_and_time",
                                                          {"currency": "ETH", "kind": "option"}, HIST_START_MS, now_ms)
    p = OUT / "perp_trades.parquet"
    pp = pd.read_parquet(p) if p.exists() else fetch_paged(p, "get_last_trades_by_instrument_and_time",
                                                          {"instrument_name": "ETH-PERPETUAL"}, now_ms - PERP_DAYS * 86_400_000, now_ms)
    p = OUT / "instruments.parquet"
    if not p.exists():
        pd.DataFrame(_get(LIVE, "get_instruments", currency="ETH", kind="option"))[
            ["instrument_name", "creation_timestamp", "expiration_timestamp"]].to_parquet(p)
    ins = pd.read_parquet(p)
    p = OUT / "perp_5m.parquet"
    if not p.exists():
        bars, t0 = [], now_ms - 50 * 86_400_000
        while t0 < now_ms:
            t1 = min(t0 + 15 * 86_400_000, now_ms)
            r = _get(LIVE, "get_tradingview_chart_data", instrument_name="ETH-PERPETUAL", start_timestamp=t0, end_timestamp=t1, resolution="5")
            bars.append(pd.DataFrame({k: r[k] for k in ("ticks", "open", "high", "low", "close")})); t0 = t1
        pd.concat(bars).drop_duplicates("ticks").to_parquet(p)
    bars = pd.read_parquet(p)
    snap = pd.read_parquet(SRC / "snap.parquet")
    db = pd.read_parquet(SRC / "db_trades.parquet")
    return tr, pp, ins, bars, snap, db, now_ms


# ───────────────────────── 그릭스(수집기 _gamma 와 같은 식) ─────────────────────────
def greeks(F, K, iv, T, sg):
    """선도가 F·IV(소수)·연 T → (델타, 감마). 감마$=γ·F²·1% , 델타$=δ·F 는 호출자가."""
    ok = (iv > 0) & (T > 0) & (F > 0)
    iv_, T_ = np.where(ok, iv, 1.0), np.where(ok, T, 1.0)
    d1 = (np.log(np.where(ok, F, 1.0) / K) + 0.5 * iv_ ** 2 * T_) / (iv_ * np.sqrt(T_))
    nd1 = 0.5 * (1.0 + np.array([math.erf(x / math.sqrt(2.0)) for x in d1]))
    gam = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (np.where(ok, F, 1.0) * iv_ * np.sqrt(T_))
    return np.where(ok, np.where(sg > 0, nd1, nd1 - 1.0), 0.0), np.where(ok, gam, 0.0)


# ───────────────────────── 추정기 ─────────────────────────
def oi_bounded(net_w: np.ndarray, oi: np.ndarray, n_before: float) -> np.ndarray:
    """OI 제약판 딜러 포지션 경로. 창 k 의 테이커 순 net_w[k], 창 끝 OI oi[k].
    첫 창: D = clip(−(그 전 누적 + 창 순), ±OI). 이후 ΔOI>0 이면 D += clip(−net, ±ΔOI), ΔOI<0 이면 D *= OI_new/OI_old,
    그리고 |D| ≤ OI 로 자른다. ponytail: 신규/청산 판별 없이 ΔOI 부호만 -- 정밀판은 다른 에이전트 몫."""
    d = np.empty(len(oi)); prev_oi = None; D = 0.0
    for k in range(len(oi)):
        if prev_oi is None:
            D = -(n_before + net_w[k])
        elif oi[k] > prev_oi:
            dO = oi[k] - prev_oi; D += float(np.clip(-net_w[k], -dO, dO))
        elif oi[k] < prev_oi:
            D *= oi[k] / prev_oi if prev_oi > 0 else 0.0
        D = float(np.clip(D, -oi[k], oi[k])); d[k] = D; prev_oi = oi[k]
    return d


def assign_windows(tr_ts: np.ndarray, snap_ts: np.ndarray) -> np.ndarray:
    """체결 시각 → 창 번호(스냅샷 k 이하 · k−1 초과면 k). 첫 스냅샷 이전은 0, 마지막 뒤는 len(snap)(버림)."""
    return np.searchsorted(snap_ts, tr_ts, side="left")


# ───────────────────────── 분석 ─────────────────────────
def prep(tr, snap):
    tr = tr.copy()
    tr["sgn"] = np.where(tr["direction"] == "buy", 1.0, -1.0)
    tr["q"] = tr["sgn"] * tr["amount"]
    tr["is_block"] = tr["block_trade_id"].notna() if "block_trade_id" in tr else False
    tr["is_liq"] = tr["liquidation"].notna() if "liquidation" in tr else False
    tr["is_combo"] = tr["combo_id"].notna() if "combo_id" in tr else False
    pr = tr["instrument_name"].map(parse)
    tr = tr[pr.notna()].copy(); pr = pr[pr.notna()]
    tr["K"] = [x[0] for x in pr]; tr["exp"] = pd.to_datetime([x[1] for x in pr]); tr["sg"] = [x[2] for x in pr]
    snap = snap.copy()
    snap["ts"] = snap["recorded_at_utc"].astype("int64") // 1000 if str(snap["recorded_at_utc"].dtype).startswith("datetime64[us") else snap["recorded_at_utc"].astype("int64") // 1_000_000
    snap["sg"] = np.where(snap["option_type"] == "call", 1.0, -1.0)
    return tr, snap


def q1_coverage(ins, snap, t0_ms, now_ms):
    last = snap[snap["ts"] == snap["ts"].max()].merge(ins[["instrument_name", "creation_timestamp"]], on="instrument_name", how="left")
    last = last[last["expiration_ts"] > pd.Timestamp(now_ms, unit="ms", tz="UTC")]
    last["cov"] = last["creation_timestamp"] >= t0_ms
    last["dte"] = (last["expiration_ts"] - pd.Timestamp(now_ms, unit="ms", tz="UTC")).dt.total_seconds() / 86400
    front = last["expiration_ts"].min()
    cur = {}
    for nm, sel in (("front", last["expiration_ts"] == front), ("week", last["dte"] <= 7), ("all", last["dte"] > 0)):
        g = last[sel]; cur[nm] = round(float(g.loc[g["cov"], "open_interest"].sum() / g["open_interest"].sum()), 4)
    # 만기별 표
    by = last.groupby("expiration_ts").apply(lambda g: pd.Series({"n": len(g), "oi": g["open_interest"].sum(),
                                                               "cov_frac": g.loc[g["cov"], "open_interest"].sum() / max(g["open_interest"].sum(), 1e-9),
                                                               "created_min": pd.Timestamp(g["creation_timestamp"].min(), unit="ms")}), include_groups=False)
    by["type"] = [exp_type(e.tz_convert(None)) for e in by.index]
    by["dte"] = [(e - pd.Timestamp(now_ms, unit="ms", tz="UTC")).total_seconds() / 86400 for e in by.index]
    # 예측: 종류별 OI(DTE) 곡선(지금 단면에서 로그 선형보간) × 만기별 커버 비율. 미래 상장분은 전부 커버.
    curve = {t: g.sort_values("dte")[["dte", "oi"]].to_numpy() for t, g in by.groupby("type")}
    def f(t, dte):
        c = curve.get(t)
        if c is None or len(c) == 0:
            return 0.0
        return float(np.exp(np.interp(dte, c[:, 0], np.log(np.maximum(c[:, 1], 1.0)))))
    lead = {"daily": 4, "weekly": 22, "monthly": 92, "quarterly": 366}          # 상장 선행 일수(현 목록에서 실측)
    fc = []
    nowd = pd.Timestamp(now_ms, unit="ms", tz="UTC").normalize()
    for D in pd.date_range(nowd + pd.Timedelta(days=1), pd.Timestamp("2027-10-01", tz="UTC"), freq="D"):
        Dt = D + pd.Timedelta(hours=9)       # 그날 09:00 UTC(08:00 만기 직후)
        num = den = num_lb = den_lb = 0.0
        exps = set(by.index[by.index > Dt])
        # 미래 만기 후보(상장 규칙대로 D 시점에 존재할 만기)
        for e in pd.date_range(Dt.normalize(), Dt + pd.Timedelta(days=370), freq="D"):
            e8 = e + pd.Timedelta(hours=8)
            if e8 <= Dt or e8 in by.index:
                continue
            t = exp_type(e8.tz_convert(None))
            # 지금은 아직 상장 전(선행 일수 밖)이고 D 에는 상장돼 있을 만기 = 수집 뒤 상장 → 전부 커버
            if (e8 - Dt).days <= lead[t] and (e8 - pd.Timestamp(now_ms, unit="ms", tz="UTC")).days > lead[t]:
                exps.add(e8)
        for e in exps:
            t = exp_type(e.tz_convert(None)); dte = (e - Dt).total_seconds() / 86400
            w = f(t, dte); cf = float(by.loc[e, "cov_frac"]) if e in by.index else 1.0
            num += w * cf; den += w
            if e in by.index:
                o = float(by.loc[e, "oi"]); num_lb += o * cf; den_lb += o
        fc.append({"date": D.strftime("%Y-%m-%d"), "cov_model": num / den if den else None,
                   "cov_existing_only": num_lb / den_lb if den_lb else None})
    fc = pd.DataFrame(fc)
    hit90 = fc.loc[fc["cov_model"] >= 0.9, "date"].min()
    hit100_lb = fc.loc[fc["cov_existing_only"].isna() | (fc["cov_existing_only"] >= 0.9999), "date"].min()
    by_out = by.reset_index().assign(expiration_ts=lambda x: x["expiration_ts"].dt.strftime("%Y-%m-%d"),
                                     created_min=lambda x: x["created_min"].dt.strftime("%Y-%m-%d"))
    return {"t0_utc": str(pd.Timestamp(t0_ms, unit="ms")), "current": cur, "by_expiry": by_out.round(4).to_dict("records"),
            "forecast_monthly": fc.iloc[::15].round(4).to_dict("records"), "date_all_90pct_model": hit90,
            "date_existing_all_covered": hit100_lb}, fc


def q2_assumption(tr, snap, now_ms):
    """첫 체결(seq==1)부터 본 종목만. 필터별 |N|/OI(마지막 스냅샷) · 경로 위반율 · z."""
    first = tr.groupby("instrument_name")["trade_seq"].min()
    full = set(first[first == 1].index)
    trf = tr[tr["instrument_name"].isin(full)]
    s = snap[snap["instrument_name"].isin(full)].sort_values("ts")
    filters = {"all": np.ones(len(trf), bool), "no_block": ~trf["is_block"].to_numpy(),
               "no_block_no_liq": ~(trf["is_block"] | trf["is_liq"]).to_numpy(),
               "no_block_no_combo": ~(trf["is_block"] | trf["is_combo"]).to_numpy(),
               "small_only": (trf["amount"] <= SMALL).to_numpy() & ~trf["is_block"].to_numpy(),
               "large_only": (trf["amount"] > SMALL).to_numpy() | trf["is_block"].to_numpy()}
    out = {"n_full_instruments": len(full)}
    # 마지막 스냅샷(만기 전) OI
    s_live = s[s["ts"] < s["expiration_ts"].astype("int64") // 1000]
    lastoi = s_live.groupby("instrument_name").agg(ts=("ts", "max"))
    lastoi = s_live.merge(lastoi, on=["instrument_name", "ts"])[["instrument_name", "ts", "open_interest", "expiration_ts"]]
    lastoi["type"] = [exp_type(e.tz_convert(None)) for e in lastoi["expiration_ts"]]
    for nm, m in filters.items():
        t = trf[m]
        rows = []
        tt = t.merge(lastoi[["instrument_name", "ts"]], on="instrument_name")
        tt = tt[tt["timestamp"] <= tt["ts"]]
        tt = tt.assign(a2=tt["amount"] ** 2, rs=tt["amount"] * np.random.default_rng(7).choice([-1.0, 1.0], len(tt)))
        agg = tt.groupby("instrument_name").agg(N=("q", "sum"), vol=("amount", "sum"), ss=("a2", "sum"), nt=("q", "size"), Nr=("rs", "sum"))
        a = lastoi.set_index("instrument_name").join(agg, how="inner")
        a = a[a["open_interest"] > 0]
        a["r"] = a["N"].abs() / a["open_interest"]
        a["z"] = a["N"] / np.sqrt(a["ss"])
        res = {"n": len(a), "sum_absN_over_sum_oi": round(float(a["N"].abs().sum() / a["open_interest"].sum()), 3),
               "share_instr_absN_gt_oi": round(float((a["r"] > 1).mean()), 3),
               "oi_w_share_absN_gt_oi": round(float(a.loc[a["r"] > 1, "open_interest"].sum() / a["open_interest"].sum()), 3),
               "median_r": round(float(a["r"].median()), 3), "vol_over_oi_median": round(float((a["vol"] / a["open_interest"]).median()), 2),
               # z = N/sqrt(Σa²): 방향이 동전이면 |z| 중앙값 ≈ 0.674. 체결 30건 이상 종목만(1건 종목은 |z|≡1)
               "median_abs_z_nt30": round(float(a.loc[a["nt"] >= 30, "z"].abs().median()), 3), "n_nt30": int((a["nt"] >= 30).sum()),
               # 같은 체결 크기에 방향만 동전으로 바꾼 대조
               "null_random_sign_sum_absN_over_sum_oi": round(float(a["Nr"].abs().sum() / a["open_interest"].sum()), 3)}
        res["by_type"] = {ty: {"n": len(g), "sum_absN_over_sum_oi": round(float(g["N"].abs().sum() / g["open_interest"].sum()), 3),
                               "share_gt1": round(float((g["r"] > 1).mean()), 3)} for ty, g in a.groupby("type")}
        out[nm] = res
        if nm == "all":
            a_all = a
    # 경로: 모든 스냅샷에서 누적 N vs OI (전량 필터)
    ts_ = s[["instrument_name", "ts", "open_interest"]].copy()
    pts = []
    by_inst = dict(tuple(ts_.groupby("instrument_name")))
    for inst, g in trf.groupby("instrument_name"):
        gs = by_inst.get(inst)
        if gs is None:
            continue
        cN = np.cumsum(g["q"].to_numpy()); idx = np.searchsorted(g["timestamp"].to_numpy(), gs["ts"].to_numpy(), side="right")
        Ns = np.where(idx > 0, cN[np.maximum(idx - 1, 0)], 0.0)
        pts.append(pd.DataFrame({"inst": inst, "N": Ns, "oi": gs["open_interest"].to_numpy()}))
    P = pd.concat(pts)
    P = P[P["oi"] > 0]
    out["path_share_absN_gt_oi"] = round(float((P["N"].abs() > P["oi"]).mean()), 3)
    out["path_oiw_share_absN_gt_oi"] = round(float(P.loc[P["N"].abs() > P["oi"], "oi"].sum() / P["oi"].sum()), 3)
    sp = P.groupby("inst").apply(lambda g: g["N"].abs().corr(g["oi"], method="spearman") if len(g) > 5 else np.nan, include_groups=False)
    out["path_spearman_absN_vs_oi_median"] = round(float(sp.median()), 3)
    # 무작위 대조: 같은 체결 크기에 방향을 동전으로 → |N|/OI
    rng = np.random.default_rng(7)
    sims = []
    tt = trf.merge(lastoi[["instrument_name", "ts"]], on="instrument_name"); tt = tt[tt["timestamp"] <= tt["ts"]]
    for _ in range(20):
        qq = tt["amount"] * rng.choice([-1.0, 1.0], len(tt))
        n = qq.groupby(tt["instrument_name"]).sum().abs()
        o = lastoi.set_index("instrument_name")["open_interest"].reindex(n.index)
        sims.append(float(n.sum() / o[o > 0].sum()))
    out["random_sign_null_sum_absN_over_sum_oi"] = round(float(np.mean(sims)), 3)
    return out


def dealer_positions(tr, snap, t_eval: np.ndarray):
    """평가 시각들(t_eval, 스냅샷 ts)마다 종목별 딜러 포지션(방법별). 반환 long DataFrame[ts, inst, oi, F, iv, K, sg, exp, M*]."""
    snaps = snap[snap["ts"].isin(t_eval)].copy()
    res = []
    trs = tr.sort_values("timestamp")
    groups = dict(tuple(trs.groupby("instrument_name")))
    full_snap = snap.sort_values("ts")
    all_groups = dict(tuple(full_snap.groupby("instrument_name")))
    for inst, g in snaps.groupby("instrument_name"):
        t = groups.get(inst)
        gs = g.sort_values("ts")
        if t is None:
            N = {k: np.zeros(len(gs)) for k in ("M1", "M2", "M4")}; Mb = np.zeros(len(gs))
        else:
            tts = t["timestamp"].to_numpy()
            idx = np.searchsorted(tts, gs["ts"].to_numpy(), side="right")
            def cum(mask):
                c = np.concatenate([[0.0], np.cumsum(np.where(mask, t["q"].to_numpy(), 0.0))]); return c[idx]
            nb = ~t["is_block"].to_numpy()
            N = {"M1": -cum(np.ones(len(t), bool)), "M2": -cum(nb), "M4": -cum(nb & (t["amount"].to_numpy() <= SMALL))}
            # OI 제약판: 이 종목의 전체 스냅샷 창으로 돌린 뒤 평가 시각에서 뽑는다
            fa = all_groups[inst]; fts = fa["ts"].to_numpy(); foi = fa["open_interest"].to_numpy()
            w = assign_windows(tts, fts)
            netw = np.bincount(w, weights=t["q"].to_numpy(), minlength=len(fts) + 1)
            n_before = 0.0          # 첫 스냅샷 이전 체결은 창 0 에 이미 들어 있다(searchsorted left)
            path = oi_bounded(netw[:len(fts)], foi, n_before)
            Mb = pd.Series(path, index=fts).reindex(gs["ts"].to_numpy()).to_numpy()
        res.append(gs.assign(M0=gs["sg"] * gs["open_interest"], M1=N["M1"], M2=N["M2"], M4=N["M4"], M3=Mb))
    return pd.concat(res)


def exposures(P: pd.DataFrame, cols=("M0", "M1", "M2", "M3", "M4")):
    """스냅샷 ts × 범위(front/week/all)별 DEX$·GEX$ (딜러 기준)."""
    exp_ms = P["expiration_ts"].astype("int64") // 1000
    P = P[exp_ms > P["ts"]].copy(); exp_ms = exp_ms[P.index]
    T = (exp_ms - P["ts"]) / (365 * 86_400_000)
    F = P["underlying_price"].to_numpy()
    dl, gm = greeks(F, P["strike"].to_numpy(), (P["mark_iv"] / 100).to_numpy(), T.to_numpy(), P["sg"].to_numpy())
    P["dD"] = dl * F; P["gG"] = gm * F * F * 0.01
    P["dte"] = T * 365
    P["front"] = exp_ms == exp_ms.groupby(P["ts"]).transform("min")
    rows = []
    for (ts, ), g in P.groupby(["ts"]):
        for sc, sel in (("front", g["front"]), ("week", g["dte"] <= 7), ("all", g["dte"] > 0)):
            h = g[sel]
            r = {"ts": ts, "scope": sc}
            for c in cols:
                r[f"dex_{c}"] = float((h["dD"] * h[c]).sum()); r[f"gex_{c}"] = float((h["gG"] * h[c]).sum())
            rows.append(r)
    return pd.DataFrame(rows)


def q4_hedge(tr, pp, bars):
    """(i) 1분 고객 델타 흐름 X(ETH) → 페르프 테이커 순 Y(ETH) 같은 분·다음 1~5분."""
    t = tr[tr["timestamp"] >= pp["timestamp"].min()].copy()
    T = (t["exp"].astype("int64") // 1_000_000 - t["timestamp"]) / (365 * 86_400_000)
    dl, _ = greeks(t["index_price"].to_numpy(), t["K"].to_numpy(), (t["iv"] / 100).to_numpy(), T.to_numpy(), t["sg"].to_numpy())
    t["xd"] = t["q"] * dl
    t["m"] = t["timestamp"] // 60_000
    pp = pp.copy(); pp["m"] = pp["timestamp"] // 60_000
    pp["y"] = np.where(pp["direction"] == "buy", 1.0, -1.0) * pp["amount"] / pp["price"]      # USD 계약 → ETH
    mins = np.arange(pp["m"].min(), pp["m"].max() + 1)
    df = pd.DataFrame(index=mins)
    df["X"] = t.groupby("m")["xd"].sum(); df["Xs"] = t[~t["is_block"]].groupby("m")["xd"].sum(); df["Xb"] = t[t["is_block"]].groupby("m")["xd"].sum()
    df["Y"] = pp.groupby("m")["y"].sum()
    px = pp.groupby("m")["price"].last().reindex(mins).ffill()
    df["r"] = np.log(px).diff()
    df = df.fillna(0.0)
    df["Yf"] = sum(df["Y"].shift(-k) for k in range(1, 6))
    df["rf"] = sum(df["r"].shift(-k) for k in range(1, 6))
    df = df.dropna()
    out = {"minutes": len(df), "days": round(len(df) / 1440, 2)}
    # 회귀 Yf ~ X + Y + r (동시 가격·흐름 통제), 일 블록 부트스트랩 CI
    def beta(d, xcol):
        A = np.column_stack([np.ones(len(d)), d[xcol], d["Y"], d["r"]]); return np.linalg.lstsq(A, d["Yf"].to_numpy(), rcond=None)[0][1]
    day = (df.index // 1440).to_numpy()
    rng = np.random.default_rng(11); ud = np.unique(day)
    for xcol in ("X", "Xs", "Xb"):
        b = beta(df, xcol)
        bs = [beta(df[np.isin(day, rng.choice(ud, len(ud)))], xcol) for _ in range(300)]
        out[f"beta_next5_{xcol}"] = [round(b, 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        out[f"corr_same_min_{xcol}"] = round(float(df[xcol].corr(df["Y"])), 4)
    # 교란 확인: X 가 다음 5분 가격도 예측하나(정보거래면 +)
    out["corr_X_rf"] = round(float(df["X"].corr(df["rf"])), 4)
    out["corr_X_r_same"] = round(float(df["X"].corr(df["r"])), 4)
    return out


def q4_level(E: pd.DataFrame, bars: pd.DataFrame):
    """(ii) 매시 GEX(방법별, 범위별) vs 다음 4h 로그 변동폭(고−저). 스피어만(raw) + 통제판(직전 4h·24h 변동폭·시각 더미를
    순위에서 회귀로 뺀 잔차 상관 = 부분 스피어만). 일 블록 부트스트랩(복원추출) CI."""
    b = bars.copy(); b["h"] = b["ticks"] // 3_600_000
    hh = b.groupby("h").agg(hi=("high", "max"), lo=("low", "min"))
    hh = hh.reindex(np.arange(hh.index.min(), hh.index.max() + 1))
    def rng(a, k):   # 시각 a 부터 k 시간 로그 변동폭
        hi = hh["hi"].rolling(k).max().shift(-(k - 1)); lo = hh["lo"].rolling(k).min().shift(-(k - 1))
        return np.log(hi / lo)
    fwd4 = rng(None, 4)
    E = E.copy(); E["h"] = E["ts"] // 3_600_000 + 1       # 스냅샷 다음 정시부터 4시간
    E = E.drop_duplicates(["h", "scope"])
    E["y"] = E["h"].map(fwd4)
    E["p4"] = (E["h"] - 4).map(fwd4)                       # 직전 4시간(h−4..h−1)
    E["p24"] = (E["h"] - 24).map(rng(None, 24))
    E["hod"] = E["h"] % 24
    out = {}
    rngb = np.random.default_rng(5)
    def stats(g, c):
        rho = g[f"gex_{c}"].corr(g["y"], method="spearman")
        C = np.column_stack([np.ones(len(g)), g["p4"].rank(), g["p24"].rank(), pd.get_dummies(g["hod"]).to_numpy(float)[:, 1:]])
        rx = g[f"gex_{c}"].rank().to_numpy(); ry = g["y"].rank().to_numpy()
        def pc(C):
            ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]; ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
            return float(np.corrcoef(ex, ey)[0, 1])
        # 일 고정효과까지(하루 안 변동만) -- 여러 날 지속되는 레짐·추세 교란 제거
        Cd = np.column_stack([C, pd.get_dummies(g["h"] // 24).to_numpy(float)[:, 1:]])
        return rho, pc(C), pc(Cd)
    for sc, g in E.dropna(subset=["y", "p4", "p24"]).groupby("scope"):
        g = g[g["gex_M3"].notna()].reset_index(drop=True)
        day = (g["h"] // 24).to_numpy(); ud = np.unique(day); byd = {d: np.flatnonzero(day == d) for d in ud}
        o = {"n_hours": len(g), "n_days": len(ud)}
        for c in ("M0", "M1", "M2", "M3"):
            r0, r1, r2 = stats(g, c)
            bs = np.array([stats(g.iloc[np.concatenate([byd[d] for d in rngb.choice(ud, len(ud))])], c) for _ in range(300)])
            o[c] = {"raw": [round(r0, 3)] + [round(float(x), 3) for x in np.nanpercentile(bs[:, 0], [2.5, 97.5])],
                    "partial": [round(r1, 3)] + [round(float(x), 3) for x in np.nanpercentile(bs[:, 1], [2.5, 97.5])],
                    "partial_dayfe": [round(r2, 3)] + [round(float(x), 3) for x in np.nanpercentile(bs[:, 2], [2.5, 97.5])]}
        out[sc] = o
    return out


def block_direction_check(tr):
    """방향 규약 확인: 체결가−마크가 부호가 direction 과 같으면 direction = 가격을 양보한 쪽(테이커)."""
    t = tr[tr["mark_price"].notna() & (tr["mark_price"] > 0)].copy()
    t["edge"] = np.sign(t["price"] - t["mark_price"]) * t["sgn"]
    def frac(m):
        e = t.loc[m, "edge"]; e = e[e != 0]; return [round(float((e > 0).mean()), 3), int(len(e))]
    rfq = t["block_rfq_id"].notna() if "block_rfq_id" in t else pd.Series(False, index=t.index)
    return {"screen": frac(~t["is_block"]), "block_rfq": frac(t["is_block"] & rfq), "block_non_rfq": frac(t["is_block"] & ~rfq)}


def main():
    tr0, pp, ins, bars, snap0, db, now_ms = load_all()
    print(f"history 옵션 체결 {len(tr0):,} · perp {len(pp):,} · 스냅샷 {snap0['recorded_at_utc'].nunique()} · DB 체결 {len(db):,}", flush=True)
    tr, snap = prep(tr0, snap0)
    # 양성 대조: 수집기 DB 와 history 의 겹치는 구간 trade_id 일치
    dbe = db[db["instrument_name"].str.startswith("ETH-")]
    ov = tr[tr["timestamp"] >= dbe["ts_ms"].min()]
    ctrl = {"db_n": len(dbe), "hist_overlap_n": len(ov), "db_in_hist": round(float(dbe["trade_id"].isin(tr["trade_id"]).mean()), 5),
            "hist_in_db": round(float(ov["trade_id"].isin(dbe["trade_id"]).mean()), 5),
            "hist_has_block_field": bool("block_trade_id" in tr0), "hist_block_n": int(tr["is_block"].sum()), "hist_liq_n": int(tr["is_liq"].sum())}
    # 블록 행 표시가 history 에 없으면 DB 의 블록 표시를 빌려 온다(겹치는 구간만)
    if not ctrl["hist_has_block_field"]:
        tr["is_block"] = tr["trade_id"].isin(dbe.loc[dbe["is_block"], "trade_id"])
    tr = tr.merge(dbe[["trade_id", "block_rfq_id"]], on="trade_id", how="left", suffixes=("", "_db")) if "block_rfq_id" not in tr else tr
    print("대조", ctrl, flush=True)
    t0_ms = int(dbe["ts_ms"].min())
    q1, fc = q1_coverage(ins, snap, t0_ms, now_ms)
    fc.to_csv(OUT / "coverage_forecast.csv", index=False)
    print("Q1", json.dumps({k: v for k, v in q1.items() if k != "by_expiry"}, ensure_ascii=False)[:1500], flush=True)
    q2 = q2_assumption(tr, snap, now_ms)
    print("Q2", json.dumps(q2, ensure_ascii=False)[:3000], flush=True)
    bdc = block_direction_check(tr)
    print("방향규약", bdc, flush=True)
    # Q3: 매시 스냅샷(시각별 첫 스냅샷) + 최신
    st = snap[["ts"]].drop_duplicates().sort_values("ts")
    hourly = st.groupby(st["ts"] // 3_600_000)["ts"].min().to_numpy()
    t_eval = np.union1d(hourly, [snap["ts"].max()])
    P = dealer_positions(tr, snap, t_eval)
    E = exposures(P)
    E.to_parquet(OUT / "exposures_hourly.parquet")
    last = E[E["ts"] == E["ts"].max()].set_index("scope")
    q3 = {"latest_utc": str(pd.Timestamp(int(E["ts"].max()), unit="ms")),
          "latest": {sc: {k: round(v / 1e6, 2) for k, v in r.items() if k != "ts"} for sc, r in last.iterrows()}}
    agree = {}
    for sc, g in E.groupby("scope"):
        a = {}
        for c in ("M1", "M2", "M3", "M4"):
            a[c] = {"gex_sign_agree_M0": round(float((np.sign(g[f"gex_{c}"]) == np.sign(g["gex_M0"])).mean()), 3),
                    "dex_sign_agree_M0": round(float((np.sign(g[f"dex_{c}"]) == np.sign(g["dex_M0"])).mean()), 3),
                    "gex_sign_agree_M1": round(float((np.sign(g[f"gex_{c}"]) == np.sign(g["gex_M1"])).mean()), 3),
                    "gex_spearman_M0": round(float(g[f"gex_{c}"].corr(g["gex_M0"], method="spearman")), 3)}
        a["n_hours"] = len(g); agree[sc] = a
    q3["sign_agreement_hourly"] = agree
    # 딜러 포지션 수준 비교(최신, 전 만기): Σ|D| / ΣOI · 종목별 부호 일치(M1 vs M0)
    Pl = P[P["ts"] == P["ts"].max()]; Pl = Pl[Pl["open_interest"] > 0]
    q3["latest_positions"] = {c: {"sum_absD_over_oi": round(float(Pl[c].abs().sum() / Pl["open_interest"].sum()), 3),
                                  "oiw_sign_agree_M0": round(float(Pl.loc[np.sign(Pl[c]) == np.sign(Pl["M0"]), "open_interest"].sum() / Pl["open_interest"].sum()), 3)}
                              for c in ("M1", "M2", "M3", "M4")}
    q3["latest_positions"]["calls_short_by_M1_oiw"] = round(float(Pl.loc[(Pl["sg"] > 0) & (Pl["M1"] < 0), "open_interest"].sum() / Pl.loc[Pl["sg"] > 0, "open_interest"].sum()), 3)
    q3["latest_positions"]["puts_short_by_M1_oiw"] = round(float(Pl.loc[(Pl["sg"] < 0) & (Pl["M1"] < 0), "open_interest"].sum() / Pl.loc[Pl["sg"] < 0, "open_interest"].sum()), 3)
    print("Q3", json.dumps(q3, ensure_ascii=False)[:4000], flush=True)
    q4 = {"hedge_flow": q4_hedge(tr, pp, bars), "gex_vs_next4h_range": q4_level(E, bars)}
    print("Q4", json.dumps(q4, ensure_ascii=False), flush=True)
    (OUT / "results.json").write_text(json.dumps({"control": ctrl, "q1": q1, "q2": q2, "block_direction": bdc, "q3": q3, "q4": q4},
                                                 ensure_ascii=False, indent=1, default=str))
    print("저장", OUT / "results.json")


def selftest():
    # OI 제약판: 경로가 |D| ≤ OI 를 지키고, 순수 신규 흐름이면 −N 과 같다
    d = oi_bounded(np.array([5.0, 3.0, -2.0, 0.0]), np.array([5.0, 8.0, 10.0, 5.0]), 0.0)
    assert np.allclose(d, [-5.0, -8.0, -6.0, -3.0]), d
    d = oi_bounded(np.array([50.0, 0.0]), np.array([5.0, 5.0]), 0.0)
    assert np.allclose(d, [-5.0, -5.0])           # 테이커 순이 OI 를 넘어도 잘린다
    assert list(assign_windows(np.array([1, 5, 10, 11]), np.array([5, 10]))) == [0, 0, 1, 2]
    assert exp_type(pd.Timestamp("2026-12-25")) == "quarterly" and exp_type(pd.Timestamp("2026-10-30")) == "monthly"
    assert exp_type(pd.Timestamp("2026-10-09")) == "weekly" and exp_type(pd.Timestamp("2026-10-01")) == "daily"
    dl, gm = greeks(np.array([100.0, 100.0]), np.array([100.0, 100.0]), np.array([0.5, 0.5]), np.array([0.1, 0.1]), np.array([1.0, -1.0]))
    assert abs(dl[0] - dl[1] - 1.0) < 1e-9 and gm[0] == gm[1] > 0
    assert parse("ETH-25DEC26-3000-C")[0] == 3000.0 and parse("ETH-PERPETUAL") is None
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else main()
