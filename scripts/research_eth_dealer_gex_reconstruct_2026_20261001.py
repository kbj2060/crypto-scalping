#!/usr/bin/env python3
"""ETH 체결 기반 딜러 GEX 2026 재구성 → 예측 타당성 (2026-10-01, 사용자 «간접 검증 1번: 긴 기간 재구성 → 예측 타당성»).

딜러 = 테이커의 반대편. 종목별 딜러 순포지션 = −Σ(테이커 매수 − 매도), 종목 첫 체결(trade_seq 1)부터 누적.
감마$ = γ·F²·1% · 델타$ = δ·w·S (수집기 options_summary._gamma 와 같은 식). 범위 front(가장 가까운 만기)/week(7일 안)/all.

재구성(2026-01-01 ~ 09-30 매 정시 t):
  - 체결: 캐시(tmp/dealer_position_accuracy_20260930/opt_trades_hist.parquet, 2025-12-25~) + history.deribit.com 에서
    2026 에 살아 있던 12-25 이전 상장 종목의 seq 1 부터(종목별 seq 페이징) + 캐시 끝~지금 꼬리.
    t 의 포지션은 **ts < t** 체결만. 종목이 t 에서 «완결» = t 전 체결의 seq 가 1 부터 끊김 없음. 미완결은 빼고 커버 % 기록.
  - IV 대안(체인 스냅샷은 08-15~ 뿐): (A) 종목 직전 체결 iv · (B) A × DVOL(t)/DVOL(그 체결 시각).
    선도가 대안 = ETH-PERPETUAL 5분봉 종가(t 에 닫힌 봉). 08-15~ 스냅샷 mark_iv·underlying_price 로 잰 GEX 와 대조해
    week GEX 부호 일치율이 높은 쪽을 본 검정에 쓴다(결과가 아니라 재구성 오차로 고른다).

사전등록 (결과 보기 전 고정, 2026-10-01):
  라벨 y_k = log 실현변동성(ETH-PERPETUAL 5분 로그수익 제곱합의 제곱근), t **이후** 연 봉부터 k = 1·4·24h.
  통제 = log RV(직전 1h·24h·7d) + log DVOL(t 에 닫힌 1h 봉) + UTC 시 더미 23개 + 상수.
  X = 딜러 GEX$ 의 반기 내 백분위 순위(0~1) → β = «GEX 최저→최고일 때 log RV 차이».
  반기 = 전반 2026-01-01~05-31 · 후반 06-01~09-30. CI = 일 블록 부트스트랩 500회 95% 백분위.
  H1(핵심): week 범위, k=4h 에서 β < 0 이 **두 반기 모두 CI 0 배제**면 통과. 1h·24h·front·all 은 보조(같은 기준으로 표시).
       일 고정효과판(하루 안 변동만)도 두 반기 CI<0 이면 «강건». 보조 척도: GEX$ / 직전 7일 perp 시간당 USD 거래대금.
  H2: week 딜러 플립이 있을 때 below = (S < 플립) 더미의 β(k=4h) > 0 이 두 반기 CI 0 배제면 통과(플립 없는 시각 제외).
  H3(대조군): 관행 GEX(콜 +OI · 풋 −OI)는 미결제가 필요해 스냅샷 구간(08-15~09-30)에서만 가능 — 그 구간을 둘로 나눠
       같은 회귀로 딜러(체결·mark_iv) vs 관행 β 를 나란히. «체결 기반이 낫다» = 딜러 β 가 두 반쪽 모두 CI<0 이고 관행은 아님.
  누수 점검: GEX 를 1시간 늦춰(t−1h 값으로 t 라벨) 결과가 유지되는지. 옵션 데이터는 2026 만 쓴다(사용자 규칙).
출력 tmp/dealer_gex_reconstruct_2026_20261001/ : hourly_dealer_gex.parquet(시간별 재구성) · iv_proxy_validation.parquet · results.json.

사용:
  python scripts/research_eth_dealer_gex_reconstruct_2026_20261001.py
  python scripts/research_eth_dealer_gex_reconstruct_2026_20261001.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_eth_dealer_position_accuracy_20260930 as D  # noqa: E402  parse · _get · fetch_paged · HIST · LIVE

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "tmp/dealer_position_accuracy_20260930"
OUT = ROOT / "tmp/dealer_gex_reconstruct_2026_20261001"
H_MS, B_MS = 3_600_000, 300_000
T0 = pd.Timestamp("2026-01-01", tz="UTC").value // 10**6
CACHE_START = D.HIST_START_MS
HALF = pd.Timestamp("2026-06-01", tz="UTC").value // 10**6
SCOPES = ("front", "week", "all")
NB = 500


# ───────────────────────── 받기(캐시) ─────────────────────────
def fetch_prelisted(cache_min_seq: pd.Series) -> pd.DataFrame:
    """2026 에 살아 있던 12-25 이전 상장 종목: seq 1 부터 캐시 첫 seq 직전까지(캐시에 없으면 끝까지)."""
    p = OUT / "prelisted_trades.parquet"
    if p.exists():
        return pd.read_parquet(p)
    ins = pd.DataFrame(json.loads((OUT / "hist_instruments.json").read_text())["result"]) if (OUT / "hist_instruments.json").exists() \
        else pd.DataFrame(D._get(D.HIST, "get_instruments", currency="ETH", kind="option", expired="true", include_old="true"))
    ins = ins[ins["instrument_name"].str.match(r"^ETH-\d")]
    pre = ins[(ins["expiration_timestamp"] > T0) & (ins["creation_timestamp"] < CACHE_START)]["instrument_name"]
    names = sorted(set(pre) | set(cache_min_seq[(cache_min_seq > 1)].index[
        [D.parse(n) is not None and D.parse(n)[1].timestamp() * 1000 > T0 for n in cache_min_seq[cache_min_seq > 1].index]]))
    rows = []
    for i, n in enumerate(names):
        stop = int(cache_min_seq.get(n, 10**9)) - 1
        s = 1
        while s <= stop:
            e = min(s + 999, stop)
            tr = D._get(D.HIST, "get_last_trades_by_instrument", instrument_name=n, start_seq=s, end_seq=e,
                        count=1000, include_old="true", sorting="asc")["trades"]
            rows += tr
            if len(tr) < e - s + 1:
                break
            s = e + 1
            time.sleep(0.1)
        if i % 50 == 0:
            print(f"  사전상장 {i}/{len(names)} {n} 누적 {len(rows):,}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("trade_id")
    df.to_parquet(p)
    return df


def fetch_series(now_ms: int):
    """ETH-PERPETUAL 5분봉(2025-12-23~) · DVOL 1h."""
    p, q = OUT / "perp_5m.parquet", OUT / "dvol_1h.parquet"
    t_start = T0 - 9 * 86_400_000
    if not p.exists():
        bars, t0 = [], t_start
        while t0 < now_ms:
            t1 = min(t0 + 14 * 86_400_000, now_ms)
            r = D._get(D.LIVE, "get_tradingview_chart_data", instrument_name="ETH-PERPETUAL", start_timestamp=t0, end_timestamp=t1, resolution="5")
            bars.append(pd.DataFrame({k: r[k] for k in ("ticks", "close", "cost")})); t0 = t1
        b = pd.concat(bars).drop_duplicates("ticks").sort_values("ticks")
        b = b[b["ticks"] + B_MS <= now_ms]                       # 형성 중 봉 버림
        b.to_parquet(p)
    if not q.exists():
        rows, t0 = [], t_start
        while t0 < now_ms:
            t1 = min(t0 + 30 * 86_400_000, now_ms)
            rows += D._get(D.LIVE, "get_volatility_index_data", currency="ETH", resolution="3600", start_timestamp=t0, end_timestamp=t1)["data"]
            t0 = t1
        v = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"]).drop_duplicates("ts").sort_values("ts")
        v = v[v["ts"] + H_MS <= now_ms]
        v.to_parquet(q)
    return pd.read_parquet(p), pd.read_parquet(q)


def load_trades(now_ms: int) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["trade_id", "trade_seq", "timestamp", "instrument_name", "direction", "amount", "iv"]
    c = pd.read_parquet(CACHE / "opt_trades_hist.parquet", columns=cols)
    tail_p = OUT / "tail_trades.parquet"
    tail = pd.read_parquet(tail_p) if tail_p.exists() else D.fetch_paged(
        tail_p, "get_last_trades_by_currency_and_time", {"currency": "ETH", "kind": "option"}, int(c["timestamp"].max()), now_ms)
    pre = fetch_prelisted(c.groupby("instrument_name")["trade_seq"].min())
    tr = pd.concat([c, tail[cols], pre[cols]]).drop_duplicates("trade_id")
    pr = tr["instrument_name"].map(D.parse)
    tr = tr[pr.notna()].copy(); pr = pr[pr.notna()]
    tr["K"] = [x[0] for x in pr]; tr["exp_ms"] = [int(x[1].timestamp() * 1000) for x in pr]; tr["sg"] = [x[2] for x in pr]
    tr = tr[tr["exp_ms"] > T0 - 9 * 86_400_000]
    tr["q"] = np.where(tr["direction"] == "buy", 1.0, -1.0) * tr["amount"]
    tr["iv"] = tr["iv"].where(tr["iv"] > 0)
    return tr.sort_values(["instrument_name", "timestamp", "trade_seq"]).reset_index(drop=True)


# ───────────────────────── 재구성 ─────────────────────────
def positions_at(tr: pd.DataFrame, t_eval: np.ndarray) -> pd.DataFrame:
    """평가 시각마다 종목별 딜러 포지션(ts < t 체결만). 행 = (평가 인덱스 k, 종목) — t 전 체결이 1건 이상이고 t < 만기인 것만.
    cov = t 전 체결의 seq 가 1 부터 끊김 없음. iv_last = t 전 마지막 유효 iv · t_last = 그 체결 시각."""
    names = tr["instrument_name"].to_numpy(); cut = np.flatnonzero(names[1:] != names[:-1]) + 1
    bounds = np.concatenate([[0], cut, [len(tr)]])
    ts_all, q_all, seq_all = tr["timestamp"].to_numpy(), tr["q"].to_numpy(), tr["trade_seq"].to_numpy()
    iv_all = tr["iv"].to_numpy(); amt_all = tr["amount"].to_numpy()
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        ts = ts_all[a:b]; exp = int(tr["exp_ms"].iat[a])
        lo = np.searchsorted(t_eval, ts[0], "right"); hi = np.searchsorted(t_eval, exp, "left")
        if hi <= lo:
            continue
        te = t_eval[lo:hi]
        n = np.searchsorted(ts, te, "left")                 # t 전 체결 건수(≥1)
        seq = seq_all[a:b]
        brk = np.flatnonzero(seq != np.arange(1, len(seq) + 1))
        ok_n = brk[0] if len(brk) else len(seq)             # 앞에서부터 끊김 없는 체결 수
        ivs = pd.Series(iv_all[a:b]).ffill().to_numpy(); tiv = pd.Series(np.where(np.isnan(iv_all[a:b]), np.nan, ts)).ffill().to_numpy()
        out.append(pd.DataFrame({"k": np.arange(lo, hi), "inst": names[a], "K": tr["K"].iat[a], "sg": tr["sg"].iat[a], "exp_ms": exp,
                                 "pos": -np.cumsum(q_all[a:b])[n - 1], "cov": n <= ok_n, "vol": np.cumsum(amt_all[a:b])[n - 1],
                                 "iv_last": ivs[n - 1], "t_last": tiv[n - 1]}))
    return pd.concat(out, ignore_index=True)


def bs(F, K, iv_pct, T):
    """(콜 델타 N(d1), 감마). r=0. 무효 입력은 0."""
    ok = (iv_pct > 0) & (T > 0) & (F > 0) & (K > 0)
    s = np.where(ok, iv_pct / 100.0, 1.0); T_ = np.where(ok, T, 1.0); F_ = np.where(ok, F, 1.0)
    d1 = (np.log(F_ / np.where(ok, K, 1.0)) + 0.5 * s * s * T_) / (s * np.sqrt(T_))
    return np.where(ok, ndtr(d1), 0.0), np.where(ok, np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (F_ * s * np.sqrt(T_)), 0.0)


def scope_masks(P: pd.DataFrame, t_eval: np.ndarray) -> dict:
    tt = t_eval[P["k"].to_numpy()]
    front = P["exp_ms"].to_numpy() == P.groupby("k")["exp_ms"].transform("min").to_numpy()
    return {"front": front, "week": (P["exp_ms"].to_numpy() - tt) <= 7 * 86_400_000, "all": np.ones(len(P), bool)}


def exposures(P: pd.DataFrame, t_eval: np.ndarray, S: np.ndarray, iv_col: str, w_col: str = "w", fwd_col: str | None = None) -> pd.DataFrame:
    """시각 k × 범위별 GEX$ · DEX$ · 플립(±15% 25점 곡선에서 S 에 가장 가까운 부호 전환, 없으면 NaN).
    F = 행의 선도가(fwd_col, 없으면 S). 가격점 px 로 옮길 때 F·px/S (수집기와 같다)."""
    k = P["k"].to_numpy(); nk = len(t_eval)
    T = (P["exp_ms"].to_numpy() - t_eval[k]) / (365 * 86_400_000)
    F0 = P[fwd_col].to_numpy() if fwd_col else S[k]; F0 = np.where(F0 > 0, F0, S[k])
    K, sg, iv, w = P["K"].to_numpy(), P["sg"].to_numpy(), P[iv_col].to_numpy(), P[w_col].to_numpy()
    iv = np.nan_to_num(iv, nan=0.0)
    m = scope_masks(P, t_eval)
    grid = 0.85 + 0.0125 * np.arange(25)
    prof = {sc: np.zeros((nk, 25)) for sc in SCOPES}
    res = {}
    for j, x in enumerate(grid):
        F = F0 * x
        nd1, gam = bs(F, K, iv, T)
        gg = gam * F * F * 0.01 * w
        for sc in SCOPES:
            prof[sc][:, j] = np.bincount(k, weights=np.where(m[sc], gg, 0.0), minlength=nk)
        if j == 12:
            dd = np.where(sg > 0, nd1, nd1 - 1.0) * w * S[k]
            for sc in SCOPES:
                res[f"gex_{sc}"] = prof[sc][:, 12].copy()
                res[f"dex_{sc}"] = np.bincount(k, weights=np.where(m[sc], dd, 0.0), minlength=nk)
    for sc in SCOPES:
        res[f"flip_{sc}"] = flip_of(prof[sc], S, grid)
    return pd.DataFrame(res)


def flip_of(prof: np.ndarray, S: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """행마다 곡선의 부호 전환(선형 보간) 중 S 에 가장 가까운 것. 없으면 NaN."""
    px = S[:, None] * grid[None, :]
    a, b = prof[:, :-1], prof[:, 1:]
    ch = (a < 0) != (b < 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cand = px[:, :-1] + (px[:, 1:] - px[:, :-1]) * (-a) / (b - a)
    d = np.where(ch, np.abs(cand - S[:, None]), np.inf)
    j = d.argmin(1)
    return np.where(np.isfinite(d.min(1)), cand[np.arange(len(S)), j], np.nan)


# ───────────────────────── 라벨·통제 ─────────────────────────
class Bars:
    def __init__(self, bars: pd.DataFrame, dvol: pd.DataFrame):
        g = np.arange(bars["ticks"].min(), bars["ticks"].max() + 1, B_MS)
        b = bars.set_index("ticks").reindex(g)
        self.missing = int(b["close"].isna().sum())
        self.open = g; self.close = b["close"].ffill().to_numpy(); cost = b["cost"].fillna(0.0).to_numpy()
        r = np.diff(np.log(self.close), prepend=np.log(self.close[0]))
        self.c2 = np.concatenate([[0.0], np.cumsum(r * r)]); self.cc = np.concatenate([[0.0], np.cumsum(cost)])
        self.dvol = dvol.set_index("ts")["c"]

    def at(self, s: np.ndarray) -> pd.DataFrame:
        """결정 시각 s(ms): 과거 = 닫힌 봉(open+5m ≤ s) · 미래 = open ≥ s 인 봉부터. 범위 밖은 NaN."""
        fut = np.searchsorted(self.open, s, "left"); end = np.searchsorted(self.open, s - B_MS, "right")
        n = len(self.open)
        def rv(a, bnd):
            ok = (a >= 1) & (bnd <= n) & (a < bnd)
            return np.where(ok, np.sqrt(self.c2[np.clip(bnd, 0, n)] - self.c2[np.clip(a, 0, n)]), np.nan)
        out = {"S": np.where(end >= 1, self.close[np.maximum(end - 1, 0)], np.nan)}
        for h in (1, 4, 24):
            out[f"y{h}"] = np.log(rv(fut, fut + 12 * h) + 1e-6)
        for h, nm in ((1, "p1"), (24, "p24"), (168, "p168")):
            out[nm] = np.log(rv(end - 12 * h, end) + 1e-6)
        out["vol7d"] = np.where(end >= 2016, (self.cc[end] - self.cc[np.maximum(end - 2016, 0)]) / 168, np.nan)
        out["dvol"] = np.log(self.dvol.reindex((s // H_MS) * H_MS - H_MS).to_numpy())
        out["hod"] = (s // H_MS) % 24
        out["day"] = s // 86_400_000
        return pd.DataFrame(out)


# ───────────────────────── 회귀 ─────────────────────────
def beta_test(df: pd.DataFrame, x: str, y: str, fe: bool, rank: bool = True, seed: int = 3) -> dict:
    d = df[[x, y, "p1", "p24", "p168", "dvol", "hod", "day"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 200:
        return {"n": len(d)}
    xv = d[x].rank(pct=True).to_numpy() if rank else d[x].to_numpy(float)
    C = np.column_stack([xv, d[["p1", "p24", "p168", "dvol"]].to_numpy(), pd.get_dummies(d["hod"]).to_numpy(float)[:, 1:], np.ones(len(d))])
    yv = d[y].to_numpy(); day = d["day"].to_numpy()
    if fe:        # 일 고정효과 = 날마다 평균 빼기(상수는 사라진다)
        C = C[:, :-1]
        for arr in (C, yv):
            mu = pd.DataFrame(arr).groupby(day).transform("mean").to_numpy()
            arr -= mu.reshape(arr.shape)
    b = np.linalg.lstsq(C, yv, rcond=None)[0][0]
    ud = np.unique(day); byd = {u: np.flatnonzero(day == u) for u in ud}
    rng = np.random.default_rng(seed); bs_ = []
    for _ in range(NB):
        ix = np.concatenate([byd[u] for u in rng.choice(ud, len(ud))])
        bs_.append(np.linalg.lstsq(C[ix], yv[ix], rcond=None)[0][0])
    lo, hi = np.percentile(bs_, [2.5, 97.5])
    return {"beta": round(float(b), 4), "ci": [round(float(lo), 4), round(float(hi), 4)], "n": len(d), "days": len(ud)}


def halves(df, x, y, rank=True, split=HALF, col="ts"):
    return {nm: {"ctrl": beta_test(g, x, y, False, rank), "dayfe": beta_test(g, x, y, True, rank)}
            for nm, g in (("H_a", df[df[col] < split]), ("H_b", df[df[col] >= split]))}


def verdict(r: dict, sign: int) -> dict:
    def ok(t):
        return "ci" in t and (t["ci"][1] < 0 if sign < 0 else t["ci"][0] > 0)
    return {"pass_ctrl": all(ok(r[h]["ctrl"]) for h in ("H_a", "H_b")), "pass_dayfe": all(ok(r[h]["dayfe"]) for h in ("H_a", "H_b"))}


# ───────────────────────── 메인 ─────────────────────────
def main():
    now_ms = int(time.time() * 1000)
    tr = load_trades(now_ms)
    bars, dvol = fetch_series(now_ms)
    B = Bars(bars, dvol)
    print(f"체결 {len(tr):,} · 종목 {tr['instrument_name'].nunique():,} · 5분봉 {len(B.open):,}(빈 봉 {B.missing}) · DVOL {len(dvol)}", flush=True)
    res = {"n_trades": len(tr), "bars_missing": B.missing}

    # 완결성 요약: 2026 에 살아 있던 종목 중 seq 1 부터 끊김 없는 비율
    g = tr[tr["exp_ms"] > T0].groupby("instrument_name")["trade_seq"].agg(["min", "max", "size"])
    res["instr_complete"] = {"n": len(g), "complete": int(((g["min"] == 1) & (g["max"] == g["size"])).sum())}
    print("완결", res["instr_complete"], flush=True)

    # ── 1) 매 정시 재구성
    t_eval = np.arange(T0, (now_ms // H_MS) * H_MS + 1, H_MS)
    L = B.at(t_eval); S = L["S"].to_numpy()
    P = positions_at(tr, t_eval)
    dv = B.dvol
    P["iv_A"] = P["iv_last"]
    dv_now = dv.reindex((t_eval[P["k"]] // H_MS) * H_MS - H_MS).to_numpy()
    dv_then = dv.reindex((P["t_last"].to_numpy() // H_MS).astype("int64") * H_MS - H_MS).to_numpy()
    P["iv_B"] = P["iv_last"] * dv_now / dv_then
    P["w"] = np.where(P["cov"], P["pos"], 0.0)
    cov_vol = pd.DataFrame({"k": P["k"], "v": P["vol"], "vc": np.where(P["cov"], P["vol"], 0.0)}).groupby("k").sum()
    res["coverage_volume_weighted"] = {"min": round(float((cov_vol["vc"] / cov_vol["v"]).min()), 5),
                                       "mean": round(float((cov_vol["vc"] / cov_vol["v"]).mean()), 5)}
    print("커버(누적 거래량 가중)", res["coverage_volume_weighted"], flush=True)

    # ── 2) IV 대안 검증(스냅샷 08-15~)
    snap = pd.read_parquet(ROOT / "tmp/dpa20260930/snap.parquet")
    snap["ts"] = snap["recorded_at_utc"].astype("int64") // 1000
    st = np.sort(snap["ts"].unique()); s_eval = pd.Series(st).groupby(st // H_MS).min().to_numpy()   # 시각별 첫 스냅샷
    Ls = B.at(s_eval); Ss = Ls["S"].to_numpy()
    Ps = positions_at(tr, s_eval)
    sn = snap[snap["ts"].isin(s_eval)].copy(); sn["k"] = np.searchsorted(s_eval, sn["ts"].to_numpy())
    Ps = Ps.merge(sn[["k", "instrument_name", "mark_iv", "underlying_price", "open_interest"]].rename(columns={"instrument_name": "inst"}),
                  on=["k", "inst"], how="inner")
    Ps["iv_A"] = Ps["iv_last"]
    Ps["iv_B"] = Ps["iv_last"] * dv.reindex((s_eval[Ps["k"]] // H_MS) * H_MS - H_MS).to_numpy() / \
        dv.reindex((Ps["t_last"].to_numpy() // H_MS).astype("int64") * H_MS - H_MS).to_numpy()
    Ps["w"] = np.where(Ps["cov"], Ps["pos"], 0.0)
    Etrue = exposures(Ps, s_eval, Ss, "mark_iv", fwd_col="underlying_price")
    val = {"n_snapshots": len(s_eval)}
    EA = exposures(Ps, s_eval, Ss, "iv_A"); EB = exposures(Ps, s_eval, Ss, "iv_B")
    for nm, E in (("A_last_trade_iv", EA), ("B_dvol_scaled", EB)):
        v = {}
        for sc in SCOPES:
            a, t_ = E[f"gex_{sc}"], Etrue[f"gex_{sc}"]
            fa, ft = E[f"flip_{sc}"], Etrue[f"flip_{sc}"]
            both = fa.notna() & ft.notna()
            v[sc] = {"gex_corr": round(float(a.corr(t_)), 4), "gex_spearman": round(float(a.corr(t_, method="spearman")), 4),
                     "gex_sign_agree": round(float((np.sign(a) == np.sign(t_)).mean()), 4),
                     "gex_median_abs_rel_err": round(float(((a - t_).abs() / t_.abs()).median()), 4),
                     "dex_corr": round(float(E[f"dex_{sc}"].corr(Etrue[f"dex_{sc}"])), 4),
                     "flip_exists_agree": round(float((fa.notna() == ft.notna()).mean()), 4),
                     "flip_median_abs_pct_diff": round(float(((fa - ft).abs() / Ss)[both].median() * 100), 3),
                     "below_flip_agree": round(float(((Ss < fa) == (Ss < ft))[both].mean()), 4)}
        wgt = Ps["w"].abs() > 0
        v["iv_median_abs_err_pt"] = round(float((Ps.loc[wgt, f"iv_{nm[0]}"] - Ps.loc[wgt, "mark_iv"]).abs().median()), 3)
        val[nm] = v
    best = "A" if val["A_last_trade_iv"]["week"]["gex_sign_agree"] >= val["B_dvol_scaled"]["week"]["gex_sign_agree"] else "B"
    val["chosen"] = best
    # 스냅샷 구간 커버(미결제 가중): 체인 전 종목 중 완결 종목 미결제 비중
    snc = sn.merge(Ps[["k", "inst", "cov"]].rename(columns={"inst": "instrument_name"}), on=["k", "instrument_name"], how="left")
    snc = snc[snc["expiration_ts"].astype("int64") // 1000 > snc["ts"]]
    snc["cov"] = snc["cov"].fillna(False).astype(bool)      # 체결 없는 종목(딜러 0)은 미결제도 0 이 대부분 -- 아래 비율에 그대로 둔다
    cr = snc.groupby("k").apply(lambda g: g.loc[g["cov"], "open_interest"].sum() / max(g["open_interest"].sum(), 1e-9), include_groups=False)
    val["oi_weighted_coverage"] = {"min": round(float(cr.min()), 4), "mean": round(float(cr.mean()), 4)}
    res["iv_proxy_validation"] = val
    pd.concat([pd.DataFrame({"ts": s_eval, "S": Ss}), Etrue.add_prefix("true_"), EA.add_prefix("A_"), EB.add_prefix("B_")], axis=1) \
        .to_parquet(OUT / "iv_proxy_validation.parquet")
    print("IV 대안", json.dumps(val, ensure_ascii=False), flush=True)

    # ── 3) 본 재구성(선택한 IV)
    E = exposures(P, t_eval, S, f"iv_{best}")
    cnt = pd.DataFrame({"k": P["k"], "c": P["cov"].astype(int), "n": 1}).groupby("k").sum().reindex(range(len(t_eval))).fillna(0)
    Hh = pd.concat([pd.DataFrame({"ts": t_eval, "n_inst": cnt["n"].to_numpy(), "n_cov": cnt["c"].to_numpy()}), L, E], axis=1)
    Hh["cov_vol"] = (cov_vol["vc"] / cov_vol["v"]).reindex(range(len(t_eval))).to_numpy()
    Hh["iv_proxy"] = best
    Hh.to_parquet(OUT / "hourly_dealer_gex.parquet")
    print("시간별 저장", OUT / "hourly_dealer_gex.parquet", len(Hh), flush=True)

    # ── 4) 검정
    Hh = Hh[Hh["ts"] < pd.Timestamp("2026-10-01", tz="UTC").value // 10**6].copy()
    for sc in SCOPES:
        Hh[f"gexn_{sc}"] = Hh[f"gex_{sc}"] / Hh["vol7d"]
        Hh[f"gexlag_{sc}"] = Hh[f"gex_{sc}"].shift(1)
    Hh["below"] = np.where(Hh["flip_week"].notna(), (Hh["S"] < Hh["flip_week"]).astype(float), np.nan)
    h1 = {}
    for sc in SCOPES:
        for hz in (1, 4, 24):
            r = halves(Hh, f"gex_{sc}", f"y{hz}")
            h1[f"{sc}_{hz}h"] = {**r, **verdict(r, -1)}
    h1["week_4h_norm_by_perp_volume"] = {**(r := halves(Hh, "gexn_week", "y4")), **verdict(r, -1)}
    h1["week_4h_lag1h_leak_check"] = {**(r := halves(Hh, "gexlag_week", "y4")), **verdict(r, -1)}
    res["H1"] = h1
    print("H1 week 4h", json.dumps(h1["week_4h"], ensure_ascii=False), flush=True)
    h2 = {**(r := halves(Hh, "below", "y4", rank=False)), **verdict(r, +1),
          "share_hours_with_flip": round(float(Hh["flip_week"].notna().mean()), 3), "share_below": round(float(Hh["below"].mean()), 3)}
    res["H2"] = h2
    print("H2", json.dumps(h2, ensure_ascii=False), flush=True)

    # H3: 스냅샷 구간 — 딜러(mark_iv) vs 관행(콜 +OI · 풋 −OI), 체인 전 종목
    sn2 = sn[sn["expiration_ts"].astype("int64") // 1000 > sn["ts"]].rename(columns={"instrument_name": "inst"}).copy()
    sn2["K"] = sn2["strike"]; sn2["sg"] = np.where(sn2["option_type"] == "call", 1.0, -1.0)
    sn2["exp_ms"] = sn2["expiration_ts"].astype("int64") // 1000; sn2["w"] = sn2["sg"] * sn2["open_interest"]
    Easm = exposures(sn2, s_eval, Ss, "mark_iv", fwd_col="underlying_price")
    H3 = pd.concat([pd.DataFrame({"ts": s_eval}), Ls, Etrue.add_prefix("d_"), Easm.add_prefix("a_")], axis=1)
    mid = int(np.median(s_eval))
    h3 = {"split_utc": str(pd.Timestamp(mid, unit="ms")), "n_hours": len(H3)}
    for nm, col in (("dealer_week", "d_gex_week"), ("assumed_week", "a_gex_week"), ("dealer_all", "d_gex_all"), ("assumed_all", "a_gex_all"),
                    ("dealer_front", "d_gex_front"), ("assumed_front", "a_gex_front")):
        h3[nm] = {**(r := halves(H3, col, "y4", split=mid)), **verdict(r, -1)}
    h3["corr_dealer_vs_assumed_week"] = round(float(H3["d_gex_week"].corr(H3["a_gex_week"], method="spearman")), 3)
    res["H3"] = h3
    print("H3", json.dumps(h3, ensure_ascii=False), flush=True)
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장", OUT / "results.json")


def selftest():
    # 합성 체결: 종목 X(콜, 만기 10시간) 테이커 매수 3 @t=1h+1, 매도 1 @t=2h(정각) · 종목 Y(seq 끊김)
    h = H_MS
    tr = pd.DataFrame({"instrument_name": ["X", "X", "Y", "Y"], "timestamp": [h + 1, 2 * h, h + 5, 2 * h + 5], "trade_seq": [1, 2, 1, 3],
                       "q": [3.0, -1.0, 2.0, 1.0], "amount": [3.0, 1.0, 2.0, 1.0], "iv": [50.0, np.nan, 60.0, 60.0],
                       "K": [100.0, 100.0, 100.0, 100.0], "sg": [1.0, 1.0, -1.0, -1.0], "exp_ms": [10 * h, 10 * h, 10 * h, 10 * h]})
    t_eval = np.array([h, 2 * h, 3 * h, 10 * h])
    P = positions_at(tr, t_eval).set_index(["k", "inst"])
    assert (0, "X") not in P.index                              # t=1h 에는 체결 전
    assert P.loc[(1, "X"), "pos"] == -3.0                       # t=2h: 정각 체결(ts=2h)은 아직 안 봄
    assert P.loc[(2, "X"), "pos"] == -2.0 and P.loc[(2, "X"), "iv_last"] == 50.0
    assert (3, "X") not in P.index                              # 만기 시각엔 없음
    assert P.loc[(1, "Y"), "cov"] and not P.loc[(2, "Y"), "cov"]   # seq 2 빠진 체결 뒤부터 미완결
    # GEX 부호: 딜러가 콜을 판(테이커 매수) → 음감마, 산 → 양감마. 풋도 같은 규칙(감마는 콜·풋 같음)
    Q = pd.DataFrame({"k": [0, 1], "K": [100.0, 100.0], "sg": [1.0, -1.0], "exp_ms": [30 * 86_400_000] * 2, "w": [-5.0, 5.0], "iv": [50.0, 50.0]})
    E = exposures(Q, np.array([0, 0]), np.array([100.0, 100.0]), "iv")
    assert E["gex_all"][0] < 0 < E["gex_all"][1] and E["gex_week"][0] == 0     # 30일 만기는 week 밖
    assert E["dex_all"][0] < 0 and E["dex_all"][1] < 0                          # 딜러 콜 숏 = 델타 −, 딜러 풋 롱 = 델타 −
    # 플립: 곡선이 S 위에서 + → − 로 바뀌면 그 사이
    prof = np.array([[1.0, 1.0, -1.0, -1.0]]); f = flip_of(prof, np.array([100.0]), np.array([0.9, 1.0, 1.1, 1.2]))
    assert abs(f[0] - 105.0) < 1e-9
    assert np.isnan(flip_of(np.ones((1, 4)), np.array([100.0]), np.array([0.9, 1.0, 1.1, 1.2]))[0])
    # 라벨은 결정 시각 뒤 봉만: 가격이 t 이후에만 움직이면 과거 RV≈0, 미래 RV>0
    ticks = np.arange(0, 400 * B_MS, B_MS); close = np.where(ticks >= 200 * B_MS, 101.0 + (ticks // B_MS) % 2, 100.0)
    Bx = Bars(pd.DataFrame({"ticks": ticks, "close": close, "cost": 1.0}), pd.DataFrame({"ts": [0], "c": [50.0]}))
    L = Bx.at(np.array([199 * B_MS + B_MS]))                    # t = 200 번째 봉 open(= 199 번째 봉 종가 시각)
    assert L["p1"][0] < -10 and L["y1"][0] > -10 and L["S"][0] == 100.0
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else main()
