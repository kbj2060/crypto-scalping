"""짧은 지평에서 기각된 ETH 흐름 실험 4계열을 긴 지평(4h·12h·1d·3d)으로 재검정 (2026-10-16).

계열 (정의는 원 연구 그대로, 지평만 늘림 · 문턱 재조정 없음)
 F1 CVD 다이버전스  5분봉 4h(48봉) 신고가/신저가인데 CVD 는 직전 48봉 극값을 못 넘음 → 반대(페이드).
                    원: 2h 연장(+0.5ATR) 실패 +3.1~3.5pp (orderflow_report5_revalidation_20261011).
                    목적 지표 = H 안 연장 실패(문턱 0.25·0.5·1 ATR, ATR=5분 12봉) · 방향 = 페이드 수익.
                    가격만 대조 = 같은 신고가 사건 전체(CVD 무관) 페이드.
 F2 체결↔가격 불일치 W=4h·1d 비겹침 봉, sign(봉 순매수) ≠ sign(봉 수익) → 가격 방향 추종.
                    대조 = 일치 봉 |수익| 십분위 × |체결 z| 삼분위 매칭 (flow_price_divergence_no_edge_20260925).
 F3 고래↔리테일 맞대결 1분 테이프 row 단위, 직전 W분 그룹 순매수 z(자기 과거 30일 창합 SD), 반대·|z|≥0.5 → 고래 쪽.
                    (a) 원 사건 W=60분 그대로 H 보유 · (b) W 를 지평에 맞춤 4h/12h/1d, 3d 는 W=1d.
 F4 청산 버스트 프록시 5분 |수익| ≥ 과거 30일 q99 & ΔOI ≤ 과거 30일 q05(청산형) vs ΔOI>0(비청산형) → 추종/페이드.
                    참고 셀: doi_12h 하위 10%(인과 확장창·당일 제외·최소 56일) → 롱 (direction_past_failures_20260915).

공통: 사건은 결정 봉 τ 까지의 데이터로만, 수익은 τ 종가 → τ+H 종가. 비용 4bp 왕복. 주 2025-01-01~ / 보조 그 이전.
통과 = 주 순평균>0 & 일(3d 는 주) 블록 CI 하한>0 & 위약(같은 기간 무작위 시각 · 같은 방향 구성, 1000회) 차이 CI 하한>0 & 보조 같은 부호.
OI: 2024-03-04 부터 스탬프 t 값 = t+5분 스냅샷 → 한 행 늦춰 «행 t = 봉 t 시작 스냅샷»으로 통일(binance_metrics_oi_stamp_switch_20240304).
실행: nice -n 10 env OMP_NUM_THREADS=3 python scripts/research_eth_longh_flow_20261016.py [--selftest]
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(subprocess.run(["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True,
                           cwd=HERE).stdout.strip()).resolve().parent
OUT = HERE / "tmp/longh_flow_20261016"
TAPE = ROOT / "tmp/worktree_salvage_20260930/scalping-entry-analysis-c0c47c/files/tmp/whale/tape/ETHUSDT"
COST = 4.0
HMIN = {"4h": 240, "12h": 720, "1d": 1440, "3d": 4320}
MAIN0 = pd.Timestamp("2025-01-01")
OI_SWITCH = pd.Timestamp("2024-03-04")
B = 1000


# ---------------------------------------------------------------- 데이터
def load_panel() -> pd.DataFrame:
    bv = ROOT / "data/binance_vision"
    cols = ["timestamp", "high", "low", "close", "volume", "taker_buy_base", "sum_open_interest"]
    fs = [bv / "panel/ETHUSDT.parquet"] + sorted(glob.glob(str(bv / "panel_daily/ETHUSDT-5m-*.parquet")))
    p = pd.concat([pd.read_parquet(f, columns=cols) for f in fs]).drop_duplicates("timestamp").sort_values("timestamp")
    p = p.set_index("timestamp").asfreq("5min").reset_index()
    p["close"] = p.close.ffill()
    p["oi"] = fix_oi(p.timestamp, p.sum_open_interest.where(p.sum_open_interest > 0))
    p["tdec"] = p.timestamp + pd.Timedelta("5min")            # 결정 시각 = 봉 종가 시각
    return p


def fix_oi(ts: pd.Series, oi: pd.Series) -> np.ndarray:
    """스위치 이후 행 t 의 값은 t+5분 스냅샷 → 한 행 늦춘다. 결과: 모든 행 t = 봉 t 시작(=직전 봉 끝) 스냅샷."""
    o = oi.to_numpy(float).copy()
    sh = np.r_[np.nan, o[:-1]]
    post = (ts >= OI_SWITCH).to_numpy()
    o[post] = sh[post]
    return o


def load_tape() -> pd.DataFrame:
    cols = ["c", "row_ret_sn", "row_mid_sn", "row_whl_sn"]
    T = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(TAPE / "*.parquet")))]).sort_index()
    T = T[~T.index.duplicated()].asfreq("1min")
    T["c"] = T.c.ffill()
    T.index = T.index.tz_localize(None)
    T = T.reset_index().rename(columns={"b": "timestamp"})
    T["tdec"] = T.timestamp + pd.Timedelta("1min")
    return T


# ---------------------------------------------------------------- 사건 (모두 인과: 결정 인덱스까지의 행만 쓴다)
def dedup(idx: np.ndarray, gap: int) -> np.ndarray:
    keep, last = [], -10 ** 12
    for i in np.sort(idx):
        if i - last >= gap:
            keep.append(i); last = i
    return np.asarray(keep, dtype=int)


def atr12(h, l, c):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    return pd.Series(tr).rolling(12, min_periods=12).mean().to_numpy()


def ev_cvd(p: pd.DataFrame) -> pd.DataFrame:
    """F1. 원 s2 (b) 그대로: 48봉 신고가/신저가 후보를 쪽별 12봉 간격으로 dedup, ev=CVD 미확인."""
    h, l, c = (p[k].to_numpy(float) for k in ("high", "low", "close"))
    cvd = np.cumsum(np.nan_to_num(2 * p.taker_buy_base.to_numpy(float) - p.volume.to_numpy(float)))
    atr = atr12(h, l, c)
    S = pd.Series
    hh, ll = S(h).shift(1).rolling(48).max().to_numpy(), S(l).shift(1).rolling(48).min().to_numpy()
    ch, cl = S(cvd).shift(1).rolling(48).max().to_numpy(), S(cvd).shift(1).rolling(48).min().to_numpy()
    rows = []
    for sgn in (1, -1):
        cand = np.flatnonzero((h > hh) if sgn > 0 else (l < ll))
        for i in dedup(cand[cand > 300], 12):
            if not np.isfinite(atr[i]):
                continue
            ev = int(cvd[i] < ch[i]) if sgn > 0 else int(cvd[i] > cl[i])
            q = ((h[i] - hh[i]) if sgn > 0 else (ll[i] - l[i])) / atr[i]
            rows.append((i, sgn, ev, q, atr[i], h[i] if sgn > 0 else l[i]))
    return pd.DataFrame(rows, columns=["i", "brk", "ev", "q", "atr", "lvl"]).sort_values("i").reset_index(drop=True)


def ev_mismatch(p: pd.DataFrame, W: int) -> pd.DataFrame:
    """F2. W 봉(5분 단위) 비겹침, UTC 정렬. 창 = 행 t−W+1..t, 결정 = 행 t 종가."""
    c = p.close.to_numpy(float)
    d = ((2 * p.taker_buy_base - p.volume) * p.close).fillna(0).to_numpy(float)
    t = np.arange(2 * W - 1, len(c), W)
    ret = np.log(c[t] / c[t - W]) * 1e4
    dl = pd.Series(d).rolling(W).sum().to_numpy()[t]
    bpd = 288 / W
    sd = pd.Series(dl).rolling(int(30 * bpd), min_periods=int(7 * bpd)).std().shift(1).to_numpy()
    E = pd.DataFrame({"i": t, "ret": ret, "dz": dl / sd})
    E = E[np.isfinite(E.dz) & (E.ret != 0)].reset_index(drop=True)
    E["sgn"] = np.sign(E.ret).astype(int)
    E["div"] = np.sign(E.dz) != np.sign(E.ret)
    return E


def ev_whale(T: pd.DataFrame, W: int, th: float = 0.5) -> pd.DataFrame:
    """F3. analyze.py/disagree.py 정의: 격자 t≡0 (mod W) 분, s=(t−W,t] 합, sd=같은 격자 과거 30일(최소 7일) 표준편차(현재 제외)."""
    n = len(T)
    t = np.arange(W + 30 * 1440, n, W)
    Z = {}
    for g in ("ret", "whl"):
        cs = np.r_[0.0, np.cumsum(T[f"row_{g}_sn"].fillna(0).to_numpy(float))]
        grid = np.arange(0, n, W)
        s_grid = cs[grid + 1] - cs[np.maximum(grid + 1 - W, 0)]
        sd = pd.Series(s_grid).rolling(30 * 1440 // W, min_periods=7 * 1440 // W).std().shift(1).to_numpy()
        Z[g] = (s_grid / sd)[t // W]
    ok = np.isfinite(Z["ret"]) & np.isfinite(Z["whl"])
    dis = ok & (np.sign(Z["whl"]) != np.sign(Z["ret"])) & (np.abs(Z["whl"]) >= th) & (np.abs(Z["ret"]) >= th)
    return pd.DataFrame({"i": t[dis], "sgn": np.sign(Z["whl"][dis]).astype(int)}), t[ok]


def ev_liq(p: pd.DataFrame) -> pd.DataFrame:
    """F4. 봉 τ 수익·ΔOI(봉 τ 동안 = oi[τ+1]−oi[τ], τ 끝 스냅샷) → 아는 시점 τ+1 종가에 결정.
    문턱 = 과거 30일(현재 제외) 분위. 종류별 12봉(1h) dedup."""
    c = p.close.to_numpy(float)
    lo = np.log(p.oi.to_numpy(float))
    r = np.r_[np.nan, np.diff(np.log(c))] * 1e4
    doi = np.r_[lo[1:] - lo[:-1], np.nan]                 # 행 τ: 봉 τ 동안 OI 변화(행 τ+1 = τ 끝 스냅샷)
    w = 30 * 288
    q99 = pd.Series(np.abs(r)).rolling(w, min_periods=7 * 288).quantile(0.99).shift(1).to_numpy()
    q05 = pd.Series(doi).rolling(w, min_periods=7 * 288).quantile(0.05).shift(1).to_numpy()
    big = np.abs(r) >= q99
    out = []
    for kind, m in (("liq", big & (doi <= q05)), ("nonliq", big & (doi > 0))):
        tau = dedup(np.flatnonzero(m & np.isfinite(q05) & np.isfinite(q99)), 12)
        tau = tau[tau + 1 < len(c)]
        out.append(pd.DataFrame({"i": tau + 1, "sgn": np.sign(r[tau]).astype(int), "kind": kind}))
    return pd.concat(out, ignore_index=True)


def causal_lo(v: np.ndarray, dayno: np.ndarray, q: float = 0.10, min_days: int = 56, cap: int = 50_000):
    """확장창 분위, 당일 제외, 최소 56일 (research_direction_event_expansion_20260915.causal_thresholds)."""
    D = int(dayno.max()) + 1
    starts = np.searchsorted(dayno, np.arange(D + 1))
    lo = np.full(len(v), np.nan)
    for d in range(min_days, D):
        hst = v[:starts[d]]; hst = hst[np.isfinite(hst)]
        if len(hst) < 1000:
            continue
        if len(hst) > cap:
            hst = hst[:: len(hst) // cap + 1]
        lo[starts[d]:starts[d + 1]] = np.quantile(hst, q)
    return lo


def ev_doi(p: pd.DataFrame) -> pd.DataFrame:
    """참고 셀: doi_12h = log OI[τ] − log OI[τ−144] (행 τ = 봉 τ 시작 스냅샷, 결정은 τ 종가) ≤ 인과 10% → 롱, 4h 순차 비겹침."""
    lo = np.log(p.oi.to_numpy(float))
    doi = np.full(len(lo), np.nan); doi[144:] = lo[144:] - lo[:-144]
    dayno = ((p.timestamp - p.timestamp.iloc[0].normalize()).dt.days).to_numpy()
    th = causal_lo(doi, dayno)
    idx = np.flatnonzero(np.isfinite(th) & (doi <= th))
    keep = dedup(idx, 48)
    return pd.DataFrame({"i": keep, "sgn": 1})


# ---------------------------------------------------------------- 평가
def fwd(c: np.ndarray, idx: np.ndarray, H: int) -> np.ndarray:
    out = np.full(len(idx), np.nan)
    ok = idx + H < len(c)
    out[ok] = np.log(c[idx[ok] + H] / c[idx[ok]]) * 1e4
    return out


def block_of(tdec: pd.Series, week: bool) -> np.ndarray:
    d = (tdec.dt.floor("D") - pd.Timestamp("2020-01-06")).dt.days.to_numpy()
    return d // 7 if week else d


def boot(v, blk, rng, B=B):
    u, inv = np.unique(blk, return_inverse=True)
    s, n = np.bincount(inv, weights=v, minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    w = rng.multinomial(len(u), np.ones(len(u)) / len(u), size=B).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w @ s) / (w @ n), w, u


def boot_diff(v1, b1, v0, b0, rng):
    """같은 블록 가중을 양쪽에 적용한 평균 차(v1−v0) 분포."""
    u = np.unique(np.r_[b1, b0])
    w = rng.multinomial(len(u), np.ones(len(u)) / len(u), size=B).astype(float)
    i1, i0 = np.searchsorted(u, b1), np.searchsorted(u, b0)
    m = []
    for v, ii in ((v1, i1), (v0, i0)):
        s, n = np.bincount(ii, weights=v, minlength=len(u)), np.bincount(ii, minlength=len(u)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            m.append((w @ s) / (w @ n))
    return m[0] - m[1]


def ci(x):
    return [round(float(np.nanpercentile(x, 2.5)), 2), round(float(np.nanpercentile(x, 97.5)), 2)]


def trend_daily(p: pd.DataFrame) -> pd.Series:
    """기준선: 직전 28일 수익 부호를 다음 1일 보유 (bp). 인덱스 = 보유한 날."""
    dc = p.set_index("tdec").close.resample("1D", label="left", closed="right").last()   # 그 날 끝 종가
    dc.index = dc.index + pd.Timedelta("1D")          # 이 날짜 0시에 아는 값
    s = np.sign(np.log(dc / dc.shift(28)))
    r = np.log(dc.shift(-1) / dc) * 1e4
    return (s * r).dropna()


def evaluate(name, c, tdec, idx, sgn, H, pool, trend, rng, extra=None):
    """idx: 결정 인덱스(c 기준) · sgn: 방향 · H: 보유 행 수 · pool: 위약 후보 결정 인덱스 전체."""
    f = fwd(c, idx, H) * sgn
    ok = np.isfinite(f)
    idx, sgn, f = idx[ok], sgn[ok], f[ok]
    net = f - COST
    td = tdec.iloc[idx].reset_index(drop=True)
    main = (td >= MAIN0).to_numpy()
    week = H * (tdec.iloc[1] - tdec.iloc[0]) > pd.Timedelta("1D")
    blk = block_of(td, week)
    r = {"cell": name, "n_all": int(len(idx)), "n_main": int(main.sum()), "n_aux": int((~main).sum())}
    if main.sum() < 20:
        return r | {"PASS": False, "note": "n<20"}
    bm, _, _ = boot(net[main], blk[main], rng)
    r |= {"main_gross": round(float(f[main].mean()), 2), "main_net": round(float(net[main].mean()), 2), "main_ci": ci(bm),
          "aux_net": round(float(net[~main].mean()), 2) if (~main).sum() else None}
    # 위약: 같은 주 기간 · 같은 수 · 같은 방향 구성(순열), 무작위 결정 시각
    pl = pool[(tdec.iloc[pool] >= MAIN0).to_numpy() & (pool + H < len(c))]
    fp = np.log(c[pl + H] / c[pl]) * 1e4
    k = int(main.sum())
    draw = rng.integers(0, len(pl), size=(B, k))
    dirs = rng.permuted(np.tile(sgn[main], (B, 1)), axis=1)
    pmean = (dirs * fp[draw]).mean(1) - COST
    dd = bm - pmean
    r |= {"placebo_net": round(float(pmean.mean()), 2), "plac_diff": round(float(net[main].mean() - pmean.mean()), 2),
          "plac_diff_ci": ci(dd)}
    yr = td.dt.year.to_numpy()
    r["years"] = {int(y): round(float(net[yr == y].mean()), 1) for y in np.unique(yr)}
    # 순차 비겹침
    keep, last = [], -10 ** 12
    for pos in np.argsort(idx, kind="stable"):
        if idx[pos] - last >= H:
            keep.append(pos); last = idx[pos]
    keep = np.asarray(keep, dtype=int)
    km = keep[main[keep]]
    if len(km) >= 20:
        bk, _, _ = boot(net[km], blk[km], rng)
        r |= {"n_nonov_main": int(len(km)), "nonov_main_net": round(float(net[km].mean()), 2), "nonov_ci": ci(bk),
              "nonov_aux_net": round(float(net[keep[~main[keep]]].mean()), 2) if (~main[keep]).sum() else None}
    else:
        r |= {"n_nonov_main": int(len(km))}
    # 일 손익 vs 추세 기준선 상관 (주 기간, 사건 없는 날 0)
    dser = pd.Series(net[main], index=td[main].dt.floor("D").to_numpy()).groupby(level=0).sum()
    tr = trend[trend.index >= MAIN0]
    tr = tr[tr.index <= td[main].max()]
    r["trend_corr"] = round(float(np.corrcoef(dser.reindex(tr.index, fill_value=0).to_numpy(), tr.to_numpy())[0, 1]), 3)
    r["PASS"] = bool(r["main_net"] > 0 and r["main_ci"][0] > 0 and r["plac_diff_ci"][0] > 0
                     and (r["aux_net"] or 0) > 0)
    if extra:
        r |= extra
    return r


def strat_diff(val, ev, q, blk, rng):
    """층화 차(사건−대조, 대조 비중 가중), 블록 부트스트랩. 원 strat_ci 와 같은 추정량."""
    u, di = np.unique(blk, return_inverse=True)
    Q, D = int(q.max()) + 1, len(u)
    ix = di * Q + q
    S = [np.bincount(ix[ev == g], weights=val[ev == g], minlength=D * Q).reshape(D, Q) for g in (1, 0)]
    N = [np.bincount(ix[ev == g], minlength=D * Q).reshape(D, Q).astype(float) for g in (1, 0)]
    pi = np.where((N[0].sum(0) > 0) & (N[1].sum(0) > 0), N[0].sum(0), 0.0); pi = pi / pi.sum()

    def est(w):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nansum(((w @ S[0]) / (w @ N[0]) - (w @ S[1]) / (w @ N[1])) * pi, -1)
    w = rng.multinomial(D, np.ones(D) / D, size=B).astype(float)
    return float(est(np.ones(D))), np.nanpercentile(est(w), [2.5, 97.5])


# ---------------------------------------------------------------- 실행
def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261016)
    p = load_panel()
    c, tdec = p.close.to_numpy(float), p.tdec
    trend = trend_daily(p)
    pool5 = np.arange(300, len(c))
    res, extn = [], []
    print("panel", p.shape, p.timestamp.iloc[0], p.timestamp.iloc[-1], flush=True)

    # F1 CVD 다이버전스
    E = ev_cvd(p)
    E["main"] = (tdec.iloc[E.i].to_numpy() >= MAIN0)
    qe = np.searchsorted(np.nanpercentile(E.q, np.arange(10, 100, 10)), E.q.to_numpy())
    h, l = p.high.to_numpy(float), p.low.to_numpy(float)
    print("F1 events", len(E), "div share", round(E.ev.mean(), 3), flush=True)
    for hn, Hm in HMIN.items():
        H = Hm // 5
        fmax = pd.Series(h).rolling(H, min_periods=H).max().shift(-H).to_numpy()
        fmin = pd.Series(l).rolling(H, min_periods=H).min().shift(-H).to_numpy()
        ii = E.i.to_numpy()
        blk = block_of(tdec.iloc[ii].reset_index(drop=True), hn == "3d")
        okh = np.isfinite(fmax[ii])
        for th in (0.25, 0.5, 1.0):
            ext = np.where(E.brk > 0, fmax[ii] >= E.lvl + th * E.atr, fmin[ii] <= E.lvl - th * E.atr)
            noext = (~ext).astype(float)
            row = {"cell": f"F1 연장실패 {th}ATR H={hn}"}
            for per, m in (("main", E.main.to_numpy() & okh), ("aux", ~E.main.to_numpy() & okh)):
                pt, cc = strat_diff(noext[m], E.ev.to_numpy()[m], qe[m], blk[m], rng)
                row |= {f"{per}_div_rate": round(float(noext[m & (E.ev == 1).to_numpy()].mean()), 3),
                        f"{per}_conf_rate": round(float(noext[m & (E.ev == 0).to_numpy()].mean()), 3),
                        f"{per}_diff_pp": round(100 * pt, 2), f"{per}_ci_pp": [round(100 * x, 2) for x in cc]}
            row["PASS"] = bool(row["main_ci_pp"][0] > 0 and row["aux_diff_pp"] > 0)
            extn.append(row); print(row, flush=True)
        # 방향: 다이버 사건 페이드 · 가격만 대조(같은 신고가 사건 전체) 차이
        d1 = E[E.ev == 1]
        f_all = fwd(c, ii, H) * -E.brk.to_numpy() - COST
        m_all = E.main.to_numpy() & np.isfinite(f_all)
        m_div = m_all & (E.ev == 1).to_numpy()
        m_cf = m_all & (E.ev == 0).to_numpy()
        dpo = boot_diff(f_all[m_div], blk[m_div], f_all[m_all], blk[m_all], rng)
        dcf = boot_diff(f_all[m_div], blk[m_div], f_all[m_cf], blk[m_cf], rng)
        extra = {"price_only_net": round(float(f_all[m_all].mean()), 2),
                 "div_minus_priceonly": round(float(f_all[m_div].mean() - f_all[m_all].mean()), 2), "dpo_ci": ci(dpo),
                 "div_minus_confirm": round(float(f_all[m_div].mean() - f_all[m_cf].mean()), 2), "dcf_ci": ci(dcf)}
        r = evaluate(f"F1 CVD다이버 페이드 H={hn}", c, tdec, d1.i.to_numpy(), -d1.brk.to_numpy(), H, pool5, trend, rng, extra)
        res.append(r); print(r, flush=True)
        r = evaluate(f"F1 가격만(신고가 전체) 페이드 H={hn}", c, tdec, E.i.to_numpy(), -E.brk.to_numpy(), H, pool5, trend, rng)
        r["control_only"] = True; res.append(r); print(r, flush=True)

    # F2 체결↔가격 불일치
    for Wn, W in (("4h", 48), ("1d", 288)):
        M = ev_mismatch(p, W)
        print(f"F2 W={Wn} bars", len(M), "div share", round(M["div"].mean(), 3), flush=True)
        M["main"] = tdec.iloc[M.i].to_numpy() >= MAIN0
        for hn, Hm in HMIN.items():
            H = Hm // 5
            M["f"] = fwd(c, M.i.to_numpy(), H) * M.sgn
            # 매칭 대조(주 기간 안에서 분위)
            G = M[M.main & np.isfinite(M.f)].copy()
            G["aq"] = pd.qcut(G.ret.abs().rank(method="first"), 10, labels=False)
            G["zq"] = pd.qcut(G.dz.abs().rank(method="first"), 3, labels=False)
            cm = G[~G["div"]].groupby(["aq", "zq"]).f.mean()
            ev = G[G["div"]]
            v = ev.f.to_numpy() - cm.reindex(pd.MultiIndex.from_frame(ev[["aq", "zq"]])).to_numpy()
            okv = np.isfinite(v)
            bv, _, _ = boot(v[okv], block_of(tdec.iloc[ev.i.to_numpy()[okv]].reset_index(drop=True), hn == "3d"), rng)
            extra = {"matched_diff": round(float(v[okv].mean()), 2), "matched_ci": ci(bv)}
            D = M[M["div"]]
            r = evaluate(f"F2 불일치 W={Wn} 가격추종 H={hn}", c, tdec, D.i.to_numpy(), D.sgn.to_numpy(), H,
                         M.i.to_numpy(), trend, rng, extra)
            res.append(r); print(r, flush=True)

    # F4 청산 프록시 + doi 참고
    L = ev_liq(p)
    print("F4 events", L.kind.value_counts().to_dict(), flush=True)
    for hn, Hm in HMIN.items():
        H = Hm // 5
        for kind in ("liq", "nonliq"):
            g = L[L.kind == kind]
            for side, nm in ((1, "추종"), (-1, "페이드")):
                r = evaluate(f"F4 {kind} {nm} H={hn}", c, tdec, g.i.to_numpy(), side * g.sgn.to_numpy(), H, pool5, trend, rng)
                if kind == "nonliq":
                    r["control_only"] = True
                res.append(r); print(r, flush=True)
        g1, g0 = L[L.kind == "liq"], L[L.kind == "nonliq"]
        f1 = fwd(c, g1.i.to_numpy(), H) * g1.sgn.to_numpy(); f0 = fwd(c, g0.i.to_numpy(), H) * g0.sgn.to_numpy()
        m1 = np.isfinite(f1) & (tdec.iloc[g1.i].to_numpy() >= MAIN0); m0 = np.isfinite(f0) & (tdec.iloc[g0.i].to_numpy() >= MAIN0)
        b1 = block_of(tdec.iloc[g1.i.to_numpy()[m1]].reset_index(drop=True), hn == "3d")
        b0 = block_of(tdec.iloc[g0.i.to_numpy()[m0]].reset_index(drop=True), hn == "3d")
        dd = boot_diff(f1[m1], b1, f0[m0], b0, rng)
        res.append({"cell": f"F4 청산형−비청산형(추종) H={hn}", "diff": round(float(f1[m1].mean() - f0[m0].mean()), 2),
                    "ci": ci(dd), "control_only": True})
        print(res[-1], flush=True)
    Dd = ev_doi(p)
    print("doi events", len(Dd), flush=True)
    for hn, Hm in HMIN.items():
        r = evaluate(f"REF doi_12h 하위10% 롱 H={hn}", c, tdec, Dd.i.to_numpy(), Dd.sgn.to_numpy(), Hm // 5, pool5, trend, rng)
        res.append(r); print(r, flush=True)
    del p

    # F3 고래 맞대결 (1분 테이프)
    T = load_tape()
    ct, tt = T.c.to_numpy(float), T.tdec
    print("tape", T.shape, T.timestamp.iloc[0], T.timestamp.iloc[-1], flush=True)
    cache = {}
    for W in (60, 240, 720, 1440):
        cache[W] = ev_whale(T, W)
    ev60, pool60 = cache[60]
    r = evaluate("F3 재현 W=60 H=60m", ct, tt, ev60.i.to_numpy(), ev60.sgn.to_numpy(), 60, pool60, trend, rng)
    r["control_only"] = True; res.append(r); print(r, flush=True)
    for hn, Hm in HMIN.items():
        r = evaluate(f"F3a 맞대결 W=60 고래 H={hn}", ct, tt, ev60.i.to_numpy(), ev60.sgn.to_numpy(), Hm, pool60, trend, rng)
        res.append(r); print(r, flush=True)
    for hn, Hm, W in (("4h", 240, 240), ("12h", 720, 720), ("1d", 1440, 1440), ("3d", 4320, 1440)):
        e, pl = cache[W]
        r = evaluate(f"F3b 맞대결 W={W}m 고래 H={hn}", ct, tt, e.i.to_numpy(), e.sgn.to_numpy(), Hm, pl, trend, rng)
        res.append(r); print(r, flush=True)

    tested = [r for r in res if not r.get("control_only")]
    summ = {"direction_cells": len(tested), "direction_pass": [r["cell"] for r in tested if r.get("PASS")],
            "ext_cells": len(extn), "ext_pass": [r["cell"] for r in extn if r["PASS"]]}
    (OUT / "results.json").write_text(json.dumps({"direction": res, "extension": extn, "summary": summ},
                                                 ensure_ascii=False, indent=1, default=str))
    pd.DataFrame(res).to_csv(OUT / "direction_cells.csv", index=False)
    pd.DataFrame(extn).to_csv(OUT / "extension_cells.csv", index=False)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- 자체 점검
def selftest():
    rng = np.random.default_rng(1)
    n = 70 * 288
    ts = pd.date_range("2024-02-01", periods=n, freq="5min")
    c = 2000 * np.exp(np.cumsum(rng.normal(0, 8e-4, n)))
    v = rng.uniform(100, 1000, n)
    p = pd.DataFrame({"timestamp": ts, "close": c, "high": c * (1 + rng.uniform(0, 1e-3, n)),
                      "low": c * (1 - rng.uniform(0, 1e-3, n)), "volume": v, "taker_buy_base": v * rng.uniform(0.2, 0.8, n),
                      "sum_open_interest": 1e5 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))})
    p["oi"] = fix_oi(p.timestamp, p.sum_open_interest)
    # OI 보정: 스위치 이전 그대로, 이후 한 행 늦춤
    k = int(np.searchsorted(ts, OI_SWITCH))
    assert p.oi.iloc[k - 1] == p.sum_open_interest.iloc[k - 1] and p.oi.iloc[k + 5] == p.sum_open_interest.iloc[k + 4]
    # 라벨: τ 종가 → τ+H 종가
    assert np.isclose(fwd(c, np.array([10]), 3)[0], np.log(c[13] / c[10]) * 1e4) and np.isnan(fwd(c, np.array([n - 2]), 3)[0])
    assert dedup(np.array([0, 5, 12, 13, 30]), 12).tolist() == [0, 12, 30]

    # 미래참조 없음: cut 이후 행을 흔들어도 cut 이하 사건은 그대로
    cut = 62 * 288 + 77
    q = p.copy()
    j = np.arange(cut + 1, n)
    for col in ("close", "high", "low", "volume", "taker_buy_base", "sum_open_interest", "oi"):
        q.loc[j, col] = q.loc[j, col].to_numpy() * rng.uniform(0.9, 1.1, len(j))

    def same(a, b, cols):
        a, b = a[a.i <= cut][cols].reset_index(drop=True), b[b.i <= cut][cols].reset_index(drop=True)
        assert len(a) > 0, cols
        pd.testing.assert_frame_equal(a, b)
    same(ev_cvd(p), ev_cvd(q), ["i", "brk", "ev", "q"])
    same(ev_mismatch(p, 48), ev_mismatch(q, 48), ["i", "sgn", "div", "dz"])
    # ev_liq 는 행 τ 의 ΔOI 에 행 τ+1 OI 를 쓰고 결정은 τ+1 — 결정 i ≤ cut 이면 i 행까지만 본다
    same(ev_liq(p), ev_liq(q), ["i", "sgn", "kind"])
    same(ev_doi(p), ev_doi(q), ["i", "sgn"])

    # 고래 z: 합성 테이프
    m = 45 * 1440
    T = pd.DataFrame({"c": 2000 + np.cumsum(rng.normal(0, 1, m)), "row_ret_sn": rng.normal(0, 1, m),
                      "row_mid_sn": 0.0, "row_whl_sn": rng.normal(0, 1, m)})
    T["row_whl_sn"] -= 0.7 * T.row_ret_sn            # 반대로 서는 일이 잦게
    e1, _ = ev_whale(T, 60)
    cutm = 40 * 1440 + 7
    T2 = T.copy(); jj = np.arange(cutm + 1, m)
    T2.loc[jj, "row_whl_sn"] = rng.normal(0, 5, len(jj)); T2.loc[jj, "row_ret_sn"] = rng.normal(0, 5, len(jj))
    e2, _ = ev_whale(T2, 60)
    a, b = e1[e1.i <= cutm].reset_index(drop=True), e2[e2.i <= cutm].reset_index(drop=True)
    assert len(a) > 50; pd.testing.assert_frame_equal(a, b)
    # 고래 z 직접 계산 대조 (원 pandas 식)
    W = 60; t0 = int(e1.i.iloc[3])
    s = T.row_whl_sn.rolling(W).sum()
    sd = s.iloc[::W].rolling(30 * 1440 // W, min_periods=7 * 1440 // W).std().reindex(s.index).ffill().shift(W)
    z_orig = s.iloc[t0] / sd.iloc[t0]
    assert np.sign(z_orig) == e1.sgn.iloc[3] and abs(z_orig) >= 0.5

    # 평가기: 심은 신호는 통과, 무작위 사건은 통과 못 함(대개)
    N = 600 * 288
    tsd = pd.Series(pd.date_range("2024-06-01", periods=N, freq="5min") + pd.Timedelta("5min"))
    cc = 2000 * np.exp(np.cumsum(rng.normal(0, 1e-3, N)))
    ev = np.sort(rng.choice(np.arange(300, N - 100), 1500, replace=False))
    for i in ev:                                       # 사건 뒤 48봉에 +60bp 심기
        cc[i + 1:] *= np.exp(0.006 / 48 * np.minimum(np.arange(1, N - i), 48))
    tr = pd.Series(0.0, index=pd.date_range("2024-06-01", periods=600, freq="D"))
    r = evaluate("plant", cc, tsd, ev, np.ones(len(ev), int), 48, np.arange(300, N), tr, rng)
    assert r["PASS"], r
    r0 = evaluate("null", cc, tsd, np.sort(rng.choice(np.arange(300, N - 300), 1500, replace=False)),
                  rng.choice([-1, 1], 1500), 48, np.arange(300, N), tr, rng)
    assert abs(r0["plac_diff"]) < 15, r0
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else run()
