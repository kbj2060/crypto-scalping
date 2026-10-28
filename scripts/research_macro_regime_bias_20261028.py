#!/usr/bin/env python3
"""매크로 «현재 국면»(A)·«4주 편향»(B) 모델 -- 사전등록 docs/prereg/macro_regime_bias_4w_20261028.md 그대로(2026-10-28).

단계(순서 고정):
  --stage fit    적합 구간(2020–23)에서만 지표 선정·부호·계수 → docs/prereg/macro_regime_bias_4w_frozen_20261028.json (채점 전 커밋)
  --stage score  동결 사양으로 2024-01~2026-05 고정 채점(주 판정)
  --stage test   2026-06~ 시험 1회(표식 파일이 있으면 거부, --force 로만)
  --stage r2test 2라운드(docs/prereg/macro_regime_bias_r2_20261028.md): 1라운드 선정·부호로 절편 없는 «부호 × 13주 변화 ÷ 104주 sd» 평균 Z,
                 2026-06~ 시험 1회(현재 빈티지 + 달러지수·STLFSI4 최초 공표값) · 2024-01~26-05 는 «사후 참고»로만. FRED_API_KEY 필요
  --selftest     네트워크·패널 없이: CW 식 · 심은 신호 선정 · 인과성(τ 뒤 패널을 바꿔도 τ 예보 불변)
입력 패널: research_macro_crypto_panel_20261027.py 산출 panel_daily.parquet(CC BY-NC, git 밖) -- --panel 로 준다.
  python scripts/research_macro_regime_bias_20261028.py --panel <panel_daily.parquet> --stage fit
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FROZEN = ROOT / "docs/prereg/macro_regime_bias_4w_frozen_20261028.json"
OUT = ROOT / "docs/experiments/macro_regime_bias_20261028"
FIT = ("2020-01-03", "2023-12-29")
HALF = "2021-12-31"
A_FIT_START = "2020-04-03"
SCORE = ("2024-01-05", "2026-05-29")
TEST_START = "2026-06-05"
HORIZONS = (4, 8, 13)
R_MIN = {"B": 0.10, "A": 0.15}
COVER = 0.80
SEED = 20261028
MIN_N = 30   # 이보다 적으면 위약·블록 부트스트랩을 안 한다(시험 구간 n≈17, 10-28 리뷰)


@functools.lru_cache(maxsize=None)
def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verdict():
    return _load("verdict", "scripts/research_macro_trend_p3_verdict_20261028.py")


def sample_week(panel: pd.DataFrame, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """판정표와 같은 주간 표본(ffill 10일 → W-FRI last, DGS10 끝 뒤 주 제거). 전진 기록기도 이 함수를 부른다(패리티).
    end = 전진 기록기의 결정 금요일 -- 🔴토 01:30 UTC 엔 FRED DGS10 의 금요일 값이 아직 없어 표본기가 그 주를 잘라낸다(10-28 리뷰).
    그 날 DGS10 에 직전 값을 넣어 절단만 우회한다(τ 피쳐는 1주 미룬 값이라 τ 의 DGS10 을 안 쓴다 -- 누수 없음)."""
    if end is not None:
        end = pd.Timestamp(end)
        panel = panel.copy()
        if end not in panel.index:
            panel.loc[end] = np.nan
            panel = panel.sort_index()
        panel.loc[end, "DGS10"] = panel["DGS10"].loc[:end].ffill().iloc[-1]
    w = _verdict().load_sampler()(panel, "W")
    for a in ("BTC", "ETH"):   # 🔴표본기의 ffill(10일)이 코인 마지막 날 뒤 주에 묵은 가격을 채운다(수익 0) -- 그 주는 버린다(10-28 리뷰)
        if a in panel:
            w.loc[w.index > panel[a].last_valid_index(), a] = np.nan
    return w


def feats_for(w: pd.DataFrame, names) -> dict:
    """지표 이름 → «판정표 변환의 13주 변화, 1주 미룸»."""
    v = _verdict()
    t = {**v.SERIES, **v.EXTRA}
    return {n: v.level(w, t[n][0], t[n][1]).diff(13).shift(1) for n in names}


def weekly(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """판정표와 같은 주간 표본 + 지표 27개(SERIES + EXTRA)의 «13주 변화, 1주 미룸»."""
    v = _verdict()
    w = sample_week(panel)
    table = {**v.SERIES, **v.EXTRA}
    missing = [c for c, _ in table.values() if c not in w]
    assert len(table) == 27 and not missing, (len(table), missing)   # 지표가 조용히 빠지지 않게
    return w, feats_for(w, table)


SD_WEEKS, SD_MIN = 104, 52


def selected(fz: dict, asset: str, model: str) -> dict:
    return {k: s for k, s in fz["assets"][asset][model]["indicators"].items() if s["selected"]}


def z_combo(feats: dict, spec: dict) -> tuple[pd.Series, pd.Series]:
    """2라운드: 절편 없는 Z = 평균_i sign(r_i)·x_i / sd_i(τ 까지 104주, 최소 52주). 반환 (Z, 시점별 지표 수)."""
    zs = [np.sign(s["r"]) * feats[k] / feats[k].rolling(SD_WEEKS, min_periods=SD_MIN).std().where(lambda x: x > 0) for k, s in spec.items()]   # sd 0 = 그 지표 빠짐
    m = pd.concat(zs, axis=1)
    return m.mean(axis=1), m.notna().sum(axis=1)


def labels(w: pd.DataFrame, asset: str) -> dict[str, pd.Series]:
    lp = 100 * np.log(w[asset])
    out = {f"y{h}": lp.shift(-h) - lp for h in HORIZONS}   # 결정 τ 다음부터 h 주
    out["c13"] = lp - lp.shift(13)                          # 같은 13주(국면 A)
    return out


def fit_one(x: pd.Series, y: pd.Series, lo: str, hi: str, rmin: float) -> dict | None:
    d = pd.concat([x, y], axis=1).loc[lo:hi]
    n_all = len(d)
    d = d.dropna()
    if n_all == 0 or len(d) < COVER * n_all:
        return None
    r = float(d.corr().iloc[0, 1])
    ra = float(d.loc[:HALF].corr().iloc[0, 1])
    rb = float(d.loc[pd.Timestamp(HALF) + pd.Timedelta(days=1):].corr().iloc[0, 1])
    b, a = np.polyfit(d.iloc[:, 0], d.iloc[:, 1], 1)
    keep = abs(r) >= rmin and np.sign(ra) == np.sign(r) and np.sign(rb) == np.sign(r)
    return {"r": r, "r_2020_21": ra, "r_2022_23": rb, "a": float(a), "b": float(b), "n": int(len(d)), "selected": bool(keep)}


def combine(feats: dict, spec: dict, count: bool = False):
    """선정 지표 1변수 예보의 평균(그 시각 값이 있는 것만). count=True 면 시점별 사용 지표 수도."""
    preds = [spec[k]["a"] + spec[k]["b"] * feats[k] for k, s in spec.items() if s["selected"]]
    m = pd.concat(preds, axis=1) if preds else pd.DataFrame()
    out = m.mean(axis=1) if preds else pd.Series(dtype=float)
    return (out, m.notna().sum(axis=1) if preds else pd.Series(dtype=int)) if count else out


def logit(x: pd.Series, up: pd.Series) -> dict:
    from sklearn.linear_model import LogisticRegression
    d = pd.concat([x, up], axis=1).dropna()
    m = LogisticRegression(C=1e6).fit(d.iloc[:, [0]].values, d.iloc[:, 1].astype(int).values)
    return {"coef": float(m.coef_[0, 0]), "intercept": float(m.intercept_[0])}


def stage_fit(panel: pd.DataFrame, force: bool = False) -> dict:
    if FROZEN.exists() and not force:   # 🔴채점을 본 뒤 다시 적합해 선정을 바꾸지 못하게(10-28 리뷰)
        raise SystemExit(f"동결 사양이 이미 있다: {FROZEN} -- 다시 적합하려면 새 사전등록")
    w, feats = weekly(panel)
    frozen = {"prereg": "docs/prereg/macro_regime_bias_4w_20261028.md", "fit": FIT, "assets": {}}
    for asset in ("ETH", "BTC"):
        L = labels(w, asset)
        fa = {}
        for h in HORIZONS:
            hi = str((pd.Timestamp(FIT[1]) - pd.Timedelta(weeks=h)).date())   # 라벨 끝 ≤ 2023-12-29
            spec = {k: s for k, x in feats.items() if (s := fit_one(x, L[f"y{h}"], FIT[0], hi, R_MIN["B"]))}
            yh = combine(feats, spec)
            fa[f"B{h}"] = {"fit_hi": hi, "indicators": spec,
                           "logit": logit(yh.loc[FIT[0]:hi], (L[f"y{h}"] > 0).loc[FIT[0]:hi]) if len(yh) else None}
            fa[f"B{h}"]["all"] = {k: {**s, "selected": True} for k, s in spec.items()}   # 보조: 자격(관측 80%) 있는 지표 전부 평균
            fa[f"B{h}"]["excluded_coverage"] = sorted(set(feats) - set(spec))
        spec = {k: s for k, x in feats.items() if (s := fit_one(x, L["c13"], A_FIT_START, FIT[1], R_MIN["A"]))}
        ch = combine(feats, spec)
        fa["A"] = {"indicators": spec, "logit": logit(ch.loc[A_FIT_START:FIT[1]], (L["c13"] > 0).loc[A_FIT_START:FIT[1]]) if len(ch) else None,
                   "excluded_coverage": sorted(set(feats) - set(spec))}
        frozen["assets"][asset] = fa
    FROZEN.write_text(json.dumps(frozen, ensure_ascii=False, indent=1))
    return frozen


# ── 채점 ────────────────────────────────────────────────────────────────────────────────────────────
def expanding_mean(y: pd.Series, h: int, start: str) -> pd.Series:
    """τ 시각에 끝난 라벨(τ−h 주 이전 결정)만의 평균 -- 적합 시작부터."""
    ys = y.loc[start:]
    return ys.shift(h).expanding().mean().reindex(y.index)


def clark_west(y, f_null, f_alt, lag: int) -> float:
    import statsmodels.api as sm
    f = (y - f_null) ** 2 - ((y - f_alt) ** 2 - (f_null - f_alt) ** 2)
    r = sm.OLS(np.asarray(f, dtype=float), np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(r.tvalues[0])


def r2_os(y, f_null, f_alt) -> float:
    return float(1 - ((y - f_alt) ** 2).sum() / ((y - f_null) ** 2).sum())


def block_boot(stat, n: int, block: int, rng, reps: int = 2000) -> np.ndarray:
    k = int(np.ceil(n / block))
    out = []
    for _ in range(reps):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block + 1, k)])[:n]
        out.append(stat(idx))
    return np.array(out)


def score_B(y, yhat, bench, h, rng) -> dict:
    d = pd.concat({"y": y, "f": yhat, "b": bench}, axis=1).dropna()
    yv, fv, bv = d.y.values, d.f.values, d.b.values
    up, pr = yv > 0, fv > 0
    excess = lambda i: float((pr[i] == up[i]).mean() - up[i].mean())   # noqa: E731
    r2 = r2_os(yv, bv, fv)
    out = {"n": int(len(d)), "n_indep": int(len(d) // h), "cw_t": clark_west(yv, bv, fv, h - 1), "r2_os": r2,
           "hit": float((pr == up).mean()), "always_up": float(up.mean()), "excess": excess(np.arange(len(d))), "pred_up_share": float(pr.mean()),
           "first": str(d.index[0].date()), "last": str(d.index[-1].date())}
    if len(d) < MIN_N:   # 시험 구간(n≈17): 위약 이동 후보·독립 블록이 거의 없다 -- 사전등록대로 부호·적중만(판정 칸은 닫힌 쪽)
        return {**out, "note": f"n<{MIN_N}: 위약·부트스트랩 생략(판정력 없음)", "excess_lo90": None, "placebo_p90": None, "placebo_pct": None}
    bs = block_boot(excess, len(d), 8, rng)
    plc = np.array([r2_os(yv, bv, np.roll(fv, k)) for k in rng.integers(9, len(d) - 8, 1000)])
    return {**out, "excess_lo90": float(np.quantile(bs, 0.10)), "placebo_p90": float(np.quantile(plc, 0.90)), "placebo_pct": float((plc < r2).mean())}


def score_A(c, chat, rng) -> dict:
    d = pd.concat({"c": c, "f": chat}, axis=1).dropna()
    cv, fv = d.c.values, d.f.values
    corr = lambda i: float(np.corrcoef(cv[i], fv[i])[0, 1])   # noqa: E731
    exc = lambda i: float(((fv[i] > 0) == (cv[i] > 0)).mean() - (cv[i] > 0).mean())   # noqa: E731
    all_i = np.arange(len(d))
    out = {"n": int(len(d)), "n_indep": int(len(d) // 13), "corr": corr(all_i), "hit": float(((fv > 0) == (cv > 0)).mean()),
           "always_up": float((cv > 0).mean()), "excess": exc(all_i), "regime_up_share": float((fv > 0).mean()),
           "first": str(d.index[0].date()), "last": str(d.index[-1].date())}
    if len(d) < MIN_N:
        return {**out, "note": f"n<{MIN_N}: 부트스트랩 생략(판정력 없음)", "corr_lo90": None, "excess_lo90": None}
    return {**out, "corr_lo90": float(np.quantile(block_boot(corr, len(d), 13, rng), .10)), "excess_lo90": float(np.quantile(block_boot(exc, len(d), 13, rng), .10))}


def z91(panel: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    """10-15 동결 규칙 Z(달러·순유동성) -- 원 스크립트의 liq_score·발표 지연 그대로, 결정 금요일 값."""
    t = _load("trend", "scripts/research_trend_mid_long_20261015.py")
    day = pd.date_range(panel.index.min(), panel.index.max(), freq="D")
    ins = [t._asof(panel[k], day, t.FRED_LAG[k]) for k in ("DTWEXBGS", "WALCL", "RRPONTSYD", "WTREGEN")]
    return t.liq_score(*ins).reindex(idx)


def stage_score(panel: pd.DataFrame, lo: str, hi: str | None, tag: str) -> dict:
    import hashlib
    import subprocess
    fz = json.loads(FROZEN.read_text())
    w, feats = weekly(panel)
    rng = np.random.default_rng(SEED)
    res, rows = {"window": [lo, hi], "frozen_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
                 "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                 "frozen_dirty": bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", str(FROZEN)], capture_output=True, text=True).stdout.strip()),
                 "assets": {}}, []
    zz = z91(panel, w.index)
    for asset in ("ETH", "BTC"):
        L, fa, ra = labels(w, asset), fz["assets"][asset], {}
        for h in HORIZONS:
            sp = fa[f"B{h}"]
            end = hi and str((pd.Timestamp(hi) - pd.Timedelta(weeks=h)).date())
            y = L[f"y{h}"].loc[lo:end].dropna()
            bench = expanding_mean(L[f"y{h}"], h, FIT[0]).reindex(y.index)
            yh, cnt = combine(feats, sp["indicators"], count=True)
            yh, cnt = yh.reindex(y.index), cnt.reindex(y.index)
            ra[f"B{h}"] = score_B(y, yh, bench, h, rng) if yh.notna().any() else {"note": "선정 지표 0개"}
            if yh.notna().any():
                ra[f"B{h}"]["n_indicators_min_mean"] = [int(cnt.min()), float(cnt.mean())]
            ra[f"B{h}_all"] = score_B(y, combine(feats, sp["all"]).reindex(y.index), bench, h, rng)
            for nm, sig in (("Z91_k0", zz), ("TSMOM13", L["c13"])):   # 방향만 맞대기(크기 없음 → 적중만)
                d = pd.concat([y, sig.reindex(y.index)], axis=1).dropna()
                ra[f"B{h}_{nm}"] = {"n": int(len(d)), "hit": float(((d.iloc[:, 1] > 0) == (d.iloc[:, 0] > 0)).mean()), "always_up": float((d.iloc[:, 0] > 0).mean())}
            d = pd.concat([y, zz.reindex(y.index)], axis=1).dropna()
            m = d.iloc[:, 1].abs() >= 1.0
            ra[f"B{h}_Z91_k1"] = {"n": int(m.sum()), "cov": float(m.mean()), "hit": float(((d.iloc[:, 1][m] > 0) == (d.iloc[:, 0][m] > 0)).mean()) if m.any() else None}
            if sp["logit"] and yh.notna().any():
                p = 1 / (1 + np.exp(-(sp["logit"]["intercept"] + sp["logit"]["coef"] * yh)))
                p0 = (L[f"y{h}"].loc[FIT[0]:] > 0).astype(float).shift(h).expanding().mean().reindex(y.index)
                d = pd.concat({"u": (y > 0).astype(float), "p": p, "p0": p0}, axis=1).dropna()
                ra[f"B{h}"]["brier_skill"] = float(1 - ((d.u - d.p) ** 2).mean() / ((d.u - d.p0) ** 2).mean())
            if h == 4:
                rows.append(pd.DataFrame({"asset": asset, "y4": y, "yhat4": yh, "bench4": bench, "z91": zz.reindex(y.index)}))
            if yh.notna().any():
                yy = pd.DataFrame({"y": y, "f": yh}).dropna()
                ra[f"B{h}_by_year"] = {str(g): {"n": int(len(x)), "hit": float(((x.f > 0) == (x.y > 0)).mean()), "always_up": float((x.y > 0).mean())}
                                       for g, x in yy.groupby(yy.index.year)}
        c = L["c13"].loc[lo:hi].dropna()
        ch, cnt = combine(feats, fa["A"]["indicators"], count=True)
        ch, cnt = ch.reindex(c.index), cnt.reindex(c.index)
        ra["A"] = score_A(c, ch, rng) if ch.notna().any() else {"note": "선정 지표 0개"}
        if ch.notna().any():
            ra["A"]["n_indicators_min_mean"] = [int(cnt.min()), float(cnt.mean())]
        if ch.notna().any():   # 보조: 국면별 다음 4주 실현 변동성(일 수익 표준편차 × √365, %)
            lr = np.log(panel[asset]).diff()
            vol = pd.Series({t: float(lr.loc[t + pd.Timedelta(days=1): t + pd.Timedelta(days=28)].std() * np.sqrt(365) * 100) for t in c.index})
            g = pd.DataFrame({"reg": ch > 0, "vol": vol}).dropna()
            ra["A_vol4w"] = {"up": float(g.vol[g.reg].mean()) if g.reg.any() else None, "down": float(g.vol[~g.reg].mean()) if (~g.reg).any() else None,
                             "n_up": int(g.reg.sum()), "n_down": int((~g.reg).sum())}
            rows.append(pd.DataFrame({"asset": asset, "c13": c, "chat13": ch}))
        res["assets"][asset] = ra
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    pd.concat(rows).to_csv(OUT / f"{tag}_predictions.csv", float_format="%.5f")
    return res


def verdicts(res: dict) -> dict:
    b, a = res["assets"]["ETH"]["B4"], res["assets"]["ETH"]["A"]
    g = lambda d, k, dflt: dflt if d.get(k) is None else d[k]   # noqa: E731 -- 없음(None·소표본)은 닫힌 쪽(실패)
    vb = {"cw>=1.282": g(b, "cw_t", -9) >= 1.282, "r2_os>0": g(b, "r2_os", -9) > 0, "excess_lo90>0": g(b, "excess_lo90", -9) > 0,
          "r2>placebo_p90": g(b, "r2_os", -9) > g(b, "placebo_p90", 9)}
    va = {"corr_lo90>0": g(a, "corr_lo90", -9) > 0, "excess_lo90>0": g(a, "excess_lo90", -9) > 0}
    return {"B_ETH_4w": {**vb, "PASS": all(vb.values())}, "A_ETH": {**va, "PASS": all(va.values())}}


# ── 2라운드 ─────────────────────────────────────────────────────────────────────────────────────────
def initial_release(panel: pd.DataFrame, sids=("DTWEXBGS", "STLFSI4")) -> pd.DataFrame:
    """개정 계열을 ALFRED 최초 공표값으로 바꾼 패널(2019-06~). FRED_API_KEY 필요."""
    import os
    import requests
    p = panel.copy()
    for sid in sids:
        ob = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=60, params={
            "series_id": sid, "api_key": os.environ["FRED_API_KEY"], "file_type": "json", "observation_start": "2019-06-01",
            "output_type": 4, "realtime_start": "1776-07-04", "realtime_end": "9999-12-31"}).json()["observations"]
        ini = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in ob if o["value"] not in (".", "")})
        ix = ini.index.intersection(p.index)
        p.loc[ix, sid] = ini.reindex(ix)
    return p


def r2_window(panel: pd.DataFrame, fz: dict, lo: str, hi: str | None) -> tuple[dict, pd.DataFrame]:
    w = sample_week(panel)
    zz = z91(panel, w.index)
    out, rows = {}, []
    hit = lambda s, y: float(((s > 0) == (y > 0)).mean())   # noqa: E731
    for asset in ("ETH", "BTC"):
        L, ra = labels(w, asset), {}
        for h in HORIZONS:
            sp = selected(fz, asset, f"B{h}")
            Z, cnt = z_combo(feats_for(w, sp), sp)
            end = hi and str((pd.Timestamp(hi) - pd.Timedelta(weeks=h)).date())
            d = pd.concat({"y": L[f"y{h}"], "z": Z, "n": cnt, "z91": zz}, axis=1).loc[lo:end].dropna(subset=["y", "z"])
            k = d.z.abs() >= 0.5
            ra[f"B{h}"] = {"n": int(len(d)), "n_indep": int(len(d) // h), "hit": hit(d.z, d.y), "always_up": float((d.y > 0).mean()),
                           "excess": hit(d.z, d.y) - float((d.y > 0).mean()), "pred_up_share": float((d.z > 0).mean()),
                           "k05_n": int(k.sum()), "k05_hit": hit(d.z[k], d.y[k]) if k.any() else None,
                           "z91_k0_hit": hit(d.z91.dropna(), d.y[d.z91.notna()]) if d.z91.notna().any() else None,
                           "n_indicators_min": int(d.n.min()) if len(d) else None, "first": str(d.index[0].date()) if len(d) else None,
                           "last": str(d.index[-1].date()) if len(d) else None}
            if h == 4:
                rows.append(d.assign(asset=asset)[["asset", "y", "z", "n", "z91"]].rename(columns={"y": "y4", "z": "z_b4"}))
        sp = selected(fz, asset, "A")
        Z, cnt = z_combo(feats_for(w, sp), sp)
        d = pd.concat({"c": L["c13"], "z": Z, "n": cnt}, axis=1).loc[lo:hi].dropna()
        ra["A"] = {"n": int(len(d)), "n_indep": int(len(d) // 13), "n_indicators_min": int(d.n.min()) if len(d) else None,
                   "hit": hit(d.z, d.c), "always_up": float((d.c > 0).mean()),
                   "excess": hit(d.z, d.c) - float((d.c > 0).mean()), "regime_up_share": float((d.z > 0).mean()),
                   "corr": float(d[["c", "z"]].corr().iloc[0, 1]) if len(d) > 2 else None}
        out[asset] = ra
    return out, pd.concat(rows) if rows else pd.DataFrame()


def stage_r2test(panel: pd.DataFrame, force: bool = False) -> dict:
    import hashlib
    mark = OUT / "r2_test_done.txt"
    if mark.exists() and not force:
        raise SystemExit(f"2라운드 시험은 한 번만: {mark} 가 있다")
    fz = json.loads(FROZEN.read_text())
    res = {"prereg": "docs/prereg/macro_regime_bias_r2_20261028.md", "frozen_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest()}
    res["test_current"], pred = r2_window(panel, fz, TEST_START, None)
    res["test_initial_release"], _ = r2_window(initial_release(panel), fz, TEST_START, None)
    res["posthoc_2024_2026_05_NOT_A_VERDICT"], _ = r2_window(panel, fz, SCORE[0], SCORE[1])
    v = {}
    for tag in ("test_current", "test_initial_release"):
        e = res[tag]["ETH"]
        deg = lambda share: " · 🔴퇴화(예보가 한쪽뿐 = 정보 없음)" if share in (0.0, 1.0) else ""   # noqa: E731 -- 1라운드 실패 모드(10-28 리뷰)
        v[tag] = {"B4": ("시험 통과(약)" if e["B4"]["excess"] >= 0 else "실패") + deg(e["B4"]["pred_up_share"]),
                  "A": ("시험 통과(약)" if e["A"]["excess"] >= 0 else "실패") + deg(e["A"]["regime_up_share"])}
    res["verdict"] = v
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r2_test_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    pred.to_csv(OUT / "r2_test_predictions.csv", float_format="%.5f")
    mark.write_text(pd.Timestamp.now().isoformat())
    return res


# ── 자체점검 ─────────────────────────────────────────────────────────────────────────────────────────
def selftest() -> None:
    import statsmodels.api as sm
    rng = np.random.default_rng(0)
    y, f0, f1 = rng.normal(size=200), np.zeros(200), rng.normal(size=200) * .1
    f = (y - f0) ** 2 - ((y - f1) ** 2 - (f0 - f1) ** 2)                     # CW 식(2.1) 손계산
    assert abs(clark_west(y, f0, f1, 0) - f.mean() / np.sqrt(((f - f.mean()) ** 2).mean() / len(f))) < 1e-9   # HAC 지연 0 = HC0
    assert clark_west(y, f0, f1, 3) != sm.OLS(f, np.ones(200)).fit().tvalues[0]                              # 지연이 실제로 들어간다
    assert r2_os(np.array([1., -1]), np.zeros(2), np.array([1., -1])) == 1.0
    idx = pd.date_range("2020-01-03", "2023-12-29", freq="W-FRI")
    sig = pd.Series(rng.normal(size=len(idx)), idx)
    yy = 3 * sig + rng.normal(size=len(idx))
    s = fit_one(sig, yy, FIT[0], FIT[1], .1)
    assert s["selected"] and s["b"] > 2, s                                   # 심은 신호는 뽑힌다
    assert not fit_one(pd.Series(rng.normal(size=len(idx)), idx), yy, FIT[0], FIT[1], .3)["selected"]   # 잡음은 안 뽑힌다
    # 확장 평균: τ 의 기준값은 라벨이 끝난(τ−h 이전) 것만
    yl = pd.Series(np.arange(10.0), pd.date_range("2020-01-03", periods=10, freq="W-FRI"))
    em = expanding_mean(yl, 4, "2020-01-03")
    assert np.isnan(em.iloc[3]) and em.iloc[4] == 0.0 and em.iloc[9] == np.mean(np.arange(6)), em.tolist()
    # 인과성: 날짜 T 뒤 패널을 바꿔도 T 다음 금요일까지의 피쳐(13주 변화·1주 미룸)는 그대로
    day = pd.date_range("2019-01-01", "2024-06-30", freq="D")
    lvl = pd.Series(np.exp(np.cumsum(rng.normal(0, .01, len(day)))), day)
    T = pd.Timestamp("2023-06-02")
    lvl2 = lvl.copy()
    lvl2[lvl2.index > T] *= 1.5
    feat = lambda s: (100 * np.log(s.resample("W-FRI").last())).diff(13).shift(1)   # noqa: E731
    a1, a2 = feat(lvl), feat(lvl2)
    assert np.allclose(a1.loc[:T + pd.Timedelta(days=7)].dropna(), a2.loc[:T + pd.Timedelta(days=7)].dropna())
    assert not np.allclose(a1.loc[T + pd.Timedelta(days=14):].dropna().iloc[:3], a2.loc[T + pd.Timedelta(days=14):].dropna().iloc[:3])
    ix = pd.date_range("2026-06-05", periods=17, freq="W-FRI")
    small = score_B(pd.Series(rng.normal(size=17), ix), pd.Series(rng.normal(size=17), ix), pd.Series(np.zeros(17), ix), 4, rng)   # 시험 구간 크기
    assert small["placebo_p90"] is None and not verdicts({"assets": {"ETH": {"B4": small, "A": {}}}})["B_ETH_4w"]["PASS"], small
    fz_ = {"a": {"r": -0.3}, "b": {"r": 0.2}}
    ix2 = pd.date_range("2020-01-03", periods=200, freq="W-FRI")
    xa, xb = pd.Series(rng.normal(size=200), ix2), pd.Series(rng.normal(size=200), ix2)
    Z, cnt = z_combo({"a": xa, "b": xb}, fz_)
    assert Z.iloc[:51].isna().all() and cnt.iloc[60] == 2                   # 52주 전엔 값 없음
    t = 120
    za = -xa.iloc[t] / xa.iloc[t - 103:t + 1].std(); zb = xb.iloc[t] / xb.iloc[t - 103:t + 1].std()
    assert abs(Z.iloc[t] - (za + zb) / 2) < 1e-12                          # 절편 없음 · 부호 × 변화 ÷ τ 까지 104주 sd
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel")
    ap.add_argument("--stage", choices=("fit", "score", "test", "r2test"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        raise SystemExit
    panel = pd.read_parquet(a.panel)
    if a.stage == "fit":
        fz = stage_fit(panel, a.force)
        for asset, fa in fz["assets"].items():
            for k in ("B4", "A"):
                print(asset, k, {n: (round(s["r"], 3), round(s["r_2020_21"], 3), round(s["r_2022_23"], 3)) for n, s in fa[k]["indicators"].items() if s["selected"]})
    elif a.stage == "r2test":
        r = stage_r2test(panel, a.force)
        print(json.dumps(r["verdict"], ensure_ascii=False))
    elif a.stage == "score":
        r = stage_score(panel, SCORE[0], SCORE[1], "score")
        print(json.dumps(verdicts(r), ensure_ascii=False))
    else:
        mark = OUT / "test_done.txt"
        if mark.exists() and not a.force:
            raise SystemExit(f"시험은 한 번만: {mark} 가 있다(--force 는 사용자 지시가 있을 때만)")
        r = stage_score(panel, TEST_START, None, "test")
        OUT.mkdir(parents=True, exist_ok=True)
        mark.write_text(pd.Timestamp.now().isoformat())
        print(json.dumps(verdicts(r), ensure_ascii=False))
