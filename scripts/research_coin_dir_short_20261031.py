#!/usr/bin/env python3
"""BTC·SOL 15분·1시간 방향 예측 — 배포 ETH 레시피(65열판) 이식 검정 (사전등록 docs/prereg/coin_dir_short_btc_sol_20261031.md).

XRP 판(scripts/research_xrp_dir_short_20261031.py)을 그대로 import — 데이터셋·표적·학습 행·채점은 그 파일 것.
바뀐 점 하나: 팔 E(ETH 학습)는 대상 코인과 무관하게 같은 모형이므로 (지평, 달)마다 한 번 학습해 BTC·SOL·XRP 행을 함께 예측한다.
XRP 행 = 양성 대조(XRP 실험의 ETH_* 예측 재현) · XRP 팔 X 는 2026-06 한 달만(대조).

    python scripts/research_coin_dir_short_20261031.py --selftest
    python scripts/research_coin_dir_short_20261031.py            # 학습 → tmp/coin_dir_short_20261031/{BTC,SOL,XRP_control}/preds.npz
    python scripts/research_coin_dir_short_20261031.py --control  # 양성 대조 판정(통과해야 --score)
    python scripts/research_coin_dir_short_20261031.py --score    # BTC·SOL 채점 → <COIN>/report.json
"""
from __future__ import annotations

import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_dir_short_20261028 as ds  # noqa: E402
import research_xrp_dir_short_20261031 as xr  # noqa: E402

OUT = ROOT / "tmp/coin_dir_short_20261031"
REF = ROOT / "tmp/xrp_dir_short_20261031"          # XRP 실험 산출(preds.npz·report.json) 사본 — 양성 대조 기준
COINS = ("BTC", "SOL")
CTRL_MONTH = pd.Timestamp("2026-06-01")
NJ = 1                                               # 공용 머신 스레드 3 = 프로세스 3 × 1 (XRP 판은 3 × 4 → 대조는 1e-9 일치가 아니라 근사 일치로 판정)
D: dict = {}


def fit_month(job):
    """(학습 코인, h, 달, 예측 대상들) → {대상: 시드 5 × 그 달 행}. 학습·스케일은 xr.fit_month 와 같은 식."""
    import lightgbm as lgb
    src, h, month, tgts = job
    t, X, ret, sig = D[src]; t_ns = t.asi8
    m0 = int(np.searchsorted(t_ns, month.value)); m1 = int(np.searchsorted(t_ns, (month + pd.offsets.MonthBegin(1)).value))
    y = ds.fwd(ret, sig, h); ok = np.isfinite(ret) & (sig > 0) & np.isfinite(y)
    tr = ds.train_rows(t_ns, ok, m0, h); hl = ds.HZ[h]
    w = None if hl is None else 0.5 ** ((t_ns[m0] - t_ns[tr]) / ds.DAY / hl)
    out = {g: [] for g in tgts}
    for seed in ds.SEEDS:
        acc = {g: 0.0 for g in tgts}
        for k in ds.ENS:
            m = lgb.LGBMRegressor(**ds.BASE, **({"objective": "huber", "alpha": 1.0} if k == "huber" else {}), n_jobs=NJ, verbose=-1,
                                  random_state=seed).fit(X[tr], ds.target(y[tr], k), sample_weight=w)
            sc = float(m.predict(X[tr[::50]]).std() + 1e-12)
            for g in tgts:
                acc[g] = acc[g] + m.predict(D[g][1][m0:m1]) / sc
        for g in tgts:
            out[g].append(acc[g])
    print(f"  {src} h{h} {month.date()} 학습 {len(tr):,}행 → {','.join(tgts)} 예측 {m1 - m0:,}", flush=True)
    return (src, h), m0, {g: np.array(v) for g, v in out.items()}


def run():
    for s in ("ETH", "XRP") + COINS:
        D[s] = xr.dataset65(xr.frames_for(s)); print("dataset", s, D[s][1].shape, flush=True)
    t = D["ETH"][0]; n = len(t)
    assert all((D[s][0] == t).all() for s in D), "코인별 결정 시각 격자가 다르다"
    jobs = [("ETH", h, m, ("BTC", "SOL", "XRP")) for h in ds.HZ for m in xr.MONTHS]
    jobs += [(c, h, m, (c,)) for c in COINS for h in ds.HZ for m in xr.MONTHS]
    jobs += [("XRP", h, CTRL_MONTH, ("XRP",)) for h in ds.HZ]
    P = {(g, a, h): np.full((len(ds.SEEDS), n), np.nan) for g in COINS + ("XRP",) for a in (g, "ETH") for h in ds.HZ}
    with get_context("fork").Pool(3) as pool:
        for (src, h), m0, preds in pool.imap_unordered(fit_month, jobs):
            for g, pr in preds.items():
                P[(g, src, h)][:, m0:m0 + pr.shape[1]] = pr
    for g in COINS + ("XRP",):
        d = OUT / (g if g in COINS else "XRP_control"); d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "preds.npz", t=t.asi8, ret=D[g][2], sig=D[g][3],
                            **{f"{a}_{h}": P[(g, a, h)] for a in (g, "ETH") for h in ds.HZ})
        print("saved", d / "preds.npz", flush=True)


def control():
    """사전등록 양성 대조: ① 저장 XRP 예측을 고친 채점기로 다시 채점 = report.json 그대로 ② 새 코드 XRP 예측 ≈ 저장 예측."""
    ref = json.loads((REF / "report.json").read_text())
    xr.score("XRP", REF)   # report.json 을 같은 내용으로 다시 쓴다(난수 시드 고정)
    again = json.loads((REF / "report.json").read_text())
    same = json.dumps(ref, sort_keys=True) == json.dumps(again, sort_keys=True)
    a, b = np.load(REF / "preds.npz"), np.load(OUT / "XRP_control" / "preds.npz")
    assert (a["t"] == b["t"]).all() and np.allclose(a["ret"], b["ret"], equal_nan=True) and np.allclose(a["sig"], b["sig"], equal_nan=True)
    t = pd.DatetimeIndex(a["t"]); rep = {"rescore_identical": same}; ok = same
    for h, nm in ds.NAMES.items():
        y = ds.fwd(a["ret"], a["sig"], h); mt = (t >= xr.TEST[0]) & (t < xr.TEST[1]) & np.isfinite(y)
        e_old, e_new = a[f"ETH_{h}"].mean(0), b[f"ETH_{h}"].mean(0)
        mm = (t >= CTRL_MONTH) & (t < CTRL_MONTH + pd.offsets.MonthBegin(1)) & np.isfinite(y)
        x_old, x_new = a[f"XRP_{h}"].mean(0)[mm], b[f"XRP_{h}"].mean(0)[mm]
        r = {"E_corr": float(np.corrcoef(e_old[mt], e_new[mt])[0, 1]),
             "E_test_ic_old": xr.spear(e_old[mt], y[mt]), "E_test_ic_new": xr.spear(e_new[mt], y[mt]),
             "X_ctrl_month_corr": float(np.corrcoef(x_old, x_new)[0, 1])}
        r["pass"] = r["E_corr"] >= .99 and r["X_ctrl_month_corr"] >= .99 and abs(r["E_test_ic_new"] - r["E_test_ic_old"]) <= .003
        ok &= r["pass"]; rep[nm] = r
    rep["pass"] = bool(ok)
    (OUT / "control.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1)); print(json.dumps(rep, ensure_ascii=False, indent=1))
    return ok


def score():
    assert json.loads((OUT / "control.json").read_text())["pass"], "양성 대조 불통과 — 채점 금지(사전등록)"
    for c in COINS:
        print("==", c); xr.score(c, OUT / c)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        for c in ("XRP",) + COINS:
            xr.selftest(c)
    elif "--control" in sys.argv:
        sys.exit(0 if control() else 1)
    elif "--score" in sys.argv:
        score()
    else:
        run()
