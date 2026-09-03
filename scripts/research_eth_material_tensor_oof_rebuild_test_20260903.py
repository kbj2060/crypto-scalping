#!/usr/bin/env python3
"""재료 텐서 **OOF 재생성** + 하류 재시험 (2026-09-03).

선행 진단(`docs/experiments/eth_evidence_signal_tp_truncation_vs_horizon_hold_20260903.md` 후속2):
현행 텐서의 `_proba`는 **TRAIN 전체에서 in-sample**이라, TRAIN 안을 시간순으로 잘라도
내부검증이 +0.48인데 진짜 표본외는 −0.02다. 용량을 8배 줄여도 innerVAL이 불변이고, 학습
곡선이 교과서적으로 건강한데 표본외가 음수이며, 5시드가 소수점 4자리까지 같았다 -- 전부 누수의
서명이다. **과적합이 아니었다.**

여기서는 진입 모델이 쓴 것과 **동일한 OOF 산출**(`tmp/eth_entry_oof_metalabel_20260903/`,
워밍업 2024-01~04 제외 · 2024-05~2025-08 확장창 4-fold · 2025-09 이후는 TRAIN 전체 학습)을
텐서에 넣어 다시 묻는다: **"이제는 학습이 값어치 있는가?"**

⭐사전등록(데이터 보기 전 고정):
  ①**누수가 실제로 줄었는가** -- innerVAL과 VAL의 간극이 좁아져야 한다(현행 +0.48 vs −0.02).
    안 좁아지면 원인 귀속이 틀린 것이므로 그 사실을 그대로 보고한다.
  ②**학습이 값어치 있는가** -- OOF 텐서로 학습한 모델이 **OOF 원시 합**을 VAL·OOS 양 창에서
    이겨야 한다. 미달이면 재료의 사용 형태는 여전히 원시 합이다.
비교 공정성을 위해 기준선도 **OOF 원시 합**으로 다시 만든다(현행 텐서 기준선과 섞지 않는다).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, OOS_START, VAL_START)

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OOFD = ROOT / "tmp/eth_entry_oof_metalabel_20260903"
ETH_REGIME = ROOT / "tmp/eth_regime_s12k3_clean_20260902/predictions.parquet"
BTC_REGIME = ROOT / "tmp/btc_regime_s24k3_clean_20260902/predictions.parquet"
OUT = ROOT / "tmp/eth_material_oof_rebuild_20260903"
FWD, INNER = 24, 0.85
STEP, MAX_ITER, PATIENCE = 20, 600, 5
SEEDS5 = [76010, 130820, 194636, 331076, 703883]
WARMUP_END = pd.Timestamp("2024-05-01")


def log(m): print(f"[oofreb] {m}", flush=True)


def build(use_oof: bool):
    """use_oof=True면 pct_oof/proba_oof, False면 현행 in-sample 열을 쓴다."""
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines
    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    kl = load_klines()[["timestamp", "close"]].sort_values("timestamp").reset_index(drop=True)
    n = len(kl); ts = pd.DatetimeIndex(kl["timestamp"])
    pos = {t: i for i, t in enumerate(ts)}
    c = kl["close"].to_numpy(float)
    out = pd.DataFrame({"timestamp": kl["timestamp"]}); cols = []
    S = np.zeros(n)
    for name, cc in cfg.items():
        H = int(cc["horizon"])
        if use_oof:
            d = pd.read_csv(OOFD / f"{name}_oof.csv", parse_dates=["timestamp"])
            d = d[np.isfinite(d["pct_oof"])].copy()
            d["p"], d["pc"] = d["proba_oof"], d["pct_oof"]
        else:
            f = SRC / f"{name}_causal_proba_cal.csv"
            d = pd.read_csv(f if f.exists() else SRC / f"{name}_causal_proba.csv",
                            parse_dates=["timestamp"])
            trp = np.sort(d.loc[d.split == "TRAIN", "proba"].to_numpy(float))
            d["p"] = d["proba"]
            d["pc"] = np.searchsorted(trp, d["proba"].to_numpy(float),
                                      side="right") / max(len(trp), 1)
        d = d[d.timestamp.isin(pos)].copy()
        d["i"] = [pos[t] for t in d.timestamp]
        d["dir"] = np.where(d.is_bottom == 1, 1.0, -1.0)
        d = d.sort_values(["i", "p"]).drop_duplicates("i", keep="last")
        fire = np.zeros(n); fire[d["i"].to_numpy()] = d["dir"].to_numpy()
        pi = {int(i): (float(a), float(b), float(x)) for i, a, b, x in
              zip(d["i"], d["p"], d["pc"], d["dir"])}
        P = np.zeros((n, 3)); age = np.ones(n)
        last, li = (0.0, 0.0, 0.0), -10**9
        for i in range(n):
            if i in pi: last, li = pi[i], i
            el = i - li
            if el < H:
                P[i] = last; age[i] = el / H
        sgn = P[:, 1] * P[:, 2]
        S += sgn
        for tag, arr in (("fire", fire), ("proba", P[:, 0]), ("pct", P[:, 1]),
                         ("signed", sgn), ("age", age)):
            out[f"{name}_{tag}"] = arr; cols.append(f"{name}_{tag}")
    for tag, p in (("eth", ETH_REGIME), ("btc", BTC_REGIME)):
        if p.exists():
            r = pd.read_parquet(p)
            out = out.merge(r.rename(columns={"regime": f"regime_{tag}"}), on="timestamp", how="left")
            out[f"regime_{tag}"] = out[f"regime_{tag}"].ffill().fillna(-1).astype(int)
            cols.append(f"regime_{tag}")
    y = np.concatenate([(c[FWD:] - c[:-FWD]) / c[:-FWD], np.full(FWD, np.nan)])
    split = np.where(ts < VAL_START, "TRAIN", np.where(ts < OOS_START, "VAL",
                     np.where(ts < HOLDOUT_START, "OOS", "HOLDOUT")))
    return out, cols, y, split, S, ts


def ic(p, y, m):
    return float(spearmanr(p[m], y[m])[0]) if m.sum() > 200 else np.nan


def fit_ts(X, y, itr, ival, seed, hp, verbose=False):
    m = HistGradientBoostingRegressor(random_state=seed, loss="squared_error",
                                      early_stopping=False, warm_start=True, max_iter=STEP, **hp)
    best, bit, bad = -np.inf, STEP, 0
    for it in range(STEP, MAX_ITER + 1, STEP):
        m.set_params(max_iter=it); m.fit(X[itr], y[itr])
        tr_ic = float(spearmanr(m.predict(X[itr]), y[itr])[0])
        va_ic = float(spearmanr(m.predict(X[ival]), y[ival])[0])
        if verbose:
            log(f"    iter={it:4d} train_IC={tr_ic:+.4f} inner_val_IC={va_ic:+.4f}"
                f"{'  ← best' if va_ic > best else ''}")
        if va_ic > best: best, bit, bad = va_ic, it, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    f = HistGradientBoostingRegressor(random_state=seed, loss="squared_error",
                                      early_stopping=False, max_iter=bit, **hp)
    f.fit(X[itr | ival], y[itr | ival])
    return f, bit, best


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    W = ("VAL", "OOS", "HOLDOUT")
    res = {}
    for tag, use_oof in (("현행(in-sample)", False), ("OOF", True)):
        D, cols, y, split, S, ts = build(use_oof)
        ok = np.isfinite(y)
        # ⭐OOF는 워밍업(2024-01~04)을 학습에서 제외한다 -- 그 구간엔 OOF 값이 없다
        # OOF는 워밍업(2024-01~04) 구간에 값이 없으므로 학습에서 제외한다
        warm = np.asarray(ts >= WARMUP_END) if use_oof else np.ones(len(y), bool)
        tr_all = (split == "TRAIN") & ok & warm
        idx = np.flatnonzero(tr_all); cut = idx[int(len(idx) * INNER)]
        itr = tr_all & (np.arange(len(y)) < cut); ival = tr_all & (np.arange(len(y)) >= cut)
        base = {w: ic(S, y, (split == w) & ok) for w in W}
        log(f"\n════ {tag} ════  TRAIN {int(tr_all.sum()):,} (내부 {int(itr.sum()):,}/{int(ival.sum()):,}) "
            f"· 피쳐 {len(cols)}")
        log(f"  기준선(원시 signed 합)  " + " ".join(f"{w} {base[w]:+.4f}" for w in W))
        X = D[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X[tr_all].median()).to_numpy(float)

        sc = StandardScaler().fit(X[tr_all]); Xs = sc.transform(X)
        rg = {}
        for a in (1.0, 100.0, 10000.0):
            p = Ridge(alpha=a).fit(Xs[tr_all], y[tr_all]).predict(Xs)
            rg[a] = {w: ic(p, y, (split == w) & ok) for w in W}
            log(f"  Ridge α={a:>7.0f}  " + " ".join(f"{w} {rg[a][w]:+.4f}" for w in W))

        hp = {"max_leaf_nodes": 31, "min_samples_leaf": 200, "l2_regularization": 1.0,
              "learning_rate": 0.03}
        ps, its, ivs = [], [], []
        for k, sd in enumerate(SEEDS5):
            m, bit, biv = fit_ts(X, y, itr, ival, sd, hp, verbose=(k == 0))
            ps.append(m.predict(X)); its.append(bit); ivs.append(biv)
        p = np.mean(ps, axis=0)
        hg = {w: ic(p, y, (split == w) & ok) for w in W}
        innerVAL = float(np.mean(ivs))
        log(f"  HGB 5시드(중단 iter {its}) innerVAL {innerVAL:+.4f}  "
            + " ".join(f"{w} {hg[w]:+.4f}" for w in W))
        log(f"  ⭐누수지표 innerVAL − VAL = {innerVAL - hg['VAL']:+.4f}")
        res[tag] = {"base": base, "ridge": rg, "hgb": hg, "innerVAL": innerVAL,
                    "gap": innerVAL - hg["VAL"], "n_train": int(tr_all.sum())}

    A, B = res["현행(in-sample)"], res["OOF"]
    print(f"\n{'':28s}" + "".join(f"{w:>10s}" for w in W))
    for tag, r in res.items():
        print(f"{tag+' 원시 합':28s}" + "".join(f"{r['base'][w]:+10.4f}" for w in W))
        print(f"{tag+' HGB':28s}" + "".join(f"{r['hgb'][w]:+10.4f}" for w in W))
    print(f"\n⭐사전등록 판정")
    print(f"  ①누수 감소: 간극 현행 {A['gap']:+.4f} → OOF {B['gap']:+.4f} "
          f"{'✅ 좁아짐' if abs(B['gap']) < abs(A['gap']) else '❌ 안 좁아짐'}")
    win = all(B["hgb"][w] > B["base"][w] for w in ("VAL", "OOS"))
    print(f"  ②학습 값어치(OOF HGB > OOF 원시 합, VAL·OOS): {'✅' if win else '❌'}")
    print(f"     VAL {B['hgb']['VAL']:+.4f} vs {B['base']['VAL']:+.4f} · "
          f"OOS {B['hgb']['OOS']:+.4f} vs {B['base']['OOS']:+.4f}")
    print(f"  → {'**OOF 텐서로 학습이 값어치 있다**' if win else '**여전히 원시 합이 사용 형태다**'}")
    json.dump(res, open(OUT / "result.json", "w"), ensure_ascii=False, indent=2, default=str)
    log(f"\n{time.time()-t0:.0f}초 · 산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
