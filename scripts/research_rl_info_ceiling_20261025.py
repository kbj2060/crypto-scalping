#!/usr/bin/env python3
"""강화학습 1단계 — 복제 기계 피쳐의 방향 정보 상한(손익분기 IC). 사전 등록 docs/prereg/rl_info_ceiling_20261025.md (718cdce6).

    python3 scripts/research_rl_info_ceiling_20261025.py

IC 하한이 손익분기 IC 를 넘는 칸이 없으면 RL 2단계는 돌리지 않는다. 산출 tmp/rl_info_ceiling_20261025/report.json.
"""
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import research_system_trend_ruleb_20261025 as SYS    # noqa: E402
import research_user_clone_ml_20261025 as CL           # noqa: E402

OUT = SYS.ROOT / "tmp" / "rl_info_ceiling_20261025"
FEATS = ["r1h", "r4h", "r24h", "r7d", "rv1h", "rv4h", "rv24h", "rv7d", "rv_ratio", "park24", "jump24", "lo24", "hi24", "trend", "hsin", "hcos"]
HS = {"1h": 12, "4h": 48, "24h": 288}
TRAIN = ("2022-01-08", "2024-12-31 23:59"); TEST = ("2025-01-01", "2026-07-31 23:59"); COST = 4e-4


def rows(sym, x):
    f = CL.mkt(x); idx = np.flatnonzero((x.index + pd.Timedelta("5min")).minute == 0); c = x.c.values
    r = f.iloc[idx][FEATS].copy(); r["t"] = x.index[idx] + pd.Timedelta("5min"); r["sym"] = sym
    for k, h in HS.items():
        j = idx + h; r[f"y{k}"] = np.where(j < len(c), np.log(c[np.minimum(j, len(c) - 1)] / c[idx]), np.nan)   # 봉 i+1..i+h(결정 봉 뒤부터)
    return r


def ic_ci(pred, y, day, n=2000, seed=0):
    ic = float(spearmanr(pred, y).statistic); ud = np.unique(day); gi = {d: np.flatnonzero(day == d) for d in ud}
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        ix = np.concatenate([gi[d] for d in rng.choice(ud, len(ud))]); bs.append(spearmanr(pred[ix], y[ix]).statistic)
    return ic, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([rows(s, SYS.bars5(s)[0]) for s in SYS.COINS]).replace([np.inf, -np.inf], np.nan)
    rep = {"prereg": "docs/prereg/rl_info_ceiling_20261025.md@718cdce6", "cost_roundtrip": COST, "cells": {}}
    for k in HS:
        d = df.dropna(subset=FEATS + [f"y{k}"]); tr = d[(d.t >= TRAIN[0]) & (d.t <= TRAIN[1])]; te = d[(d.t >= TEST[0]) & (d.t <= TEST[1])].copy()
        tr = tr[tr.t + pd.Timedelta(hours=int(k[:-1])) <= pd.Timestamp(TRAIN[1])]   # 학습 표적이 시험 구간으로 넘어가지 않게
        hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False, random_state=0)
        hgb.fit(tr[FEATS], tr[f"y{k}"]); mu, sd = tr[FEATS].mean(), tr[FEATS].std()
        rid = Ridge(alpha=1.0).fit((tr[FEATS] - mu) / sd, tr[f"y{k}"])
        te["p_hgb"] = hgb.predict(te[FEATS]); te["p_ridge"] = rid.predict((te[FEATS] - mu) / sd)
        if k == "4h":   # T2 누수 트립와이어: 표적을 피쳐에 섞으면 IC 폭증
            leak = HistGradientBoostingRegressor(max_iter=100, random_state=0).fit(np.c_[tr[FEATS], tr[f"y{k}"]], tr[f"y{k}"])
            lic = float(spearmanr(leak.predict(np.c_[te[FEATS], te[f"y{k}"]]), te[f"y{k}"]).statistic); assert lic > 0.9, lic
            rep["T2_leak_ic_4h"] = lic
        for scope, g in (("ETH", te[te.sym == "ETH"]), ("pool", te)):
            sig = float(g[f"y{k}"].std()); be = COST / (1.4 * sig); day = g.t.dt.floor("D").values
            cell = {"sigma_bp": sig * 1e4, "ic_breakeven": be, "n": len(g)}
            for mname in ("p_hgb", "p_ridge"):
                ic, ci = ic_ci(g[mname].values, g[f"y{k}"].values, day); cell[mname] = {"ic": ic, "ci": ci, "edge_bp_est": ic * sig * 1.4 * 1e4}
            cell["pass"] = any(cell[mn]["ci"][0] > be for mn in ("p_hgb", "p_ridge")); rep["cells"][f"{scope}_{k}"] = cell
            print(scope, k, f"σ {sig * 1e4:.0f}bp · 손익분기 IC {be:.4f} · HGB IC {cell['p_hgb']['ic']:+.4f} {np.round(cell['p_hgb']['ci'], 4).tolist()}"
                  f" · 리지 {cell['p_ridge']['ic']:+.4f} {np.round(cell['p_ridge']['ci'], 4).tolist()} · 통과 {cell['pass']}", flush=True)
    rep["go_stage2"] = any(c["pass"] for c in rep["cells"].values())
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print("go_stage2:", rep["go_stage2"], "· T2 누수판 IC", round(rep["T2_leak_ic_4h"], 3))


if __name__ == "__main__":
    sys.exit(main())
