#!/usr/bin/env python3
"""B6: 표본 확장 -- 슬롯 증가 + 깊이/대기 격자 확장 (2026-09-03).

B4/B5가 정반대 결과를 냈다:
  τ 임계값 방식  -- 무작위 필터 대조군 3창 통과(p=0.000)인데 n=32~67, 독립블록 9일. 표본 부족.
  M 순위배분 방식 -- 표본 충분(n=201~297)인데 무작위 M을 못 이김(p=0.25~0.55). 순위 무가치.
차이는 거르는 기준이다 -- τ는 **팔의 절대 품질**(전체 상위 12%), M은 **동시 대기 중 상대 순위**.
대기 중인 것들끼리는 서로 비슷해 순위 정보가 없다.

그래서 τ 방식을 유지한 채 **체결 자체를 늘린다**:
  ① 슬롯 1→2,3,4,6
  ② 깊이 {2.0,2.5,3.0,3.5} × 대기 {6,12}
     대기는 상위집합이므로 wait=12로 한 번 시뮬하고 bars_to_fill<=6으로 걸러 wait=6을 파생한다.

모델은 깊이/대기를 피쳐로 넣어 **한 번만** 학습한다(조합별 학습은 선택 표면을 키운다).
체결 탐지는 벡터화(오프셋 루프), 트레일링만 건별 루프.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, OOS_START, VAL_START)
from research_eth_entry_direction_oracle_ceiling_20260903 import NOTIONAL, COST, trail_out  # noqa: E402
from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_regime_s12k3_label_train_20260902 import GBM3_MODEL_PATH, load_frame  # noqa: E402

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OUT = ROOT / "tmp/eth_entry_b6_expand_20260903"
DEPTHS = [2.0, 2.5, 3.0, 3.5]
WAIT_MAX = 12
WAITS = [6, 12]
SLOTS = [1, 2, 3, 4, 6]
TAUS = [0.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030]


def log(m): print(f"[b6] {m}", flush=True)


def slotN(df, n_slots):
    if df.empty: return np.array([])
    d = df.sort_values("fi")
    taken, busy = [], []
    for fi, ei, y in zip(d.fi.to_numpy(), d.ei.to_numpy(), d.y.to_numpy()):
        busy = [b for b in busy if b > fi]
        if len(busy) < n_slots:
            taken.append(y); busy.append(ei)
    return np.asarray(taken, float)


def main() -> int:
    cfg = json.loads((SRC / "config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    hz = {k: int(v["horizon"]) for k, v in cfg["cfg"].items()}
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(kl["timestamp"])
    h, l, c = (kl[k].to_numpy(float) for k in ("high", "low", "close"))
    pos_of = {t: i for i, t in enumerate(ts)}
    n = len(kl)

    T = []
    for name in cfg["cfg"]:
        d0 = pd.read_csv(SRC / f"{name}_causal_fires.csv", parse_dates=["timestamp"])
        d0 = d0[d0.timestamp.isin(pos_of)].copy()
        d0["i"] = [pos_of[t] for t in d0.timestamp]
        d0["signal"] = name
        sel_cols = list(dict.fromkeys(["i", "timestamp", "side", "atr_pct", "signal"] + base))
        T.append(d0[sel_cols])
    T = pd.concat(T, ignore_index=True)
    T = T[np.isfinite(T.atr_pct) & (T.atr_pct > 0)].reset_index(drop=True)
    log(f"트리거 {len(T):,}")

    recs = []
    for depth in DEPTHS:
        for armv in (1, 0):
            sd = np.where(T.side.to_numpy() == "bottom", 1, -1) * (1 if armv else -1)
            i = T.i.to_numpy(); a = T.atr_pct.to_numpy()
            below = sd > 0
            lim = np.where(below, c[i] * (1 - depth * a), c[i] * (1 + depth * a))
            fill = np.full(len(T), -1, dtype=np.int64)
            for off in range(1, WAIT_MAX + 1):
                k = i + off
                ok = (fill < 0) & (k < n)
                if not ok.any(): continue
                kk = k[ok]
                hit = np.where(below[ok], l[kk] <= lim[ok], h[kk] >= lim[ok])
                idx = np.flatnonzero(ok)[hit]
                fill[idx] = i[idx] + off
            got = fill > 0
            H = T.signal.map(hz).to_numpy()
            ok = got & (fill + H < n)
            sub = T[ok].copy()
            sub["arm"] = armv; sub["depth"] = depth
            sub["fi"] = fill[ok]; sub["btf"] = fill[ok] - i[ok]
            sub["ei"] = fill[ok] + H[ok]
            sub["lim"] = lim[ok]; sub["sd"] = sd[ok]
            ys = []
            for e, aa, s_, f_, hh in zip(sub.lim, sub.atr_pct, sub.sd, sub.fi, H[ok]):
                f_ = int(f_)
                mv = trail_out(int(s_), float(e), float(aa), h[f_:f_ + hh], l[f_:f_ + hh], c[f_:f_ + hh])
                ys.append(float(mv * NOTIONAL - COST * NOTIONAL))
            sub["y"] = ys
            recs.append(sub)
        log(f"  depth {depth}: 누적 체결 {sum(len(r) for r in recs):,}")
    D = pd.concat(recs, ignore_index=True)
    D["split"] = np.where(D.timestamp < VAL_START, "TRAIN",
                   np.where(D.timestamp < OOS_START, "VAL",
                   np.where(D.timestamp < HOLDOUT_START, "OOS", "HOLDOUT")))
    D = D[D.timestamp >= pd.Timestamp("2024-05-01")].reset_index(drop=True)
    log(f"\n총 체결 {len(D):,} | " + " ".join(f"{k} {int(v):,}" for k, v in D.split.value_counts().items()))

    # 136 패널 조인
    src = joblib.load(GBM3_MODEL_PATH); cols = list(dict.fromkeys(src["feature_cols"])); med = src["feature_medians"]
    rf = load_frame()
    rf = rf.loc[:, ~pd.Index(rf.columns).duplicated()]
    x = rf[["timestamp"] + cols].copy()
    x = x.drop_duplicates("timestamp").reset_index(drop=True)
    for cc in cols:
        x[cc] = pd.to_numeric(x[cc], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med.get(cc, 0.0))
    dup = [cc for cc in cols if cc in D.columns]
    x = x.rename(columns={cc: cc + "_r136" for cc in dup})
    R136 = [(cc + "_r136" if cc in dup else cc) for cc in cols]
    D = D.reset_index(drop=True).merge(x, on="timestamp", how="left")
    D = D.loc[:, ~pd.Index(D.columns).duplicated()].dropna(subset=R136).reset_index(drop=True)
    D["sig_id"] = pd.Categorical(D.signal).codes
    FEATS = base + ["arm", "sig_id", "atr_pct", "depth"] + R136
    X = D[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (D.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    D["pred"] = np.mean([HistGradientBoostingRegressor(random_state=s, **HP)
                         .fit(X[tr], D.loc[tr, "y"]).predict(X) for s in SEEDS], axis=0)
    log(f"학습 완료 TRAIN {int(tr.sum()):,} · 피쳐 {len(FEATS)}")

    rows = []
    for depth in DEPTHS:
        for wait in WAITS:
            sel = (D.depth == depth) & (D.btf <= wait)
            for tau in TAUS:
                for ns in SLOTS:
                    r = {"depth": depth, "wait": wait, "tau_bp": tau * 1e4, "slots": ns}
                    for wn in ("VAL", "OOS", "HOLDOUT"):
                        w = D[sel & (D.split == wn) & (D.pred > tau)]
                        v = slotN(w, ns); nn, m, pf = stat(v)
                        r[f"{wn}_bp"] = round(m, 2); r[f"{wn}_n"] = nn
                    rows.append(r)
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / "grid.csv", index=False)
    D.to_csv(OUT / "fills.csv", index=False)

    log("\n=== 표본이 충분한(VAL n>=200 & OOS n>=150) 조합 중 VAL 상위 12 ===")
    g = R[(R.VAL_n >= 200) & (R.OOS_n >= 150)].sort_values("VAL_bp", ascending=False)
    pd.set_option("display.width", 220)
    print(g.head(12)[["depth", "wait", "tau_bp", "slots", "VAL_bp", "VAL_n",
                      "OOS_bp", "OOS_n", "HOLDOUT_bp", "HOLDOUT_n"]].to_string(index=False))
    log(f"\n조건 충족 조합 {len(g)}/{len(R)}")
    base_rows = R[(R.tau_bp == 0.0)]
    log("\n=== 참고: 무필터(τ=0) 기준선 중 VAL 상위 6 ===")
    print(base_rows.sort_values("VAL_bp", ascending=False).head(6)[
        ["depth", "wait", "slots", "VAL_bp", "VAL_n", "OOS_bp", "OOS_n", "HOLDOUT_bp", "HOLDOUT_n"]].to_string(index=False))
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
