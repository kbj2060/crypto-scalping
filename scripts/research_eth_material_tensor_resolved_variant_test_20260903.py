#!/usr/bin/env python3
"""재료 텐서 유지규칙 3변형 하류 테스트 (2026-09-03).

`docs/experiments/eth_evidence_signal_tp_truncation_vs_horizon_hold_20260903.md`의 권고
("둘 중 고르지 말고 `_resolved` 열을 추가하라")를 실제 하류 과제로 검증한다.

  A 대시보드 로직  -- 라벨 확정 시 유지 중단 (`_fill_until_tp_or_horizon`과 동일)
  B 현행 텐서      -- HORIZON 내내 유지
  C B + `_resolved` -- 유지하되 확정 여부를 열로 준다 (정보상 A와 B를 모두 포함)
  D 대조군          -- C와 같되 `_resolved`를 **셔플**(이득이 노이즈인지 판정)

⭐사전등록(데이터 보기 전 고정): **C가 VAL·OOS 양 창에서 A와 B를 모두 이겨야** 열 추가를
채택한다. 미달이면 현행 B 유지. 그리고 C가 D(셔플)를 양 창에서 이기지 못하면 이득은 노이즈다.

과제: 전방 24봉 수익 회귀(HGB 5시드). 지표는 예측-실현 Spearman IC.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, OOS_START, VAL_START)
from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP  # noqa: E402

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
ETH_REGIME = ROOT / "tmp/eth_regime_s12k3_clean_20260902/predictions.parquet"
BTC_REGIME = ROOT / "tmp/btc_regime_s24k3_clean_20260902/predictions.parquet"
OUT = ROOT / "tmp/eth_material_resolved_variant_20260903"
HIT_RESOLUTION = {"fib_extension_exhaustion": "touch_and_mae"}
FIB_K_LOSS_MULT = 2.0
FWD = 24
RNG = np.random.default_rng(20260903)


def log(m): print(f"[variant] {m}", flush=True)


def resolve_bar(i, side, k, hz, mode, c, h, lo, ap, n):
    end = min(i + hz, n - 1)
    if not np.isfinite(ap):
        return end
    t = k * ap
    if mode == "touch":
        lv = c[i] * (1 - t) if side == "top" else c[i] * (1 + t)
    else:
        lv = c[i] * (1 + FIB_K_LOSS_MULT * t) if side == "top" else c[i] * (1 - FIB_K_LOSS_MULT * t)
    for b in range(i + 1, end + 1):
        if mode == "touch":
            done = (lo[b] <= lv) if side == "top" else (h[b] >= lv)
        else:
            done = (h[b] >= lv) if side == "top" else (lo[b] <= lv)
        if done:
            return b
    return end


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines

    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    kl = load_klines()[["timestamp", "high", "low", "close"]].sort_values(
        "timestamp").reset_index(drop=True)
    n = len(kl)
    ts = pd.DatetimeIndex(kl["timestamp"])
    pos_of = {t: i for i, t in enumerate(ts)}
    c, h, lo = (kl[x].to_numpy(float) for x in ("close", "high", "low"))
    log(f"기준 {n:,}봉 {ts.min()} ~ {ts.max()}")

    base = pd.DataFrame({"timestamp": kl["timestamp"]})
    colsA, colsB, res_cols = [], [], []
    for name, cc in cfg.items():
        H, k = int(cc["horizon"]), float(cc["k"])
        mode = HIT_RESOLUTION.get(name, "touch")
        f = SRC / f"{name}_causal_proba_cal.csv"
        d = pd.read_csv(f if f.exists() else SRC / f"{name}_causal_proba.csv",
                        parse_dates=["timestamp"])
        if "proba_cal" not in d.columns:
            d["proba_cal"] = d["proba"]
        ap = pd.read_csv(SRC / f"{name}_causal_fires.csv",
                         parse_dates=["timestamp"])[["timestamp", "atr_pct"]]
        d = d.merge(ap, on="timestamp", how="left")
        trp = np.sort(d.loc[d.split == "TRAIN", "proba"].to_numpy(float))
        d["proba_pct"] = np.searchsorted(trp, d["proba"].to_numpy(float),
                                         side="right") / max(len(trp), 1)
        d = d[d["timestamp"].isin(pos_of)].copy()
        d["i"] = [pos_of[t] for t in d["timestamp"]]
        d["dir"] = np.where(d["is_bottom"] == 1, 1.0, -1.0)
        d = d.sort_values(["i", "proba"]).drop_duplicates("i", keep="last")

        fire = np.zeros(n)
        fire[d["i"].to_numpy()] = d["dir"].to_numpy()
        # 각 발동의 확정 봉을 미리 구한다(인과적: 시점 b의 resolved는 b까지의 정보만 씀)
        rb = {int(i): resolve_bar(int(i), ("bottom" if dd > 0 else "top"), k, H, mode,
                                  c, h, lo, float(a), n)
              for i, dd, a in zip(d["i"], d["dir"], d["atr_pct"])}
        pi = {int(i): (float(a), float(b), float(cx), float(dd)) for i, a, b, cx, dd in
              zip(d["i"], d["proba"], d["proba_cal"], d["proba_pct"], d["dir"])}
        pB = np.zeros(n); calB = np.zeros(n); pctB = np.zeros(n); dB = np.zeros(n)
        ageB = np.ones(n); resolved = np.zeros(n)
        last, last_i = (0.0, 0.0, 0.0, 0.0), -10**9
        for i in range(n):
            if i in pi:
                last, last_i = pi[i], i
            el = i - last_i
            if el < H:
                pB[i], calB[i], pctB[i], dB[i] = last
                ageB[i] = el / H
                resolved[i] = 1.0 if i > rb.get(last_i, 10**9) else 0.0
            else:
                ageB[i] = 1.0
        m = resolved == 0                                  # A = 확정 전 구간만
        for tag, arr in (("proba", pB), ("proba_cal", calB), ("pct", pctB),
                         ("signed", pctB * dB), ("age", ageB), ("fire", fire)):
            base[f"{name}_{tag}_B"] = arr
            if tag == "age":
                base[f"{name}_{tag}_A"] = np.where(m, arr, 1.0)
            elif tag == "fire":
                base[f"{name}_{tag}_A"] = arr              # 발동 자체는 동일
            else:
                base[f"{name}_{tag}_A"] = np.where(m, arr, 0.0)
            colsB.append(f"{name}_{tag}_B"); colsA.append(f"{name}_{tag}_A")
        base[f"{name}_resolved"] = resolved
        res_cols.append(f"{name}_resolved")
        log(f"  {name:26s} H={H:<3} 발동 {len(d):6,} · 커버리지 B {(pB>0).mean():.1%} "
            f"→ A {((pB>0)&m).mean():.1%} · 확정구간 {resolved.mean():.1%}")

    reg = []
    for tag, p in (("eth", ETH_REGIME), ("btc", BTC_REGIME)):
        if p.exists():
            r = pd.read_parquet(p)
            base = base.merge(r.rename(columns={"regime": f"regime_{tag}"}),
                              on="timestamp", how="left")
            base[f"regime_{tag}"] = base[f"regime_{tag}"].ffill().fillna(-1).astype(int)
            reg.append(f"regime_{tag}")

    y = np.concatenate([(c[FWD:] - c[:-FWD]) / c[:-FWD], np.full(FWD, np.nan)])
    split = np.where(ts < VAL_START, "TRAIN", np.where(ts < OOS_START, "VAL",
                     np.where(ts < HOLDOUT_START, "OOS", "HOLDOUT")))
    ok = np.isfinite(y)
    tr = (split == "TRAIN") & ok
    base["_resolved_shuf"] = 0.0
    D = base.copy()
    for rc in res_cols:
        D[rc + "_shuf"] = RNG.permutation(D[rc].to_numpy())

    VAR = {"A 대시보드(확정시 중단)": colsA + reg,
           "B 현행(HORIZON 유지)": colsB + reg,
           "C B+resolved": colsB + reg + res_cols,
           "D 대조군(resolved 셔플)": colsB + reg + [r + "_shuf" for r in res_cols]}
    log(f"\n학습 TRAIN {int(tr.sum()):,}봉 · 목표 전방 {FWD}봉 수익 · 시드 {len(SEEDS)}\n")
    print(f"{'변형':26s} {'피쳐':>5s} " + " ".join(f"{w:>9s}" for w in ("TRAIN", "VAL", "OOS", "HOLDOUT")))
    res = {}
    for nm, cols in VAR.items():
        X = D[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X[tr].median())
        p = np.mean([HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
                     .fit(X[tr], y[tr]).predict(X) for s in SEEDS], axis=0)
        row = {}
        for wn in ("TRAIN", "VAL", "OOS", "HOLDOUT"):
            m2 = (split == wn) & ok
            row[wn] = float(spearmanr(p[m2], y[m2])[0])
        res[nm] = row
        print(f"{nm:26s} {len(cols):5d} " + " ".join(f"{row[w]:+9.4f}" for w in
              ("TRAIN", "VAL", "OOS", "HOLDOUT")))

    A, B, C, Dv = (res[k] for k in VAR)
    win = all(C[w] > A[w] and C[w] > B[w] for w in ("VAL", "OOS"))
    nz = all(C[w] > Dv[w] for w in ("VAL", "OOS"))
    print(f"\n⭐사전등록 판정")
    print(f"  ①C가 VAL·OOS 양 창에서 A·B를 모두 이김: {'✅' if win else '❌'}")
    print(f"     VAL  C {C['VAL']:+.4f} vs A {A['VAL']:+.4f} / B {B['VAL']:+.4f}")
    print(f"     OOS  C {C['OOS']:+.4f} vs A {A['OOS']:+.4f} / B {B['OOS']:+.4f}")
    print(f"  ②C가 셔플 대조군을 양 창에서 이김: {'✅' if nz else '❌'}")
    print(f"     VAL  C {C['VAL']:+.4f} vs D {Dv['VAL']:+.4f} · OOS  C {C['OOS']:+.4f} vs D {Dv['OOS']:+.4f}")
    print(f"  → {'**_resolved 열 추가 채택**' if (win and nz) else '**현행 B 유지**'}")
    json.dump(res, open(OUT / "result.json", "w"), ensure_ascii=False, indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
