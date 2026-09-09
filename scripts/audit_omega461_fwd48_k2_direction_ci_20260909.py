"""K2 판정 — 전향 레짐 라벨(fwd48)의 전방수익 bull−bear 격차 CI.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
사전등록: `scripts/build_omega461_forward_regime_label_20260909.py` docstring 의 K2.
  "전방수익 bull−bear 격차가 7일 블록 부트스트랩에서 CI 가 0 을 배제해야 한다."
기존 라벨(wide24 HMM / balgbm / balnobb / s12k3)은 이 검정에서 전부 실패했다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "fwd48(전향)": (ROOT / "data/ensemble/supervised/omega461_fwd48_cut2509_20260909", "regime3_fwd48_cut2509_"),
    "balnobb(후행)": (ROOT / "data/ensemble/supervised/omega461_balnobb_cut2509_20260909", "regime3_balnobb_cut2509_"),
}
CLASSES = ("bull", "bear", "chop")
SPLITS = {"validation": ("2025-10-01", "2025-12-31 23:55:00"), "oos": ("2026-01-01", "2026-02-28 23:55:00")}
HORIZONS, BLOCK, NBOOT = (12, 48, 288), 2016, 2000
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"

def load():
    b = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
                    usecols=["timestamp", "close"], parse_dates=["timestamp"])
    b2 = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2025.csv",
                     usecols=["timestamp", "close"], parse_dates=["timestamp"])
    b = pd.concat([b2, b], ignore_index=True)
    for _, (d, pref) in ARMS.items():
        parts = [pd.read_csv(d / f"training_features_{t}_{pref}sidecar.csv", parse_dates=["timestamp"],
                             usecols=["timestamp"] + [f"{pref}{c}_prob" for c in CLASSES])
                 for t in ("2025", "2026_rebuilt")]
        b = b.merge(pd.concat(parts, ignore_index=True), on="timestamp", how="inner")
    return b.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

def spread(pred, fwd):
    mb, mr = pred == 0, pred == 1
    if mb.sum() < 30 or mr.sum() < 30: return np.nan
    return float((np.mean(fwd[mb]) - np.mean(fwd[mr])) * 1e4)

def main() -> int:
    df, rng = load(), np.random.default_rng(20260909)
    rep = {}
    for split, (s, e) in SPLITS.items():
        m = ((df.timestamp >= s) & (df.timestamp <= e)).to_numpy()
        d = df[m].reset_index(drop=True)
        close = pd.to_numeric(d["close"], errors="raise").to_numpy(np.float64)
        preds = {n: d[[f"{p}{c}_prob" for c in CLASSES]].to_numpy(np.float64).argmax(1)
                 for n, (_, p) in ARMS.items()}
        blocks = np.arange(len(d)) // BLOCK; uniq = np.unique(blocks)
        print(f"\n{'='*80}\n[{split}] {len(d):,}봉  독립 블록 {len(uniq)}개", flush=True)
        rep[split] = {}
        for h in HORIZONS:
            fwd = np.full(len(close), np.nan); fwd[:-h] = (close[h:] - close[:-h]) / close[:-h]
            ok = np.isfinite(fwd)
            pt = {n: spread(p[ok], fwd[ok]) for n, p in preds.items()}
            draws = {n: [] for n in preds}
            for _ in range(NBOOT):
                take = rng.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([np.flatnonzero(blocks == b) for b in take]); idx = idx[ok[idx]]
                if len(idx) < 200: continue
                for n, p in preds.items():
                    v = spread(p[idx], fwd[idx])
                    if np.isfinite(v): draws[n].append(v)
            print(f"  h{h} ({h*5//60}h)", flush=True)
            rep[split][f"h{h}"] = {}
            for n in preds:
                a = np.array(draws[n]); lo, hi = np.quantile(a, [0.025, 0.975])
                ex = bool(lo > 0 or hi < 0)
                rep[split][f"h{h}"][n] = {"point": round(pt[n], 2), "lo": round(float(lo), 2),
                                          "hi": round(float(hi), 2), "excludes_zero": ex}
                print(f"    {n:14s} {pt[n]:+8.1f}bp  CI[{lo:+8.1f},{hi:+8.1f}] "
                      f"{'✅ K2 통과' if ex else '❌ K2 실패'}", flush=True)
    (OUT / "fwd48_k2_direction_ci.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n산출물: {OUT}/fwd48_k2_direction_ci.json", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
