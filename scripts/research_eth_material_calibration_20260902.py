#!/usr/bin/env python3
"""증거신호 재료 텐서의 확률 보정 측정 (2026-09-02, 재료화 3단계).

보정(calibration)은 값의 범위가 아니라 **의미**에 관한 것이다 -- "모델이 0.7이라고 할 때 실제로
70%가 맞는가". AUC는 순위만 재므로, 순위를 잘 매기면서 절대값이 틀릴 수 있다.

재는 것:
  reliability diagram (분위 10구간: 구간별 평균 예측값 vs 실제 적중률)
  ECE  기대보정오차 = sum (n_b/N) * |pred_b - obs_b|      (0에 가까울수록 좋음)
  MCE  최대보정오차 = max |pred_b - obs_b|
  Brier = mean((p-y)^2),  BSS = 1 - Brier/Brier(기저율)   (0보다 커야 기저율보다 나음)
  예측 분포 폭 (모델이 확률 범위를 실제로 쓰는가)
  기저율 드리프트 (TRAIN 대비 VAL/OOS 실제 양성률 변화 -- 모델 잘못이 아닌 보정 오차의 원인)

⚠️HOLDOUT은 건드리지 않는다.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OUT_DIR = ROOT / "tmp/eth_material_calibration_20260902"
NBIN = 10


def log(m): print(f"[calib] {m}", flush=True)


def curve(p, y, nbin=NBIN):
    """분위 구간 기반 reliability curve. 반환 (pred_mean, obs_rate, counts)."""
    q = np.unique(np.quantile(p, np.linspace(0, 1, nbin + 1)))
    if len(q) < 3:
        return np.array([p.mean()]), np.array([y.mean()]), np.array([len(p)])
    b = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
    pm, om, cn = [], [], []
    for k in range(len(q) - 1):
        m = b == k
        if m.sum() == 0: continue
        pm.append(p[m].mean()); om.append(y[m].mean()); cn.append(int(m.sum()))
    return np.array(pm), np.array(om), np.array(cn)


def metrics(p, y):
    pm, om, cn = curve(p, y)
    w = cn / cn.sum()
    ece = float((w * np.abs(pm - om)).sum())
    mce = float(np.abs(pm - om).max())
    br = float(np.mean((p - y) ** 2))
    base = float(y.mean())
    br0 = float(np.mean((base - y) ** 2))
    return {"ece": round(ece, 4), "mce": round(mce, 4), "brier": round(br, 4),
            "brier_base": round(br0, 4), "bss": round(1 - br / br0, 4) if br0 > 0 else 0.0,
            "base_rate": round(base, 4), "pred_mean": round(float(p.mean()), 4),
            "pred_std": round(float(p.std()), 4),
            "pred_p05": round(float(np.quantile(p, 0.05)), 4),
            "pred_p95": round(float(np.quantile(p, 0.95)), 4),
            "bias": round(float(p.mean() - base), 4)}


def main() -> int:
    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    names = list(cfg)
    rows, curves = [], {}
    for name in names:
        f = SRC / f"{name}_causal_proba.csv"
        if not f.exists(): continue
        d = pd.read_csv(f)
        tr = d[d.split == "TRAIN"]
        for wn in ("VAL", "OOS"):
            w = d[d.split == wn]
            if len(w) < 100: continue
            p, y = w["proba"].to_numpy(float), w["hit"].to_numpy(int)
            m = metrics(p, y)
            m.update({"signal": name, "window": wn, "n": len(w),
                      "train_base": round(float(tr["hit"].mean()), 4)})
            m["base_drift"] = round(m["base_rate"] - m["train_base"], 4)
            rows.append(m)
            curves[(name, wn)] = curve(p, y)
    r = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT_DIR / "calibration_metrics.csv", index=False)

    pd.set_option("display.width", 250)
    log("\n=== 보정 지표 ===")
    print(r[["signal", "window", "n", "train_base", "base_rate", "base_drift",
             "pred_mean", "bias", "pred_std", "pred_p05", "pred_p95",
             "ece", "mce", "brier", "bss"]].to_string(index=False))

    log("\n=== 해석 요약 ===")
    for name in names:
        sub = r[r.signal == name]
        if sub.empty: continue
        v = sub[sub.window == "VAL"].iloc[0]; o = sub[sub.window == "OOS"].iloc[0]
        verdict = []
        if max(abs(v.bias), abs(o.bias)) > 0.05: verdict.append(f"편향 {v.bias:+.3f}/{o.bias:+.3f}")
        if max(v.ece, o.ece) > 0.05: verdict.append(f"ECE 큼 {v.ece:.3f}/{o.ece:.3f}")
        if max(v.pred_std, o.pred_std) < 0.08: verdict.append(f"폭 좁음 std {v.pred_std:.3f}/{o.pred_std:.3f}")
        if min(v.bss, o.bss) <= 0: verdict.append(f"BSS<=0 {v.bss:+.3f}/{o.bss:+.3f}")
        if max(abs(v.base_drift), abs(o.base_drift)) > 0.05:
            verdict.append(f"기저율드리프트 {v.base_drift:+.3f}/{o.base_drift:+.3f}")
        log(f"  {name:26s} {'· '.join(verdict) if verdict else '양호'}")

    # ---- 차트 ----
    fig, axes = plt.subplots(2, 4, figsize=(30, 15))
    for ax, name in zip(axes.ravel(), names):
        ax.plot([0, 1], [0, 1], "k--", lw=2, alpha=0.5, label="perfect")
        for wn, col in (("VAL", "#1f77b4"), ("OOS", "#d62728")):
            if (name, wn) not in curves: continue
            pm, om, cn = curves[(name, wn)]
            ax.plot(pm, om, "o-", color=col, lw=3, ms=11, label=wn)
            sub = r[(r.signal == name) & (r.window == wn)]
            if len(sub):
                b = float(sub.iloc[0]["base_rate"])
                ax.axhline(b, color=col, ls=":", lw=2, alpha=0.55)
        ax.set_title(name.replace("_", " "), fontsize=21)
        ax.set_xlabel("predicted probability", fontsize=18)
        ax.set_ylabel("observed hit rate", fontsize=18)
        ax.tick_params(labelsize=16)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        ax.legend(fontsize=16, loc="upper left")
    fig.suptitle("Causal-population metalabel calibration (dotted = base rate)", fontsize=28)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "reliability_diagrams.png", dpi=145)
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
