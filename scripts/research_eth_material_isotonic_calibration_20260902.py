#!/usr/bin/env python3
"""재료 텐서 확률 보정 적용 -- isotonic / Platt (2026-09-02, 재료화 3단계-b).

측정(`research_eth_material_calibration_20260902.py`) 결과: 대부분 단조 상승하지만 대각선보다
납작하다(예측 폭 > 실제 폭). BSS는 전부 0 근처라 확률로서의 정보는 얇지만, **보정을 하면 8개
신호가 같은 척도가 된다** -- 보정 후에는 어느 신호든 0.7이 똑같이 70%를 뜻하므로 기저율 차이가
숫자 안에 올바르게 반영된다. 기저율 나눗셈보다 원리적으로 나은 해법이다.

절차
----
  1. VAL에서 isotonic과 Platt(로지스틱) 둘 다 적합. **VAL 내부 5-fold CV로 승자 선택** --
     OOS를 선택에 쓰지 않기 위해서다.
  2. 승자를 VAL 전체로 재적합해 OOS에 적용, **정직한 검증**:
       ECE/BSS가 실제로 개선되는가 (VAL 과적합이 아닌가)
       AUC는 불변인가 (단조 변환이므로 그래야 정상 -- 아니면 버그)
  3. 전 구간(HOLDOUT 포함)에 매핑을 적용해 `<sig>_proba_cal`로 저장. 원본은 남긴다.

⚠️보정 매핑은 VAL에서만 적합한다. OOS는 검증 전용이고 선택에 쓰지 않는다.
⚠️HOLDOUT은 매핑 적용만 하고(추론) 성능을 재지 않는다.
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
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from research_eth_material_calibration_20260902 import metrics  # noqa: E402

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OUT_DIR = ROOT / "tmp/eth_material_calibration_20260902"
SEED = 7529


def log(m): print(f"[iso] {m}", flush=True)


def fit_iso(p, y):
    m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    return lambda x: np.clip(m.predict(x), 1e-6, 1 - 1e-6)


def fit_platt(p, y):
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, solver="lbfgs").fit(z, y)
    def f(x):
        zz = np.log(np.clip(x, eps, 1 - eps) / (1 - np.clip(x, eps, 1 - eps))).reshape(-1, 1)
        return np.clip(m.predict_proba(zz)[:, 1], 1e-6, 1 - 1e-6)
    return f


def cv_ece(p, y, fitter, k=5):
    """VAL 내부 CV로 일반화 ECE 추정 -- OOS를 선택에 쓰지 않기 위함."""
    if len(np.unique(y)) < 2 or len(y) < 5 * k:
        return np.inf
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    out = []
    for tr, te in skf.split(p.reshape(-1, 1), y):
        try:
            f = fitter(p[tr], y[tr])
            out.append(metrics(f(p[te]), y[te])["ece"])
        except Exception:
            return np.inf
    return float(np.mean(out))


def main() -> int:
    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    rows = []
    for name in cfg:
        f = SRC / f"{name}_causal_proba.csv"
        if not f.exists(): continue
        d = pd.read_csv(f)
        v = d.split == "VAL"; o = d.split == "OOS"
        pv, yv = d.loc[v, "proba"].to_numpy(float), d.loc[v, "hit"].to_numpy(int)
        po, yo = d.loc[o, "proba"].to_numpy(float), d.loc[o, "hit"].to_numpy(int)
        if len(pv) < 100 or len(po) < 100: continue

        e_iso, e_pl = cv_ece(pv, yv, fit_iso), cv_ece(pv, yv, fit_platt)
        win = "isotonic" if e_iso <= e_pl else "platt"
        fitter = fit_iso if win == "isotonic" else fit_platt
        g = fitter(pv, yv)
        d["proba_cal"] = g(d["proba"].to_numpy(float))

        before_o, after_o = metrics(po, yo), metrics(g(po), yo)
        before_v, after_v = metrics(pv, yv), metrics(g(pv), yv)
        auc_b = roc_auc_score(yo, po); auc_a = roc_auc_score(yo, g(po))
        rec = {"signal": name, "winner": win, "cv_ece_iso": round(e_iso, 4),
               "cv_ece_platt": round(e_pl, 4),
               "VAL_ece_before": before_v["ece"], "VAL_ece_after": after_v["ece"],
               "OOS_ece_before": before_o["ece"], "OOS_ece_after": after_o["ece"],
               "OOS_bss_before": before_o["bss"], "OOS_bss_after": after_o["bss"],
               "OOS_bias_before": before_o["bias"], "OOS_bias_after": after_o["bias"],
               "OOS_auc_before": round(float(auc_b), 4), "OOS_auc_after": round(float(auc_a), 4),
               "auc_delta": round(float(auc_a - auc_b), 5),
               "ece_improved": bool(after_o["ece"] < before_o["ece"]),
               "bss_improved": bool(after_o["bss"] > before_o["bss"])}
        rows.append(rec)
        d.to_csv(SRC / f"{name}_causal_proba_cal.csv", index=False)
        log(f"{name:26s} {win:8s} | OOS ECE {before_o['ece']:.4f}->{after_o['ece']:.4f} "
            f"BSS {before_o['bss']:+.4f}->{after_o['bss']:+.4f} "
            f"AUC {auc_b:.4f}->{auc_a:.4f} (Δ{auc_a-auc_b:+.5f})")

    r = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT_DIR / "isotonic_results.csv", index=False)
    pd.set_option("display.width", 250)
    log("\n=== 결과 ===")
    print(r[["signal", "winner", "OOS_ece_before", "OOS_ece_after", "OOS_bss_before",
             "OOS_bss_after", "OOS_auc_before", "OOS_auc_after", "auc_delta",
             "ece_improved", "bss_improved"]].to_string(index=False))
    log(f"\nOOS ECE 개선: {int(r.ece_improved.sum())}/{len(r)} | "
        f"OOS BSS 개선: {int(r.bss_improved.sum())}/{len(r)} | "
        f"AUC 최대 변화 {r.auc_delta.abs().max():+.5f} (0에 가까워야 정상)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
