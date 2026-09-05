#!/usr/bin/env python3
"""나이 인지 확률 모델 — **신호별 캘리브레이션 감사** (2026-09-06).

모델카드는 나이별 캘리브레이션만 담았다(격차 −0.002~−0.018). 그런데 이 모델은 8종 **공용 단일 모델**이고,
신호마다 기저율이 크게 다르다(demarker는 K=0.70이라 hit률 ≈0.90, 나머지는 0.2~0.5).
라이브 실측에서 demarker가 발동 봉 0.921 → 나이 인지 0.705로 벌어졌다 — 신호 원핫이 그 차이를 다 흡수
못했을 수 있다. **집계로는 맞는데 신호별로 어긋나는** 상태를 여기서 잡는다.

신호 × 창별로: n · 실제 도달률 · 예측 평균 · 격차 · AUC. 격차 |Δ| > 0.05면 경고.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ART = ROOT / "data/models/eth_chip_age_aware_proba_20260906"
OUT = ROOT / "data/research/eth_chip_age_aware_calibration_audit_20260906"


def log(m): print(f"[calib] {m}", flush=True)


def main() -> int:
    import joblib
    from sklearn.metrics import roc_auc_score
    import train_eth_chip_age_aware_proba_20260906 as T
    from live_evidence_signal_dashboard_20260823 import compute_signals
    from live_evidence_signal_metalabel_20260829 import METALABEL_SIGNALS
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import FEATURE_COLUMNS, build_indicator_frame

    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    b = joblib.load(ART / "model.joblib")
    SIGNALS = b["signals"]; H = b["horizon"]; K = b["k"]
    kl, btc = T.load_kl(T.KL_ETH), T.load_kl(T.KL_BTC)
    sig = compute_signals(kl.copy(), btc_df=btc)
    frame = build_indicator_frame(kl.copy())
    base_cols = [c for c in FEATURE_COLUMNS if c != "is_bottom"]
    XB = frame[base_cols].to_numpy(float)
    n = len(kl)
    h, l, c = (kl[x].to_numpy(float) for x in ("high", "low", "close"))
    prev = np.r_[np.nan, c[:-1]]
    trr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    ap = (pd.Series(trr).rolling(14, min_periods=14).mean().to_numpy()) / c
    ts = kl["timestamp"].to_numpy()

    rows = []
    for si, s in enumerate(SIGNALS):
        H_, k_ = H[s], K[s]
        for side, ib in (("bottom", 1), ("top", 0)):
            col = f"{side}_{s}"
            if col not in sig.columns:
                continue
            ff = T.main.__globals__ and None
            keep = np.zeros(n, bool); last = -10 ** 9
            for j in np.flatnonzero(sig[col].fillna(False).to_numpy(bool)):
                if j - last > T.GAP:
                    keep[j] = True
                last = j
            for i in np.flatnonzero(keep):
                if not np.isfinite(ap[i]) or i + H_ >= n:
                    continue
                lvl = c[i] * (1 + k_ * ap[i]) if side == "bottom" else c[i] * (1 - k_ * ap[i])
                hit = (h[i + 1:i + H_ + 1] >= lvl) if side == "bottom" else (l[i + 1:i + H_ + 1] <= lvl)
                to = int(np.argmax(hit)) + 1 if hit.any() else 10 ** 6
                for a in range(H_):
                    if to <= a:
                        break
                    rows.append((i + a, a, H_ - a, ib, si, 1 if to <= H_ else 0, ts[i], s))
    R = pd.DataFrame(rows, columns=["bar", "age", "left", "is_bottom", "si", "y", "fire_ts", "signal"])
    R["split"] = "NONE"; tsi = pd.DatetimeIndex(R["fire_ts"])
    for w, (a, bnd) in T.SPLITS.items():
        R.loc[(tsi >= pd.Timestamp(a)) & (tsi < pd.Timestamp(bnd)), "split"] = w
    R = R.loc[R["split"] != "NONE"].reset_index(drop=True)
    oh = np.zeros((len(R), len(SIGNALS))); oh[np.arange(len(R)), R["si"].to_numpy()] = 1.0
    X = np.hstack([XB[R["bar"].to_numpy()], R["is_bottom"].to_numpy().reshape(-1, 1).astype(float), oh,
                   R[["age", "left"]].to_numpy(float)])
    p = np.mean([m.predict_proba(X)[:, 1] for m in b["models"]], axis=0)
    y = R["y"].to_numpy()
    log(f"행 {len(R):,} ({time.time()-t0:.0f}s)")
    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "per_signal": {}, "warnings": []}
    print(f"{'신호':26s} {'창':6} {'n':>7} {'실제':>7} {'예측':>7} {'격차':>7} {'AUC':>7}")
    for s in SIGNALS:
        rep["per_signal"][s] = {}
        for w in ("TRAIN", "VAL", "OOS"):
            m = (R["signal"] == s).to_numpy() & (R["split"] == w).to_numpy()
            if m.sum() < 200 or len(np.unique(y[m])) < 2:
                continue
            gap = float(p[m].mean() - y[m].mean())
            d = {"n": int(m.sum()), "actual": round(float(y[m].mean()), 3), "pred": round(float(p[m].mean()), 3),
                 "gap": round(gap, 3), "auc": round(float(roc_auc_score(y[m], p[m])), 4)}
            rep["per_signal"][s][w] = d
            flag = " ⚠️" if abs(gap) > 0.05 and w in ("VAL", "OOS") else ""
            if flag:
                rep["warnings"].append(f"{s}/{w} gap {gap:+.3f}")
            print(f"{s:26s} {w:6s} {d['n']:>7} {d['actual']:>7.3f} {d['pred']:>7.3f} {gap:>+7.3f} {d['auc']:>7.4f}{flag}")
    # ── 신호 × 나이 교차 (2026-09-06 추가): 위 표는 나이를 합산해서 "나이 0에서 특정 신호가 어긋나는"
    # 경우를 못 잡는다. 사용자가 라이브에서 demarker 0.921 -> 0.705를 보고 물었다. 그 자리를 직접 잰다.
    print()
    print(f"{'신호':26s} {'나이':>4} " + " ".join(f"{w+' n/실제/예측/격차':>26}" for w in ("TRAIN", "VAL", "OOS")))
    rep["per_signal_by_age"] = {}
    for s in SIGNALS:
        rep["per_signal_by_age"][s] = {}
        for a in (0, 2, 4):
            if a >= H[s]:
                continue
            line = f"{s:26s} {a:>4} "; cell = {}
            for w in ("TRAIN", "VAL", "OOS"):
                m = (R["signal"] == s).to_numpy() & (R["split"] == w).to_numpy() & (R["age"].to_numpy() == a)
                if m.sum() < 80 or len(np.unique(y[m])) < 2:
                    line += " " * 27; continue
                g = float(p[m].mean() - y[m].mean())
                cell[w] = {"n": int(m.sum()), "actual": round(float(y[m].mean()), 3), "pred": round(float(p[m].mean()), 3), "gap": round(g, 3)}
                line += f"{m.sum():>6}/{y[m].mean():>5.3f}/{p[m].mean():>5.3f}/{g:>+6.3f}   "
            rep["per_signal_by_age"][s][f"a{a}"] = cell
            print(line)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    log(f"완료 · |격차|>0.05 경고 {len(rep['warnings'])}건: {rep['warnings']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
