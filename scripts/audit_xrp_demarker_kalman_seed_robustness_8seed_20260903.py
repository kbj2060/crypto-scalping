#!/usr/bin/env python3
"""XRP `demarker_extreme` / `kalman_deviation_meanrev` **8시드 견고성** -- N>=5 요건 충족.

## 왜

두 신호는 4시드(`[20260829, 141592, 271828, 577215]`)로 평가됐다. 시드는 **진짜 다양하고**
(고정 간격 증가 아님) per-seed AUC도 리포트에 있으며 **부호 일관성도 만족**했지만,
CLAUDE.md Seed-Diversity 게이트의 **N>=5** 요건에 하나 모자란다.

나머지 3종(str_z/taker/orthogonal)은 같은 날 8시드로 채웠으므로, 여기서 두 신호도 맞춘다.

## 설계

발동/라벨 빌드는 결정론적이므로 신호당 1회만 만들고 **TabPFN만 시드별 재적합**한다.
각 신호의 **원본 모듈**(`build_final_fires` / `build_fires_and_features`)을 그대로 import한다.

⚠️**VAL + OOS만** 본다. HOLDOUT은 이미 1회 소진됐고(demarker 0.6759 / kalman 0.6223)
시드별 재평가는 재노출이다.

## 판정 (실행 전 고정)

  모든 시드에서 OOS AUC > 0.5 (부호 일치).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CAND_CSV = (ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903"
            / "xrp_5m_evidence_signal_candidates_tier0.csv")
EXPECTED_ROWS = 272_490
OUT = ROOT / "data/research/xrp_demarker_kalman_seed_robustness_8seed_20260903.json"

# 3종 감사와 **같은** 8시드 (랜덤 추출, 고정 간격 증가 아님)
SEEDS = [20260903, 811453, 30011, 947, 260317, 5387291, 68041, 1299709]

SPECS = {
    "demarker_extreme": {
        "script": "research_xrp_demarker_extreme_metalabel_tabpfn_20260903.py",
        "kind": "demarker", "h": 2, "k": 1.5,
        "recorded_4seed": {"VAL": 0.7018, "OOS": 0.6859}},
    "kalman_deviation_meanrev": {
        "script": "research_xrp_kalman_deviation_meanrev_metalabel_tabpfn_20260903.py",
        "kind": "kalman", "h": 5, "k": 2.0,
        "recorded_4seed": {"VAL": 0.6894, "OOS": 0.6103}},
}


def log(m): print(f"[dk-seed] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier

    rep = {"asset": "XRPUSDT", "seeds": SEEDS, "n_seeds": len(SEEDS),
           "seed_selection": "랜덤 추출(고정 간격 증가 아님)", "holdout_touched": False,
           "note": "VAL+OOS만 평가 -- HOLDOUT은 1회 소진(demarker 0.6759 / kalman 0.6223)",
           "signals": {}}

    for name, spec in SPECS.items():
        sp = importlib.util.spec_from_file_location(f"m_{name}", ROOT / "scripts" / spec["script"])
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)

        f = pd.read_csv(CAND_CSV)
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True).dt.tz_localize(None)
        f = mod.add_missing_features(f)
        if abs(len(f) - EXPECTED_ROWS) > 200:
            raise RuntimeError(f"{name}: 행수 {len(f):,} != XRP 기대치 {EXPECTED_ROWS:,}")

        if spec["kind"] == "demarker":
            out = mod.build_final_fires(f, spec["h"], spec["k"], mod.CLUSTER_GAP)
        else:
            f["kalman_dev_z"] = mod.compute_kalman_dev_z(f["close"].to_numpy())
            bt = (f["kalman_dev_z"] <= -2.0).fillna(False).to_numpy()
            tt = (f["kalman_dev_z"] >= 2.0).fillna(False).to_numpy()
            out = mod.build_fires_and_features(f, bt, tt, spec["h"], spec["k"], mod.CLUSTER_GAP)
        fires = out[0] if isinstance(out, tuple) else out
        ts = pd.to_datetime(fires["timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        fires = fires.assign(timestamp=ts)
        feats = [c for c in mod.FEATURE_COLUMNS if c in fires.columns]

        tr = fires.loc[fires["timestamp"] < mod.VAL_START].reset_index(drop=True)
        splits = {"VAL": fires.loc[(fires["timestamp"] >= mod.VAL_START)
                                   & (fires["timestamp"] < mod.OOS_START)].reset_index(drop=True),
                  "OOS": fires.loc[(fires["timestamp"] >= mod.OOS_START)
                                   & (fires["timestamp"] < mod.HOLDOUT_START)].reset_index(drop=True)}
        log("")
        log(f"=== {name}  (touch H={spec['h']} K={spec['k']} GAP={mod.CLUSTER_GAP}) ===")
        log(f"  TRAIN {len(tr):,} (hit {tr['hit'].mean():.4f}) | "
            f"VAL {len(splits['VAL'])} / OOS {len(splits['OOS'])}")
        per = {"VAL": [], "OOS": []}
        rows = []
        for sd in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=sd)
            clf.fit(tr[feats], tr["hit"].to_numpy().astype(int))
            row = {"seed": sd}
            for s_, ev in splits.items():
                p = clf.predict_proba(ev[feats])[:, 1]
                a = float(roc_auc_score(ev["hit"].astype(int), p))
                row[s_] = round(a, 4); per[s_].append(a)
            rows.append(row)
            log(f"  seed={sd:<9} VAL {row['VAL']}  OOS {row['OOS']}")
        res = {"horizon": spec["h"], "k": spec["k"], "n_train": int(len(tr)), "per_seed": rows}
        for s_ in ("VAL", "OOS"):
            v = np.array(per[s_])
            res[s_] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                       "min": float(v.min()), "max": float(v.max()),
                       "all_above_half": bool((v > 0.5).all())}
            log(f"  {s_} 평균 {v.mean():.4f} ± {v.std(ddof=1):.4f}  "
                f"[{v.min():.4f}, {v.max():.4f}]  전부>0.5: {'✅' if (v > 0.5).all() else '❌'}")
        for s_ in ("VAL", "OOS"):
            r4 = spec["recorded_4seed"][s_]
            log(f"  기록 4시드 {s_} {r4:.4f}  vs  8시드 평균 {res[s_]['mean']:.4f}  "
                f"(차 {res[s_]['mean'] - r4:+.4f})")
        res["recorded_4seed"] = spec["recorded_4seed"]
        rep["signals"][name] = res

    log("")
    log("=" * 70)
    log("판정 (사전 고정: 모든 시드에서 OOS AUC > 0.5)")
    log("=" * 70)
    allok = True
    for n, v in rep["signals"].items():
        ok = v["OOS"]["all_above_half"]; allok &= ok
        log(f"  {n:<26} OOS {v['OOS']['mean']:.4f} ± {v['OOS']['std']:.4f}  "
            f"[{v['OOS']['min']:.4f}, {v['OOS']['max']:.4f}]  {'✅' if ok else '❌'}")
    log("")
    log(f"⇒ {'✅**부호 일관성 통과** (N=8, 랜덤 추출)' if allok else '⚠️**일부 시드에서 0.5 이하**'}")
    rep["all_oos_above_half"] = bool(allok)
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
