#!/usr/bin/env python3
"""증거신호 칩 **인과 모집단 동결 컨텍스트** 내보내기 (2026-09-04).

`research_eth_evidence_chip_accuracy_upgrade_20260904.py --stage build`가 만든 프레임(raw 발동 + Tier0 + 신호 고유 피쳐 +
레짐 OOF one-hot + 청산맵 거리)의 **TRAIN 행**을 라이브 모듈 `_predict_proba()`가 읽는 스키마(timestamp, hit, feature cols…)로 저장한다.
어느 팔(F0/F1/F2/F3)을 쓸지는 `--arm-map '{"demarker_extreme":"F0", ...}'`(기본: 전 신호 F0). 지정 안 된 신호는 내보내지 않는다.
⚠️TRAIN(<2025-09-01)만. VAL/OOS/HOLDOUT 행은 컨텍스트에 절대 넣지 않는다(라이브 확률이 검증 구간을 학습하면 검증이 무효가 된다).
TabPFN 기본 한도(10,000행)를 넘는 신호는 없다(최대 kalman 9,668).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import importlib.util
_s = importlib.util.spec_from_file_location("chipacc", ROOT / "scripts/research_eth_evidence_chip_accuracy_upgrade_20260904.py")
CA = importlib.util.module_from_spec(_s); _s.loader.exec_module(CA)
OUT = ROOT / "data/labels/eth_5m_evidence_chip_causal_20260904"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm-map", default=None)
    ap.add_argument("--population", choices=["all", "live"], default="all",
                    help="all=TRAIN raw 발동 전부 / live=라이브 캐시가 새로 추론하는 봉만(같은 측면 raw 발동이 직전 horizon_bars 안에 없는 발동)")
    a = ap.parse_args()
    arm_map = json.loads(a.arm_map) if a.arm_map else {s: "F0" for s in CA.SIGNALS}
    OUT.mkdir(parents=True, exist_ok=True); manifest = {}
    # horizon_bars는 라이브 METALABEL_SIGNALS와 같은 값이 인과 모집단 config.json에 있다(라이브 모듈 import는 torch 체인을 타서 로컬 불가).
    HZ = {k: int(v["horizon"]) for k, v in json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())["cfg"].items()}
    for s, arm in arm_map.items():
        d = pd.read_parquet(CA.OUT / "frames" / f"{s}.parquet")
        if a.population == "live":
            H = HZ[s]; m = np.zeros(len(d), bool)
            for sd in (0, 1):
                idx = np.flatnonzero(d["is_bottom"].to_numpy() == sd); pos = d["pos"].to_numpy()[idx]; last = -10**9
                for j, p_ in enumerate(pos):
                    if p_ - last >= H:
                        m[idx[j]] = True
                    last = p_
            d = d.loc[m]
        T0 = CA.tier0_cols(d, s) + ["is_bottom"]
        feats = {"F0": T0, "F1": T0 + CA.REG_COLS, "F2": T0 + CA.LIQ_COLS, "F3": T0 + CA.REG_COLS + CA.LIQ_COLS}[arm]
        tr = d.loc[d["split"] == "TRAIN", ["pos", "timestamp", "side", "hit", "move_atr_mult", "split"] + feats].copy()
        tr = tr.dropna(subset=feats)
        assert len(tr) <= 10000 and tr["timestamp"].max() < pd.Timestamp("2025-09-01"), (s, len(tr), tr["timestamp"].max())
        path = OUT / f"{s}_train_context_causal_{arm}_{a.population}_20260904.csv"; tr.to_csv(path, index=False)
        manifest[s] = {"arm": arm, "population": a.population, "path": str(path.relative_to(ROOT)), "rows": int(len(tr)), "hit_rate": round(float(tr["hit"].mean()), 4),
                       "feature_columns": feats, "timestamp_max": str(tr["timestamp"].max())}
        print(f"{s:26s} {arm} rows {len(tr):5d} hit {tr['hit'].mean():.3f} feats {len(feats)} -> {path.name}")
    (OUT / f"manifest_{a.population}.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
