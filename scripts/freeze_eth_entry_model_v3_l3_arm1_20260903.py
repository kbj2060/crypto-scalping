#!/usr/bin/env python3
"""ETH 진입 모델 **v3 동결 -- L3 정직 라벨 · arm1만** (2026-09-03).

v1(HGB)·v2(TabPFN)는 **L0 오염 라벨** 위에 있었다. `trail_out`이 체결 봉 자체부터 평가해
**체결 이전 고가**(중앙 1.76 ATR, 82.3%가 ARM 초과)를 진입 후 이익으로 크레딧했고, 1분봉
해상으로 확증한 결과 전체 후보 **PF 2.86 → 0.95**였다.
전문: `docs/experiments/eth_entry_intrabar_fill_bar_credit_artifact_20260903.md`

v3에서 바뀌는 것은 셋뿐이다:
  ① **L3 라벨로 학습** -- 1분봉으로 체결 시점을 특정하고 그 이후 구간만 크레딧
  ② **arm1(신호방향)만 제출** -- 구조·모델 재시험 양쪽에서 arm1만이 양팔보다 나았다(8/10 조합).
     역방향 팔이 아티팩트의 최대 수혜자였다
  ③ 임계값을 **arm1 후보의 TRAIN 분포**에서 유도

⚠️**바꾸지 않은 것과 그 이유**: depth 3.0 / wait 6 / slots 4 / SL3.0·ARM1.0·Trail0.1 ·
유지율 0.2037. 전부 L0에서 골라졌지만, **지금 다시 고르면 이미 수십 번 소진된
VAL/OOS/HOLDOUT을 또 태운다.** 재시험에서 ARM 0.5나 SL 4.0이 약간 나아 보였으나 그 차이는
확립 불가능한 크기다. 구조 재선택은 **신선한 창을 벌고 나서** 할 일이다.

⚠️⚠️**이것은 승격 후보가 아니다.** 오늘 확인된 바로 이 전략에는 확립된 엣지가 없다 --
트리거가 무작위 봉보다 못하고(VAL +1.48 vs +2.87), 독립 일수가 42~45일뿐이라 일 단위 군집
부트스트랩 CI 하한이 음수다. v3의 목적은 **정직한 기반 위에서 전진 데이터를 모으는 것**이고,
그것이 남은 유일한 증거원이다.

동결 대상: TabPFN 컨텍스트(주) + HGB 모델(대조). 둘 다 섀도우에서 나란히 채점한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
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

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
L3D = ROOT / "tmp/eth_entry_1m_resolved_20260903/labels_1m_all.csv"
V1 = ROOT / "tmp/eth_entry_limit_fade_v1_20260903"
MODEL_ID = "eth_entry_limit_fade_v3_l3arm1_20260903"
OUT = ROOT / f"tmp/{MODEL_ID}"
DEPTH, WAIT, NSLOT, KEEP0, SUB = 3.0, 6, 4, 0.2037, 18000
LABEL_THR = 0.0040
POLICY = {
    "depth_atr": DEPTH, "wait_bars": WAIT, "slots": NSLOT,
    "arms": "signal_direction_only",              # ⭐v1/v2의 both_arms에서 변경
    "scorer": "tabpfn_classifier_5member", "label": "L3(1m-resolved) y > 40bp",
    "exit": {"sl_atr": 3.0, "arm_atr": 1.0, "trail_atr": 0.1},
    "exit_convention": "L3 -- fill bar contributes only its POST-FILL minutes (1m resolved)",
    "cost_roundtrip": 0.0010, "margin_fraction": 0.30, "leverage": 3.0,
    "cancel_if_unfilled": True,
}


def log(m): print(f"[v3] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier

    card1 = json.loads((V1 / "model_card.json").read_text())
    FEATS = card1["feature_cols"]
    LAB = pd.read_csv(L3D, parse_dates=["timestamp"])
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    A = D.merge(LAB[["timestamp", "signal", "arm", "depth", "btf", "y_L3"]],
                on=["timestamp", "signal", "arm", "depth", "btf"], how="left")
    A = A[np.isfinite(A.y_L3)].reset_index(drop=True)
    X = A[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (A.split == "TRAIN").to_numpy()
    med = X[tr].median()
    X = X.fillna(med).to_numpy(np.float32)
    y = A["y_L3"].to_numpy(float); lab = (y > LABEL_THR).astype(int)
    # ⭐학습은 전 모집단(모든 깊이·대기·팔), 평가/임계값은 arm1 후보만
    dsel = ((A.depth == DEPTH) & (A.btf <= WAIT) & (A.arm == 1)).to_numpy()
    itr = np.flatnonzero(tr); prow = np.flatnonzero(dsel)
    log(f"학습 {len(itr):,} · arm1 후보 {len(prow):,} · 피쳐 {len(FEATS)}")

    def expand(v):
        f = np.full(len(A), -np.inf); f[prow] = v; return f

    def pol(mask, score, thr):
        w = A[mask & (score > thr)]
        t = slotN(w.assign(y=y[mask & (score > thr)]), NSLOT)
        return {"n": int(len(t)), "mean_bp": round(float(np.mean(t) * 1e4), 4) if len(t) else 0.0,
                "mean_bp_exact": float(np.mean(t) * 1e4) if len(t) else 0.0,
                "pf": round(float(stat(t)[2]), 4) if len(t) else 0.0}

    W = ("TRAIN", "VAL", "OOS", "HOLDOUT")
    M = {w: dsel & (A.split == w).to_numpy() for w in W}
    nofil = {w: pol(M[w], np.zeros(len(A)), -1.0) for w in W}
    log("무필터(arm1) " + " ".join(f"{w} {nofil[w]['mean_bp']:+.2f}(n{nofil[w]['n']})" for w in W))

    # --- HGB 대조 ---
    hgb = [HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
           .fit(X[tr], y[tr]) for s in SEEDS]
    pH = expand(np.mean([m.predict(X[prow]) for m in hgb], axis=0))
    thrH = float(np.quantile(pH[M["TRAIN"]], 1 - KEEP0))
    mH = {w: pol(M[w], pH, thrH) for w in W}

    # --- TabPFN 주 모델 ---
    log(f"TabPFN 5멤버 적합 (컨텍스트 {SUB:,})...")
    ps = []
    for k, sd in enumerate(SEEDS):
        rs = np.random.default_rng(sd).choice(itr, size=min(SUB, len(itr)), replace=False)
        m = TabPFNClassifier(device="cuda", random_state=sd)
        m.fit(X[rs], lab[rs])
        ps.append(m.predict_proba(X[prow])[:, 1])
        log(f"  멤버{k} seed={sd}")
    pT = expand(np.mean(ps, axis=0))
    thrT = float(np.quantile(pT[M["TRAIN"]], 1 - KEEP0))
    mT = {w: pol(M[w], pT, thrT) for w in W}
    log(f"⭐임계값: TabPFN p_thr {thrT:.6f} · HGB τ {thrH*1e4:+.2f}bp (둘 다 TRAIN·arm1에서만 유도)")

    print(f"\n{'':22s}" + "".join(f"{w:>12s}" for w in W))
    print(f"{'무필터(arm1)':22s}" + "".join(f"{nofil[w]['mean_bp']:+12.2f}" for w in W))
    print(f"{'HGB 회귀 (대조)':22s}" + "".join(f"{mH[w]['mean_bp']:+12.2f}" for w in W))
    print(f"{'TabPFN 분류 (주)':22s}" + "".join(f"{mT[w]['mean_bp']:+12.2f}" for w in W))
    print(f"{'  체결 수 (TabPFN)':22s}" + "".join(f"{mT[w]['n']:12d}" for w in W))

    payload = {
        "model_id": MODEL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        "kind": "entry_filter_l3_arm1",
        "supersedes": ["eth_entry_limit_fade_v1_20260903 (L0 contaminated)",
                       "eth_entry_limit_fade_v2_tabpfn_20260903 (L0 contaminated)"],
        "label": "L3 = 1m-resolved: fill bar contributes only post-fill minutes",
        "seeds": list(SEEDS), "context_subsample": SUB,
        "feature_cols": FEATS, "feature_medians": {k: float(v) for k, v in med.items()},
        "signal_code_map": card1["signal_code_map"], "signal_horizons": card1["signal_horizons"],
        "policy": {**POLICY, "p_threshold": thrT, "hgb_tau": thrH},
        "train_range": card1["train_range"], "splits": card1["splits"],
        "frozen_metrics_tabpfn": mT, "frozen_metrics_hgb": mH, "no_filter_arm1": nofil,
        "context_X": X[itr], "context_y": lab[itr].astype(np.int8), "context_index": itr,
        "hgb_models": hgb,
        "NOT_A_PROMOTION": (
            "This strategy has NO established edge as of 2026-09-03. Triggers do not beat random "
            "bars (VAL +1.48 vs +2.87); day-clustered bootstrap CI on (filter - no filter) includes "
            "zero on VAL and OOS with only 42-45 independent days; a float32/float64 change flips "
            "VAL's sign. v3 exists to collect FORWARD shadow data on an honest basis -- that is the "
            "only remaining evidence source. All VAL/OOS/HOLDOUT numbers here are diagnostic."),
        "inherited_unre_selected": (
            "depth/wait/slots/exit/keep-rate are all inherited from the L0 era and were NOT "
            "re-selected, because re-selecting would burn windows that are already spent many times "
            "over. The retest suggested ARM 0.5 and SL 4.0 look slightly better but the differences "
            "are not establishable."),
        "changed_from_v2": ["L3 labels instead of L0", "arm1 (signal direction) only instead of "
                            "both arms", "threshold derived on arm1 TRAIN candidates"],
    }
    ap = OUT / "model.joblib"
    joblib.dump(payload, ap)
    meta = {k: v for k, v in payload.items()
            if k not in ("context_X", "context_y", "context_index", "hgb_models")}
    (OUT / "model_card.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    sha = hashlib.sha256(ap.read_bytes()).hexdigest()[:16]
    (OUT / "SHA256").write_text(sha + "\n")
    log(f"\n동결 저장: {ap} ({ap.stat().st_size/1e6:.1f} MB) sha256:{sha}")

    # 재적재 검증
    log("재적재 검증...")
    Q = joblib.load(ap)
    loc = {int(v): i for i, v in enumerate(Q["context_index"])}
    ps2 = []
    for sd in Q["seeds"]:
        rs = np.random.default_rng(sd).choice(Q["context_index"], size=SUB, replace=False)
        sel = np.array([loc[int(v)] for v in rs])
        m = TabPFNClassifier(device="cuda", random_state=sd)
        m.fit(Q["context_X"][sel], Q["context_y"][sel])
        ps2.append(m.predict_proba(X[prow])[:, 1])
    p2 = expand(np.mean(ps2, axis=0))
    ok = True
    for w in W:
        r = pol(M[w], p2, Q["policy"]["p_threshold"])
        d = abs(r["mean_bp_exact"] - Q["frozen_metrics_tabpfn"][w]["mean_bp_exact"])
        log(f"  {w:8s} {r['mean_bp']:+.2f} vs {Q['frozen_metrics_tabpfn'][w]['mean_bp']:+.2f} "
            f"|Δ|={d:.2e} {'✅' if d < 1e-6 else '❌'}")
        ok &= d < 1e-6
    log("✅ 재적재 검증 통과" if ok else "❌ 재적재 검증 실패")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
