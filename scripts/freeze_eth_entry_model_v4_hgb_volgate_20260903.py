#!/usr/bin/env python3
"""ETH 진입 v4 동결 -- **HGB + 변동성 게이트, 경량** (2026-09-03).

v3(TabPFN 5멤버 GPU 상주)를 대체한다. 두 가지 이유다:

  ① **성능 근거가 없다** -- 오늘 실측에서 `게이트만`과 `게이트+모델`의 선택이 **완전히
     동일**했다(VAL n=82, OOS n=64, HOLDOUT n=26 전부 일치). 고변동성 후보를 모델이 전부
     통과시키므로 게이트 ⊆ 모델이고, 그 부분집합이 전체보다 낫다(모델의 나머지가 희석).
     161피쳐 모델이 하는 일을 1파라미터 규칙이 한다.
  ② **운영 비용** -- TabPFN은 한 행 채점 68.7초에 GPU 5멤버 상주다. 2026-09-03 서버(WSL2)가
     굳어 재부팅됐고 그 부하가 기여했을 개연성이 있다. HGB는 CPU 마이크로초다.

## ⭐배치 시점에 필터를 확정하지 않는다

`게이트만 = 게이트+모델`이므로 배치 시점 필터링에 이득이 없다. 대신 **arm1 후보를 전부
제출하고 점수·게이트 플래그를 기록**하면 어떤 필터든 원장에서 사후 평가할 수 있다.
증거 축적이 **약 7배** 빨라진다 -- 체결 1.34건/일 → 약 9.2건/일, 200건까지 150일 → 약 22일.
전진 데이터가 유일한 증거원인 상황에서 이 속도 차이가 결정적이다.

기록 항목: `pred_hgb`(L3 학습) · `atr_pct` · `vol_pct`(TRAIN 분포 백분위) · `gate_p90` 플래그.
슬롯 정책도 사후 적용한다(원장에 `fi`/`ei`가 있으므로 재구성 가능).

⚠️**승격 후보가 아니다.** 이 전략에는 확립된 엣지가 없다 -- 일 단위 군집 부트스트랩에서
게이트도 VAL 3/3 실패했고 독립 일수가 8~22일뿐이며 p90→p95에서 VAL 부호가 뒤집힌다.
목적은 오직 **오염되지 않은 전진 눈금 확보**다.
"""
from __future__ import annotations

import hashlib
import json
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
MODEL_ID = "eth_entry_limit_fade_v4_hgb_volgate_20260903"
OUT = ROOT / f"tmp/{MODEL_ID}"
DEPTH, WAIT, NSLOT, KEEP0, VOL_CUT = 3.0, 6, 4, 0.2037, 90
POLICY = {
    "depth_atr": DEPTH, "wait_bars": WAIT, "slots_for_accounting": NSLOT,
    "arms": "signal_direction_only",
    "placement": "ALL arm1 candidates -- no filter at placement; filters evaluated offline",
    "scorer": "HistGradientBoostingRegressor squared_error x5 (L3 labels)",
    "vol_gate": f"atr_pct >= TRAIN p{VOL_CUT}",
    "exit": {"sl_atr": 3.0, "arm_atr": 1.0, "trail_atr": 0.1},
    "exit_convention": "L3 -- fill bar contributes only its POST-FILL minutes (1m resolved)",
    "cost_roundtrip": 0.0010, "margin_fraction": 0.30, "leverage": 3.0,
    "cancel_if_unfilled": True,
}


def log(m): print(f"[v4] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
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
    X = X.fillna(med)
    y = A["y_L3"].to_numpy(float)
    dsel = ((A.depth == DEPTH) & (A.btf <= WAIT) & (A.arm == 1)).to_numpy()
    log(f"학습 {int(tr.sum()):,} · arm1 후보 {int(dsel.sum()):,} · 피쳐 {len(FEATS)}")

    hgb = [HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
           .fit(X[tr], y[tr]) for s in SEEDS]
    p = np.mean([m.predict(X) for m in hgb], axis=0)
    tau = float(np.quantile(p[tr & dsel], 1 - KEEP0))
    a = pd.to_numeric(A["atr_pct"], errors="coerce").to_numpy(float)
    vthr = float(np.nanpercentile(a[tr & dsel], VOL_CUT))
    gate = a >= vthr
    log(f"⭐임계값(TRAIN·arm1에서만): HGB τ {tau*1e4:+.2f}bp · 변동성 p{VOL_CUT} = {vthr:.6f}")

    W = ("TRAIN", "VAL", "OOS", "HOLDOUT")
    M = {w: dsel & (A.split == w).to_numpy() for w in W}

    def perf(mask):
        d = A[mask]
        if not len(d): return 0.0, 0
        t = slotN(d.assign(y=y[mask]), NSLOT)
        return (float(np.mean(t) * 1e4) if len(t) else 0.0), int(len(t))

    variants = {"무필터": np.ones(len(A), bool), "HGB만": p > tau,
                "게이트만": gate, "게이트+HGB": gate & (p > tau)}
    print(f"\n{'':16s}" + "".join(f"{w:>14s}" for w in W))
    res = {}
    for nm, mk in variants.items():
        r = {w: perf(M[w] & mk) for w in W}
        res[nm] = r
        print(f"{nm:16s}" + "".join(f"{r[w][0]:+9.2f}(n{r[w][1]:4d})" for w in W))

    payload = {
        "model_id": MODEL_ID, "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        "kind": "entry_evidence_collector_hgb_volgate",
        "supersedes": ["v1/v2 (L0 contaminated)", "v3 (TabPFN, no performance basis + GPU cost)"],
        "label": "L3 = 1m-resolved fill bar",
        "seeds": list(SEEDS), "hp": HP, "models": hgb,
        "feature_cols": FEATS, "feature_medians": {k: float(v) for k, v in med.items()},
        "signal_code_map": card1["signal_code_map"], "signal_horizons": card1["signal_horizons"],
        "policy": {**POLICY, "hgb_tau": tau, "vol_threshold_atr_pct": vthr},
        "train_range": card1["train_range"], "splits": card1["splits"],
        "frozen_metrics": {k: {w: {"mean_bp": round(v[w][0], 4), "n": v[w][1]} for w in W}
                           for k, v in res.items()},
        "NOT_A_PROMOTION": (
            "No established edge. Day-clustered bootstrap: the vol gate fails VAL in all 3 "
            "variants; only 8-22 independent days; p90->p95 flips VAL's sign. The model's "
            "selection is a superset of the gate's and performs worse. This artifact exists to "
            "collect UNCONTAMINATED FORWARD data -- the only remaining evidence source."),
        "design_note": (
            "Places ALL arm1 candidates and records pred_hgb / atr_pct / vol_pct / gate flag, so "
            "any filter (and any slot policy) can be evaluated offline from the ledger. This is ~7x "
            "faster evidence accumulation than filtering at placement (9.2 vs 1.34 fills/day)."),
    }
    ap = OUT / "model.joblib"
    joblib.dump(payload, ap)
    (OUT / "model_card.json").write_text(json.dumps(
        {k: v for k, v in payload.items() if k != "models"}, indent=2, ensure_ascii=False))
    sha = hashlib.sha256(ap.read_bytes()).hexdigest()[:16]
    (OUT / "SHA256").write_text(sha + "\n")
    log(f"\n동결: {ap} ({ap.stat().st_size/1e6:.1f} MB) sha256:{sha}")

    Q = joblib.load(ap)
    p2 = np.mean([m.predict(X) for m in Q["models"]], axis=0)
    d = float(np.abs(p2 - p).max())
    log(f"재적재 검증: 예측 max|Δ| = {d:.2e} {'✅' if d < 1e-12 else '❌'}")
    return 0 if d < 1e-12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
