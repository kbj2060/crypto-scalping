#!/usr/bin/env python3
"""ETH 진입 모델 **v2 동결 -- TabPFN 직접 분류** (2026-09-03).

⚠️⚠️**사전등록 기준 미달 상태에서 사용자 결정으로 채택한다.** 감사 가능하도록 명시한다.

  B12/B14/B15/B16에서 TabPFN 분류는 **3창 전부 HGB를 앞섰다**(B16: +2.9/+3.4/+8.9bp,
  B14: +0.91/+1.79/+5.52 동일 데이터량 기준). 대조군도 3창 p=0.000으로 통과했다.
  그러나 **순열검정을 한 번도 통과하지 못했다** -- B15 시드페어드 VAL p=0.1391 / OOS p=0.0518,
  B16 봉단위스왑 VAL p=0.3092 / OOS p=0.2843. 사전등록은 "양 창 p<0.05"였다.

  **채택 근거(사용자, 2026-09-03)**: "TabPFN 직접으로 확정하자. 이게 더 앞으로 어떤 추가
  모듈을 붙여도 더 잘 배울 것 같아." -- 즉 현재 표본에서의 통계적 유의성이 아니라, 재료·모듈이
  추가될 때의 **확장성**에 근거한 선택이다. v1(HGB)은 폐기하지 않고 `tmp/`에 남긴다.

⚠️**운영 비용을 함께 기록한다**(2026-09-03 실측): 한 행 채점 p50 **68.7초** / p95 86.1초로
HGB(15.4ms)의 약 4,800배. 후보 봉이 41.2봉/일이므로 **하루 약 47분 GPU**, 주문 제출이 **69초
지연**된다. 상주 GPU 1.065GB. 5분봉·3ATR·대기6봉이라 치명적이진 않으나 기준가가 1분 낡는다.

## ⭐고정 임계값 -- 이게 v1과 가장 다른 점

v1은 "예측 > 40bp"라는 절대 임계값을 썼다. TabPFN은 **확률**을 내므로 같은 규칙을 쓸 수 없다.
그리고 연구(B12~B16)의 TabPFN 수치는 **창마다 HGB의 유지비율에 맞춘 분위수**로 뽑았는데,
**라이브는 미래 창의 분포를 모르므로 그걸 할 수 없다.**

그래서 여기서는 **TRAIN 후보 분포에서만** 임계값을 유도한다:
    p_thr = quantile(proba[TRAIN 후보], 1 - keep_frac_TRAIN)
그리고 VAL/OOS/HOLDOUT을 **그 고정 임계값 하나로** 다시 평가한다. 이 수치가 라이브가 실제로
낼 수 있는 것이고, 연구 표의 창별 매칭 수치와 다르다 -- 둘 다 카드에 남긴다.

## 동결 대상 = 컨텍스트

TabPFN은 in-context 학습이라 "가중치 동결"이 아니라 **컨텍스트 동결**이다.
TRAIN 후보 X(float32) + 라벨 + 시드 5개를 저장하면 5개 멤버 컨텍스트가 정확히 재현된다.
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
V1 = ROOT / "tmp/eth_entry_limit_fade_v1_20260903"
MODEL_ID = "eth_entry_limit_fade_v2_tabpfn_20260903"
OUT = ROOT / f"tmp/{MODEL_ID}"
DEPTH, WAIT, TAU_V1, NSLOT = 3.0, 6, 0.0040, 4
LABEL_THR, SUB = 0.0040, 18000
POLICY = {
    "depth_atr": DEPTH, "wait_bars": WAIT, "slots": NSLOT,
    "scorer": "tabpfn_classifier_5member", "label": "y > 40bp",
    "exit": {"sl_atr": 3.0, "arm_atr": 1.0, "trail_atr": 0.1},
    "cost_roundtrip": 0.0010, "margin_fraction": 0.30, "leverage": 3.0,
    "both_arms": True, "cancel_if_unfilled": True,
}


def log(m): print(f"[freeze2] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier

    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    card1 = json.loads((V1 / "model_card.json").read_text())
    FEATS = card1["feature_cols"]
    X = D[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (D.split == "TRAIN").to_numpy()
    med = X[tr].median()
    X = X.fillna(med)
    y = D["y"].to_numpy(); lab = (y > LABEL_THR).astype(int)
    dsel = ((D.depth == DEPTH) & (D.btf <= WAIT)).to_numpy()
    itr = np.flatnonzero(tr)
    # ⭐채점은 **후보 행만** 한다. 정책이 후보에서만 고르므로 나머지 12만행은 낭비다
    # (TabPFN은 한 호출이 수십 초라 이 차이가 9배다).
    prow = np.flatnonzero(dsel)
    Xp = X.iloc[prow].to_numpy()

    def expand(v):
        f = np.full(len(D), -np.inf); f[prow] = v; return f

    log(f"TRAIN {len(itr):,} · 피쳐 {len(FEATS)} · 후보(depth{DEPTH}/wait{WAIT}) {len(prow):,}")

    # --- v1 HGB 기준선 (같은 코드로 재계산해 카드에 병기) ---
    pA = np.mean([HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
                  .fit(X[tr], y[tr]).predict(X) for s in SEEDS], axis=0)

    def pol(mask, score, thr):
        w = D[mask]; s = score[mask]
        t = slotN(w[s > thr], NSLOT)
        return {"n": int(len(t)), "mean_bp": round(float(np.mean(t) * 1e4), 4) if len(t) else 0.0,
                "mean_bp_exact": float(np.mean(t) * 1e4) if len(t) else 0.0,
                "pf": round(float(stat(t)[2]), 4) if len(t) else 0.0,
                "keep_frac": round(float((s > thr).mean()), 4)}

    W = ("TRAIN", "VAL", "OOS", "HOLDOUT")
    m_of = {w: dsel & (D.split == w).to_numpy() for w in W}
    v1m = {w: pol(m_of[w], pA, TAU_V1) for w in W}
    log("v1 HGB(τ=40bp) " + " ".join(f"{w} {v1m[w]['mean_bp']:+.2f}(n{v1m[w]['n']})" for w in W))

    # --- v2 TabPFN 5멤버 ---
    log(f"\nTabPFN 5멤버 적합 (컨텍스트 {SUB:,}행)...")
    # ⭐**저장할 배열에서 그대로 적합한다.** 앞선 시도는 컨텍스트를 float32로 저장하면서
    # 동결 수치는 float64로 냈고, 그 정밀도 차이가 재적재 재현을 최대 6.8bp 어긋나게 했다.
    # 저장물과 계산물을 같은 바이트로 맞춰야 재적재 검증이 성립한다.
    ctx, ps = [], []
    Xtr = X.iloc[itr].to_numpy(np.float32)
    Xp = Xp.astype(np.float32)
    loc = {int(v): i for i, v in enumerate(itr)}
    for k, sd in enumerate(SEEDS):
        rs = np.random.default_rng(sd).choice(itr, size=min(SUB, len(itr)), replace=False)
        sel = np.array([loc[int(v)] for v in rs])
        m = TabPFNClassifier(device="cuda", random_state=sd)
        m.fit(Xtr[sel], lab[itr][sel])
        ps.append(expand(m.predict_proba(Xp)[:, 1]))
        ctx.append(int(len(rs)))
        log(f"  멤버{k} seed={sd} 컨텍스트 {len(rs):,}행")
    P = np.mean(ps, axis=0)

    # --- ⭐고정 임계값: TRAIN 후보 분포에서만 유도 ---
    keep_tr = v1m["TRAIN"]["keep_frac"]
    p_thr = float(np.quantile(P[m_of["TRAIN"]], 1 - keep_tr))
    log(f"\n⭐고정 임계값 유도: TRAIN 유지비율 {keep_tr:.4f} → p_thr = {p_thr:.6f}")
    v2m = {w: pol(m_of[w], P, p_thr) for w in W}

    # 참고: 연구 표가 쓴 '창별 매칭' 수치도 함께 남긴다 (라이브 불가, 비교용)
    v2_matched = {w: pol(m_of[w], P, float(np.quantile(P[m_of[w]], 1 - v1m[w]["keep_frac"])))
                  for w in W}

    print(f"\n{'':26s}" + "".join(f"{w:>12s}" for w in W))
    print(f"{'v1 HGB (τ=40bp)':26s}" + "".join(f"{v1m[w]['mean_bp']:+12.2f}" for w in W))
    print(f"{'v2 TabPFN (고정 p_thr)':26s}" + "".join(f"{v2m[w]['mean_bp']:+12.2f}" for w in W))
    print(f"{'  (참고) 창별 매칭':26s}" + "".join(f"{v2_matched[w]['mean_bp']:+12.2f}" for w in W))
    print(f"{'  체결 수 (고정)':26s}" + "".join(f"{v2m[w]['n']:12d}" for w in W))

    sha_fills = hashlib.sha256(open(B6 / "fills.csv", "rb").read(1 << 22)).hexdigest()[:16]
    payload = {
        "model_id": MODEL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        "kind": "entry_filter_tabpfn_classifier",
        "target": "P(per-arm net return at FILL price after roundtrip cost > 40bp)",
        "scorer": "TabPFNClassifier(device=cuda) x 5 members, in-context",
        "seeds": list(SEEDS), "context_rows_per_member": ctx, "context_subsample": SUB,
        "feature_cols": FEATS, "feature_medians": {k: float(v) for k, v in med.items()},
        "signal_code_map": card1["signal_code_map"], "signal_horizons": card1["signal_horizons"],
        "policy": {**POLICY, "p_threshold": p_thr},
        "threshold_derivation": (
            "p_thr = quantile(proba[TRAIN candidates], 1 - keep_frac_TRAIN) where keep_frac_TRAIN "
            "is v1 HGB's TRAIN keep fraction at tau=40bp. Derived from TRAIN ONLY -- live can "
            "apply it without knowing any future window's distribution."),
        "train_range": card1["train_range"], "splits": card1["splits"],
        "frozen_metrics": v2m,
        "frozen_metrics_window_matched_NOT_LIVE_ACHIEVABLE": v2_matched,
        "v1_hgb_same_run": v1m,
        "context_X_dtype": "float32", "context_X_shape": list(Xtr.shape),
        "fills_head_sha256": sha_fills,
        "adoption": {
            "decision": "user decision 2026-09-03, ADOPTED DESPITE PRE-REGISTRATION FAILURE",
            "user_rationale": "앞으로 추가 모듈을 붙여도 더 잘 배울 것 -- 확장성 근거",
            "preregistered_criterion": "beat HGB on both VAL and OOS AND permutation p<0.05 both",
            "criterion_result": "arm 1 PASSED (3/3 windows), arm 2 FAILED every time",
            "permutation_p_values": {"B15_seed_paired": {"VAL": 0.1391, "OOS": 0.0518},
                                     "B16_bar_swap": {"VAL": 0.3092, "OOS": 0.2843}},
            "note": "v1 HGB artifact is retained at tmp/eth_entry_limit_fade_v1_20260903 and is "
                    "the fallback if forward shadow contradicts this choice."},
        "operational_cost_measured_20260903": {
            "score_latency_ms_n1": {"p50": 68701.1, "p95": 86076.3},
            "hgb_equivalent_ms": 15.4, "ratio": "~4800x",
            "resident_gpu_gb": 1.065, "candidate_bars_per_day": 41.2,
            "gpu_saturated_minutes_per_day": 47,
            "consequence": "order placement delayed ~69s; reference price ~1min stale"},
        "caveats": {
            **card1["caveats"],
            "preregistration_failed": "permutation test never passed; adopted on user judgement",
            "threshold_is_new": "v1 used an absolute 40bp cut; v2 uses a TRAIN-derived probability "
                                "quantile. Research tables used per-window matched fractions which "
                                "live cannot reproduce -- see frozen_metrics vs the *_NOT_LIVE_* key.",
            "inference_cost": "~69s per scoring call on shared GPU; contends with dashboard/shadows"},
        "context_X": Xtr, "context_y": lab[itr].astype(np.int8), "context_index": itr,
    }
    ap = OUT / "model.joblib"
    joblib.dump(payload, ap)
    meta = {k: v for k, v in payload.items() if k not in ("context_X", "context_y", "context_index")}
    (OUT / "model_card.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    sha = hashlib.sha256(ap.read_bytes()).hexdigest()[:16]
    (OUT / "SHA256").write_text(sha + "\n")
    log(f"\n동결 저장: {ap} ({ap.stat().st_size/1e6:.1f} MB) sha256:{sha}")

    # --- 재적재 검증: 컨텍스트로 멤버를 다시 세워 동결 수치를 재현하는가 ---
    log("\n재적재 검증...")
    Q = joblib.load(ap)
    ps2 = []
    loc2 = {int(v): i for i, v in enumerate(Q["context_index"])}
    for sd in Q["seeds"]:
        rs = np.random.default_rng(sd).choice(Q["context_index"], size=SUB, replace=False)
        sel = np.array([loc2[int(v)] for v in rs])
        m = TabPFNClassifier(device="cuda", random_state=sd)
        m.fit(Q["context_X"][sel], Q["context_y"][sel])
        ps2.append(expand(m.predict_proba(Xp)[:, 1]))
    P2 = np.mean(ps2, axis=0)
    ok = True
    for w in W:
        r = pol(m_of[w], P2, Q["policy"]["p_threshold"])
        d = abs(r["mean_bp_exact"] - Q["frozen_metrics"][w]["mean_bp_exact"])
        log(f"  {w:8s} 재현 {r['mean_bp']:+.2f} vs 동결 {Q['frozen_metrics'][w]['mean_bp']:+.2f} "
            f"· |Δ|={d:.2e} {'✅' if d < 1e-6 else '❌'}")
        ok &= d < 1e-6
    log(f"\n{'✅ 재적재 검증 통과' if ok else '❌ 재적재 검증 실패'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
