#!/usr/bin/env python3
"""ETH 진입 모델 동결 -- 배포 가능한 아티팩트 생성 (2026-09-03).

지금까지 b1~b9가 매번 메모리에서 새로 학습하고 버렸다. 재현은 되지만 **고정된 물건이 없다.**
전진 섀도우를 돌리려면 동결 아티팩트가 선행돼야 하므로 여기서 만든다.

동결 대상 (전부 사전등록 판정으로 확정됨)
  트리거  8종 raw (`compute_signals`, ⚠️`cluster_dedup` 금지 -- 앵커 선택이 미래참조)
  진입    양팔 지정가 depth 3.0xATR · 대기 6봉 · 미체결 취소
  필터    HistGradientBoostingRegressor(squared_error) 5시드 평균 · 161피쳐 · 예측 > 40bp
  슬롯    4
  청산    트레일링 SL3.0/ARM1.0/Trail0.1, horizon은 신호별
  비용    왕복 10bp (peg-maker 실측 대기)

⚠️이 아티팩트는 **연구 동결본**이지 승격 승인이 아니다. HOLDOUT은 8종 신호가 이미 소진했고
여기서 깊이/τ 선별에도 참고했으므로 +59.66bp는 진단이다. 승격 근거는 전진 섀도우로만 번다.

산출 후 **재적재 검증**을 돌려 동결 수치를 그대로 재현하는지 확인한다.
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
MODEL_ID = "eth_entry_limit_fade_v1_20260903"
OUT = ROOT / f"tmp/{MODEL_ID}"
POLICY = {
    "depth_atr": 3.0, "wait_bars": 6, "tau": 0.0040, "slots": 4,
    "exit": {"sl_atr": 3.0, "arm_atr": 1.0, "trail_atr": 0.1},
    "cost_roundtrip": 0.0010, "margin_fraction": 0.30, "leverage": 3.0,
    "both_arms": True, "cancel_if_unfilled": True,
}


def log(m): print(f"[freeze] {m}", flush=True)


def main() -> int:
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    excl = set(base + ["arm", "sig_id", "atr_pct", "depth", "y", "split", "timestamp", "i",
                       "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"])
    R = [c for c in D.columns if c.endswith("_r136")] + \
        [c for c in D.columns if c not in excl and not c.endswith("_r136")]
    R = list(dict.fromkeys([c for c in R if D[c].dtype.kind in "fiub"]))
    FEATS = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R))
    X = D[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (D.split == "TRAIN").to_numpy()
    med = X[tr].median()
    X = X.fillna(med)
    y = D["y"].to_numpy()
    log(f"TRAIN {int(tr.sum()):,} · 피쳐 {len(FEATS)} · 전체 {len(D):,}")

    models, preds = [], []
    for s in SEEDS:
        m = HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP).fit(X[tr], y[tr])
        models.append(m); preds.append(m.predict(X))
        log(f"  시드 {s} 학습 완료")
    pred = np.mean(preds, axis=0)

    dsel = ((D.depth == POLICY["depth_atr"]) & (D.btf <= POLICY["wait_bars"])).to_numpy()
    frozen = {}
    for wn in ("TRAIN", "VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == wn).to_numpy()
        w = D[m]; pv = pred[m]
        v = slotN(w[pv > POLICY["tau"]], POLICY["slots"])
        nn, bp, pf = stat(v)
        frozen[wn] = {"n": nn, "mean_bp": round(bp, 4), "pf": round(pf, 4),
                      "mean_bp_exact": float(bp),          # 재적재 검증용 (반올림 없음)
                      "keep_frac": round(float((pv > POLICY["tau"]).mean()), 4)}
        log(f"  {wn:8s} n={nn:5d} {bp:+7.2f}bp PF{pf:5.2f} (유지 {(pv > POLICY['tau']).mean():.1%})")

    # 신호별 horizon (청산 창)
    hz = {k: int(v["horizon"]) for k, v in cfg["cfg"].items()}
    sig_codes = dict(zip(pd.Categorical(D.signal).categories,
                         range(len(pd.Categorical(D.signal).categories))))
    try:
        git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
    except Exception:
        git_sha = "unknown"

    payload = {
        "model_id": MODEL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "kind": "entry_filter_regressor",
        "target": "per-arm net return at FILL price after roundtrip cost (price_move*notional - cost)",
        "models": models, "seeds": SEEDS, "hp": HP, "loss": "squared_error",
        "feature_cols": FEATS, "feature_medians": {k: float(v) for k, v in med.items()},
        "signal_code_map": sig_codes, "signal_horizons": hz,
        "policy": POLICY,
        "train_range": [str(D.loc[tr, "timestamp"].min()), str(D.loc[tr, "timestamp"].max())],
        "splits": {"VAL": "2025-09-01", "OOS": "2026-01-01", "HOLDOUT": "2026-04-01"},
        "frozen_metrics": frozen,
        "provenance": {
            "fills_source": str(B6 / "fills.csv"),
            "scripts": ["research_eth_entry_direction_oracle_v2_20260903.py",
                        "research_eth_entry_b6_expand_20260903.py",
                        "research_eth_entry_b7_final_controls_20260903.py",
                        "research_eth_entry_b8_featsel_expanded_20260903.py",
                        "research_eth_entry_b9_arch_sweep_20260903.py"],
            "controls_passed": ["random-filter p=0.000 (3 windows)", "5 random seeds 5/5 (3 windows)",
                                "time-block cluster bootstrap CI above no-filter (3 windows)",
                                "momentum-flip loses in all 3 windows",
                                "DSR 1.0000 / PBO 0.0437 (172 trials)"],
            "feature_selection": "B8: k=15..161 flat; selected best (k=25) lost on OOS -> full 161 frozen",
            "architecture": "B9: absolute/quantile/winsor did not beat squared on BOTH windows -> squared frozen",
        },
        "caveats": {
            "holdout_spent": "8 signals each consumed their single holdout; also consulted for depth/tau -> diagnostic only",
            "cost_unverified": "roundtrip 10bp assumed; peg-maker measurement pending >=2026-09-04",
            "fill_model_unverified": "assumes fill at limit when low<=limit (or high>=limit); not validated at 3xATR depth",
            "not_a_promotion": "research freeze only; fresh holdout must be earned by forward shadow",
            "cluster_dedup_forbidden": "trigger set MUST come from raw compute_signals() -- anchor selection is lookahead",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    ap = OUT / "model.joblib"
    joblib.dump(payload, ap)
    meta = {k: v for k, v in payload.items() if k != "models"}
    (OUT / "model_card.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    sha = hashlib.sha256(ap.read_bytes()).hexdigest()[:16]
    (OUT / "SHA256").write_text(sha + "\n")
    log(f"\n동결 저장: {ap} ({ap.stat().st_size/1e6:.1f} MB) sha256:{sha}")

    # ---- 재적재 검증 ----
    log("\n=== 재적재 검증 ===")
    P = joblib.load(ap)
    X2 = D[P["feature_cols"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X2 = X2.fillna(pd.Series(P["feature_medians"]))
    p2 = np.mean([m.predict(X2) for m in P["models"]], axis=0)
    ok = True
    for wn in ("VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == wn).to_numpy()
        w = D[m]; pv = p2[m]
        v = slotN(w[pv > P["policy"]["tau"]], P["policy"]["slots"])
        nn, bp, _ = stat(v)
        same = (nn == frozen[wn]["n"]) and abs(bp - frozen[wn]["mean_bp_exact"]) < 1e-9
        ok &= same
        log(f"  {wn:8s} n={nn:5d} {bp:+7.2f}bp  {'✅일치' if same else '❌불일치'}")
    log(f"\n{'✅ 동결 검증 통과' if ok else '❌ 동결 검증 실패'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
