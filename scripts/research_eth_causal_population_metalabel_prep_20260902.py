#!/usr/bin/env python3
"""증거신호 8종 -- 인과 모집단으로 메타라벨 재학습 준비 (2026-09-02, 재료화 1단계).

WHY
---
2026-09-02 승격 감사에서 드러난 학습/추론 모집단 불일치를 고친다. 배포된 메타라벨은
**cluster-anchored 발동집합**으로 학습됐는데(라이브 스코어러 주석: "the model was trained on
cluster-anchored RAW fire"), 라이브 추론은 **raw 트리거 봉**에서 일어난다. 그 차이가 작지 않다 --
str_z는 raw 8,415 vs 앵커 4,522(1.9배), demarker는 9,992 vs 2,581(3.9배)이고, 여분은 전부
클라이맥스가 아닌 봉이다.

표시 전용이면 "확률 톤이 다르다" 수준이지만, **하류 DL/RL의 재료로 쓸 거면 훨씬 나쁘다** --
어긋난 분포에서 나온 확률을 하류 모델이 진짜 신호로 알고 학습한다.

WHAT
----
라벨 정의(K, HORIZON)와 피처(Tier0 23)는 **배포본과 정확히 동일하게 고정**하고, 모집단만
앵커 -> raw 트리거 전체로 바꾼다. 그래야 AUC 차이가 순수하게 모집단 효과다.
K/HORIZON 출처는 `live_evidence_signal_metalabel_20260829.py`의 METALABEL_SIGNALS(정본).

⚠️K는 앵커 모집단에서 ~50/50으로 보정된 값이다. raw 모집단에서는 hit rate가 달라진다.
그건 의도된 것이다 -- 라벨을 바꾸면 AUC 비교가 무의미해지므로(저장소 규칙), 라벨은 고정하고
양성률 변화를 그대로 보고한다.

트리거는 `live_evidence_signal_dashboard_20260823.py::compute_signals()`에서 가져온다 --
라이브가 실제로 쓰는 바로 그 정의이고 완전히 인과적이다. cluster_dedup을 쓰지 않는다.

분할: TRAIN < 2025-09-01 / VAL 2025-09~12 / OOS 2026-01~03 / HOLDOUT >= 2026-04-01(미접촉)
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

OUT_DIR = ROOT / "tmp/eth_causal_population_metalabel_20260902"
BTC_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"
START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

# (k, horizon_bars) -- live_evidence_signal_metalabel_20260829.py::METALABEL_SIGNALS 정본
CFG = {
    "taker_delta_z_climax":     (2.00, 24),
    "short_term_return_z":      (1.75, 12),
    "liquidity_sweep":          (4.00, 30),
    "orthogonal_combo":         (3.571, 24),
    "smt_divergence":           (4.20, 72),
    "fib_extension_exhaustion": (2.35, 20),
    "demarker_extreme":         (0.70, 8),
    "kalman_deviation_meanrev": (2.50, 12),
}


def log(m): print(f"[prep] {m}", flush=True)


def main() -> int:
    from live_evidence_signal_dashboard_20260823 import compute_signals
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (
        FEATURE_COLUMNS, build_indicator_frame)

    kl = load_klines()
    log(f"klines {len(kl):,}봉 {kl['timestamp'].min()} ~ {kl['timestamp'].max()}")
    ind = build_indicator_frame(kl)
    # smt_divergence는 ETH-BTC 다이버전스라 BTC 고저가 필요하다. 없으면 조용히 발동 0이 된다.
    btc = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    btc = btc.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    log(f"BTC {len(btc):,}봉 {btc['timestamp'].min()} ~ {btc['timestamp'].max()}")
    sig = compute_signals(kl, btc_df=btc)
    log(f"compute_signals 완료: {sig.shape[1]}열 | smt 발동 "
        f"{int(sig['bottom_smt_divergence'].fillna(False).sum() + sig['top_smt_divergence'].fillna(False).sum()):,}")

    high, low, close = (kl[c].to_numpy(float) for c in ("high", "low", "close"))
    atr = ind["atr_pct"].to_numpy(float)
    ts = pd.DatetimeIndex(kl["timestamp"])
    n = len(kl)
    base = [c for c in FEATURE_COLUMNS if c != "is_bottom"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, (K, H) in CFG.items():
        cb, ct = f"bottom_{name}", f"top_{name}"
        if cb not in sig.columns:
            log(f"⚠️ {name}: 트리거 컬럼 없음, 건너뜀"); continue
        rows = []
        for side, col in (("bottom", cb), ("top", ct)):
            idx = np.flatnonzero(sig[col].fillna(False).to_numpy(dtype=bool))
            idx = idx[(idx < n - H) & (ts[idx].to_numpy() >= np.datetime64(START))]
            idx = idx[np.isfinite(atr[idx]) & (atr[idx] > 0)]
            entry = close[idx]
            if side == "bottom":
                ext = np.array([high[i + 1:i + H + 1].max() for i in idx])
                mv = (ext - entry) / entry
            else:
                ext = np.array([low[i + 1:i + H + 1].min() for i in idx])
                mv = (entry - ext) / entry
            rows.append(pd.DataFrame({
                "pos": idx, "timestamp": ts[idx], "side": side,
                "move_atr_mult": mv / atr[idx],
                "hit": (mv >= K * atr[idx]).astype(int),
                "is_bottom": 1 if side == "bottom" else 0}))
        d = pd.concat(rows, ignore_index=True).sort_values("pos").reset_index(drop=True)
        for c in base:
            d[c] = ind[c].to_numpy()[d["pos"].to_numpy()]
        split = np.where(d.timestamp < VAL_START, "TRAIN",
                 np.where(d.timestamp < OOS_START, "VAL",
                 np.where(d.timestamp < HOLDOUT_START, "OOS", "HOLDOUT")))
        d["split"] = split
        d.to_csv(OUT_DIR / f"{name}_causal_fires.csv", index=False)
        cnt = d.split.value_counts()
        rec = {"signal": name, "k": K, "horizon": H, "n_total": len(d),
               "n_train": int(cnt.get("TRAIN", 0)), "n_val": int(cnt.get("VAL", 0)),
               "n_oos": int(cnt.get("OOS", 0)), "n_holdout": int(cnt.get("HOLDOUT", 0)),
               "hit_rate_train": round(float(d.loc[d.split == "TRAIN", "hit"].mean()), 4),
               "hit_rate_all": round(float(d.hit.mean()), 4),
               "tabpfn_train_ok": bool(int(cnt.get("TRAIN", 0)) <= 18000)}
        summary.append(rec)
        log(f"{name:26s} K={K:<5} H={H:<3} | 총 {len(d):6,} "
            f"(TR {rec['n_train']:5,} VAL {rec['n_val']:4,} OOS {rec['n_oos']:4,} HOLD {rec['n_holdout']:5,}) "
            f"| TRAIN 양성률 {rec['hit_rate_train']:.3f} | TabPFN컨텍스트 {'OK' if rec['tabpfn_train_ok'] else '초과⚠️'}")

    s = pd.DataFrame(summary)
    s.to_csv(OUT_DIR / "population_summary.csv", index=False)
    (OUT_DIR / "config.json").write_text(json.dumps(
        {"cfg": {k: {"k": v[0], "horizon": v[1]} for k, v in CFG.items()},
         "population": "raw triggers from compute_signals() -- NO cluster_dedup (causal)",
         "features": base + ["is_bottom"], "splits": {"VAL": str(VAL_START), "OOS": str(OOS_START),
         "HOLDOUT": str(HOLDOUT_START)}, "holdout_touched": False}, indent=2))
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
