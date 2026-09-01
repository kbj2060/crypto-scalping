#!/usr/bin/env python3
"""V자반등 재학습 -- 후보풀(라벨 생성용 트리거 구성) 3종 비교.

## 질문

V_REBOUND를 재학습한다면, 라벨/후보를 만드는 데 쓰는 증거신호 구성을 **기존 그대로** 둘 것인가,
아니면 2026-09-01 세션이 찾아낸 **필터링들을 적용**할 것인가?

## 비교하는 3개 풀

  A_baseline   : 현행 배포판 9트리거 그대로 (raw fires, dedup 없음, orthogonal_combo 포함)
  B_dedup      : A에서 ①orthogonal_combo 제거(순증분 0건, 수학적으로 손실 없음 확인됨)
                 ②kalman_deviation_meanrev/demarker_extreme에 GAP=12 cluster_dedup
                 (ETH에서 한 번도 안 고쳐졌던 raw 미중복제거 버그)
  C_dedup_atr  : B + 각 트리거를 **자기 발동시 ATR 중앙값** 이상인 봉으로만 제한
                 (2026-09-01 저ATR 결함 발견의 후보풀 적용판)

A ⊇ B ⊇ C (필터는 제거만 한다).

## ⚠️ 평가 설계 (풀마다 모집단이 다르므로)

풀이 다르면 후보 모집단이 다르고, 그러면 각자의 AUC를 직접 비교하는 건 무효다
(feedback_cross_model_auc_comparison_requires_matched_label_difficulty_20260901). 따라서 둘 다 본다:

  1. **공통 기준 평가**: 전 모델을 **A풀의 VAL 후보** 위에서 평가 -> 사과 대 사과 모델 품질.
     A는 최대 집합이므로 어느 풀로 학습했든 배포시 마주할 수 있는 모집단의 상한이다.
  2. **자기 풀 평가**: 각 모델을 자기 풀의 VAL에서 -- 그 구성으로 게이트까지 바꿨을 때의 실제
     배포 성능. **풀 간 직접비교 금지**로 명시한다.

추가로 풀 자체의 품질 지표(V자반등 라벨 비율, held_up 얽힘 잔존도)도 함께 본다 -- 모델과
무관하게 "이 풀이 더 좋은 후보를 담고 있는가"를 보는 축.

라벨식은 현행 V0(giveback) 그대로. GBM 프록시(TabPFN 아님), 5시드. TRAIN < 2025-09-01 /
VAL 2025-09-01~2025-12-31. ⚠️OOS/HOLDOUT 미터치.

Run with the quant_ai conda env (CPU only):
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_v_rebound_pool_variant_retrain_comparison_20260901.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/kbj20/crypto-scalping")
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_btc_demarker_extreme_metalabel_tabpfn_20260901 import cluster_dedup  # noqa: E402

FEAS = ROOT / "scripts/research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py"
_fspec = importlib.util.spec_from_file_location("everybar_feas_poolcmp", FEAS)
_feas = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(_feas)
_vs = _feas._vs

OUT_JSON = ROOT / "data/research/eth_v_rebound_pool_variant_comparison_20260901/report.json"

FEATURE_COLUMNS = _feas.FEATURE_COLUMNS
TRAIN_END = _feas.TRAIN_END
VAL_END = _feas.VAL_END
START = _feas.START
SEEDS = [20260901, 141592, 271828, 577215, 20260829]
W = _vs.W
GAP = 12

NAMED8 = _vs.NAMED8
ALL9 = _vs.ALL9

# 각 트리거의 발동시 ATR 중앙값(bp) -- 2026-09-01 실측(라이브 경로 동일, HOLDOUT 이전).
# 8개는 live_evidence_signal_metalabel_20260829.METALABEL_SIGNALS의 atr_median_bp와 같은 값이고,
# local_extreme은 증거신호가 아니라 이 스크립트를 위해 같은 방식으로 새로 측정했다.
ATR_MEDIAN_BP = {
    "taker_delta_z_climax": 31.5, "short_term_return_z": 37.0, "liquidity_sweep": 26.4,
    "orthogonal_combo": 32.3, "smt_divergence": 24.4, "fib_extension_exhaustion": 26.2,
    "demarker_extreme": 32.6, "kalman_deviation_meanrev": 36.6, "local_extreme": 23.4,
}
DEDUP_SIGNALS = {"kalman_deviation_meanrev": "kalman_dev_z", "demarker_extreme": "dem"}


def log(msg: str) -> None:
    print(f"[pool_cmp] {msg}", flush=True)


def build_base():
    feat = _feas.build_all_bar_frame()
    eth = _vs.load_klines(_feas.ETH_CSV)
    btc = _vs.load_klines(_feas.BTC_CSV)
    impl = _vs.load_impl()
    causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())
    sig = compute_signals(eth, btc_df=btc, funding_df=None)
    sig["atr"] = causal["atr"].to_numpy()

    n = len(sig)
    low, high = sig["low"].to_numpy(), sig["high"].to_numpy()
    lo = np.zeros(n, dtype=bool); hi = np.zeros(n, dtype=bool)
    for i in range(W, n - W):
        if low[i] == low[i - W:i + W + 1].min():
            lo[i] = True
        if high[i] == high[i - W:i + W + 1].max():
            hi[i] = True
    sig["bottom_local_extreme"] = lo
    sig["top_local_extreme"] = hi

    log("applying GAP=12 dedup to kalman/demarker (B/C pools only)...")
    for name, col in DEDUP_SIGNALS.items():
        v = sig[col].to_numpy(dtype=float)
        for side in ("bottom", "top"):
            raw = sig[f"{side}_{name}"].fillna(False).to_numpy().astype(bool) & np.isfinite(v)
            idx = np.flatnonzero(raw)
            kept = cluster_dedup(idx, v[idx], most_negative=(side == "bottom"), gap=GAP) if len(idx) else idx
            ded = np.zeros(n, dtype=bool); ded[kept] = True
            sig[f"{side}_{name}_ded"] = ded
            log(f"  {side}_{name}: {int(raw.sum())} -> {int(ded.sum())}")
    return sig, feat


def pool_mask(sig: pd.DataFrame, feat_atr_bp: np.ndarray, side: str, variant: str) -> np.ndarray:
    """해당 side에서 이 풀의 후보인 봉 마스크."""
    if variant == "A":
        names, use_ded, atr_filter = ALL9, False, False
    elif variant == "B":
        names, use_ded, atr_filter = [n for n in ALL9 if n != "orthogonal_combo"], True, False
    else:
        names, use_ded, atr_filter = [n for n in ALL9 if n != "orthogonal_combo"], True, True

    m = np.zeros(len(sig), dtype=bool)
    for nm in names:
        col = f"{side}_{nm}_ded" if (use_ded and nm in DEDUP_SIGNALS) else f"{side}_{nm}"
        fired = sig[col].fillna(False).to_numpy().astype(bool)
        if atr_filter:
            fired = fired & (feat_atr_bp >= ATR_MEDIAN_BP[nm])
        m |= fired
    return m


def build_long(sig, feat, held_up, statuses, pools) -> pd.DataFrame:
    rows = []
    for side, is_down in (("bottom", True), ("top", False)):
        d = pd.DataFrame({"timestamp": sig["timestamp"], "status": statuses[side],
                          "held_up": held_up[side]})
        for v in ("A", "B", "C"):
            d[f"pool_{v}"] = pools[(side, v)]
        merged = d.merge(feat, on="timestamp", how="inner", suffixes=("", "_f"))
        sub = pd.DataFrame({"timestamp": merged["timestamp"], "side": side})
        sub["is_downside"] = np.int8(1 if is_down else 0)
        level = merged["sweep_level_low"].to_numpy() if is_down else merged["sweep_level_high"].to_numpy()
        atr = merged["atr"].to_numpy(dtype=float)
        pen = (level - merged["low"].to_numpy()) if is_down else (merged["high"].to_numpy() - level)
        sub["sweep_penetration_atr"] = pen / atr
        sub["atr"] = atr
        sub["atr_percentile_864"] = merged["atr_percentile_864"].to_numpy()
        sub["range_width_pct"] = merged["range_width_pct"].to_numpy()
        sub["hour_utc"] = merged["hour_utc"].to_numpy()
        sub["weekday"] = merged["weekday"].to_numpy()
        dz = merged["delta_z"].to_numpy(dtype=float)
        sub["delta_z"] = dz
        sub["flow_aligned_delta_z"] = dz if is_down else -dz
        for c in ["p_fast", "p_slow", "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio",
                  "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile", "ret3_z", "rsi"]:
            sub[c] = merged[c].to_numpy()
        sub["status"] = merged["status"].to_numpy()
        sub["held_up"] = merged["held_up"].to_numpy()
        for v in ("A", "B", "C"):
            sub[f"pool_{v}"] = merged[f"pool_{v}"].to_numpy()
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    long = long.loc[(long["timestamp"] >= START) & (long["timestamp"] < VAL_END)].reset_index(drop=True)
    long = long.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    long["label"] = np.where(long["status"] == "v_rebound", 1.0,
                     np.where(long["status"] == "chop", 0.0, np.nan))
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN", "VAL")
    return long


def main() -> int:
    t0 = time.time()
    sig, feat = build_base()

    log("computing V0 labels + held_up on all bars...")
    statuses = {"bottom": _vs.label_variant(sig, True, "wick", 0),
                "top": _vs.label_variant(sig, False, "wick", 0)}
    low, high = sig["low"].to_numpy(), sig["high"].to_numpy()
    held_up = {"bottom": _vs.fwd_window(low, 1, W, "min") >= low,
               "top": _vs.fwd_window(high, 1, W, "max") <= high}

    atr_bp_map = dict(zip(feat["timestamp"], feat["atr_pct"].to_numpy() * 1e4))
    feat_atr_bp = sig["timestamp"].map(atr_bp_map).to_numpy(dtype=float)

    pools = {(side, v): pool_mask(sig, feat_atr_bp, side, v)
             for side in ("bottom", "top") for v in ("A", "B", "C")}
    long = build_long(sig, feat, held_up, statuses, pools)

    log("=== 풀 자체 품질 (모델 무관, TRAIN+VAL) ===")
    log(f"  {'pool':12s} {'후보수':>8s} {'라벨가능':>8s} {'V자반등률':>9s} {'held_up=T비중':>12s}")
    pool_stats = {}
    for v in ("A", "B", "C"):
        m = long[f"pool_{v}"].to_numpy()
        lab = m & long["label"].notna().to_numpy()
        rate = float(long.loc[lab, "label"].mean())
        hu = float(long.loc[m, "held_up"].mean())
        pool_stats[v] = {"n_candidates": int(m.sum()), "n_labeled": int(lab.sum()),
                         "v_rebound_rate": round(rate, 4), "held_up_share": round(hu, 4)}
        log(f"  {v:12s} {int(m.sum()):>8d} {int(lab.sum()):>8d} {rate:>9.2%} {hu:>12.2%}")

    ref_val = long.loc[(long["split"] == "VAL") & long["label"].notna() & long["pool_A"]]
    log(f"\n공통 기준 평가셋 = A풀의 VAL 후보: n={len(ref_val)}, base={ref_val['label'].mean():.4f}")

    results = {}
    for v in ("A", "B", "C"):
        tr = long.loc[(long["split"] == "TRAIN") & long["label"].notna() & long[f"pool_{v}"]]
        own_val = long.loc[(long["split"] == "VAL") & long["label"].notna() & long[f"pool_{v}"]]
        log(f"\n=== 풀 {v} 학습 (TRAIN n={len(tr)}, base={tr['label'].mean():.4f}) ===")
        ref_a, own_a = [], []
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(random_state=sd, max_iter=300, early_stopping=True,
                                               validation_fraction=0.15)
            m.fit(tr[FEATURE_COLUMNS], tr["label"].to_numpy())
            ref_a.append(roc_auc_score(ref_val["label"].to_numpy(), m.predict_proba(ref_val[FEATURE_COLUMNS])[:, 1]))
            own_a.append(roc_auc_score(own_val["label"].to_numpy(), m.predict_proba(own_val[FEATURE_COLUMNS])[:, 1]))
        ra, oa = np.array(ref_a), np.array(own_a)
        results[v] = {
            "train_n": int(len(tr)), "train_base_rate": round(float(tr["label"].mean()), 4),
            "own_val_n": int(len(own_val)), "own_val_base_rate": round(float(own_val["label"].mean()), 4),
            "auc_on_COMMON_ref_A_val": {"mean": round(float(ra.mean()), 4), "std": round(float(ra.std()), 4)},
            "auc_on_own_pool_NOT_COMPARABLE": {"mean": round(float(oa.mean()), 4), "std": round(float(oa.std()), 4)},
        }
        log(f"  공통기준(A풀 VAL) AUC {ra.mean():.4f} ± {ra.std():.4f}   "
            f"| (참고,비교불가) 자기풀 AUC {oa.mean():.4f} ± {oa.std():.4f} (n={len(own_val)})")

    base = results["A"]["auc_on_COMMON_ref_A_val"]["mean"]
    log("\n=== 공통 기준, A 대비 ===")
    for v in ("A", "B", "C"):
        d = results[v]["auc_on_COMMON_ref_A_val"]["mean"] - base
        results[v]["delta_vs_A_on_common_ref"] = round(d, 4)
        log(f"  {v}: delta={d:+.4f}  (std {results[v]['auc_on_COMMON_ref_A_val']['std']:.4f})")

    report = {
        "signal": "v_rebound_pool_variant_retrain_comparison", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {
            "screening_only": True, "model": "HistGradientBoostingClassifier proxy (not TabPFN)",
            "live_code_changed": False, "holdout_touched": False, "oos_touched": False,
            "label_formula": "V0 unchanged (current live label_side())",
            "splits": {"TRAIN": f"< {TRAIN_END}", "VAL": f"{TRAIN_END} .. {VAL_END}"},
            "evaluation_note": ("풀마다 모집단이 달라 자기풀 AUC 직접비교는 무효. 전 모델을 A풀 "
                                "VAL이라는 고정 기준에서도 평가해 사과 대 사과로 맞췄다."),
        },
        "pools": {
            "A_baseline": "현행 배포판 9트리거 (raw fires, dedup 없음, orthogonal_combo 포함)",
            "B_dedup": "A - orthogonal_combo + kalman/demarker GAP=12 dedup",
            "C_dedup_atr": "B + 각 트리거를 자기 발동시 ATR 중앙값 이상으로 제한",
        },
        "atr_median_bp": ATR_MEDIAN_BP, "seeds": SEEDS,
        "pool_quality": pool_stats, "results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"\nreport saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
