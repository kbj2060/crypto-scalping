#!/usr/bin/env python3
"""매 봉 스코어링 재설계 -- 트리거 발동여부를 피쳐로 추가하는 테스트 (GBM 프록시, CPU only).

research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py의 후속. 그 스크립트가
"전체 봉 재학습이면 AUC 0.6953으로 작동한다"를 확인했고, 이번엔 Tier0 23피쳐에 **어느 트리거가
발동했는가**를 추가하면 개선되는지 본다. 현행 파이프라인은 이 정보를 아예 모델에 안 준다
(트리거는 순수 후보선택용 audit 컬럼일 뿐, TIER0 입력에 없음) -- 게이트를 없앤 설계에서는
"이 봉에서 스윕이 터졌다"가 모델이 활용할 수 있는 진짜 정보가 된다.

## 이전 증거신호 분석자료를 반영한 3가지 수정

1) **kalman_deviation_meanrev / demarker_extreme에 GAP=12 cluster_dedup 적용**
   (eth_v_rebound_feeder_role_audit_and_dedup_fix_20260901): 라벨빌더와 라이브 서버 둘 다
   이 두 신호를 raw 미중복제거 per-bar 불리언으로 쓰고 있었다 -- 롤링 z-score/오실레이터라
   한번 임계값을 넘으면 되돌아올 때까지 연속 여러 봉이 계속 "발동중"으로 pinned되는 구조.
   BTC는 이미 고쳤으나 ETH엔 한 번도 적용된 적 없던 버그. 피쳐로 쓸 때도 sticky한 raw 불리언은
   노이즈가 되므로 dedup판을 별도 피쳐셋으로 나눠 직접 비교한다.
   held_up 층화 재검증(eth_v_rebound_feeder_role_held_up_stratified_reranking_20260901)에서
   이 dedup이 held_up 아티팩트가 아닌 진짜 개선임이 확인됐다.

2) **orthogonal_combo는 제외하지 않고 포함**: 순증분 후보 0건(발동하는 모든 봉이 이미 다른
   트리거와 겹침)이라 "후보풀 기여도"는 0이지만, **피쳐로서는 다른 질문**이다 -- 그 봉이
   orthogonal_combo 조건까지 만족하는 특정 confluence 패턴임을 표시하는 정보는 union 불리언과
   구별된다. 선험적으로 빼지 않고 순열중요도로 판정한다.

3) **local_extreme은 피쳐에서 제외 (룩어헤드)**: research_eth_v_rebound_label_redesign_variant_
   screen_20260901.py가 확증했듯 local_extreme[i]는 bar i+6에서야 알 수 있다(불일치 0/210,630).
   bar i 시점 피쳐로 넣으면 명백한 룩어헤드. 다만 **얼마나 부풀리는지 정량화하는 대조군(F4)을
   일부러 포함** -- 배포 불가임을 명시적으로 표시.

나머지 8트리거(liquidity_sweep/taker_delta_z_climax/short_term_return_z/orthogonal_combo/
smt_divergence/fib_extension_exhaustion/demarker_extreme/kalman_deviation_meanrev)는 전부
compute_signals()의 인과적 라이브 신호라 bar i 시점에 알 수 있다.

## 피쳐셋

  F1_tier0_only        : 23 Tier0 (이전 스크립트의 모델 B와 동일, 기준선)
  F2_plus_raw_triggers : + 8트리거 raw 불리언 + n_triggers(발동개수)
  F3_plus_deduped      : + 8트리거(kalman/demarker만 GAP=12 dedup) + n_triggers   <- 권장안
  F4_LOOKAHEAD_CONTROL : F3 + local_extreme  <- 배포 불가, 룩어헤드 인플레 정량화 전용

⚠️ 진단 전용: 라이브 코드 변경 없음, OOS/HOLDOUT 미터치(TRAIN < 2025-09-01, VAL 2025-09-01~
2025-12-31). 라벨식은 현행 V0 그대로.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_v_rebound_every_bar_trigger_features_20260901.py
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

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_btc_demarker_extreme_metalabel_tabpfn_20260901 import cluster_dedup  # noqa: E402

FEAS_SCRIPT = ROOT / "scripts/research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py"
_fspec = importlib.util.spec_from_file_location("everybar_feas_20260901", FEAS_SCRIPT)
_feas = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(_feas)

_vs = _feas._vs  # label variant helpers (label_variant / fwd_window / load_klines / load_impl)

ETH_CSV = _feas.ETH_CSV
BTC_CSV = _feas.BTC_CSV
OUT_JSON = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/every_bar_trigger_features_report.json"

TRAIN_END = _feas.TRAIN_END
VAL_END = _feas.VAL_END
START = _feas.START
SEED = _feas.SEED
W = _feas.W
TIER0 = _feas.FEATURE_COLUMNS

CAUSAL8 = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z", "orthogonal_combo",
           "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
DEDUP_SIGNALS = {"kalman_deviation_meanrev": "kalman_dev_z", "demarker_extreme": "dem"}
DEDUP_GAP = 12

TRIG_RAW = [f"trig_{n}" for n in CAUSAL8] + ["n_triggers"]
TRIG_DED = [f"trigd_{n}" for n in CAUSAL8] + ["n_triggers_ded"]

FEATURE_SETS = {
    "F1_tier0_only": TIER0,
    "F2_plus_raw_triggers": TIER0 + TRIG_RAW,
    "F3_plus_deduped_triggers": TIER0 + TRIG_DED,
    "F4_LOOKAHEAD_CONTROL_plus_local_extreme": TIER0 + TRIG_DED + ["trig_local_extreme_LOOKAHEAD"],
}


def log(msg: str) -> None:
    print(f"[trigger_features] {msg}", flush=True)


def build_long_frame_with_triggers() -> pd.DataFrame:
    log("building per-bar indicator frame (whole history)...")
    feat = _feas.build_all_bar_frame()

    log("computing triggers (compute_signals + local_extreme) ...")
    eth = _vs.load_klines(ETH_CSV)
    btc = _vs.load_klines(BTC_CSV)
    impl = _vs.load_impl()
    causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())
    sig = compute_signals(eth, btc_df=btc, funding_df=None)
    sig["atr"] = causal["atr"].to_numpy()

    n = len(sig)
    low, high = sig["low"].to_numpy(), sig["high"].to_numpy()
    lo_flag = np.zeros(n, dtype=bool)
    hi_flag = np.zeros(n, dtype=bool)
    for i in range(W, n - W):
        if low[i] == low[i - W:i + W + 1].min():
            lo_flag[i] = True
        if high[i] == high[i - W:i + W + 1].max():
            hi_flag[i] = True
    sig["bottom_local_extreme"] = lo_flag
    sig["top_local_extreme"] = hi_flag

    log(f"applying GAP={DEDUP_GAP} cluster_dedup to {list(DEDUP_SIGNALS)} (the never-fixed ETH raw-fire bug)...")
    dedup_stats = {}
    for name, col_name in DEDUP_SIGNALS.items():
        col = sig[col_name].to_numpy(dtype=float)
        for side in ("bottom", "top"):
            raw = sig[f"{side}_{name}"].fillna(False).to_numpy().astype(bool) & np.isfinite(col)
            idx_raw = np.flatnonzero(raw)
            kept = (cluster_dedup(idx_raw, col[idx_raw], most_negative=(side == "bottom"), gap=DEDUP_GAP)
                    if len(idx_raw) else idx_raw)
            ded = np.zeros(n, dtype=bool)
            ded[kept] = True
            sig[f"{side}_{name}_ded"] = ded
            dedup_stats[f"{side}_{name}"] = {"raw_fires": int(raw.sum()), "deduped_fires": int(ded.sum())}
            log(f"  {side}_{name}: {int(raw.sum())} -> {int(ded.sum())} fires")
    for name in CAUSAL8:
        if name not in DEDUP_SIGNALS:
            for side in ("bottom", "top"):
                sig[f"{side}_{name}_ded"] = sig[f"{side}_{name}"].fillna(False).to_numpy().astype(bool)

    log("computing V0 (current live) labels on ALL bars, both sides...")
    st_b = _vs.label_variant(sig, is_down=True, anchor_mode="wick", shift=0)
    st_t = _vs.label_variant(sig, is_down=False, anchor_mode="wick", shift=0)
    fwd_low_min = _vs.fwd_window(low, 1, W, "min")
    fwd_high_max = _vs.fwd_window(high, 1, W, "max")

    keep = {"timestamp": sig["timestamp"], "st_b": st_b, "st_t": st_t,
            "held_up_b": fwd_low_min >= low, "held_up_t": fwd_high_max <= high,
            "le_b": lo_flag, "le_t": hi_flag}
    for name in CAUSAL8:
        for side in ("bottom", "top"):
            keep[f"{side}_{name}"] = sig[f"{side}_{name}"].fillna(False).to_numpy().astype(bool)
            keep[f"{side}_{name}_ded"] = sig[f"{side}_{name}_ded"].to_numpy().astype(bool)
    trig_frame = pd.DataFrame(keep)

    merged = trig_frame.merge(feat, on="timestamp", how="inner", suffixes=("", "_f"))
    log(f"merged per-bar frame: {len(merged)} bars")

    rows = []
    for side, is_down in (("bottom", True), ("top", False)):
        d = merged
        st = d["st_b"] if is_down else d["st_t"]
        sub = pd.DataFrame({"timestamp": d["timestamp"], "side": side})
        sub["is_downside"] = np.int8(1 if is_down else 0)

        level = d["sweep_level_low"].to_numpy() if is_down else d["sweep_level_high"].to_numpy()
        atr = d["atr"].to_numpy(dtype=float)
        pen = (level - d["low"].to_numpy()) if is_down else (d["high"].to_numpy() - level)
        sub["sweep_penetration_atr"] = pen / atr
        sub["atr"] = atr
        sub["atr_percentile_864"] = d["atr_percentile_864"].to_numpy()
        sub["range_width_pct"] = d["range_width_pct"].to_numpy()
        sub["hour_utc"] = d["hour_utc"].to_numpy()
        sub["weekday"] = d["weekday"].to_numpy()
        dz = d["delta_z"].to_numpy(dtype=float)
        sub["delta_z"] = dz
        sub["flow_aligned_delta_z"] = dz if is_down else -dz
        for col in ["p_fast", "p_slow", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
                    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi",
                    "bb_width_pctile", "ret3_z", "rsi"]:
            sub[col] = d[col].to_numpy()

        for name in CAUSAL8:
            sub[f"trig_{name}"] = d[f"{side}_{name}"].to_numpy().astype(np.int8)
            sub[f"trigd_{name}"] = d[f"{side}_{name}_ded"].to_numpy().astype(np.int8)
        sub["n_triggers"] = sub[[f"trig_{n}" for n in CAUSAL8]].sum(axis=1).astype(np.int8)
        sub["n_triggers_ded"] = sub[[f"trigd_{n}" for n in CAUSAL8]].sum(axis=1).astype(np.int8)
        sub["trig_local_extreme_LOOKAHEAD"] = (d["le_b"] if is_down else d["le_t"]).to_numpy().astype(np.int8)

        sub["status"] = st.to_numpy()
        sub["held_up"] = (d["held_up_b"] if is_down else d["held_up_t"]).to_numpy()
        rows.append(sub)

    long = pd.concat(rows, ignore_index=True)
    long = long.loc[(long["timestamp"] >= START) & (long["timestamp"] < VAL_END)].reset_index(drop=True)
    long["label"] = np.where(long["status"] == "v_rebound", 1.0,
                     np.where(long["status"] == "chop", 0.0, np.nan))
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN", "VAL")
    long.attrs["dedup_stats"] = dedup_stats
    return long


def auc_or_none(y, p):
    if len(np.unique(y)) < 2:
        return None
    return round(float(roc_auc_score(y, p)), 4)


def shuffle_importance(model, val: pd.DataFrame, cols: list[str], target_cols: list[str],
                       base_auc: float, n_repeats: int = 3) -> dict:
    """Permutation importance restricted to the trigger columns (cheap: only the columns we care
    about, not all 32). delta_auc = base - shuffled; positive means the feature genuinely helps."""
    rng = np.random.default_rng(SEED)
    y = val["label"].to_numpy()
    out = {}
    for c in target_cols:
        if c not in cols:
            continue
        deltas = []
        for _ in range(n_repeats):
            X = val[cols].copy()
            X[c] = rng.permutation(X[c].to_numpy())
            deltas.append(base_auc - auc_or_none(y, model.predict_proba(X)[:, 1]))
        out[c] = {"delta_auc_mean": round(float(np.mean(deltas)), 5),
                  "delta_auc_std": round(float(np.std(deltas)), 5),
                  "fire_rate_val": round(float(val[c].mean()), 5)}
    return out


def main() -> int:
    t0 = time.time()
    long = build_long_frame_with_triggers()
    dedup_stats = long.attrs.get("dedup_stats", {})

    all_feature_cols = sorted({c for cols in FEATURE_SETS.values() for c in cols})
    labeled = long.loc[long["label"].notna()].dropna(subset=all_feature_cols).reset_index(drop=True)
    tr = labeled.loc[labeled["split"] == "TRAIN"]
    va = labeled.loc[labeled["split"] == "VAL"]
    log(f"labeled+clean: TRAIN={len(tr)} VAL={len(va)} "
        f"(base rate TRAIN={tr['label'].mean():.4f} VAL={va['label'].mean():.4f})")

    results = {}
    for fname, cols in FEATURE_SETS.items():
        is_control = "LOOKAHEAD" in fname
        log(f"=== {fname} ({len(cols)} features){'  [LOOKAHEAD CONTROL - NOT DEPLOYABLE]' if is_control else ''} ===")
        m = HistGradientBoostingClassifier(random_state=SEED, max_iter=300, early_stopping=True,
                                           validation_fraction=0.15)
        m.fit(tr[cols], tr["label"].to_numpy())
        p = m.predict_proba(va[cols])[:, 1]
        y = va["label"].to_numpy()
        base_auc = auc_or_none(y, p)
        hu = va["held_up"].to_numpy().astype(bool)
        entry = {
            "n_features": len(cols), "is_lookahead_control": is_control,
            "val_all_bars_auc": base_auc,
            "auc_held_up_true": auc_or_none(y[hu], p[hu]),
            "auc_held_up_false": auc_or_none(y[~hu], p[~hu]),
            "auc_proba_vs_held_up_itself": auc_or_none(hu.astype(int), p),
        }
        log(f"  VAL 전체봉 AUC={base_auc} | held_up내부 T={entry['auc_held_up_true']} "
            f"F={entry['auc_held_up_false']} | proba↔held_up={entry['auc_proba_vs_held_up_itself']}")

        targets = [c for c in cols if c.startswith(("trig_", "trigd_", "n_triggers"))]
        if targets:
            imp = shuffle_importance(m, va, cols, targets, base_auc)
            entry["trigger_permutation_importance"] = imp
            for c, v in sorted(imp.items(), key=lambda kv: -kv[1]["delta_auc_mean"]):
                log(f"    {c:42s} delta_auc={v['delta_auc_mean']:+.5f} (±{v['delta_auc_std']:.5f}) "
                    f"fire_rate={v['fire_rate_val']:.4f}")
        results[fname] = entry

    base = results["F1_tier0_only"]["val_all_bars_auc"]
    for fname, e in results.items():
        e["delta_vs_F1_tier0_only"] = round(e["val_all_bars_auc"] - base, 4)

    report = {
        "signal": "v_rebound_every_bar_trigger_features", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {
            "screening_only": True, "model": "HistGradientBoostingClassifier proxy (not TabPFN)",
            "tabpfn_training_done": False, "economic_cost_gate_done": False,
            "live_code_changed": False, "holdout_touched": False, "oos_touched": False,
            "label_formula": "V0 unchanged (current live label_side())",
            "population": "ALL bars x both sides (every-bar scoring design)",
            "splits": {"TRAIN": f"< {TRAIN_END}", "VAL": f"{TRAIN_END} .. < {VAL_END}"},
        },
        "prior_analysis_corrections_applied": {
            "kalman_demarker_gap12_dedup": ("ETH label builder + live server both used raw un-deduped "
                                            "per-bar booleans; GAP=12 cluster_dedup applied in F3/F4"),
            "orthogonal_combo_kept": ("net-new candidate contribution is 0, but as a FEATURE it marks a "
                                      "distinct confluence pattern -- judged by permutation importance, "
                                      "not dropped a priori"),
            "local_extreme_excluded": ("confirmed lookahead: local_extreme[i] is only knowable at i+6 "
                                       "(0/210,630 mismatches) -- present ONLY in F4 as a control"),
        },
        "dedup_stats": dedup_stats,
        "population_sizes": {"train": int(len(tr)), "val": int(len(va)),
                             "train_base_rate": round(float(tr["label"].mean()), 4),
                             "val_base_rate": round(float(va["label"].mean()), 4)},
        "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
        "results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    log(f"total runtime: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
