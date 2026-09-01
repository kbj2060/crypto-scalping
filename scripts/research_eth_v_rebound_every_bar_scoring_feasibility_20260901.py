#!/usr/bin/env python3
"""V_REBOUND "매 봉 스코어링" 재설계 -- 실현가능성 결정테스트 (GBM 프록시, CPU only).

사용자 제안(2026-09-01): local_extreme/증거신호는 **학습 라벨 생성**에만 쓰고, 신호 체계는
"트리거가 발동해야 점수를 낸다"에서 "매 5분봉마다 모델이 직접 판단한다"로 바꾼다.

## 왜 이게 held_up 얽힘의 해법이 되는가

held_up 얽힘(docs/homer/v_rebound_open_issues_20260901.md #6)의 근본 원인은 라벨 수식이 아니라
**선택(selection)**이다: local_extreme은 정의상 low[i]==min(low[i-6:i+7])이라 held_up
(low[i+1:i+7]>=low[i])을 100% 보장하고, 이 held_up은 라벨의 fast_move가 커지기 위한 선행조건이다.
즉 트리거가 "이미 절반쯤 정답인 봉"만 골라서 모델에 공급해왔다. 게이트를 없애고 매 봉을
스코어링하면 이 공짜 크레딧이 사라진다 -- 모델이 held_up 여부 자체를 스스로 예측해야 한다.
라벨의 의미("진짜 저점에서의 급반등")는 그대로 보존되므로, 평가창을 뒤로 미루는 shift 변형
(research_eth_v_rebound_label_redesign_variant_screen_20260901.py의 V2~V4, 발생률이 22%->4%로
붕괴하며 문제가 "이미 반등한 뒤 또 반등할까"로 바뀜)보다 교환조건이 낫다.

## 이 스크립트가 답하는 결정적 질문

후보풀(9트리거 발동봉, 전체의 23.45%)에서 학습한 모델을 **전체 봉**에 적용해도 스킬이 남는가?

이 저장소는 정확히 이 패턴에 두 번 당한 전례가 있다 -- orthogonal_combo의 kept-only 헤드라인
과대평가(eth_project_homer_evidence_signal_dl_migration_20260830), trend_continuation의
"확실한 사건만 학습"(eth_trend_continuation_at_evidence_signal_fires_20260831 후속5): 둘 다
필터링된 부분집합에서는 좋아 보이다가 전체 모집단 평가에서 소멸했다. 그래서 반드시
**두 모집단 모두에서** 평가한다.

  A  : TRAIN 후보봉만 학습     (현행 배포 구성의 프록시)
  B  : TRAIN 전체봉 학습        (제안 설계)
  B_sub: TRAIN 전체봉을 A와 같은 표본수로 서브샘플 학습 (표본수 효과 분리용 대조군)

각 모델을 VAL 후보봉 / VAL 전체봉 두 모집단에서 평가한다. 추가로 held_up 층화 AUC도 계산해
"매 봉 스코어링이 실제로 얽힘 크레딧을 제거하는가"를 직접 확인한다.

GBM(HistGradientBoosting)은 이 저장소의 표준 저비용 프록시다 -- TabPFN 절대수치를 재현하려는
게 아니라 **두 모집단 간 상대비교**가 목적이므로 프록시로 충분하다(통과 시에만 GPU/TabPFN).

⚠️ 진단 전용: 라이브 코드 변경 없음, OOS/HOLDOUT 미터치(TRAIN < 2025-09-01, VAL 2025-09-01~
2025-12-31, CLAUDE.md Fresh-Forward 계약). 라벨식 자체는 현행 V0 그대로(변경 없음).

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py
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

TIER0_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_v_rebound_features_tier0_20260829.py"
_spec = importlib.util.spec_from_file_location("tier0_feat_everybar_20260901", TIER0_SCRIPT)
_tier0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tier0)

VARIANT_SCRIPT = ROOT / "scripts/research_eth_v_rebound_label_redesign_variant_screen_20260901.py"
_vspec = importlib.util.spec_from_file_location("label_variants_20260901", VARIANT_SCRIPT)
_vs = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(_vs)

ETH_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_CSV = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
OUT_JSON = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/every_bar_scoring_feasibility_report.json"

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
START = pd.Timestamp("2024-01-01", tz="UTC")  # canonical START convention
SEED = 20260901
W = 6

FEATURE_COLUMNS = _tier0.FEATURE_COLUMNS + ["rsi"]
NAMED8 = _vs.NAMED8
ALL9 = _vs.ALL9


def log(msg: str) -> None:
    print(f"[every_bar_feasibility] {msg}", flush=True)


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def build_all_bar_frame() -> pd.DataFrame:
    """Per-bar indicator frame + the 3 side-dependent columns' raw inputs. Formulas copied verbatim
    from build_eth_5m_v_rebound_multitrigger_features_tier0_20260831.py::main() so the every-bar
    features are byte-identical in construction to the deployed candidate features."""
    sweep_impl = _tier0.load_sweep_impl()
    indicator_frame = _tier0.build_indicator_frame(sweep_impl)
    sweep_frame = sweep_impl.add_causal_columns(sweep_impl.load_5m(_tier0.SOURCE))
    assert len(indicator_frame) == len(sweep_frame)
    assert (indicator_frame["timestamp"].to_numpy() == sweep_frame["timestamp"].to_numpy()).all()

    f = indicator_frame.copy()
    f["sweep_level_low"] = sweep_frame["sweep_level_low"]
    f["sweep_level_high"] = sweep_frame["sweep_level_high"]
    f["atr"] = sweep_frame["atr"]
    f["atr_percentile_864"] = f["atr"].rolling(864, min_periods=864).rank(pct=True)
    f["range_width_pct"] = (f["sweep_level_high"] - f["sweep_level_low"]) / f["close"]
    f["hour_utc"] = f["timestamp"].dt.hour
    f["weekday"] = f["timestamp"].dt.weekday
    f["rsi"] = rsi_wilder(f["close"])
    return f


def build_long_frame() -> pd.DataFrame:
    """One row per (bar, side): 23 Tier0 features + V0 label + is_candidate + held_up."""
    log("building per-bar indicator frame (whole history)...")
    feat = build_all_bar_frame()

    log("computing triggers (compute_signals + local_extreme) on the same frame...")
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

    log("computing V0 (current live) labels on ALL bars, both sides...")
    st_b = _vs.label_variant(sig, is_down=True, anchor_mode="wick", shift=0)
    st_t = _vs.label_variant(sig, is_down=False, anchor_mode="wick", shift=0)

    fwd_low_min = _vs.fwd_window(low, 1, W, "min")
    fwd_high_max = _vs.fwd_window(high, 1, W, "max")
    held_up_b = fwd_low_min >= low
    held_up_t = fwd_high_max <= high

    cand_b = np.any([sig[f"bottom_{nm}"].fillna(False).to_numpy() for nm in ALL9], axis=0)
    cand_t = np.any([sig[f"top_{nm}"].fillna(False).to_numpy() for nm in ALL9], axis=0)

    trig_frame = pd.DataFrame({
        "timestamp": sig["timestamp"],
        "st_b": st_b, "st_t": st_t,
        "held_up_b": held_up_b, "held_up_t": held_up_t,
        "cand_b": cand_b, "cand_t": cand_t,
        "le_b": lo_flag, "le_t": hi_flag,
    })

    merged = trig_frame.merge(feat, on="timestamp", how="inner", suffixes=("", "_f"))
    log(f"merged per-bar frame: {len(merged)} bars")

    rows = []
    for side, is_down in (("bottom", True), ("top", False)):
        d = merged.copy()
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

        sub["status"] = st.to_numpy()
        sub["held_up"] = (d["held_up_b"] if is_down else d["held_up_t"]).to_numpy()
        sub["is_candidate"] = (d["cand_b"] if is_down else d["cand_t"]).to_numpy()
        sub["is_local_extreme"] = (d["le_b"] if is_down else d["le_t"]).to_numpy()
        rows.append(sub)

    long = pd.concat(rows, ignore_index=True)
    long = long.loc[long["timestamp"] >= START].reset_index(drop=True)
    long = long.loc[long["timestamp"] < VAL_END].reset_index(drop=True)
    long["label"] = np.where(long["status"] == "v_rebound", 1.0,
                     np.where(long["status"] == "chop", 0.0, np.nan))
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN", "VAL")
    return long


def fit_gbm(X: pd.DataFrame, y: np.ndarray, seed: int = SEED) -> HistGradientBoostingClassifier:
    m = HistGradientBoostingClassifier(random_state=seed, max_iter=300, early_stopping=True,
                                        validation_fraction=0.15)
    m.fit(X, y)
    return m


def auc_or_none(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return round(float(roc_auc_score(y, p)), 4)


def evaluate(model, val: pd.DataFrame, tag: str) -> dict:
    X = val[FEATURE_COLUMNS]
    p = model.predict_proba(X)[:, 1]
    y = val["label"].to_numpy()
    out = {"population_n": int(len(val)), "base_rate": round(float(y.mean()), 4),
           "auc": auc_or_none(y, p)}
    hu = val["held_up"].to_numpy().astype(bool)
    for name, mask in (("held_up_true", hu), ("held_up_false", ~hu)):
        if mask.sum() > 10:
            out[f"auc_{name}"] = auc_or_none(y[mask], p[mask])
            out[f"n_{name}"] = int(mask.sum())
            out[f"base_rate_{name}"] = round(float(y[mask].mean()), 4)
    out["auc_proba_vs_held_up_itself"] = auc_or_none(hu.astype(int), p)
    log(f"  [{tag}] n={out['population_n']:7d} base={out['base_rate']:.4f} AUC={out['auc']} "
        f"| held_up내부 T={out.get('auc_held_up_true')} F={out.get('auc_held_up_false')} "
        f"| proba↔held_up={out['auc_proba_vs_held_up_itself']}")
    return out


def main() -> int:
    t0 = time.time()
    long = build_long_frame()

    labeled = long.loc[long["label"].notna()].copy()
    labeled = labeled.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    log(f"labeled+clean rows: {len(labeled)} (of {len(long)} bar-sides)")

    tr = labeled.loc[labeled["split"] == "TRAIN"]
    va = labeled.loc[labeled["split"] == "VAL"]
    tr_cand = tr.loc[tr["is_candidate"]]
    va_cand = va.loc[va["is_candidate"]]
    log(f"TRAIN all={len(tr)} cand={len(tr_cand)} | VAL all={len(va)} cand={len(va_cand)}")
    log(f"base rates -- TRAIN all={tr['label'].mean():.4f} cand={tr_cand['label'].mean():.4f} "
        f"| VAL all={va['label'].mean():.4f} cand={va_cand['label'].mean():.4f}")

    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(len(tr), size=min(len(tr_cand), len(tr)), replace=False)
    tr_sub = tr.iloc[np.sort(sub_idx)]

    models = {}
    log("=== fitting GBM proxies ===")
    log(f"  A     : TRAIN candidates only  (n={len(tr_cand)})")
    models["A_train_on_candidates"] = fit_gbm(tr_cand[FEATURE_COLUMNS], tr_cand["label"].to_numpy())
    log(f"  B     : TRAIN all bars         (n={len(tr)})")
    models["B_train_on_all_bars"] = fit_gbm(tr[FEATURE_COLUMNS], tr["label"].to_numpy())
    log(f"  B_sub : TRAIN all bars, subsampled to A's size (n={len(tr_sub)})")
    models["B_sub_all_bars_subsampled"] = fit_gbm(tr_sub[FEATURE_COLUMNS], tr_sub["label"].to_numpy())

    results = {}
    for mname, model in models.items():
        log(f"=== evaluating {mname} ===")
        results[mname] = {
            "eval_on_VAL_candidates": evaluate(model, va_cand, f"{mname} -> VAL cand"),
            "eval_on_VAL_all_bars": evaluate(model, va, f"{mname} -> VAL all"),
        }

    report = {
        "signal": "v_rebound_every_bar_scoring_feasibility", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {
            "screening_only": True, "model": "HistGradientBoostingClassifier proxy (not TabPFN)",
            "tabpfn_training_done": False, "economic_cost_gate_done": False,
            "live_code_changed": False, "holdout_touched": False, "oos_touched": False,
            "label_formula": "V0 unchanged (current live label_side())",
            "splits": {"TRAIN": f"< {TRAIN_END}", "VAL": f"{TRAIN_END} .. < {VAL_END}"},
            "purpose": ("Decide whether the user's proposed redesign -- keep local_extreme/evidence "
                        "signals for LABEL generation but score EVERY 5m bar instead of gating on a "
                        "trigger fire -- retains model skill on the full population."),
        },
        "population_sizes": {
            "train_all": int(len(tr)), "train_candidates": int(len(tr_cand)),
            "train_subsampled": int(len(tr_sub)),
            "val_all": int(len(va)), "val_candidates": int(len(va_cand)),
            "train_base_rate_all": round(float(tr["label"].mean()), 4),
            "train_base_rate_candidates": round(float(tr_cand["label"].mean()), 4),
            "val_base_rate_all": round(float(va["label"].mean()), 4),
            "val_base_rate_candidates": round(float(va_cand["label"].mean()), 4),
        },
        "feature_columns": FEATURE_COLUMNS,
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
