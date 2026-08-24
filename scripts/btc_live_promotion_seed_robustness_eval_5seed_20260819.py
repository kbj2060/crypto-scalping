#!/usr/bin/env python3
"""'라이브로 승격된 BTC h48qual+swingtransition 자체'가 시드에 강건한가를 검증 -- Seed-Diversity
Ensemble Promotion Gate 적용 (CLAUDE.md), N=5(기존 라이브 번들 260620 + 신규 랜덤 4개:
750703416/160125165/626578270/179796523, random.SystemRandom().sample로 추출, 등간격 아님).

ETH의 대응 스크립트(eth_live_promotion_seed_robustness_eval_3seed_20260819.py, 다만 그쪽은 N=3
예비검증이었고 여기는 정식 N=5)와 달리, BTC는 dual-component(h48qual+zig075) 라우터가 아니라
단일 컴포넌트(h48qual, swingtransition 피쳐 추가)이므로 replay_omega4_6_1_greedy_router_20260706.
greedy_replay 대신, BTC 자신의 검증된 단일-컴포넌트 리플레이 엔진
train_eval_omega4_2_risk_sidecar_btc_20260708.py::_replay_with_risk를 그대로 재사용한다 -- 이건
실제 배포된 "HEADLINE" BTC 평가 스크립트(apply_final_scale_map_btc_freshforward_ext_
swingtransition_20260806.py, audit_live_models_fresh_forward_20260808.py가 "BTC h48qual+
swingtransition (promoted live)"의 HEADLINE으로 명시한 그 스크립트)가 실제로 쓰는 바로 그 함수다.
이 스크립트는 그 main()의 로직(_scaled_margin_leverage/_compound_metrics 등, 그대로 import해서
재사용, 재구현 아님)을 따르되:
  1) precomputed prediction CSV(원본 번들 전용 freshforward_ext 디렉토리) 대신, 각 시드 자신의
     seed_variant 학습이 이미 만들어둔 q055 예측 CSV(train/validation/oos_predictions_q055.csv)를
     사용 -- 신규 시드 4개는 전부 자기 자신의 bundle로 만든 프레시 추론 결과이므로 "저장된 과거
     ledger 재사용" 아니다. 기존 라이브 번들(seed260620_original)도 자기 자신의 원 학습 시점에
     만들어진 q055 예측 CSV를 그대로 쓴다(재추론 없음 -- 라이브 그 자체이므로).
  2) val/oos 2창이 아니라 6창(2025q1/q2/q3 컨텍스트 + val + oos_q1/q2, ETH의
     eth_omega461_multiwindow_confirmation_gate_20260814.py::WINDOW_DEFS와 동일 날짜 경계를
     BTC 데이터에 적용) 전부.
  3) risk sidecar는 5개 시드 전부 원본(260620) frozen 재사용 -- ETH/일리아스1과 동일한 명시적
     단순화(시드별 전용 sidecar 재학습은 범위 밖).

Fresh-Forward 준수: fresh_forward_bar_by_bar=true(각 창의 진입 결정은 그 시드 bundle이 학습
스크립트 자신의 predict_raw()로 이미 만들어둔, 해당 창 구간의 causal 예측이고, 청산은
_replay_with_risk의 단일 순방향 bar-by-bar 루프), trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import btc_live_promotion_seed_robustness_canonicaldata_20260819 as canon_wrap  # noqa: E402
import apply_final_scale_map_btc_freshforward_ext_swingtransition_20260806 as scale_ref  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega4_2_risk_sidecar_btc_20260708 as sidecar  # noqa: E402

parent_script = canon_wrap.parent_script
omega = canon_wrap.omega
DEVICE = torch.device("cpu")

LIVE_BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition"
SIDECAR_PKL = ROOT / "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260806_swingtransition/risk_sidecar.pkl"
DIRECTION_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/btc_zigzag_action_labels_20260708"
QUALITY_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_live_promotion_seed_robustness_20260819_eval"

Q_TAG = "q055"
LONG_SCALE, SHORT_SCALE = 0.5, 2.5
EXIT_THRESHOLD = 0.95
COST_MULT = 3.0
ATR_KWARGS = dict(atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12)

# Same 6 pre-registered calendar windows as eth_omega461_multiwindow_confirmation_gate_20260814.py
# ::WINDOW_DEFS, applied to BTC's own train_raw(2025-01-01..09-30)/val_raw(2025-10-01..12-31)/
# oos_raw(2026-01-01 onward, native coverage confirmed through at least 2026-07-12) splits.
WINDOW_DEFS = {
    "2025q1": {"start": "2025-01-01", "end": "2025-03-31 23:59:59", "split": "train", "oof": True},
    "2025q2": {"start": "2025-04-01", "end": "2025-06-30 23:59:59", "split": "train", "oof": True},
    "2025q3": {"start": "2025-07-01", "end": "2025-09-30 23:59:59", "split": "train", "oof": True},
    "val":    {"start": "2025-10-01", "end": "2025-12-31", "split": "validation", "oof": True},
    "oos_q1": {"start": "2026-01-01", "end": "2026-03-31", "split": "oos", "oof": False},
    "oos_q2": {"start": "2026-04-01", "end": "2026-06-30", "split": "oos", "oof": False},
}
_PRED_SPLIT_PREFIX = {"train": "train_predictions", "validation": "validation_predictions", "oos": "oos_predictions"}

NEW_SEEDS = [750703416, 160125165, 626578270, 179796523]
SEED_LABELS = ["260620_original", *[str(s) for s in NEW_SEEDS]]


def _bundle_dir_for(seed_label: str) -> Path:
    if seed_label == "260620_original":
        return LIVE_BUNDLE_DIR
    return ROOT / f"tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition_livepromo_seedvariant_{seed_label}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("stage=prepare_frames (shared across all seeds -- features/labels are seed-independent)", flush=True)
    frames = parent_script._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=DIRECTION_LABEL_DIR,
        quality_mode="quality_label_action",
        quality_label_dir=QUALITY_LABEL_DIR,
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    raw_by_split = {"train": frames["train_raw"], "validation": frames["val_raw"], "oos": frames["oos_raw"]}
    for split, df in raw_by_split.items():
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print(f"  {split}_raw rows={len(df)} range=[{df['timestamp'].min()}, {df['timestamp'].max()}]", flush=True)
    fee, slip = omega._load_fee_slip()

    with open(SIDECAR_PKL, "rb") as f:
        pkl = pickle.load(f)

    report: dict[str, Any] = {
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "risk_sizing_source": "frozen_risk_sidecar_reused_across_all_5_seeds -- same simplification as "
                               "the ETH N=3 preliminary (eth_live_promotion_seed_robustness_eval_3seed_20260819.py) "
                               "and Ilias1 N=5 axis: no per-seed risk sidecar retrained, explicit caveat.",
        "n_seeds": len(SEED_LABELS), "seed_labels": SEED_LABELS,
        "quality_threshold": 0.55, "q_tag": Q_TAG,
        "long_scale": LONG_SCALE, "short_scale": SHORT_SCALE, "exit_threshold": EXIT_THRESHOLD, "cost_mult": COST_MULT,
        "windows": {},
    }

    all_results: dict[str, dict[str, Any]] = {}
    for seed_label in SEED_LABELS:
        bundle_dir = _bundle_dir_for(seed_label)
        bundle_path = bundle_dir / "true_3head_tabm_bundle.pt"
        if not bundle_path.exists():
            raise RuntimeError(f"{seed_label}: bundle not found at {bundle_path} -- training not finished yet?")
        print(f"########## seed={seed_label} ##########", flush=True)
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        models = bundle["models"]
        base_cols = list(bundle["base_cols"])
        loaded = parent._load_payloads(models, device=DEVICE)

        windows: dict[str, Any] = {}
        for wname, wd in WINDOW_DEFS.items():
            frame_full = raw_by_split[wd["split"]].reset_index(drop=True)
            pred_path = bundle_dir / f"{_PRED_SPLIT_PREFIX[wd['split']]}_{Q_TAG}.csv"
            if not pred_path.exists():
                raise RuntimeError(f"{seed_label}/{wname}: prediction csv not found at {pred_path}")
            pred_full = pd.read_csv(pred_path)
            pred_full["timestamp"] = pd.to_datetime(pred_full["timestamp"])
            if len(pred_full) != len(frame_full) or not pred_full["timestamp"].equals(frame_full["timestamp"]):
                raise RuntimeError(f"{seed_label}/{wname}: prediction/frame timestamp mismatch (pred={len(pred_full)}, frame={len(frame_full)})")

            mask = (frame_full["timestamp"] >= wd["start"]) & (frame_full["timestamp"] <= wd["end"])
            frame = frame_full.loc[mask].reset_index(drop=True)
            pred = pred_full.loc[mask].reset_index(drop=True)
            if len(frame) == 0:
                raise RuntimeError(f"{seed_label}/{wname}: empty window slice")

            missing = sorted(set(base_cols) - set(frame.columns))
            if missing:
                raise RuntimeError(f"{seed_label}/{wname}: frame missing base_cols: {missing[:20]}")
            x = parent._base_input(frame, base_cols)
            dec_base = parent._to_decisions(pred, oof=bool(wd["oof"]))
            dec, _ = atr_eval._apply_atr_safety_sltp(dec_base, frame, **ATR_KWARGS)
            atr = atr_eval._atr_pct(frame, ATR_KWARGS["atr_window"])

            features = sidecar._risk_feature_frame(frame, pred, dec, base_cols, atr_pct=atr, feature_mode=pkl["risk_feature_mode"])
            x_all, _ = sidecar._feature_matrix(features, pkl["feature_columns"])
            side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
            score = (
                sidecar._predict_side_split_models(pkl["model"], x_all, side_all)
                if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=np.float64)
            )
            mapping = pkl["selected_mapping"]
            base_margin = sidecar._risk_margins(
                dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"],
                **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS},
            )
            base_leverage = (
                sidecar._risk_leverage(
                    dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"],
                    **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS},
                )
                if pkl["dynamic_leverage"] else np.ones(len(dec))
            )
            margin, leverage = scale_ref._scaled_margin_leverage(dec, base_margin, base_leverage, long_scale=LONG_SCALE, short_scale=SHORT_SCALE)

            _m, ledger = sidecar._replay_with_risk(
                frame, x, dec, loaded,
                risk_margin_fraction=margin, risk_leverage=leverage, exit_threshold=EXIT_THRESHOLD,
                fee=fee, slip=slip, cost_mult=COST_MULT, notional_scaled_sltp=False,
                exit_sizing_input_mode="actual", device=DEVICE,
            )
            metrics = scale_ref._compound_metrics(ledger)
            windows[wname] = {"rows": int(len(frame)), **metrics}
            print(f"seed={seed_label:20} window={wname:8} pnl={metrics['pnl']:9.2f}% mdd={metrics['mdd']:8.2f}% trades={metrics['trades']:3d}", flush=True)
            ledger_path = OUT_DIR / f"ledger_{seed_label}_{wname}.csv"
            ledger.to_csv(ledger_path, index=False)

        all_results[seed_label] = windows

    report["windows"] = all_results

    print(flush=True)
    print("=== 5-시드 부호일치 요약 (compound PnL, %) ===")
    header = f"{'window':10}" + "".join(f"{s:>18}" for s in SEED_LABELS) + "  sign_consistent"
    print(header)
    sign_flip_windows = []
    for wname in WINDOW_DEFS:
        pnls = [all_results[s][wname]["pnl"] for s in SEED_LABELS]
        signs = {p >= 0 for p in pnls}
        consistent = len(signs) == 1
        if not consistent:
            sign_flip_windows.append(wname)
        row = f"{wname:10}" + "".join(f"{p:18.2f}" for p in pnls) + f"  {'YES' if consistent else 'NO -- SIGN FLIP'}"
        print(row, flush=True)

    print(flush=True)
    print(f"sign_flip_windows={sign_flip_windows}")
    print("N=5, CLAUDE.md Seed-Diversity Ensemble Promotion Gate 기준(N>=5 진짜 랜덤시드) 충족.", flush=True)

    report["sign_flip_windows"] = sign_flip_windows
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
