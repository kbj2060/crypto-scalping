#!/usr/bin/env python3
"""h48qual/zig075 posfix(canonical-data) 재학습 번들의 Fresh-Forward 6창 평가.

CLAUDE.md Fresh-Forward Validation/OOS/Test Rule 준수:
- fresh_forward_bar_by_bar=true: 진입 결정은 프레임의 각 bar 시점에 확정된 피쳐만으로 그 번들
  자신이 직접 추론(_predict_payload)해서 만든다 -- 저장된 과거 ledger/prediction을 재사용하지
  않는다. 청산은 greedy.greedy_replay의 단일 순방향(bar 증가 순서) 루프로 결정.
- trade_ledgers_used_as_input=false / saved_parent_exit_timestamps_used=false /
  future_rows_used_for_entry=false.

6개 사전등록 창(eth_omega461_multiwindow_confirmation_gate_20260814.py의 WINDOW_DEFS와 동일 정의,
그러나 예측 CSV는 그 모듈의 gate.load_all_windows()가 하드코딩한 sweep.EXT_PRED_DIR(구 번들 예측)
대신, 이 두 posfix 번들 자신의 예측을 새로 생성해서 쓴다 -- gate 모듈 자체는 "entries hardcoded to
old frozen CSVs"라 전체교체 번들 평가에 못 씀(2026-08-18 조사 확인, docs/experiments/
eth_odyssey4_exit_head_liveatr_barrier_and_label_reaudit_20260818.md 참고).

예측 생성은 build_omega4_6_1_extended_parent_predictions_20260706.py와 동일한 메커니즘
(parent._base_input -> parent._predict_payload per expert -> parent._routed -> parent.
_prediction_output)을 그대로 재사용 -- 새로 발명한 게 아니라 이미 검증된 패턴.
"""
from __future__ import annotations

import json
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

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_posfix_canonicaldata_freshforward_20260818"
DEVICE = parent._device("cpu")

# Filled in once both canonical-data retrains complete and are pulled locally.
BUNDLES = {
    "h48qual": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_canonicaldata_20260818/true_3head_tabm_bundle.pt",
        "q_tag": "q050",
        "threshold": 0.50,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
        "sidecar_pkl": sweep.COMPONENTS["h48qual"]["sidecar_pkl"],  # no fresh sidecar trained for the posfix bundle -- explicit caveat, see report
        "exit_threshold": sweep.BASELINE_EXIT_THRESHOLD,
    },
    "zig075": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_canonicaldata_20260818/true_3head_tabm_bundle.pt",
        "q_tag": "q075",
        "threshold": 0.75,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
        "sidecar_pkl": sweep.COMPONENTS["zig075"]["sidecar_pkl"],
        "exit_threshold": sweep.BASELINE_EXIT_THRESHOLD,
    },
}


# cmamba/risk overlays have zero real 2025 source data (memory: omega_cmamba_risk_overlay_dead_
# code) -- the canonical-data training wrapper fed these bundles ZERO-FILLED placeholders for
# these 11 columns too (same mechanism, see train_eval_omega4_3head_parent72_eth_canonicaldata_
# posfix_20260818.py), so `_base_input`'s silent reindex+fillna(0.0) for these specific columns at
# eval time reproduces training exactly, not a mismatch. Any OTHER missing column would be a real
# problem and must still abort.
_EXPECTED_ZERO_COLS = {
    "regime3_stability_h6_score", "regime3_transition_h6_risk_prob", "regime3_transition_h6_risk_pred",
    "regime3_churn_h6_risk_score", "regime3_cmamba_h6_sidecar_bull_prob", "regime3_cmamba_h6_sidecar_bear_prob",
    "regime3_cmamba_h6_sidecar_chop_prob", "regime3_cmamba_h6_sidecar_class_id", "regime3_cmamba_h6_sidecar_confidence",
    "regime3_cmamba_h6_sidecar_transition_prob", "regime3_cmamba_h6_sidecar_stability_score",
}


def generate_predictions(name: str, cfg: dict, frame: pd.DataFrame, *, oof: bool) -> pd.DataFrame:
    """Fresh bar-by-bar-consistent entry predictions directly from `cfg['bundle']`, matching
    build_omega4_6_1_extended_parent_predictions_20260706.py's proven mechanism -- no stored
    ledger, no old bundle's predictions, genuine inference on this frame's own point-in-time
    features. Non-sequential per-row inference is fine here because entry decisions are
    stateless per bar (no position-history dependence) -- only the EXIT side needs a true
    sequential walk, which greedy.greedy_replay provides below."""
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols = bundle["base_cols"]
    models = bundle["models"]
    missing = set(base_cols) - set(frame.columns)
    unexpected_missing = sorted(missing - _EXPECTED_ZERO_COLS)
    if unexpected_missing:
        raise RuntimeError(f"{name}: frame missing {len(unexpected_missing)} UNEXPECTED base_cols (not the known cmamba/risk zero-placeholders): {unexpected_missing[:20]}")
    x = parent._base_input(frame, base_cols)
    route = hard._route_id(frame)
    preds = {expert: parent._predict_payload(models[expert], x, device=DEVICE) for expert in hard.EXPERT_NAMES}
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    # _predict_payload is a single, non-fold-specific forward pass -- "oof" has no separate
    # computation here, it is purely a column-naming convention _to_fixed_decisions dispatches on
    # downstream. Compute once with the "_oof" prefix, rename to the non-oof prefix when oof=False
    # -- exactly what build_omega4_6_1_extended_parent_predictions_20260706.py already does.
    out = parent._prediction_output(frame, direction, quality, threshold=float(cfg["threshold"]), prefix="omega1_regime3_expertdq_oof")
    if not oof:
        out = out.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in out.columns})
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "risk_sizing_source": "original_prefix_bundle_frozen_sidecar_reused_on_posfix_decisions -- "
                               "no risk sidecar has been trained yet for either posfix bundle, so "
                               "this eval reuses the ORIGINAL (pre-fix, June-trained) bundle's own "
                               "frozen sidecar model, fed this posfix bundle's fresh decisions/frame. "
                               "Margin/leverage ARE real, per-bar, dynamic sidecar outputs (verified: "
                               "non-constant, e.g. 0.278/0/0.31/... across bars) -- NOT a flat "
                               "BASE_TEMPLATE constant. But it is still an interim substitute, not "
                               "live-equivalent sizing for a bundle whose OWN sidecar doesn't exist "
                               "yet -- flagged explicitly, not silently passed off as fully live-parity.",
        "val_start_note": "sweep.VAL_START=2025-10-01, not CLAUDE.md's nominal 2025-09-01 -- "
                           "Sept 2025 is in-sample for this bundle's own parent training "
                           "(parent.SPLIT_TS=2025-10-01), pre-existing repo-wide convention, not "
                           "introduced by this eval.",
        "base_feature_count_note": "Both posfix bundles were retrained against CANONICAL data "
                                    "(data/splits/year_oos/training_features_*.csv, same lineage "
                                    "sweep.load_frame/this eval use) specifically so this Fresh-"
                                    "Forward eval's frame has full coverage of every bundle base_col "
                                    "-- verified via generate_predictions()'s explicit missing-column "
                                    "check, not assumed.",
        "windows": {},
    }

    windows: dict[str, dict[str, Any]] = {}
    for wname, wd in gate.WINDOW_DEFS.items():
        frame = sweep.load_frame(wd["start"], wd["end"], base_csv=wd["base_csv"], wide24_csv=wd["wide24_csv"])
        frame, n_dropped = gate._drop_route_nan(frame)
        windows[wname] = {"frame": frame, "oof": wd["oof"], "tier": wd["tier"], "split": wd["split"], "route_nan_dropped": n_dropped}
        print(f"window={wname} rows={len(frame)} range=[{frame['timestamp'].min()}, {frame['timestamp'].max()}] route_nan_dropped={n_dropped}", flush=True)

    pred_dir = OUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_paths: dict[str, dict[str, Path]] = {name: {} for name in BUNDLES}
    for wname, w in windows.items():
        for name, cfg in BUNDLES.items():
            preds = generate_predictions(name, cfg, w["frame"], oof=w["oof"])
            out_path = pred_dir / f"{name}_{wname}_predictions_{cfg['q_tag']}.csv"
            preds.to_csv(out_path, index=False)
            pred_paths[name][wname] = out_path
            nonzero = float((preds.filter(like="final_action").iloc[:, 0] != 0).mean())
            print(f"  predictions component={name} window={wname} rows={len(preds)} nonzero_action_rate={nonzero:.3f}", flush=True)

    for wname, w in windows.items():
        components: dict[str, Any] = {}
        for name, cfg in BUNDLES.items():
            pred = pd.read_csv(pred_paths[name][wname])
            pred["timestamp"] = pd.to_datetime(pred["timestamp"])
            if not pred["timestamp"].equals(w["frame"]["timestamp"]):
                raise RuntimeError(f"{name}/{wname}: fresh prediction/frame timestamp mismatch")
            greedy_cfg = dict(cfg)
            # greedy.prepare_component hardcodes _to_decisions(..., oof=False) (its own main() only
            # ever scores OOS predictions); portfolio._prepare_component_val is the byte-for-byte
            # oof=True counterpart research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py
            # already established for VAL-tier predictions -- same dispatch gate.run_portfolio_
            # variant uses (`portfolio._prepare_component_val if w["oof"] else greedy.prepare_
            # component`), not a new choice invented here.
            prep_fn = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component
            components[name] = prep_fn(w["frame"], pred_paths[name][wname], greedy_cfg, DEVICE)
        _diag, ledger = greedy.greedy_replay(w["frame"], components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=DEVICE)
        ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_posfix_canonicaldata.csv"
        ledger.to_csv(ledger_path, index=False)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, w["frame"], greedy.DURATION_THRESHOLD)
        report["windows"][wname] = {
            "tier": w["tier"], "frame_rows": int(len(w["frame"])), "route_nan_dropped": w["route_nan_dropped"],
            "no_gate": no_gate, "with_gate": with_gate, "ledger_path": str(ledger_path),
        }
        print(f"window={wname} tier={w['tier']} no_gate={no_gate['pnl']:.2f}%/{no_gate['mdd']:.2f}%/{no_gate['trades']}t "
              f"with_gate={with_gate['pnl']:.2f}%/{with_gate['mdd']:.2f}%/{with_gate['trades']}t", flush=True)

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
