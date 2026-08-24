#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH Odyssey4 섀도우(live_eth_odyssey4_zig075_entry_veto_shadow_cleanroom_
20260816.py, systemd eth-odyssey4-shadow.service, 서버 상시 가동 중) 고유 신규축의 N=5 시드
강건성 평가.

=== 배경 ===
오늘(2026-08-19) ETH/BTC/SOL 라이브 dual(h48qual+zig075) 자체는 이미 N=5 랜덤시드로 검증 완료
(docs/experiments/eth_live_promotion_seed_robustness_3seed_20260819.md 등, 결과: 6창 중 4창 부호
플립 -- seed-robust하지 않음 CONFIRMED). 오디세이4 섀도우는 그 라이브 dual 위에 두 계층을 얹는다:
  (a) h48qual의 exit_head를 liveATR-relabel판(research_eth_omega461_exit_head_liveatr_relabel_
      20260813.py, 2026-08-13)으로 교체 -- encoder/direction_head/quality_head는 라이브와
      bit-identical, exit_head만 다름.
  (b) SustainedUptrendDetector(rolling 1주 dual_momentum>0 비율 > p90=0.8025793650793651,
      2025-Q1+Q2만으로 캘리브레이션, 자유변수 0개, 결정론적 규칙) 기반: 활성 시 (b-1) zig075 SHORT
      진입 스킵, (b-2) h48qual 보유 포지션의 exit 판정을 liveATR-relabel exit_head 대신 h48qual
      원본(재라벨 이전) exit_head로 전환.
(b)는 순수 규칙(신규 자유변수 0개)이라 시드축이 없다 -- 유일한 신규 "학습" 컴포넌트는 (a) h48qual
liveATR-relabel exit_head뿐이다. 이 스크립트는 그 축 하나만 5개 시드로 바꿔가며 평가한다:
260813(원본, encoder/direction/quality를 라이브 h48qual 번들에서 freeze하고 exit_head만 이
시드로 학습) + 497101020/912177061/29403054/458139929(같은 스크립트, 같은 encoder freeze, exit_
head 재학습만 시드 변경 -- scripts/eth_live_promotion_seed_robustness_odyssey4_exithead_seed_
variant_20260819.py 그대로 재사용해 오늘 세션 중 학습).

h48qual "guard" 번들(원본, 재라벨 이전 -- (b-2)가 감지기 ON일 때 전환해 쓰는 쪽)과 zig075는 둘 다
라이브 원본(260620) 하나로 고정한다. 이유: 오늘 세션 전체(ETH/BTC/SOL 라이브 dual N=5 검증)에서
지킨 원칙과 동일 -- risk sidecar를 frozen으로 고정해 인코더 시드효과를 명확히 격리했던 것처럼,
여기서도 "h48qual liveATR-relabel exit_head 시드효과"를 다른 두 축(guard 번들의 encoder 시드,
zig075의 encoder 시드 -- 둘 다 이미 오늘 별도로 N=5 검증된 축)과 뒤섞지 않기 위해서다.

=== 기존 백테스트 엔진 재사용 (재구현 금지 원칙) ===
research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py::greedy_replay_entry_veto
가 바로 그 엔진이다 -- Odyssey3의 h48qual 레짐인지형 exit 가드(research_eth_omega461_regime_aware_
exit_head_uptrend_guard_20260814.py::greedy_replay_regime_aware_exit_guard의 renamed copy)에
zig075 SHORT entry veto 한 블록만 추가한 것으로, 라이브 섀도우가 실제로 구현하는 두 계층 (a)(b-1)
(b-2)를 정확히 함께 재현하는 유일한 기존 엔진이다. 그대로 import해서 재사용한다 -- 이 파일은 그
엔진에 넣을 h48qual "liveatr"(default) 번들 경로를 시드별로 바꿔주는 컴포넌트 준비 함수
(prepare_regime_aware_components_seeded) 하나만 신규 추가한다. 이것도 research_eth_omega461_
regime_aware_exit_head_uptrend_guard_20260814.py::prepare_regime_aware_components의 renamed copy
(h48qual liveatr 번들 경로 파라미터화 한 줄 외 전부 동일) -- 이 리포지토리의 기존 관례(greedy_
replay_entry_veto 자신도 greedy_replay_regime_aware_exit_guard의 renamed copy)를 그대로 따른다.

fresh_forward_bar_by_bar=true (greedy_replay_entry_veto는 causal 단일 순방향 패스, i 증가, detector
는 순수 backward-looking rolling mean). trade_ledgers_used_as_input=false (렛저는 출력 전용).
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py /
trading_bot_modules/runtime_config.py / .env / live_eth_odyssey4_zig075_entry_veto_shadow_
cleanroom_20260816.py. Does NOT modify any imported module -- research_eth_omega461_zig075_short_
entry_veto_sustained_uptrend_20260814.py, research_eth_omega461_regime_aware_exit_head_uptrend_
guard_20260814.py, research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py,
eth_omega461_multiwindow_confirmation_gate_20260814.py, replay_omega4_6_1_greedy_router_20260706.py,
research_eth_omega461_exit_sweep_20260721.py, research_eth_omega461_live_sltp_mfe_width_20260813.py
are all imported and read only. No retraining, no GPU (DEVICE=cpu, matches every script in this
lineage).

사용: python scripts/eval_eth_odyssey4_shadow_exithead_seed_robustness_20260819.py
      [--seeds 260813_original,497101020,...]  (기본값 = 5개 전부)
"""
from __future__ import annotations

import argparse
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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as veto  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_shadow_exithead_seed_robustness_20260819"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05

LIVEATR_ROOT = ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500"
SEED_BUNDLES: dict[str, Path] = {
    "260813_original": portfolio.NEW_H48QUAL_BUNDLE,  # == LIVEATR_ROOT/h48qual/true_3head_tabm_bundle.pt
    "497101020": LIVEATR_ROOT.parent / f"{LIVEATR_ROOT.name}_seedvariant_497101020" / "h48qual" / "true_3head_tabm_bundle.pt",
    "912177061": LIVEATR_ROOT.parent / f"{LIVEATR_ROOT.name}_seedvariant_912177061" / "h48qual" / "true_3head_tabm_bundle.pt",
    "29403054": LIVEATR_ROOT.parent / f"{LIVEATR_ROOT.name}_seedvariant_29403054" / "h48qual" / "true_3head_tabm_bundle.pt",
    "458139929": LIVEATR_ROOT.parent / f"{LIVEATR_ROOT.name}_seedvariant_458139929" / "h48qual" / "true_3head_tabm_bundle.pt",
}

# G0 reference -- Odyssey4 contract G0 table verbatim (docs/model_contracts/
# odyssey4_eth_entry_veto_baseline_contract_20260814.md), (no_gate, with_gate) per window. Seed
# 260813_original's bundle IS portfolio.NEW_H48QUAL_BUNDLE, so reproducing this table exactly is a
# pure copy-fidelity check of prepare_regime_aware_components_seeded (below) against the unmodified
# guard.prepare_regime_aware_components + veto._attach_veto_mask combination it stands in for.
G0_ODYSSEY4 = {
    "2025q1": ({"pnl": 97.70, "mdd": -20.62, "trades": 28}, {"pnl": 44.98, "mdd": -20.62, "trades": 20}),
    "2025q2": ({"pnl": 65.83, "mdd": -14.17, "trades": 31}, {"pnl": 5.62, "mdd": -23.59, "trades": 19}),
    "2025q3": ({"pnl": -10.63, "mdd": -29.66, "trades": 23}, {"pnl": 20.17, "mdd": -19.72, "trades": 17}),
    "val": ({"pnl": 41.13, "mdd": -21.70, "trades": 35}, {"pnl": 77.31, "mdd": -21.76, "trades": 26}),
    "oos_q1": ({"pnl": 93.27, "mdd": -15.48, "trades": 24}, {"pnl": 67.25, "mdd": -15.48, "trades": 19}),
    "oos_q2": ({"pnl": -9.55, "mdd": -20.76, "trades": 13}, {"pnl": -12.69, "mdd": -20.76, "trades": 10}),
}


def log(msg: str) -> None:
    print(f"[odyssey4_shadow_exithead_seedrobust] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


# =====================================================================================================
# Renamed copy of guard.prepare_regime_aware_components -- the ONLY change is that the h48qual
# "liveatr" (default) config's bundle path is a parameter instead of hardcoded to gate.COMP_CFGS_
# ASYMMETRIC_TABM_LIVEATR["h48qual"]. The guard bundle (h48qual ORIGINAL, pre-relabel) and zig075
# are untouched -- both still sourced from gate's own single-reference configs, exactly as in the
# unmodified function, so this script's only free axis is h48qual_liveatr_bundle.
# =====================================================================================================
def prepare_regime_aware_components_seeded(
    window_name: str, windows: dict[str, Any], score_by_base: dict[Path, pd.DataFrame], threshold: float,
    out_dir: Path, device: torch.device, h48qual_liveatr_bundle: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    w = windows[window_name]
    split = gate.WINDOW_DEFS[window_name]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, out_dir)
    prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component

    h48qual_liveatr_cfg = portfolio._component_cfg("h48qual", bundle_override=h48qual_liveatr_bundle)
    h48qual_liveatr = prep(aligned_frame, aligned_paths["h48qual"], h48qual_liveatr_cfg, device)
    h48qual_original = prep(aligned_frame, aligned_paths["h48qual"], gate.COMP_CFGS_BASELINE_BOTH_ORIGINAL["h48qual"], device)
    zig075 = prep(aligned_frame, aligned_paths["zig075"], gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["zig075"], device)

    mask, n_nan = guard._detector_mask_for_frame(aligned_frame, window_name, score_by_base, threshold)

    h48qual_guarded = dict(h48qual_liveatr)
    h48qual_guarded["guard_base_np"] = h48qual_original["base_np"]
    h48qual_guarded["guard_exit_runtime"] = h48qual_original["exit_runtime"]
    h48qual_guarded["guard_pos_idx"] = h48qual_original["pos_idx"]
    h48qual_guarded["guard_exit_threshold"] = h48qual_original["exit_threshold"]
    h48qual_guarded["sustained_uptrend_mask"] = mask

    zig075_veto = dict(zig075)
    zig075_veto["short_entry_veto_mask"] = mask

    components = {"h48qual": h48qual_guarded, "zig075": zig075_veto}
    diag = {
        "n_bars": int(len(aligned_frame)), "detector_nan_bars": n_nan,
        "detector_active_bars": int(mask.sum()), "detector_active_frac": float(mask.mean()),
    }
    return aligned_frame, components, diag


def _write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="", help="comma-separated subset of SEED_BUNDLES keys; default = all 5")
    args = ap.parse_args()

    seed_labels = [s.strip() for s in args.seeds.split(",") if s.strip()] or list(SEED_BUNDLES.keys())
    unknown = [s for s in seed_labels if s not in SEED_BUNDLES]
    if unknown:
        print(f"unknown seed labels: {unknown}, known: {list(SEED_BUNDLES.keys())}", flush=True)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    greedy.PRIORITY = ("h48qual", "zig075")  # Odyssey4 contract's locked priority tuple (defensive
    # reset -- matches eth_omega461_multiwindow_confirmation_gate_20260814.py's own precedent of
    # explicitly resetting this shared-module global before use, in case another concurrent session
    # mutated it).

    report: dict[str, Any] = {
        "design": (
            "ETH Odyssey4 shadow's ONLY new-training axis (h48qual liveATR-relabel exit_head) "
            "evaluated across N=5 seeds via the existing combined entry-veto+exit-guard replay "
            "engine (research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814."
            "greedy_replay_entry_veto, reused unmodified). SustainedUptrendDetector, h48qual guard "
            "bundle (original pre-relabel), and zig075 all held fixed at their single live-reference "
            "configs -- isolates the seed axis to exactly the h48qual liveATR-relabel exit_head, "
            "matching this session's ETH/BTC/SOL live-dual N=5 methodology (frozen risk sidecar)."
        ),
        "seed_bundles": {k: str(v) for k, v in SEED_BUNDLES.items()},
        "seed_labels_evaluated_this_run": seed_labels,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }

    missing = {label: str(SEED_BUNDLES[label]) for label in seed_labels if not SEED_BUNDLES[label].exists()}
    if missing:
        report["stage_reached"] = "bundle_check"
        report["gate_pass"] = False
        report["missing_bundles"] = missing
        _write_report(report)
        log(f"stage=ABORT missing bundles: {missing}")
        return 1

    log("=== stage=load_windows ===")
    windows = gate.load_all_windows()

    log("=== stage=detector_build (reused from guard module) ===")
    score_by_base, robustness_thresholds, threshold = guard.build_detector()
    if abs(threshold - veto.EXPECTED_PRIMARY_THRESHOLD) > 1e-12:
        report["stage_reached"] = "detector_build"
        report["gate_pass"] = False
        report["note"] = f"recomputed p90 threshold {threshold!r} != locked value {veto.EXPECTED_PRIMARY_THRESHOLD!r} -- data drift, aborting."
        _write_report(report)
        log("stage=ABORT threshold drift")
        return 1
    log(f"  primary(p90)={threshold:.10f} == locked Odyssey3/4 value")
    report["detector"] = {"threshold_used": threshold, "matches_locked_value": True}

    log(f"=== stage=seed_sweep (seeds={seed_labels}) ===")
    results: dict[str, Any] = {}
    for seed_label in seed_labels:
        bundle_path = SEED_BUNDLES[seed_label]
        log(f"--- seed={seed_label} bundle={bundle_path} ---")
        seed_result: dict[str, Any] = {}
        for wname in gate.ALL_WINDOWS:
            aligned_frame, components, prep_diag = prepare_regime_aware_components_seeded(
                wname, windows, score_by_base, threshold, OUT_DIR, device, bundle_path,
            )
            diag, ledger = veto.greedy_replay_entry_veto(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
            ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_seed{seed_label}.csv"
            ledger.to_csv(ledger_path, index=False)
            no_gate = portfolio._ledger_metrics(ledger)
            with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
            seed_result[wname] = {
                "no_gate": no_gate, "with_gate": with_gate, "ledger_path": str(ledger_path),
                "detector_diag": prep_diag,
                "veto_bars": int(diag["veto_bars"]),
                "h48qual_guard_active_bars": int(diag["h48qual_guard_active_bars"]),
                "h48qual_guard_decision_differs_bars": int(diag["h48qual_guard_decision_differs_bars"]),
            }
            log(f"  {wname:8s} no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d}  "
                f"with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d}  "
                f"veto_bars={diag['veto_bars']:3d} guard_active_bars={diag['h48qual_guard_active_bars']:3d} "
                f"guard_decision_differs={diag['h48qual_guard_decision_differs_bars']:3d}")
        results[seed_label] = seed_result

    report["results_by_seed"] = results

    if "260813_original" in results:
        log("=== stage=g0_fidelity_check (seed260813_original vs Odyssey4 contract G0 table) ===")
        g0: dict[str, Any] = {}
        for wname in gate.ALL_WINDOWS:
            ng, wg = results["260813_original"][wname]["no_gate"], results["260813_original"][wname]["with_gate"]
            ref_ng, ref_wg = G0_ODYSSEY4[wname]
            ok_ng, ok_wg = _close(ng, ref_ng), _close(wg, ref_wg)
            g0[wname] = {"no_gate_match": ok_ng, "with_gate_match": ok_wg,
                         "actual_no_gate": ng, "reference_no_gate": ref_ng,
                         "actual_with_gate": wg, "reference_with_gate": ref_wg}
            log(f"  {wname:8s} no_gate_match={ok_ng} with_gate_match={ok_wg}")
        g0_pass = all(v["no_gate_match"] and v["with_gate_match"] for v in g0.values())
        report["g0_fidelity_seed260813_vs_odyssey4_contract"] = {"windows": g0, "pass": g0_pass}
        log(f"stage=g0_fidelity_result pass={g0_pass}")
        if not g0_pass:
            log("WARNING: G0 fidelity check FAILED -- prepare_regime_aware_components_seeded does not "
                "reproduce the locked Odyssey4 contract reference for the original bundle. Treat all "
                "seed results below as suspect until this is resolved.")

    log("=== stage=sign_agreement ===")
    sign_table: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        pnl_by_seed = {s: results[s][wname]["with_gate"]["pnl"] for s in seed_labels}
        signs = {s: (v > 0) for s, v in pnl_by_seed.items()}
        all_same_sign = len(set(signs.values())) == 1
        sign_table[wname] = {"with_gate_pnl_by_seed": pnl_by_seed, "all_same_sign": all_same_sign,
                             "tier": gate.WINDOW_DEFS[wname]["tier"]}
        log(f"  {wname:8s} with_gate_pnl_by_seed={pnl_by_seed} all_same_sign={all_same_sign}")
    report["sign_agreement"] = sign_table

    report["stage_reached"] = "done"
    report["gate_pass"] = True
    _write_report(report)
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
