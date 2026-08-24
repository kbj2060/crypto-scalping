#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH Odyssey4 섀도우를 h48qual exit_head뿐 아니라 zig075 인코더까지 함께
페어링해서 재시드하면 시드플립이 재현되는지 검증.

=== 배경 ===
eval_eth_odyssey4_shadow_exithead_seed_robustness_20260819.py(N=5)는 h48qual liveATR-relabel
exit_head만 시드별로 바꾸고 zig075/guard는 라이브 원본(260620)에 고정해 6창 전부 부호일치를
얻었다. 그런데 그 창들에서 zig075가 트레이드의 69~83%를 차지한다는 게 사후분석으로 드러났다
(source_component_counts 직접확인) -- 즉 "부호가 안정적이었던" 진짜 이유는 exit_head가 무해해서가
아니라, 부호를 실제로 좌우하는 zig075가 애초에 시드 변수가 아니었기 때문일 가능성이 크다.

이 스크립트는 그 가설을 직접 검증한다: 오늘(2026-08-19) 이미 완료된 ETH 라이브 dual N=5 검증에서
학습된 zig075 인코더 5개(신규 학습 없음, 그대로 재사용)를, 같은 날 학습된 h48qual liveATR-relabel
exit_head 5개와 인덱스로 페어링(둘 다 독립적으로 뽑은 진짜 랜덤시드이므로 페어링 자체가 임의 조합을
만든다)해서 zig075/h48qual 둘 다 진짜로 다른 시드가 섞인 5개 트라이얼을 만든다. guard 번들(h48qual
원본, 감지기 ON일 때만 조회)과 SustainedUptrendDetector(자유변수 0개)는 원 스크립트와 동일하게
고정 -- 이번에 새로 여는 유일한 축은 zig075 인코더뿐이다.

페어링(인덱스 순, ETH 라이브 N=5 시드 -> 오디세이4 exit_head N=5 시드):
  260620(원본) <-> 260813_original(원본)  -- 원 세션의 격리축 테스트와 동일 조합, 재현성 확인용
  94046540     <-> 497101020
  524707103    <-> 912177061
  312069414    <-> 29403054
  44751167     <-> 458139929

=== 기존 백테스트 엔진 재사용 ===
eval_eth_odyssey4_shadow_exithead_seed_robustness_20260819.py의 prepare_regime_aware_components_
seeded를 그대로 복사하되 zig075_bundle 파라미터를 하나 추가한 것뿐 -- greedy_replay_entry_veto
(research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py) 자체는 미변경.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false. 신규 학습 없음(zig075/h48qual 둘 다 이미 학습된
번들 재사용), GPU 불필요.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py / runtime_config.py / .env /
live_eth_odyssey4_zig075_entry_veto_shadow_cleanroom_20260816.py. 기존 모듈은 전부 읽기 전용
import."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_shadow_full_reseed_20260820"
DEVICE = portfolio.DEVICE

LIVEATR_ROOT = ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500"
H48QUAL_BUNDLES: dict[str, Path] = {
    "pair1": portfolio.NEW_H48QUAL_BUNDLE,  # == LIVEATR_ROOT/h48qual/true_3head_tabm_bundle.pt (seed 260813_original)
    "pair2": LIVEATR_ROOT.parent / f"{LIVEATR_ROOT.name}_seedvariant_497101020" / "h48qual" / "true_3head_tabm_bundle.pt",
    "pair3": LIVEATR_ROOT.parent / f"{LIVEATR_ROOT.name}_seedvariant_912177061" / "h48qual" / "true_3head_tabm_bundle.pt",
    "pair4": LIVEATR_ROOT.parent / f"{LIVEATR_ROOT.name}_seedvariant_29403054" / "h48qual" / "true_3head_tabm_bundle.pt",
    "pair5": LIVEATR_ROOT.parent / f"{LIVEATR_ROOT.name}_seedvariant_458139929" / "h48qual" / "true_3head_tabm_bundle.pt",
}
ZIG075_ROOT_PREFIX = "omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k"
ZIG075_BUNDLES: dict[str, Path] = {
    "pair1": sweep.COMPONENTS["zig075"]["bundle"],  # live original, seed 260620
    "pair2": ROOT / f"tmp/causal_regen_20260516/{ZIG075_ROOT_PREFIX}_livepromo_seedvariant_94046540/true_3head_tabm_bundle.pt",
    "pair3": ROOT / f"tmp/causal_regen_20260516/{ZIG075_ROOT_PREFIX}_livepromo_seedvariant_524707103/true_3head_tabm_bundle.pt",
    "pair4": ROOT / f"tmp/causal_regen_20260516/{ZIG075_ROOT_PREFIX}_livepromo_seedvariant_312069414/true_3head_tabm_bundle.pt",
    "pair5": ROOT / f"tmp/causal_regen_20260516/{ZIG075_ROOT_PREFIX}_livepromo_seedvariant_44751167/true_3head_tabm_bundle.pt",
}
PAIR_SEED_LABELS = {
    "pair1": "zig075=260620(live) x h48qual=260813_original",
    "pair2": "zig075=94046540 x h48qual=497101020",
    "pair3": "zig075=524707103 x h48qual=912177061",
    "pair4": "zig075=312069414 x h48qual=29403054",
    "pair5": "zig075=44751167 x h48qual=458139929",
}


def log(msg: str) -> None:
    print(f"[odyssey4_shadow_full_reseed] {msg}", flush=True)


def prepare_regime_aware_components_dual_seeded(
    window_name: str, windows: dict[str, Any], score_by_base: dict[Path, pd.DataFrame], threshold: float,
    out_dir: Path, device: torch.device, h48qual_liveatr_bundle: Path, zig075_bundle: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """eval_eth_odyssey4_shadow_exithead_seed_robustness_20260819.py::prepare_regime_aware_
    components_seeded의 renamed copy -- zig075_bundle 파라미터 하나만 추가, 나머지 전부 동일
    (guard 번들은 여전히 h48qual 원본 고정, detector도 고정)."""
    w = windows[window_name]
    split = gate.WINDOW_DEFS[window_name]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, out_dir)
    prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component

    h48qual_liveatr_cfg = portfolio._component_cfg("h48qual", bundle_override=h48qual_liveatr_bundle)
    h48qual_liveatr = prep(aligned_frame, aligned_paths["h48qual"], h48qual_liveatr_cfg, device)
    h48qual_original = prep(aligned_frame, aligned_paths["h48qual"], gate.COMP_CFGS_BASELINE_BOTH_ORIGINAL["h48qual"], device)
    zig075_cfg = portfolio._component_cfg("zig075", bundle_override=zig075_bundle)
    zig075 = prep(aligned_frame, aligned_paths["zig075"], zig075_cfg, device)

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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    greedy.PRIORITY = ("h48qual", "zig075")

    missing = {}
    for label in H48QUAL_BUNDLES:
        if not H48QUAL_BUNDLES[label].exists():
            missing[f"h48qual/{label}"] = str(H48QUAL_BUNDLES[label])
        if not ZIG075_BUNDLES[label].exists():
            missing[f"zig075/{label}"] = str(ZIG075_BUNDLES[label])
    if missing:
        print(f"missing bundles: {missing}", flush=True)
        _write_report({"stage_reached": "bundle_check", "gate_pass": False, "missing_bundles": missing})
        return 1

    report: dict[str, Any] = {
        "design": (
            "Odyssey4 shadow re-evaluated with BOTH h48qual liveATR-relabel exit_head AND zig075 "
            "encoder paired-reseeded (5 index-paired trials from the two already-completed N=5 "
            "seed sweeps done earlier today), isolating whether zig075's own seed variance -- known "
            "from the ETH live-dual N=5 result (4/6 window sign flips) -- reproduces once zig075 is "
            "no longer held fixed. Guard bundle and SustainedUptrendDetector remain fixed (zero free "
            "parameters, as established)."
        ),
        "pair_labels": PAIR_SEED_LABELS,
        "h48qual_bundles": {k: str(v) for k, v in H48QUAL_BUNDLES.items()},
        "zig075_bundles": {k: str(v) for k, v in ZIG075_BUNDLES.items()},
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }

    log("=== stage=load_windows ===")
    windows = gate.load_all_windows()

    log("=== stage=detector_build ===")
    score_by_base, robustness_thresholds, threshold = guard.build_detector()
    if abs(threshold - veto.EXPECTED_PRIMARY_THRESHOLD) > 1e-12:
        _write_report({"stage_reached": "detector_build", "gate_pass": False, "note": "threshold drift"})
        log("stage=ABORT threshold drift")
        return 1
    report["detector"] = {"threshold_used": threshold, "matches_locked_value": True}

    log("=== stage=pair_sweep ===")
    results: dict[str, Any] = {}
    for pair_label in H48QUAL_BUNDLES:
        h48qual_bundle = H48QUAL_BUNDLES[pair_label]
        zig075_bundle = ZIG075_BUNDLES[pair_label]
        log(f"--- {pair_label}: {PAIR_SEED_LABELS[pair_label]} ---")
        pair_result: dict[str, Any] = {}
        for wname in gate.ALL_WINDOWS:
            aligned_frame, components, prep_diag = prepare_regime_aware_components_dual_seeded(
                wname, windows, score_by_base, threshold, OUT_DIR, device, h48qual_bundle, zig075_bundle,
            )
            diag, ledger = veto.greedy_replay_entry_veto(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
            ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_{pair_label}.csv"
            ledger.to_csv(ledger_path, index=False)
            no_gate = portfolio._ledger_metrics(ledger)
            with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
            pair_result[wname] = {
                "no_gate": no_gate, "with_gate": with_gate, "ledger_path": str(ledger_path),
                "detector_diag": prep_diag,
                "veto_bars": int(diag["veto_bars"]),
                "h48qual_guard_active_bars": int(diag["h48qual_guard_active_bars"]),
                "h48qual_guard_decision_differs_bars": int(diag["h48qual_guard_decision_differs_bars"]),
            }
            log(f"  {wname:8s} no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d}  "
                f"with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d}  "
                f"src={no_gate.get('source_component_counts')}")
        results[pair_label] = pair_result
    report["results_by_pair"] = results

    log("=== stage=sign_agreement ===")
    sign_table: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        pnl_by_pair = {p: results[p][wname]["with_gate"]["pnl"] for p in H48QUAL_BUNDLES}
        signs = {p: (v > 0) for p, v in pnl_by_pair.items()}
        all_same_sign = len(set(signs.values())) == 1
        sign_table[wname] = {"with_gate_pnl_by_pair": pnl_by_pair, "all_same_sign": all_same_sign,
                             "tier": gate.WINDOW_DEFS[wname]["tier"]}
        log(f"  {wname:8s} pnl_by_pair={pnl_by_pair} all_same_sign={all_same_sign}")
    report["sign_agreement"] = sign_table

    report["stage_reached"] = "done"
    report["gate_pass"] = True
    _write_report(report)
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
