#!/usr/bin/env python3
"""RESEARCH ONLY -- isolates entry-veto FROM exit-guard for the ETH Odyssey4 shadow mechanism,
to answer: does zig075 SHORT-during-uptrend entry-veto ALONE (no h48qual exit-guard) already
achieve the 6/6 seed-sign-stability seen with both mechanisms together?

=== 배경 ===
eval_eth_odyssey4_shadow_full_reseed_20260820.py는 zig075 5시드 + h48qual(liveATR-relabel)
5시드를 페어링해 entry-veto+exit-guard 엔진(greedy_replay_entry_veto, 둘 다 항상 동시에 켜짐)으로
평가해 6/6 부호일치를 얻었다. 같은 zig075 5시드를 h48qual(liveATR-relabel) 고정+PLAIN 엔진(veto도
guard도 없음)으로 돌리면 6/6 전부 플립한다(eval_eth_dual_single_factor_isolation_20260820.py의
Test B-liveatr). 즉 "메커니즘이 있으면 0/6, 없으면 6/6"까지는 확정됐지만, entry-veto와 exit-guard
둘 중 무엇이 실제 기여자인지는 분리검증된 적이 없다 -- greedy_replay_entry_veto가 항상 두 메커니즘을
동시에 포함하기 때문이다.

사용자 질문(2026-08-20): "BTC엔 detector/veto가 왜 이식이 안 되냐" -> 답변 과정에서 이 미해결
교란변수가 드러남(BTC 이식은 exit-guard용 h48qual 대체번들이 없어 entry-veto만 테스트했는데, ETH의
0/6 결과 자체가 entry-veto 단독 효과인지 exit-guard와의 결합 효과인지 불명확했음). 사용자가 이
분리검증을 명시적으로 요청("그렇게 해줘").

=== 방법: 기존 함수 재구성만, 신규 리플레이 로직 0줄 ===
greedy_replay_entry_veto의 exit-guard 분기는 `comp.get("sustained_uptrend_mask")`가 None이면
완전히 no-op이 되도록 이미 설계돼 있다(guard 모듈 자체의 docstring: "No mask attached ->
byte-identical to the unmodified greedy_replay's own behaviour" -- 직접 소스 확인, 이 스크립트가
새로 만든 동작이 아니라 기존 코드가 이미 지원하던 경로). 따라서:
  - eval_eth_odyssey4_shadow_full_reseed_20260820.py::prepare_regime_aware_components_dual_seeded의
    renamed copy에서 h48qual_guarded 빌드 단계(guard_base_np/guard_exit_runtime/guard_pos_idx/
    guard_exit_threshold/sustained_uptrend_mask 부착)를 전부 생략 -- h48qual_liveatr을 그대로 사용.
  - zig075엔 short_entry_veto_mask는 동일하게 부착.
  - greedy_replay_entry_veto 자체(research_eth_omega461_zig075_short_entry_veto_sustained_
    uptrend_20260814.py)는 단 한 줄도 수정하지 않고 그대로 import해서 호출.
  - 자체 검증: 매 (pair, window)에서 diag["h48qual_guard_active_bars"]가 0이어야 한다 -- guard가
    실제로 완전히 비활성화됐는지의 직접 증거(코드가 스스로 카운트하는 진단값이라 별도 assertion
    불필요, 리포트에 그대로 기록).

같은 5개 페어(zig075 N=5 x h48qual liveATR-relabel N=5, 인덱스 순), 같은 6창. 신규 학습 없음
(기존 번들 100% 재사용), GPU 불필요.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py / runtime_config.py / .env.
기존 모듈은 전부 읽기 전용 import."""
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

import eval_eth_odyssey4_shadow_full_reseed_20260820 as full_reseed  # noqa: E402

omega = full_reseed.omega
greedy = full_reseed.greedy
sweep = full_reseed.sweep
portfolio = full_reseed.portfolio
mfe_width = full_reseed.mfe_width
gate = full_reseed.gate
guard = full_reseed.guard
veto = full_reseed.veto

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_shadow_entry_veto_only_20260820"
DEVICE = full_reseed.DEVICE
H48QUAL_BUNDLES = full_reseed.H48QUAL_BUNDLES
ZIG075_BUNDLES = full_reseed.ZIG075_BUNDLES
PAIR_SEED_LABELS = full_reseed.PAIR_SEED_LABELS


def log(msg: str) -> None:
    print(f"[odyssey4_entry_veto_only] {msg}", flush=True)


def prepare_entry_veto_only_dual_seeded(
    window_name: str, windows: dict[str, Any], score_by_base, threshold: float,
    out_dir: Path, device: torch.device, h48qual_liveatr_bundle: Path, zig075_bundle: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """prepare_regime_aware_components_dual_seeded의 renamed copy -- 유일한 차이: h48qual에
    guard_*/sustained_uptrend_mask를 부착하는 5줄을 전부 생략(h48qual_liveatr을 그대로 사용).
    h48qual_original도 더 이상 필요 없어 준비하지 않음. zig075의 short_entry_veto_mask 부착은
    동일하게 유지."""
    w = windows[window_name]
    split = gate.WINDOW_DEFS[window_name]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, out_dir)
    prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component

    h48qual_liveatr_cfg = portfolio._component_cfg("h48qual", bundle_override=h48qual_liveatr_bundle)
    h48qual_liveatr = prep(aligned_frame, aligned_paths["h48qual"], h48qual_liveatr_cfg, device)
    zig075_cfg = portfolio._component_cfg("zig075", bundle_override=zig075_bundle)
    zig075 = prep(aligned_frame, aligned_paths["zig075"], zig075_cfg, device)

    mask, n_nan = guard._detector_mask_for_frame(aligned_frame, window_name, score_by_base, threshold)

    zig075_veto = dict(zig075)
    zig075_veto["short_entry_veto_mask"] = mask

    # h48qual_liveatr passed through UNMODIFIED -- no guard_* keys, no sustained_uptrend_mask.
    components = {"h48qual": dict(h48qual_liveatr), "zig075": zig075_veto}
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
            "Isolates entry-veto from exit-guard for the ETH Odyssey4 shadow mechanism. Same 5 "
            "zig075 x h48qual(liveATR-relabel) index-paired trials as eval_eth_odyssey4_shadow_"
            "full_reseed_20260820.py, same greedy_replay_entry_veto engine (unmodified), but h48qual "
            "components carry NO guard_*/sustained_uptrend_mask -- exit-guard is structurally "
            "disabled (comp.get('sustained_uptrend_mask') is None -> guard branch never fires, "
            "verified per-window via h48qual_guard_active_bars==0 in the diagnostics below)."
        ),
        "pair_labels": PAIR_SEED_LABELS,
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

    log("=== stage=pair_sweep (entry-veto only, guard structurally disabled) ===")
    results: dict[str, Any] = {}
    guard_never_fired = True
    for pair_label in H48QUAL_BUNDLES:
        h48qual_bundle = H48QUAL_BUNDLES[pair_label]
        zig075_bundle = ZIG075_BUNDLES[pair_label]
        log(f"--- {pair_label}: {PAIR_SEED_LABELS[pair_label]} ---")
        pair_result: dict[str, Any] = {}
        for wname in gate.ALL_WINDOWS:
            aligned_frame, components, prep_diag = prepare_entry_veto_only_dual_seeded(
                wname, windows, score_by_base, threshold, OUT_DIR, device, h48qual_bundle, zig075_bundle,
            )
            diag, ledger = veto.greedy_replay_entry_veto(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
            guard_active = int(diag["h48qual_guard_active_bars"])
            guard_never_fired = guard_never_fired and guard_active == 0
            ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_{pair_label}.csv"
            ledger.to_csv(ledger_path, index=False)
            no_gate = portfolio._ledger_metrics(ledger)
            with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
            pair_result[wname] = {
                "no_gate": no_gate, "with_gate": with_gate, "ledger_path": str(ledger_path),
                "detector_diag": prep_diag,
                "veto_bars": int(diag["veto_bars"]),
                "h48qual_guard_active_bars_expected_zero": guard_active,
            }
            log(f"  {wname:8s} no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d}  "
                f"with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d}  "
                f"veto_bars={diag['veto_bars']}  guard_active_bars={guard_active}")
        results[pair_label] = pair_result
    report["results_by_pair"] = results
    report["guard_structurally_disabled_everywhere"] = guard_never_fired
    log(f"guard_structurally_disabled_everywhere={guard_never_fired}")

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
    report["gate_pass"] = bool(guard_never_fired)
    _write_report(report)
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
