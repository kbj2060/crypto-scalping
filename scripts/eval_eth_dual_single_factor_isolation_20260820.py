#!/usr/bin/env python3
"""ETH 라이브 dual(h48qual+zig075) N=5 시드검증(6창 중 4창 플립)의 원인이 h48qual 인코더
변동인지 zig075 인코더 변동인지 단일요인격리(one-factor-at-a-time)로 분리.

=== 배경/논리사슬 ===
- Test D(2026-08-19, 이미완료): h48qual+zig075 둘 다 완전 재학습(시드 5개 페어) -> 6창 중 4창 플립.
- Test A(2026-08-19, 이미완료, 오디세이4축): h48qual **exit_head만**(encoder는 라이브고정) 시드
  변경 + zig075 **완전고정** -> 6창 전부 부호일치.
- Test B(2026-08-20, 이미완료, 오디세이4축): h48qual exit_head만 시드변경(encoder고정) + zig075
  **완전 재학습**(인코더까지) 시드변경 -> 그래도 6창 전부 부호일치(예상밖 -- zig075 인코더가
  진짜로 바뀌어도 안 흔들림).
Test A/B 둘 다 h48qual의 encoder(direction/quality_head)는 항상 라이브 고정이었다는 공통점이
있다 -- 남은 유일한 미검증 축은 "h48qual의 encoder까지 완전히 바뀌면 어떻게 되는가"다.

이 스크립트는 그 축을 정확히 겨냥한다. 신규 학습 없음 -- 둘 다 오늘 이전 세션에서 이미 완료된
ETH 라이브 dual N=5 검증의 산출물(h48qual/zig075 각각 독립적으로 학습된 완전 3-head 부트스트랩
5개씩)을 재사용, PLAIN dual replay 엔진(veto/guard 없음 -- Test D를 만든 것과 정확히 같은 엔진,
eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818.py)으로 두 개의 단일요인 트라이얼을
동시에 만든다:

  Test C  (h48qual만 변경): h48qual in {260620,94046540,524707103,312069414,44751167}(전부 ETH
    라이브 N=5 검증에서 이미 학습된 완전 부트스트랩) x zig075 = 260620(라이브) 고정.
  Test B-plain (zig075만 변경, veto/guard 없는 엔진으로 재확인): zig075 in {260620,94046540,
    524707103,312069414,44751167} x h48qual = 260620(라이브) 고정.

둘 다 threshold(0.50/0.75)·risk sidecar(원본 frozen)는 라이브 그대로 -- 이번 세션 전체에서
지킨 원칙과 동일.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false. 신규 학습 없음, GPU 불필요.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))

import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402

ev._EXPECTED_ZERO_COLS = set()

ETH_LIVE_N5_SEEDS = ["260620", "94046540", "524707103", "312069414", "44751167"]


def _bundle_for(component: str, seed_label: str) -> Path:
    if seed_label == "260620":
        return sweep.COMPONENTS[component]["bundle"]
    out_suffix_map = {
        "h48qual": f"zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_livepromo_seedvariant_{seed_label}",
        "zig075": f"current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_livepromo_seedvariant_{seed_label}",
    }
    return ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_{out_suffix_map[component]}/true_3head_tabm_bundle.pt"


def _cfg_for(component: str, bundle: Path) -> dict:
    src = sweep.COMPONENTS[component]
    return {
        "bundle": bundle, "q_tag": src["q_tag"], "threshold": src["quality_threshold"],
        "atr_window": src["atr_window"], "tp_mult": src["tp_mult"], "sl_mult": src["sl_mult"],
        "min_tp": src["min_tp"], "min_sl": src["min_sl"], "max_tp": src["max_tp"], "max_sl": src["max_sl"],
        "sidecar_pkl": src["sidecar_pkl"], "exit_threshold": sweep.BASELINE_EXIT_THRESHOLD,
    }


def run_single_factor(varying: str, out_subdir: str, fixed_bundle_override: Path | None = None, fixed_label: str = "260620") -> dict:
    """varying='h48qual' -> Test C (h48qual varies, zig075 fixed at 260620).
    varying='zig075' -> Test B-plain (zig075 varies, h48qual fixed at 260620), or Test B-liveatr
    (zig075 varies, h48qual fixed at fixed_bundle_override=liveATR-relabel bundle) when
    fixed_bundle_override is given -- isolates whether Test B(오디세이4축)'s 6/6 stability came
    from the veto/guard mechanism or from this different h48qual baseline, independent of veto/guard
    (this script never adds veto/guard -- always the plain dual engine)."""
    fixed = "zig075" if varying == "h48qual" else "h48qual"
    fixed_bundle = fixed_bundle_override if fixed_bundle_override is not None else _bundle_for(fixed, "260620")
    all_results: dict[str, dict] = {}
    for seed_label in ETH_LIVE_N5_SEEDS:
        print(f"########## {varying}={seed_label} (fixed {fixed}={fixed_label}) ##########", flush=True)
        ev.BUNDLES = {
            varying: _cfg_for(varying, _bundle_for(varying, seed_label)),
            fixed: _cfg_for(fixed, fixed_bundle),
        }
        ev.OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{out_subdir}_{seed_label}"
        ev.main()
        report = json.loads((ev.OUT_DIR / "report.json").read_text(encoding="utf-8"))
        windows = {w: {"no_gate": d["no_gate"], "with_gate": d["with_gate"]} for w, d in report["windows"].items()}
        all_results[seed_label] = windows
        for w, wd in windows.items():
            wg = wd["with_gate"]
            print(f"{seed_label:12} {w:8} with_gate pnl={wg['pnl']:8.2f}% mdd={wg['mdd']:8.2f}% trades={wg['trades']:3d}", flush=True)
    return all_results


def _sign_table(all_results: dict, label: str) -> list:
    print()
    print(f"=== {label}: 부호일치 요약 (with_gate PnL, %) ===")
    header = f"{'window':10}" + "".join(f"{s:>14}" for s in ETH_LIVE_N5_SEEDS) + "  sign_consistent"
    print(header)
    sign_flip_windows = []
    for w in next(iter(all_results.values())).keys():
        pnls = [all_results[s][w]["with_gate"]["pnl"] for s in ETH_LIVE_N5_SEEDS]
        signs = {p >= 0 for p in pnls}
        consistent = len(signs) == 1
        if not consistent:
            sign_flip_windows.append(w)
        row = f"{w:10}" + "".join(f"{p:14.2f}" for p in pnls) + f"  {'YES' if consistent else 'NO -- FLIP'}"
        print(row, flush=True)
    print(f"sign_flip_windows={sign_flip_windows}")
    return sign_flip_windows


def main() -> int:
    # Test C / Test B-plain already completed in an earlier run of this script (2026-08-20) --
    # results archived in tmp/causal_regen_20260516/eth_dual_single_factor_isolation_20260820_
    # summary.json. This run adds ONLY the disambiguating third test.
    print("################ TEST B-liveatr: zig075 varies, h48qual FIXED at liveATR-relabel bundle (plain engine, no veto/guard) ################", flush=True)
    results_bl = run_single_factor(
        "zig075", "eth_dual_isolation_20260820_testBliveatr_zig075_varies",
        fixed_bundle_override=portfolio.NEW_H48QUAL_BUNDLE, fixed_label="260813_original(liveATR-relabel)",
    )
    flips_bl = _sign_table(results_bl, "Test B-liveatr (zig075 varies, h48qual=liveATR-relabel fixed)")

    summary = {"testB_liveatr_zig075_varies_h48qual_fixed_liveatr": {"results": results_bl, "sign_flip_windows": flips_bl}}
    out_path = ROOT / "tmp/causal_regen_20260516/eth_dual_single_factor_isolation_20260820_testBliveatr_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nresults={out_path}", flush=True)
    print(f"\nTest B-liveatr sign_flip_windows(zig075단독, h48qual=liveATR-relabel고정)={flips_bl}", flush=True)
    print("Test B-plain(h48qual=순수라이브고정, 2026-08-20 완료) sign_flip_windows=['2025q1','2025q2','val','oos_q1','oos_q2']", flush=True)
    print("Test B(오디세이4 veto+guard엔진, h48qual=liveATR-relabel고정, 2026-08-20 완료) sign_flip_windows=[] (6/6 stable)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
