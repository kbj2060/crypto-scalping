#!/usr/bin/env python3
"""[[eth_ilias1_zig075_standalone_always_direction_benchmark_20260818]]의 always-long/always-
short 벤치마크를 N=5 신규 랜덤시드(+ 원본 seed=260620)에 대해 반복 -- 2025-Q2/VAL에서만 real이
두 always를 압도하고 OOS 두 창은 스킬 증거가 없다는 패턴이 시드에 걸쳐 재현되는지 검증. 로직은
eval_eth_ilias1_zig075_always_direction_benchmark_20260818.py의 run_variant를 그대로 재사용,
bundle 경로만 시드별로 바꿔 반복."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_eth_ilias1_zig075_always_direction_benchmark_20260818 as bench  # noqa: E402

SEEDS = {
    "seed260620_original": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818/true_3head_tabm_bundle.pt",
    "seed121026": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818_seedvariant_121026/true_3head_tabm_bundle.pt",
    "seed337153": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818_seedvariant_337153/true_3head_tabm_bundle.pt",
    "seed390529": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818_seedvariant_390529/true_3head_tabm_bundle.pt",
    "seed640787": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818_seedvariant_640787/true_3head_tabm_bundle.pt",
    "seed794920": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818_seedvariant_794920/true_3head_tabm_bundle.pt",
}

all_results = {}
for seed_label, bundle_path in SEEDS.items():
    bench.ZIG_CFG["bundle"] = bundle_path
    bench.OUT_DIR = ROOT / f"tmp/causal_regen_20260516/eth_ilias1_zig075_always_direction_benchmark_20260818_{seed_label}"
    print(f"########## {seed_label} ({bundle_path.parent.name}) ##########", flush=True)
    real = bench.run_variant("real_direction", None)
    always_long = bench.run_variant("always_long", bench.omega.ACTION_LONG)
    always_short = bench.run_variant("always_short", bench.omega.ACTION_SHORT)
    all_results[seed_label] = {"real": real, "always_long": always_long, "always_short": always_short}

print()
print("=== FULL SUMMARY: does real beat BOTH always_long and always_short? (with_gate pnl) ===")
print(f"{'seed':22} {'window':8} {'real':>9} {'a_long':>9} {'a_short':>9}  beats_both")
for seed_label, res in all_results.items():
    for wname in bench.gate.WINDOW_DEFS:
        r = res["real"]["windows"][wname]["with_gate"]["pnl"]
        al = res["always_long"]["windows"][wname]["with_gate"]["pnl"]
        as_ = res["always_short"]["windows"][wname]["with_gate"]["pnl"]
        beats_both = r > al and r > as_
        print(f"{seed_label:22} {wname:8} {r:9.2f} {al:9.2f} {as_:9.2f}  {'YES' if beats_both else 'no'}")

out_path = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/2a48cbbb-f754-4dea-b53f-3262b328fe32/scratchpad/multiseed_always_benchmark_results.json")
serializable = {sl: {v: r["windows"] for v, r in res.items()} for sl, res in all_results.items()}
out_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"results={out_path}", flush=True)
