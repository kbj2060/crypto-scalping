#!/usr/bin/env python3
"""일리아스1의 zig075 단독 / h48qual 단독 PnL을 6창 Fresh-Forward로 측정 -- dual(포트폴리오,
h48qual+zig075 단일슬롯)은 이미 eval_eth_odyssey4_pinned102_freshforward_20260818.py로 계산된
"일리아스 1" 공식 수치(VAL-best threshold 0.40/0.80)가 있으므로 재사용하고, 여기서는 각
컴포넌트를 ev.BUNDLES에 하나만 남겨 greedy.greedy_replay가 그 컴포넌트만으로 단일슬롯을
운용하게 만든다 -- L0~L10 전체 파이프라인(진짜 TP/SL·사이징·exit_head 포함) 그대로, 로직
재구현 없음."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402

FULL_BUNDLES = {
    "h48qual": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_pinned102_20260818/true_3head_tabm_bundle.pt",
        "q_tag": "q040", "threshold": 0.40,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
        "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_h48qual_pinned102_q040_20260818/risk_sidecar.pkl",
        "exit_threshold": 0.95,
    },
    "zig075": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818/true_3head_tabm_bundle.pt",
        "q_tag": "q080", "threshold": 0.80,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
        "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_zig075_pinned102_q080_20260818/risk_sidecar.pkl",
        "exit_threshold": 0.95,
    },
}
ev._EXPECTED_ZERO_COLS = set()

RUNS = [
    ("zig075_standalone", {"zig075": FULL_BUNDLES["zig075"]}),
    ("h48qual_standalone", {"h48qual": FULL_BUNDLES["h48qual"]}),
]

summary = {}
for label, bundles in RUNS:
    print(f"=== stage={label} ===", flush=True)
    ev.BUNDLES = bundles
    ev.OUT_DIR = ROOT / f"tmp/causal_regen_20260516/eth_ilias1_{label}_freshforward_20260818"
    ev.main()
    report = json.loads((ev.OUT_DIR / "report.json").read_text(encoding="utf-8"))
    summary[label] = {w: {"tier": d["tier"], "with_gate": d["with_gate"]} for w, d in report["windows"].items()}

print()
print("=== SUMMARY (with_gate) ===")
for label, windows in summary.items():
    for w, d in windows.items():
        g = d["with_gate"]
        print(f"{label:20} {w:8} tier={d['tier']:12} pnl={g['pnl']:8.2f}% mdd={g['mdd']:8.2f}% trades={g['trades']:3d}")

out_path = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/2a48cbbb-f754-4dea-b53f-3262b328fe32/scratchpad/ilias1_standalone_summary.json")
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"summary={out_path}", flush=True)
