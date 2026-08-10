"""Fresh-forward compliance audit of every live/shadow model's evidence chain (2026-08-08).

CLAUDE.md's Fresh-Forward rule requires a promotion report to state four flags explicitly:
  fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
  saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false
and forbids using a saved trade ledger / candidate-event ledger / saved parent exit timestamps as
INPUT to any number used for promotion, model selection, or a test claim.

This walks each live model's evidence chain -- parent bundle -> risk sidecar -> final scale map /
router -> any shadow extension -- and reports, per artifact: whether the four flags are present,
what they say, and whether the artifact's own method string suggests a ledger-based composition.
Absence of the flags is reported as UNDECLARED (not as a violation): it means the claim is not
verifiable against the project's own checklist, which is itself an audit finding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/live_models_oos_20260808/fresh_forward_audit.json"
FF = ["fresh_forward_bar_by_bar", "trade_ledgers_used_as_input",
      "saved_parent_exit_timestamps_used", "future_rows_used_for_entry"]
EXPECTED = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}

CHAIN = {
    "ETH Omega4.6.1 (live component, greedy router)": [
        ("risk sidecar h48qual",
         "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/report.json"),
        ("risk sidecar zig075",
         "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_zig075_q075_precomputed_20260630/report.json"),
        ("extended OOS build", "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/build_report.json"),
        ("HEADLINE greedy router (+145.34%)",
         "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_result.json"),
        ("event-flat fresh-forward variant",
         "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/event_flat_fresh_forward_report.json"),
    ],
    "SOL zig075 v2 (adaptive_squeeze)": [
        ("risk sidecar",
         "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_q070_20260720/report.json"),
        ("HEADLINE final scale map",
         "tmp/causal_regen_20260516/sol_final_scale_map_adaptive_squeeze_20260720/report.json"),
    ],
    "BTC h48qual+swingtransition (promoted live)": [
        ("risk sidecar",
         "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260806_swingtransition/report.json"),
        ("HEADLINE final scale map (freshforward_ext)",
         "tmp/causal_regen_20260516/btc_final_scale_map_swingtransition_freshforward_ext_20260806/report.json"),
    ],
    "BTC multislot shadow (N=3 x1.5)": [
        ("multislot gate report",
         "tmp/causal_regen_20260516/btc_swingtransition_multislot_20260807/report.json"),
    ],
    "BTC multislot + czz_trend risk overlay (new shadow)": [
        ("overlay risk evaluation", "tmp/btc_regime_sizing_risk_20260808/risk_evaluation.json"),
        ("shadow combination replay", "tmp/btc_multislot_shadow_regime_sizing_20260808/results.json"),
    ],
}

LEDGER_HINTS = ("ledger", "router", "rescale", "reconcile")


def audit_one(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    try:
        r = json.load(open(path))
    except Exception as e:
        return {"exists": True, "parse_error": str(e)}
    present = {k: r.get(k) for k in FF if k in r}
    if not present:
        verdict = "UNDECLARED (플래그 없음)"
    elif len(present) < 4:
        verdict = "PARTIAL (일부 플래그만)"
    elif all(present[k] == EXPECTED[k] for k in FF):
        verdict = "OK"
    else:
        verdict = "VIOLATION (플래그가 규칙과 불일치)"
    method = str(r.get("method") or r.get("model_id") or "")
    note = str(r.get("note") or "")
    hints = [h for h in LEDGER_HINTS if h in (method + " " + note).lower()]
    return {"exists": True, "flags": present or None, "verdict": verdict,
            "method": method[:140] or None, "note": note[:220] or None,
            "ledger_wording_hits": hints or None}


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {}
    for model, arts in CHAIN.items():
        report[model] = {}
        print(f"=== {model}")
        for label, rel in arts:
            a = audit_one(ROOT / rel)
            report[model][label] = {"path": rel, **a}
            if not a.get("exists"):
                print(f"  {label:44} MISSING")
                continue
            print(f"  {label:44} {a['verdict']:28} flags={a.get('flags')}"
                  + (f" ledger-wording={a['ledger_wording_hits']}" if a.get("ledger_wording_hits") else ""))
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
