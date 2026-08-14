#!/usr/bin/env python3
"""Odyssey2 follow-up to #1 (레짐별 quality_threshold, 2026-08-13, 기준선 미달로 종결).
사용자 제안(2026-08-14): exit_head 비대칭 채택과 동일 패턴 -- h48qual만 레짐별 threshold를
쓰고 zig075는 원본 전역 threshold(0.75, 전 레짐 불변)로 완전히 그대로 둔다. 원 실험의
"joint" 맵은 zig075의 bear/chop도 0.30/0.35로 같이 낮춰서 zig075 컴포넌트가 악화됐었다
(PnL+40.31%->+31.65%, MDD-13.07%->-13.38%) -- 이 비대칭 조합은 아직 테스트된 적 없다.

재학습 불필요, 원 스크립트(research_eth_omega461_regime_specific_quality_threshold_20260813.py)의
검증된 evaluate_component/portfolio_eval을 그대로 import해 재사용 -- 새 로직 없음, 새 조합만.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_regime_specific_quality_threshold_20260813 as base  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_regime_threshold_h48qual_only_asymmetric_20260814"


def log(msg: str) -> None:
    print(f"[h48only_regime_thr] {msg}", flush=True)


def run_val() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_frame = base.base_sweep.load_frame(base.base_sweep.VAL_START, base.base_sweep.VAL_END,
                                            base_csv=base.base_sweep.BASE_2025, wide24_csv=base.base_sweep.WIDE24_2025)
    prefix = base.base_sweep.omega._tabm_prefix(True)

    raw = {}
    for name, cfg in base.base_sweep.COMPONENTS.items():
        pred_csv = base.base_sweep.EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"
        raw[name] = base.load_raw(name, cfg, "VAL", pred_csv)

    # asymmetric map: h48qual gets the previously-found regime-specific VAL-best thresholds,
    # zig075 stays flat at its live global baseline (0.75) in EVERY regime -- fully untouched.
    asym_map = {
        "h48qual": {"bull": 0.30, "bear": 0.30, "chop": 0.35},
        "zig075": {r: 0.75 for r in base.REGIMES},
    }
    baseline_map = {name: {r: base.GLOBAL_BASELINE[name] for r in base.REGIMES} for name in base.base_sweep.COMPONENTS}

    log("stage=G0 baseline")
    prepped_baseline = {}
    for name, cfg in base.base_sweep.COMPONENTS.items():
        m, p = base.evaluate_component(name, cfg, val_frame, raw[name], prefix, baseline_map[name], oof=True)
        prepped_baseline[name] = p
        log(f"  baseline component={name} pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']}")
    no_gate_base, with_gate_base = base.portfolio_eval(val_frame, prepped_baseline)
    g0_ok = abs(no_gate_base["pnl"] - 36.82) < 0.5 and abs(no_gate_base["mdd"] - (-24.34)) < 0.5 and no_gate_base["trades"] == 29
    log(f"  G0 baseline portfolio no_gate={no_gate_base} -> {'PASS' if g0_ok else 'FAIL'}")
    if not g0_ok:
        raise RuntimeError("G0 self-consistency failed")

    log("stage=asymmetric (h48qual regime-specific, zig075 fully original)")
    prepped_asym = {}
    component_rows = {}
    for name, cfg in base.base_sweep.COMPONENTS.items():
        m, p = base.evaluate_component(name, cfg, val_frame, raw[name], prefix, asym_map[name], oof=True)
        component_rows[name] = m
        prepped_asym[name] = p
        log(f"  asym component={name} pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']}")
    # sanity: zig075 asym component MUST exactly equal its baseline (untouched thresholds)
    zig_matches_baseline = (abs(component_rows["zig075"]["pnl"] - 40.31) < 0.1)
    log(f"  sanity check zig075 asym == zig075 baseline: {zig_matches_baseline} (pnl={component_rows['zig075']['pnl']:.2f}%, expect ~40.31%)")

    no_gate_asym, with_gate_asym = base.portfolio_eval(val_frame, prepped_asym)
    log(f"  asym portfolio no_gate={no_gate_asym}")
    log(f"  asym portfolio with_gate={with_gate_asym}")

    beats = (no_gate_asym["pnl"] >= no_gate_base["pnl"] and no_gate_asym["mdd"] >= no_gate_base["mdd"] and
             with_gate_asym["pnl"] >= with_gate_base["pnl"] and with_gate_asym["mdd"] >= with_gate_base["mdd"])
    log(f"Gate (pnl+mdd nonworse, no_gate+with_gate): {'PASS' if beats else 'FAIL'}")

    result = {
        "g0_ok": g0_ok, "baseline_no_gate": no_gate_base, "baseline_with_gate": with_gate_base,
        "asym_map": asym_map, "component_h48qual": component_rows["h48qual"], "component_zig075": component_rows["zig075"],
        "zig075_matches_baseline_sanity": zig_matches_baseline,
        "asym_no_gate": no_gate_asym, "asym_with_gate": with_gate_asym, "gate_pass": bool(beats), "oos_run": False,
    }
    (OUT_DIR / "val_report.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def run_oos() -> dict:
    oos_frame = base.base_sweep.load_frame(base.base_sweep.OOS_START, base.base_sweep.OOS_END,
                                            base_csv=base.base_sweep.BASE_2026, wide24_csv=base.base_sweep.WIDE24_2026)
    prefix = base.base_sweep.omega._tabm_prefix(False)
    raw = {}
    for name, cfg in base.base_sweep.COMPONENTS.items():
        pred_csv = base.base_sweep.EXT_PRED_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        raw[name] = base.load_raw(name, cfg, "OOS", pred_csv)

    asym_map = {"h48qual": {"bull": 0.30, "bear": 0.30, "chop": 0.35}, "zig075": {r: 0.75 for r in base.REGIMES}}
    baseline_map = {name: {r: base.GLOBAL_BASELINE[name] for r in base.REGIMES} for name in base.base_sweep.COMPONENTS}

    prepped_base, prepped_asym = {}, {}
    for name, cfg in base.base_sweep.COMPONENTS.items():
        _, prepped_base[name] = base.evaluate_component(name, cfg, oos_frame, raw[name], prefix, baseline_map[name], oof=False)
        _, prepped_asym[name] = base.evaluate_component(name, cfg, oos_frame, raw[name], prefix, asym_map[name], oof=False)
    no_gate_base, with_gate_base = base.portfolio_eval(oos_frame, prepped_base)
    no_gate_asym, with_gate_asym = base.portfolio_eval(oos_frame, prepped_asym)
    survives = (no_gate_asym["pnl"] >= no_gate_base["pnl"] and no_gate_asym["mdd"] >= no_gate_base["mdd"])
    log(f"OOS baseline no_gate={no_gate_base}")
    log(f"OOS asym     no_gate={no_gate_asym}")
    log(f"OOS with_gate baseline={with_gate_base} asym={with_gate_asym}")
    log(f"OOS survives (no_gate pnl+mdd): {survives}")
    result = {"oos_baseline_no_gate": no_gate_base, "oos_asym_no_gate": no_gate_asym,
              "oos_baseline_with_gate": with_gate_base, "oos_asym_with_gate": with_gate_asym,
              "oos_survives": bool(survives)}
    existing = json.loads((OUT_DIR / "val_report.json").read_text())
    existing.update({"oos_run": True, **result})
    (OUT_DIR / "val_report.json").write_text(json.dumps(existing, indent=2, default=str))
    return result


if __name__ == "__main__":
    r = run_val()
    if r["gate_pass"]:
        log("VAL gate PASSED -- opening single-touch OOS")
        run_oos()
    else:
        log("VAL gate FAILED -- OOS NOT opened")
