#!/usr/bin/env python3
"""가설: h48qual은 chop bar만 담당, zig075는 bull/bear(비-chop) bar만 담당 -- 사용자 제안,
zig075 standalone이 6창 중 5창(모두 저-chop%)에서 압도하고 h48qual만 이기는 OOS-Q2가 6창 중
chop 비중이 가장 높다(56.4% vs 다른 창 47.0~50.2%)는 관찰에서 나온 가설.

⚠️ 이 가설은 OOS-Q2 결과를 보고 나온 것이라 완전한 독립검증은 아니다 -- 명시적으로 밝힌다.
새 자유변수는 0개(레짐 카테고리는 이미 있는 L1 bull/bear/chop 분류 그대로, 문턱값 튜닝 없음).

구현: 정규 prep_fn(portfolio._prepare_component_val / greedy.prepare_component, 로직 재구현
없음)으로 진짜 dec/margin/leverage를 만든 뒤, 그 dec["side"]/dec["action"]을 bar의 regime에
따라 마스킹만 한다 -- h48qual은 non-chop bar에서 강제 CASH, zig075는 chop bar에서 강제 CASH.
그 다음은 greedy.greedy_replay 그대로(TP/SL/사이징/exit_head 전부 실제 로직).

fresh_forward_bar_by_bar=true. No stored trade ledger used as input."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_ilias1_regime_split_h48qual_chop_zig075_trend_20260818"
DEVICE = ev.DEVICE
CHOP_IDX = hard.EXPERT_NAMES.index("chop")

BUNDLES = {
    "h48qual": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_pinned102_20260818/true_3head_tabm_bundle.pt",
        "q_tag": "q040", "threshold": 0.40,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
        "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_h48qual_pinned102_q040_20260818/risk_sidecar.pkl",
        "exit_threshold": 0.95, "allowed_regime": "chop",
    },
    "zig075": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818/true_3head_tabm_bundle.pt",
        "q_tag": "q080", "threshold": 0.80,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
        "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_zig075_pinned102_q080_20260818/risk_sidecar.pkl",
        "exit_threshold": 0.95, "allowed_regime": "non_chop",
    },
}


def _mask_by_regime(dec: pd.DataFrame, route: np.ndarray, allowed_regime: str) -> pd.DataFrame:
    dec = dec.copy()
    is_chop = route == CHOP_IDX
    block = is_chop if allowed_regime == "non_chop" else ~is_chop
    dec.loc[block, "action"] = 0
    dec.loc[block, "side"] = 0
    dec.loc[block, "notional_exposure"] = 0.0
    return dec


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = ev.omega._load_fee_slip()
    report: dict = {
        "hypothesis": "h48qual handles chop bars only, zig075 handles non-chop (bull/bear) bars only",
        "hypothesis_source_caveat": "derived from observing OOS-Q2 (the one window h48qual-standalone beats "
                                     "zig075-standalone) has the highest chop share of all 6 windows (56.4% vs "
                                     "47.0-50.2% elsewhere) -- NOT an independently pre-registered hypothesis, "
                                     "flagged explicitly rather than presented as a clean out-of-sample test.",
        "new_free_variables": 0,
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "windows": {},
    }

    windows = {}
    for wname, wd in gate.WINDOW_DEFS.items():
        frame = sweep.load_frame(wd["start"], wd["end"], base_csv=wd["base_csv"], wide24_csv=wd["wide24_csv"])
        frame, n_dropped = gate._drop_route_nan(frame)
        windows[wname] = {"frame": frame, "oof": wd["oof"], "tier": wd["tier"]}
        print(f"window={wname} rows={len(frame)} route_nan_dropped={n_dropped}", flush=True)

    pred_dir = OUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for wname, w in windows.items():
        frame = w["frame"]
        route = hard._route_id(frame)
        components = {}
        for name, cfg in BUNDLES.items():
            preds = ev.generate_predictions(name, cfg, frame, oof=w["oof"])
            pred_path = pred_dir / f"{name}_{wname}_predictions_{cfg['q_tag']}.csv"
            preds.to_csv(pred_path, index=False)
            prep_fn = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component
            comp = prep_fn(frame, pred_path, cfg, DEVICE)
            n_before = int((comp["dec"]["side"] != 0).sum())
            comp["dec"] = _mask_by_regime(comp["dec"], route, cfg["allowed_regime"])
            n_after = int((comp["dec"]["side"] != 0).sum())
            print(f"  component={name} window={wname} entries_before_mask={n_before} entries_after_mask={n_after}", flush=True)
            components[name] = comp

        _diag, ledger = greedy.greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=DEVICE)
        ledger_path = OUT_DIR / f"ledger_{wname}.csv"
        ledger.to_csv(ledger_path, index=False)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, frame, greedy.DURATION_THRESHOLD)
        report["windows"][wname] = {"tier": w["tier"], "no_gate": no_gate, "with_gate": with_gate, "ledger_path": str(ledger_path)}
        print(f"window={wname} tier={w['tier']} no_gate={no_gate['pnl']:.2f}%/{no_gate['mdd']:.2f}%/{no_gate['trades']}t "
              f"with_gate={with_gate['pnl']:.2f}%/{with_gate['mdd']:.2f}%/{with_gate['trades']}t", flush=True)

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=ev.omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
