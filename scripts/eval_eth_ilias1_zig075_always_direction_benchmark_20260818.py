#!/usr/bin/env python3
"""zig075 단독의 강한 replay 성과(VAL +192.21% 등)가 진짜 방향예측 스킬인지, 아니면 quality
게이트의 진입 타이밍 선택 + TP:SL 비대칭 구조만으로 방향과 무관하게 나오는 결과인지 검증 --
[[h48qual_standalone_replay_invalid]]("검증된 역할≠편향없음, 항상 max(always) 벤치마크 대조")
가 이미 h48qual에 대해 확립한 방법론을 zig075에 그대로 적용. Odyssey1이 ungated direction_head
를 always-short와 비교했던 것과 같은 계열의 대조(단, 여기는 quality게이트로 이미 선별된 진입
타이밍은 유지하고 방향만 고정 -- "언제 걸지는 그대로, 어느 쪽으로 걸지"만 always-long/short로
바꿔서 방향예측의 한계기여를 분리).

방법: zig075의 실제 fresh 예측에서 final_action이 CASH(0)가 아닌 bar는 그대로 진입 타이밍으로
쓰되, final_action 값 자체를 전부 LONG(1) 또는 전부 SHORT(2)로 강제 치환한 예측 CSV를 만들어
같은 prepare_component/greedy_replay(진짜 TP/SL·사이징·exit_head 전부 포함)로 리플레이.
quality_for_action/dir_confidence 등 사이드카 피쳐는 원본 모델이 실제로 계산한 값 그대로
남김(그 bar에 대한 모델의 확신도 자체는 유지, 방향 콜만 고정) -- 로직 재구현 없음, 기존
prepare_component/greedy_replay 그대로 재사용."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_ilias1_zig075_always_direction_benchmark_20260818"
DEVICE = ev.DEVICE
ZIG_CFG = {
    "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818/true_3head_tabm_bundle.pt",
    "q_tag": "q080", "threshold": 0.80,
    "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
    "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
    "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_zig075_pinned102_q080_20260818/risk_sidecar.pkl",
    "exit_threshold": 0.95,
}


def _force_action(pred: pd.DataFrame, prefix: str, forced_action: int) -> pd.DataFrame:
    out = pred.copy()
    col = f"{prefix}final_action"
    nonzero = out[col] != omega.ACTION_CASH
    out.loc[nonzero, col] = forced_action
    return out


def run_variant(label: str, forced_action: int | None) -> dict:
    out_dir = OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()
    report = {"windows": {}}
    for wname, wd in gate.WINDOW_DEFS.items():
        frame = sweep.load_frame(wd["start"], wd["end"], base_csv=wd["base_csv"], wide24_csv=wd["wide24_csv"])
        frame, _n = gate._drop_route_nan(frame)
        preds = ev.generate_predictions("zig075", ZIG_CFG, frame, oof=wd["oof"])
        prefix = "omega1_regime3_expertdq_oof_" if wd["oof"] else "omega1_regime3_expertdq_"
        if forced_action is not None:
            preds = _force_action(preds, prefix, forced_action)
        pred_path = out_dir / f"zig075_{wname}_predictions.csv"
        preds.to_csv(pred_path, index=False)

        prep_fn = portfolio._prepare_component_val if wd["oof"] else greedy.prepare_component
        comp = prep_fn(frame, pred_path, ZIG_CFG, DEVICE)
        components = {"zig075": comp}
        _diag, ledger = greedy.greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=DEVICE)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, frame, greedy.DURATION_THRESHOLD)
        report["windows"][wname] = {"tier": wd["tier"], "no_gate": no_gate, "with_gate": with_gate}
        print(f"{label:16} {wname:8} tier={wd['tier']:12} with_gate pnl={with_gate['pnl']:8.2f}% mdd={with_gate['mdd']:8.2f}% trades={with_gate['trades']:3d} wr={with_gate.get('wr', float('nan')):.3f}", flush=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    return report


def main() -> int:
    real = run_variant("real_direction", None)
    always_long = run_variant("always_long", omega.ACTION_LONG)
    always_short = run_variant("always_short", omega.ACTION_SHORT)

    print()
    print("=== SUMMARY (with_gate pnl / mdd / trades) ===")
    for wname in gate.WINDOW_DEFS:
        r = real["windows"][wname]["with_gate"]
        al = always_long["windows"][wname]["with_gate"]
        as_ = always_short["windows"][wname]["with_gate"]
        print(f"{wname:8} real={r['pnl']:7.2f}%/{r['mdd']:7.2f}%/{r['trades']:3d}t  "
              f"always_long={al['pnl']:7.2f}%/{al['mdd']:7.2f}%/{al['trades']:3d}t  "
              f"always_short={as_['pnl']:7.2f}%/{as_['mdd']:7.2f}%/{as_['trades']:3d}t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
