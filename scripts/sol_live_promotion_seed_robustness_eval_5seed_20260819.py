#!/usr/bin/env python3
"""'라이브로 승격된 SOL zig075 v2(adaptive_squeeze) 단일 컴포넌트 자체'가 시드에 강건한가를
검증 -- Seed-Diversity Ensemble Promotion Gate 적용 (CLAUDE.md, N=5).

ETH 쪽(scripts/eth_live_promotion_seed_robustness_eval_3seed_20260819.py)과 동일한 목적/기법을
SOL에 적용. 원본(260720 학습, 재학습 없이 라이브 번들 그대로 재사용)+신규 랜덤시드 4개
(848498120/732130789/193749676/534479280, SystemRandom 샘플링)를 sol_live_promotion_seed_
robustness_zig075_seed_variant_20260819.py로 재학습, 5개 전부 Fresh-Forward 평가.

Risk sidecar는 5개 시드 전부 원본(adaptive_squeeze_q070_20260720) 것을 frozen 재사용 -- ETH
쪽과 동일한 단순화(시드별 전용 sidecar 재학습은 범위 밖, 명시적 caveat).

⚠️ 재사용한 기존 검증된 엔진 (재구현 아님):
- 프레임 빌드: sol_live_promotion_seed_robustness_canonicaldata_20260819.parent_script.
  _prepare_frames (= git HEAD sol_20260707.py 원본 코드, adaptive_squeeze 데이터 + 재생성된
  REGIME3_CURRENT 오버레이 적용).
- VAL/OOS 경계: scripts/eval_sol_dual_structure_router_20260729.py의 SPLIT_TS(2025-09-01)/
  VAL_END(2026-01-01)/OOS_END(2026-04-01) 상수를 그대로 재사용(CLAUDE.md Fresh-Forward 기본값과
  일치하는, 이미 SOL에서 검증된 관례) -- 이 스크립트가 직접 재정의하지 않는다.
- 컴포넌트 준비: eval_sol_dual_structure_router_20260729.py::prepare_component을 그대로 호출
  (SOL 전용 sidecar 모듈 train_eval_omega4_2_risk_sidecar_sol_20260707을 이미 올바르게
  참조하고 있어 자체 omega/sidecar 전역상태에 의존하지 않음 -- oof 파라미터도 이미 내장).
- 실제 bar-by-bar 워크: replay_omega4_6_1_greedy_router_20260706.py::greedy_replay -- 단
  greedy.omega/greedy.sidecar를 SOL 모듈로 monkey-patch해야 한다(이 함수 자체가 모듈 전역
  omega/sidecar를 직접 참조, 기본값은 ETH용). ⚠️ eval_sol_dual_structure_router_20260729.py
  자신의 dual_replay()는 SCALE_MAP(zig075_L/S 롱숏 비대칭 레버리지)을 전혀 적용하지 않는다
  (그 라우터 자신의 replay_variant()가 greedy.SCALE_MAP을 전부 1.0으로 중립화하고 자체 risk_
  scale/regime_margin_scale 그리드서치를 씀 -- 이는 별도 dual-router 구조탐색 축의 설계고,
  실제 라이브 SOL 설정(scale_map={zig075_L:1.0, zig075_S:1.75}, trading_bot_modules/
  runtime_config.py OMEGA4_6_1_SHADOW_ASSET_CONFIG["sol"])과 다르다) -- 그래서 dual_replay가
  아니라 greedy_replay를 쓴다(SCALE_MAP.get(f"{name}_{'L'/'S'}") 를 실제로 적용하는 쪽).
- 포스트혹 duration 게이트: research_eth_omega461_live_sltp_mfe_width_20260813.py::
  _duration_gated, 지표: research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py::
  _ledger_metrics -- 둘 다 자산무관(ledger/frame의 공통 컬럼만 사용).

⚠️ Duration gate 값: trading_bot_modules/runtime_config.py의 OMEGA4_6_1_SHADOW_ASSET_CONFIG
["sol"]["duration_threshold"]=0.0055208323 (FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_DURATION_GATE_OFF
기본값 False일 때 실제로 쓰이는 값, trading_bot.py:11327-11331 확인). docs/model_contracts/
sol_adaptive_squeeze_v2_20260720.md는 v1/v2 비교를 "gate off"로 서술하지만 그건 그 문서 자체의
평가방법론(비교용) 서술이고, 오늘(2026-08-19) 시점 runtime_config.py 기본값은 게이트가 켜져
있다 -- 이 스크립트는 no_gate/with_gate 둘 다 계산해 이 불일치를 명시적으로 남긴다.

⚠️ 3창(6창 아님): CLAUDE.md 필수 2창(val/oos_q1)에 oos_q2(2026-04-01~06-30, adaptive_squeeze
EVAL_CSV가 07-21까지 커버해 데이터는 있음)를 보너스로 추가했다. ETH의 6창(2025q1/q2/q3 포함)은
parent.SPLIT_TS=2025-10-01 기준 2025 Q1-Q3가 train_raw(인샘플)에 들어가 "OOS형" 진단과 다른
성격이고, SOL 전용 WINDOW_DEFS 인프라도 없어 새로 만드는 대신 이미 검증된 eval_sol_dual_
structure_router_20260729.py의 2창 관례+데이터가 허용하는 만큼의 보너스 1창으로 범위를 좁혔다."""
from __future__ import annotations

import json
import pickle
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

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import eval_sol_dual_structure_router_20260729 as router_script  # noqa: E402
import sol_live_promotion_seed_robustness_canonicaldata_20260819 as canon_wrap  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sol_live_promotion_seed_robustness_20260819"
DEVICE = parent._device("cpu")

sol_omega = canon_wrap.omega  # train_eval_omega1_2_tabm_diffusion_risk_sol_20260707, TRAIN_CSV/EVAL_CSV/REGIME3_CURRENT_* already overridden
sol_sidecar = router_script.sidecar  # train_eval_omega4_2_risk_sidecar_sol_20260707

OOS_Q2_END = pd.Timestamp("2026-07-01")
SOL_DURATION_THRESHOLD = 0.0055208323  # trading_bot_modules/runtime_config.py OMEGA4_6_1_SHADOW_ASSET_CONFIG["sol"]["duration_threshold"]
SCALE_MAP = {"zig075_L": 1.0, "zig075_S": 1.75}  # same source, ["sol"]["scale_map"]

FROZEN_SIDECAR_PKL = ROOT / "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_q070_20260720/risk_sidecar.pkl"
ZIG075_CFG_BASE = {
    "q_tag": "q070", "threshold": 0.70,
    "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
    "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
    "sidecar_pkl": FROZEN_SIDECAR_PKL,
    "exit_threshold": 0.95,  # confirmed against the sidecar's own report.json contract.exit_threshold
}

SEED_LABELS = ["seed1_live_original", "848498120", "732130789", "193749676", "534479280"]


def _bundle_path(seed_label: str) -> Path:
    if seed_label == "seed1_live_original":
        return ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720/true_3head_tabm_bundle.pt"
    return ROOT / f"tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_seedvariant_{seed_label}/true_3head_tabm_bundle.pt"


def generate_predictions(bundle_path: Path, frame: pd.DataFrame, *, threshold: float, oof: bool, out_path: Path) -> Path:
    """Fresh bar-by-bar-consistent entry predictions directly from `bundle_path` -- no stored
    ledger, no old bundle's predictions, genuine inference on this frame's own point-in-time
    features. Mirrors eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818.py::
    generate_predictions (same mechanism, asset-agnostic body -- only parent/hard needed, both
    shared modules)."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    base_cols = bundle["base_cols"]
    models = bundle["models"]
    missing = sorted(set(base_cols) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{bundle_path}: frame missing {len(missing)} base_cols: {missing[:20]}")
    x = parent._base_input(frame, base_cols)
    route = hard._route_id(frame)
    preds = {expert: parent._predict_payload(models[expert], x, device=DEVICE) for expert in hard.EXPERT_NAMES}
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    out = parent._prediction_output(frame, direction, quality, threshold=float(threshold), prefix="omega1_regime3_expertdq_oof")
    if not oof:
        out = out.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in out.columns})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = sol_omega._load_fee_slip()

    # Missing bundles fail fast with a clear message instead of a deep traceback mid-loop.
    for seed_label in SEED_LABELS:
        bp = _bundle_path(seed_label)
        if not bp.exists():
            raise RuntimeError(f"{seed_label}: bundle not found at {bp} -- training not finished yet?")
    if not FROZEN_SIDECAR_PKL.exists():
        raise RuntimeError(f"frozen risk sidecar not found: {FROZEN_SIDECAR_PKL}")

    # VAL/OOS boundaries: exact reuse of eval_sol_dual_structure_router_20260729.py's own
    # SPLIT_TS/VAL_END/OOS_END (already the SOL-established, CLAUDE.md-matching convention).
    parent.SPLIT_TS = router_script.SPLIT_TS
    sol_omega.SPLIT_TS = router_script.SPLIT_TS
    frames = canon_wrap.parent_script._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=canon_wrap.parent_script.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    val_frame = frames["val_raw"].loc[frames["val_raw"]["timestamp"] < router_script.VAL_END].reset_index(drop=True)
    oos_all = frames["oos_raw"]
    oos_q1_frame = oos_all.loc[oos_all["timestamp"] < router_script.OOS_END].reset_index(drop=True)
    oos_q2_frame = oos_all.loc[(oos_all["timestamp"] >= router_script.OOS_END) & (oos_all["timestamp"] < OOS_Q2_END)].reset_index(drop=True)
    windows: dict[str, dict[str, Any]] = {
        "val": {"frame": val_frame, "oof": True},
        "oos_q1": {"frame": oos_q1_frame, "oof": False},
        "oos_q2": {"frame": oos_q2_frame, "oof": False},
    }
    for wname, w in windows.items():
        if w["frame"].empty:
            raise RuntimeError(f"window {wname} produced an empty frame")
        print(f"window={wname} rows={len(w['frame'])} range=[{w['frame']['timestamp'].min()}, {w['frame']['timestamp'].max()}] oof={w['oof']}", flush=True)

    # greedy_replay itself references module-level `omega`/`sidecar` globals (defaults to ETH's
    # modules) -- must monkey-patch to SOL's before calling it. router_script.prepare_component
    # does NOT need this (it uses its own top-level SOL-scoped omega/sidecar imports directly).
    greedy.omega = sol_omega
    greedy.sidecar = sol_sidecar
    greedy.PRIORITY = ("zig075",)
    greedy.SCALE_MAP = dict(SCALE_MAP)

    pred_dir = OUT_DIR / "predictions"
    all_results: dict[str, dict[str, Any]] = {}
    for seed_label in SEED_LABELS:
        bundle_path = _bundle_path(seed_label)
        print(f"\n########## seed={seed_label} bundle={bundle_path} ##########", flush=True)
        all_results[seed_label] = {}
        for wname, w in windows.items():
            frame = w["frame"]
            cfg = dict(ZIG075_CFG_BASE, bundle=bundle_path)
            pred_csv = pred_dir / f"{seed_label}_{wname}_predictions_q070.csv"
            generate_predictions(bundle_path, frame, threshold=cfg["threshold"], oof=w["oof"], out_path=pred_csv)
            component = router_script.prepare_component(frame, pred_csv, cfg, DEVICE, oof=w["oof"])
            _diag, ledger = greedy.greedy_replay(frame, {"zig075": component}, fee=fee, slip=slip, cost_mult=3.0, device=DEVICE)
            ledger_path = OUT_DIR / f"portfolio_ledger_{seed_label}_{wname}.csv"
            ledger.to_csv(ledger_path, index=False)
            no_gate = portfolio._ledger_metrics(ledger)
            with_gate = mfe_width._duration_gated(ledger, frame, SOL_DURATION_THRESHOLD)
            all_results[seed_label][wname] = {"no_gate": no_gate, "with_gate": with_gate, "ledger_path": str(ledger_path)}
            print(f"{seed_label:20} {wname:8} no_gate pnl={no_gate['pnl']:8.2f}% mdd={no_gate['mdd']:8.2f}% trades={no_gate['trades']:3d}  |  "
                  f"with_gate pnl={with_gate['pnl']:8.2f}% mdd={with_gate['mdd']:8.2f}% trades={with_gate['trades']:3d}", flush=True)

    print()
    for gate_name in ("no_gate", "with_gate"):
        print(f"=== 5-시드 부호일치 요약 ({gate_name} PnL, %) ===")
        header = f"{'window':10}" + "".join(f"{s:>20}" for s in SEED_LABELS) + "  sign_consistent"
        print(header)
        sign_flip_windows = []
        for wname in windows:
            pnls = [all_results[s][wname][gate_name]["pnl"] for s in SEED_LABELS]
            signs = {p >= 0 for p in pnls}
            consistent = len(signs) == 1
            if not consistent:
                sign_flip_windows.append(wname)
            row = f"{wname:10}" + "".join(f"{p:20.2f}" for p in pnls) + f"  {'YES' if consistent else 'NO -- SIGN FLIP'}"
            print(row, flush=True)
        print(f"sign_flip_windows({gate_name})={sign_flip_windows}")
        print()

    report = {
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "n_seeds": len(SEED_LABELS),
        "seed_labels": SEED_LABELS,
        "risk_sizing_source": "frozen_original_sidecar_(adaptive_squeeze_q070_20260720)_reused_across_all_5_seeds -- "
                               "no per-seed risk sidecar retrained (same simplification as the ETH N=3 precedent), "
                               "explicit caveat, not a silent flat-constant substitute (margin/leverage remain real "
                               "per-bar sidecar outputs, just computed against each seed's own fresh decisions).",
        "duration_gate_note": f"SOL_DURATION_THRESHOLD={SOL_DURATION_THRESHOLD} matches trading_bot_modules/"
                               "runtime_config.py's current default (FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_DURATION_GATE_OFF=False). "
                               "docs/model_contracts/sol_adaptive_squeeze_v2_20260720.md's own v1-vs-v2 comparison table used "
                               "gate-off methodology instead -- both no_gate and with_gate are reported here to cover that "
                               "discrepancy explicitly.",
        "scale_map": SCALE_MAP,
        "windows_note": "3 windows (val/oos_q1/oos_q2), not ETH's 6 -- see module docstring for why.",
        "windows": {wname: {"range": [str(w["frame"]["timestamp"].min()), str(w["frame"]["timestamp"].max())], "rows": int(len(w["frame"]))} for wname, w in windows.items()},
        "results": all_results,
    }
    out_path = OUT_DIR / "summary_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"report={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
