#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_frozen_v27_offline_rl_exit_overlay_v33 as v33  # noqa: E402
from scripts.eval_hf_v13_v31_rl_surrounding_v49_v50_v51 import ClosePolicyNet, TorchClosePolicy, _feature_audit, _numeric_cols, _patch_v33_state  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "hf_v13_v49_tail_reentry_v57_20260513"
DEFAULT_PARENT = v31.DEFAULT_PARENT
DEFAULT_JACKPOT = v31.DEFAULT_JACKPOT
DEFAULT_V27 = v31.DEFAULT_V27
DEFAULT_V49 = ROOT / "data/ensemble/supervised/hf_v13_v31_rl_surrounding_v49_v50_v51_20260512/v49_exit_rl_raw_all.pkl"
DEFAULT_TRAIN = v31.DEFAULT_TRAIN
DEFAULT_EVAL = v31.DEFAULT_EVAL
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v49_tail_reentry_v57_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v49_tail_reentry_v57_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v49_tail_reentry_v57_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v49_tail_reentry_v57_20260513_grid.csv"

V31_BASELINE = {"cost1": {"pnl": 277.0679629973942, "mdd": -31.74}, "cost2": {"pnl": 112.79326141840412, "mdd": -31.46}, "cost3": {"pnl": 20.933695032758784, "mdd": -43.09}}


@dataclass(frozen=True)
class TailConfig:
    name: str
    min_realized: float
    min_mfe: float
    same_utility_min: float
    opposite_utility_max: float
    delay_bars: int
    notional: float
    tp: float
    sl: float
    max_hold: int
    max_reentries: int


def _configs() -> list[TailConfig]:
    rows = [TailConfig("v57_identity_v49", 99.0, 99.0, 99.0, -99.0, 99, 0.0, 0.0, 0.0, 0, 0)]
    i = 0
    for realized, mfe in ((0.010, 0.025), (0.020, 0.035), (0.030, 0.045)):
        for same, opp in ((0.010, 0.010), (0.014, 0.006)):
            for delay in (1, 2):
                for notional, tp, sl, hold in ((0.8, 0.026, 0.014, 24), (1.2, 0.035, 0.018, 36)):
                    rows.append(TailConfig(f"v57_tail_{i}", realized, mfe, same, opp, delay, notional, tp, sl, hold, 1))
                    i += 1
    return rows


def _install_pickle_aliases() -> None:
    import __main__

    setattr(__main__, "TorchClosePolicy", TorchClosePolicy)
    setattr(__main__, "ClosePolicyNet", ClosePolicyNet)


def _load_v49(path: Path) -> dict[str, Any]:
    _install_pickle_aliases()
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "policy" not in obj:
        raise TypeError(f"{path} is not a V49 payload")
    return obj


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.15 * c3["pnl"] - 0.28 * abs(c1["mdd"]) + 0.15 * min(c1.get("runner_actions", {}).get("tail_reentry", 0), 50))


def _flow_pass(row: pd.Series, side: int) -> bool:
    taker = float(row.get("net_taker_ratio", 0.0) or 0.0)
    accel = float(row.get("taker_acceleration", 0.0) or 0.0)
    intensity = float(row.get("trade_intensity", 0.0) or 0.0)
    return side * taker >= -0.10 and side * accel >= -0.20 and intensity >= -3.0


def backtest_tail(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    overlay_model: Any,
    overlay: v33.OverlayConfig,
    cfg: TailConfig,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    decisions: pd.DataFrame,
    record: bool = False,
) -> dict[str, Any]:
    close = _close(df)
    fee_eff = fee * cost_mult
    slip_eff = slip * cost_mult
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    owner = ""
    entry_price = entry_equity = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = deep_cooldown = 0
    add_done = False
    mfe = mae = 0.0
    entry_edge = entry_margin = 0.0
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None
    tail: dict[str, Any] | None = None
    reentries_used = 0

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        dd_abs = max(0.0, 1.0 - eq / max(peak, 1e-12))
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            reason = ""
            if owner in {"deep_alpha", "tail_reentry"} and hold >= overlay.min_hold:
                p_close = v33._predict_close_prob(overlay_model, v33._deep_state_row(df, i, pos, entry_edge, entry_margin, hold, unreal, mfe, mae))
                if p_close >= overlay.close_p_th:
                    reason = f"{owner}_rl_exit_overlay"
            if not reason:
                if take_profit > 0.0 and unreal >= take_profit:
                    reason = f"{owner}_take_profit"
                elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                    reason = f"{owner}_stop_loss"
                elif max_hold > 0 and hold >= max_hold:
                    reason = f"{owner}_max_hold"
            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                x = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    fill_i = min(i + 1, len(df) - 1)
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    add_px = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    cash -= cash * fee_eff * delta
                    notional = new_notional
                    actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                else:
                    actions["v21_reject"] = actions.get("v21_reject", 0) + 1
                add_done = True
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee_eff * notional
                realized = cash / max(entry_equity, 1e-12) - 1.0
                exit_side = pos
                exit_mfe = mfe
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "exit_reason": reason, "realized_net_pct": float(realized * 100.0), "final_notional_exposure": float(notional), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0), "fee_exit_pct": float(fee_eff * notional * 100.0), "cash_after": float(cash)})
                    records.append(out)
                profit_exit = realized >= cfg.min_realized and exit_mfe >= cfg.min_mfe and "stop_loss" not in reason and owner != "tail_reentry"
                if profit_exit and reentries_used < cfg.max_reentries:
                    tail = {"side": exit_side, "ready_i": i + cfg.delay_bars, "source_reason": reason}
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(overlay.cooldown))
                add_done = False
                open_record = None
                continue
        if pos != 0:
            continue
        if tail is not None and i >= int(tail["ready_i"]):
            side = int(tail["side"])
            same = float(deep_q[i, 0]) if side > 0 else float(deep_q[i, 1])
            opp = float(deep_q[i, 1]) if side > 0 else float(deep_q[i, 0])
            if same >= cfg.same_utility_min and opp <= cfg.opposite_utility_max and _flow_pass(df.iloc[i], side):
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "tail_reentry"
                entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                entry_equity = cash
                entry_idx = i
                parent_notional = notional = float(cfg.notional)
                take_profit = float(cfg.tp)
                stop_loss = float(cfg.sl)
                max_hold = int(cfg.max_hold)
                next_cooldown = 0
                entry_edge = same
                entry_margin = same - opp
                cash -= cash * fee_eff * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                reentries_used += 1
                actions["tail_reentry"] = actions.get("tail_reentry", 0) + 1
                tail = None
                if record:
                    open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
                continue
            if i > int(tail["ready_i"]) + 3:
                actions["tail_reject"] = actions.get("tail_reject", 0) + 1
                tail = None
        if cooldown > 0:
            cooldown -= 1
            continue
        if deep_cooldown > 0:
            deep_cooldown -= 1
        dec = decisions.iloc[i]
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
            fill_i = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            entry_edge = 0.0
            entry_margin = 0.0
            cash -= cash * fee_eff * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec.leverage)
            mfe = mae = 0.0
            add_done = False
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            if record:
                open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "leverage": float(dec.leverage), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
            continue
        if deep_cooldown <= 0 and i >= v33.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= overlay.edge_th and margin >= overlay.margin_th:
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "deep_alpha"
                entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                entry_equity = cash
                entry_idx = i
                parent_notional = notional = float(overlay.notional)
                take_profit = float(overlay.base_tp)
                stop_loss = float(overlay.base_sl)
                max_hold = int(overlay.base_hold)
                next_cooldown = int(overlay.cooldown)
                entry_edge = edge
                entry_margin = margin
                cash -= cash * fee_eff * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
                if record:
                    open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "deep_edge": float(edge), "deep_margin": float(margin), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
    if pos != 0:
        fill_i = len(df) - 1
        exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n = max(long_entries + short_entries, 1)
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "deep_entries": int(deep_entries), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "exits": exits, "runner_actions": actions}
    if record:
        out["trade_records"] = records
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V57 V49 tail-harvest reentry sleeve.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--v49-policy", type=Path, default=DEFAULT_V49)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading stack", flush=True)
    parent = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(args.v27_model)
    v49 = _load_v49(args.v49_policy)
    policy = v49["policy"]
    overlay = v33.OverlayConfig(**dict(v49["selected_config"]))
    base = dict(parent["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    raw_cols = list(v49.get("feature_cols") or _numeric_cols(train_all, eval_df))
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    state_audit = _feature_audit(raw_cols, train_all, eval_df)
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    with _patch_v33_state(raw_cols):
        for cfg in _configs():
            print(f"[{MODEL_ID}] validation cfg={cfg.name}", flush=True)
            v1 = backtest_tail(val, parent, jackpot_model, add_cfg, val_q, policy, overlay, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec)
            v2 = backtest_tail(val, parent, jackpot_model, add_cfg, val_q, policy, overlay, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec)
            v3 = backtest_tail(val, parent, jackpot_model, add_cfg, val_q, policy, overlay, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec)
            row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
        assert best is not None
        selected = TailConfig(**best["config"])
        metrics: dict[str, Any] = {}
        ledgers: dict[str, str] = {}
        for mult in (1, 2, 3):
            r = backtest_tail(eval_df, parent, jackpot_model, add_cfg, eval_q, policy, overlay, selected, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
            if mult == 1:
                ledger = pd.DataFrame(r.pop("trade_records", []))
                lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
                lp.parent.mkdir(parents=True, exist_ok=True)
                ledger.to_csv(lp, index=False)
                ledgers["cost1"] = str(lp)
            metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "v57_tail_reentry_manifest.json"
    manifest.write_text(json.dumps({"model_id": MODEL_ID, "selected_config": asdict(selected), "overlay": asdict(overlay), "v49_policy": str(args.v49_policy), "raw_state_feature_count": len(raw_cols)}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([{**{f"cfg_{k}": v for k, v in r["config"].items()}, "selection_score": r["selection_score"], "val_pnl": r["validation_cost1"]["pnl"], "val_mdd": r["validation_cost1"]["mdd"], "val_trades": r["validation_cost1"]["trades"], "val_tail_reentries": r["validation_cost1"].get("runner_actions", {}).get("tail_reentry", 0), "val_c2_pnl": r["validation_cost2"]["pnl"], "val_c3_pnl": r["validation_cost3"]["pnl"]} for r in rows]).to_csv(args.grid_out, index=False)
    blocking = list(parent_audit.get("blocking", [])) + [f"state:{x}" for x in state_audit.get("blocking", [])]
    warnings = list(parent_audit.get("warnings", [])) + [f"state:{x}" for x in state_audit.get("warnings", [])]
    if metrics["cost1"]["pnl"] <= V31_BASELINE["cost1"]["pnl"]:
        warnings.append("oos_cost1_did_not_beat_v31")
    if metrics["cost1"]["pnl"] <= 500:
        warnings.append("target_pnl_500_not_reached")
    if metrics["cost2"]["pnl"] <= 0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > 500 and metrics["cost2"]["pnl"] > 0 else "iterate"
    audit = {"status": "pass" if not blocking else "fail", "verdict": verdict, "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "policy": "v49_tail_reentry_v57", "entry_owner_frozen": True, "v49_exit_policy_reused": True, "tail_reentry_only_after_profit_exit": True, "parent_audit": parent_audit, "state_feature_audit": state_audit, "selected_config": asdict(selected), "metrics": metrics, "baseline_v31": V31_BASELINE}
    report = {"model_id": MODEL_ID, "design": "V57 adds a same-side tail reentry sleeve after profitable V31/V49 exits only when V27 same-side utility remains high, opposite utility is blocked, and flow proxies are not adverse.", "selected_config": asdict(selected), "selection_result": best, "metrics": metrics, "audit": audit, "artifacts": {"manifest": str(manifest), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "manifest": str(manifest), "selected_config": asdict(selected), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
