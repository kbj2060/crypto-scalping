#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402
import train_eval_omega1_2_1_tp_runner_meta_selector_20260610 as meta  # noqa: E402


MODEL_ID = "omega1_2_1_tp_runner_partial_selector_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class ActionSpec:
    name: str
    close_frac: float
    extend: bool


ACTIONS = (
    ActionSpec("take_all", 1.0, False),
    ActionSpec("extend_all", 0.0, True),
    ActionSpec("close30_extend70", 0.30, True),
    ActionSpec("close50_extend50", 0.50, True),
    ActionSpec("close70_extend30", 0.70, True),
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _event_features(frame: pd.DataFrame, state: pd.DataFrame, pos: base.Position, i: int, unreal: float) -> dict[str, float]:
    return meta._event_features(frame, state, pos, i, unreal)


def _finish_after_action(
    cash: float,
    arrays: dict[str, np.ndarray],
    pos: base.Position,
    i: int,
    action: ActionSpec,
    template: meta.RunnerTemplate,
    *,
    fee_eff: float,
    slip_eff: float,
) -> float:
    before = max(float(cash), 1e-12)
    p = base.Position(**pos.__dict__)
    c = float(cash)
    if action.close_frac > 0.0:
        c, p, _ = base._close_fraction(c, arrays, p, int(i), float(action.close_frac), fee_eff, slip_eff)
        if p.side == 0:
            return float(c / before - 1.0)
    if action.extend:
        old_tp = float(p.take_profit)
        p.floor_unreal = max(float(p.floor_unreal), old_tp * float(template.floor_frac))
        p.take_profit = old_tp * float(template.extend_mult)
    else:
        c, p, _ = base._close_fraction(c, arrays, p, int(i), 1.0, fee_eff, slip_eff)
        return float(c / before - 1.0)

    extensions = 1
    last_i = len(arrays["close"]) - 1
    for j in range(int(i), last_i):
        unreal = base._unreal(arrays, p, int(j), slip_eff)
        p.mfe = max(float(p.mfe), float(unreal))
        p.mae = min(float(p.mae), float(unreal))
        if p.take_profit > 0.0 and unreal >= p.take_profit:
            if extensions < int(template.max_extensions):
                extensions += 1
                old_tp = float(p.take_profit)
                p.floor_unreal = max(float(p.floor_unreal), old_tp * float(template.floor_frac))
                p.take_profit = old_tp * float(template.extend_mult)
                continue
            c, p, _ = base._close_fraction(c, arrays, p, int(j), 1.0, fee_eff, slip_eff)
            return float(c / before - 1.0)
        if p.floor_unreal > -abs(p.stop_loss) and unreal <= p.floor_unreal:
            c, p, _ = base._close_fraction(c, arrays, p, int(j), 1.0, fee_eff, slip_eff)
            return float(c / before - 1.0)
        if p.stop_loss > 0.0 and unreal <= -abs(p.stop_loss):
            c, p, _ = base._close_fraction(c, arrays, p, int(j), 1.0, fee_eff, slip_eff)
            return float(c / before - 1.0)
    c, p, _ = base._close_fraction(c, arrays, p, last_i, 1.0, fee_eff, slip_eff)
    return float(c / before - 1.0)


def _collect_tp_events(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    template: meta.RunnerTemplate,
) -> pd.DataFrame:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    pos = base.Position()
    rows: list[dict[str, Any]] = []
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(float(pos.mfe), float(unreal))
            pos.mae = min(float(pos.mae), float(unreal))
            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                feat = _event_features(frame, state, pos, i, unreal)
                rewards = {
                    spec.name: _finish_after_action(cash, arrays, pos, i, spec, template, fee_eff=fee_eff, slip_eff=slip_eff)
                    for spec in ACTIONS
                }
                best_name = max(rewards, key=rewards.get)
                rows.append(
                    {
                        "event_i": int(i),
                        "entry_signal_i": int(pos.entry_signal_i),
                        "entry_i": int(pos.entry_i),
                        "event_time": str(frame["timestamp"].iloc[int(i)]),
                        "best_action": best_name,
                        "best_action_id": int([a.name for a in ACTIONS].index(best_name)),
                        "take_all_ret": float(rewards["take_all"]),
                        "extend_all_ret": float(rewards["extend_all"]),
                        "split30_ret": float(rewards["close30_extend70"]),
                        "split50_ret": float(rewards["close50_extend50"]),
                        "split70_ret": float(rewards["close70_extend30"]),
                        "best_ret": float(rewards[best_name]),
                        "best_edge_vs_take": float(rewards[best_name] - rewards["take_all"]),
                        **feat,
                    }
                )
                reason = "take_profit"
            elif pos.floor_unreal > -abs(pos.stop_loss) and unreal <= pos.floor_unreal:
                reason = "meta_runner_profit_lock_exit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"
            if reason:
                cash, pos, _ = base._close_fraction(cash, arrays, pos, i, 1.0, fee_eff, slip_eff)
            continue
        if bool(active[i]):
            cash, pos, _entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("empty TP event dataset")
    return df


def _train_selector(events: pd.DataFrame, *, seed: int) -> tuple[Any | None, list[str], dict[str, Any]]:
    drop = {
        "event_i",
        "entry_signal_i",
        "entry_i",
        "event_time",
        "best_action",
        "best_action_id",
        "take_all_ret",
        "extend_all_ret",
        "split30_ret",
        "split50_ret",
        "split70_ret",
        "best_ret",
        "best_edge_vs_take",
    }
    feature_cols = [c for c in events.columns if c not in drop]
    y = events["best_action_id"].astype(int).to_numpy()
    diag = {
        "events": int(len(events)),
        "best_action_counts": events["best_action"].value_counts().to_dict(),
        "mean_best_edge_vs_take": float(events["best_edge_vs_take"].mean()),
        "feature_cols": feature_cols,
    }
    if len(np.unique(y)) < 2 or len(events) < 8:
        return None, feature_cols, {**diag, "reason": "insufficient_multiclass_events"}
    clf = HistGradientBoostingClassifier(
        max_iter=50,
        max_leaf_nodes=4,
        min_samples_leaf=3,
        l2_regularization=2.0,
        learning_rate=0.04,
        random_state=int(seed),
    )
    clf.fit(events[feature_cols].to_numpy(dtype=np.float64), y)
    return clf, feature_cols, {**diag, "reason": "ok", "seed": int(seed)}


def _choose_action(
    policy: str,
    clf: Any | None,
    feature_cols: list[str],
    frame: pd.DataFrame,
    state: pd.DataFrame,
    pos: base.Position,
    i: int,
    unreal: float,
) -> ActionSpec:
    if policy == "take_all":
        return ACTIONS[0]
    if policy == "extend_all":
        return ACTIONS[1]
    if policy == "split30":
        return ACTIONS[2]
    if policy == "split50":
        return ACTIONS[3]
    if policy == "split70":
        return ACTIONS[4]
    if policy == "hgb_selector" and clf is not None:
        feat = _event_features(frame, state, pos, i, unreal)
        x = np.asarray([[float(feat[c]) for c in feature_cols]], dtype=np.float64)
        action_id = int(clf.predict(x)[0])
        return ACTIONS[int(np.clip(action_id, 0, len(ACTIONS) - 1))]
    raise RuntimeError(f"unknown policy: {policy}")


def _metrics(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], actions: dict[str, int], long_entries: int, short_entries: int) -> dict[str, Any]:
    eq = np.asarray(equity_curve if equity_curve else [1.0], dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12) - 1.0) * 100.0
    arr = np.asarray(trades, dtype=np.float64)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(dd.min()),
        "trades": int(len(trades)),
        "wr": float(np.mean(arr > 0.0)) if len(arr) else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": dict(reasons),
        "runner_actions": dict(actions),
    }


def _simulate(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    template: meta.RunnerTemplate,
    policy: str,
    clf: Any | None,
    feature_cols: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    equity_curve = [cash]
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    actions: dict[str, int] = {}
    pos = base.Position()
    extensions = 0
    long_entries = short_entries = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(float(pos.mfe), float(unreal))
            pos.mae = min(float(pos.mae), float(unreal))
            equity_curve.append(cash * (1.0 + unreal))
            reason = ""
            log_pos_override: base.Position | None = None
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                before_action_pos = base.Position(**pos.__dict__)
                spec = _choose_action(policy, clf, feature_cols, frame, state, pos, i, unreal)
                actions[spec.name] = actions.get(spec.name, 0) + 1
                if spec.close_frac > 0.0:
                    cash, pos, _ = base._close_fraction(cash, arrays, pos, i, float(spec.close_frac), fee_eff, slip_eff)
                if pos.side != 0 and spec.extend:
                    extensions += 1
                    old_tp = float(pos.take_profit)
                    pos.floor_unreal = max(float(pos.floor_unreal), old_tp * float(template.floor_frac))
                    pos.take_profit = old_tp * float(template.extend_mult)
                else:
                    reason = "take_profit"
                    log_pos_override = before_action_pos
            elif pos.floor_unreal > -abs(pos.stop_loss) and unreal <= pos.floor_unreal:
                reason = "meta_runner_profit_lock_exit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"
            if reason:
                close_pos = log_pos_override if log_pos_override is not None else base.Position(**pos.__dict__)
                if pos.side != 0:
                    cash, pos, _ = base._close_fraction(cash, arrays, pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(runner._ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, extensions))
                extensions = 0
            continue
        equity_curve.append(cash)
        if bool(active[i]):
            side = int(dec.iloc[int(i)].get("side", 0) or 0)
            cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
            if entered:
                long_entries += int(side > 0)
                short_entries += int(side < 0)
                extensions = 0
    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(runner._ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end", extensions))
    return _metrics(cash, equity_curve, trades, reasons, actions, long_entries, short_entries), pd.DataFrame(rows)


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
        f"{prefix}_actions": metrics["runner_actions"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = runner._build()
    print(json.dumps({"stage": "build_done", "sec": round(time.time() - t0, 3)}), flush=True)
    template = meta.TEMPLATES[0]
    events = _collect_tp_events(
        data["validation"]["frame"],
        data["validation"]["dec"],
        data["validation"]["state"],
        fee=float(data["validation"]["fee"]),
        slip=float(data["validation"]["slip"]),
        cost_mult=3.0,
        template=template,
    )
    events.to_csv(OUT_DIR / "validation_tp_partial_counterfactual_events.csv", index=False)
    clf, feature_cols, train_diag = _train_selector(events, seed=260613)
    print(json.dumps({"stage": "events_done", **train_diag, "sec": round(time.time() - t0, 3)}, default=_json_default), flush=True)

    policies = ["take_all", "extend_all", "split30", "split50", "split70", "hgb_selector"]
    rows: list[dict[str, Any]] = []
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    for policy in policies:
        print(json.dumps({"stage": "simulate_start", "policy": policy, "sec": round(time.time() - t0, 3)}), flush=True)
        row: dict[str, Any] = {"policy": policy}
        ledgers[policy] = {}
        for split in ("validation", "oos"):
            metrics, ledger = _simulate(
                data[split]["frame"],
                data[split]["dec"],
                data[split]["state"],
                fee=float(data[split]["fee"]),
                slip=float(data[split]["slip"]),
                cost_mult=3.0,
                template=template,
                policy=policy,
                clf=clf,
                feature_cols=feature_cols,
            )
            row.update(_row(split, metrics))
            ledgers[policy][split] = ledger
        rows.append(row)

    ranking = pd.DataFrame(rows)
    base_oos = float(ranking.loc[ranking["policy"].eq("take_all"), "oos_pnl"].iloc[0])
    base_val = float(ranking.loc[ranking["policy"].eq("take_all"), "validation_pnl"].iloc[0])
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - base_oos
    ranking["delta_validation_pnl"] = ranking["validation_pnl"] - base_val
    ranking["score"] = ranking["oos_pnl"] + 0.45 * ranking["validation_pnl"] + 0.25 * ranking["oos_mdd"] + 0.20 * ranking["validation_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_mdd"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "tp_runner_partial_selector_ranking.csv", index=False)
    for policy in ["take_all", "extend_all", *ranking["policy"].head(4).tolist()]:
        if policy in ledgers:
            for split, ledger in ledgers[policy].items():
                ledger.to_csv(OUT_DIR / f"{split}_{policy}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "method": "TP-touched event policy: take all, extend all, or partial close then extend. Counterfactual labels are generated on validation TP events only; OOS applies selected policies without retraining on OOS.",
        "template": {
            "name": template.name,
            "extend_mult": float(template.extend_mult),
            "floor_frac": float(template.floor_frac),
            "max_extensions": int(template.max_extensions),
            "quality_min": float(template.quality_min),
            "momentum_min": float(template.momentum_min),
        },
        "train_diag": train_diag,
        "top": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "events": str(OUT_DIR / "validation_tp_partial_counterfactual_events.csv"),
            "ranking": str(OUT_DIR / "tp_runner_partial_selector_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
