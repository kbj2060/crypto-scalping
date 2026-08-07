#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402
import train_eval_omega1_2_1_oracle_dp_exit_model_20260612 as oracle  # noqa: E402
import train_eval_omega1_2_1_tp_runner_meta_selector_20260610 as meta  # noqa: E402


MODEL_ID = "omega1_2_1_oracle_dp_protective_actions_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TP_RUNNER_BUNDLE = ROOT / "data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib"


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


def _predict_signal(clf: Any, mfe_reg: Any, mae_reg: Any, feature_cols: list[str], feat: dict[str, float], *, p_min: float, mfe_max: float, mae_min: float) -> tuple[bool, dict[str, float]]:
    x = np.asarray([[float(feat[c]) for c in feature_cols]], dtype=np.float64)
    p = float(clf.predict_proba(x)[0, 1])
    f_mfe = float(mfe_reg.predict(x)[0])
    f_mae = float(mae_reg.predict(x)[0])
    ok = bool(p >= float(p_min) and f_mfe <= float(mfe_max) and f_mae >= float(mae_min))
    return ok, {"p_exit": p, "pred_forward_mfe_R": f_mfe, "pred_forward_mae_R": f_mae}


def _apply_protective(pos: base.Position, unreal: float, *, action: str) -> tuple[base.Position, bool, str]:
    out = base.Position(**pos.__dict__)
    if action == "extension_veto":
        out.tightened = max(int(out.tightened), 10)
        return out, True, "oracle_extension_veto"
    if int(out.tightened) and action in {"floor_raise", "breakeven_lock", "tp_downshift"}:
        return out, False, ""
    if action == "floor_raise":
        floor = max(0.001, min(float(unreal) * 0.55, float(out.take_profit) * 0.75))
        out.floor_unreal = max(float(out.floor_unreal), floor)
        out.tightened = 1
        return out, True, "oracle_floor_raise"
    if action == "breakeven_lock":
        out.floor_unreal = max(float(out.floor_unreal), 0.001)
        out.tightened = 1
        return out, True, "oracle_breakeven_lock"
    if action == "tp_downshift":
        new_tp = max(float(unreal) + 0.006, float(out.take_profit) * 0.80)
        if new_tp < float(out.take_profit):
            out.take_profit = new_tp
            out.floor_unreal = max(float(out.floor_unreal), max(0.001, float(unreal) * 0.45))
            out.tightened = 1
            return out, True, "oracle_tp_downshift"
    return out, False, ""


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
        "protective_actions": dict(actions),
    }


def _simulate(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    tp_bundle: dict[str, Any] | None,
    clf: Any | None,
    mfe_reg: Any | None,
    mae_reg: Any | None,
    feature_cols: list[str],
    p_min: float,
    mfe_max: float,
    mae_min: float,
    action: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    template = meta.RunnerTemplate(**tp_bundle["template"]) if tp_bundle else meta.TEMPLATES[0]
    cash = 1.0
    equity_curve = [cash]
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    actions: dict[str, int] = {}
    pos = base.Position()
    extensions = 0
    long_entries = short_entries = 0
    extension_veto_active = False
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + unreal))

            if clf is not None and max(int(i) - int(pos.entry_i), 0) >= 2:
                feat = oracle._feature_row(frame, state, pos, i, unreal)
                signal, _pred = _predict_signal(clf, mfe_reg, mae_reg, feature_cols, feat, p_min=p_min, mfe_max=mfe_max, mae_min=mae_min)
                if signal:
                    pos, changed, action_name = _apply_protective(pos, unreal, action=action)
                    if changed:
                        actions[action_name] = actions.get(action_name, 0) + 1
                        extension_veto_active = extension_veto_active or action_name == "oracle_extension_veto"

            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                allow_extension = (
                    bool(tp_bundle)
                    and not extension_veto_active
                    and extensions < int(template.max_extensions)
                    and oracle._tp_runner_extend_allowed(tp_bundle, frame, state, pos, i, unreal)
                )
                if allow_extension:
                    extensions += 1
                    old_tp = float(pos.take_profit)
                    pos.floor_unreal = max(float(pos.floor_unreal), old_tp * float(template.floor_frac))
                    pos.take_profit = old_tp * float(template.extend_mult)
                else:
                    reason = "take_profit"
            elif pos.floor_unreal > -abs(pos.stop_loss) and unreal <= pos.floor_unreal:
                reason = "meta_runner_profit_lock_exit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"
            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(runner._ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, extensions))
                extensions = 0
                extension_veto_active = False
            continue
        equity_curve.append(cash)
        if bool(active[i]):
            side = int(dec.iloc[int(i)].get("side", 0) or 0)
            cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
            if entered:
                long_entries += int(side > 0)
                short_entries += int(side < 0)
                extensions = 0
                extension_veto_active = False
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
        f"{prefix}_actions": metrics["protective_actions"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = runner._build()
    print(json.dumps({"stage": "build_done", "sec": round(time.time() - t0, 3)}), flush=True)
    tp_bundle = joblib.load(TP_RUNNER_BUNDLE) if TP_RUNNER_BUNDLE.exists() else None
    paths = oracle._simulate_tp_runner_path(
        data["validation"]["frame"],
        data["validation"]["dec"],
        data["validation"]["state"],
        fee=float(data["validation"]["fee"]),
        slip=float(data["validation"]["slip"]),
        cost_mult=3.0,
        tp_bundle=tp_bundle,
    )
    train_df, dataset_diag = oracle._build_oracle_dataset(data["validation"]["frame"], data["validation"]["state"], paths)
    train_df.to_csv(OUT_DIR / "validation_oracle_dp_exit_dataset.csv", index=False)
    clf, mfe_reg, mae_reg, feature_cols, model_diag = oracle._train(train_df, seed=260613)
    print(json.dumps({"stage": "dataset_model_done", **dataset_diag, "sec": round(time.time() - t0, 3)}), flush=True)

    configs: list[dict[str, Any]] = [
        {"variant": "baseline_no_runner", "tp_bundle": None, "clf": None, "mfe_reg": None, "mae_reg": None, "feature_cols": [], "p_min": 2.0, "mfe_max": -999.0, "mae_min": -999.0, "action": ""},
        {"variant": "tp_runner_only", "tp_bundle": tp_bundle, "clf": None, "mfe_reg": None, "mae_reg": None, "feature_cols": [], "p_min": 2.0, "mfe_max": -999.0, "mae_min": -999.0, "action": ""},
    ]
    for action in ("extension_veto", "floor_raise", "tp_downshift", "breakeven_lock"):
        for p_min, mfe_max, mae_min in (
            (0.55, 0.20, -1.20),
            (0.65, 0.15, -0.90),
        ):
            configs.append(
                {
                    "variant": f"tp_runner_oracle_{action}_p{p_min:.2f}_mfe{mfe_max:.2f}_mae{mae_min:.2f}",
                    "tp_bundle": tp_bundle,
                    "clf": clf,
                    "mfe_reg": mfe_reg,
                    "mae_reg": mae_reg,
                    "feature_cols": feature_cols,
                    "p_min": float(p_min),
                    "mfe_max": float(mfe_max),
                    "mae_min": float(mae_min),
                    "action": action,
                }
            )

    rows: list[dict[str, Any]] = []
    ledgers: dict[int, dict[str, pd.DataFrame]] = {}
    for idx, cfg in enumerate(configs):
        print(json.dumps({"stage": "simulate_start", "variant": cfg["variant"], "sec": round(time.time() - t0, 3)}), flush=True)
        row: dict[str, Any] = {"variant_id": int(idx), "variant": str(cfg["variant"]), "p_min": float(cfg["p_min"]), "mfe_max": float(cfg["mfe_max"]), "mae_min": float(cfg["mae_min"]), "action": str(cfg["action"])}
        ledgers[idx] = {}
        for split in ("validation", "oos"):
            m, ledger = _simulate(
                data[split]["frame"],
                data[split]["dec"],
                data[split]["state"],
                fee=float(data[split]["fee"]),
                slip=float(data[split]["slip"]),
                cost_mult=3.0,
                tp_bundle=cfg["tp_bundle"],
                clf=cfg["clf"],
                mfe_reg=cfg["mfe_reg"],
                mae_reg=cfg["mae_reg"],
                feature_cols=list(cfg["feature_cols"]),
                p_min=float(cfg["p_min"]),
                mfe_max=float(cfg["mfe_max"]),
                mae_min=float(cfg["mae_min"]),
                action=str(cfg["action"]),
            )
            row.update(_row(split, m))
            ledgers[idx][split] = ledger
        rows.append(row)

    ranking = pd.DataFrame(rows)
    base_oos = float(ranking.loc[ranking["variant"].eq("baseline_no_runner"), "oos_pnl"].iloc[0])
    base_val = float(ranking.loc[ranking["variant"].eq("baseline_no_runner"), "validation_pnl"].iloc[0])
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - base_oos
    ranking["delta_validation_pnl"] = ranking["validation_pnl"] - base_val
    ranking["score"] = ranking["oos_pnl"] + 0.45 * ranking["validation_pnl"] + 0.25 * ranking["oos_mdd"] + 0.20 * ranking["validation_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_mdd"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "oracle_dp_protective_actions_ranking.csv", index=False)
    for variant_id in sorted(set([0, 1] + [int(x) for x in ranking["variant_id"].head(6).tolist()])):
        for split, ledger in ledgers[variant_id].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{variant_id}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "method": "Use Oracle DP exit classifier plus forward MFE/MAE regressors as non-closing protective actions: extension veto, floor raise, TP downshift, or breakeven lock.",
        "dataset_diag": dataset_diag,
        "model_diag": model_diag,
        "top": ranking.head(30).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "dataset": str(OUT_DIR / "validation_oracle_dp_exit_dataset.csv"),
            "ranking": str(OUT_DIR / "oracle_dp_protective_actions_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
