#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402


MODEL_ID = "omega1_2_1_horizon_router_20260611"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class RotationSpec:
    name: str
    tp_scale: float
    sl_scale: float
    floor_activate: float
    floor_frac: float


ROTATION_SPECS = (
    RotationSpec("rot_tp045_sl060_floor40_35", 0.45, 0.60, 0.40, 0.35),
    RotationSpec("rot_tp055_sl070_floor45_35", 0.55, 0.70, 0.45, 0.35),
    RotationSpec("rot_tp065_sl080_floor50_35", 0.65, 0.80, 0.50, 0.35),
    RotationSpec("rot_tp075_sl085_floor55_25", 0.75, 0.85, 0.55, 0.25),
    RotationSpec("rot_tp085_sl090_floor65_20", 0.85, 0.90, 0.65, 0.20),
)

DROP_LABEL_COLS = {
    "i",
    "runner_ret",
    "runner_hold",
    "runner_reason",
    "rotation_ret",
    "rotation_hold",
    "rotation_reason",
    "rotation_edge",
    "label_rotation",
}


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


def _entry_features(frame: pd.DataFrame, state: pd.DataFrame, dec: pd.DataFrame, i: int) -> dict[str, float]:
    row = state.iloc[int(i)]
    out = {str(c): float(row[c]) for c in state.columns}
    side = float(dec.iloc[int(i)].get("side", 0) or 0)
    close = pd.to_numeric(frame["close"], errors="raise")
    out["entry_side"] = side
    for lag in (1, 3, 6, 12, 24, 48):
        ret = float(close.pct_change(lag).iloc[int(i)] if int(i) >= lag else 0.0)
        out[f"entry_ret{lag}_side"] = ret * side
    out["entry_hour"] = float(pd.to_datetime(frame["timestamp"].iloc[int(i)]).hour)
    return out


def _apply_rotation(pos: base.Position, spec: RotationSpec) -> base.Position:
    out = base.Position(**pos.__dict__)
    out.take_profit = max(float(out.take_profit) * float(spec.tp_scale), 1e-8)
    out.stop_loss = max(float(out.stop_loss) * float(spec.sl_scale), 1e-8)
    out.floor_unreal = -abs(float(out.stop_loss))
    return out


def _maybe_rotation_floor(pos: base.Position, unreal: float, spec: RotationSpec) -> base.Position:
    out = base.Position(**pos.__dict__)
    if out.tightened == 0 and out.take_profit > 0.0 and unreal >= float(out.take_profit) * float(spec.floor_activate):
        out.floor_unreal = max(float(out.floor_unreal), float(out.mfe) * float(spec.floor_frac), 0.001)
        out.tightened = 1
    return out


def _simulate_entry_path(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    dec: pd.DataFrame,
    i: int,
    *,
    fee_eff: float,
    slip_eff: float,
    rotation: RotationSpec | None,
) -> dict[str, Any]:
    cash = 1.0
    cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
    if not entered:
        return {"ret": 0.0, "hold": 0, "reason": "not_filled", "mfe": 0.0, "mae": 0.0}
    if rotation is not None:
        pos = _apply_rotation(pos, rotation)
    last = len(frame) - 1
    reason = "forced_end"
    exit_i = last
    for j in range(int(pos.entry_i), last + 1):
        unreal = base._unreal(arrays, pos, j, slip_eff)
        pos.mfe = max(pos.mfe, unreal)
        pos.mae = min(pos.mae, unreal)
        if rotation is not None:
            pos = _maybe_rotation_floor(pos, unreal, rotation)
        reason = base._hit_reason(base._unreal(arrays, pos, j, slip_eff), pos)
        if reason:
            exit_i = int(j)
            break
    cash, _pos, _ = base._close_fraction(cash, arrays, pos, exit_i, 1.0, fee_eff, slip_eff)
    return {
        "ret": float((cash - 1.0) * 100.0),
        "hold": int(exit_i) - int(pos.entry_i),
        "reason": str(reason),
        "mfe": float(pos.mfe * 100.0),
        "mae": float(pos.mae * 100.0),
    }


def _build_counterfactual_labels(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    spec: RotationSpec,
    edge: float,
) -> pd.DataFrame:
    arrays = base._arrays(frame)
    active_idxs = np.flatnonzero(base.omega._active(dec))
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    rows: list[dict[str, Any]] = []
    for i in active_idxs:
        if int(i) >= len(frame) - 2:
            continue
        runner_path = _simulate_entry_path(frame, arrays, dec, int(i), fee_eff=fee_eff, slip_eff=slip_eff, rotation=None)
        rotation_path = _simulate_entry_path(frame, arrays, dec, int(i), fee_eff=fee_eff, slip_eff=slip_eff, rotation=spec)
        rows.append(
            {
                "i": int(i),
                "runner_ret": float(runner_path["ret"]),
                "runner_hold": int(runner_path["hold"]),
                "runner_reason": str(runner_path["reason"]),
                "rotation_ret": float(rotation_path["ret"]),
                "rotation_hold": int(rotation_path["hold"]),
                "rotation_reason": str(rotation_path["reason"]),
                "rotation_edge": float(rotation_path["ret"] - runner_path["ret"]),
                "label_rotation": int(float(rotation_path["ret"] - runner_path["ret"]) > float(edge)),
                **_entry_features(frame, state, dec, int(i)),
            }
        )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _feature_cols(labels: pd.DataFrame) -> list[str]:
    return [c for c in labels.columns if c not in DROP_LABEL_COLS]


def _fit_router(labels: pd.DataFrame, *, kind: str, seed: int) -> tuple[Any | None, list[str], dict[str, Any]]:
    cols = _feature_cols(labels)
    y = labels["label_rotation"].astype(int).to_numpy()
    diag: dict[str, Any] = {
        "kind": kind,
        "seed": int(seed),
        "rows": int(len(labels)),
        "positive": int(y.sum()),
        "feature_count": int(len(cols)),
    }
    if len(labels) < 20 or len(np.unique(y)) < 2:
        return None, cols, {**diag, "reason": "insufficient_or_single_class"}
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=5,
            l2_regularization=2.0,
            learning_rate=0.045,
            random_state=int(seed),
        )
    elif kind == "et":
        clf = ExtraTreesClassifier(
            n_estimators=220,
            max_depth=4,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=int(seed),
        )
    else:
        raise RuntimeError(f"unknown router kind: {kind}")
    clf.fit(labels[cols].to_numpy(dtype=np.float64), y)
    return clf, cols, {**diag, "reason": "ok"}


def _rotation_proba(model: Any | None, cols: list[str], frame: pd.DataFrame, state: pd.DataFrame, dec: pd.DataFrame, i: int) -> float:
    if model is None:
        return 0.0
    feat = _entry_features(frame, state, dec, int(i))
    x = np.asarray([[float(feat[c]) for c in cols]], dtype=np.float64)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x)[0, 1])
    return float(model.predict(x)[0])


def _ledger_row(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    pos: base.Position,
    exit_i: int,
    cash: float,
    net_pct: float,
    reason: str,
    route: str,
) -> dict[str, Any]:
    row = runner._ledger_row(frame, arrays, pos, exit_i, cash, net_pct, reason, 0)
    row["horizon_route"] = route
    row["hold_bars"] = int(exit_i) - int(pos.entry_i)
    return row


def _metric_with_hold(
    cash: float,
    equity_curve: list[float],
    trades: list[float],
    reasons: dict[str, int],
    long_entries: int,
    short_entries: int,
    ledger: pd.DataFrame,
) -> dict[str, Any]:
    out = runner._metric(cash, equity_curve, trades, reasons, long_entries, short_entries)
    if ledger.empty:
        out.update({"avg_hold_bars": 0.0, "median_hold_bars": 0.0, "max_hold_bars": 0, "runner_trades": 0, "rotation_trades": 0})
        return out
    hold = pd.to_numeric(ledger["hold_bars"], errors="raise")
    routes = ledger["horizon_route"].astype(str)
    out.update(
        {
            "avg_hold_bars": float(hold.mean()),
            "median_hold_bars": float(hold.median()),
            "max_hold_bars": int(hold.max()),
            "runner_trades": int(routes.eq("runner").sum()),
            "rotation_trades": int(routes.eq("rotation").sum()),
        }
    )
    return out


def _simulate_router(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    spec: RotationSpec | None,
    model: Any | None,
    feature_cols: list[str],
    proba_min: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    equity_curve: list[float] = [cash]
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    pos = base.Position()
    route = "runner"
    long_entries = 0
    short_entries = 0

    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            if route == "rotation" and spec is not None:
                pos = _maybe_rotation_floor(pos, unreal, spec)
            equity_curve.append(cash * (1.0 + base._unreal(arrays, pos, i, slip_eff)))
            reason = base._hit_reason(base._unreal(arrays, pos, i, slip_eff), pos)
            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(_ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, route))
                route = "runner"
            continue

        equity_curve.append(cash)
        if not bool(active[i]):
            continue
        choose_rotation = spec is not None and _rotation_proba(model, feature_cols, frame, state, dec, i) >= float(proba_min)
        route = "rotation" if choose_rotation else "runner"
        side = int(dec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
        if entered:
            long_entries += int(side > 0)
            short_entries += int(side < 0)
            if choose_rotation and spec is not None:
                pos = _apply_rotation(pos, spec)
        else:
            route = "runner"

    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(_ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end", route))

    ledger = pd.DataFrame(rows)
    return _metric_with_hold(cash, equity_curve, trades, reasons, long_entries, short_entries, ledger), ledger


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_avg_hold": float(metrics["avg_hold_bars"]),
        f"{prefix}_median_hold": float(metrics["median_hold_bars"]),
        f"{prefix}_max_hold": int(metrics["max_hold_bars"]),
        f"{prefix}_runner_trades": int(metrics["runner_trades"]),
        f"{prefix}_rotation_trades": int(metrics["rotation_trades"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    rows: list[dict[str, Any]] = []
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    label_diags: dict[str, Any] = {}
    model_diags: dict[str, Any] = {}

    baseline_ledgers: dict[str, pd.DataFrame] = {}
    baseline_row: dict[str, Any] = {
        "variant_id": 0,
        "variant": "baseline_runner_only",
        "rotation_spec": "none",
        "model_kind": "none",
        "proba_min": 2.0,
    }
    for split in ("validation", "oos"):
        metrics, ledger = _simulate_router(
            data[split]["frame"],
            data[split]["dec"],
            data[split]["state"],
            fee=float(data[split]["fee"]),
            slip=float(data[split]["slip"]),
            cost_mult=3.0,
            spec=None,
            model=None,
            feature_cols=[],
            proba_min=2.0,
        )
        baseline_row.update(_row(split, metrics))
        baseline_ledgers[split] = ledger
    rows.append(baseline_row)
    ledgers["0"] = baseline_ledgers

    variant_id = 1
    for spec in ROTATION_SPECS:
        labels = _build_counterfactual_labels(
            data["validation"]["frame"],
            data["validation"]["dec"],
            data["validation"]["state"],
            fee=float(data["validation"]["fee"]),
            slip=float(data["validation"]["slip"]),
            cost_mult=3.0,
            spec=spec,
            edge=0.10,
        )
        labels.to_csv(OUT_DIR / f"validation_labels_{spec.name}.csv", index=False)
        label_diags[spec.name] = {
            "rows": int(len(labels)),
            "rotation_positive": int(labels["label_rotation"].sum()) if len(labels) else 0,
            "mean_rotation_edge": float(labels["rotation_edge"].mean()) if len(labels) else 0.0,
            "median_rotation_edge": float(labels["rotation_edge"].median()) if len(labels) else 0.0,
        }
        for kind in ("hgb", "et"):
            model, feature_cols, diag = _fit_router(labels, kind=kind, seed=260611)
            model_diags[f"{spec.name}_{kind}"] = diag
            if model is None:
                continue
            for proba_min in (0.45, 0.55, 0.65, 0.75):
                row: dict[str, Any] = {
                    "variant_id": int(variant_id),
                    "variant": f"{spec.name}_{kind}_p{str(proba_min).replace('.', '')}",
                    "rotation_spec": spec.name,
                    "model_kind": kind,
                    "proba_min": float(proba_min),
                }
                split_ledgers: dict[str, pd.DataFrame] = {}
                for split in ("validation", "oos"):
                    metrics, ledger = _simulate_router(
                        data[split]["frame"],
                        data[split]["dec"],
                        data[split]["state"],
                        fee=float(data[split]["fee"]),
                        slip=float(data[split]["slip"]),
                        cost_mult=3.0,
                        spec=spec,
                        model=model,
                        feature_cols=feature_cols,
                        proba_min=float(proba_min),
                    )
                    row.update(_row(split, metrics))
                    split_ledgers[split] = ledger
                rows.append(row)
                ledgers[str(variant_id)] = split_ledgers
                variant_id += 1

    ranking = pd.DataFrame(rows)
    base_row = ranking.loc[ranking["variant"].eq("baseline_runner_only")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(base_row["oos_pnl"])
    ranking["delta_validation_pnl"] = ranking["validation_pnl"] - float(base_row["validation_pnl"])
    ranking["delta_oos_avg_hold"] = ranking["oos_avg_hold"] - float(base_row["oos_avg_hold"])
    ranking["delta_oos_max_hold"] = ranking["oos_max_hold"] - int(base_row["oos_max_hold"])
    ranking["score"] = (
        ranking["oos_pnl"]
        + 0.35 * ranking["validation_pnl"]
        + 0.25 * ranking["oos_mdd"]
        - 0.018 * ranking["oos_avg_hold"]
        - 0.006 * ranking["oos_max_hold"]
    )
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_mdd", "oos_avg_hold"], ascending=[False, False, False, True]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "horizon_router_ranking.csv", index=False)

    promotable = ranking[
        (ranking["variant"] != "baseline_runner_only")
        & (ranking["oos_pnl"] >= float(base_row["oos_pnl"]))
        & (ranking["validation_pnl"] >= float(base_row["validation_pnl"]) * 0.90)
        & (ranking["oos_mdd"] >= float(base_row["oos_mdd"]) - 1.0)
        & ((ranking["oos_avg_hold"] < float(base_row["oos_avg_hold"])) | (ranking["oos_max_hold"] < int(base_row["oos_max_hold"])))
    ].copy()
    promotable.to_csv(OUT_DIR / "horizon_router_promotable.csv", index=False)

    save_ids = sorted(set([0] + [int(x) for x in ranking["variant_id"].head(10).tolist()] + [int(x) for x in promotable["variant_id"].head(10).tolist()]))
    for sid in save_ids:
        for split, ledger in ledgers[str(sid)].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{sid}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Entry-time horizon router: preserve runner trades, route selected entries to faster rotation TP/SL/floor policy.",
        "baseline": base_row.to_dict(),
        "label_diags": label_diags,
        "model_diags": model_diags,
        "promotable_count": int(len(promotable)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "promotable": promotable.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "horizon_router_ranking.csv"),
            "promotable": str(OUT_DIR / "horizon_router_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
