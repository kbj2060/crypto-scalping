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
import train_eval_omega1_2_1_horizon_router_20260611 as hr  # noqa: E402


MODEL_ID = "omega1_2_1_capped_runner_router_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DROP_LABEL_COLS = {
    "i",
    "runner_ret",
    "runner_hold",
    "runner_reason",
    "capped_ret",
    "capped_hold",
    "capped_reason",
    "capped_edge",
    "label_capped",
}


@dataclass(frozen=True)
class CappedSpec:
    name: str
    cap_bars: int
    min_unreal: float


SPECS = (
    CappedSpec("cap1024_floor0", 1024, 0.0),
    CappedSpec("cap1536_floor0", 1536, 0.0),
    CappedSpec("cap2048_floor0", 2048, 0.0),
    CappedSpec("cap1024_profit2", 1024, 0.02),
    CappedSpec("cap1536_profit2", 1536, 0.02),
    CappedSpec("cap2048_profit2", 2048, 0.02),
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


def _simulate_entry_path(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    dec: pd.DataFrame,
    i: int,
    *,
    fee_eff: float,
    slip_eff: float,
    cap_spec: CappedSpec | None,
) -> dict[str, Any]:
    cash = 1.0
    cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
    if not entered:
        return {"ret": 0.0, "hold": 0, "reason": "not_filled", "mfe": 0.0, "mae": 0.0}
    last = len(frame) - 1
    reason = "forced_end"
    exit_i = last
    for j in range(int(pos.entry_i), last + 1):
        unreal = base._unreal(arrays, pos, j, slip_eff)
        pos.mfe = max(pos.mfe, unreal)
        pos.mae = min(pos.mae, unreal)
        reason = base._hit_reason(base._unreal(arrays, pos, j, slip_eff), pos)
        if reason:
            exit_i = int(j)
            break
        if cap_spec is not None and int(j) - int(pos.entry_i) >= int(cap_spec.cap_bars):
            if unreal >= float(cap_spec.min_unreal):
                reason = "capped_runner_exit"
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


def _build_labels(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    spec: CappedSpec,
    edge: float,
    hold_penalty: float,
) -> pd.DataFrame:
    arrays = base._arrays(frame)
    active_idxs = np.flatnonzero(base.omega._active(dec))
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    rows: list[dict[str, Any]] = []
    for i in active_idxs:
        if int(i) >= len(frame) - 2:
            continue
        runner_path = _simulate_entry_path(frame, arrays, dec, int(i), fee_eff=fee_eff, slip_eff=slip_eff, cap_spec=None)
        capped_path = _simulate_entry_path(frame, arrays, dec, int(i), fee_eff=fee_eff, slip_eff=slip_eff, cap_spec=spec)
        runner_u = float(runner_path["ret"]) - float(hold_penalty) * float(runner_path["hold"])
        capped_u = float(capped_path["ret"]) - float(hold_penalty) * float(capped_path["hold"])
        rows.append(
            {
                "i": int(i),
                "runner_ret": float(runner_path["ret"]),
                "runner_hold": int(runner_path["hold"]),
                "runner_reason": str(runner_path["reason"]),
                "capped_ret": float(capped_path["ret"]),
                "capped_hold": int(capped_path["hold"]),
                "capped_reason": str(capped_path["reason"]),
                "capped_edge": float(capped_u - runner_u),
                "label_capped": int(float(capped_u - runner_u) > float(edge)),
                **hr._entry_features(frame, state, dec, int(i)),
            }
        )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _feature_cols(labels: pd.DataFrame) -> list[str]:
    return [c for c in labels.columns if c not in DROP_LABEL_COLS]


def _fit(labels: pd.DataFrame, *, kind: str, seed: int) -> tuple[Any | None, list[str], dict[str, Any]]:
    cols = _feature_cols(labels)
    y = labels["label_capped"].astype(int).to_numpy()
    diag = {"kind": kind, "seed": int(seed), "rows": int(len(labels)), "positive": int(y.sum()), "feature_count": int(len(cols))}
    if len(labels) < 20 or len(np.unique(y)) < 2:
        return None, cols, {**diag, "reason": "insufficient_or_single_class"}
    if kind == "hgb":
        model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=5, l2_regularization=2.0, learning_rate=0.045, random_state=int(seed))
    elif kind == "et":
        model = ExtraTreesClassifier(n_estimators=240, max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=int(seed))
    else:
        raise RuntimeError(f"unknown kind: {kind}")
    model.fit(labels[cols].to_numpy(dtype=np.float64), y)
    return model, cols, {**diag, "reason": "ok"}


def _proba(model: Any | None, cols: list[str], frame: pd.DataFrame, state: pd.DataFrame, dec: pd.DataFrame, i: int) -> float:
    if model is None:
        return 0.0
    feat = hr._entry_features(frame, state, dec, int(i))
    x = np.asarray([[float(feat[c]) for c in cols]], dtype=np.float64)
    return float(model.predict_proba(x)[0, 1]) if hasattr(model, "predict_proba") else float(model.predict(x)[0])


def _ledger_row(frame: pd.DataFrame, arrays: dict[str, np.ndarray], pos: base.Position, exit_i: int, cash: float, net_pct: float, reason: str, route: str) -> dict[str, Any]:
    row = runner._ledger_row(frame, arrays, pos, exit_i, cash, net_pct, reason, 0)
    row["capped_route"] = route
    row["hold_bars"] = int(exit_i) - int(pos.entry_i)
    return row


def _metric(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int, ledger: pd.DataFrame) -> dict[str, Any]:
    out = runner._metric(cash, equity_curve, trades, reasons, long_entries, short_entries)
    if ledger.empty:
        out.update({"avg_hold_bars": 0.0, "median_hold_bars": 0.0, "max_hold_bars": 0, "capped_trades": 0, "runner_trades": 0})
        return out
    hold = pd.to_numeric(ledger["hold_bars"], errors="raise")
    route = ledger["capped_route"].astype(str)
    out.update(
        {
            "avg_hold_bars": float(hold.mean()),
            "median_hold_bars": float(hold.median()),
            "max_hold_bars": int(hold.max()),
            "capped_trades": int(route.eq("capped").sum()),
            "runner_trades": int(route.eq("runner").sum()),
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
    spec: CappedSpec,
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
            equity_curve.append(cash * (1.0 + base._unreal(arrays, pos, i, slip_eff)))
            reason = base._hit_reason(base._unreal(arrays, pos, i, slip_eff), pos)
            if not reason and route == "capped" and int(i) - int(pos.entry_i) >= int(spec.cap_bars):
                if unreal >= float(spec.min_unreal):
                    reason = "capped_runner_exit"
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
        choose_capped = _proba(model, feature_cols, frame, state, dec, int(i)) >= float(proba_min)
        route = "capped" if choose_capped else "runner"
        side = int(dec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = base._enter(cash, arrays, dec, int(i), fee_eff, slip_eff)
        if entered:
            long_entries += int(side > 0)
            short_entries += int(side < 0)
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
    return _metric(cash, equity_curve, trades, reasons, long_entries, short_entries, ledger), ledger


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
        f"{prefix}_capped_trades": int(metrics["capped_trades"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    rows: list[dict[str, Any]] = []
    ledgers_by_id: dict[int, dict[str, pd.DataFrame]] = {}
    diags: dict[str, Any] = {}

    variant_id = 0
    base_spec = SPECS[0]
    base_row: dict[str, Any] = {"variant_id": 0, "variant": "baseline_runner", "spec": "none", "kind": "none", "seed": 0, "edge": 999.0, "hold_penalty": 0.0, "proba_min": 2.0}
    base_ledgers: dict[str, pd.DataFrame] = {}
    for split in ("validation", "oos"):
        payload = data[split]
        metrics, ledger = _simulate_router(payload["frame"], payload["dec"], payload["state"], fee=float(payload["fee"]), slip=float(payload["slip"]), cost_mult=3.0, spec=base_spec, model=None, feature_cols=[], proba_min=2.0)
        base_row.update(_row(split, metrics))
        base_ledgers[split] = ledger
    rows.append(base_row)
    ledgers_by_id[0] = base_ledgers
    variant_id += 1

    for spec in SPECS:
        for hold_penalty in (0.0005, 0.0010, 0.0020):
            labels = _build_labels(data["validation"]["frame"], data["validation"]["dec"], data["validation"]["state"], fee=float(data["validation"]["fee"]), slip=float(data["validation"]["slip"]), cost_mult=3.0, spec=spec, edge=0.0, hold_penalty=float(hold_penalty))
            labels.to_csv(OUT_DIR / f"labels_{spec.name}_hp{hold_penalty:g}.csv", index=False)
            for edge in (0.0, 0.25, 0.50):
                relabeled = labels.copy()
                relabeled["label_capped"] = (pd.to_numeric(relabeled["capped_edge"], errors="raise") > float(edge)).astype(int)
                for kind in ("hgb", "et"):
                    for seed in (260611, 260612):
                        model, cols, diag = _fit(relabeled, kind=kind, seed=seed)
                        diags[f"{spec.name}_hp{hold_penalty:g}_e{edge:g}_{kind}_s{seed}"] = diag
                        if model is None:
                            continue
                        for proba_min in (0.45, 0.55, 0.65, 0.75):
                            row: dict[str, Any] = {"variant_id": int(variant_id), "variant": f"{spec.name}_{kind}_s{seed}_e{edge:g}_hp{hold_penalty:g}_p{str(proba_min).replace('.', '')}", "spec": spec.name, "kind": kind, "seed": int(seed), "edge": float(edge), "hold_penalty": float(hold_penalty), "proba_min": float(proba_min)}
                            split_ledgers: dict[str, pd.DataFrame] = {}
                            for split in ("validation", "oos"):
                                payload = data[split]
                                metrics, ledger = _simulate_router(payload["frame"], payload["dec"], payload["state"], fee=float(payload["fee"]), slip=float(payload["slip"]), cost_mult=3.0, spec=spec, model=model, feature_cols=cols, proba_min=float(proba_min))
                                row.update(_row(split, metrics))
                                split_ledgers[split] = ledger
                            rows.append(row)
                            ledgers_by_id[variant_id] = split_ledgers
                            variant_id += 1

    ranking = pd.DataFrame(rows)
    baseline = ranking.loc[ranking["variant"].eq("baseline_runner")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(baseline["oos_pnl"])
    ranking["delta_validation_pnl"] = ranking["validation_pnl"] - float(baseline["validation_pnl"])
    ranking["delta_oos_trades"] = ranking["oos_trades"] - int(baseline["oos_trades"])
    ranking["delta_oos_avg_hold"] = ranking["oos_avg_hold"] - float(baseline["oos_avg_hold"])
    ranking["delta_oos_max_hold"] = ranking["oos_max_hold"] - int(baseline["oos_max_hold"])
    ranking["score"] = ranking["oos_pnl"] + 0.55 * ranking["validation_pnl"] + 0.35 * ranking["oos_mdd"] + 0.35 * ranking["delta_oos_trades"] - 0.020 * ranking["oos_avg_hold"] - 0.008 * ranking["oos_max_hold"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_trades", "oos_mdd"], ascending=[False, False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "capped_runner_router_ranking.csv", index=False)

    balanced = ranking[
        (ranking["variant"] != "baseline_runner")
        & (ranking["oos_pnl"] >= float(baseline["oos_pnl"]) * 0.90)
        & (ranking["validation_pnl"] >= float(baseline["validation_pnl"]) * 0.90)
        & (ranking["oos_mdd"] >= float(baseline["oos_mdd"]) - 2.0)
        & (ranking["oos_trades"] >= int(baseline["oos_trades"]))
        & ((ranking["oos_avg_hold"] < float(baseline["oos_avg_hold"])) | (ranking["oos_max_hold"] < int(baseline["oos_max_hold"])))
    ].copy()
    balanced.to_csv(OUT_DIR / "capped_runner_router_balanced.csv", index=False)

    save_ids = sorted(set([0] + [int(x) for x in ranking["variant_id"].head(20).tolist()] + [int(x) for x in balanced["variant_id"].head(20).tolist()]))
    for sid in save_ids:
        for split, ledger in ledgers_by_id[int(sid)].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{sid}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Entry-time router for capped runner profile to target extreme hold while preserving normal runner behavior.",
        "baseline": baseline.to_dict(),
        "diagnostics": diags,
        "balanced_count": int(len(balanced)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "balanced": balanced.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "capped_runner_router_ranking.csv"),
            "balanced": str(OUT_DIR / "capped_runner_router_balanced.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "balanced_count": int(len(balanced)), "top10": ranking.head(10).to_dict(orient="records"), "balanced": balanced.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
