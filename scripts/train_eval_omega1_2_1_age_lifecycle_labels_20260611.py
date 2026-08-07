#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402
import train_eval_omega1_2_1_tp_runner_meta_selector_20260610 as meta  # noqa: E402


MODEL_ID = "omega1_2_1_age_lifecycle_labels_20260611"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TP_RUNNER_BUNDLE = ROOT / "data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib"

AGE_GRID = (1, 3, 6, 12, 24, 48, 96, 192, 384, 768)


@dataclass
class StateSnap:
    i: int
    cash: float
    pos: base.Position
    unreal: float


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


def _feature_row(frame: pd.DataFrame, state: pd.DataFrame, pos: base.Position, i: int, unreal: float) -> dict[str, float]:
    feat = meta._event_features(frame, state, pos, i, unreal)
    tp = max(float(pos.take_profit), 1e-8)
    sl = max(abs(float(pos.stop_loss)), 1e-8)
    age = max(int(i) - int(pos.entry_i), 0)
    feat.update(
        {
            "age_bucket_1": float(age <= 1),
            "age_bucket_3": float(1 < age <= 3),
            "age_bucket_6": float(3 < age <= 6),
            "age_bucket_12": float(6 < age <= 12),
            "age_bucket_24": float(12 < age <= 24),
            "age_bucket_48": float(24 < age <= 48),
            "age_bucket_96": float(48 < age <= 96),
            "age_bucket_192p": float(age > 96),
            "tp_progress": float(unreal / tp),
            "sl_progress": float(-unreal / sl),
            "dist_tp": float(pos.take_profit - unreal),
            "dist_sl": float(unreal + abs(pos.stop_loss)),
            "floor_unreal": float(pos.floor_unreal),
            "mfe_minus_unreal": float(max(pos.mfe, unreal) - unreal),
        }
    )
    return feat


def _close_now_pct(arrays: dict[str, np.ndarray], snap: StateSnap, fee_eff: float, slip_eff: float) -> float:
    close_pos = base.Position(**snap.pos.__dict__)
    cash_after, _pos, _ = base._close_fraction(float(snap.cash), arrays, close_pos, int(snap.i), 1.0, fee_eff, slip_eff)
    return float((cash_after / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)


def _snap_wanted(age: int) -> bool:
    if age in AGE_GRID:
        return True
    return age > 0 and age % 96 == 0


def _collect_age_dataset(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    min_edge_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    pos = base.Position()
    snaps: list[StateSnap] = []
    label_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    trades: list[float] = []
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            age = max(int(i) - int(pos.entry_i), 0)
            if _snap_wanted(age):
                snaps.append(StateSnap(i=int(i), cash=float(cash), pos=base.Position(**pos.__dict__), unreal=float(unreal)))
            reason = base._hit_reason(unreal, pos)
            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                final_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(final_pct)
                ledger_rows.append(runner._ledger_row(frame, arrays, close_pos, i, cash, final_pct, reason, 0))
                for snap in snaps:
                    exit_now_pct = _close_now_pct(arrays, snap, fee_eff, slip_eff)
                    edge = exit_now_pct - final_pct
                    label_rows.append(
                        {
                            "event_i": int(snap.i),
                            "entry_signal_i": int(close_pos.entry_signal_i),
                            "final_exit_i": int(i),
                            "exit_now_pct": float(exit_now_pct),
                            "final_pct": float(final_pct),
                            "edge_pct": float(edge),
                            "label_exit_now": int(edge > float(min_edge_pct)),
                            "label_hold": int(edge <= float(min_edge_pct)),
                            **_feature_row(frame, state, snap.pos, snap.i, snap.unreal),
                        }
                    )
                snaps = []
            continue
        if bool(active[i]):
            cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
            if entered:
                snaps = []
    if pos.side != 0:
        i = len(frame) - 1
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
        final_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        ledger_rows.append(runner._ledger_row(frame, arrays, close_pos, i, cash, final_pct, "forced_end", 0))
        for snap in snaps:
            exit_now_pct = _close_now_pct(arrays, snap, fee_eff, slip_eff)
            edge = exit_now_pct - final_pct
            label_rows.append(
                {
                    "event_i": int(snap.i),
                    "entry_signal_i": int(close_pos.entry_signal_i),
                    "final_exit_i": int(i),
                    "exit_now_pct": float(exit_now_pct),
                    "final_pct": float(final_pct),
                    "edge_pct": float(edge),
                    "label_exit_now": int(edge > float(min_edge_pct)),
                    "label_hold": int(edge <= float(min_edge_pct)),
                    **_feature_row(frame, state, snap.pos, snap.i, snap.unreal),
                }
            )
    return pd.DataFrame(label_rows), pd.DataFrame(ledger_rows)


def _train_model(rows: pd.DataFrame, *, kind: str, seed: int) -> tuple[Any | None, list[str], dict[str, Any]]:
    drop = {"event_i", "entry_signal_i", "final_exit_i", "exit_now_pct", "final_pct", "edge_pct", "label_exit_now", "label_hold"}
    feature_cols = [c for c in rows.columns if c not in drop]
    y = rows["label_exit_now"].astype(int).to_numpy()
    diag = {
        "rows": int(len(rows)),
        "positive": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "feature_cols": feature_cols,
    }
    if len(rows) < 20 or len(np.unique(y)) < 2:
        return None, feature_cols, {**diag, "reason": "insufficient_or_single_class"}
    if kind == "et":
        model = ExtraTreesClassifier(
            n_estimators=220,
            max_depth=4,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=int(seed),
        )
    elif kind == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=70,
            max_leaf_nodes=6,
            learning_rate=0.04,
            l2_regularization=2.0,
            random_state=int(seed),
        )
    else:
        raise RuntimeError(kind)
    model.fit(rows[feature_cols].to_numpy(dtype=np.float64), y)
    return model, feature_cols, {**diag, "reason": "ok", "kind": kind, "seed": int(seed)}


def _tp_runner_extend_allowed(bundle: dict[str, Any] | None, frame: pd.DataFrame, state: pd.DataFrame, pos: base.Position, i: int, unreal: float) -> bool:
    if not bundle:
        return False
    template = meta.RunnerTemplate(**bundle["template"])
    return meta._selector_allowed(
        bundle.get("model"),
        list(bundle.get("feature_cols", [])),
        frame,
        state,
        pos,
        int(i),
        float(unreal),
        template=template,
        proba_min=float(bundle.get("proba_min", 2.0)),
    )


def _exit_now_allowed(
    model: Any | None,
    feature_cols: list[str],
    frame: pd.DataFrame,
    state: pd.DataFrame,
    pos: base.Position,
    i: int,
    unreal: float,
    *,
    proba_min: float,
    min_unreal: float,
    min_age: int,
) -> bool:
    if model is None:
        return False
    age = max(int(i) - int(pos.entry_i), 0)
    if unreal < float(min_unreal) or age < int(min_age):
        return False
    feat = _feature_row(frame, state, pos, i, unreal)
    x = np.asarray([[float(feat[c]) for c in feature_cols]], dtype=np.float64)
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(x)[0, 1])
    else:
        p = float(model.predict(x)[0])
    return p >= float(proba_min)


def _metrics(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int) -> dict[str, Any]:
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
    model: Any | None,
    feature_cols: list[str],
    proba_min: float,
    min_unreal: float,
    min_age: int,
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
    pos = base.Position()
    long_entries = short_entries = 0
    extensions = 0
    template = meta.RunnerTemplate(**tp_bundle["template"]) if tp_bundle else meta.TEMPLATES[0]
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + unreal))
            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                if tp_bundle and extensions < int(template.max_extensions) and _tp_runner_extend_allowed(tp_bundle, frame, state, pos, i, unreal):
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
            elif _exit_now_allowed(
                model,
                feature_cols,
                frame,
                state,
                pos,
                i,
                unreal,
                proba_min=proba_min,
                min_unreal=min_unreal,
                min_age=min_age,
            ):
                reason = "age_lifecycle_exit"
            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
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
    return _metrics(cash, equity_curve, trades, reasons, long_entries, short_entries), pd.DataFrame(rows)


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = runner._build()
    print(json.dumps({"stage": "build_done", "sec": round(time.time() - t0, 3)}), flush=True)
    tp_bundle = joblib.load(TP_RUNNER_BUNDLE) if TP_RUNNER_BUNDLE.exists() else None
    labels_path = OUT_DIR / "validation_age_lifecycle_labels.csv"
    ledger_path = OUT_DIR / "validation_label_source_baseline_ledger.csv"
    if labels_path.exists() and ledger_path.exists():
        val_rows = pd.read_csv(labels_path)
        print(json.dumps({"stage": "labels_reused", "rows": int(len(val_rows)), "sec": round(time.time() - t0, 3)}), flush=True)
    else:
        val_rows, val_base_ledger = _collect_age_dataset(
            data["validation"]["frame"],
            data["validation"]["dec"],
            data["validation"]["state"],
            fee=float(data["validation"]["fee"]),
            slip=float(data["validation"]["slip"]),
            cost_mult=3.0,
            min_edge_pct=0.35,
        )
        val_rows.to_csv(labels_path, index=False)
        val_base_ledger.to_csv(ledger_path, index=False)
        print(json.dumps({"stage": "labels_built", "rows": int(len(val_rows)), "sec": round(time.time() - t0, 3)}), flush=True)

    configs: list[dict[str, Any]] = [
        {"variant": "baseline_no_runner", "tp_bundle": None, "model": None, "feature_cols": [], "proba_min": 2.0, "min_unreal": 999.0, "min_age": 999},
        {"variant": "tp_runner_only", "tp_bundle": tp_bundle, "model": None, "feature_cols": [], "proba_min": 2.0, "min_unreal": 999.0, "min_age": 999},
    ]
    model_diags: dict[str, Any] = {}
    model, feature_cols, diag = _train_model(val_rows, kind="et", seed=260613)
    model_diags["et_260613"] = diag
    configs.append(
        {
            "variant": "tp_runner_age_et_s260613_p085_u040",
            "tp_bundle": tp_bundle,
            "model": model,
            "feature_cols": feature_cols,
            "proba_min": 0.85,
            "min_unreal": 0.040,
            "min_age": 3,
        }
    )
    configs.append(
        {
            "variant": "tp_runner_age_et_s260613_p070_u025",
            "tp_bundle": tp_bundle,
            "model": model,
            "feature_cols": feature_cols,
            "proba_min": 0.70,
            "min_unreal": 0.025,
            "min_age": 3,
        }
    )
    rows: list[dict[str, Any]] = []
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    for idx, cfg in enumerate(configs):
        print(json.dumps({"stage": "simulate_start", "variant": cfg["variant"], "sec": round(time.time() - t0, 3)}), flush=True)
        row = {"variant_id": int(idx), "variant": cfg["variant"]}
        split_ledgers: dict[str, pd.DataFrame] = {}
        for split in ("validation", "oos"):
            metrics, ledger = _simulate(
                data[split]["frame"],
                data[split]["dec"],
                data[split]["state"],
                fee=float(data[split]["fee"]),
                slip=float(data[split]["slip"]),
                cost_mult=3.0,
                tp_bundle=cfg["tp_bundle"],
                model=cfg["model"],
                feature_cols=list(cfg["feature_cols"]),
                proba_min=float(cfg["proba_min"]),
                min_unreal=float(cfg["min_unreal"]),
                min_age=int(cfg["min_age"]),
            )
            row.update(_row(split, metrics))
            split_ledgers[split] = ledger
        ledgers[str(idx)] = split_ledgers
        rows.append(row)

    ranking = pd.DataFrame(rows)
    base_oos = float(ranking.loc[ranking["variant"].eq("baseline_no_runner"), "oos_pnl"].iloc[0])
    base_val = float(ranking.loc[ranking["variant"].eq("baseline_no_runner"), "validation_pnl"].iloc[0])
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - base_oos
    ranking["delta_validation_pnl"] = ranking["validation_pnl"] - base_val
    ranking["score"] = ranking["oos_pnl"] + 0.45 * ranking["validation_pnl"] + 0.25 * ranking["oos_mdd"] + 0.20 * ranking["validation_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_mdd"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "age_lifecycle_ranking.csv", index=False)
    for variant_id in sorted(set([0, 1] + [int(x) for x in ranking["variant_id"].head(5).tolist()])):
        for split, ledger in ledgers[str(variant_id)].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{variant_id}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "method": "Build lifecycle labels from SLTP baseline entries at fixed holding ages. Train exit-now selector; parent entries and true-leverage risk contract are frozen.",
        "label_diag": {
            "rows": int(len(val_rows)),
            "positive": int(val_rows["label_exit_now"].sum()) if len(val_rows) else 0,
            "positive_rate": float(val_rows["label_exit_now"].mean()) if len(val_rows) else 0.0,
            "edge_mean": float(val_rows["edge_pct"].mean()) if len(val_rows) else 0.0,
            "edge_median": float(val_rows["edge_pct"].median()) if len(val_rows) else 0.0,
        },
        "model_diags": model_diags,
        "baseline": ranking[ranking["variant"].eq("baseline_no_runner")].to_dict(orient="records")[0],
        "tp_runner_only": ranking[ranking["variant"].eq("tp_runner_only")].to_dict(orient="records")[0],
        "top20": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "labels": str(OUT_DIR / "validation_age_lifecycle_labels.csv"),
            "ranking": str(OUT_DIR / "age_lifecycle_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "label_diag": report["label_diag"], "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
