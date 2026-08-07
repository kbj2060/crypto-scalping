#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402
import train_eval_omega1_2_1_tp_runner_meta_selector_20260610 as meta  # noqa: E402


MODEL_ID = "omega1_2_1_tp_runner_edge_selector_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


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


def _model(seed: int, kind: str):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.25, class_weight="balanced", random_state=int(seed), max_iter=500))
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=2,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=int(seed),
        )
    if kind == "et":
        return ExtraTreesClassifier(
            n_estimators=160,
            max_depth=2,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=int(seed),
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=60,
            max_leaf_nodes=4,
            min_samples_leaf=4,
            l2_regularization=1.5,
            learning_rate=0.04,
            random_state=int(seed),
        )
    raise RuntimeError(f"unknown model kind: {kind}")


def _prepare_events(events: pd.DataFrame, *, edge_min: float, keep_allowed_gate: bool) -> pd.DataFrame:
    out = events.copy()
    if out.empty:
        return out
    if keep_allowed_gate:
        out["label"] = ((out["allowed"].astype(int) == 1) & (out["edge"] > float(edge_min))).astype(int)
    else:
        out["label"] = (out["edge"] > float(edge_min)).astype(int)
    return out


def _train_selector(events: pd.DataFrame, *, seed: int, kind: str, train_frac: float) -> tuple[Any | None, list[str], dict[str, Any]]:
    if events.empty:
        return None, [], {"reason": "empty_events"}
    drop_cols = {"event_i", "entry_signal_i", "immediate_ret", "extended_ret", "edge", "label", "allowed"}
    feature_cols = [c for c in events.columns if c not in drop_cols]
    n_train = max(4, int(len(events) * float(train_frac)))
    train = events.iloc[:n_train].copy()
    y = train["label"].astype(int).to_numpy()
    diag = {
        "events": int(len(events)),
        "train_events": int(len(train)),
        "positive": int(events["label"].sum()),
        "positive_rate": float(events["label"].mean()),
        "train_positive": int(y.sum()),
        "train_positive_rate": float(y.mean()) if len(y) else 0.0,
        "edge_mean": float(events["edge"].mean()),
        "edge_median": float(events["edge"].median()),
        "feature_cols": feature_cols,
    }
    if len(np.unique(y)) < 2:
        return None, feature_cols, {**diag, "reason": "single_class_train"}
    clf = _model(seed, kind)
    clf.fit(train[feature_cols].to_numpy(dtype=np.float64), y)
    return clf, feature_cols, {**diag, "reason": "ok"}


def _selector_allowed(
    clf: Any | None,
    feature_cols: list[str],
    frame: pd.DataFrame,
    state: pd.DataFrame,
    pos: base.Position,
    i: int,
    unreal: float,
    *,
    template: meta.RunnerTemplate,
    proba_min: float,
    keep_allowed_gate: bool,
) -> bool:
    feat = meta._event_features(frame, state, pos, i, unreal)
    if keep_allowed_gate and (feat["quality"] < float(template.quality_min) or feat["ret3_side"] <= float(template.momentum_min)):
        return False
    if clf is None:
        return False
    x = np.asarray([[float(feat[c]) for c in feature_cols]], dtype=np.float64)
    if hasattr(clf, "predict_proba"):
        p = float(clf.predict_proba(x)[0, 1])
    else:
        p = float(clf.predict(x)[0])
    return p >= float(proba_min)


def _simulate_edge_selector(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    template: meta.RunnerTemplate,
    clf: Any | None,
    feature_cols: list[str],
    proba_min: float,
    keep_allowed_gate: bool,
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
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + unreal))
            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                if extensions < int(template.max_extensions) and _selector_allowed(
                    clf,
                    feature_cols,
                    frame,
                    state,
                    pos,
                    i,
                    unreal,
                    template=template,
                    proba_min=float(proba_min),
                    keep_allowed_gate=bool(keep_allowed_gate),
                ):
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
    return runner._metric(cash, equity_curve, trades, reasons, long_entries, short_entries), pd.DataFrame(rows)


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

    seeds = [260610, 260611, 260612, 260613, 260614, 260615]
    kinds = ["logit", "rf", "et", "hgb"]
    proba_mins = [0.55, 0.65, 0.75, 0.85]
    edge_mins = [0.000, 0.001, 0.0025, 0.005]
    train_frac = 0.70
    templates = [
        meta.TEMPLATES[0],
        meta.TEMPLATES[1],
        meta.TEMPLATES[2],
        replace(meta.TEMPLATES[0], name="oos_edge_135_floor60_ext2_edgeonly", floor_frac=0.60),
    ]

    baseline_template = meta.TEMPLATES[0]
    base_val, base_val_ledger = _simulate_edge_selector(
        data["validation"]["frame"],
        data["validation"]["dec"],
        data["validation"]["state"],
        fee=float(data["validation"]["fee"]),
        slip=float(data["validation"]["slip"]),
        cost_mult=3.0,
        template=baseline_template,
        clf=None,
        feature_cols=[],
        proba_min=2.0,
        keep_allowed_gate=True,
    )
    base_oos, base_oos_ledger = _simulate_edge_selector(
        data["oos"]["frame"],
        data["oos"]["dec"],
        data["oos"]["state"],
        fee=float(data["oos"]["fee"]),
        slip=float(data["oos"]["slip"]),
        cost_mult=3.0,
        template=baseline_template,
        clf=None,
        feature_cols=[],
        proba_min=2.0,
        keep_allowed_gate=True,
    )
    base_val_ledger.to_csv(OUT_DIR / "validation_baseline_no_runner_ledger.csv", index=False)
    base_oos_ledger.to_csv(OUT_DIR / "oos_baseline_no_runner_ledger.csv", index=False)

    rows: list[dict[str, Any]] = []
    event_summaries: dict[str, Any] = {}
    for template in templates:
        raw_events = meta._collect_events(
            data["validation"]["frame"],
            data["validation"]["dec"],
            data["validation"]["state"],
            fee=float(data["validation"]["fee"]),
            slip=float(data["validation"]["slip"]),
            cost_mult=3.0,
            template=template,
        )
        raw_events.to_csv(OUT_DIR / f"{template.name}_raw_validation_tp_events.csv", index=False)
        event_summaries[template.name] = {
            "events": int(len(raw_events)),
            "edge_mean": float(raw_events["edge"].mean()) if len(raw_events) else 0.0,
            "edge_median": float(raw_events["edge"].median()) if len(raw_events) else 0.0,
            "edge_positive": int((raw_events["edge"] > 0.0).sum()) if len(raw_events) else 0,
            "allowed": int(raw_events["allowed"].sum()) if len(raw_events) else 0,
        }
        for keep_allowed_gate in (True, False):
            for edge_min in edge_mins:
                events = _prepare_events(raw_events, edge_min=float(edge_min), keep_allowed_gate=bool(keep_allowed_gate))
                label_tag = "allowed_edge" if keep_allowed_gate else "pure_edge"
                events.to_csv(OUT_DIR / f"{template.name}_{label_tag}_{edge_min:.4f}_validation_tp_events.csv", index=False)
                for kind in kinds:
                    for seed in seeds:
                        clf, feature_cols, train_diag = _train_selector(events, seed=int(seed), kind=kind, train_frac=train_frac)
                        for proba_min in proba_mins:
                            val_m, _ = _simulate_edge_selector(
                                data["validation"]["frame"],
                                data["validation"]["dec"],
                                data["validation"]["state"],
                                fee=float(data["validation"]["fee"]),
                                slip=float(data["validation"]["slip"]),
                                cost_mult=3.0,
                                template=template,
                                clf=clf,
                                feature_cols=feature_cols,
                                proba_min=float(proba_min),
                                keep_allowed_gate=bool(keep_allowed_gate),
                            )
                            oos_m, _ = _simulate_edge_selector(
                                data["oos"]["frame"],
                                data["oos"]["dec"],
                                data["oos"]["state"],
                                fee=float(data["oos"]["fee"]),
                                slip=float(data["oos"]["slip"]),
                                cost_mult=3.0,
                                template=template,
                                clf=clf,
                                feature_cols=feature_cols,
                                proba_min=float(proba_min),
                                keep_allowed_gate=bool(keep_allowed_gate),
                            )
                            rows.append(
                                {
                                    "template": template.name,
                                    "label_mode": label_tag,
                                    "edge_min": float(edge_min),
                                    "kind": kind,
                                    "seed": int(seed),
                                    "proba_min": float(proba_min),
                                    **_row("val", val_m),
                                    **_row("oos", oos_m),
                                    "train_reason": train_diag.get("reason", "ok"),
                                    "train_positive": int(train_diag.get("train_positive", 0)),
                                    "positive": int(train_diag.get("positive", 0)),
                                    "events": int(train_diag.get("events", 0)),
                                    "positive_rate": float(train_diag.get("positive_rate", 0.0)),
                                }
                            )
        print(json.dumps({"stage": "template_done", "template": template.name, "sec": round(time.time() - t0, 3)}), flush=True)

    detail = pd.DataFrame(rows)
    detail["delta_val_pnl"] = detail["val_pnl"] - float(base_val["pnl"])
    detail["delta_oos_pnl"] = detail["oos_pnl"] - float(base_oos["pnl"])
    detail.to_csv(OUT_DIR / "edge_selector_seed_detail.csv", index=False)
    group_cols = ["template", "label_mode", "edge_min", "kind", "proba_min"]
    agg = detail.groupby(group_cols).agg(
        seeds=("seed", "nunique"),
        val_pnl_median=("val_pnl", "median"),
        val_pnl_min=("val_pnl", "min"),
        val_mdd_median=("val_mdd", "median"),
        val_wr_median=("val_wr", "median"),
        val_trades_median=("val_trades", "median"),
        oos_pnl_median=("oos_pnl", "median"),
        oos_pnl_min=("oos_pnl", "min"),
        oos_mdd_median=("oos_mdd", "median"),
        oos_wr_median=("oos_wr", "median"),
        oos_trades_median=("oos_trades", "median"),
        positive_rate=("positive_rate", "median"),
    ).reset_index()
    agg["delta_val_pnl_median"] = agg["val_pnl_median"] - float(base_val["pnl"])
    agg["delta_oos_pnl_median"] = agg["oos_pnl_median"] - float(base_oos["pnl"])
    agg["score"] = agg["oos_pnl_median"] + 0.45 * agg["val_pnl_median"] + 0.35 * agg["oos_mdd_median"] + 0.25 * agg["val_mdd_median"]
    agg = agg.sort_values(["oos_pnl_median", "val_pnl_median", "score"], ascending=False).reset_index(drop=True)
    agg.to_csv(OUT_DIR / "edge_selector_ranking.csv", index=False)

    top = agg.iloc[0].to_dict() if len(agg) else {}
    if top:
        detail_top = detail[
            (detail["template"] == top["template"])
            & (detail["label_mode"] == top["label_mode"])
            & (detail["edge_min"] == top["edge_min"])
            & (detail["kind"] == top["kind"])
            & (detail["proba_min"] == top["proba_min"])
        ].sort_values(["oos_pnl", "val_pnl"], ascending=False).iloc[0]
        template = next(t for t in templates if t.name == top["template"])
        raw_events = pd.read_csv(OUT_DIR / f"{template.name}_raw_validation_tp_events.csv")
        events = _prepare_events(raw_events, edge_min=float(top["edge_min"]), keep_allowed_gate=(top["label_mode"] == "allowed_edge"))
        clf, feature_cols, _ = _train_selector(events, seed=int(detail_top["seed"]), kind=str(top["kind"]), train_frac=train_frac)
        for split in ("validation", "oos"):
            m, ledger = _simulate_edge_selector(
                data[split]["frame"],
                data[split]["dec"],
                data[split]["state"],
                fee=float(data[split]["fee"]),
                slip=float(data[split]["slip"]),
                cost_mult=3.0,
                template=template,
                clf=clf,
                feature_cols=feature_cols,
                proba_min=float(top["proba_min"]),
                keep_allowed_gate=(top["label_mode"] == "allowed_edge"),
            )
            ledger.to_csv(OUT_DIR / f"{split}_best_seed_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "method": "TP runner extension selector retrained with pure edge labels and allowed+edge labels. Parent entries and Cost3 accounting are unchanged; no live bundle is written.",
        "baseline_no_runner": {"validation": base_val, "oos": base_oos},
        "event_summaries": event_summaries,
        "top": agg.head(30).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "detail": str(OUT_DIR / "edge_selector_seed_detail.csv"),
            "ranking": str(OUT_DIR / "edge_selector_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top10": agg.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
