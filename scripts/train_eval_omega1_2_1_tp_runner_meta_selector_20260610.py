#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import eval_omega1_2_1_tp_runner_20260610 as runner
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_tp_runner_meta_selector_20260610"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LIVE_BUNDLE_DIR = ROOT / "data/ensemble/supervised" / MODEL_ID


@dataclass(frozen=True)
class RunnerTemplate:
    name: str
    extend_mult: float
    floor_frac: float
    max_extensions: int
    quality_min: float
    momentum_min: float


TEMPLATES = (
    RunnerTemplate("oos_edge_135_floor45_ext2", 1.35, 0.45, 2, 0.70, 0.0),
    RunnerTemplate("val_strong_175_floor90_ext1", 1.75, 0.90, 1, 0.70, 0.0),
    RunnerTemplate("balanced_150_floor90_ext1", 1.50, 0.90, 1, 0.70, 0.0),
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
    row = state.iloc[int(i)]
    close = pd.to_numeric(frame["close"], errors="raise")
    ret3 = float(close.pct_change(3).iloc[int(i)] if int(i) >= 3 else 0.0)
    ret6 = float(close.pct_change(6).iloc[int(i)] if int(i) >= 6 else 0.0)
    ret12 = float(close.pct_change(12).iloc[int(i)] if int(i) >= 12 else 0.0)
    side = float(pos.side)
    mfe = max(float(pos.mfe), float(unreal))
    giveback = (mfe - float(unreal)) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
    return {
        "side": side,
        "hold_bars": float(max(int(i) - int(pos.entry_i), 0)),
        "unreal": float(unreal),
        "mfe": float(mfe),
        "mae": float(pos.mae),
        "giveback": float(np.clip(giveback, 0.0, 10.0)),
        "ret3_side": float(ret3 * side),
        "ret6_side": float(ret6 * side),
        "ret12_side": float(ret12 * side),
        "ret3_abs": float(abs(ret3)),
        "ret6_abs": float(abs(ret6)),
        "quality": float(row.get("tabm_quality_for_action", 0.0)),
        "router_confidence": float(row.get("tabm_router_confidence", 0.0)),
        "router_margin": float(row.get("tabm_router_margin", 0.0)),
        "dir_confidence": float(row.get("tabm_dir_confidence", 0.0)),
        "dir_side_edge": float(row.get("tabm_dir_side_edge", 0.0)),
        "dir_trade_prob": float(row.get("tabm_dir_trade_prob", 0.0)),
        "p_long_minus_short": float(row.get("tabm_dir_p_long", 0.0) - row.get("tabm_dir_p_short", 0.0)),
        "atr14_pct": float(row.get("atr14_pct", 0.0)),
        "bar_range_pct": float(row.get("bar_range_pct", 0.0)),
        "ema9_21_gap_side": float(row.get("ema9_21_gap", 0.0) * side),
        "tod_sin": float(row.get("tod_sin", 0.0)),
        "tod_cos": float(row.get("tod_cos", 0.0)),
    }


def _immediate_return(cash: float, arrays: dict[str, np.ndarray], pos: base.Position, i: int, fee_eff: float, slip_eff: float) -> float:
    before = max(float(cash), 1e-12)
    new_cash, _new_pos, _ = base._close_fraction(float(cash), arrays, pos, int(i), 1.0, fee_eff, slip_eff)
    return float(new_cash / before - 1.0)


def _extended_return(
    cash: float,
    arrays: dict[str, np.ndarray],
    pos: base.Position,
    i: int,
    *,
    fee_eff: float,
    slip_eff: float,
    template: RunnerTemplate,
) -> float:
    before = max(float(cash), 1e-12)
    p = base.Position(**pos.__dict__)
    old_tp = float(p.take_profit)
    p.floor_unreal = max(float(p.floor_unreal), old_tp * float(template.floor_frac))
    p.take_profit = old_tp * float(template.extend_mult)
    extensions = 1
    last_i = len(arrays["close"]) - 1
    for j in range(int(i), last_i):
        unreal = base._unreal(arrays, p, j, slip_eff)
        p.mfe = max(p.mfe, unreal)
        p.mae = min(p.mae, unreal)
        if p.take_profit > 0.0 and unreal >= p.take_profit:
            if extensions < int(template.max_extensions):
                extensions += 1
                old_tp = float(p.take_profit)
                p.floor_unreal = max(float(p.floor_unreal), old_tp * float(template.floor_frac))
                p.take_profit = old_tp * float(template.extend_mult)
                continue
            new_cash, _new_pos, _ = base._close_fraction(float(cash), arrays, p, j, 1.0, fee_eff, slip_eff)
            return float(new_cash / before - 1.0)
        if p.floor_unreal > -abs(p.stop_loss) and unreal <= p.floor_unreal:
            new_cash, _new_pos, _ = base._close_fraction(float(cash), arrays, p, j, 1.0, fee_eff, slip_eff)
            return float(new_cash / before - 1.0)
        if p.stop_loss > 0.0 and unreal <= -abs(p.stop_loss):
            new_cash, _new_pos, _ = base._close_fraction(float(cash), arrays, p, j, 1.0, fee_eff, slip_eff)
            return float(new_cash / before - 1.0)
    new_cash, _new_pos, _ = base._close_fraction(float(cash), arrays, p, last_i, 1.0, fee_eff, slip_eff)
    return float(new_cash / before - 1.0)


def _collect_events(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    template: RunnerTemplate,
) -> pd.DataFrame:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    pos = base.Position()
    events: list[dict[str, Any]] = []
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            reason = base._hit_reason(unreal, pos)
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                feat = _event_features(frame, state, pos, i, unreal)
                imm = _immediate_return(cash, arrays, base.Position(**pos.__dict__), i, fee_eff, slip_eff)
                ext = _extended_return(cash, arrays, base.Position(**pos.__dict__), i, fee_eff=fee_eff, slip_eff=slip_eff, template=template)
                allowed = (
                    feat["quality"] >= float(template.quality_min)
                    and feat["ret3_side"] > float(template.momentum_min)
                )
                events.append(
                    {
                        "event_i": int(i),
                        "entry_signal_i": int(pos.entry_signal_i),
                        "immediate_ret": float(imm),
                        "extended_ret": float(ext),
                        "edge": float(ext - imm),
                        "label": int(allowed and (ext > imm + 0.001)),
                        "allowed": int(allowed),
                        **feat,
                    }
                )
            if reason:
                cash, pos, _ = base._close_fraction(cash, arrays, pos, i, 1.0, fee_eff, slip_eff)
            continue
        if bool(active[i]):
            cash, pos, _entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
    return pd.DataFrame(events)


def _model(seed: int, kind: str):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.35, class_weight="balanced", random_state=int(seed), max_iter=500))
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=80,
            max_depth=2,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=int(seed),
        )
    if kind == "et":
        return ExtraTreesClassifier(
            n_estimators=120,
            max_depth=2,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=int(seed),
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=40,
            max_leaf_nodes=4,
            l2_regularization=1.0,
            learning_rate=0.05,
            random_state=int(seed),
        )
    raise RuntimeError(f"unknown model kind: {kind}")


def _train_selector(events: pd.DataFrame, *, seed: int, kind: str, train_frac: float) -> tuple[Any | None, list[str], dict[str, Any]]:
    if events.empty:
        return None, [], {"reason": "empty_events"}
    feature_cols = [c for c in events.columns if c not in {"event_i", "entry_signal_i", "immediate_ret", "extended_ret", "edge", "label", "allowed"}]
    n_train = max(4, int(len(events) * float(train_frac)))
    train = events.iloc[:n_train].copy()
    y = train["label"].astype(int).to_numpy()
    diag = {
        "events": int(len(events)),
        "train_events": int(len(train)),
        "positive": int(events["label"].sum()),
        "train_positive": int(y.sum()),
        "feature_cols": feature_cols,
    }
    if len(np.unique(y)) < 2:
        return None, feature_cols, {**diag, "reason": "single_class_train"}
    clf = _model(seed, kind)
    clf.fit(train[feature_cols].to_numpy(dtype=np.float64), y)
    return clf, feature_cols, diag


def _selector_allowed(
    clf: Any | None,
    feature_cols: list[str],
    frame: pd.DataFrame,
    state: pd.DataFrame,
    pos: base.Position,
    i: int,
    unreal: float,
    *,
    template: RunnerTemplate,
    proba_min: float,
) -> bool:
    feat = _event_features(frame, state, pos, i, unreal)
    if feat["quality"] < float(template.quality_min) or feat["ret3_side"] <= float(template.momentum_min):
        return False
    if clf is None:
        return False
    x = np.asarray([[float(feat[c]) for c in feature_cols]], dtype=np.float64)
    if hasattr(clf, "predict_proba"):
        p = float(clf.predict_proba(x)[0, 1])
    else:
        p = float(clf.predict(x)[0])
    return p >= float(proba_min)


def _simulate_meta(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    template: RunnerTemplate,
    clf: Any | None,
    feature_cols: list[str],
    proba_min: float,
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
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    seeds = [260610, 260611, 260612, 260613, 260614, 260615, 260616, 260617]
    kinds = ["logit", "rf", "et", "hgb"]
    proba_mins = [0.55, 0.65, 0.75, 0.85]

    base_val, base_val_ledger = _simulate_meta(
        data["validation"]["frame"],
        data["validation"]["dec"],
        data["validation"]["state"],
        fee=float(data["validation"]["fee"]),
        slip=float(data["validation"]["slip"]),
        cost_mult=3.0,
        template=TEMPLATES[0],
        clf=None,
        feature_cols=[],
        proba_min=2.0,
    )
    base_oos, base_oos_ledger = _simulate_meta(
        data["oos"]["frame"],
        data["oos"]["dec"],
        data["oos"]["state"],
        fee=float(data["oos"]["fee"]),
        slip=float(data["oos"]["slip"]),
        cost_mult=3.0,
        template=TEMPLATES[0],
        clf=None,
        feature_cols=[],
        proba_min=2.0,
    )
    base_val_ledger.to_csv(OUT_DIR / "validation_baseline_ledger.csv", index=False)
    base_oos_ledger.to_csv(OUT_DIR / "oos_baseline_ledger.csv", index=False)
    rows: list[dict[str, Any]] = []
    event_summaries: dict[str, Any] = {}
    for template in TEMPLATES:
        events = _collect_events(
            data["validation"]["frame"],
            data["validation"]["dec"],
            data["validation"]["state"],
            fee=float(data["validation"]["fee"]),
            slip=float(data["validation"]["slip"]),
            cost_mult=3.0,
            template=template,
        )
        events.to_csv(OUT_DIR / f"{template.name}_validation_tp_events.csv", index=False)
        event_summaries[template.name] = {
            "events": int(len(events)),
            "positive": int(events["label"].sum()) if len(events) else 0,
            "mean_edge": float(events["edge"].mean()) if len(events) else 0.0,
        }
        for kind in kinds:
            for seed in seeds:
                clf, feature_cols, train_diag = _train_selector(events, seed=seed, kind=kind, train_frac=0.70)
                for proba_min in proba_mins:
                    val_m, _val_ledger = _simulate_meta(
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
                    )
                    oos_m, _oos_ledger = _simulate_meta(
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
                    )
                    rows.append(
                        {
                            "template": template.name,
                            "kind": kind,
                            "seed": int(seed),
                            "proba_min": float(proba_min),
                            **_row("val", val_m),
                            **_row("oos", oos_m),
                            "train_reason": train_diag.get("reason", "ok"),
                            "train_positive": int(train_diag.get("train_positive", 0)),
                            "events": int(train_diag.get("events", 0)),
                        }
                    )

    detail = pd.DataFrame(rows)
    detail["delta_val_pnl"] = detail["val_pnl"] - float(base_val["pnl"])
    detail["delta_oos_pnl"] = detail["oos_pnl"] - float(base_oos["pnl"])
    detail.to_csv(OUT_DIR / "meta_selector_seed_detail.csv", index=False)

    group_cols = ["template", "kind", "proba_min"]
    agg = detail.groupby(group_cols).agg(
        seeds=("seed", "nunique"),
        val_pnl_median=("val_pnl", "median"),
        val_pnl_min=("val_pnl", "min"),
        val_mdd_median=("val_mdd", "median"),
        val_wr_median=("val_wr", "median"),
        oos_pnl_median=("oos_pnl", "median"),
        oos_pnl_min=("oos_pnl", "min"),
        oos_mdd_median=("oos_mdd", "median"),
        oos_wr_median=("oos_wr", "median"),
        oos_trades_median=("oos_trades", "median"),
    ).reset_index()
    agg["delta_val_pnl_median"] = agg["val_pnl_median"] - float(base_val["pnl"])
    agg["delta_oos_pnl_median"] = agg["oos_pnl_median"] - float(base_oos["pnl"])
    agg["score"] = agg["oos_pnl_median"] + 0.45 * agg["val_pnl_median"] + 0.35 * agg["oos_mdd_median"] + 0.25 * agg["val_mdd_median"]
    agg = agg.sort_values(["oos_pnl_median", "val_pnl_median", "score"], ascending=False).reset_index(drop=True)
    agg.to_csv(OUT_DIR / "meta_selector_seed_ranking.csv", index=False)
    promotable = agg[
        (agg["oos_pnl_median"] > float(base_oos["pnl"]))
        & (agg["val_pnl_median"] > float(base_val["pnl"]) * 0.85)
        & (agg["oos_mdd_median"] >= float(base_oos["mdd"]) * 1.25)
        & (agg["val_mdd_median"] >= float(base_val["mdd"]) * 1.30)
    ].copy()
    promotable.to_csv(OUT_DIR / "meta_selector_promotable.csv", index=False)
    if len(agg):
        best = agg.iloc[0].to_dict()
        btemplate = next(t for t in TEMPLATES if t.name == best["template"])
        best_seed_row = detail[
            (detail["template"] == best["template"])
            & (detail["kind"] == best["kind"])
            & (detail["proba_min"] == best["proba_min"])
        ].sort_values(["oos_pnl", "val_pnl"], ascending=False).iloc[0]
        events = pd.read_csv(OUT_DIR / f"{btemplate.name}_validation_tp_events.csv")
        clf, feature_cols, _diag = _train_selector(events, seed=int(best_seed_row["seed"]), kind=str(best["kind"]), train_frac=0.70)
        for split in ("validation", "oos"):
            m, ledger = _simulate_meta(
                data[split]["frame"],
                data[split]["dec"],
                data[split]["state"],
                fee=float(data[split]["fee"]),
                slip=float(data[split]["slip"]),
                cost_mult=3.0,
                template=btemplate,
                clf=clf,
                feature_cols=feature_cols,
                proba_min=float(best["proba_min"]),
            )
            ledger.to_csv(OUT_DIR / f"{split}_best_seed_ledger.csv", index=False)
        LIVE_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_id": MODEL_ID,
            "status": "research_candidate_shadow_required",
            "selector_kind": str(best["kind"]),
            "selector_seed": int(best_seed_row["seed"]),
            "train_frac": 0.70,
            "proba_min": float(best["proba_min"]),
            "template": {
                "name": btemplate.name,
                "extend_mult": float(btemplate.extend_mult),
                "floor_frac": float(btemplate.floor_frac),
                "max_extensions": int(btemplate.max_extensions),
                "quality_min": float(btemplate.quality_min),
                "momentum_min": float(btemplate.momentum_min),
            },
            "feature_cols": list(feature_cols),
            "model": clf,
            "baseline": {"validation": base_val, "oos": base_oos},
            "selected_metrics": {
                "validation": _row("val", _simulate_meta(
                    data["validation"]["frame"],
                    data["validation"]["dec"],
                    data["validation"]["state"],
                    fee=float(data["validation"]["fee"]),
                    slip=float(data["validation"]["slip"]),
                    cost_mult=3.0,
                    template=btemplate,
                    clf=clf,
                    feature_cols=feature_cols,
                    proba_min=float(best["proba_min"]),
                )[0]),
                "oos": _row("oos", _simulate_meta(
                    data["oos"]["frame"],
                    data["oos"]["dec"],
                    data["oos"]["state"],
                    fee=float(data["oos"]["fee"]),
                    slip=float(data["oos"]["slip"]),
                    cost_mult=3.0,
                    template=btemplate,
                    clf=clf,
                    feature_cols=feature_cols,
                    proba_min=float(best["proba_min"]),
                )[0]),
            },
        }
        joblib.dump(bundle, LIVE_BUNDLE_DIR / "tp_runner_meta_selector.joblib")

    report = {
        "model_id": MODEL_ID,
        "baseline": {"validation": base_val, "oos": base_oos},
        "event_summaries": event_summaries,
        "promotable_count": int(len(promotable)),
        "top": agg.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "detail": str(OUT_DIR / "meta_selector_seed_detail.csv"),
            "ranking": str(OUT_DIR / "meta_selector_seed_ranking.csv"),
            "promotable": str(OUT_DIR / "meta_selector_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
            "live_bundle": str(LIVE_BUNDLE_DIR / "tp_runner_meta_selector.joblib"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top5": agg.head(5).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
