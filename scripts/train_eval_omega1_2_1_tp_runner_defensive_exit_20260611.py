#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
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


MODEL_ID = "omega1_2_1_tp_runner_defensive_exit_20260611"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TP_RUNNER_BUNDLE = ROOT / "data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib"


@dataclass(frozen=True)
class DefensiveRule:
    name: str
    min_unreal: float
    min_tp_progress: float
    giveback_frac: float
    ret3_side_max: float
    quality_max: float


RULES = (
    DefensiveRule("rule_gb35_prog60", 0.020, 0.60, 0.35, -999.0, -999.0),
    DefensiveRule("rule_gb45_prog70", 0.025, 0.70, 0.45, -999.0, -999.0),
    DefensiveRule("rule_mom_prog70", 0.025, 0.70, 10.0, -0.0010, -999.0),
    DefensiveRule("rule_quality_prog70", 0.025, 0.70, 10.0, -999.0, 0.62),
    DefensiveRule("rule_combo_soft", 0.020, 0.60, 0.50, -0.0005, 0.65),
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


def _position_features(frame: pd.DataFrame, state: pd.DataFrame, pos: base.Position, i: int, unreal: float) -> dict[str, float]:
    feat = meta._event_features(frame, state, pos, i, unreal)
    tp = max(float(pos.take_profit), 1e-8)
    sl = max(abs(float(pos.stop_loss)), 1e-8)
    feat.update(
        {
            "tp_progress": float(unreal / tp),
            "sl_progress": float(-unreal / sl),
            "dist_tp": float(pos.take_profit - unreal),
            "dist_sl": float(unreal + abs(pos.stop_loss)),
            "floor_unreal": float(pos.floor_unreal),
            "reduced": float(getattr(pos, "reduced", 0)),
            "tightened": float(getattr(pos, "tightened", 0)),
        }
    )
    return feat


def _close_return(cash: float, arrays: dict[str, np.ndarray], pos: base.Position, i: int, fee_eff: float, slip_eff: float) -> float:
    before = max(float(cash), 1e-12)
    new_cash, _new_pos, _ = base._close_fraction(float(cash), arrays, base.Position(**pos.__dict__), int(i), 1.0, fee_eff, slip_eff)
    return float(new_cash / before - 1.0)


def _continue_return(cash: float, arrays: dict[str, np.ndarray], pos: base.Position, i: int, fee_eff: float, slip_eff: float) -> float:
    before = max(float(cash), 1e-12)
    p = base.Position(**pos.__dict__)
    last_i = len(arrays["close"]) - 1
    for j in range(int(i), last_i):
        unreal = base._unreal(arrays, p, j, slip_eff)
        p.mfe = max(p.mfe, unreal)
        p.mae = min(p.mae, unreal)
        reason = base._hit_reason(unreal, p)
        if reason:
            new_cash, _new_pos, _ = base._close_fraction(float(cash), arrays, p, j, 1.0, fee_eff, slip_eff)
            return float(new_cash / before - 1.0)
    new_cash, _new_pos, _ = base._close_fraction(float(cash), arrays, p, last_i, 1.0, fee_eff, slip_eff)
    return float(new_cash / before - 1.0)


def _collect_defensive_events(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    min_unreal: float,
    min_hold: int,
    edge: float,
    sample_stride: int,
    max_events: int,
) -> pd.DataFrame:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    pos = base.Position()
    rows: list[dict[str, Any]] = []
    candidates_seen = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            reason = base._hit_reason(unreal, pos)
            hold = int(i) - int(pos.entry_i)
            if not reason and unreal >= float(min_unreal) and hold >= int(min_hold):
                candidates_seen += 1
                if (candidates_seen - 1) % max(int(sample_stride), 1) == 0:
                    immediate = _close_return(cash, arrays, pos, i, fee_eff, slip_eff)
                    cont = _continue_return(cash, arrays, pos, i, fee_eff, slip_eff)
                    rows.append(
                        {
                            "event_i": int(i),
                            "entry_signal_i": int(pos.entry_signal_i),
                            "immediate_ret": float(immediate),
                            "continue_ret": float(cont),
                            "edge": float(immediate - cont),
                            "label": int(immediate > cont + float(edge)),
                            **_position_features(frame, state, pos, i, unreal),
                        }
                    )
                    if int(max_events) > 0 and len(rows) >= int(max_events):
                        break
            if reason:
                cash, pos, _ = base._close_fraction(cash, arrays, pos, i, 1.0, fee_eff, slip_eff)
            continue
        if bool(active[i]):
            cash, pos, _entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
    return pd.DataFrame(rows)


def _train_defensive_model(events: pd.DataFrame, *, kind: str, seed: int) -> tuple[Any | None, list[str], dict[str, Any]]:
    if events.empty:
        return None, [], {"reason": "empty_events"}
    drop = {"event_i", "entry_signal_i", "immediate_ret", "continue_ret", "edge", "label"}
    feature_cols = [c for c in events.columns if c not in drop]
    train = events.iloc[: max(8, int(len(events) * 0.70))].copy()
    y = train["label"].astype(int).to_numpy()
    diag = {
        "events": int(len(events)),
        "positive": int(events["label"].sum()),
        "train_events": int(len(train)),
        "train_positive": int(y.sum()),
        "feature_cols": feature_cols,
    }
    if len(np.unique(y)) < 2:
        return None, feature_cols, {**diag, "reason": "single_class_train"}
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=45,
            max_leaf_nodes=5,
            l2_regularization=1.5,
            learning_rate=0.045,
            random_state=int(seed),
        )
    elif kind == "et":
        clf = ExtraTreesClassifier(
            n_estimators=160,
            max_depth=3,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=int(seed),
        )
    else:
        raise RuntimeError(f"unknown defensive model kind: {kind}")
    clf.fit(train[feature_cols].to_numpy(dtype=np.float64), y)
    return clf, feature_cols, {**diag, "reason": "ok", "kind": kind, "seed": int(seed)}


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


def _rule_exit(rule: DefensiveRule, frame: pd.DataFrame, state: pd.DataFrame, pos: base.Position, i: int, unreal: float) -> bool:
    feat = _position_features(frame, state, pos, i, unreal)
    if unreal < float(rule.min_unreal) or feat["tp_progress"] < float(rule.min_tp_progress):
        return False
    checks = []
    if rule.giveback_frac < 9.0:
        checks.append(feat["giveback"] >= float(rule.giveback_frac))
    if rule.ret3_side_max > -900.0:
        checks.append(feat["ret3_side"] <= float(rule.ret3_side_max))
    if rule.quality_max > -900.0:
        checks.append(feat["quality"] <= float(rule.quality_max))
    return bool(checks and any(checks))


def _model_exit(
    clf: Any | None,
    feature_cols: list[str],
    frame: pd.DataFrame,
    state: pd.DataFrame,
    pos: base.Position,
    i: int,
    unreal: float,
    *,
    proba_min: float,
    min_unreal: float,
    min_tp_progress: float,
) -> bool:
    feat = _position_features(frame, state, pos, i, unreal)
    if feat["unreal"] < float(min_unreal) or feat["tp_progress"] < float(min_tp_progress):
        return False
    if clf is None:
        return False
    x = np.asarray([[float(feat[c]) for c in feature_cols]], dtype=np.float64)
    if hasattr(clf, "predict_proba"):
        p = float(clf.predict_proba(x)[0, 1])
    else:
        p = float(clf.predict(x)[0])
    return p >= float(proba_min)


def _simulate(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    tp_bundle: dict[str, Any] | None,
    rule: DefensiveRule | None,
    clf: Any | None,
    feature_cols: list[str],
    proba_min: float,
    min_unreal: float,
    min_tp_progress: float,
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
            elif rule is not None and _rule_exit(rule, frame, state, pos, i, unreal):
                reason = "defensive_rule_lower_tp_exit"
            elif _model_exit(
                clf,
                feature_cols,
                frame,
                state,
                pos,
                i,
                unreal,
                proba_min=proba_min,
                min_unreal=min_unreal,
                min_tp_progress=min_tp_progress,
            ):
                reason = "defensive_model_lower_tp_exit"

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
    data = runner._build()
    tp_bundle = joblib.load(TP_RUNNER_BUNDLE) if TP_RUNNER_BUNDLE.exists() else None

    val_events = _collect_defensive_events(
        data["validation"]["frame"],
        data["validation"]["dec"],
        data["validation"]["state"],
        fee=float(data["validation"]["fee"]),
        slip=float(data["validation"]["slip"]),
        cost_mult=3.0,
        min_unreal=0.020,
        min_hold=3,
        edge=0.001,
        sample_stride=2,
        max_events=1500,
    )
    val_events.to_csv(OUT_DIR / "validation_defensive_exit_events.csv", index=False)

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}

    configs: list[dict[str, Any]] = [
        {"variant": "baseline_no_runner", "tp_bundle": None, "rule": None, "clf": None, "feature_cols": [], "proba_min": 2.0, "min_unreal": 999.0, "min_tp_progress": 999.0},
        {"variant": "tp_runner_only", "tp_bundle": tp_bundle, "rule": None, "clf": None, "feature_cols": [], "proba_min": 2.0, "min_unreal": 999.0, "min_tp_progress": 999.0},
    ]
    for rule in (RULES[1], RULES[4]):
        configs.append({"variant": f"tp_runner_{rule.name}", "tp_bundle": tp_bundle, "rule": rule, "clf": None, "feature_cols": [], "proba_min": 2.0, "min_unreal": 999.0, "min_tp_progress": 999.0})

    model_diags: dict[str, Any] = {}
    for kind in ("et", "hgb"):
        for seed in (260613,):
            clf, feature_cols, diag = _train_defensive_model(val_events, kind=kind, seed=seed)
            model_diags[f"{kind}_{seed}"] = diag
            for proba_min in (0.65,):
                for min_prog in (0.60, 0.75):
                    configs.append(
                        {
                            "variant": f"tp_runner_model_{kind}_s{seed}_p{str(proba_min).replace('.', '')}_prog{str(min_prog).replace('.', '')}",
                            "tp_bundle": tp_bundle,
                            "rule": None,
                            "clf": clf,
                            "feature_cols": feature_cols,
                            "proba_min": float(proba_min),
                            "min_unreal": 0.010,
                            "min_tp_progress": float(min_prog),
                        }
                    )

    for idx, cfg in enumerate(configs):
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
                rule=cfg["rule"],
                clf=cfg["clf"],
                feature_cols=cfg["feature_cols"],
                proba_min=float(cfg["proba_min"]),
                min_unreal=float(cfg["min_unreal"]),
                min_tp_progress=float(cfg["min_tp_progress"]),
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
    ranking.to_csv(OUT_DIR / "defensive_exit_ranking.csv", index=False)

    for variant_id in sorted(set([0, 1] + [int(x) for x in ranking["variant_id"].head(5).tolist()])):
        for split, ledger in ledgers[str(variant_id)].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{variant_id}_ledger.csv", index=False)

    promotable = ranking[
        (ranking["oos_pnl"] > base_oos)
        & (ranking["validation_pnl"] > base_val * 0.85)
        & (ranking["oos_mdd"] >= float(ranking.loc[ranking["variant"].eq("baseline_no_runner"), "oos_mdd"].iloc[0]) * 1.25)
        & (ranking["validation_mdd"] >= float(ranking.loc[ranking["variant"].eq("baseline_no_runner"), "validation_mdd"].iloc[0]) * 1.30)
    ].copy()
    promotable.to_csv(OUT_DIR / "defensive_exit_promotable.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "method": "TP-runner extension plus defensive lower-TP/early-exit overlay. Parent entries and risk contract are frozen.",
        "baseline": ranking[ranking["variant"].eq("baseline_no_runner")].to_dict(orient="records")[0],
        "tp_runner_only": ranking[ranking["variant"].eq("tp_runner_only")].to_dict(orient="records")[0],
        "event_diag": {
            "validation_events": int(len(val_events)),
            "validation_positive": int(val_events["label"].sum()) if len(val_events) else 0,
            "validation_mean_edge": float(val_events["edge"].mean()) if len(val_events) else 0.0,
        },
        "model_diags": model_diags,
        "promotable_count": int(len(promotable)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "defensive_exit_ranking.csv"),
            "events": str(OUT_DIR / "validation_defensive_exit_events.csv"),
            "promotable": str(OUT_DIR / "defensive_exit_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
