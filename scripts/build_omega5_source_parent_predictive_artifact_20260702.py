#!/usr/bin/env python3
"""Build a runtime-native predictive source-parent artifact for Omega5.

This script intentionally does not train on the OOS split for the proof model.
It trains a validation-only distillation model, evaluates that model on OOS,
then separately writes a live artifact trained on validation+OOS for runtime
use.  The live artifact is marked as a deployment artifact, not an OOS score
source.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.prove_omega5_runtime_native_walkforward_20260701 import (  # noqa: E402
    BASE_FEATURES,
    SPLIT_WINDOWS,
    TRADE_CANDIDATES,
    load_enriched_frame,
    prepare_replay_frame,
)


MODEL_ID = "omega4_6_2_source_parent_predictive_distill_20260702"
SOURCE_PARENT_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_POLICY_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
PROOF_BUNDLE = OUT_DIR / "proof_validation_only_bundle.joblib"
LIVE_BUNDLE = OUT_DIR / "live_val_oos_bundle.joblib"
REPORT_JSON = OUT_DIR / "report.json"
REPORT_MD = ROOT / "docs/audits/omega5_source_parent_predictive_distill_20260702.md"
SOURCE_PARENT_REPORT = ROOT / "tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701/report.json"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701/report.json"
LIVE_SNAPSHOT = ROOT / "data/live/decision_feature_frame_snapshot.pkl.gz"
EPS = 1.0e-12

FORBIDDEN_FEATURE_SUBSTRINGS = (
    "timestamp",
    "entry_timestamp",
    "exit_timestamp",
    "entry_timestamp_dt",
    "exit_timestamp_dt",
    "trade_return",
    "net_per_notional",
    "raw_exit_price_move",
    "mfe_price_move",
    "mae_price_move",
    "take_profit",
    "stop_loss",
    "reason",
    "source_ledger",
    "source_report",
    "source_alias",
    "win",
    "label",
    "target",
)


@dataclass(frozen=True)
class SplitData:
    split: str
    frame: pd.DataFrame
    labels: pd.DataFrame


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _selected_artifacts(report_path: Path) -> dict[str, Path]:
    report = _read_json(report_path)
    artifacts = dict(report["artifacts"])
    return {
        "validation": _resolve(artifacts["selected_validation_ledger"]),
        "oos": _resolve(artifacts["selected_oos_ledger"]),
    }


def _window_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    start_raw, end_raw = SPLIT_WINDOWS[split]
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    mask = (ts >= pd.Timestamp(start_raw)) & (ts <= pd.Timestamp(end_raw))
    out = frame.loc[mask].copy().reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"{split}: empty feature window")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    return out


def _load_policy_tables() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    parent_paths = _selected_artifacts(SOURCE_PARENT_REPORT)
    reference_paths = _selected_artifacts(REFERENCE_REPORT)
    parent: dict[str, pd.DataFrame] = {}
    reference: dict[str, pd.DataFrame] = {}
    for split in ("validation", "oos"):
        p = pd.read_csv(parent_paths[split])
        r = pd.read_csv(reference_paths[split])
        for df in (p, r):
            df["entry_dt"] = pd.to_datetime(df["entry_timestamp"], errors="raise")
            df["exit_dt"] = pd.to_datetime(df["exit_timestamp"], errors="raise")
            df["side"] = pd.to_numeric(df["side"], errors="coerce").fillna(0).astype(int)
            df["notional"] = pd.to_numeric(df["notional"], errors="coerce").fillna(0.0)
        parent[split] = p.sort_values(["entry_dt", "exit_dt"]).reset_index(drop=True)
        reference[split] = r.sort_values(["entry_dt", "exit_dt"]).reset_index(drop=True)
    return parent, reference


def _active_parent_at(parent_df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    rows = parent_df[(parent_df["entry_dt"] <= ts) & (ts < parent_df["exit_dt"])]
    rows = rows[pd.to_numeric(rows["notional"], errors="coerce").fillna(0.0) > EPS]
    if len(rows) == 0:
        return None
    if len(rows) > 1:
        raise RuntimeError(f"multiple active source-parent intervals at {ts}: {len(rows)}")
    return rows.iloc[0]


def _labels_for_split(frame: pd.DataFrame, split: str, parent_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    labels = pd.DataFrame(
        {
            "timestamp": ts,
            "action_class": np.zeros(len(frame), dtype=np.int8),
            "side": np.zeros(len(frame), dtype=np.int8),
            "notional": np.zeros(len(frame), dtype=float),
            "leverage": np.ones(len(frame), dtype=float),
            "margin_fraction": np.zeros(len(frame), dtype=float),
            "roundtrip_cost": np.zeros(len(frame), dtype=float),
            "event": np.zeros(len(frame), dtype=np.int8),
        }
    )
    index_by_ts = {pd.Timestamp(v): i for i, v in enumerate(labels["timestamp"])}
    for _, ref in reference_df.iterrows():
        event_ts = pd.Timestamp(ref["entry_dt"])
        i = index_by_ts.get(event_ts)
        if i is None or float(ref["notional"]) <= EPS:
            continue
        parent_row = _active_parent_at(parent_df, event_ts)
        if parent_row is None:
            continue
        side = int(parent_row["side"])
        if side not in {-1, 1}:
            continue
        if int(ref["side"]) != side:
            continue
        notional = float(parent_row["notional"])
        leverage = float(parent_row["leverage"])
        margin = float(parent_row["margin_fraction"])
        if notional <= EPS or leverage <= EPS or margin <= EPS:
            continue
        labels.loc[i, "action_class"] = 1 if side > 0 else 2
        labels.loc[i, "side"] = side
        labels.loc[i, "notional"] = notional
        labels.loc[i, "leverage"] = leverage
        labels.loc[i, "margin_fraction"] = margin
        labels.loc[i, "roundtrip_cost"] = float(ref.get("roll8_roundtrip_cost", 0.000612) or 0.000612)
        labels.loc[i, "event"] = 1
    return labels


def _load_split(split: str, parent_df: pd.DataFrame, reference_df: pd.DataFrame) -> SplitData:
    frame = _window_frame(prepare_replay_frame(load_enriched_frame(split)), split)
    labels = _labels_for_split(frame, split, parent_df, reference_df)
    return SplitData(split=split, frame=frame, labels=labels)


def _numeric_feature_columns(frames: list[pd.DataFrame], live_frame: pd.DataFrame | None) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    if live_frame is not None:
        common &= set(live_frame.columns)
    out: list[str] = []
    for col in sorted(common):
        low = col.lower()
        if any(token in low for token in FORBIDDEN_FEATURE_SUBSTRINGS):
            continue
        if all(pd.api.types.is_numeric_dtype(frame[col]) for frame in frames):
            out.append(col)
    if not out:
        raise RuntimeError("no common numeric feature columns")
    return out


def _matrix(frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    x = frame[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x.fillna(0.0).to_numpy(dtype=np.float32)


def _sample_weights(y: np.ndarray) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float32)
    pos = y != 0
    pos_count = max(int(pos.sum()), 1)
    neg_count = max(int((~pos).sum()), 1)
    weights[pos] = min(float(neg_count / pos_count), 100.0)
    return weights


def _train_bundle(train_data: SplitData, feature_cols: list[str], *, train_tag: str) -> dict[str, Any]:
    x = _matrix(train_data.frame, feature_cols)
    y = train_data.labels["action_class"].to_numpy(dtype=np.int8)
    weights = _sample_weights(y)
    classifier = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.035,
        max_leaf_nodes=31,
        min_samples_leaf=12,
        l2_regularization=0.05,
        random_state=4625,
    )
    classifier.fit(x, y, sample_weight=weights)
    pos = y != 0
    if int(pos.sum()) >= 8:
        regressor = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=4,
            l2_regularization=0.05,
            random_state=4626,
        )
        regressor.fit(x[pos], train_data.labels.loc[pos, "notional"].to_numpy(dtype=float))
        median_notional = float(train_data.labels.loc[pos, "notional"].median())
    else:
        regressor = None
        median_notional = float(train_data.labels.loc[pos, "notional"].median()) if int(pos.sum()) else 0.0
    roundtrip_cost = float(train_data.labels.loc[pos, "roundtrip_cost"].replace(0.0, np.nan).median())
    if not np.isfinite(roundtrip_cost):
        roundtrip_cost = 0.000612
    return {
        "model_id": MODEL_ID,
        "source_parent_model_id": SOURCE_PARENT_ID,
        "reference_policy_model_id": REFERENCE_POLICY_ID,
        "train_tag": train_tag,
        "feature_cols": feature_cols,
        "classifier": classifier,
        "notional_regressor": regressor,
        "median_notional": median_notional,
        "leverage_cap": 5.0,
        "max_margin_fraction": 1.0,
        "max_hold_bars": int(90 * 12),
        "roundtrip_cost": roundtrip_cost,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _threshold_search(bundle: dict[str, Any], data: SplitData) -> dict[str, Any]:
    x = _matrix(data.frame, bundle["feature_cols"])
    proba = bundle["classifier"].predict_proba(x)
    classes = list(bundle["classifier"].classes_)
    class_to_col = {int(c): i for i, c in enumerate(classes)}
    long_p = proba[:, class_to_col.get(1, 0)] if 1 in class_to_col else np.zeros(len(data.frame))
    short_p = proba[:, class_to_col.get(2, 0)] if 2 in class_to_col else np.zeros(len(data.frame))
    y_true = data.labels["action_class"].to_numpy(dtype=np.int8)
    best: dict[str, Any] | None = None
    for threshold in np.linspace(0.01, 0.99, 99):
        y_pred = np.zeros(len(y_true), dtype=np.int8)
        choose_long = (long_p >= short_p) & (long_p >= threshold)
        choose_short = (short_p > long_p) & (short_p >= threshold)
        y_pred[choose_long] = 1
        y_pred[choose_short] = 2
        event_f1 = f1_score(y_true != 0, y_pred != 0, zero_division=0)
        side_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        pred_events = int((y_pred != 0).sum())
        true_events = int((y_true != 0).sum())
        count_penalty = abs(pred_events - true_events) / max(true_events, 1)
        score = float(event_f1 + 0.25 * side_f1 - 0.05 * count_penalty)
        candidate = {
            "threshold": float(threshold),
            "score": score,
            "event_f1": float(event_f1),
            "side_macro_f1": float(side_f1),
            "pred_events": pred_events,
            "true_events": true_events,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is None:
        raise RuntimeError("threshold search failed")
    bundle["entry_probability_threshold"] = float(best["threshold"])
    return best


def _predict_classes(bundle: dict[str, Any], data: SplitData) -> tuple[np.ndarray, np.ndarray]:
    x = _matrix(data.frame, bundle["feature_cols"])
    proba = bundle["classifier"].predict_proba(x)
    classes = list(bundle["classifier"].classes_)
    class_to_col = {int(c): i for i, c in enumerate(classes)}
    long_p = proba[:, class_to_col.get(1, 0)] if 1 in class_to_col else np.zeros(len(data.frame))
    short_p = proba[:, class_to_col.get(2, 0)] if 2 in class_to_col else np.zeros(len(data.frame))
    threshold = float(bundle["entry_probability_threshold"])
    y_pred = np.zeros(len(data.frame), dtype=np.int8)
    choose_long = (long_p >= short_p) & (long_p >= threshold)
    choose_short = (short_p > long_p) & (short_p >= threshold)
    y_pred[choose_long] = 1
    y_pred[choose_short] = 2
    return y_pred, np.maximum(long_p, short_p)


def _evaluate(bundle: dict[str, Any], data: SplitData) -> dict[str, Any]:
    y_true = data.labels["action_class"].to_numpy(dtype=np.int8)
    y_pred, confidence = _predict_classes(bundle, data)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true != 0, y_pred != 0, average="binary", zero_division=0)
    side_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "split": data.split,
        "rows": int(len(y_true)),
        "true_events": int((y_true != 0).sum()),
        "pred_events": int((y_pred != 0).sum()),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "side_macro_f1": float(side_macro_f1),
        "mean_pred_confidence": float(np.mean(confidence[y_pred != 0])) if int((y_pred != 0).sum()) else 0.0,
        "confusion_matrix_labels_0_cash_1_long_2_short": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True),
    }


def _live_feature_coverage(feature_cols: list[str], live_frame: pd.DataFrame | None) -> dict[str, Any]:
    if live_frame is None:
        return {"live_snapshot_available": False, "missing": feature_cols, "coverage": 0.0}
    missing = sorted(set(feature_cols) - set(live_frame.columns))
    nonfinite: list[str] = []
    if len(live_frame):
        last = live_frame.iloc[-1]
        for col in feature_cols:
            if col in live_frame.columns:
                try:
                    v = float(last[col])
                    if not np.isfinite(v):
                        nonfinite.append(col)
                except Exception:
                    nonfinite.append(col)
    return {
        "live_snapshot_available": True,
        "live_snapshot_path": str(LIVE_SNAPSHOT),
        "feature_count": len(feature_cols),
        "missing": missing,
        "nonfinite_latest": sorted(nonfinite),
        "coverage": float((len(feature_cols) - len(missing)) / max(len(feature_cols), 1)),
        "latest_timestamp": str(live_frame.iloc[-1].get("timestamp", "")) if len(live_frame) else "",
    }


def _load_live_snapshot_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("frame"), pd.DataFrame):
        return obj["frame"]
    raise RuntimeError(f"unsupported live snapshot payload: {path} type={type(obj).__name__}")


def _write_report(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Omega5 Source Parent Predictive Distill - 2026-07-02",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Proof bundle: `{payload['artifacts']['proof_bundle']}`",
        f"- Live bundle: `{payload['artifacts']['live_bundle']}`",
        f"- Feature count: `{payload['feature_count']}`",
        f"- Live feature coverage: `{payload['live_feature_coverage']['coverage']:.4f}`",
        "",
        "## Proof Metrics",
        "",
        f"- Validation event F1: `{payload['proof_metrics']['validation']['event_f1']:.4f}`",
        f"- OOS event F1: `{payload['proof_metrics']['oos']['event_f1']:.4f}`",
        f"- OOS predicted/true events: `{payload['proof_metrics']['oos']['pred_events']}` / `{payload['proof_metrics']['oos']['true_events']}`",
        "",
        "## Contract",
        "",
        "- Proof model trains on validation only and evaluates OOS without OOS training.",
        "- Live model trains on validation+OOS for deployment; it is not used as OOS proof.",
        "- The artifact predicts current-bar source-parent action/side/notional from causal features, not historical policy rows.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parent_tables, reference_tables = _load_policy_tables()
    validation = _load_split("validation", parent_tables["validation"], reference_tables["validation"])
    oos = _load_split("oos", parent_tables["oos"], reference_tables["oos"])
    live_frame = _load_live_snapshot_frame(LIVE_SNAPSHOT)
    feature_cols = _numeric_feature_columns([validation.frame, oos.frame], live_frame)

    proof_bundle = _train_bundle(validation, feature_cols, train_tag="validation_only")
    threshold_info = _threshold_search(proof_bundle, validation)
    proof_metrics = {
        "threshold_selection": threshold_info,
        "validation": _evaluate(proof_bundle, validation),
        "oos": _evaluate(proof_bundle, oos),
    }

    combined = SplitData(
        split="validation_oos",
        frame=pd.concat([validation.frame, oos.frame], ignore_index=True),
        labels=pd.concat([validation.labels, oos.labels], ignore_index=True),
    )
    live_bundle = _train_bundle(combined, feature_cols, train_tag="validation_plus_oos_live_deployment")
    live_bundle["entry_probability_threshold"] = proof_bundle["entry_probability_threshold"]
    live_bundle["proof_report"] = str(REPORT_JSON)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(proof_bundle, PROOF_BUNDLE)
    joblib.dump(live_bundle, LIVE_BUNDLE)

    coverage = _live_feature_coverage(feature_cols, live_frame)
    pass_gate = (
        coverage["coverage"] >= 1.0
        and not coverage["nonfinite_latest"]
        and proof_metrics["oos"]["event_f1"] > 0.0
        and proof_metrics["oos"]["pred_events"] > 0
    )
    payload = {
        "model_id": MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "SOURCE_PARENT_PREDICTIVE_ARTIFACT_READY" if pass_gate else "SOURCE_PARENT_PREDICTIVE_ARTIFACT_NEEDS_REVIEW",
        "source_parent_model_id": SOURCE_PARENT_ID,
        "reference_policy_model_id": REFERENCE_POLICY_ID,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "live_feature_coverage": coverage,
        "proof_metrics": proof_metrics,
        "label_counts": {
            "validation_events": int(validation.labels["event"].sum()),
            "oos_events": int(oos.labels["event"].sum()),
            "combined_events": int(combined.labels["event"].sum()),
        },
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "proof_bundle": str(PROOF_BUNDLE),
            "live_bundle": str(LIVE_BUNDLE),
            "report_json": str(REPORT_JSON),
            "report_md": str(REPORT_MD),
            "trade_candidates": {k: str(v) for k, v in TRADE_CANDIDATES.items()},
            "base_features": {k: str(v) for k, v in BASE_FEATURES.items()},
            "source_parent_report": str(SOURCE_PARENT_REPORT),
            "reference_report": str(REFERENCE_REPORT),
        },
        "contract": {
            "proof_training": "validation_only",
            "proof_evaluation": "oos_holdout",
            "live_training": "validation_plus_oos",
            "historical_policy_row_lookup_used_by_live_artifact": False,
            "causal_feature_only_prediction": True,
        },
    }
    _write_report(payload)
    print(json.dumps({"verdict": payload["verdict"], "report": str(REPORT_JSON), "live_bundle": str(LIVE_BUNDLE)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
