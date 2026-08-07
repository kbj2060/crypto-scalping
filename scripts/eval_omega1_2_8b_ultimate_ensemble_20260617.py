#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_8b_paper_fixes_20260617 as paper  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616 as exp  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_8b_ultimate_ensemble_20260617"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BUNDLE_PATH = paper.BUNDLE_PATH


@dataclass(frozen=True)
class MemberPrediction:
    name: str
    motive: str
    val_action: np.ndarray
    val_conf: np.ndarray
    oos_action: np.ndarray
    oos_conf: np.ndarray
    val_metrics: dict[str, Any]
    oos_metrics: dict[str, Any]
    val_score: float
    oos_score: float


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _reason_count(metrics: dict[str, Any], key: str) -> int:
    reasons = metrics.get("exit_reasons") or {}
    if not isinstance(reasons, dict):
        return 0
    return int(reasons.get(key, 0) or 0)


def _selection_score(metrics: dict[str, Any], base: dict[str, Any]) -> float:
    pnl_delta = float(metrics["pnl"]) - float(base["pnl"])
    stop_loss = _reason_count(metrics, "fallback_stop_loss")
    takeover = _reason_count(metrics, "fallback_primary_takeover")
    entries = int(metrics.get("fallback_entries", 0) or 0)
    wr_drop = max(float(base["wr"]) - float(metrics["wr"]), 0.0)
    stop_rate = float(stop_loss / max(entries, 1))
    return (
        pnl_delta
        + 0.04 * entries
        + 8.0 * float(metrics["wr"])
        + 0.20 * float(metrics["mdd"])
        - 1.50 * stop_loss
        - 0.50 * takeover
        - 18.0 * wr_drop
        - 6.0 * stop_rate
    )


def _metric_record(
    name: str,
    family: str,
    val_m: dict[str, Any],
    oos_m: dict[str, Any],
    base_val: dict[str, Any],
    base_oos: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    row = {"candidate": name, "family": family, **params}
    row.update(sleeve._metric_row("val", val_m))
    row.update(sleeve._metric_row("oos", oos_m))
    row["val_delta_pnl"] = float(val_m["pnl"]) - float(base_val["pnl"])
    row["oos_delta_pnl"] = float(oos_m["pnl"]) - float(base_oos["pnl"])
    row["val_fallback_stop_loss"] = _reason_count(val_m, "fallback_stop_loss")
    row["oos_fallback_stop_loss"] = _reason_count(oos_m, "fallback_stop_loss")
    row["val_fallback_primary_takeover"] = _reason_count(val_m, "fallback_primary_takeover")
    row["oos_fallback_primary_takeover"] = _reason_count(oos_m, "fallback_primary_takeover")
    row["selection_score_val_only"] = _selection_score(val_m, base_val)
    row["diagnostic_score_oos"] = _selection_score(oos_m, base_oos)
    return row


def _weighted_vote(
    members: list[MemberPrediction],
    split: str,
    weights: np.ndarray,
    *,
    min_votes: int,
    min_weight: float,
    min_margin: float,
    max_entries: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    actions = [m.val_action if split == "val" else m.oos_action for m in members]
    confs = [m.val_conf if split == "val" else m.oos_conf for m in members]
    n = len(actions[0])
    long_score = np.zeros(n, dtype=np.float64)
    short_score = np.zeros(n, dtype=np.float64)
    long_votes = np.zeros(n, dtype=np.int64)
    short_votes = np.zeros(n, dtype=np.int64)
    for action, conf, weight in zip(actions, confs, weights):
        long_mask = action == sleeve.ACTION_LONG
        short_mask = action == sleeve.ACTION_SHORT
        long_score += long_mask.astype(np.float64) * conf * float(weight)
        short_score += short_mask.astype(np.float64) * conf * float(weight)
        long_votes += long_mask.astype(np.int64)
        short_votes += short_mask.astype(np.int64)
    best_long = long_score >= short_score
    best_score = np.where(best_long, long_score, short_score)
    other_score = np.where(best_long, short_score, long_score)
    best_votes = np.where(best_long, long_votes, short_votes)
    action = np.where(
        (best_votes >= int(min_votes))
        & (best_score >= float(min_weight))
        & ((best_score - other_score) >= float(min_margin)),
        np.where(best_long, sleeve.ACTION_LONG, sleeve.ACTION_SHORT),
        sleeve.ACTION_CASH,
    ).astype(np.int64)
    conf = np.clip(best_score / max(float(np.sum(weights)), 1.0e-12), 0.0, 1.0)
    conf = np.where(action != sleeve.ACTION_CASH, conf, 0.0).astype(np.float64)
    if max_entries is not None and int(max_entries) >= 0:
        active = np.flatnonzero(action != sleeve.ACTION_CASH)
        if len(active) > int(max_entries):
            keep = active[np.argsort(best_score[active])[::-1][: int(max_entries)]]
            gated_action = np.zeros_like(action)
            gated_conf = np.zeros_like(conf)
            gated_action[keep] = action[keep]
            gated_conf[keep] = conf[keep]
            action, conf = gated_action, gated_conf
    return action, conf


def _agreement_overlay(
    candidates: list[MemberPrediction],
    split: str,
    *,
    min_agree: int,
    max_entries: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    actions = [m.val_action if split == "val" else m.oos_action for m in candidates]
    confs = [m.val_conf if split == "val" else m.oos_conf for m in candidates]
    n = len(actions[0])
    long_votes = np.sum([a == sleeve.ACTION_LONG for a in actions], axis=0)
    short_votes = np.sum([a == sleeve.ACTION_SHORT for a in actions], axis=0)
    long_conf = np.mean([np.where(a == sleeve.ACTION_LONG, c, 0.0) for a, c in zip(actions, confs)], axis=0)
    short_conf = np.mean([np.where(a == sleeve.ACTION_SHORT, c, 0.0) for a, c in zip(actions, confs)], axis=0)
    best_long = long_votes >= short_votes
    best_votes = np.where(best_long, long_votes, short_votes)
    best_conf = np.where(best_long, long_conf, short_conf)
    action = np.where(
        best_votes >= int(min_agree),
        np.where(best_long, sleeve.ACTION_LONG, sleeve.ACTION_SHORT),
        sleeve.ACTION_CASH,
    ).astype(np.int64)
    conf = np.where(action != sleeve.ACTION_CASH, best_conf, 0.0).astype(np.float64)
    if max_entries is not None and int(max_entries) >= 0:
        active = np.flatnonzero(action != sleeve.ACTION_CASH)
        if len(active) > int(max_entries):
            keep = active[np.argsort(conf[active])[::-1][: int(max_entries)]]
            gated_action = np.zeros_like(action)
            gated_conf = np.zeros_like(conf)
            gated_action[keep] = action[keep]
            gated_conf[keep] = conf[keep]
            action, conf = gated_action, gated_conf
    return action, conf


def _variants() -> list[paper.Variant]:
    return [
        paper.Variant("live_contract_support_gate", "CQL/OOD support blocking + existing conformal lower-bound", 0.0, 0.0, None, None, 0.92, 8.0, 0.0, None),
        paper.Variant("spci_stricter_lower_bound", "SPCI-style more conservative residual lower-bound", 0.0015, 0.0, None, None, 0.92, 8.0, 0.0, None),
        paper.Variant("cql_strict_support", "CQL-style stricter behavior-support filter", 0.0, 0.0, None, None, 0.95, 6.0, 0.0, None),
        paper.Variant("cql_very_strict_support", "CQL-style high-confidence in-support only", 0.0, 0.0, None, None, 0.98, 4.0, 0.0, None),
        paper.Variant("utility_margin_conservative", "Conservative utility agreement and margin filter", 0.0, 0.0, 0.001, 0.001, 0.92, 8.0, 0.0, None),
        paper.Variant("mmdrex_router_confidence", "MM-DREX-inspired dynamic router confidence gate", 0.0, 0.0, None, None, 0.92, 8.0, 0.55, None),
        paper.Variant("combined_spci_cql_router", "Combined SPCI lower-bound + CQL support + router confidence", 0.001, 0.001, 0.001, 0.001, 0.95, 6.0, 0.50, None),
    ]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(BUNDLE_PATH)
    val_payload, oos_payload, meta = exp._build_payloads()
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)[list(bundle["feature_cols"])]
    x_oos = oos_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)[list(bundle["feature_cols"])]
    fee = float(meta["fee"])
    slip = float(meta["slip"])
    base_val_raw = omega._metrics(val_payload["frame"], val_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_oos_raw = omega._metrics(oos_payload["frame"], oos_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_val = {**base_val_raw, "primary_entries": base_val_raw["long_entries"] + base_val_raw["short_entries"], "fallback_entries": 0, "primary_takeovers": 0, "exit_reasons": base_val_raw.get("exit_reasons", {})}
    base_oos = {**base_oos_raw, "primary_entries": base_oos_raw["long_entries"] + base_oos_raw["short_entries"], "fallback_entries": 0, "primary_takeovers": 0, "exit_reasons": base_oos_raw.get("exit_reasons", {})}

    members: list[MemberPrediction] = []
    rows = [_metric_record("parent_only_baseline", "control", base_val, base_oos, base_val, base_oos, {"member_count": 0})]
    for variant in _variants():
        val_a, val_c, _val_diag = paper._predict_actions(x_val, bundle, variant)
        oos_a, oos_c, _oos_diag = paper._predict_actions(x_oos, bundle, variant)
        val_m = sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], exp.RISK, val_a, val_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        oos_m = sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], exp.RISK, oos_a, oos_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        val_score = _selection_score(val_m, base_val)
        oos_score = _selection_score(oos_m, base_oos)
        members.append(MemberPrediction(variant.name, variant.paper_motive, val_a, val_c, oos_a, oos_c, val_m, oos_m, val_score, oos_score))
        rows.append(_metric_record(variant.name, "single_member", val_m, oos_m, base_val, base_oos, {"member_count": 1, "members": variant.name}))

    ranked_members = sorted(members, key=lambda m: (m.val_score, float(m.val_metrics["pnl"])), reverse=True)
    member_name_to_obj = {m.name: m for m in members}
    live_fallback_entries = int(member_name_to_obj["live_contract_support_gate"].val_metrics["fallback_entries"])
    caps = [None, live_fallback_entries, max(live_fallback_entries - 4, 0), max(live_fallback_entries - 8, 0)]

    for k in range(2, min(6, len(ranked_members)) + 1):
        for subset in combinations(ranked_members[:6], k):
            subset = list(subset)
            names = "+".join(m.name for m in subset)
            raw_weights = np.asarray([max(m.val_score, 0.0) for m in subset], dtype=np.float64)
            if float(raw_weights.sum()) <= 0.0:
                raw_weights = np.ones(len(subset), dtype=np.float64)
            weights = raw_weights / float(raw_weights.sum())
            for min_votes in range(1 if k == 2 else 2, k + 1):
                for min_weight in (0.20, 0.34, 0.50):
                    for min_margin in (0.0, 0.05, 0.10):
                        for cap in caps:
                            val_a, val_c = _weighted_vote(subset, "val", weights, min_votes=min_votes, min_weight=min_weight, min_margin=min_margin, max_entries=cap)
                            oos_a, oos_c = _weighted_vote(subset, "oos", weights, min_votes=min_votes, min_weight=min_weight, min_margin=min_margin, max_entries=cap)
                            val_m = sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], exp.RISK, val_a, val_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
                            oos_m = sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], exp.RISK, oos_a, oos_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
                            rows.append(
                                _metric_record(
                                    f"weighted_vote_k{k}_v{min_votes}_w{min_weight:.2f}_m{min_margin:.2f}_cap{cap}",
                                    "weighted_vote",
                                    val_m,
                                    oos_m,
                                    base_val,
                                    base_oos,
                                    {"member_count": k, "members": names, "min_votes": min_votes, "min_weight": min_weight, "min_margin": min_margin, "max_entries": cap},
                                )
                            )
            for min_agree in range(2, k + 1):
                for cap in caps:
                    val_a, val_c = _agreement_overlay(subset, "val", min_agree=min_agree, max_entries=cap)
                    oos_a, oos_c = _agreement_overlay(subset, "oos", min_agree=min_agree, max_entries=cap)
                    val_m = sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], exp.RISK, val_a, val_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
                    oos_m = sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], exp.RISK, oos_a, oos_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
                    rows.append(
                        _metric_record(
                            f"agreement_k{k}_a{min_agree}_cap{cap}",
                            "agreement_overlay",
                            val_m,
                            oos_m,
                            base_val,
                            base_oos,
                            {"member_count": k, "members": names, "min_agree": min_agree, "max_entries": cap},
                        )
                    )

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ultimate_ensemble_ranking.csv", index=False)
    top_val = ranking.iloc[0].to_dict()
    top_oos = ranking.sort_values(["oos_pnl", "diagnostic_score_oos", "val_pnl"], ascending=False).iloc[0].to_dict()
    robust = ranking[
        (ranking["val_delta_pnl"] > 0.0)
        & (ranking["oos_delta_pnl"] > 0.0)
        & (ranking["val_mdd"] >= -12.0)
        & (ranking["oos_mdd"] >= -9.0)
        & (ranking["val_fallback_stop_loss"] <= 2)
        & (ranking["oos_fallback_stop_loss"] <= 3)
    ].copy()
    robust = robust.sort_values(["selection_score_val_only", "oos_pnl"], ascending=False).reset_index(drop=True)
    robust.to_csv(OUT_DIR / "ultimate_ensemble_robust_candidates.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "status": "research_eval_complete",
        "method": "Validation-selected ensemble grid over Omega1.2.8b paper-fix members. OOS is diagnostic only.",
        "bundle": str(BUNDLE_PATH),
        "risk": exp.RISK.__dict__,
        "baseline": {"validation": base_val, "oos": base_oos},
        "members_by_validation_score": [
            {
                "name": m.name,
                "motive": m.motive,
                "val_score": m.val_score,
                "oos_score": m.oos_score,
                "validation": m.val_metrics,
                "oos": m.oos_metrics,
            }
            for m in ranked_members
        ],
        "selected_by_validation": top_val,
        "best_oos_diagnostic": top_oos,
        "top20_validation_selected": ranking.head(20).to_dict(orient="records"),
        "top20_robust": robust.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "ultimate_ensemble_ranking.csv"),
            "robust_candidates": str(OUT_DIR / "ultimate_ensemble_robust_candidates.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected_by_validation": top_val, "best_oos_diagnostic": top_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
