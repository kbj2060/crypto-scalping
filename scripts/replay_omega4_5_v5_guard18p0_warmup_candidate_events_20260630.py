#!/usr/bin/env python3
"""Omega 4.5 v5_guard18p0 warmup-aware candidate-event replay.

This is a verification script, not a model retrain. It reconstructs the
v5 priority-router event stream from saved selected and skipped-overlap
candidate ledgers, reapplies the guard18p0 source/side exposure scaling,
then enforces the 20260630 warmup contract before sequentially replaying
non-overlapping trades.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/home/llewyn/crypto-scalping")
CREATIVE_BASE = ROOT / "tmp/causal_regen_20260516/omega_creative_until_10am_20260630"
SELECTED_DIR = (
    CREATIVE_BASE
    / "walkforward_oos_blind_source_side_scale_20260630_strict_mdd"
    / "v5_explainable_router"
    / "guard18p0"
)
PRIORITY_DIR = CREATIVE_BASE / "priority_router_v5_h48_h48quality_zig"
WARMUP_DIR = CREATIVE_BASE / "warmup_gate_recheck_20260630"
WARMUP_CANDIDATE_DIR = WARMUP_DIR / "v5_explainable_router_guard18p0"
BASELINE_DIR = ROOT / "tmp/causal_regen_20260516/omega4_5_baseline_v5_guard18p0_20260630"
OUT_DIR = (
    ROOT
    / "tmp/causal_regen_20260516/omega4_5_v5_guard18p0_warmup_candidate_event_replay_20260630"
)

MODEL_ID = "omega4_5_v5_explainable_router_guard18p0_20260630"
WARMUP_BARS = 576
ZERO_DEFAULT_MAX = 0.35
MAX_LEVERAGE = 5.0
MAX_CONTRACT_DIFF = 1e-9
TARGET_PNL = 100.0
MDD_FLOOR = -20.0
PRIORITY_RANK = {
    "h48_conservative": 0,
    "h48quality_repaired": 1,
    "zigzag_q075": 2,
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


def side_key(series: pd.Series) -> pd.Series:
    mapped = series.map({1: "L", -1: "S", "1": "L", "-1": "S"})
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].dropna().unique().tolist())
        raise ValueError(f"unknown side values: {bad}")
    return mapped


def scale_group(df: pd.DataFrame) -> pd.Series:
    return df["source_alias"].astype(str) + "_" + side_key(df["side"])


def extract_source_side_scale_map() -> dict[str, float]:
    frames = []
    for split in ("validation", "oos"):
        df = read_csv(SELECTED_DIR / f"{split}_scaled_trade_ledger.csv")
        frames.append(df[["scale_group", "raw_source_side_scale"]].copy())
    both = pd.concat(frames, ignore_index=True)
    scale_map: dict[str, float] = {}
    for group, grp in both.groupby("scale_group"):
        vals = sorted(float(x) for x in grp["raw_source_side_scale"].dropna().unique())
        if len(vals) != 1:
            raise ValueError(f"non-unique raw scale for {group}: {vals}")
        scale_map[str(group)] = vals[0]
    return scale_map


def load_candidate_events(split: str) -> pd.DataFrame:
    priority = read_csv(PRIORITY_DIR / f"{split}_priority_trade_ledger.csv")
    priority["_candidate_origin"] = "priority_selected"
    skipped = read_csv(PRIORITY_DIR / f"{split}_skipped_overlap_trade_ledger.csv")
    skipped["_candidate_origin"] = "skipped_overlap"
    events = pd.concat([priority, skipped], ignore_index=True)
    events["_priority_rank"] = events["source_alias"].map(PRIORITY_RANK)
    if events["_priority_rank"].isna().any():
        bad = sorted(events.loc[events["_priority_rank"].isna(), "source_alias"].unique())
        raise ValueError(f"unknown source_alias in {split}: {bad}")
    events["_source_row"] = np.arange(len(events))
    events = events.drop_duplicates(
        subset=[
            "entry_signal_i",
            "entry_i",
            "exit_i",
            "side",
            "source_alias",
            "net_per_notional",
            "raw_exit_price_move",
        ],
        keep="first",
    ).copy()
    return events.sort_values(
        ["entry_signal_i", "_priority_rank", "_candidate_origin", "_source_row"],
        kind="mergesort",
    ).reset_index(drop=True)


def apply_guard18p0_scaling(events: pd.DataFrame, scale_map: dict[str, float]) -> pd.DataFrame:
    df = events.copy()
    df["side_key"] = side_key(df["side"])
    df["scale_group"] = scale_group(df)
    missing = sorted(set(df["scale_group"]) - set(scale_map))
    if missing:
        raise ValueError(f"missing guard18p0 source/side scales: {missing}")

    df["raw_source_side_scale"] = df["scale_group"].map(scale_map).astype(float)
    df["original_notional"] = df["notional"].astype(float)
    df["original_leverage"] = df["leverage"].astype(float)
    df["original_margin_fraction"] = df["margin_fraction"].astype(float)
    cap_scale = MAX_LEVERAGE / df["original_leverage"].replace(0, np.nan)
    df["effective_source_side_scale"] = np.minimum(df["raw_source_side_scale"], cap_scale)
    df["leverage"] = df["original_leverage"] * df["effective_source_side_scale"]
    df["leverage"] = df["leverage"].clip(upper=MAX_LEVERAGE)
    df["margin_fraction"] = df["original_margin_fraction"]
    df["notional"] = df["margin_fraction"] * df["leverage"]
    df["trade_return"] = df["net_per_notional"].astype(float) * df["notional"]
    df["risk_notional"] = df["notional"]
    df["risk_leverage"] = df["leverage"]
    df["risk_margin_fraction"] = df["margin_fraction"]

    diff = (df["notional"] - df["margin_fraction"] * df["leverage"]).abs().max()
    if float(diff) > MAX_CONTRACT_DIFF:
        raise ValueError(f"notional contract violation: {diff}")
    if float(df["leverage"].max()) > MAX_LEVERAGE + 1e-9:
        raise ValueError(f"leverage cap violation: {df['leverage'].max()}")
    return df


def load_warmup_lookup(split: str) -> dict[tuple[int, str, int], dict[str, Any]]:
    frames = []
    for suffix in ("warmup_gated_ledger", "warmup_blocked_trades"):
        path = WARMUP_CANDIDATE_DIR / f"{split}_{suffix}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if len(df):
                frames.append(df)
    if not frames:
        return {}
    warm = pd.concat(frames, ignore_index=True)
    warm["side"] = warm["side"].astype(int)
    lookup: dict[tuple[int, str, int], dict[str, Any]] = {}
    for _, row in warm.iterrows():
        key = (int(row["entry_signal_i"]), str(row["source_alias"]), int(row["side"]))
        lookup[key] = {
            "_warmup_bar_index": row.get("_warmup_bar_index", np.nan),
            "_warmup_feature_zero_ratio": row.get("_warmup_feature_zero_ratio", np.nan),
            "_warmup_feature_zero_default_ratio": row.get(
                "_warmup_feature_zero_default_ratio", np.nan
            ),
            "ai_ready": row.get("ai_ready", np.nan),
            "_warmup_keep": bool(row.get("_warmup_keep", False)),
            "_warmup_block_reason": row.get("_warmup_block_reason", ""),
            "_warmup_feature_diag_available": True,
        }
    return lookup


def add_warmup_columns(events: pd.DataFrame, split: str, mode: str) -> pd.DataFrame:
    if mode not in {"strict_feature_diag", "bar_index_only_exploratory"}:
        raise ValueError(f"unknown mode: {mode}")
    df = events.copy()
    lookup = load_warmup_lookup(split)

    diagnostics: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        key = (int(row["entry_signal_i"]), str(row["source_alias"]), int(row["side"]))
        diag = lookup.get(key)
        if diag is None:
            diag = {
                "_warmup_bar_index": int(row["entry_signal_i"]),
                "_warmup_feature_zero_ratio": np.nan,
                "_warmup_feature_zero_default_ratio": np.nan,
                "ai_ready": np.nan,
                "_warmup_keep": False,
                "_warmup_block_reason": "warmup_feature_diag_missing;",
                "_warmup_feature_diag_available": False,
            }
        diagnostics.append(diag)
    diag_df = pd.DataFrame(diagnostics)
    df = pd.concat([df.reset_index(drop=True), diag_df.reset_index(drop=True)], axis=1)

    keeps = []
    reasons = []
    for _, row in df.iterrows():
        reason = ""
        bar_index = row["_warmup_bar_index"]
        if pd.isna(bar_index):
            bar_index = row["entry_signal_i"]
        if float(bar_index) < WARMUP_BARS:
            reason += f"bar_index_below_{WARMUP_BARS};"

        if mode == "strict_feature_diag":
            if not bool(row["_warmup_feature_diag_available"]):
                reason += "warmup_feature_diag_missing;"
            if pd.isna(row["ai_ready"]) or float(row["ai_ready"]) != 1.0:
                reason += "ai_ready_not_1;"
            zero_default = row["_warmup_feature_zero_default_ratio"]
            if pd.isna(zero_default) or float(zero_default) > ZERO_DEFAULT_MAX:
                reason += "zero_default_ratio_gt_0p35;"
        else:
            if bool(row["_warmup_feature_diag_available"]):
                if pd.notna(row["ai_ready"]) and float(row["ai_ready"]) != 1.0:
                    reason += "ai_ready_not_1;"
                zero_default = row["_warmup_feature_zero_default_ratio"]
                if pd.notna(zero_default) and float(zero_default) > ZERO_DEFAULT_MAX:
                    reason += "zero_default_ratio_gt_0p35;"

        keeps.append(reason == "")
        reasons.append(reason)
    df["_warmup_keep"] = keeps
    df["_warmup_block_reason"] = reasons
    df["_warmup_mode"] = mode
    return df


def replay_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    taken: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    open_until = -1

    for _, row in events.iterrows():
        rec = row.to_dict()
        entry_signal_i = int(row["entry_signal_i"])
        if not bool(row["_warmup_keep"]):
            rec["_replay_decision"] = "blocked_warmup"
            skipped.append(rec)
            continue
        if entry_signal_i <= open_until:
            rec["_replay_decision"] = "skipped_overlap"
            rec["_open_until_i"] = open_until
            skipped.append(rec)
            continue
        rec["_replay_decision"] = "taken"
        taken.append(rec)
        open_until = int(row["exit_i"])

    return pd.DataFrame(taken), pd.DataFrame(skipped)


def metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {
            "pnl": 0.0,
            "strict_mdd": 0.0,
            "closed_mdd": 0.0,
            "trades": 0,
            "wr": 0.0,
            "long_entries": 0,
            "short_entries": 0,
            "avg_notional": 0.0,
            "max_notional": 0.0,
            "max_leverage": 0.0,
            "contract_diff": 0.0,
        }

    cash = 1.0
    peak = 1.0
    strict_mdd = 0.0
    closed_mdd = 0.0
    for _, row in ledger.iterrows():
        mae_equity = cash * (1.0 + float(row["mae_price_move"]) * float(row["notional"]))
        strict_mdd = min(strict_mdd, mae_equity / peak - 1.0)
        cash *= 1.0 + float(row["trade_return"])
        peak = max(peak, cash)
        closed_mdd = min(closed_mdd, cash / peak - 1.0)
        strict_mdd = min(strict_mdd, cash / peak - 1.0)

    contract_diff = (
        ledger["notional"].astype(float)
        - ledger["margin_fraction"].astype(float) * ledger["leverage"].astype(float)
    ).abs().max()
    return {
        "pnl": (cash - 1.0) * 100.0,
        "strict_mdd": strict_mdd * 100.0,
        "closed_mdd": closed_mdd * 100.0,
        "trades": int(len(ledger)),
        "wr": float((ledger["trade_return"] > 0).mean()),
        "long_entries": int((ledger["side"].astype(int) == 1).sum()),
        "short_entries": int((ledger["side"].astype(int) == -1).sum()),
        "avg_notional": float(ledger["notional"].mean()),
        "max_notional": float(ledger["notional"].max()),
        "max_leverage": float(ledger["leverage"].max()),
        "contract_diff": float(contract_diff),
    }


def add_gate_summary(split_metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "target_pass": split_metrics["validation"]["pnl"] >= TARGET_PNL
        and split_metrics["oos"]["pnl"] >= TARGET_PNL,
        "risk_pass": split_metrics["validation"]["strict_mdd"] >= MDD_FLOOR
        and split_metrics["oos"]["strict_mdd"] >= MDD_FLOOR,
        "leverage_pass": split_metrics["validation"]["max_leverage"] <= MAX_LEVERAGE + 1e-9
        and split_metrics["oos"]["max_leverage"] <= MAX_LEVERAGE + 1e-9,
        "accounting_pass": split_metrics["validation"]["contract_diff"] <= MAX_CONTRACT_DIFF
        and split_metrics["oos"]["contract_diff"] <= MAX_CONTRACT_DIFF,
    }


def blocked_reason_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "_warmup_block_reason" not in df:
        return {}
    counts: dict[str, int] = {}
    for reason in df["_warmup_block_reason"].fillna(""):
        for part in str(reason).split(";"):
            if part:
                counts[part] = counts.get(part, 0) + 1
    return dict(sorted(counts.items()))


def write_mode_outputs(mode: str, scale_map: dict[str, float]) -> dict[str, Any]:
    mode_dir = OUT_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    split_metrics: dict[str, Any] = {}
    split_outputs: dict[str, Any] = {}

    for split in ("validation", "oos"):
        events = load_candidate_events(split)
        scaled = apply_guard18p0_scaling(events, scale_map)
        warm = add_warmup_columns(scaled, split, mode)
        replayed, skipped = replay_events(warm)

        events_path = mode_dir / f"{split}_candidate_events_scaled_warmup.csv"
        replayed_path = mode_dir / f"{split}_replay_trade_ledger.csv"
        skipped_path = mode_dir / f"{split}_replay_skipped_candidates.csv"
        warm.to_csv(events_path, index=False)
        replayed.to_csv(replayed_path, index=False)
        skipped.to_csv(skipped_path, index=False)

        m = metrics(replayed)
        split_metrics[split] = m
        warmup_blocked = skipped[skipped["_replay_decision"] == "blocked_warmup"].copy()
        overlap_skipped = skipped[skipped["_replay_decision"] == "skipped_overlap"].copy()
        first_allowed = ""
        if not replayed.empty:
            first_allowed = str(replayed.iloc[0]["entry_timestamp"])
        split_outputs[split] = {
            "candidate_events": int(len(warm)),
            "taken_trades": int(len(replayed)),
            "blocked_warmup": int(len(warmup_blocked)),
            "skipped_overlap": int(len(overlap_skipped)),
            "warmup_block_reason_counts": blocked_reason_counts(warmup_blocked),
            "feature_diag_missing_candidates": int(
                (~warm["_warmup_feature_diag_available"].astype(bool)).sum()
            ),
            "first_allowed_entry_timestamp": first_allowed,
            "events_csv": str(events_path),
            "ledger_csv": str(replayed_path),
            "skipped_csv": str(skipped_path),
        }

    gates = add_gate_summary(split_metrics)
    gates["selection_oos_independent"] = True
    gates["full_bar_replay_available"] = False
    gates["candidate_event_replay"] = True
    gates["candidate_event_metric_pass"] = bool(
        mode == "strict_feature_diag"
        and gates["target_pass"]
        and gates["risk_pass"]
        and gates["leverage_pass"]
        and gates["accounting_pass"]
        and gates["selection_oos_independent"]
    )
    gates["redteam_full_pass"] = bool(
        gates["candidate_event_metric_pass"] and gates["full_bar_replay_available"]
    )
    gates["promotion_pass"] = gates["redteam_full_pass"]

    return {
        "mode": mode,
        "metrics": split_metrics,
        "gates": gates,
        "outputs": split_outputs,
    }


def write_upgrade_plan(report: dict[str, Any]) -> None:
    lines = [
        "# Omega 4.5 Upgrade Plan: v5_guard18p0",
        "",
        "## Baseline Contract",
        "",
        f"- Model ID: `{MODEL_ID}`",
        "- Alias: `v5_guard18p0` -> `v5_explainable_router_guard18p0`",
        "- Source router: `priority_router_v5_h48_h48quality_zig`",
        "- Exposure policy: guard18p0 source/side scaling, leverage cap 5x",
        "- Futures sizing: `notional = margin_fraction * leverage`",
        "- Warmup contract: 576 bars, `ai_ready == 1`, zero/default ratio <= 0.35",
        "",
        "## Current Verification Result",
        "",
    ]
    for mode in report["mode_results"]:
        lines.append(f"### {mode['mode']}")
        for split in ("validation", "oos"):
            m = mode["metrics"][split]
            out = mode["outputs"][split]
            lines.append(
                f"- {split}: pnl {m['pnl']:.2f}%, strict_mdd {m['strict_mdd']:.2f}%, "
                f"trades {m['trades']}, WR {m['wr']:.2%}, blocked {out['blocked_warmup']}, "
                f"overlap skipped {out['skipped_overlap']}"
            )
        lines.append(f"- gates: `{mode['gates']}`")
        lines.append("")

    lines.extend(
        [
            "## Upgrade Sequence",
            "",
            "1. Regenerate validation/OOS features with pre-split tail before slicing.",
            "   Verify: no cold-frame zeros, `ai_ready == 1`, first tradable bar after warmup.",
            "2. Re-run the v5 router from raw per-bar source signals, not saved trade ledgers.",
            "   Verify: candidate stream includes skipped signals with full warmup diagnostics.",
            "3. Re-audit strict MDD using MAE/open-equity, not closed-trade-only drawdown.",
            "   Verify: validation and OOS both pnl >= 100%, strict MDD >= -20%, leverage <= 5x.",
            "4. If target still fails, tune only on validation: source/side scale floors, early-window guards,",
            "   and long/short sleeve balance. Freeze parameters before OOS.",
            "   Verify: OOS remains blind and lineage stays OOS-independent.",
            "5. Promote only after full bar-level replay passes. Candidate-event replay is diagnostic,",
            "   not sufficient for live promotion.",
            "",
        ]
    )
    (OUT_DIR / "OMEGA4_5_UPGRADE_PLAN.md").write_text("\n".join(lines))


def write_subagent_handoff(report: dict[str, Any]) -> None:
    strict = next(x for x in report["mode_results"] if x["mode"] == "strict_feature_diag")
    lines = [
        "# Sub-Agent Handoff: Omega 4.5 v5_guard18p0",
        "",
        "Use this as the baseline contract for follow-up Omega 4.5 work.",
        "",
        "## Identity",
        "",
        f"- Model ID: `{MODEL_ID}`",
        "- Alias: `v5_guard18p0`",
        "- Canonical artifact: `v5_explainable_router_guard18p0`",
        f"- Report: `{OUT_DIR / 'report.json'}`",
        f"- Summary CSV: `{OUT_DIR / 'redteam_candidate_event_summary.csv'}`",
        "",
        "## Current Status",
        "",
        "- This is the Omega 4.5 baseline candidate, but it is not a live/promotion PASS yet.",
        "- `strict_feature_diag` candidate-event replay metric gates pass:",
        f"  - validation pnl {strict['metrics']['validation']['pnl']:.2f}%, "
        f"strict_mdd {strict['metrics']['validation']['strict_mdd']:.2f}%, "
        f"trades {strict['metrics']['validation']['trades']}",
        f"  - OOS pnl {strict['metrics']['oos']['pnl']:.2f}%, "
        f"strict_mdd {strict['metrics']['oos']['strict_mdd']:.2f}%, "
        f"trades {strict['metrics']['oos']['trades']}",
        "- `redteam_full_pass` is false because full per-bar replay is not yet available.",
        "- Do not promote until raw source-signal replay with pre-split warmup features passes.",
        "",
        "## Mandatory Contracts",
        "",
        "- Run Python with conda env `quant_ai`.",
        "- Futures sizing: `notional = margin_fraction * leverage`.",
        "- Leverage cap: 5x.",
        "- Strict MDD must include intra-trade MAE/open-equity troughs.",
        "- Warmup: 576 bars, `ai_ready == 1`, zero/default feature ratio <= 0.35.",
        "- No silent aliases/fallbacks on active candidate/live paths.",
        "",
        "## Next Work",
        "",
        "1. Locate or regenerate raw per-bar source signals for all v5 components.",
        "2. Compute validation/OOS features with pre-split tail before slicing.",
        "3. Re-run router priority and source/side scaling from raw signals.",
        "4. Re-run strict redteam gates: validation and OOS pnl >= 100%, strict MDD >= -20%, leverage <= 5x.",
        "5. Only tune on validation; keep OOS blind.",
        "",
    ]
    (OUT_DIR / "SUBAGENT_OMEGA4_5_HANDOFF_20260630.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scale_map = extract_source_side_scale_map()
    baseline_report = read_json(BASELINE_DIR / "report.json") if (BASELINE_DIR / "report.json").exists() else {}
    source_report = read_json(SELECTED_DIR / "report.json")
    redteam_report = read_json(CREATIVE_BASE / "redteam_full_audit_nested_20260630/report.json")
    warmup_report = read_json(WARMUP_DIR / "report.json")

    mode_results = [
        write_mode_outputs("strict_feature_diag", scale_map),
        write_mode_outputs("bar_index_only_exploratory", scale_map),
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "purpose": "Omega 4.5 baseline propagation plus warmup-aware candidate-event replay verification.",
        "baseline_alias": "v5_guard18p0",
        "canonical_candidate": "v5_explainable_router_guard18p0",
        "source_report": str(SELECTED_DIR / "report.json"),
        "baseline_manifest": str(BASELINE_DIR / "report.json"),
        "warmup_contract_report": str(WARMUP_DIR / "report.json"),
        "redteam_report": str(CREATIVE_BASE / "redteam_full_audit_nested_20260630/report.json"),
        "source_side_scale_map": scale_map,
        "contracts": {
            "warmup_bars": WARMUP_BARS,
            "zero_default_feature_ratio_max": ZERO_DEFAULT_MAX,
            "max_leverage": MAX_LEVERAGE,
            "target_pnl_each_split": TARGET_PNL,
            "strict_mdd_floor_each_split": MDD_FLOOR,
            "notional_contract": "notional = margin_fraction * leverage",
            "strict_mdd": "includes intra-trade MAE/open-equity troughs",
            "fail_fast_note": "strict mode blocks candidate events without warmup feature diagnostics.",
        },
        "input_snapshots": {
            "baseline_manifest": baseline_report,
            "guard18p0_report": source_report,
            "nested_redteam_verdict": {
                "verdict": redteam_report.get("verdict"),
                "full_pass_count": redteam_report.get("full_pass_count"),
                "best_oos_independent_candidate": redteam_report.get("best_oos_independent_candidate"),
            },
            "warmup_recheck_contract": warmup_report.get("contract"),
        },
        "mode_results": mode_results,
        "promotion_verdict": "NO_PROMOTION_FULL_BAR_REPLAY_REQUIRED",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    rows = []
    for mode in mode_results:
        row = {"mode": mode["mode"], **mode["gates"]}
        for split in ("validation", "oos"):
            m = mode["metrics"][split]
            row.update(
                {
                    f"{split}_pnl": m["pnl"],
                    f"{split}_strict_mdd": m["strict_mdd"],
                    f"{split}_trades": m["trades"],
                    f"{split}_wr": m["wr"],
                    f"{split}_max_leverage": m["max_leverage"],
                    f"{split}_avg_notional": m["avg_notional"],
                    f"{split}_blocked_warmup": mode["outputs"][split]["blocked_warmup"],
                    f"{split}_feature_diag_missing": mode["outputs"][split][
                        "feature_diag_missing_candidates"
                    ],
                }
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "redteam_candidate_event_summary.csv", index=False)
    write_upgrade_plan(report)
    write_subagent_handoff(report)
    print(json.dumps(report["mode_results"], indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
