#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_omega462_hf_policy_bar_forward_val_oos_20260702 import json_default, write_json  # noqa: E402
from scripts.train_eval_omega462_live_native_entry_gate_20260702 import (  # noqa: E402
    DEFAULT_FEATURES_2025,
    DEFAULT_OOS_FEATURES,
    DEFAULT_TRAIN_FEATURES,
    load_policy,
)
from scripts.train_eval_omega462_live_native_sequence_entry_gate_20260703 import (  # noqa: E402
    DEFAULT_OUT as DEFAULT_SEQUENCE_OUT,
    MODEL_ID as SEQUENCE_MODEL_ID,
    load_artifact,
    parse_csv_list,
    simulate_with_sequence_gate,
)


DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/omega462_live_native_tcn_sequence_entry_gate_variants_20260703"


def parse_side_filter(text: str) -> set[int] | None:
    value = str(text).strip().lower()
    if value in {"", "both", "all", "none"}:
        return None
    if value == "short":
        return {-1}
    if value == "long":
        return {1}
    raise RuntimeError(f"unknown side filter: {text}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_csv_list(args.thresholds, cast=float)
    allowed_sides = parse_side_filter(args.side_filter)
    policy = load_policy(args)

    results: dict[str, Any] = {}
    split_artifacts: dict[str, Any] = {}
    integrity: dict[str, int] = {}
    artifact_path = Path(args.artifact)

    for threshold in thresholds:
        suffix = f"{args.name_prefix}_thr{threshold:.6f}_{args.side_filter}".replace("-", "m").replace(".", "p")
        artifact = load_artifact(artifact_path, threshold=float(threshold), name=suffix)
        validation_metrics, validation_artifacts = simulate_with_sequence_gate(
            split="validation",
            feature_path=Path(args.features_2025),
            start=args.validation_start,
            end=args.validation_end,
            parent_variant=args.parent_runtime_variant,
            policy=policy,
            out_dir=out_dir,
            artifact=artifact,
            allowed_sides=allowed_sides,
        )
        oos_metrics, oos_artifacts = simulate_with_sequence_gate(
            split="oos",
            feature_path=Path(args.oos_features),
            start=args.oos_start,
            end=args.oos_end,
            parent_variant=args.parent_runtime_variant,
            policy=policy,
            out_dir=out_dir,
            artifact=artifact,
            allowed_sides=allowed_sides,
        )
        results[artifact.name] = {
            "artifact": str(artifact_path),
            "threshold": float(threshold),
            "side_filter": str(args.side_filter),
            "validation": validation_metrics,
            "oos": oos_metrics,
        }
        split_artifacts[artifact.name] = {
            "validation": validation_artifacts,
            "oos": oos_artifacts,
        }
        integrity[f"{artifact.name}_validation_ledger_replay_trace_count"] = int(validation_metrics["ledger_replay_trace_count"])
        integrity[f"{artifact.name}_validation_non_live_native_trace_count"] = int(validation_metrics["non_live_native_trace_count"])
        integrity[f"{artifact.name}_validation_non_minus_one_policy_row_count"] = int(validation_metrics["non_minus_one_policy_row_count"])
        integrity[f"{artifact.name}_oos_ledger_replay_trace_count"] = int(oos_metrics["ledger_replay_trace_count"])
        integrity[f"{artifact.name}_oos_non_live_native_trace_count"] = int(oos_metrics["non_live_native_trace_count"])
        integrity[f"{artifact.name}_oos_non_minus_one_policy_row_count"] = int(oos_metrics["non_minus_one_policy_row_count"])

    ranked = sorted(
        [
            {
                "name": name,
                "threshold": float(payload["threshold"]),
                "side_filter": payload["side_filter"],
                "validation_compound_pnl_pct": float(payload["validation"]["compound_pnl_pct"]),
                "validation_compound_mdd_pct": float(payload["validation"]["compound_mdd_pct"]),
                "validation_trades": int(payload["validation"]["trades"]),
                "oos_compound_pnl_pct": float(payload["oos"]["compound_pnl_pct"]),
                "oos_compound_mdd_pct": float(payload["oos"]["compound_mdd_pct"]),
                "oos_trades": int(payload["oos"]["trades"]),
            }
            for name, payload in results.items()
        ],
        key=lambda row: (row["validation_compound_pnl_pct"], row["oos_compound_pnl_pct"]),
        reverse=True,
    )
    report = {
        "schema_version": "omega462.sequence_gate_variants.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sequence_model_id": SEQUENCE_MODEL_ID,
        "artifact": str(artifact_path),
        "parent_runtime_variant": str(args.parent_runtime_variant),
        "policy": policy,
        "side_filter": str(args.side_filter),
        "thresholds": thresholds,
        "training_contract": {
            "validation_rows_used_for_training": False,
            "oos_rows_used_for_training": False,
            "artifact_is_pretrained_on_train_split_only": True,
            "variant_changes_only_runtime_gate_threshold_or_side_filter": True,
        },
        "fresh_forward_definition": "fixed split, causal 5m bar-by-bar replay; sequence gate sees only current/past live-native feature rows buffered inside the split",
        "results": results,
        "ranked": ranked,
        "split_artifacts": split_artifacts,
        "integrity": integrity,
        "artifacts": {
            "out_dir": str(out_dir),
            "report": str(out_dir / "report.json"),
        },
    }
    write_json(out_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=str(DEFAULT_SEQUENCE_OUT / "tcn_seq_gate_L24_flat.pt"))
    parser.add_argument("--name-prefix", default="tcn_seq_gate_L24_flat_variant")
    parser.add_argument("--train-features", default=str(DEFAULT_TRAIN_FEATURES))
    parser.add_argument("--features-2025", default=str(DEFAULT_FEATURES_2025))
    parser.add_argument("--oos-features", default=str(DEFAULT_OOS_FEATURES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--parent-runtime-variant", choices=["source_v5", "cap220_no_v5"], default="source_v5")
    parser.add_argument("--validation-start", default="2025-09-01 00:00:00")
    parser.add_argument("--validation-end", default="2026-01-01 00:00:00")
    parser.add_argument("--oos-start", default="2026-01-01 00:00:00")
    parser.add_argument("--oos-end", default="2026-04-01 00:00:00")
    parser.add_argument("--tp", type=float, default=0.026)
    parser.add_argument("--sl", type=float, default=0.014)
    parser.add_argument("--cap", type=float, default=4.106)
    parser.add_argument("--max-hold-hours", type=float, default=90.0)
    parser.add_argument("--thresholds", default="-0.013343234360218049,-0.008412085473537445,-0.004099276661872864")
    parser.add_argument("--side-filter", choices=["both", "short", "long"], default="short")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"ranked": report["ranked"], "integrity": report["integrity"]}, ensure_ascii=False, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
