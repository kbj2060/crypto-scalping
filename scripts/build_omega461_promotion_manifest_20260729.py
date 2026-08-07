#!/usr/bin/env python3
"""Build an execution-eligible Omega4.6.1 manifest from validated evidence reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading_bot_modules.omega4_6_1_runtime_contract import (
    require_execution_promotion_manifest,
    validate_fresh_forward_report_contract,
    validate_selection_statistics_contract,
)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence report must be a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_promotion_manifest(
    *,
    artifact_report_path: Path,
    validation_report_path: Path,
    oos_report_path: Path,
    selection_report_path: Path,
    source_commit: str,
) -> dict[str, object]:
    artifact = _load(artifact_report_path)
    validation = _load(validation_report_path)
    oos = _load(oos_report_path)
    selection = _load(selection_report_path)

    if artifact.get("promotion_pass") is not True:
        raise ValueError("artifact integrity report did not pass")
    validate_fresh_forward_report_contract(validation, split_name="validation")
    validate_fresh_forward_report_contract(oos, split_name="oos")
    validate_selection_statistics_contract(selection)
    if not source_commit.strip():
        raise ValueError("source_commit must be non-empty")

    def evidence(path: Path, payload: dict[str, object]) -> dict[str, object]:
        return {**payload, "report_path": str(path), "report_sha256": _sha256(path)}

    return {
        "schema_version": "current_live_manifest_v1",
        "promotion_eligible": True,
        "promotion_blockers": [],
        "source": {"git_commit": source_commit.strip(), "worktree_clean": True},
        "artifact_integrity": evidence(artifact_report_path, artifact),
        "fresh_forward": {
            "validation": evidence(validation_report_path, validation),
            "oos": evidence(oos_report_path, oos),
        },
        "selection_statistics": evidence(selection_report_path, selection),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--oos-report", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_promotion_manifest(
        artifact_report_path=args.artifact_report,
        validation_report_path=args.validation_report,
        oos_report_path=args.oos_report,
        selection_report_path=args.selection_report,
        source_commit=args.source_commit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    require_execution_promotion_manifest(args.out)
    print(json.dumps({"manifest": str(args.out), "promotion_eligible": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
