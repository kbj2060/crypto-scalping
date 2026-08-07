#!/usr/bin/env python3
"""Print a non-secret snapshot of the active Omega4.6.1 runtime contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from trading_bot_modules.omega4_6_1_live import (  # noqa: E402
    OMEGA4_6_1_MODEL_ID,
    OMEGA4_6_1_MODEL_VERSION,
)
from trading_bot_modules.runtime_config import (  # noqa: E402
    FINAL_GOVERNOR_OMEGA4_6_1_BTC_BUNDLE_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_BTC_REGIME3_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_BTC_SIDECAR_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_ENABLE,
    FINAL_GOVERNOR_OMEGA4_6_1_ETH_NOTIONAL_MULTIPLIER,
    FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_SIDECAR_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_SHADOW_ASSETS_ENABLE,
    FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE,
    FINAL_GOVERNOR_OMEGA4_6_1_SOL_BUNDLE_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_SOL_NOTIONAL_MULTIPLIER,
    FINAL_GOVERNOR_OMEGA4_6_1_SOL_REGIME3_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_SOL_SIDECAR_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_SIDECAR_PATH,
    OMEGA4_6_1_SHADOW_ASSET_CONFIG,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _file_record(path_value: str) -> dict[str, Any]:
    path = _resolve(path_value)
    record: dict[str, Any] = {
        "path": _display_path(path),
        "exists": path.is_file(),
    }
    if path.is_file():
        record.update({"size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return record


def _sidecar_record(path_value: str) -> dict[str, Any]:
    record = _file_record(path_value)
    report_path = _resolve(path_value).parent / "report.json"
    record["report"] = _file_record(str(report_path))
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        risk_model = report.get("risk_model", {})
        contract = report.get("contract", {})
        record["selection_scope"] = risk_model.get("selection_scope")
        record["quality_threshold"] = contract.get("quality_threshold")
        record["precomputed_prediction_dir"] = risk_model.get(
            "precomputed_prediction_dir"
        )
        record["precomputed_prediction_tag"] = risk_model.get(
            "precomputed_prediction_tag"
        )
    return record


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _env_flag(name: str) -> dict[str, bool]:
    value = os.getenv(name)
    return {
        "present": value is not None,
        "enabled": str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"},
    }


def build_manifest() -> dict[str, Any]:
    status = _git_output("status", "--porcelain=v1", "-z")
    entries = [entry for entry in status.split("\0") if entry]
    tracked_changes = sum(not entry.startswith("??") for entry in entries)
    untracked_entries = sum(entry.startswith("??") for entry in entries)

    artifacts = {
        "eth": {
            "h48qual_bundle": _file_record(FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH),
            "h48qual_sidecar": _sidecar_record(FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_SIDECAR_PATH),
            "zig075_bundle": _file_record(FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH),
            "zig075_sidecar": _sidecar_record(FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_SIDECAR_PATH),
        },
        "sol": {
            "bundle": _file_record(FINAL_GOVERNOR_OMEGA4_6_1_SOL_BUNDLE_PATH),
            "sidecar": _sidecar_record(FINAL_GOVERNOR_OMEGA4_6_1_SOL_SIDECAR_PATH),
            "regime3": _file_record(FINAL_GOVERNOR_OMEGA4_6_1_SOL_REGIME3_PATH),
        },
        "btc": {
            "bundle": _file_record(FINAL_GOVERNOR_OMEGA4_6_1_BTC_BUNDLE_PATH),
            "sidecar": _sidecar_record(FINAL_GOVERNOR_OMEGA4_6_1_BTC_SIDECAR_PATH),
            "regime3": _file_record(FINAL_GOVERNOR_OMEGA4_6_1_BTC_REGIME3_PATH),
        },
    }

    blockers = ["snapshot_only_until_unified_promotion_gate_passes"]
    if entries:
        blockers.append("dirty_worktree")
    for asset, values in artifacts.items():
        for name, record in values.items():
            if "sidecar" in name and record.get("selection_scope") not in {None, "validation_only"}:
                blockers.append(f"{asset}_{name}_selection_scope_{record['selection_scope']}")

    return {
        "schema_version": "current_live_manifest_v1",
        "snapshot_date": "2026-07-29",
        "promotion_eligible": False,
        "promotion_blockers": blockers,
        "model": {
            "model_id": OMEGA4_6_1_MODEL_ID,
            "model_version": OMEGA4_6_1_MODEL_VERSION,
        },
        "runtime_flags": {
            "omega4_6_1_enabled": bool(FINAL_GOVERNOR_OMEGA4_6_1_ENABLE),
            "shadow_assets_enabled": bool(FINAL_GOVERNOR_OMEGA4_6_1_SHADOW_ASSETS_ENABLE),
            "sol_btc_real_execution_enabled": bool(
                FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE
            ),
            "binance_account": _env_flag("BINANCE_ACCOUNT_ENABLED"),
            "binance_execution": _env_flag("BINANCE_EXECUTION_ENABLED"),
        },
        "sizing": {
            "contract": "notional = margin_fraction * leverage",
            "eth_notional_multiplier": float(
                FINAL_GOVERNOR_OMEGA4_6_1_ETH_NOTIONAL_MULTIPLIER
            ),
            "sol_notional_multiplier": float(
                FINAL_GOVERNOR_OMEGA4_6_1_SOL_NOTIONAL_MULTIPLIER
            ),
        },
        "asset_config": OMEGA4_6_1_SHADOW_ASSET_CONFIG,
        "artifacts": artifacts,
        "source": {
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "worktree_clean": not entries,
            "tracked_changes": tracked_changes,
            "untracked_entries": untracked_entries,
        },
    }


if __name__ == "__main__":
    json.dump(build_manifest(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
