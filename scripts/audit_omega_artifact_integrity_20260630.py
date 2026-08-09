#!/usr/bin/env python3
"""Fail-fast artifact integrity audit for Omega promotion candidates.

Promotion candidates must not depend on regenerating parent predictions from
mutable code/data contracts. Each risk-sidecar component needs exact
precomputed parent prediction CSVs for the threshold it actually uses.

[2026-07-30 -- dataset_lineage gate, P0-2 of
docs/pipeline_integrity_and_research_redesign_20260730.md]
The Omega4.6.1 07-06 frozen baseline could not be reproduced because the feature CSV it was
trained/replayed on changed in place with no record kept of which exact bytes it used (root cause:
upstream Binance metrics zips were retroactively revised -- see project memory
project-omega461-baseline-drift-bisection-20260730). scripts/dataset_snapshot.py now lets any
dataset-producing script register a content hash for its output. This audit additionally requires
every component report.json (risk sidecar and its parent bundle) to declare
`dataset_lineage: {"features_path": ..., "features_sha256": ...}`, and fails closed if that
lineage is missing, unregistered in data/splits/DATASET_MANIFEST.json, or does not match either
the manifest's recorded hash or the current on-disk file. This deliberately fails every report.json
written before this gate existed -- that is the intended behavior, not a bug: a report that cannot
prove which exact data it used is not a valid promotion candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


@dataclass
class Check:
    name: str
    status: str
    detail: str


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def qtag(value: float) -> str:
    return f"q{int(round(float(value) * 100.0)):03d}"


def check(name: str, ok: bool, detail: str) -> Check:
    return Check(name=name, status="pass" if ok else "fail", detail=detail)


def risk_selection_contract_checks(risk_report: dict[str, Any]) -> list[Check]:
    risk_model = risk_report.get("risk_model", {})
    selected = risk_report.get("selected", {})
    constraints = selected.get("constraints", {})
    full_validation = selected.get("selected_full_replay", {}).get("validation", {})
    checks = [
        check(
            "risk_selection_scope_validation_only",
            risk_model.get("selection_scope") == "validation_only",
            f"selection_scope={risk_model.get('selection_scope')!r}",
        ),
        check(
            "risk_constraint_pass_declared",
            selected.get("constraint_pass") is True,
            f"constraint_pass={selected.get('constraint_pass')!r}",
        ),
        check(
            "risk_fallback_not_used",
            selected.get("fallback_used") is False,
            f"fallback_used={selected.get('fallback_used')!r}",
        ),
        check(
            "risk_full_replay_selection_applied",
            selected.get("full_replay_selection_applied") is True,
            f"full_replay_selection_applied={selected.get('full_replay_selection_applied')!r}",
        ),
    ]
    trade_floor = constraints.get("validation_trade_floor")
    mdd_floor = constraints.get("validation_mdd_floor")
    trades = full_validation.get("trades")
    mdd = full_validation.get("mdd")
    machine_contract_present = all(
        value is not None for value in (trade_floor, mdd_floor, trades, mdd)
    )
    checks.append(
        check(
            "risk_machine_readable_constraints_present",
            machine_contract_present,
            f"trade_floor={trade_floor!r} mdd_floor={mdd_floor!r} trades={trades!r} mdd={mdd!r}",
        )
    )
    metrics_pass = False
    if machine_contract_present:
        metrics_pass = int(trades) >= int(trade_floor) and float(mdd) >= float(mdd_floor)
    checks.append(
        check(
            "risk_full_replay_metrics_meet_constraints",
            metrics_pass,
            f"trades={trades!r}>={trade_floor!r} mdd={mdd!r}>={mdd_floor!r}",
        )
    )
    return checks


DATASET_MANIFEST_PATH = ROOT / "data/splits/DATASET_MANIFEST.json"


def _load_dataset_manifest() -> dict[str, Any]:
    if not DATASET_MANIFEST_PATH.exists():
        return {"files": {}}
    return json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))


def dataset_lineage_checks(report: dict[str, Any], *, prefix: str) -> list[Check]:
    """Verify a report.json declares which exact dataset it used, and that this still matches
    both the registered manifest entry and the current on-disk file. See module docstring."""
    checks: list[Check] = []
    lineage = report.get("dataset_lineage")
    if not isinstance(lineage, dict):
        checks.append(
            check(f"{prefix}_dataset_lineage_present", False, "dataset_lineage missing from report.json")
        )
        return checks
    checks.append(check(f"{prefix}_dataset_lineage_present", True, "dataset_lineage present"))

    features_path = lineage.get("features_path")
    features_sha256 = lineage.get("features_sha256")
    if not features_path or not features_sha256:
        checks.append(
            check(
                f"{prefix}_dataset_lineage_fields_complete",
                False,
                f"features_path={features_path!r} features_sha256={features_sha256!r}",
            )
        )
        return checks
    checks.append(check(f"{prefix}_dataset_lineage_fields_complete", True, f"features_path={features_path}"))

    abs_path = resolve_path(features_path)
    rel_key = abs_path.resolve().relative_to(ROOT).as_posix() if abs_path is not None else features_path
    manifest = _load_dataset_manifest()
    entry = manifest.get("files", {}).get(rel_key)
    if entry is None:
        checks.append(
            check(
                f"{prefix}_dataset_lineage_registered_in_manifest",
                False,
                f"{rel_key} not found in data/splits/DATASET_MANIFEST.json -- run scripts/dataset_snapshot.py register",
            )
        )
        return checks
    checks.append(
        check(f"{prefix}_dataset_lineage_registered_in_manifest", True, f"{rel_key} registered {entry.get('generated_at')}")
    )
    checks.append(
        check(
            f"{prefix}_dataset_lineage_report_matches_manifest",
            entry.get("sha256") == features_sha256,
            f"report_sha256={features_sha256} manifest_sha256={entry.get('sha256')}",
        )
    )

    if abs_path is not None and abs_path.exists():
        current_sha256 = sha256_file(abs_path)
        checks.append(
            check(
                f"{prefix}_dataset_lineage_matches_current_file",
                current_sha256 == entry.get("sha256"),
                f"current_sha256={current_sha256} manifest_sha256={entry.get('sha256')}",
            )
        )
    else:
        checks.append(check(f"{prefix}_dataset_lineage_matches_current_file", False, f"{abs_path} does not exist"))

    return checks


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    st = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def discover_component_dirs(report_path: Path) -> list[tuple[str, Path]]:
    report = read_json(report_path)
    components = report.get("components")
    if isinstance(components, dict):
        out: list[tuple[str, Path]] = []
        for alias, cfg in components.items():
            if not isinstance(cfg, dict) or "out_dir" not in cfg:
                raise RuntimeError(f"{report_path}: component {alias} missing out_dir")
            out.append((str(alias), resolve_path(str(cfg["out_dir"]))))
        return out

    source_report = report.get("source_report")
    if isinstance(source_report, str) and source_report:
        source_path = resolve_path(source_report)
        if source_path is not None and source_path.exists():
            try:
                return discover_component_dirs(source_path)
            except RuntimeError:
                pass

    source_priority_report = report.get("source_priority_report")
    if isinstance(source_priority_report, str) and source_priority_report:
        priority_path = resolve_path(source_priority_report)
        if priority_path is not None and priority_path.exists():
            return discover_component_dirs(priority_path)

    full_bar = report.get("full_bar_warmup_replay")
    if isinstance(full_bar, dict) and isinstance(full_bar.get("report"), str):
        full_bar_path = resolve_path(full_bar["report"])
        if full_bar_path is not None and full_bar_path.exists():
            full_bar_report = read_json(full_bar_path)
            priority_path = resolve_path(full_bar_report.get("source_priority_report"))
            if priority_path is not None and priority_path.exists():
                return discover_component_dirs(priority_path)

    if "baseline_bundle" in report and "risk_model" in report:
        return [(report_path.parent.name, report_path.parent)]

    raise RuntimeError(f"{report_path}: cannot discover Omega component run dirs")


def prepared_frame_timestamps(risk_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, pd.Series]:
    import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega
    import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4

    risk_model = risk_report["risk_model"]
    label_contract = parent_report.get("label_contract") or {}
    direction_label_dir = resolve_path(risk_model.get("direction_label_dir") or label_contract.get("direction_label_dir"))
    if direction_label_dir is None:
        raise RuntimeError("missing direction_label_dir")

    quality_mode = str(risk_model.get("quality_mode") or label_contract.get("quality_mode") or "same_as_direction")
    quality_label_dir = label_contract.get("quality_label_dir") if quality_mode in {"quality_label_action", "quality_label_hard_rule"} else None

    omega.TRAIN_CSV = resolve_path(risk_model["train_csv"])
    omega.EVAL_CSV = resolve_path(risk_model["eval_csv"])
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=direction_label_dir,
        quality_mode=quality_mode,
        quality_label_dir=resolve_path(quality_label_dir),
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    return {
        "train": pd.to_datetime(frames["train_raw"]["timestamp"], errors="raise").reset_index(drop=True),
        "validation": pd.to_datetime(frames["val_raw"]["timestamp"], errors="raise").reset_index(drop=True),
        "oos": pd.to_datetime(frames["oos_raw"]["timestamp"], errors="raise").reset_index(drop=True),
    }


def prediction_timestamp_check(path: Path, expected: pd.Series) -> tuple[bool, str]:
    pred = pd.read_csv(path, usecols=["timestamp"])
    got = pd.to_datetime(pred["timestamp"], errors="raise").reset_index(drop=True)
    if len(got) != len(expected):
        return False, f"row_count_mismatch got={len(got)} expected={len(expected)}"
    if not got.equals(expected):
        first_bad = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), None)
        return False, f"timestamp_mismatch first_bad={first_bad} got={got.iloc[first_bad] if first_bad is not None else None} expected={expected.iloc[first_bad] if first_bad is not None else None}"
    return True, f"rows={len(got)}"


def audit_component(alias: str, component_dir: Path, *, require_train: bool) -> dict[str, Any]:
    report_path = component_dir / "report.json"
    risk_report = read_json(report_path)
    parent_bundle = resolve_path(risk_report.get("baseline_bundle"))
    parent_report_path = parent_bundle.parent / "report.json" if parent_bundle is not None else None
    parent_report = read_json(parent_report_path) if parent_report_path is not None and parent_report_path.exists() else {}
    risk_model = risk_report.get("risk_model", {})
    contract = risk_report.get("contract", {})
    threshold = float(contract.get("quality_threshold"))
    tag = qtag(threshold)
    pre_dir = resolve_path(risk_model.get("precomputed_prediction_dir"))
    pre_tag = risk_model.get("precomputed_prediction_tag")
    prediction_dir = pre_dir if pre_dir is not None else (parent_bundle.parent if parent_bundle is not None else component_dir)

    checks: list[Check] = []
    checks.append(check("component_report_exists", report_path.exists(), str(report_path)))
    checks.extend(risk_selection_contract_checks(risk_report))
    checks.append(check("parent_bundle_exists", parent_bundle is not None and parent_bundle.exists(), str(parent_bundle)))
    checks.append(check("parent_report_exists", parent_report_path is not None and parent_report_path.exists(), str(parent_report_path)))
    checks.extend(dataset_lineage_checks(risk_report, prefix="risk"))
    checks.extend(dataset_lineage_checks(parent_report, prefix="parent"))
    checks.append(
        check(
            "risk_sidecar_uses_precomputed_parent_predictions",
            pre_dir is not None and str(pre_tag) == tag,
            f"precomputed_prediction_dir={pre_dir} precomputed_prediction_tag={pre_tag} required_tag={tag}",
        )
    )

    parent_label_contract = parent_report.get("label_contract") or {}
    risk_quality_mode = risk_model.get("quality_mode")
    parent_quality_mode = parent_label_contract.get("quality_mode")
    checks.append(
        check(
            "risk_report_parent_quality_mode_match",
            parent_quality_mode is None or risk_quality_mode == parent_quality_mode or pre_dir is not None,
            f"risk_quality_mode={risk_quality_mode} parent_quality_mode={parent_quality_mode} precomputed_dir={pre_dir}",
        )
    )

    required_splits = ["validation", "oos"]
    if require_train:
        required_splits.insert(0, "train")

    prediction_files = {split: prediction_dir / f"{split}_predictions_{tag}.csv" for split in required_splits}
    missing = [split for split, path in prediction_files.items() if not path.exists()]
    checks.append(
        check(
            "exact_threshold_parent_prediction_files_present",
            not missing,
            json.dumps({split: str(path) for split, path in prediction_files.items()}, ensure_ascii=False),
        )
    )

    timestamp_checks: dict[str, str] = {}
    if not missing:
        try:
            expected_ts = prepared_frame_timestamps(risk_report, parent_report)
            for split, path in prediction_files.items():
                ok, detail = prediction_timestamp_check(path, expected_ts[split])
                timestamp_checks[split] = detail
                checks.append(check(f"{split}_prediction_timestamps_match_runtime_frame", ok, detail))
        except Exception as exc:  # noqa: BLE001 - audit must report the contract failure.
            checks.append(check("runtime_frame_timestamp_check_available", False, repr(exc)))

    artifacts = {
        "component_report": file_fingerprint(report_path),
        "parent_bundle": file_fingerprint(parent_bundle) if parent_bundle is not None else {"exists": False},
        "parent_report": file_fingerprint(parent_report_path) if parent_report_path is not None else {"exists": False},
        "risk_sidecar": file_fingerprint(component_dir / "risk_sidecar.pkl"),
        "prediction_files": {split: file_fingerprint(path) for split, path in prediction_files.items()},
    }
    failures = [c for c in checks if c.status == "fail"]
    return {
        "alias": alias,
        "component_dir": str(component_dir),
        "required_quality_threshold": threshold,
        "required_prediction_tag": tag,
        "checks": [c.__dict__ for c in checks],
        "pass": not failures,
        "failures": [c.__dict__ for c in failures],
        "timestamp_checks": timestamp_checks,
        "artifacts": artifacts,
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        f"# Omega Artifact Integrity Audit - {audit['created_at']}",
        "",
        f"- Report: `{audit['input_report']}`",
        f"- Promotion pass: `{audit['promotion_pass']}`",
        f"- Components: `{len(audit['components'])}`",
        "",
        "## Component Results",
        "",
    ]
    for comp in audit["components"]:
        lines.append(
            f"- `{comp['alias']}`: pass={comp['pass']} threshold={comp['required_quality_threshold']} tag=`{comp['required_prediction_tag']}`"
        )
        for failure in comp["failures"]:
            lines.append(f"  - FAIL `{failure['name']}`: {failure['detail']}")
    lines.extend(
        [
            "",
            "## Promotion Rule",
            "",
            "A candidate is promotable only when every component uses exact-threshold precomputed parent predictions and those prediction timestamps match the runtime frame.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--no-require-train", action="store_true")
    args = ap.parse_args()

    report_path = resolve_path(args.report)
    if report_path is None:
        raise RuntimeError("--report resolved to None")
    components = discover_component_dirs(report_path)
    out_dir = resolve_path(args.out_dir) if args.out_dir is not None else report_path.parent
    if out_dir is None:
        raise RuntimeError("--out-dir resolved to None")
    out_dir.mkdir(parents=True, exist_ok=True)

    component_results = [
        audit_component(alias, component_dir, require_train=not bool(args.no_require_train))
        for alias, component_dir in components
    ]
    promotion_pass = all(bool(item["pass"]) for item in component_results)
    audit = {
        "audit_id": "omega_artifact_integrity_audit_20260630",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_report": str(report_path),
        "promotion_pass": bool(promotion_pass),
        "component_count": len(component_results),
        "components": component_results,
        "policy": {
            "precomputed_parent_predictions_required": True,
            "exact_threshold_tag_required": True,
            "timestamp_match_required": True,
            "validation_only_risk_selection_required": True,
            "risk_constraint_pass_required": True,
            "risk_fallback_allowed": False,
            "full_validation_replay_constraint_required": True,
            "historical_trade_ledger_fallback_allowed_for_promotion": False,
            "dataset_lineage_required": True,
            "dataset_lineage_manifest": str(DATASET_MANIFEST_PATH.relative_to(ROOT)),
        },
    }
    json_path = out_dir / "omega_artifact_integrity_audit_20260630.json"
    md_path = out_dir / "omega_artifact_integrity_audit_20260630.md"
    write_json(json_path, audit)
    write_markdown(md_path, audit)
    print(json.dumps({"promotion_pass": promotion_pass, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0 if promotion_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
