"""Fail-fast runtime contracts shared by Omega4.6.1 decision paths."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class EntryOverlayStatus(str, Enum):
    PASS = "PASS"
    VETO = "VETO"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class SizingDecision:
    margin_fraction: float
    leverage: float
    notional: float

    def __post_init__(self) -> None:
        values = (self.margin_fraction, self.leverage, self.notional)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Omega4.6.1 sizing values must be finite")
        if self.margin_fraction <= 0.0:
            raise ValueError("Omega4.6.1 margin_fraction must be positive for an entry")
        if self.leverage <= 0.0 or self.notional <= 0.0:
            raise ValueError("Omega4.6.1 leverage and notional must be positive for an entry")
        expected_notional = self.margin_fraction * self.leverage
        if not math.isclose(self.notional, expected_notional, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                "Omega4.6.1 sizing contract mismatch: "
                "notional must equal margin_fraction * leverage"
            )


def finalize_sizing(
    *,
    margin_fraction: float,
    requested_notional: float,
    max_leverage: float,
    max_notional: float,
) -> SizingDecision:
    """Apply final caps once, after all notional modifiers and portfolio scaling."""
    values = {
        "margin_fraction": float(margin_fraction),
        "requested_notional": float(requested_notional),
        "max_leverage": float(max_leverage),
        "max_notional": float(max_notional),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Omega4.6.1 sizing inputs must be finite: {values}")
    if values["margin_fraction"] <= 0.0:
        raise ValueError("Omega4.6.1 margin_fraction must be positive")
    if values["requested_notional"] <= 0.0:
        raise ValueError("Omega4.6.1 requested_notional must be positive")
    if values["max_leverage"] <= 0.0 or values["max_notional"] <= 0.0:
        raise ValueError("Omega4.6.1 sizing caps must be positive")

    notional = min(
        values["requested_notional"],
        values["max_notional"],
        values["margin_fraction"] * values["max_leverage"],
    )
    leverage = notional / values["margin_fraction"]
    return SizingDecision(
        margin_fraction=values["margin_fraction"],
        leverage=leverage,
        notional=notional,
    )


def direction_overlay_status(
    *, entry_side: int, predicted_direction: int | None
) -> EntryOverlayStatus:
    if entry_side not in {-1, 1}:
        raise ValueError(f"invalid entry side: {entry_side}")
    if predicted_direction is None:
        return EntryOverlayStatus.UNAVAILABLE
    if predicted_direction not in {-1, 0, 1}:
        raise ValueError(f"invalid predicted direction: {predicted_direction}")
    if predicted_direction and predicted_direction != entry_side:
        return EntryOverlayStatus.VETO
    return EntryOverlayStatus.PASS


def strict_feature_values(
    feature_columns: list[str], feature_values: dict[str, object]
) -> list[float]:
    """Return finite feature values in artifact order; never synthesize missing data."""
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Omega4.6.1 sidecar feature contract has duplicate columns")
    missing = [column for column in feature_columns if column not in feature_values]
    if missing:
        raise ValueError(f"Omega4.6.1 sidecar feature contract missing columns: {missing}")

    ordered: list[float] = []
    for column in feature_columns:
        try:
            value = float(feature_values[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Omega4.6.1 sidecar feature {column!r} is not numeric"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"Omega4.6.1 sidecar feature {column!r} is non-finite"
            )
        ordered.append(value)
    return ordered


def validate_sidecar_lineage(
    *,
    repo_root: str | Path,
    bundle_path: str | Path,
    sidecar_path: str | Path,
    quality_threshold: float,
    allowed_selection_scopes: frozenset[str] = frozenset({"validation_only"}),
) -> dict[str, str]:
    """Validate the runtime sidecar against its exact parent prediction artifact."""
    root = Path(repo_root).resolve()

    def resolve(path_value: str | Path) -> Path:
        path = Path(path_value)
        return (path if path.is_absolute() else root / path).resolve()

    bundle = resolve(bundle_path)
    sidecar = resolve(sidecar_path)
    report_path = sidecar.parent / "report.json"
    for label, path in (("bundle", bundle), ("sidecar", sidecar), ("report", report_path)):
        if not path.is_file():
            raise ValueError(f"Omega4.6.1 missing {label} artifact: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    risk_model = report.get("risk_model", {})
    contract = report.get("contract", {})
    selection_scope = risk_model.get("selection_scope")
    if selection_scope not in allowed_selection_scopes:
        raise ValueError(
            f"Omega4.6.1 sidecar selection_scope must be one of {sorted(allowed_selection_scopes)}, "
            f"got {selection_scope!r}"
        )

    expected_tag = f"q{int(round(float(quality_threshold) * 100)):03d}"
    prediction_tag = risk_model.get("precomputed_prediction_tag")
    if prediction_tag != expected_tag:
        raise ValueError(
            f"Omega4.6.1 prediction tag mismatch: {prediction_tag!r} != {expected_tag!r}"
        )
    report_threshold = contract.get("quality_threshold")
    if report_threshold is None or not math.isclose(
        float(report_threshold), float(quality_threshold), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "Omega4.6.1 quality threshold mismatch: "
            f"{report_threshold!r} != {quality_threshold!r}"
        )

    prediction_dir_value = risk_model.get("precomputed_prediction_dir")
    if not prediction_dir_value:
        raise ValueError("Omega4.6.1 report is missing precomputed_prediction_dir")
    prediction_dir = resolve(prediction_dir_value)
    if prediction_dir != bundle.parent.resolve():
        raise ValueError(
            "Omega4.6.1 parent prediction lineage mismatch: "
            f"{prediction_dir} != {bundle.parent.resolve()}"
        )
    missing_predictions = [
        str(prediction_dir / f"{split}_predictions_{expected_tag}.csv")
        for split in ("train", "validation", "oos")
        if not (prediction_dir / f"{split}_predictions_{expected_tag}.csv").is_file()
    ]
    if missing_predictions:
        raise ValueError(
            "Omega4.6.1 missing exact prediction artifacts: "
            + ", ".join(missing_predictions)
        )
    return {
        "selection_scope": selection_scope,
        "prediction_dir": str(prediction_dir),
        "prediction_tag": prediction_tag,
        "report_path": str(report_path),
    }


def validate_fresh_forward_report_contract(
    report: dict[str, object], *, split_name: str
) -> None:
    required = {
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    mismatches = {
        key: report.get(key)
        for key, expected in required.items()
        if report.get(key) is not expected
    }
    if mismatches:
        raise ValueError(
            f"Omega4.6.1 {split_name} fresh-forward contract mismatch: {mismatches}"
        )


def validate_selection_statistics_contract(
    selection_statistics: dict[str, object]
) -> None:
    if selection_statistics.get("gate_pass") is not True:
        raise ValueError("Omega4.6.1 statistical selection evidence did not pass")
    required_statistics = (
        "deflated_sharpe_ratio",
        "minimum_deflated_sharpe_ratio",
        "probability_backtest_overfit",
        "maximum_probability_backtest_overfit",
    )
    try:
        statistics = {
            key: float(selection_statistics[key]) for key in required_statistics
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Omega4.6.1 statistical selection evidence is incomplete") from exc
    if not all(math.isfinite(value) for value in statistics.values()):
        raise ValueError("Omega4.6.1 statistical selection evidence is non-finite")
    if (
        statistics["deflated_sharpe_ratio"]
        < statistics["minimum_deflated_sharpe_ratio"]
        or statistics["probability_backtest_overfit"]
        > statistics["maximum_probability_backtest_overfit"]
    ):
        raise ValueError("Omega4.6.1 statistical selection thresholds did not pass")


def require_execution_promotion_manifest(path_value: str | Path) -> dict[str, object]:
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"Omega4.6.1 promotion manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "current_live_manifest_v1":
        raise ValueError(
            "Omega4.6.1 promotion manifest schema mismatch: "
            f"{manifest.get('schema_version')!r}"
        )
    blockers = list(manifest.get("promotion_blockers", []) or [])
    if manifest.get("promotion_eligible") is not True or blockers:
        raise ValueError(
            "Omega4.6.1 promotion manifest is not eligible: "
            f"blockers={blockers}"
        )

    artifact_integrity = manifest.get("artifact_integrity")
    if not isinstance(artifact_integrity, dict) or artifact_integrity.get("promotion_pass") is not True:
        raise ValueError("Omega4.6.1 artifact integrity evidence did not pass")

    fresh_forward = manifest.get("fresh_forward")
    if not isinstance(fresh_forward, dict):
        raise ValueError("Omega4.6.1 promotion manifest is missing fresh-forward evidence")
    for split_name in ("validation", "oos"):
        split_report = fresh_forward.get(split_name)
        if not isinstance(split_report, dict):
            raise ValueError(
                f"Omega4.6.1 promotion manifest is missing {split_name} fresh-forward evidence"
            )
        validate_fresh_forward_report_contract(split_report, split_name=split_name)

    selection_statistics = manifest.get("selection_statistics")
    if not isinstance(selection_statistics, dict):
        raise ValueError("Omega4.6.1 promotion manifest is missing statistical evidence")
    validate_selection_statistics_contract(selection_statistics)
    return manifest
