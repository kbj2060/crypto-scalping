import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/ensemble/reports/btc_sol_micro_scalp_transfer_adapters_v1_20260718.json"


def test_transfer_adapter_report_uses_chronological_selection_boundaries() -> None:
    report = json.loads(REPORT.read_text())
    splits = report["split_times"]
    ordered = [
        pd.Timestamp(splits[name])
        for name in (
            "calibration_start",
            "tune_start",
            "validation_start",
            "development_start",
            "development_end",
            "fresh_forward_start",
        )
    ]
    assert ordered == sorted(ordered)
    assert len(set(ordered)) == len(ordered)
    assert report["parent_weights_frozen"] is True
    assert report["training_performed"] is False
    assert report["parameter_updates"] == 0
    assert report["selection_uses_only_tune_split"] is True
    assert report["validation_used_for_reporting_and_gate_only"] is True
    assert report["development_used_for_selection"] is False
    assert report["activation_allowed"] is False
    assert report["order_submission_supported"] is False


def test_transfer_artifacts_are_shadow_only_and_exact_parent_bound() -> None:
    report = json.loads(REPORT.read_text())
    for asset, asset_report in report["assets"].items():
        artifact = json.loads(Path(asset_report["artifact"]).read_text())
        assert artifact["asset"] == asset
        assert artifact["parent_model_id"] == report["parent_model_id"]
        assert artifact["parent_model_sha256"] == report["parent_model_sha256"]
        assert artifact["parent_weights_frozen"] is True
        assert artifact["training_performed"] is False
        assert artifact["parameter_updates"] == 0
        assert artifact["artifact_execution_policy"]["enabled"] is False
        assert artifact["activation_allowed"] is False
        assert artifact["order_submission_supported"] is False
        assert artifact["fixed_holding_period_used"] is False
