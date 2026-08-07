#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_DIR = ROOT / "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528"
DEFAULT_EXPECTED_MODEL_ID = "alpha7_submodel_01965_decontam_v2_tp_20260528"
EXPECTED_CURRENT_REGIME = "clean_regime4_state24_sticky090_v2_20260517"

ENV_LIVE_DIR = "ALPHA7_LIVE_BASELINE_DIR"
ENV_EXPECTED_MODEL_ID = "ALPHA7_LIVE_BASELINE_MODEL_ID"


@dataclass(frozen=True)
class Alpha7LiveBaseline:
    live_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    model_id: str
    primary_parent: Path
    primary_summary: Path
    fallback_parent: Path
    fallback_summary: Path
    combo_summary: Path
    tp_sl_path_edge: Path


def _must_exist(path: Path, *, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"alpha7 baseline {kind} missing: {path}")


def get_live_baseline() -> Alpha7LiveBaseline:
    live_dir_raw = os.environ.get(ENV_LIVE_DIR)
    live_dir = Path(live_dir_raw) if live_dir_raw else DEFAULT_LIVE_DIR
    manifest_path = live_dir / "alpha7_live_manifest.json"
    _must_exist(manifest_path, kind="manifest")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_id = str(manifest.get("model_id", "")).strip()
    expected_model_id = os.environ.get(ENV_EXPECTED_MODEL_ID, DEFAULT_EXPECTED_MODEL_ID)
    if model_id != expected_model_id:
        raise ValueError(
            f"alpha7 baseline model_id mismatch: expected={expected_model_id} actual={model_id} "
            f"(manifest={manifest_path})"
        )

    lineage = manifest.get("lineage", {})
    current_regime = str(lineage.get("current_regime", "")).strip()
    if current_regime != EXPECTED_CURRENT_REGIME:
        raise ValueError(
            f"alpha7 baseline regime mismatch: expected={EXPECTED_CURRENT_REGIME} actual={current_regime} "
            f"(manifest={manifest_path})"
        )

    primary_parent = live_dir / "primary_parent.pkl"
    primary_summary = live_dir / "primary_summary.json"
    fallback_parent = live_dir / "fallback_alpha43_no_legacy_parent.pkl"
    fallback_summary = live_dir / "fallback_alpha43_no_legacy_summary.json"
    combo_summary = live_dir / "fallback_combo_summary.json"
    tp_sl_path_edge = live_dir / "tp_sl_path_edge_predictor.pkl"

    _must_exist(primary_parent, kind="primary_parent")
    _must_exist(primary_summary, kind="primary_summary")
    _must_exist(fallback_parent, kind="fallback_parent")
    _must_exist(fallback_summary, kind="fallback_summary")
    _must_exist(combo_summary, kind="combo_summary")
    _must_exist(tp_sl_path_edge, kind="tp_sl_path_edge_predictor")

    return Alpha7LiveBaseline(
        live_dir=live_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        model_id=model_id,
        primary_parent=primary_parent,
        primary_summary=primary_summary,
        fallback_parent=fallback_parent,
        fallback_summary=fallback_summary,
        combo_summary=combo_summary,
        tp_sl_path_edge=tp_sl_path_edge,
    )
