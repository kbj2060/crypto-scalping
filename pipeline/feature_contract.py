from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "docs" / "feature_contract_manifest.json"


def load_feature_contract(manifest_path: str | Path | None = None) -> dict:
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    if not path.is_absolute():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_feature_groups(groups: dict[str, list[str]] | None) -> list[str]:
    if not groups:
        return []
    out: list[str] = []
    for vals in groups.values():
        out.extend(vals)
    return list(dict.fromkeys(out))


def cleanup_candidates(contract: dict, priorities: Iterable[str]) -> list[str]:
    cleanup = contract.get("cleanup_priorities", {})
    out: list[str] = []
    for p in priorities:
        out.extend(cleanup.get(p, []))
    return list(dict.fromkeys(out))


def apply_feature_drop(df: pd.DataFrame, drop_cols: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in dict.fromkeys(drop_cols) if c in df.columns]
    if not cols:
        return df, []
    return df.drop(columns=cols), cols


def rl_passthrough_keep(contract: dict) -> set[str]:
    keep = set()
    keep.update(flatten_feature_groups(contract.get("shared_base_features", {})))
    keep.update(flatten_feature_groups(contract.get("m7_outputs", {})))
    return keep
