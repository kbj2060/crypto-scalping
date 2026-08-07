#!/usr/bin/env python3
"""Build exact-timestamp probability ensembles from a fixed parent seed set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "validation", "oos")
ENSEMBLE_SCOPE = "entry_direction_and_quality_probabilities_only"
PROMOTION_BLOCKERS = (
    "exit_head_not_ensembled",
    "live_parent_bundle_not_built",
)
PROBABILITY_SUFFIXES = (
    "dir_p_cash",
    "dir_p_long",
    "dir_p_short",
    "quality_p_cash",
    "quality_p_long",
    "quality_p_short",
)
ROUTER_SUFFIXES = ("router_expert", "router_confidence", "router_margin")


def _prefix(frame: pd.DataFrame) -> str:
    matches = [column for column in frame.columns if column.endswith("dir_p_cash")]
    if len(matches) != 1:
        raise ValueError(f"expected one direction probability prefix, got {matches}")
    return matches[0][: -len("dir_p_cash")]


def _require_probabilities(frame: pd.DataFrame, columns: list[str], *, seed: int) -> None:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"seed {seed} contains non-finite probabilities")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError(f"seed {seed} contains probabilities outside [0, 1]")
    for group in (columns[:3], columns[3:]):
        sums = frame[group].to_numpy(dtype=np.float64).sum(axis=1)
        if not np.allclose(sums, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(f"seed {seed} probability rows do not sum to one")


def ensemble_prediction_frames(
    seed_frames: dict[int, pd.DataFrame], *, quality_threshold: float
) -> pd.DataFrame:
    if len(seed_frames) < 2:
        raise ValueError("fixed-seed ensemble requires at least two distinct seeds")
    if not np.isfinite(quality_threshold) or not 0.0 <= quality_threshold <= 1.0:
        raise ValueError(f"invalid quality_threshold: {quality_threshold}")

    ordered = sorted((int(seed), frame.reset_index(drop=True)) for seed, frame in seed_frames.items())
    first_seed, first = ordered[0]
    if "timestamp" not in first.columns:
        raise ValueError(f"seed {first_seed} is missing timestamp")
    prefix = _prefix(first)
    probability_columns = [prefix + suffix for suffix in PROBABILITY_SUFFIXES]
    required = ["timestamp", *probability_columns, *(prefix + suffix for suffix in ROUTER_SUFFIXES)]
    missing = [column for column in required if column not in first.columns]
    if missing:
        raise ValueError(f"seed {first_seed} is missing columns: {missing}")

    for seed, frame in ordered:
        if list(frame.columns) != list(first.columns):
            raise ValueError(f"seed {seed} prediction column contract mismatch")
        if not frame["timestamp"].equals(first["timestamp"]):
            raise ValueError(f"seed {seed} timestamp contract mismatch")
        if _prefix(frame) != prefix:
            raise ValueError(f"seed {seed} prediction prefix mismatch")
        for suffix in ROUTER_SUFFIXES:
            column = prefix + suffix
            if not frame[column].equals(first[column]):
                raise ValueError(f"seed {seed} deterministic router column mismatch: {column}")
        _require_probabilities(frame, probability_columns, seed=seed)

    output = first.copy()
    for column in probability_columns:
        output[column] = np.mean(
            [frame[column].to_numpy(dtype=np.float64) for _, frame in ordered], axis=0
        )

    direction_columns = [prefix + suffix for suffix in PROBABILITY_SUFFIXES[:3]]
    quality_columns = [prefix + suffix for suffix in PROBABILITY_SUFFIXES[3:]]
    direction = output[direction_columns].to_numpy(dtype=np.float64)
    quality = output[quality_columns].to_numpy(dtype=np.float64)
    action = np.argmax(direction, axis=1).astype(np.int64)
    row_index = np.arange(len(output))
    quality_for_action = quality[row_index, action]
    final_action = np.where(
        (action != 0) & (quality_for_action >= float(quality_threshold)), action, 0
    ).astype(np.int64)

    output[prefix + "dir_confidence"] = direction.max(axis=1)
    output[prefix + "dir_side_edge"] = direction[:, 1] - direction[:, 2]
    output[prefix + "dir_trade_prob"] = direction[:, 1] + direction[:, 2]
    output[prefix + "dir_action"] = action
    output[prefix + "quality_for_action"] = quality_for_action
    output[prefix + "quality_threshold"] = float(quality_threshold)
    output[prefix + "final_action"] = final_action
    return output[list(first.columns)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_seed_dir(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("--seed-dir must be SEED=PATH")
    return int(seed_text), Path(path_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", action="append", type=_parse_seed_dir, required=True)
    parser.add_argument("--quality-threshold", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    seed_dirs = dict(args.seed_dir)
    if len(seed_dirs) != len(args.seed_dir):
        raise RuntimeError("duplicate seed in --seed-dir")
    if len(seed_dirs) < 2:
        raise RuntimeError("at least two fixed seeds are required")
    tag = f"q{int(round(float(args.quality_threshold) * 100)):03d}"
    args.out_dir.mkdir(parents=True, exist_ok=False)

    input_artifacts: dict[str, dict[str, dict[str, object]]] = {}
    output_artifacts: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        frames: dict[int, pd.DataFrame] = {}
        input_artifacts[split] = {}
        for seed, seed_dir in sorted(seed_dirs.items()):
            path = seed_dir / f"{split}_predictions_{tag}.csv"
            if not path.is_file():
                raise RuntimeError(f"missing seed prediction artifact: {path}")
            frames[seed] = pd.read_csv(path)
            input_artifacts[split][str(seed)] = {
                "path": str(path),
                "sha256": _sha256(path),
            }
        output = ensemble_prediction_frames(
            frames, quality_threshold=float(args.quality_threshold)
        )
        output_path = args.out_dir / f"{split}_predictions_{tag}.csv"
        output.to_csv(output_path, index=False)
        output_artifacts[split] = {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "rows": int(len(output)),
        }

    manifest = {
        "schema_version": "fixed_seed_prediction_ensemble_v1",
        "seeds": sorted(seed_dirs),
        "ensemble_method": "arithmetic_mean_probabilities_before_threshold",
        "ensemble_scope": ENSEMBLE_SCOPE,
        "exit_head_ensemble": False,
        "live_parent_bundle_available": False,
        "promotion_eligible": False,
        "promotion_blockers": list(PROMOTION_BLOCKERS),
        "quality_threshold": float(args.quality_threshold),
        "prediction_tag": tag,
        "trade_outcomes_used_for_seed_selection": False,
        "input_artifacts": input_artifacts,
        "output_artifacts": output_artifacts,
    }
    manifest_path = args.out_dir / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "seeds": sorted(seed_dirs), "tag": tag}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
