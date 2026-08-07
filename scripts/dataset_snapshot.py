#!/usr/bin/env python3
"""P0-1 dataset content-addressing tool (docs/pipeline_integrity_and_research_redesign_20260730.md).

Registers/verifies sha256 + row-count + timestamp range for feature CSVs under data/splits/, so a
"frozen" dataset can be checked for drift instead of silently trusted. The Omega4.6.1 frozen
baseline (07-06, +145.34%/-10.13%/24trades) could not be reproduced because
data/splits/year_oos/training_features_2026_rebuilt.csv changed in place with no record of what
changed or when (root cause: upstream binance_data/metrics/*.zip files were retroactively revised
-- see project memory project-omega461-baseline-drift-bisection-20260730). This tool prevents that
failure mode from recurring undetected, for whichever of this repo's many ad-hoc build scripts
produces a dataset CSV.

Usage:
  python scripts/dataset_snapshot.py register --all
  python scripts/dataset_snapshot.py register data/splits/year_oos/training_features_2026.csv
  python scripts/dataset_snapshot.py verify --all
  python scripts/dataset_snapshot.py verify data/splits/year_oos/training_features_2026_rebuilt.csv

register: adds files not yet in the manifest (baseline pin). Never overwrites an existing entry --
an already-registered path with different content is reported as DRIFT and left untouched unless
--adopt-drift is passed explicitly (rare, deliberate, keeps the previous hash in the record).

verify: exits 1 if any checked file's current hash differs from its manifest entry. A path with no
manifest entry is reported as UNREGISTERED (not itself a failure -- register it first).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data/splits"
MANIFEST_PATH = SPLITS_DIR / "DATASET_MANIFEST.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _ts_range(path: Path) -> tuple[str | None, str | None, int]:
    try:
        df = pd.read_csv(path, usecols=["timestamp"], low_memory=False)
        ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
        rows = len(df)
        if ts.empty:
            return None, None, rows
        return str(ts.min()), str(ts.max()), rows
    except Exception:
        return None, None, -1


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"schema_version": "dataset_manifest_v1", "files": {}}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def build_lineage_record(path: Path | str) -> dict:
    """For scripts writing a new report.json: returns a `dataset_lineage` dict
    (`{features_path, features_sha256, manifest_version}`) that
    scripts/audit_omega_artifact_integrity_20260630.py's dataset_lineage gate can verify.
    Registers `path` first if it is not yet in the manifest (register() never overwrites an
    existing, different entry -- see cmd_register)."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    cmd_register([path], adopt_drift=False)
    manifest = _load_manifest()
    rel = _rel(path)
    entry = manifest["files"][rel]
    return {
        "features_path": rel,
        "features_sha256": entry["sha256"],
        "manifest_version": manifest["schema_version"],
    }


def cmd_register(paths: list[Path], adopt_drift: bool) -> int:
    manifest = _load_manifest()
    added = skipped = adopted = drifted = 0
    for path in paths:
        if not path.exists() or not path.is_file():
            print(f"  ! missing, skipped: {path}")
            continue
        rel = _rel(path)
        digest = _sha256_file(path)
        entry = manifest["files"].get(rel)
        if entry is None:
            ts_min, ts_max, rows = _ts_range(path)
            manifest["files"][rel] = {
                "sha256": digest, "size_bytes": path.stat().st_size, "rows": rows,
                "ts_min": ts_min, "ts_max": ts_max,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator_git_sha": _git_sha(),
            }
            added += 1
            print(f"  + registered: {rel} (rows={rows}, sha256={digest[:12]}...)")
        elif entry["sha256"] == digest:
            skipped += 1
        elif adopt_drift:
            ts_min, ts_max, rows = _ts_range(path)
            print(f"  ~ ADOPTING drift for {rel}: {entry['sha256'][:12]}... -> {digest[:12]}...")
            manifest["files"][rel] = {
                "sha256": digest, "size_bytes": path.stat().st_size, "rows": rows,
                "ts_min": ts_min, "ts_max": ts_max,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator_git_sha": _git_sha(),
                "previous_sha256": entry["sha256"],
            }
            adopted += 1
        else:
            print(f"  ! DRIFT (left as previously registered; rerun with --adopt-drift to accept): {rel}")
            print(f"      manifest sha256={entry['sha256']}")
            print(f"      current  sha256={digest}")
            drifted += 1
    _save_manifest(manifest)
    print(
        f"\nregistered={added} unchanged={skipped} adopted_drift={adopted} "
        f"drift_not_adopted={drifted} total_manifest_entries={len(manifest['files'])}"
    )
    return 1 if drifted else 0


def cmd_verify(paths: list[Path]) -> int:
    manifest = _load_manifest()
    ok = drift = unregistered = 0
    for path in paths:
        rel = _rel(path)
        entry = manifest["files"].get(rel)
        if entry is None:
            print(f"  ? UNREGISTERED: {rel}")
            unregistered += 1
            continue
        digest = _sha256_file(path)
        if digest == entry["sha256"]:
            ok += 1
        else:
            print(f"  X DRIFT: {rel}")
            print(f"      manifest sha256={entry['sha256']} (registered {entry['generated_at']})")
            print(f"      current  sha256={digest}")
            drift += 1
    print(f"\nok={ok} drift={drift} unregistered={unregistered}")
    return 1 if drift > 0 else 0


def _all_dataset_paths() -> list[Path]:
    csvs = sorted(SPLITS_DIR.rglob("*.csv"))
    baks = sorted(SPLITS_DIR.rglob("*.csv.bak_pre_extend_*"))
    return csvs + baks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="pin current file content as baseline")
    reg.add_argument("paths", nargs="*", help="CSV paths")
    reg.add_argument("--all", action="store_true", help="register every dataset CSV under data/splits/")
    reg.add_argument("--adopt-drift", action="store_true", help="explicitly accept new content for an already-registered path")

    ver = sub.add_parser("verify", help="check current file content against the manifest, exit 1 on drift")
    ver.add_argument("paths", nargs="*", help="CSV paths")
    ver.add_argument("--all", action="store_true", help="verify every dataset CSV under data/splits/")

    args = ap.parse_args()
    if args.all:
        paths = _all_dataset_paths()
    else:
        paths = [Path(p) for p in args.paths]
        if not paths:
            ap.error("provide paths or --all")

    if args.cmd == "register":
        return cmd_register(paths, args.adopt_drift)
    return cmd_verify(paths)


if __name__ == "__main__":
    raise SystemExit(main())
