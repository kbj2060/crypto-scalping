#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/zigzag_second_stage_retrain_all_20260531"


FAMILIES = [
    "ai_all_legacy",
    "ai_direction_legacy",
    "ai_role_risk_context",
    "m7_all_nonp0",
    "m7_direction_legacy",
    "m7_unsup_risk_context",
    "regime3_current_context",
    "regime3_risk_context",
    "regime3_all_context",
    "all_second_stage_nonp0",
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _run(name: str, cmd: list[str], log_path: Path) -> dict[str, Any]:
    print(f"[zigzag-retrain] start {name}", flush=True)
    print("[zigzag-retrain] cmd " + " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed rc={proc.returncode}; log={log_path}")
    print(f"[zigzag-retrain] done {name}", flush=True)
    return {"name": name, "cmd": cmd, "log": str(log_path), "returncode": int(proc.returncode)}


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_rows(source: str, audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = audit.get("pairs", audit.get("results", []))
    for item in pairs:
        score = item.get("score_metrics", {})
        train = item.get("train_metrics", {})
        rows.append(
            {
                "source": source,
                "family": item.get("family", source),
                "train_year": item.get("train_year"),
                "score_year": item.get("score_year"),
                "feature_count": item.get("feature_count"),
                "score_bacc": score.get("balanced_accuracy"),
                "score_ovr_auc": score.get("ovr_auc"),
                "train_bacc": train.get("balanced_accuracy"),
                "train_ovr_auc": train.get("ovr_auc"),
                "score_csv": item.get("score_csv"),
                "model_path": item.get("model_path"),
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--skip-ai-patch", action="store_true")
    p.add_argument("--task-type", choices=("GPU", "CPU"), default="GPU")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs = args.out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    m7_out = args.out_dir / "m7_zigzag_action_hgb"
    family_out = args.out_dir / "family_sweep"
    ai_patch_out = args.out_dir / "ai_zigzag_patchmix_catboost"

    runs.append(
        _run(
            "m7_zigzag_action_hgb",
            [
                str(PY),
                "scripts/train_wave3_m7_action_hgb_20260531.py",
                "--out-dir",
                str(m7_out),
            ],
            logs / "m7_zigzag_action_hgb.log",
        )
    )
    runs.append(
        _run(
            "zigzag_second_stage_family_sweep",
            [
                str(PY),
                "scripts/train_wave3_second_stage_family_sweep_20260531.py",
                "--out-dir",
                str(family_out),
                "--task-type",
                str(args.task_type),
                "--families",
                ",".join(FAMILIES),
            ],
            logs / "zigzag_second_stage_family_sweep.log",
        )
    )
    if not args.skip_ai_patch:
        runs.append(
            _run(
                "ai_zigzag_patchmix_catboost",
                [
                    str(PY),
                    "scripts/train_wave3_ai_patchmix_catboost_20260531.py",
                    "--out-dir",
                    str(ai_patch_out),
                    "--task-type",
                    str(args.task_type),
                ],
                logs / "ai_zigzag_patchmix_catboost.log",
            )
        )

    audits = {
        "m7_zigzag_action_hgb": str(m7_out / "m7_zigzag_action_hgb_audit.json"),
        "zigzag_second_stage_family_sweep": str(family_out / "zigzag_second_stage_family_sweep_audit.json"),
    }
    if not args.skip_ai_patch:
        audits["ai_zigzag_patchmix_catboost"] = str(ai_patch_out / "ai_zigzag_patchmix_catboost_audit.json")

    rows: list[dict[str, Any]] = []
    for source, path_str in audits.items():
        rows.extend(_pair_rows(source, _load(Path(path_str))))

    summary = {
        "type": "zigzag_second_stage_retrain_all",
        "label_contract": str(ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_label_audit.json"),
        "families": FAMILIES,
        "regime3_pred_excluded": True,
        "regime4_excluded": True,
        "runs": runs,
        "audits": audits,
        "rows": rows,
    }
    out = args.out_dir / "zigzag_second_stage_retrain_all_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"summary": str(out), "rows": len(rows)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
