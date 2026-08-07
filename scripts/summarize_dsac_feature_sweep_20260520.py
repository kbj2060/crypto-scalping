#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import torch


ROOT = Path("/home/llewyn/crypto-scalping")
SWEEP_ROOT = ROOT / "tmp/causal_regen_20260516/dsac_feature_sweep_20260520"


VAL_RE = re.compile(
    r"\[VAL\]\s+PnL:\s*([-+0-9.]+)%\s+\|\s+Tr:\s*([0-9]+)\s+\|\s+TPD:\s*([-+0-9.]+).*?L:\s*([0-9]+)\s+S:\s*([0-9]+).*?Score:([-+0-9.]+)"
)


def _load_ckpt(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"load_error": str(exc)}
    keys = [
        "epoch",
        "global_step",
        "best_val_score",
        "best_val_pnl",
        "bad_val_count",
        "base_state_dim",
        "state_dim",
        "state_schema",
    ]
    out = {k: ckpt.get(k) for k in keys if k in ckpt}
    if "meta" in ckpt and isinstance(ckpt["meta"], dict):
        meta = ckpt["meta"]
        out["all_features_enable"] = meta.get("all_features_enable")
        out["all_feature_count"] = meta.get("all_feature_count")
        out["all_feature_output_count"] = meta.get("all_feature_output_count")
        out["extra_feature_pca_enable"] = meta.get("extra_feature_pca_enable")
        out["extra_feature_pca_components"] = meta.get("extra_feature_pca_components")
        pca_meta = meta.get("extra_feature_pca_meta")
        if isinstance(pca_meta, dict):
            out["extra_pca_evr_sum"] = pca_meta.get("explained_variance_sum")
            out["extra_pca_input_dim"] = pca_meta.get("input_dim")
            out["extra_pca_output_dim"] = pca_meta.get("output_dim")
    return out


def _parse_log(path: Path) -> dict:
    if not path.exists():
        return {"val_count": 0, "last_vals": []}
    vals = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = VAL_RE.search(line)
        if not m:
            continue
        vals.append(
            {
                "pnl": float(m.group(1)),
                "trades": int(m.group(2)),
                "tpd": float(m.group(3)),
                "long": int(m.group(4)),
                "short": int(m.group(5)),
                "score": float(m.group(6)),
            }
        )
    best_by_score = max(vals, key=lambda x: x["score"], default=None)
    best_by_pnl = max(vals, key=lambda x: x["pnl"], default=None)
    return {
        "val_count": len(vals),
        "best_by_score": best_by_score,
        "best_by_pnl": best_by_pnl,
        "last_vals": vals[-3:],
    }


def main() -> None:
    rows = []
    for run_dir in sorted(p for p in SWEEP_ROOT.iterdir() if p.is_dir()):
        row = {
            "name": run_dir.name,
            "dir": str(run_dir),
        }
        row.update({f"ckpt_{k}": v for k, v in _load_ckpt(run_dir / "checkpoint.pth").items()})
        best = _load_ckpt(run_dir / "best.pth")
        row.update({f"best_{k}": v for k, v in best.items()})
        row.update(_parse_log(run_dir / "master.log"))
        rows.append(row)
    out = {
        "sweep_root": str(SWEEP_ROOT),
        "runs": rows,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
