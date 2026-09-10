#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path("/home/llewyn/crypto-scalping")
SWEEP_ROOT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else os.getenv("DSAC_SWEEP_ROOT", ROOT / "tmp/causal_regen_20260516/dsac_5d_feature_tests_20260520")
)

VAL_RE = re.compile(
    r"\[VAL\]\s+PnL:\s*([-+0-9.]+)%\s+\|\s+Tr:\s*([0-9]+)\s+\|\s+TPD:\s*([-+0-9.]+).*?WR:([0-9]+)%.*?MDD:([-+0-9.]+)%.*?L:\s*([0-9]+)\s+S:\s*([0-9]+).*?Score:([-+0-9.]+).*?pass=([0-9]+)"
)
EP_RE = re.compile(r"Ep\s+([0-9]{4}).*?PnL:\s*([-+0-9.]+)%\s+Tr:\s*([0-9]+)\s+WR:\s*([0-9]+)%")


def _parse_log(path: Path) -> dict:
    vals = []
    eps = []
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    for line in text.splitlines():
        m = VAL_RE.search(line)
        if m:
            vals.append(
                {
                    "pnl": float(m.group(1)),
                    "trades": int(m.group(2)),
                    "tpd": float(m.group(3)),
                    "wr": int(m.group(4)),
                    "mdd": float(m.group(5)),
                    "long": int(m.group(6)),
                    "short": int(m.group(7)),
                    "score": float(m.group(8)),
                    "pass": int(m.group(9)),
                }
            )
        e = EP_RE.search(line)
        if e:
            eps.append(
                {
                    "ep": int(e.group(1)),
                    "pnl": float(e.group(2)),
                    "trades": int(e.group(3)),
                    "wr": int(e.group(4)),
                }
            )
    return {
        "ep_count": len(eps),
        "last_ep": eps[-1] if eps else None,
        "val_count": len(vals),
        "best_by_score": max(vals, key=lambda x: x["score"], default=None),
        "best_by_pnl": max(vals, key=lambda x: x["pnl"], default=None),
        "last_val": vals[-1] if vals else None,
        "last_vals": vals[-3:],
    }


def main() -> None:
    runs = []
    if not SWEEP_ROOT.exists():
        print(json.dumps({"sweep_root": str(SWEEP_ROOT), "runs": []}, ensure_ascii=False, indent=2))
        return
    for d in sorted(p for p in SWEEP_ROOT.iterdir() if p.is_dir() and not p.name.startswith("router_")):
        row = {"name": d.name, "dir": str(d)}
        row.update(_parse_log(d / "master.log"))
        runs.append(row)
    ranked = sorted(runs, key=lambda r: (r.get("best_by_score") or {}).get("score", -1e18), reverse=True)
    out = {"sweep_root": str(SWEEP_ROOT), "runs": runs, "ranked_by_score": ranked}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
