#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


EP_RE = re.compile(
    r"Ep\s+(?P<ep>\d+)\s+\|\s+phase=(?P<phase>[^|]+)\s+\|\s+PnL:\s*(?P<pnl>[+-]?\d+\.\d+)%\s+"
    r"Tr:\s*(?P<trades>\d+)\s+WR:\s*(?P<wr>\d+)%"
)

VAL_RE = re.compile(
    r"\[VAL\]\s+PnL:\s*(?P<pnl>[+-]?\d+\.\d+)%\s+\|\s+Tr:\s*(?P<trades>\d+)\s+\|\s+TPD:(?P<tpd>[+-]?\d+\.\d+)\s+\|\s+"
    r"WR:(?P<wr>\d+)%\s+\|\s+MDD:(?P<mdd>[+-]?\d+\.\d+)%\s+\|\s+L:\s*(?P<long>\d+)\s+S:\s*(?P<short>\d+)\s+\|"
    r".*?Score:(?P<score>[+-]?\d+\.\d+)"
)


@dataclass
class RunSummary:
    variant: str
    run_dir: str
    spec_path: str | None
    feature_count: int | None
    episodes_target: int | None
    latest_episode: int | None
    latest_phase: str | None
    latest_train_pnl: float | None
    latest_train_trades: int | None
    latest_train_wr: int | None
    latest_val_pnl: float | None
    latest_val_trades: int | None
    latest_val_tpd: float | None
    latest_val_wr: int | None
    latest_val_mdd: float | None
    latest_val_long: int | None
    latest_val_short: int | None
    latest_val_score: float | None
    best_val_pnl: float | None
    best_val_trades: int | None
    best_val_tpd: float | None
    best_val_wr: int | None
    best_val_mdd: float | None
    best_val_long: int | None
    best_val_short: int | None
    best_val_score: float | None
    completed: bool


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_log(log_path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    latest_ep: dict[str, Any] | None = None
    vals: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ep_match = EP_RE.search(line)
        if ep_match:
            latest_ep = {
                "ep": int(ep_match.group("ep")),
                "phase": ep_match.group("phase").strip(),
                "pnl": float(ep_match.group("pnl")),
                "trades": int(ep_match.group("trades")),
                "wr": int(ep_match.group("wr")),
            }
            continue
        val_match = VAL_RE.search(line)
        if val_match:
            vals.append(
                {
                    "ep": latest_ep["ep"] if latest_ep else None,
                    "pnl": float(val_match.group("pnl")),
                    "trades": int(val_match.group("trades")),
                    "tpd": float(val_match.group("tpd")),
                    "wr": int(val_match.group("wr")),
                    "mdd": float(val_match.group("mdd")),
                    "long": int(val_match.group("long")),
                    "short": int(val_match.group("short")),
                    "score": float(val_match.group("score")),
                }
            )
    return latest_ep, vals


def _summarize_run(run_dir: Path, episode_cutoff: int | None = None) -> RunSummary | None:
    log_path = run_dir / "launcher.log"
    manifest_path = run_dir / "variant_manifest.json"
    if not log_path.exists() or not manifest_path.exists():
        return None
    manifest = _load_json(manifest_path)
    spec = manifest.get("spec", {})
    latest_ep, vals = _parse_log(log_path)
    if episode_cutoff is not None:
        if latest_ep is not None and latest_ep["ep"] > episode_cutoff:
            latest_ep = None
            for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                ep_match = EP_RE.search(line)
                if ep_match:
                    ep_num = int(ep_match.group("ep"))
                    if ep_num > episode_cutoff:
                        break
                    latest_ep = {
                        "ep": ep_num,
                        "phase": ep_match.group("phase").strip(),
                        "pnl": float(ep_match.group("pnl")),
                        "trades": int(ep_match.group("trades")),
                        "wr": int(ep_match.group("wr")),
                    }
            vals = [row for row in vals if row.get("ep") is not None and row["ep"] <= episode_cutoff]
    latest_val = vals[-1] if vals else None
    best_val = max(vals, key=lambda row: row["score"]) if vals else None
    episodes_target = manifest.get("episodes")
    if episode_cutoff is not None:
        episodes_target = episode_cutoff
    latest_episode = latest_ep["ep"] if latest_ep else None
    completed = bool(latest_episode is not None and episodes_target is not None and latest_episode >= episodes_target)
    return RunSummary(
        variant=manifest.get("variant", run_dir.name),
        run_dir=str(run_dir),
        spec_path=manifest.get("spec_path"),
        feature_count=spec.get("feature_count"),
        episodes_target=episodes_target,
        latest_episode=latest_episode,
        latest_phase=latest_ep["phase"] if latest_ep else None,
        latest_train_pnl=latest_ep["pnl"] if latest_ep else None,
        latest_train_trades=latest_ep["trades"] if latest_ep else None,
        latest_train_wr=latest_ep["wr"] if latest_ep else None,
        latest_val_pnl=latest_val["pnl"] if latest_val else None,
        latest_val_trades=latest_val["trades"] if latest_val else None,
        latest_val_tpd=latest_val["tpd"] if latest_val else None,
        latest_val_wr=latest_val["wr"] if latest_val else None,
        latest_val_mdd=latest_val["mdd"] if latest_val else None,
        latest_val_long=latest_val["long"] if latest_val else None,
        latest_val_short=latest_val["short"] if latest_val else None,
        latest_val_score=latest_val["score"] if latest_val else None,
        best_val_pnl=best_val["pnl"] if best_val else None,
        best_val_trades=best_val["trades"] if best_val else None,
        best_val_tpd=best_val["tpd"] if best_val else None,
        best_val_wr=best_val["wr"] if best_val else None,
        best_val_mdd=best_val["mdd"] if best_val else None,
        best_val_long=best_val["long"] if best_val else None,
        best_val_short=best_val["short"] if best_val else None,
        best_val_score=best_val["score"] if best_val else None,
        completed=completed,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-dir",
        default="/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/dsac_feature_screen_regime_fixed_20260521",
    )
    ap.add_argument("--episode-cutoff", type=int, default=None)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    summaries: list[RunSummary] = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        row = _summarize_run(child, episode_cutoff=args.episode_cutoff)
        if row is not None:
            summaries.append(row)

    out_rows = [asdict(row) for row in summaries]
    out_rows.sort(
        key=lambda row: (
            1 if row["completed"] else 0,
            float("-inf") if row["best_val_score"] is None else row["best_val_score"],
            -1 if row["latest_episode"] is None else row["latest_episode"],
        ),
        reverse=True,
    )
    suffix = f"_ep{args.episode_cutoff}" if args.episode_cutoff is not None else ""
    csv_path = runs_dir / f"screen_run_summary{suffix}.csv"
    json_path = runs_dir / f"screen_run_summary{suffix}.json"
    if out_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(json.dumps(out_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"rows": len(out_rows), "csv": str(csv_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
