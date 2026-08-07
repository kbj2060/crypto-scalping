#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PY = Path("/home/llewyn/miniconda3/envs/quant_ai/bin/python")
BASE_OUT = ROOT / "tmp/causal_regen_20260516/alpha6_meta_extreme_loop_20260524"
LOG_DIR = ROOT / "logs"


def _now() -> datetime:
    return datetime.now()


def _read_summary(path: Path) -> dict[str, Any] | None:
    p = path / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _best_eval(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    rows = summary.get("eval") or []
    for row in rows:
        if row.get("split") == "full_val":
            return dict(row)
    return dict(rows[-1]) if rows else {}


def _score(row: dict[str, Any]) -> float:
    pnl = float(row.get("pnl", -999.0) or -999.0)
    mdd = abs(float(row.get("mdd", -999.0) or -999.0))
    trades = int(row.get("trades", 0) or 0)
    if trades < 20:
        return pnl - 100.0
    return pnl / max(mdd, 1e-9)


def _append_ranking(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "ts",
        "name",
        "stage",
        "status",
        "score",
        "pnl",
        "mdd",
        "calmar",
        "trades",
        "wr",
        "out_dir",
        "cmd",
    ]
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def _run(name: str, stage: str, cmd: list[str], out_dir: Path, deadline: datetime, ranking: Path) -> dict[str, Any]:
    if _now() >= deadline:
        return {"name": name, "stage": stage, "status": "deadline"}
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[run] {name} stage={stage} out={out_dir}", flush=True)
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    summary = _read_summary(out_dir)
    ev = _best_eval(summary)
    status = "ok" if proc.returncode == 0 and summary else f"fail:{proc.returncode}"
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "stage": stage,
        "status": status,
        "score": _score(ev),
        "pnl": ev.get("pnl", ""),
        "mdd": ev.get("mdd", ""),
        "calmar": ev.get("calmar", ""),
        "trades": ev.get("trades", ""),
        "wr": ev.get("wr", ""),
        "out_dir": str(out_dir),
        "cmd": " ".join(cmd),
        "seconds": round(time.time() - started, 3),
    }
    _append_ranking(ranking, row)
    print(
        f"[done] {name} status={status} pnl={row['pnl']} mdd={row['mdd']} trades={row['trades']} score={row['score']}",
        flush=True,
    )
    return row


def _meta_cmd(
    out_dir: Path,
    *,
    label_edge: float,
    active_prob: float,
    margin: float,
    source: str,
    risk_episodes: int,
    warmup: int,
    cap: int,
    oof_iter: int,
    exit_iter: int,
) -> list[str]:
    return [
        str(PY),
        str(ROOT / "scripts/train_alpha6_lgbm_meta_dsac_risk_20260524.py"),
        "--out-dir",
        str(out_dir),
        "--meta-label-source",
        source,
        "--label-min-edge",
        str(label_edge),
        "--meta-active-prob-min",
        str(active_prob),
        "--meta-margin-min",
        str(margin),
        "--risk-episodes",
        str(risk_episodes),
        "--risk-warmup",
        str(warmup),
        "--risk-batch-size",
        "256",
        "--max-risk-train-candidates",
        str(cap),
        "--oof-folds",
        "2",
        "--oof-iterations",
        str(oof_iter),
        "--oof-exit-iterations",
        str(exit_iter),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Alpha6 meta/DSAC extreme search until a wall-clock deadline.")
    ap.add_argument("--until-hour", type=int, default=9)
    ap.add_argument("--until-minute", type=int, default=0)
    ap.add_argument("--base-out", type=Path, default=BASE_OUT)
    args = ap.parse_args()

    deadline = _now().replace(hour=int(args.until_hour), minute=int(args.until_minute), second=0, microsecond=0)
    if deadline <= _now():
        deadline += timedelta(days=1)
    base = args.base_out
    base.mkdir(parents=True, exist_ok=True)
    ranking = base / "ranking.csv"
    manifest: dict[str, Any] = {
        "started_at": _now().isoformat(timespec="seconds"),
        "deadline": deadline.isoformat(timespec="seconds"),
        "experiments": [],
    }
    (base / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    screen_grid: list[dict[str, Any]] = []
    for edge in (0.0, 0.0005, 0.0010, 0.0020):
        for active_prob, margin in ((0.0, 0.0), (0.45, 0.0), (0.55, 0.03), (0.65, 0.06)):
            screen_grid.append({"edge": edge, "active_prob": active_prob, "margin": margin, "source": "candidate"})
    screen_grid.append({"edge": 0.002, "active_prob": 0.65, "margin": 0.08, "source": "oracle"})

    rows: list[dict[str, Any]] = []
    for k, cfg in enumerate(screen_grid, start=1):
        if _now() + timedelta(minutes=6) >= deadline:
            break
        name = f"screen_{k:02d}_{cfg['source']}_e{cfg['edge']}_p{cfg['active_prob']}_m{cfg['margin']}".replace(".", "p")
        out_dir = base / name
        row = _run(
            name,
            "screen",
            _meta_cmd(
                out_dir,
                label_edge=float(cfg["edge"]),
                active_prob=float(cfg["active_prob"]),
                margin=float(cfg["margin"]),
                source=str(cfg["source"]),
                risk_episodes=2,
                warmup=600,
                cap=5000,
                oof_iter=45,
                exit_iter=15,
            ),
            out_dir,
            deadline,
            ranking,
        )
        row.update(cfg)
        rows.append(row)

    ranked = sorted([r for r in rows if str(r.get("status")) == "ok"], key=lambda r: float(r.get("score", -999)), reverse=True)
    promote = ranked[:3]
    for k, src in enumerate(promote, start=1):
        if _now() + timedelta(minutes=18) >= deadline:
            break
        name = f"promote_{k:02d}_from_{src['name']}"
        out_dir = base / name
        row = _run(
            name,
            "promote",
            _meta_cmd(
                out_dir,
                label_edge=float(src["edge"]),
                active_prob=float(src["active_prob"]),
                margin=float(src["margin"]),
                source=str(src["source"]),
                risk_episodes=8,
                warmup=3000,
                cap=0,
                oof_iter=120,
                exit_iter=40,
            ),
            out_dir,
            deadline,
            ranking,
        )
        row.update({"promoted_from": src["name"], "edge": src["edge"], "active_prob": src["active_prob"], "margin": src["margin"], "source": src["source"]})
        rows.append(row)

    # Preserve the strongest known rule stack as a non-DSAC fallback reference.
    if _now() + timedelta(minutes=10) < deadline:
        name = "fallback_scoring_stack_full"
        out_dir = base / name
        row = _run(
            name,
            "fallback",
            [
                str(PY),
                str(ROOT / "scripts/backtest_alpha6_label_scoring_stack_20260524.py"),
                "--grid",
                "full",
                "--out-dir",
                str(out_dir),
            ],
            out_dir,
            deadline,
            ranking,
        )
        rows.append(row)

    all_rows = []
    if ranking.exists():
        with ranking.open() as f:
            all_rows = list(csv.DictReader(f))
    best = sorted(all_rows, key=lambda r: float(r.get("score") or -999), reverse=True)[:10]
    final = {
        "finished_at": _now().isoformat(timespec="seconds"),
        "deadline": deadline.isoformat(timespec="seconds"),
        "best": best,
        "ranking_csv": str(ranking),
    }
    (base / "final_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2))
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
