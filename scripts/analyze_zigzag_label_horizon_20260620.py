#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
PRICE_FILES = {
    2025: ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv",
    2026: ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv",
}
OUT_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_label_horizon_audit_20260620"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], col: str, default: float = 0.0) -> float:
    try:
        return float(row.get(col, default))
    except Exception:
        return default


def _q(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * p))))
    return float(s[idx])


def _summ(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(vals),
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "p75": _q(vals, 0.75),
        "p90": _q(vals, 0.90),
        "p95": _q(vals, 0.95),
        "p99": _q(vals, 0.99),
        "max": float(max(vals)),
    }


def _segments(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    segs: list[dict[str, Any]] = []
    if not rows:
        return segs
    cur = int(_f(rows[0], "zigzag_action"))
    start = 0
    for i in range(1, len(rows)):
        action = int(_f(rows[i], "zigzag_action"))
        if action != cur:
            segs.append({"action": cur, "start": start, "end": i - 1, "length": i - start})
            cur = action
            start = i
    segs.append({"action": cur, "start": start, "end": len(rows) - 1, "length": len(rows) - start})
    return segs


def _future_return_holds(price: list[dict[str, str]], labels: list[dict[str, str]], horizons: list[int]) -> list[dict[str, Any]]:
    pmap = {r["timestamp"]: r for r in price}
    rows = []
    aligned = []
    for lab in labels:
        p = pmap.get(lab["timestamp"])
        if p is not None:
            aligned.append({**p, **lab})
    for i, row in enumerate(aligned):
        action = int(_f(row, "zigzag_action"))
        if action == 0 or i + 2 >= len(aligned):
            continue
        side = 1 if action == 1 else -1
        entry_i = i + 1
        entry = _f(aligned[entry_i], "open")
        if entry <= 0.0:
            continue
        best_mfe = 0.0
        first_05 = 0
        first_10 = 0
        first_20 = 0
        first_adverse_10 = 0
        max_h = min(max(horizons), len(aligned) - 1 - entry_i)
        for h in range(1, max_h + 1):
            r = aligned[entry_i + h]
            high = _f(r, "high")
            low = _f(r, "low")
            if side > 0:
                mfe = (high - entry) / entry
                mae = (low - entry) / entry
            else:
                mfe = (entry - low) / entry
                mae = (entry - high) / entry
            best_mfe = max(best_mfe, mfe)
            if not first_05 and mfe >= 0.005:
                first_05 = h
            if not first_10 and mfe >= 0.010:
                first_10 = h
            if not first_20 and mfe >= 0.020:
                first_20 = h
            if not first_adverse_10 and mae <= -0.010:
                first_adverse_10 = h
        fwd = {}
        for h in horizons:
            j = min(len(aligned) - 1, entry_i + h)
            close = _f(aligned[j], "close")
            fwd[f"ret_h{h}"] = ((close - entry) / entry if side > 0 else (entry - close) / entry)
        rows.append(
            {
                "timestamp": row["timestamp"],
                "action": action,
                "first_mfe_0p5pct_bars": first_05,
                "first_mfe_1pct_bars": first_10,
                "first_mfe_2pct_bars": first_20,
                "first_adverse_1pct_bars": first_adverse_10,
                "best_mfe_192": best_mfe,
                **fwd,
            }
        )
    return rows


def _analyze_year(year: int, label_dir: Path) -> dict[str, Any]:
    labels = _read_csv(label_dir / f"zigzag_action_labels_{year}.csv")
    price = _read_csv(PRICE_FILES[year])
    segs = _segments(labels)
    active = [s for s in segs if int(s["action"]) != 0]
    cash = [s for s in segs if int(s["action"]) == 0]
    by_action = {}
    for action, name in [(0, "cash"), (1, "long"), (2, "short")]:
        vals = [float(s["length"]) for s in segs if int(s["action"]) == action]
        by_action[name] = _summ(vals)
    holds = _future_return_holds(price, labels, [12, 24, 48, 96, 192])
    hit_cols = ["first_mfe_0p5pct_bars", "first_mfe_1pct_bars", "first_mfe_2pct_bars", "first_adverse_1pct_bars"]
    hit_summary = {}
    for col in hit_cols:
        nonzero = [float(r[col]) for r in holds if float(r[col]) > 0]
        hit_summary[col] = {
            **_summ(nonzero),
            "hit_ratio": float(len(nonzero) / len(holds)) if holds else 0.0,
        }
    ret_summary = {}
    for h in [12, 24, 48, 96, 192]:
        vals = [float(r[f"ret_h{h}"]) for r in holds]
        ret_summary[f"ret_h{h}"] = _summ(vals)
    rows = []
    for s in segs:
        rows.append(
            {
                "year": year,
                "action": s["action"],
                "start_timestamp": labels[int(s["start"])]["timestamp"],
                "end_timestamp": labels[int(s["end"])]["timestamp"],
                "length_bars": s["length"],
                "length_hours": float(s["length"]) * 5.0 / 60.0,
            }
        )
    return {
        "year": year,
        "rows": len(labels),
        "segment_count": len(segs),
        "active_segment_count": len(active),
        "cash_segment_count": len(cash),
        "counts": dict(Counter(int(_f(r, "zigzag_action")) for r in labels)),
        "segment_length_bars": {
            "all": _summ([float(s["length"]) for s in segs]),
            "active": _summ([float(s["length"]) for s in active]),
            **by_action,
        },
        "future_hold_proxy": hit_summary,
        "future_return_by_horizon": ret_summary,
        "segment_rows": rows,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    label_dir = args.label_dir if args.label_dir.is_absolute() else ROOT / args.label_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_id": "zigzag_label_horizon_audit_20260620",
        "source_label_dir": str(label_dir),
        "notes": [
            "Segment length measures consecutive identical zigzag_action rows.",
            "Future hold proxy uses next-bar entry and first future high/low touch in label direction.",
            "This is analysis only; no label mutation is performed.",
        ],
        "years": {},
    }
    all_rows = []
    for year in (2025, 2026):
        y = _analyze_year(year, label_dir)
        report["years"][str(year)] = {k: v for k, v in y.items() if k != "segment_rows"}
        all_rows.extend(y["segment_rows"])
    seg_path = out_dir / "zigzag_segment_lengths.csv"
    with seg_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    report["artifacts"] = {"segment_lengths": str(seg_path.relative_to(ROOT)), "report": str((out_dir / "report.json").relative_to(ROOT))}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
