#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_parent72_loose_20260620"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_entry_label_path_optimal_exit_labels_20260620"
FEE_RATE = 0.0005
SLIP_RATE = 0.0002
MAKER_FEE_MULT = 0.20
BASE_NOTIONAL = 0.45
BASE_LEVERAGE = 2.0
BASE_TAKE_PROFIT = 0.026
BASE_STOP_LOSS = 0.014


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def _limit_touched(rows: list[dict[str, str]], fill_i: int, price: float, side: int, *, entry: bool) -> bool:
    high = _f(rows[fill_i], "high")
    low = _f(rows[fill_i], "low")
    is_buy = (side > 0 and entry) or (side < 0 and not entry)
    return bool(low <= price) if is_buy else bool(high >= price)


def _close_fallback_price(rows: list[dict[str, str]], fill_i: int, side: int, slip_eff: float) -> float:
    px = _f(rows[fill_i], "close")
    return px * (1.0 - slip_eff) if side > 0 else px * (1.0 + slip_eff)


def _try_execution(rows: list[dict[str, str]], signal_i: int, side: int, *, entry: bool, fee_eff: float, slip_eff: float) -> tuple[bool, float, float, int, str]:
    fill_i = min(int(signal_i) + 1, len(rows) - 1)
    limit_px = _f(rows[fill_i], "open")
    if limit_px > 0.0 and _limit_touched(rows, fill_i, limit_px, side, entry=entry):
        return True, limit_px, fee_eff * MAKER_FEE_MULT, fill_i, "signal_immediate_maker_limit"
    if entry:
        return False, 0.0, 0.0, fill_i, "signal_immediate_limit_miss"
    return True, _close_fallback_price(rows, fill_i, side, slip_eff), fee_eff, fill_i, "exit_market_fallback_after_limit_miss_close"


def _exit_net(rows: list[dict[str, str]], signal_i: int, side: int, entry_price: float, cash_after_entry_fee: float, *, fee_eff: float, slip_eff: float) -> tuple[float, int, str]:
    filled, exit_px, exit_fee, fill_i, route = _try_execution(rows, signal_i, side, entry=False, fee_eff=fee_eff, slip_eff=slip_eff)
    if not filled:
        return cash_after_entry_fee - 1.0, fill_i, route
    raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
    cash = cash_after_entry_fee * (1.0 + raw_exit * BASE_NOTIONAL)
    cash -= cash_after_entry_fee * exit_fee * BASE_NOTIONAL
    return cash - 1.0, fill_i, route


def _build_for_year(label_dir: Path, out_dir: Path, year: int, *, exit_edge_min: float, cost_mult: float, max_samples: int) -> dict[str, object]:
    rows = _read_rows(label_dir / f"zigzag_action_labels_{year}.csv")
    fee_eff = FEE_RATE * cost_mult
    slip_eff = SLIP_RATE * cost_mult
    out_rows: list[dict[str, object]] = []
    edges: list[float] = []
    reason_counts: dict[str, int] = {}
    used_segments = 0
    skipped_segments = 0
    positive = 0
    segment_id = -1
    i = 0
    last_i = len(rows) - 2
    while i < last_i:
        action = int(float(rows[i]["zigzag_action"]))
        if action not in (1, 2):
            i += 1
            continue
        start_i = i
        while i < last_i and int(float(rows[i]["zigzag_action"])) == action:
            i += 1
        end_i = min(i - 1, last_i)
        segment_id += 1
        side = 1 if action == 1 else -1
        filled, entry_price, entry_fee, entry_fill_i, entry_route = _try_execution(rows, start_i, side, entry=True, fee_eff=fee_eff, slip_eff=slip_eff)
        entry_i = min(start_i + 1, len(rows) - 1)
        if not filled or end_i < entry_i:
            skipped_segments += 1
            continue
        path_idx = list(range(entry_i, end_i + 1))
        cash_after_entry_fee = 1.0 - entry_fee * BASE_NOTIONAL
        exit_net = []
        exit_fill_i = []
        exit_route = []
        for row_i in path_idx:
            net, fill_i, route = _exit_net(rows, row_i, side, entry_price, cash_after_entry_fee, fee_eff=fee_eff, slip_eff=slip_eff)
            exit_net.append(net)
            exit_fill_i.append(fill_i)
            exit_route.append(route)
        suffix_value = [0.0] * len(exit_net)
        suffix_pos = [0] * len(exit_net)
        best_value = exit_net[-1]
        best_pos = len(exit_net) - 1
        for k in range(len(exit_net) - 1, -1, -1):
            if exit_net[k] >= best_value:
                best_value = exit_net[k]
                best_pos = k
            suffix_value[k] = best_value
            suffix_pos[k] = best_pos
        mfe = 0.0
        mae = 0.0
        for k, row_i in enumerate(path_idx):
            px = _f(rows[row_i], "close")
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * BASE_NOTIONAL
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            if k == len(path_idx) - 1:
                best_future_pos = k
                best_future_net = exit_net[k]
                edge = 0.0
                label = 1
                reason = "segment_end_forced_exit"
            else:
                best_future_pos = suffix_pos[k + 1]
                best_future_net = suffix_value[k + 1]
                edge = exit_net[k] - best_future_net
                label = int(edge >= exit_edge_min)
                reason = "oracle_dp_exit_now" if label else "oracle_dp_hold"
            positive += label
            edges.append(edge)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            src = rows[row_i]
            out_rows.append(
                {
                    "timestamp": src["timestamp"],
                    "open": src["open"],
                    "high": src["high"],
                    "low": src["low"],
                    "close": src["close"],
                    "entry_zigzag_action": action,
                    "entry_zigzag_action_name": "LONG" if action == 1 else "SHORT",
                    "exit_action": label,
                    "exit_action_name": "EXIT" if label else "HOLD",
                    "exit_path_segment_id": segment_id,
                    "exit_path_side": side,
                    "exit_path_entry_signal_i": start_i,
                    "exit_path_entry_i": entry_i,
                    "exit_path_entry_fill_i": entry_fill_i,
                    "exit_path_end_i": end_i,
                    "exit_path_hold_bars": max(row_i - entry_i, 0),
                    "exit_path_entry_price": f"{entry_price:.12g}",
                    "exit_path_now_net": f"{exit_net[k]:.12g}",
                    "exit_path_best_future_net": f"{best_future_net:.12g}",
                    "exit_path_edge": f"{edge:.12g}",
                    "exit_path_best_future_i": path_idx[best_future_pos],
                    "exit_path_best_future_fill_i": exit_fill_i[best_future_pos],
                    "exit_path_mfe": f"{mfe:.12g}",
                    "exit_path_mae": f"{mae:.12g}",
                    "exit_path_unrealized": f"{unreal:.12g}",
                    "exit_path_reason": reason,
                    "entry_route": entry_route,
                    "best_future_exit_route": exit_route[best_future_pos],
                }
            )
            if max_samples > 0 and len(out_rows) >= max_samples:
                break
        used_segments += 1
        if max_samples > 0 and len(out_rows) >= max_samples:
            break
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"entry_label_path_optimal_exit_labels_{year}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    sorted_edges = sorted(edges)
    def q(p: float) -> float:
        if not sorted_edges:
            return 0.0
        return sorted_edges[min(int(len(sorted_edges) * p), len(sorted_edges) - 1)]
    return {
        "year": year,
        "path": str(out_path),
        "rows": len(out_rows),
        "positive_count": positive,
        "negative_count": len(out_rows) - positive,
        "positive_rate": positive / max(len(out_rows), 1),
        "exit_edge_mean": mean(edges) if edges else 0.0,
        "exit_edge_p50": q(0.50),
        "exit_edge_p90": q(0.90),
        "exit_edge_p99": q(0.99),
        "used_segments": used_segments,
        "skipped_segments": skipped_segments,
        "reason_counts": reason_counts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--years", default="2025,2026")
    ap.add_argument("--exit-edge-min", type=float, default=0.002)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()
    years = [int(x.strip()) for x in str(args.years).split(",") if x.strip()]
    reports = [_build_for_year(Path(args.label_dir), Path(args.out_dir), year, exit_edge_min=float(args.exit_edge_min), cost_mult=float(args.cost_mult), max_samples=int(args.max_samples)) for year in years]
    report = {
        "label_id": "omega4_entry_label_path_optimal_exit_labels_20260620",
        "source_entry_label_dir": str(args.label_dir),
        "label_contract": {
            "mode": "entry_label_path_optimal_stopping_every_in_position_bar",
            "oracle": "DP/suffix maximum realized net exit value within the same active entry-label segment",
            "exit_rule": "EXIT if exit_now_net - best_future_net >= exit_edge_min; final segment bar forced EXIT",
            "execution_contract": "entry next-open maker-limit must touch; exit next-open maker-limit else close fallback",
            "cash_rows": "excluded",
        },
        "risk_template": {
            "notional": BASE_NOTIONAL,
            "leverage": BASE_LEVERAGE,
            "take_profit": BASE_TAKE_PROFIT,
            "stop_loss": BASE_STOP_LOSS,
            "max_hold": 0,
            "cooldown": 0,
        },
        "fee": FEE_RATE,
        "slip": SLIP_RATE,
        "maker_fee_mult": MAKER_FEE_MULT,
        "cost_mult": float(args.cost_mult),
        "exit_edge_min": float(args.exit_edge_min),
        "reports": reports,
    }
    report_path = Path(args.out_dir) / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "reports": reports}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
