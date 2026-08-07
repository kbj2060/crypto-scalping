"""Kappa1 Stage 2 -- sign-constrained fusion layer, worst-quarter CMA-ES, quarterly LOWO.

Contract: docs/experiments/btc_kappa1_invariant_composite_20260807.json
Design:   docs/btc_kappa1_invariant_composite_policy_design_20260807.md

Nothing is retrained; inputs are the frozen primitive scores in the pinned dataset.
The only fitted object is an 8-parameter policy:
  [w_flow, w_evt, thr_in, tp_mult, sl_mult, size_cut1, size_cut2, anom_pen]
  - entry permission: event gate fired within the last 6 bars (30 min)
  - direction: contrarian to flow_mean_5m
  - score = w_flow*|flow_z| + w_evt*evt_raw_score, enter if score > thr_in
  - sizing: margin_fraction {0.10,0.20,0.30} by score margin, shrunk by GMM vol rank
  - exit: ATR TP/SL + trailing (0.5*SL @ 0.3*TP, the KEEP-ALIVE lever) + 288-bar time exit
  - execution: maker post-only per Stage-0 v2 (limit at next open, strict trade-through,
    cancel after 3 bars); cost = 2bps maker entry + 5bps taker exit + 1bp adverse = 8bps
Objective per CMA-ES run: maximize the MINIMUM training-quarter net PnL (activity penalty
below 15 trades/quarter). Selection: leave-one-quarter-out, held-out quarter never touched
by that fold's optimizer. Pass per contract: >=5/6 held-out quarters positive per seed,
sign agreement across seeds, >=15 trades per held-out quarter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import cma

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/splits/year_oos/btc_kappa1_primitives_5m_20260807.parquet"
PREFLIGHT = ROOT / "docs/experiments/btc_kappa1_invariant_composite_20260807.preflight.json"
OUT = ROOT / "docs/experiments/btc_kappa1_stage2_lowo_results_20260807.json"

SEEDS = [914237, 60481, 7754321, 283009, 51418]
PERMISSION_BARS = 6
CANCEL_BARS = 3
TIME_EXIT_BARS = 288
COST = 0.0008
LEVERAGE = 3
MIN_TRADES_Q = 15
BOUNDS_LO = np.array([0.0, 0.0, 0.0, 1.0, 0.5, 0.05, 0.05, 0.0])
BOUNDS_HI = np.array([1.0, 1.0, 2.0, 6.0, 3.0, 2.0, 2.0, 1.0])
CMA_ITERS = 60


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare():
    frame = pd.read_parquet(DATASET)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    flow = frame["flow_mean_5m"]
    flow_std = flow.rolling(2016, min_periods=288).std()
    frame["flow_z"] = flow / flow_std.replace(0, np.nan)
    frame["gmm_rank_norm"] = frame["gmm_cluster_rank"] / frame["gmm_cluster_rank"].max()
    fired = frame["evt_gate_fired"].to_numpy(dtype=bool)
    permission = np.zeros(len(frame), dtype=bool)
    for k in range(PERMISSION_BARS):
        permission[k:] |= fired[:len(frame) - k]
    frame["permission"] = permission
    frame["quarter"] = frame["timestamp"].dt.to_period("Q").astype(str)
    quarters = [q for q, cnt in frame["quarter"].value_counts().items() if cnt > 20000]
    quarters = sorted(quarters)
    return frame, quarters


class Backtester:
    def __init__(self, frame: pd.DataFrame):
        self.open = frame["open"].to_numpy()
        self.high = frame["high"].to_numpy()
        self.low = frame["low"].to_numpy()
        self.close = frame["close"].to_numpy()
        self.atr = frame["atr_pct_96"].to_numpy()
        self.flow = frame["flow_mean_5m"].to_numpy()
        self.flow_z = frame["flow_z"].to_numpy()
        self.evt = frame["raw_evt"].to_numpy() if "raw_evt" in frame else frame["evt_raw_score"].to_numpy()
        self.gmm = frame["gmm_rank_norm"].to_numpy()
        self.quarter = frame["quarter"].to_numpy()
        self.n = len(frame)
        self.candidates = np.flatnonzero(frame["permission"].to_numpy()
                                         & np.isfinite(self.flow_z) & np.isfinite(self.atr))

    def run(self, params: np.ndarray, decision_quarters: set[str] | None) -> dict:
        w_flow, w_evt, thr, tp_mult, sl_mult, cut1, cut2, anom_pen = params
        trades_pnl, trades_quarter = [], []
        busy_until = -1
        for i in self.candidates:
            if i <= busy_until or i + 1 >= self.n:
                continue
            if decision_quarters is not None and self.quarter[i] not in decision_quarters:
                continue
            score = w_flow * abs(self.flow_z[i]) + w_evt * self.evt[i]
            if not np.isfinite(score) or score <= thr:
                continue
            side = -1 if self.flow[i] > 0 else 1
            limit = self.open[i + 1]
            fill = -1
            for j in range(i + 1, min(i + 1 + CANCEL_BARS, self.n)):
                through = self.low[j] < limit if side == 1 else self.high[j] > limit
                if through:
                    fill = j
                    break
            if fill < 0:
                busy_until = min(i + CANCEL_BARS, self.n - 1)
                continue
            atr = self.atr[i]
            tp_move, sl_move = tp_mult * atr, sl_mult * atr
            margin = 0.10 if score - thr < cut1 else (0.20 if score - thr < cut1 + cut2 else 0.30)
            margin *= max(0.0, 1.0 - anom_pen * self.gmm[i])
            if side == 1:
                tp_price, sl_price = limit * (1 + tp_move), limit * (1 - sl_move)
                trail_trigger = limit * (1 + 0.3 * tp_move)
            else:
                tp_price, sl_price = limit * (1 - tp_move), limit * (1 + sl_move)
                trail_trigger = limit * (1 - 0.3 * tp_move)
            stop = sl_price
            trail_active = False
            end = min(fill + TIME_EXIT_BARS, self.n - 1)
            exit_ret = None
            for j in range(fill + 1, end + 1):
                if side == 1:
                    if self.low[j] <= stop:
                        exit_ret = stop / limit - 1
                        break
                    if self.high[j] >= tp_price:
                        exit_ret = tp_move
                        break
                    if not trail_active and self.close[j] >= trail_trigger:
                        trail_active = True
                    if trail_active:
                        stop = max(stop, self.close[j] - 0.5 * sl_move * limit)
                else:
                    if self.high[j] >= stop:
                        exit_ret = 1 - stop / limit
                        break
                    if self.low[j] <= tp_price:
                        exit_ret = tp_move
                        break
                    if not trail_active and self.close[j] <= trail_trigger:
                        trail_active = True
                    if trail_active:
                        stop = min(stop, self.close[j] + 0.5 * sl_move * limit)
                exit_bar = j
            if exit_ret is None:
                exit_ret = side * (self.close[end] / limit - 1)
                exit_bar = end
            else:
                exit_bar = j
            trades_pnl.append((exit_ret - COST) * margin * LEVERAGE)
            trades_quarter.append(self.quarter[i])
            busy_until = exit_bar
        if not trades_pnl:
            return {"per_quarter": {}, "total": 0.0, "trades": 0, "win_rate": float("nan"), "mdd": 0.0}
        arr = np.array(trades_pnl)
        qs = np.array(trades_quarter)
        per_quarter = {q: {"pnl": float(arr[qs == q].sum()), "trades": int((qs == q).sum())}
                       for q in np.unique(qs)}
        equity = np.cumsum(arr)
        peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
        return {"per_quarter": per_quarter, "total": float(arr.sum()), "trades": int(len(arr)),
                "win_rate": float((arr > 0).mean()), "mdd": float(np.min(equity - peak))}


def objective(bt: Backtester, params: np.ndarray, train_quarters: list[str]) -> float:
    stats = bt.run(params, set(train_quarters))
    worst = min((stats["per_quarter"].get(q, {"pnl": 0.0})["pnl"] for q in train_quarters))
    for q in train_quarters:
        n_trades = stats["per_quarter"].get(q, {"trades": 0})["trades"]
        if n_trades < MIN_TRADES_Q:
            worst -= 0.02 * (MIN_TRADES_Q - n_trades) / MIN_TRADES_Q
    return -worst  # CMA minimizes


def optimize(bt: Backtester, train_quarters: list[str], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.2, 0.8, size=8)
    es = cma.CMAEvolutionStrategy(x0, 0.25, {
        "bounds": [[0.0] * 8, [1.0] * 8], "seed": seed, "maxiter": CMA_ITERS, "verbose": -9})
    def denorm(u): return BOUNDS_LO + np.asarray(u) * (BOUNDS_HI - BOUNDS_LO)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [objective(bt, denorm(u), train_quarters) for u in solutions])
    return denorm(es.result.xbest)


def main() -> None:
    preflight = json.loads(PREFLIGHT.read_text())
    assert sha256_file(DATASET) == preflight["dataset"]["sha256"], "dataset drifted; rerun preflight"
    frame, quarters = prepare()
    bt = Backtester(frame)
    print(f"quarters: {quarters}; candidate bars: {len(bt.candidates)}")

    folds = {}
    for held_out in quarters:
        train_quarters = [q for q in quarters if q != held_out]
        fold = {}
        for seed in SEEDS:
            best = optimize(bt, train_quarters, seed)
            held_stats = bt.run(best, {held_out})
            fold[str(seed)] = {"params": [round(float(v), 4) for v in best],
                               "held_out": held_stats}
            print(f"fold {held_out} seed {seed}: held-out pnl={held_stats['total']:+.4f} "
                  f"trades={held_stats['trades']}")
        folds[held_out] = fold

    per_seed_positive = {str(s): sum(1 for q in quarters if folds[q][str(s)]["held_out"]["total"] > 0)
                         for s in SEEDS}
    sign_agreement = {q: len({folds[q][str(s)]["held_out"]["total"] > 0 for s in SEEDS}) == 1
                      for q in quarters}
    min_trades_ok = {q: min(folds[q][str(s)]["held_out"]["trades"] for s in SEEDS) >= MIN_TRADES_Q
                     for q in quarters}
    need = int(np.ceil(len(quarters) * 4 / 5))
    gate_pass = (all(v >= need for v in per_seed_positive.values())
                 and all(sign_agreement.values()) and all(min_trades_ok.values()))
    report = {
        "contract": "docs/experiments/btc_kappa1_invariant_composite_20260807.json",
        "dataset_sha256": preflight["dataset"]["sha256"],
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "quarters": quarters, "seeds": SEEDS, "cma_iters": CMA_ITERS,
        "cost_model": "maker2_takerexit5_adverse1_bps_conservative",
        "folds": folds,
        "per_seed_positive_heldout_quarters": per_seed_positive,
        "required_positive_heldout_quarters": need,
        "sign_agreement_per_fold": sign_agreement,
        "min_trades_ok_per_fold": min_trades_ok,
        "gate_pass": gate_pass,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nper-seed positive held-out quarters (need >={need}/{len(quarters)}): {per_seed_positive}")
    print(f"sign agreement: {sign_agreement}")
    print(f"min trades ok: {min_trades_ok}")
    print(f"STAGE 2 GATE: {'PASS' if gate_pass else 'FAIL'}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
