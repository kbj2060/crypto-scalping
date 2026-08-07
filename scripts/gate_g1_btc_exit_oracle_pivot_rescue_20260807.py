"""Gate G1 -- is there a ceiling worth chasing for a zigzag-pivot-based EXIT layer?

The best triple-barrier transformer's OOS fresh-entry ledger is 153 TP / 277 SL / 4 timeout. Time
exits are negligible, so the only place an exit policy can add value is by cutting some of the SL
losers before they reach their stop. Per the oracle-label-selection protocol, measure the CEILING
under perfect foresight BEFORE training any exit model:

- rescuable_rate: of the SL losers, how many were ever in profit by more than the roundtrip cost
  (price MFE > 10bps) before the stop fired? That is the population an exit model could work on.
- P1 perfect_mfe: exit at each trade's max favorable excursion bar. Absolute (unreachable) upper
  bound on any exit policy.
- P2 pivot_exit: exit at the first ORACLE zigzag pivot against the position (H pivot for a LONG,
  L pivot for a SHORT), TP/SL still active. This is the ceiling for a perfect pivot DETECTOR.
- P3 pivot_lead_k: exit at the first bar from which an adverse pivot occurs within the next k bars
  (k=12, 24). This is the ceiling for the actually-proposed Layer 2 head -- a discrete-time pivot
  HAZARD predictor -- which fires k bars ahead of the pivot rather than on it.

Two simulation modes are reported for each policy:
- fixed_entries: the baseline ledger's exact decision bars, so the only thing that changes is the
  exit rule (clean attribution).
- resimulated: the full fresh-entry decision stream re-run under the policy. Earlier exits free the
  single position slot sooner, so this admits MORE trades and pays MORE cost -- the honest version
  of what the policy would actually produce.

All mechanics (entry at next bar's open, non-overlapping single position, SL checked before TP
within a bar, notional = margin_fraction * leverage, cost charged once per trade) mirror
core/causal_futures_backtest.simulate_single_position; the policy exit is evaluated at bar CLOSE
after the intrabar TP/SL check, since a bar-close signal cannot pre-empt an intrabar barrier touch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402
from ensemble.deep_features.btc_deepfeat_encoders_20260806 import build_model  # noqa: E402

CHECKPOINT = ROOT / "tmp/btc_deepfeat_tripbarrier_20260806/flatsmooth_cw_0.9/deepfeat_bundle.pt"
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet"
PIVOT_PATH = ROOT / "data/splits/year_oos/btc_5m_pivot_transition_labels_20260806.parquet"
OUT_DIR = ROOT / "tmp/btc_gate_g1_exit_oracle_20260807"

CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS = 12, 288, 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
NOTIONAL = MARGIN_FRACTION * LEVERAGE
ACCOUNT_COST = ROUNDTRIP_COST_RATE * NOTIONAL
BREAKEVEN_PRICE_MOVE = ROUNDTRIP_COST_RATE  # price move that exactly pays the roundtrip cost
# P2 (exit exactly at the pivot) is not realisable by a predictor -- a pivot is only knowable after
# the reversal confirms it -- so the lead sweep maps how much timing slack the ceiling tolerates.
LEAD_BARS = (1, 2, 3, 6, 12, 24)
# Model-free, fully causal alternatives: 68-76% of SL losers were in profit past the roundtrip cost
# at some point, so a path-dependent stop may capture part of P2's gain with no exit model at all.
STOP_POLICIES = {
    "S1_breakeven@0.3TP": {"mode": "breakeven", "activate": 0.3},
    "S2_breakeven@0.5TP": {"mode": "breakeven", "activate": 0.5},
    "S3_trail0.5SL@0.3TP": {"mode": "trailing", "activate": 0.3, "trail": 0.5},
    "S4_trail1.0SL@0.5TP": {"mode": "trailing", "activate": 0.5, "trail": 1.0},
}


@torch.no_grad()
def _predict(model, ds, split, device, batch_size=1024):
    model.eval()
    row_idx = ds.end_idx[split]
    out = []
    for i in range(0, len(row_idx), batch_size):
        x = torch.from_numpy(ds.get_batch(row_idx[i : i + batch_size])).to(device)
        logits, _, _ = model(x)
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _fresh_entry_mask(side_state: np.ndarray) -> np.ndarray:
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh


def _simulate(decision_idx, sides, tp_moves, sl_moves, open_, high, low, close, forced_exit,
              stop_policy=None, profit_gated=False):
    """Non-overlapping single-position simulation. ``forced_exit`` is a per-side boolean matrix of
    shape (2, n_bars) -- row 0 for LONG, row 1 for SHORT -- marking bars whose CLOSE triggers a
    policy exit. Pass an all-False matrix to reproduce plain TP/SL/horizon behaviour.

    ``stop_policy`` optionally adds a path-dependent stop that needs no model at all:
    ``{"mode": "breakeven", "activate": a}`` moves the stop to the entry price once favorable
    excursion has reached ``a * tp_move``; ``{"mode": "trailing", "activate": a, "trail": d}``
    additionally trails the stop ``d * sl_move`` behind the favorable extreme. The stop level for
    bar ``j`` is derived from the favorable extreme through bar ``j-1`` only -- using bar j's own
    high/low to set the stop that bar j is then tested against would be intrabar lookahead."""
    n = len(close)
    occupied_through = -1
    cash = 1.0
    rows = []
    equity_marks = []
    for d, side, tp_move, sl_move in zip(decision_idx, sides, tp_moves, sl_moves):
        if not (np.isfinite(tp_move) and np.isfinite(sl_move)):
            continue
        entry_i = int(d) + 1
        if entry_i >= n or entry_i <= occupied_through:
            continue
        final_i = min(entry_i + HORIZON_BARS - 1, n - 1)
        if final_i < entry_i:
            continue
        entry = open_[entry_i]
        frow = forced_exit[0] if side > 0 else forced_exit[1]
        if side > 0:
            tp_level, sl_level = entry * (1.0 + tp_move), entry * (1.0 - sl_move)
        else:
            tp_level, sl_level = entry * (1.0 - tp_move), entry * (1.0 + sl_move)

        price_move, reason, exit_i = None, None, final_i
        best_fav = -np.inf  # favorable excursion through the PREVIOUS bar only
        for j in range(entry_i, final_i + 1):
            stop_move = sl_move  # loss (positive number) realised if this bar's stop is hit
            if stop_policy is not None and np.isfinite(best_fav) and best_fav >= stop_policy["activate"] * tp_move:
                if stop_policy["mode"] == "breakeven":
                    stop_move = 0.0
                else:
                    stop_move = min(sl_move, -(best_fav - stop_policy["trail"] * sl_move))
                    stop_move = min(stop_move, sl_move)
            stop_level = entry * (1.0 - stop_move) if side > 0 else entry * (1.0 + stop_move)
            if side > 0:
                if low[j] <= stop_level:
                    price_move, reason, exit_i = -stop_move, "sl" if stop_move == sl_move else "stop_moved", j
                    break
                if high[j] >= tp_level:
                    price_move, reason, exit_i = tp_move, "tp", j
                    break
            else:
                if high[j] >= stop_level:
                    price_move, reason, exit_i = -stop_move, "sl" if stop_move == sl_move else "stop_moved", j
                    break
                if low[j] <= tp_level:
                    price_move, reason, exit_i = tp_move, "tp", j
                    break
            best_fav = max(best_fav, (high[j] / entry - 1.0) if side > 0 else (1.0 - low[j] / entry))
            if frow[j]:
                move = (close[j] / entry - 1.0) if side > 0 else (1.0 - close[j] / entry)
                if not profit_gated or move > BREAKEVEN_PRICE_MOVE:
                    price_move, reason, exit_i = move, "policy", j
                    break
        if price_move is None:
            price_move = (close[final_i] / entry - 1.0) if side > 0 else (1.0 - close[final_i] / entry)
            reason, exit_i = "timeout", final_i

        # MFE over the bars actually held (recomputed for the realised holding window)
        held_high, held_low = high[entry_i : exit_i + 1], low[entry_i : exit_i + 1]
        mfe_held = float((held_high.max() / entry - 1.0) if side > 0 else (1.0 - held_low.min() / entry))

        trade_return = float(price_move * NOTIONAL - ACCOUNT_COST)
        cash *= 1.0 + trade_return
        occupied_through = exit_i
        equity_marks.append(cash)
        rows.append({
            "decision_i": int(d), "entry_i": entry_i, "exit_i": exit_i, "side": int(side),
            "reason": reason, "bars_held": int(exit_i - entry_i + 1),
            "price_move": float(price_move), "trade_return": trade_return, "mfe": mfe_held,
            "tp_move": float(tp_move), "sl_move": float(sl_move),
        })
    return pd.DataFrame(rows), np.array(equity_marks, dtype=np.float64)


def _summarize(ledger, equity_marks, name, split, mode) -> dict:
    if len(ledger) == 0:
        return {"policy": name, "split": split, "mode": mode, "n_trades": 0}
    rets = ledger["trade_return"].to_numpy(dtype=np.float64)
    equity = np.concatenate([[1.0], equity_marks])
    running_max = np.maximum.accumulate(equity)
    std_bps = float(rets.std(ddof=1) * 10000.0) if len(rets) > 1 else float("nan")
    gross_bps = float((rets.mean() + ACCOUNT_COST) * 10000.0)
    return {
        "policy": name, "split": split, "mode": mode, "n_trades": int(len(ledger)),
        "win_rate": float((rets > 0).mean()),
        "mean_ret_bps": float(rets.mean() * 10000.0),
        "gross_mean_ret_bps": gross_bps,
        "std_ret_bps": std_bps,
        "t_stat_gross": float(gross_bps / (std_bps / np.sqrt(len(rets)))) if len(rets) > 1 else None,
        "sum_ret_pct": float(rets.sum() * 100.0),
        "final_equity": float(equity[-1]),
        "trade_mdd_pct": float(((equity - running_max) / running_max).min() * 100.0),
        "median_bars_held": float(ledger["bars_held"].median()),
        "exit_reasons": ledger["reason"].value_counts().to_dict(),
    }


def _rescue_stats(ledger) -> dict:
    losers = ledger[ledger["reason"] == "sl"]
    if len(losers) == 0:
        return {"n_sl": 0}
    mfe = losers["mfe"].to_numpy(dtype=np.float64)
    return {
        "n_sl": int(len(losers)),
        "sl_share_of_trades": float(len(losers) / len(ledger)),
        "mfe_median_bps": float(np.median(mfe) * 10000.0),
        "mfe_p75_bps": float(np.percentile(mfe, 75) * 10000.0),
        "mfe_p90_bps": float(np.percentile(mfe, 90) * 10000.0),
        "rescuable_rate_breakeven": float((mfe > BREAKEVEN_PRICE_MOVE).mean()),
        "rescuable_rate_half_tp": float((mfe > 0.5 * losers["tp_move"].to_numpy(dtype=np.float64)).mean()),
        "median_bars_held_sl": float(losers["bars_held"].median()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = bundle["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = build_dataset(
        window=config["window"], label_path=LABEL_PATH, hard_col="trade_outcome_action",
        soft_cols=["trade_outcome_soft_cash", "trade_outcome_soft_long", "trade_outcome_soft_short"],
    )
    model = build_model(
        config["arch"], config["n_features"], config["category_sizes"], embed_dim=config["embed_dim"],
        d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"],
        ffn_mult=config["ffn_mult"], dropout=config["dropout"], quality_head=config["quality_head"],
        head_type=config.get("head_type", "linear"),
    ).to(device)
    model.load_state_dict(bundle["model_state"])

    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    open_ = panel["open"].to_numpy(dtype=np.float64)
    high = panel["high"].to_numpy(dtype=np.float64)
    low = panel["low"].to_numpy(dtype=np.float64)
    close = panel["close"].to_numpy(dtype=np.float64)
    n = len(panel)

    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_all, sl_all = TP_MULT * vol, SL_MULT * vol

    piv = pd.read_parquet(PIVOT_PATH, columns=["timestamp", "is_pivot", "pivot_type"])
    piv = piv.sort_values("timestamp").reset_index(drop=True)
    if not (piv["timestamp"].to_numpy() == panel["timestamp"].to_numpy()).all():
        raise RuntimeError("pivot label timestamps don't match the panel")
    is_high_pivot = ((piv["is_pivot"] == 1) & (piv["pivot_type"] == "H")).to_numpy()
    is_low_pivot = ((piv["is_pivot"] == 1) & (piv["pivot_type"] == "L")).to_numpy()
    # adverse pivot: a HIGH pivot ends an up-wave (bad for a LONG); a LOW pivot ends a down-wave.
    adverse = np.stack([is_high_pivot, is_low_pivot])  # row 0 = LONG, row 1 = SHORT

    no_exit = np.zeros((2, n), dtype=bool)
    policies: dict[str, np.ndarray] = {"P0_baseline": no_exit, "P2_pivot_exit": adverse}
    for k in LEAD_BARS:
        lead = np.zeros((2, n), dtype=bool)
        for r in (0, 1):
            hits = np.flatnonzero(adverse[r])
            for h in hits:
                lead[r, max(0, h - k) : h + 1] = True
        policies[f"P3_pivot_lead{k}"] = lead

    results, rescue, mfe_ceiling = [], [], []
    for split in ("val", "oos"):
        probs = _predict(model, ds, split, device)
        pred_hard = probs.argmax(axis=1)
        side_state = np.where(pred_hard == 1, 1, np.where(pred_hard == 2, -1, 0))
        row_idx = ds.end_idx[split]
        fresh = _fresh_entry_mask(side_state)
        d_all, s_all = row_idx[fresh], side_state[fresh]

        base_ledger, base_eq = _simulate(d_all, s_all, tp_all[d_all], sl_all[d_all], open_, high, low, close, no_exit)
        results.append(_summarize(base_ledger, base_eq, "P0_baseline", split, "resimulated"))
        rescue.append({"split": split, **_rescue_stats(base_ledger)})

        # P1: perfect-MFE exit, evaluated on the baseline's own trades (upper bound; not a policy
        # that could be re-simulated, since it needs the whole realised path of each trade).
        pm = base_ledger["mfe"].to_numpy(dtype=np.float64)
        rets_p1 = pm * NOTIONAL - ACCOUNT_COST
        mfe_ceiling.append({
            "split": split, "n_trades": int(len(base_ledger)),
            "sum_ret_pct": float(rets_p1.sum() * 100.0),
            "mean_ret_bps": float(rets_p1.mean() * 10000.0),
            "win_rate": float((rets_p1 > 0).mean()),
        })

        base_d = base_ledger["decision_i"].to_numpy(dtype=np.int64)
        base_s = base_ledger["side"].to_numpy(dtype=np.int64)
        for name, forced in policies.items():
            if name == "P0_baseline":
                continue
            led, eq = _simulate(base_d, base_s, tp_all[base_d], sl_all[base_d], open_, high, low, close, forced)
            results.append(_summarize(led, eq, name, split, "fixed_entries"))
            led, eq = _simulate(d_all, s_all, tp_all[d_all], sl_all[d_all], open_, high, low, close, forced)
            results.append(_summarize(led, eq, name, split, "resimulated"))

        # P4: same hazard signal as P3, but only allowed to close a position that is already in
        # profit past the roundtrip cost. A pure P3 exit cuts losers early at a loss, which is what
        # makes it so sensitive to lead time; gating on profit should trade some of P2's upside for
        # tolerance to an imprecise pivot forecast -- the only version a real hazard head could run.
        for k in LEAD_BARS:
            forced = policies[f"P3_pivot_lead{k}"]
            led, eq = _simulate(base_d, base_s, tp_all[base_d], sl_all[base_d], open_, high, low, close,
                                forced, profit_gated=True)
            results.append(_summarize(led, eq, f"P4_profitgated_lead{k}", split, "fixed_entries"))
            led, eq = _simulate(d_all, s_all, tp_all[d_all], sl_all[d_all], open_, high, low, close,
                                forced, profit_gated=True)
            results.append(_summarize(led, eq, f"P4_profitgated_lead{k}", split, "resimulated"))

        for name, spec in STOP_POLICIES.items():
            led, eq = _simulate(base_d, base_s, tp_all[base_d], sl_all[base_d], open_, high, low, close, no_exit, spec)
            results.append(_summarize(led, eq, name, split, "fixed_entries"))
            led, eq = _simulate(d_all, s_all, tp_all[d_all], sl_all[d_all], open_, high, low, close, no_exit, spec)
            results.append(_summarize(led, eq, name, split, "resimulated"))

    print("=== SL-loser rescue potential (baseline ledger) ===")
    for r in rescue:
        print(json.dumps(r))
    print("\n=== P1 perfect-MFE exit ceiling (unreachable upper bound, baseline trades) ===")
    for r in mfe_ceiling:
        print(json.dumps(r))

    header = (f"{'policy':<18}{'split':<6}{'mode':<14}{'trades':>7}{'win%':>7}{'grossbps':>10}"
              f"{'t':>7}{'sum%':>9}{'equity':>8}{'mdd%':>8}{'held':>6}")
    print("\n=== exit-policy comparison ===")
    print(header)
    print("-" * len(header))
    for r in results:
        if not r.get("n_trades"):
            continue
        t = f"{r['t_stat_gross']:.2f}" if r.get("t_stat_gross") is not None else "n/a"
        print(f"{r['policy']:<18}{r['split']:<6}{r['mode']:<14}{r['n_trades']:>7}{r['win_rate']*100:>7.1f}"
              f"{r['gross_mean_ret_bps']:>10.2f}{t:>7}{r['sum_ret_pct']:>9.1f}{r['final_equity']:>8.3f}"
              f"{r['trade_mdd_pct']:>8.1f}{r['median_bars_held']:>6.0f}")

    payload = {
        "config": {
            "checkpoint": str(CHECKPOINT), "tp_mult": TP_MULT, "sl_mult": SL_MULT,
            "horizon_bars": HORIZON_BARS, "notional": NOTIONAL,
            "account_cost_bps": ACCOUNT_COST * 10000.0, "lead_bars": list(LEAD_BARS),
        },
        "sl_rescue_potential": rescue, "perfect_mfe_ceiling": mfe_ceiling, "policies": results,
    }
    (OUT_DIR / "g1_exit_oracle_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_DIR}/g1_exit_oracle_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
