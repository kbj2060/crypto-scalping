#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 priority #5 (last item in the priority queue), VAL-side evaluation.
Compares the TCN h48qual exit_head trained by
scripts/train_eval_omega461_tcn_exit_head_liveatr_20260813.py against the CURRENT Odyssey2 baseline
-- h48qual's TabM live-ATR-relabel exit head (tmp/causal_regen_20260516/
eth_omega461_exit_head_liveatr_relabel_20260813_full1500/h48qual/true_3head_tabm_bundle.pt, the same
baseline priority #4/GBDT compared against) -- at both the component level (h48qual standalone
ledger) and the portfolio level (h48qual+zig075 single-account greedy router). zig075 is not
touched.

=== Why this script needs MORE than priority #4 (GBDT)'s duck-typing trick ===
GBDT (research_eth_omega461_gbdt_exit_head_val_20260813.py) could inject its model directly at
train_eval_omega4_2_risk_sidecar_20260622._predict_exit_prob_one's existing call sites because that
function only ever hands the model ONE feature row (`row = base_np[row_i]`) -- a duck-typed
`model(x)` returning `{"exit": logits}` for a single-row `x` was a perfect drop-in replacement for
ThreeHeadTabM there, and `replay_exit_variant` / `greedy_replay` needed ZERO changes.

A TCN structurally cannot use that same call site: it needs a WINDOW of history per decision, but
`_predict_exit_prob_one` (despite already holding `base_np`, the full per-bar feature matrix, and
`row_i`, the current index, as arguments) only ever slices a single row before calling the model --
the window information the function technically has access to never reaches the model call. Per the
coordinator's explicit instruction, the fix is NOT to edit that function (or its two callers) but to
make renamed, logic-preserving COPIES that slice a window instead of a single row at the exit-head
call site only:
  - _predict_exit_prob_one_windowed   (copy of train_eval_omega4_2_risk_sidecar_20260622._predict_exit_prob_one)
  - replay_exit_variant_windowed      (copy of research_eth_omega461_exit_sweep_20260721.replay_exit_variant)
  - greedy_replay_windowed            (copy of replay_omega4_6_1_greedy_router_20260706.greedy_replay)
The portfolio-level copy is required too (not just the component-level one) because greedy_replay
ALSO calls _predict_exit_prob_one directly, at its own separate call site -- same structural
limitation. Unlike the component-level copy (which only ever runs ONE model, TCN or TabM, per call),
greedy_replay_windowed must handle a MIXED portfolio (h48qual on TCN, zig075 still on its original
TabM bundle) within the SAME per-bar loop, so it dispatches per-expert-model at call time via
`getattr(model, "IS_WINDOWED", False)` rather than assuming every component is windowed -- TCNExitHeadWrapper
sets this flag, real ThreeHeadTabM instances (and GBDTExitHeadWrapper) do not have it, so zig075's
real TabM model is routed to the ORIGINAL (unwindowed, single-row) _predict_exit_prob_one exactly as
before. Feeding a real ThreeHeadTabM a 3D windowed tensor would shape-mismatch-crash immediately
(loud failure, not silent corruption) if this dispatch were missing or wrong.
research_eth_omega461_exit_sweep_20260721.py, train_eval_omega4_2_risk_sidecar_20260622.py, and
replay_omega4_6_1_greedy_router_20260706.py are none of them touched by this script -- verified via
`git diff` before and after this experiment (see the companion experiment doc's "준수 확인" section).

=== TCNExitHeadWrapper -- where standardization + branch-splitting actually happens ===
_predict_exit_prob_one_windowed hands the wrapper a RAW (unstandardized) window
`(T=window, n_base + len(POS_COLS))` slice of base_np (mean=0/std=1 identity scalers passed from the
harness side, same convention priority #4's GBDTExitHeadWrapper established -- the wrapper owns its
own standardization instead). TCNExitHeadWrapper.__call__ splits that into the market-feature
sequence (first n_base columns, all T rows -> standardized with the TCN bundle's own market_scaler,
transposed to (batch, C, T) for Conv1d) and the position-state scalar vector (last len(POS_COLS)
columns of the LAST row only, i.e. the current bar -- prior rows in the window never had this
trade's real position state computed for them, see the training script's module docstring --
standardized with the bundle's own pos_scaler), runs TCNExitClassifier(seq, pos), and reshapes the
2-class logits to (batch, k=1, 2) so the surrounding TabM-shaped softmax/ensemble-pooling machinery
(designed for TabM's k=8 internal ensemble) reproduces a plain single-model probability unchanged --
same "fake a k=1 ensemble" trick GBDTExitHeadWrapper used, just with real logits instead of
log(predict_proba).

=== G0 self-check (runs first, unconditionally) ===
Identical in spirit and code to priority #4's G0: re-derives the two ALREADY-PUBLISHED reference
numbers (component-level via h48cons._evaluate_val, portfolio-level via
research_eth_omega461_exit_head_portfolio_asymmetric_20260813.run_variant, both 100% pre-existing,
unmodified code) through this exact script's import chain, and asserts they match the published VAL
numbers (reused from research_eth_omega461_gbdt_exit_head_val_20260813.G0_REFERENCE /
_close_to_reference, imported unchanged rather than re-typed). If G0 fails this script aborts BEFORE
touching any TCN number, per this project's methodology discipline.

=== Promotion gate ===
TCN must be non-worse than the TabM live-ATR baseline on BOTH PnL and MDD at BOTH the component and
portfolio level on VAL before OOS is allowed to run (see
scripts/research_eth_omega461_tcn_exit_head_oos_20260813.py, which reads this script's report.json
and refuses to proceed if gate_pass is False -- same code-enforced pattern as priority #4's OOS
script, not researcher discretion).

fresh_forward_bar_by_bar=true (replay_exit_variant_windowed and greedy_replay_windowed are both
single causal forward passes, i increasing, only bar i and already-closed history used at bar i --
the ADDED window lookback is still strictly historical/already-closed, never a future bar).
trade_ledgers_used_as_input=false (ledgers are written-only outputs). saved_parent_exit_timestamps_
used=false. future_rows_used_for_entry=false. direction_head/quality_head/encoder are frozen and
unchanged (bit-identical across the original/TabM-liveATR/GBDT/TCN h48qual variants -- only
exit_head differs). VAL window 2025-10-01..2025-12-31
(research_eth_omega461_exit_sweep_20260721.VAL_START/VAL_END). OOS is never loaded here.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT touch zig075.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_head_h48cons_relabel_20260813 as h48cons  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_gbdt_exit_head_val_20260813 as gbdt_val  # noqa: E402
import train_eval_omega461_tcn_exit_head_liveatr_20260813 as tcn_train  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_tcn_exit_head_val_20260813"
TCN_BUNDLE = tcn_train.OUT_DIR / "h48qual" / "tcn_exit_bundle.pt"

# Reused unchanged from priority #4's val script -- same published-VAL-numbers reference this
# project's G0 self-checks are always graded against (docs/experiments/
# eth_omega461_live_exit_head_liveatr_relabel_20260813.md).
G0_REFERENCE = gbdt_val.G0_REFERENCE
G0_TOLERANCE_PP = gbdt_val.G0_TOLERANCE_PP
_close_to_reference = gbdt_val._close_to_reference


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


# ---------------------------------------------------------------------------
# Windowed duck-typing layer -- see module docstring for why this is necessary (GBDT's single-row
# duck-typing trick does not extend to a model that needs sequence context).
# ---------------------------------------------------------------------------


class TCNExitHeadWrapper:
    """Duck-types train_eval_omega1_2_tabm_3head_20260603.ThreeHeadTabM's __call__ contract for
    _predict_exit_prob_one_windowed, the same way research_eth_omega461_gbdt_exit_head_val_
    20260813.GBDTExitHeadWrapper does for _predict_exit_prob_one -- EXCEPT the input it receives is
    a raw (T=window, n_base+len(POS_COLS)) window, not a single row, because that is what a TCN
    structurally needs. Owns its own standardization (market_scaler for the sequence branch,
    pos_scaler for the current-bar position-state branch) since the harness passes identity
    mean=0/std=1 (same convention GBDTExitHeadWrapper established: the wrapper, not the shared
    harness, knows what scaling its model needs). IS_WINDOWED=True is the marker
    greedy_replay_windowed's per-bar dispatch checks to route h48qual (this wrapper) to the windowed
    prediction path while routing zig075 (a real, un-wrapped ThreeHeadTabM instance, no such
    attribute) to the original single-row path unchanged."""

    IS_WINDOWED = True

    def __init__(
        self, model: "tcn_train.TCNExitClassifier", device: torch.device, n_base: int,
        market_scaler: dict[str, np.ndarray], pos_scaler: dict[str, np.ndarray],
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.n_base = int(n_base)
        self.market_mean = torch.as_tensor(market_scaler["mean"], dtype=torch.float32, device=device)
        self.market_std = torch.as_tensor(market_scaler["std"], dtype=torch.float32, device=device)
        self.pos_mean = torch.as_tensor(pos_scaler["mean"], dtype=torch.float32, device=device)
        self.pos_std = torch.as_tensor(pos_scaler["std"], dtype=torch.float32, device=device)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (batch, T, n_base + len(POS_COLS)) raw features (harness-side identity mean/std).
        market_raw = x[:, :, : self.n_base]
        pos_raw = x[:, -1, self.n_base :]  # current bar (last window row) position state only
        market_std = torch.clamp((market_raw - self.market_mean) / self.market_std, -10.0, 10.0)
        pos_std = torch.clamp((pos_raw - self.pos_mean) / self.pos_std, -10.0, 10.0)
        seq = market_std.transpose(1, 2)  # (batch, C, T) for Conv1d
        logits = self.model(seq, pos_std)  # (batch, 2)
        return {"exit": logits.unsqueeze(1)}  # (batch, k=1, 2) -- matches TabM ensemble-pooling shape


def _load_tcn_bundle(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _tcn_loaded_models(base_cols: list[str], tcn_bundle: dict[str, Any], device: torch.device) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Shape-compatible with parent._load_payloads' return value (dict[expert] -> (model, scaler)),
    for harnesses (replay_exit_variant_windowed) that accept `loaded_models` directly."""
    if list(base_cols) != list(tcn_bundle["base_cols"]):
        raise RuntimeError("base_cols mismatch between requested harness bundle and TCN training bundle")
    cols = list(base_cols) + list(parent.POS_COLS)
    scaler = {"columns": cols, "mean": np.zeros(len(cols), dtype=np.float32), "std": np.ones(len(cols), dtype=np.float32)}
    out: dict[str, tuple[Any, dict[str, Any]]] = {}
    for expert in hard.EXPERT_NAMES:
        model = tcn_train.TCNExitClassifier(
            in_ch=len(tcn_bundle["base_cols"]), pos_dim=len(tcn_bundle["pos_cols"]), **tcn_bundle["arch"]
        )
        model.load_state_dict(tcn_bundle["models"][expert])
        wrapper = TCNExitHeadWrapper(model, device, len(tcn_bundle["base_cols"]), tcn_bundle["market_scaler"], tcn_bundle["pos_scaler"])
        out[expert] = (wrapper, scaler)
    return out


def _inject_tcn_exit_runtime(prepped: dict[str, Any], tcn_bundle: dict[str, Any], device: torch.device, base_cols: list[str]) -> dict[str, Any]:
    """Shape-compatible override for harnesses (greedy_replay_windowed, via prepare_component /
    _prepare_component_val) that already built `exit_runtime` -- replaces only that dict key,
    everything else (dec/atr/margin/leverage/route/exit_threshold) untouched. Returns a new dict
    (does not mutate `prepped`), mirroring research_eth_omega461_gbdt_exit_head_val_20260813.
    _inject_gbdt_exit_runtime exactly."""
    cols = list(base_cols) + list(parent.POS_COLS)
    n = int(prepped["base_np"].shape[1])
    if n != len(cols):
        raise RuntimeError(f"TCN injection column count mismatch: base_np width={n} vs base_cols+POS_COLS={len(cols)}")
    tcn_loaded = _tcn_loaded_models(base_cols, tcn_bundle, device)
    zeros, ones = np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
    out = dict(prepped)
    out["exit_runtime"] = {expert: (tcn_loaded[expert][0], zeros, ones) for expert in hard.EXPERT_NAMES}
    return out


@torch.no_grad()
def _predict_exit_prob_one_windowed(
    base_np: np.ndarray,
    runtime: dict[str, tuple[Any, np.ndarray, np.ndarray]],
    pos_idx: list[int],
    *,
    row_i: int,
    expert: str,
    pos_values: list[float],
    device: torch.device,
    window: int,
) -> float:
    """Copy of train_eval_omega4_2_risk_sidecar_20260622._predict_exit_prob_one, modified ONLY to
    slice a WINDOW of history (base_np[row_i-window+1 : row_i+1], left-zero-padded near a series'
    start via train_eval_omega461_tcn_exit_head_liveatr_20260813._slice_window -- the SAME helper
    the training script's windowed dataset uses, imported not reimplemented, so train-time and
    replay-time windowing are byte-identical) instead of a single row, because
    _predict_exit_prob_one structurally cannot provide a TCN the sequence context it needs (see
    module docstring). Position-state (pos_idx columns) is substituted only into the LAST (current)
    row of the window, exactly matching how the original substitutes it into the single current
    row -- earlier bars in the window never had this trade's real position state computed for
    them."""
    model, mean, std = runtime[expert]
    raw_window = tcn_train._slice_window(base_np, int(row_i), int(window)).copy()
    raw_window[-1, np.asarray(pos_idx, dtype=np.int64)] = np.asarray(pos_values, dtype=np.float32)
    x = ((raw_window - mean) / std).astype(np.float32)
    probs = torch.softmax(model(torch.from_numpy(x).unsqueeze(0).to(device))["exit"], dim=-1).mean(dim=1)
    return float(probs.detach().cpu().numpy()[0, 1])


@torch.no_grad()
def replay_exit_variant_windowed(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    exit_threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
    window: int,
    trailing_activate_frac: float | None = None,
    trailing_retain_frac: float | None = None,
    trailing_trail_frac: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Copy of research_eth_omega461_exit_sweep_20260721.replay_exit_variant -- logic is 100%
    identical (same TP/SL/trailing/exit-head order, same fill/cost model, same causal forward loop)
    EXCEPT the exit-head probability call uses _predict_exit_prob_one_windowed(..., window=window)
    instead of rs._predict_exit_prob_one(...). See module docstring for why this copy (rather than
    editing the original) is required. research_eth_omega461_exit_sweep_20260721.py itself is
    NEVER imported-and-mutated or edited by this script -- only read, once, to produce this copy.
    fresh_forward_bar_by_bar=true (single forward pass, i in increasing order, only row i and
    already-closed prior bars -- including the added window lookback, itself strictly historical --
    used at bar i); no saved ledger is used as input."""
    trailing_enabled = trailing_activate_frac is not None and (
        trailing_retain_frac is not None or trailing_trail_frac is not None)
    if trailing_retain_frac is not None and trailing_trail_frac is not None:
        raise ValueError("pass either trailing_retain_frac (proportional) or trailing_trail_frac (fixed distance)")
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_signal_i = 0
    notional = 0.0
    leverage = 1.0
    margin_fraction = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    armed = False
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    margin_sum = 0.0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    route = hard._route_id(frame)
    from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _try_execution as omega_try_execution, _fill_price as omega_fill_price

    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(base_x, loaded_models)

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            mfe = max(mfe, move)
            mae = min(mae, move)
        else:
            move = 0.0

        if pos != 0:
            reason = ""
            exit_prob = 0.0
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            elif trailing_enabled and (not armed) and mfe >= float(trailing_activate_frac) * take_profit and take_profit > 0.0:
                armed = True
            if not reason and trailing_enabled and armed and mfe > 0.0:
                if trailing_retain_frac is not None:
                    if move <= float(trailing_retain_frac) * mfe:
                        reason = "trailing_stop"
                elif move <= mfe - float(trailing_trail_frac) * abs(stop_loss):
                    reason = "trailing_stop"
            if not reason:
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = _predict_exit_prob_one_windowed(
                    base_np, exit_runtime, pos_idx, row_i=int(i), expert=expert,
                    pos_values=[
                        float(pos), float(hold), float(move), float(mfe), float(mae),
                        float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move), float(move + abs(stop_loss)),
                        float(notional), float(leverage), float(notional * leverage), float(take_profit), float(stop_loss),
                    ],
                    device=device, window=window,
                )
                exit_prob = float(prob)
                if prob >= float(exit_threshold):
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                trades += 1
                win = int(cash > entry_equity)
                wins += win
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({
                    "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(i),
                    "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                    "exit_timestamp": str(frame["timestamp"].iloc[int(i)]), "side": int(pos), "reason": reason,
                    "win": int(win), "raw_exit_price_move": float(raw_exit), "mfe_price_move": float(mfe),
                    "mae_price_move": float(mae), "trade_return": float(trade_return),
                    "net_per_notional": float(trade_return / max(notional, 1.0e-12)), "notional": float(notional),
                    "margin_fraction": float(margin_fraction), "leverage": float(leverage),
                    "exit_prob": float(exit_prob), "take_profit": float(take_profit), "stop_loss": float(stop_loss),
                })
                pos = 0
                armed = False
                continue
        eq = cash if pos == 0 else cash * (1.0 + move * notional)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, fee_paid, _route = omega_try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        row_leverage = float(risk_leverage[int(i)])
        row_margin = float(risk_margin_fraction[int(i)])
        row_notional = row_margin * row_leverage
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        leverage = row_leverage
        margin_fraction = row_margin
        notional = row_notional
        base_tp = float(row.get("take_profit", 0.0) or 0.0)
        base_sl = float(row.get("stop_loss", 0.0) or 0.0)
        if bool(notional_scaled_sltp):
            take_profit = base_tp * row_notional
            stop_loss = base_sl * row_notional
        else:
            take_profit = base_tp
            stop_loss = base_sl
        cash -= cash * fee_paid * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        margin_sum += margin_fraction
        mfe = 0.0
        mae = 0.0
        armed = False

    if pos != 0:
        exit_px = omega_fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        trades += 1
        win = int(cash > entry_equity)
        wins += win
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append({
            "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(len(frame) - 1),
            "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]), "exit_timestamp": str(frame["timestamp"].iloc[-1]),
            "side": int(pos), "reason": "forced_end", "win": int(win), "raw_exit_price_move": float(raw_exit),
            "mfe_price_move": float(mfe), "mae_price_move": float(mae), "trade_return": float(trade_return),
            "net_per_notional": float(trade_return / max(notional, 1.0e-12)), "notional": float(notional),
            "margin_fraction": float(margin_fraction), "leverage": float(leverage), "exit_prob": 0.0,
            "take_profit": float(take_profit), "stop_loss": float(stop_loss),
        })

    n_entries = max(long_entries + short_entries, 1)
    ledger = pd.DataFrame(rows)
    hold_bars = (ledger["exit_i"] - ledger["entry_i"]).clip(lower=0) if len(ledger) else pd.Series(dtype=float)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades),
            "wr": float(wins / trades) if trades else 0.0, "trades_per_day": float(trades / rs._duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries), "avg_leverage": float(leverage_sum / n_entries),
            "avg_hold_bars": float(hold_bars.mean()) if len(hold_bars) else 0.0,
            "max_trade_pnl": float(ledger["trade_return"].max() * 100.0) if len(ledger) else 0.0,
            "p95_trade_pnl": float(ledger["trade_return"].quantile(0.95) * 100.0) if len(ledger) else 0.0,
            "long_entries": int(long_entries), "short_entries": int(short_entries), "exit_reasons": reasons,
        },
        ledger,
    )


@torch.no_grad()
def greedy_replay_windowed(
    frame: pd.DataFrame, components: dict, *, fee: float, slip: float, cost_mult: float,
    device: torch.device, window: int, trailing_activate_frac: float | None = None,
    trailing_trail_frac: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Copy of replay_omega4_6_1_greedy_router_20260706.greedy_replay -- logic is 100% identical
    EXCEPT the exit-head probability call DISPATCHES per active component's model: if that expert's
    model has IS_WINDOWED=True (TCNExitHeadWrapper), it goes through
    _predict_exit_prob_one_windowed(..., window=window); otherwise (a real, unwrapped ThreeHeadTabM,
    e.g. zig075's untouched original bundle) it goes through the ORIGINAL, unmodified
    rs._predict_exit_prob_one -- required because this single replay loop can hold either
    component's position at a given time, and only h48qual's exit_head is TCN here (zig075 keeps
    its real TabM model, which cannot accept a windowed tensor). See module docstring.
    replay_omega4_6_1_greedy_router_20260706.py itself is NEVER edited by this script -- only read,
    once, to produce this copy."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    active_comp = None
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = leverage_v = margin_fraction = 0.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    armed = False
    trailing_enabled = trailing_activate_frac is not None and trailing_trail_frac is not None
    rows: list[dict] = []
    reasons: dict[str, int] = {}

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            unreal = move * notional
            mfe, mae = max(mfe, move), min(mae, move)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if not reason and trailing_enabled:
                if (not armed) and take_profit > 0.0 and mfe >= float(trailing_activate_frac) * take_profit:
                    armed = True
                if armed and mfe > 0.0 and move <= mfe - float(trailing_trail_frac) * abs(stop_loss):
                    reason = "trailing_stop"
            if not reason:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                model_for_expert = comp["exit_runtime"][expert][0]
                pos_values = [float(pos), float(hold), float(move), float(mfe), float(mae),
                              float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move),
                              float(move + abs(stop_loss)), float(notional), float(leverage_v),
                              float(notional * leverage_v), float(take_profit), float(stop_loss)]
                if getattr(model_for_expert, "IS_WINDOWED", False):
                    prob = _predict_exit_prob_one_windowed(
                        comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=pos_values, device=device, window=window,
                    )
                else:
                    prob = rs._predict_exit_prob_one(
                        comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=pos_values, device=device,
                    )
                if prob >= comp["exit_threshold"]:
                    reason = "exit_head"
            if reason:
                exit_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee_eff * notional
                trade_return = cash / max(entry_equity, 1e-12) - 1.0
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({"entry_signal_i": entry_signal_i, "entry_i": entry_i, "exit_i": i,
                             "entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]),
                             "exit_timestamp": str(frame["timestamp"].iloc[i]), "side": int(pos),
                             "source_component": active_comp, "reason": reason,
                             "win": int(cash > entry_equity), "trade_return": float(trade_return),
                             "notional": float(notional), "margin_fraction": float(margin_fraction),
                             "leverage": float(leverage_v)})
                pos, active_comp = 0, None
                continue
            continue

        for name in greedy.PRIORITY:
            if name not in components:
                continue
            comp = components[name]
            side = int(comp["dec"]["side"].iloc[i])
            if side == 0 or not bool(omega._active(comp["dec"]).iloc[i] if hasattr(omega._active(comp["dec"]), "iloc") else omega._active(comp["dec"])[i]):
                continue
            row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
            if row_margin <= 0.0:
                continue
            scale = greedy.SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(row_leverage * scale, greedy.LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
            row_leverage = row_notional / max(row_margin, 1e-12)
            if row_notional <= 0.0:
                continue
            entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
            pos, active_comp = side, name
            entry_price, entry_equity = float(entry_px), cash
            entry_i, entry_signal_i = min(i + 1, n - 1), i
            margin_fraction, leverage_v, notional = row_margin, row_leverage, row_notional
            take_profit = float(comp["dec"]["take_profit"].iloc[i])
            stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
            cash -= cash * fee_eff * notional
            mfe = mae = 0.0
            armed = False
            break

    return {"reason_counts": reasons}, pd.DataFrame(rows)


def _evaluate_component_val_tcn(tcn_bundle: dict[str, Any]) -> dict[str, Any]:
    """h48qual-standalone VAL ledger, TabM live-ATR baseline (original, unwindowed
    sweep.replay_exit_variant) vs TCN (replay_exit_variant_windowed). Mirrors
    research_eth_omega461_gbdt_exit_head_val_20260813._evaluate_component_val's single-prep,
    swap-loaded_models-only pattern."""
    cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    val_pred = sweep.EXT_PRED_DIR / "h48qual" / f"validation_predictions_{cfg['q_tag']}.csv"

    prepped = sweep.prep_component("h48qual", cfg, val_frame, val_pred, oof=True)
    m_tabm, _ledger_tabm = sweep.replay_exit_variant(
        prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
        risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=prepped["fee"], slip=prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=sweep.DEVICE,
    )

    base_cols = list(torch.load(cfg["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    tcn_loaded = _tcn_loaded_models(base_cols, tcn_bundle, sweep.DEVICE)
    m_tcn, _ledger_tcn = replay_exit_variant_windowed(
        prepped["frame"], prepped["x"], prepped["dec"], tcn_loaded,
        risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=prepped["fee"], slip=prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=sweep.DEVICE,
        window=int(tcn_bundle["window"]),
    )
    return {"tabm_liveatr": m_tabm, "tcn": m_tcn}


def _run_portfolio_variant_tcn(
    val_frame: pd.DataFrame, aligned_pred_paths: dict[str, Path], tcn_bundle: dict[str, Any], *, fee: float, slip: float,
) -> dict[str, Any]:
    h48qual_cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    zig075_cfg = portfolio._component_cfg("zig075")
    base_cols = list(torch.load(h48qual_cfg["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    h48qual_prepped = portfolio._prepare_component_val(val_frame, aligned_pred_paths["h48qual"], h48qual_cfg, portfolio.DEVICE)
    h48qual_tcn = _inject_tcn_exit_runtime(h48qual_prepped, tcn_bundle, portfolio.DEVICE, base_cols)
    zig075_prepped = portfolio._prepare_component_val(val_frame, aligned_pred_paths["zig075"], zig075_cfg, portfolio.DEVICE)
    components = {"h48qual": h48qual_tcn, "zig075": zig075_prepped}
    _diag, ledger = greedy_replay_windowed(
        val_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=portfolio.DEVICE, window=int(tcn_bundle["window"]),
    )
    ledger.to_csv(OUT_DIR / "portfolio_ledger_asymmetric_h48qual_tcn_zig075_original.csv", index=False)
    metrics = portfolio._ledger_metrics(ledger)
    print(f"  asymmetric_h48qual_tcn_zig075_original: {json.dumps({k: v for k, v in metrics.items() if k not in ('reason_counts', 'source_component_counts')})}", flush=True)
    return metrics


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== stage=G0_self_check ===", flush=True)
    g0_component = h48cons._evaluate_val("h48qual", portfolio.NEW_H48QUAL_BUNDLE)
    print(f"  component baseline_original: {g0_component['baseline']}", flush=True)
    print(f"  component tabm_liveatr: {g0_component['h48cons_relabel']}", flush=True)
    g0_ok_component_baseline = _close_to_reference(g0_component["baseline"], G0_REFERENCE["component_baseline_original"])
    g0_ok_component_tabm = _close_to_reference(g0_component["h48cons_relabel"], G0_REFERENCE["component_tabm_liveatr"])

    val_frame_raw = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    fee, slip = omega._load_fee_slip()
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in ("h48qual", "zig075")}
    val_frame, aligned_pred_paths = portfolio._align_frame_and_predictions(val_frame_raw, q_tags)
    print(f"  VAL aligned rows={len(val_frame)} (from raw {len(val_frame_raw)})", flush=True)

    portfolio_baseline = portfolio.run_variant(
        "baseline_both_original",
        {"h48qual": portfolio._component_cfg("h48qual"), "zig075": portfolio._component_cfg("zig075")},
        val_frame, aligned_pred_paths, fee=fee, slip=slip,
    )
    portfolio_tabm_liveatr = portfolio.run_variant(
        "asymmetric_h48qual_liveatr_zig075_original",
        {"h48qual": portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE), "zig075": portfolio._component_cfg("zig075")},
        val_frame, aligned_pred_paths, fee=fee, slip=slip,
    )
    g0_ok_portfolio_baseline = _close_to_reference(portfolio_baseline, G0_REFERENCE["portfolio_baseline_both_original"])
    g0_ok_portfolio_tabm = _close_to_reference(portfolio_tabm_liveatr, G0_REFERENCE["portfolio_asymmetric_tabm_liveatr"])

    g0_pass = bool(g0_ok_component_baseline and g0_ok_component_tabm and g0_ok_portfolio_baseline and g0_ok_portfolio_tabm)
    g0_report = {
        "component_baseline_original": {"actual": g0_component["baseline"], "reference": G0_REFERENCE["component_baseline_original"], "match": g0_ok_component_baseline},
        "component_tabm_liveatr": {"actual": g0_component["h48cons_relabel"], "reference": G0_REFERENCE["component_tabm_liveatr"], "match": g0_ok_component_tabm},
        "portfolio_baseline_both_original": {"actual": portfolio_baseline, "reference": G0_REFERENCE["portfolio_baseline_both_original"], "match": g0_ok_portfolio_baseline},
        "portfolio_asymmetric_tabm_liveatr": {"actual": portfolio_tabm_liveatr, "reference": G0_REFERENCE["portfolio_asymmetric_tabm_liveatr"], "match": g0_ok_portfolio_tabm},
        "tolerance_pp": G0_TOLERANCE_PP,
        "pass": g0_pass,
    }
    print(f"stage=G0_result pass={g0_pass}", flush=True)

    if not g0_pass:
        report = {
            "stage_reached": "G0_self_check",
            "g0": g0_report,
            "gate_pass": False,
            "note": "G0 failed -- this harness does not reproduce the published TabM live-ATR reference numbers. Aborting before evaluating TCN (per methodology discipline, TCN numbers from an unverified harness are not trustworthy).",
        }
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
        print(f"report={OUT_DIR / 'report.json'}", flush=True)
        print("stage=ABORT G0 failed", flush=True)
        return 1

    print("=== stage=tcn_evaluation ===", flush=True)
    if not TCN_BUNDLE.exists():
        raise FileNotFoundError(f"TCN bundle not found, run train_eval_omega461_tcn_exit_head_liveatr_20260813.py first: {TCN_BUNDLE}")
    tcn_bundle = _load_tcn_bundle(TCN_BUNDLE)

    component_tcn = _evaluate_component_val_tcn(tcn_bundle)
    print(f"  component tabm_liveatr (rechecked): {component_tcn['tabm_liveatr']}", flush=True)
    print(f"  component tcn: {component_tcn['tcn']}", flush=True)
    portfolio_tcn = _run_portfolio_variant_tcn(val_frame, aligned_pred_paths, tcn_bundle, fee=fee, slip=slip)

    gate_component_pnl = float(component_tcn["tcn"]["pnl"]) >= float(component_tcn["tabm_liveatr"]["pnl"])
    gate_component_mdd = float(component_tcn["tcn"]["mdd"]) >= float(component_tcn["tabm_liveatr"]["mdd"])
    gate_portfolio_pnl = float(portfolio_tcn["pnl"]) >= float(portfolio_tabm_liveatr["pnl"])
    gate_portfolio_mdd = float(portfolio_tcn["mdd"]) >= float(portfolio_tabm_liveatr["mdd"])
    gate_pass = bool(gate_component_pnl and gate_component_mdd and gate_portfolio_pnl and gate_portfolio_mdd)
    print(
        f"stage=gate_result component_pnl={gate_component_pnl} component_mdd={gate_component_mdd} "
        f"portfolio_pnl={gate_portfolio_pnl} portfolio_mdd={gate_portfolio_mdd} gate_pass={gate_pass}",
        flush=True,
    )

    report = {
        "stage_reached": "tcn_evaluation",
        "g0": g0_report,
        "tcn_bundle": str(TCN_BUNDLE),
        "tcn_window": int(tcn_bundle["window"]),
        "tcn_arch": tcn_bundle["arch"],
        "component_level": {
            "tabm_liveatr": component_tcn["tabm_liveatr"],
            "tcn": component_tcn["tcn"],
            "gate_pnl_nonworse": gate_component_pnl,
            "gate_mdd_nonworse": gate_component_mdd,
        },
        "portfolio_level": {
            "baseline_both_original": portfolio_baseline,
            "asymmetric_h48qual_liveatr_zig075_original": portfolio_tabm_liveatr,
            "asymmetric_h48qual_tcn_zig075_original": portfolio_tcn,
            "gate_pnl_nonworse": gate_portfolio_pnl,
            "gate_mdd_nonworse": gate_portfolio_mdd,
        },
        "gate_pass": gate_pass,
        "gate_rule": "TCN non-worse than TabM live-ATR baseline on PnL AND MDD, at BOTH component and portfolio level",
        "val_window": [sweep.VAL_START, sweep.VAL_END],
        "oos_opened": False,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
