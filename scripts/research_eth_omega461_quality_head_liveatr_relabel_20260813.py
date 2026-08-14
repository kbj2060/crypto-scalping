#!/usr/bin/env python3
"""RESEARCH ONLY -- same-night follow-up to
`research_eth_omega461_exit_head_liveatr_relabel_20260813.py` (exit head fix), applying the SAME
diagnosis+fix pattern to h48qual's QUALITY head instead.

Defect: h48qual's quality_head (the `quality_threshold=0.50` entry filter, PRIORITY-1 in
`trading_bot_modules/omega4_6_1_live.py`'s `_ComponentConfig`) is trained via
`quality_mode="quality_label_action"` (`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`)
against a label sourced from `.../sltp_h48_conservative_padded_to_zigzag_timestamps` -- confirmed
directly from the deployed bundle's own report.json `label_contract`
(`tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/report.json`)
and that label directory's own `manifest.json` (source: `sltp_triple_barrier_h48_conservative`,
`BarrierConfig("h48_conservative", horizon=48, tp_mult=1.2, sl_mult=0.8, min_tp=0.006, min_sl=0.004)`,
ATR96-relative). But the LIVE component this quality head actually gates uses a completely different
barrier (`_ComponentConfig` defaults, `trading_bot_modules/omega4_6_1_live.py:91-97`): atr_window=192,
tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12, NO fixed time horizon.
Roughly a 10x mismatch on tp_mult, 7.5x on sl_mult, 12.5x on the tp floor, 2x on the ATR window --
same defect class, similar magnitude, as the exit-head bug already found+fixed tonight.

FIX, mechanically different from the exit-head fix because the quality label itself has a different
shape. The exit head needed a label at EVERY BAR WHILE HOLDING a position (hence dense per-hold-bar
`_position_feature_row` construction -- the thing that caused the earlier OOM). The quality head's
label (confirmed via `train_eval_omega4_3head_parent72_loose_entry_quality_20260620._quality_target`
mode="quality_label_action" -> `quality_labels["zigzag_action"]`, and the deployed padded-label CSV's
own schema/manifest) is a single 0/1/2 (cash/long/short) value PER 5-MINUTE BAR (dense over the whole
timeline, not just zigzag-pivot bars, then padded onto the direction label's own timestamp grid), and
the quality/direction heads ALWAYS see POS_COLS=0 at inference (`_Component.entry_decision` zero-fills
`parent.POS_COLS` before calling the model -- confirmed in `omega4_6_1_live.py` and mirrored by
`parent._base_input`). So this relabel needs ONE simulated outcome per (bar, side), not a growing
list of per-hold-bar feature rows -- no `_position_feature_row` construction at all, and therefore
none of the ~1.27M-row Python-list-of-dicts blowup that crashed the server on the exit-head attempt.

Because the new label still needs a value at EVERY bar (not a ~37K-candidate subsample like the
exit-head fix used), and the live ATR barrier resolves over hundreds of bars, a naive per-bar Python
walk-forward loop (~210K TRAIN bars x 2 sides x up to `--max-horizon-bars` steps) would be too slow.
This script instead reuses the numba-jitted dense dual-side ATR-barrier walk pattern already proven
in this repo for an analogous problem (`build_zigzag_action_labels_barrier_matched_20260704.py`,
Omega6/L2 sub-project, same "does a trade opened at bar i clear an ATR-scaled TP before an ATR-scaled
SL" question), adapted here for: (a) the live component's floor+cap ATR clip formula (not
Omega6's plain `mult*atr` linear scale -- reproduced from `_ComponentConfig.entry_decision`,
`np.clip(max(min_tp, atr_pct*tp_mult), 0, max_tp)`), (b) intrabar high/low touch detection for
TP/SL (matching `_build_exit_dataset_entry_label_live_atr_barrier`'s convention in the exit-head-fix
template, not Omega6's close-only checks), (c) this project's own fee/slip/notional constants
(`train_eval_omega1_2_tabm_diffusion_risk_20260603.BASE_TEMPLATE`/`_load_fee_slip`), and (d) a
computational-only horizon cap instead of Omega6's semantic 288-bar max_hold (see HORIZON DESIGN
NOTE below). The numba loop keeps ALL working memory as fixed-size float64/int arrays sized `n`
(a few MB total) -- there is no unbounded per-row object accumulation for this recipe's shape, so
the h48cons predecessor's chunk-flush/circuit-breaker pattern is not mechanically applicable here.
The lightweight available-memory/RSS guard from that predecessor is still reused (imported, not
reimplemented) around the training stage as a belt-and-suspenders measure, since this is a shared,
already-fragile-tonight server.

HORIZON DESIGN NOTE (explicit reasoning, per the coordinator's request): the ORIGINAL quality label
had a fixed 48-bar horizon. The live barrier this label should now match has NO time-based exit
baked into the TP/SL trigger itself (`_ComponentConfig` has no max_hold/time-stop field; hold time
in live trading is governed dynamically by the exit_head's probability output, not by the entry-time
barrier). Two options were considered: (a) drop the fixed horizon entirely, letting each hypothetical
trade run to TP/SL resolution (or a very generous cap used ONLY as a numerical safety valve, not a
real horizon), mirroring live's actual horizon-free behavior; (b) keep some bounded-horizon
compromise on the theory that real positions are not literally held forever. This script chooses
(a): the live TP/SL trigger truly has no time-exit, the already-completed exit-head-liveatr-relabel
fix (this script's own template) made the identical choice for the SAME barrier (dropping h48cons's
48-bar horizon, using `--max-horizon-bars 6000` purely as a compute cap, gated on a pre-training
bars-to-resolution checkpoint), and keeping both heads matched to the identical barrier philosophy
avoids introducing a second, arbitrary horizon convention. `--max-horizon-bars` defaults to 6000
(~20.8 days at 5-minute bars), the same value already used and checkpoint-validated by the exit-head
fix for this identical barrier. Option (b) was rejected: there is no principled bounded-horizon value
to justify beyond "some number felt safer", which is exactly the kind of unjustified free parameter
CLAUDE.md's simplicity guidance argues against, and it would make the two heads inconsistent for no
clear benefit.

VAL-EVAL CORRECTNESS NOTE (found while reading the template's `_evaluate_val`, verified before
reuse per the coordinator's explicit instruction not to assume): `research_eth_omega461_exit_sweep_20260721.prep_component`
builds entry decisions (`dec`, i.e. side/quality-gated action) from a STATIC precomputed prediction
CSV (`pred_csv` arg), NOT from a fresh forward pass of `cfg["bundle"]`'s loaded model -- `cfg["bundle"]`
is only ever consulted for the EXIT head during `replay_exit_variant` (`rs._predict_exit_prob_one`).
That is correct and sufficient for the exit-head fix (entry-side decisions are unchanged there), but
would silently no-op a quality-head fix: reusing `h48cons._evaluate_val` unchanged here would replay
the SAME baseline entries for both "baseline" and "new" rows, because the static CSV bakes in the
OLD quality head's decisions regardless of which bundle path is passed. This script therefore adds
`_fresh_predictions` (recomputes direction+quality from a given bundle via the same
`parent._predict_payload`/`_routed`/`_prediction_output` pipeline the h48qual driver's own main()
uses to generate its prediction CSVs) and evaluates VAL by feeding FRESH predictions into
`sweep.prep_component`/`sweep.replay_exit_variant` (both reused unchanged -- only the prediction
SOURCE changes). As a self-check, the report also includes a fresh-recomputed baseline (same
pipeline, original bundle) next to the original static-CSV baseline number; they should closely
agree, and a material mismatch would flag a bug in this new eval path rather than a real result.

Retrain scope: mirrors `_fit_exit_head_only`'s freeze pattern exactly, just aimed at
`model.quality_head` instead of `model.exit_head` -- encoder, direction_head, AND exit_head are all
loaded from the currently-deployed live bundle (`research_eth_omega461_exit_sweep_20260721.COMPONENTS[component]["bundle"]`,
verified identical to `runtime_config.py`'s `FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH`/
`_ZIG075_BUNDLE_PATH`) and kept frozen (`requires_grad_(False)`); only `quality_head`'s ~579 params
train. This intentionally starts from the ORIGINAL live bundle, not from tonight's exit-head-retrained
one, so this experiment's result is attributable to the quality relabel alone (one variable at a
time), matching this sub-project's isolation discipline. `quality_threshold` (0.50 h48qual / 0.75
zig075) is held FIXED at the live value for this first pass -- not re-swept -- for the same
one-variable-at-a-time reason (mirrors the exit-head fix holding `exit_threshold` fixed too).

fresh_forward_bar_by_bar=true (VAL replay is a single causal forward pass, `sweep.replay_exit_variant`,
unchanged). trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. The dense LABEL construction (this script's `_build_dense_quality_liveatr_label`)
legitimately uses future bars relative to each labeled bar -- offline training-target construction,
the same already-established convention as every other barrier/zigzag label in this repo (see
`build_zigzag_action_labels_barrier_matched_20260704.py`'s identical disclosure); never used as a
live/replay decision input. Training itself uses only the pre-2025-10-01 TRAIN split. VAL only --
this script never loads or scores OOS data.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT overwrite any live checkpoint or the original h48qual/zig075 driver scripts/bundles.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_h48cons_relabel_20260813 as h48cons  # noqa: E402
import research_eth_omega461_exit_head_liveatr_relabel_20260813 as exit_liveatr  # noqa: E402

MODEL_ID = "eth_omega461_quality_head_liveatr_relabel_20260813"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DIRECTION_LABEL_DIR = h48cons.DIRECTION_LABEL_DIR

# Exact live defaults -- reused (not re-copied) from the exit-head-fix template, which itself
# transcribed them verbatim from `trading_bot_modules/omega4_6_1_live.py:91-97` (`_ComponentConfig`
# dataclass field defaults). Single source of truth across both relabel scripts.
LIVE_ATR_CFG = exit_liveatr.LIVE_ATR_CFG

# Same computational-only cap as the exit-head fix for this identical barrier (see HORIZON DESIGN
# NOTE in the module docstring). Not a semantic time-stop.
DEFAULT_MAX_HORIZON_BARS = 6000

# Only label a side LONG/SHORT if its net-of-cost simulated P&L would have been strictly positive
# (same convention as `build_zigzag_action_labels_barrier_matched_20260704.py`'s MIN_UTILITY=0.0).
MIN_UTILITY = 0.0

# Reused (not reimplemented) from the exit-head-fix template -- see that script's module docstring
# for the incident this guards against (a memory-crash that took the shared server down for over an
# hour). This recipe's own working set is small fixed-size numpy arrays (see module docstring), so
# this is a belt-and-suspenders check around the training stage, not a load-bearing fix here.
_available_memory_gb = exit_liveatr._available_memory_gb
_process_rss_gb = exit_liveatr._process_rss_gb
MIN_AVAILABLE_MEMORY_GB = exit_liveatr.MIN_AVAILABLE_MEMORY_GB
MAX_PROCESS_RSS_GB = exit_liveatr.MAX_PROCESS_RSS_GB


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@numba.njit(cache=True)
def _simulate_dense_dual_side_liveatr_barrier(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_pct: np.ndarray,
    n: int,
    tp_mult: float,
    sl_mult: float,
    min_tp: float,
    min_sl: float,
    max_tp: float,
    max_sl: float,
    max_horizon_bars: int,
    notional: float,
    fee_eff: float,
    slip_eff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For every bar i, simulate a hypothetical LONG and a hypothetical SHORT entered at the next
    bar's open, walking forward (intrabar high/low touch) until the live ATR-adaptive TP or SL
    barrier fires or `max_horizon_bars` is exhausted. Returns net-of-cost P&L and bars-held per side,
    plus a resolution-reason code (0=sl, 1=tp, 2=timeout). Entry/exit fill uses a flat
    open/close-based slip model (round-trip, both legs), not `omega._try_execution`'s maker-limit
    fill logic -- a deliberate simplification for LABEL construction, matching the same tradeoff
    already made by `build_zigzag_action_labels_barrier_matched_20260704.py` for an analogous dense
    per-bar barrier label; this is a training target, not a claim about live executable fills.
    Single-threaded by design (no `parallel=True`/`prange`) to stay a light, predictable load on a
    shared server rather than fanning out across every core.
    """
    long_net = np.zeros(n, dtype=np.float64)
    short_net = np.zeros(n, dtype=np.float64)
    long_bars = np.zeros(n, dtype=np.int32)
    short_bars = np.zeros(n, dtype=np.int32)
    long_reason = np.full(n, 2, dtype=np.int8)
    short_reason = np.full(n, 2, dtype=np.int8)
    for i in range(n - 1):
        entry_i = i + 1
        a = atr_pct[i]
        if a <= 0.0:
            continue
        tp_move = min(max(min_tp, a * tp_mult), max_tp)
        sl_move = min(max(min_sl, a * sl_mult), max_sl)
        end_i = entry_i + max_horizon_bars
        if end_i > n - 1:
            end_i = n - 1
        if end_i < entry_i:
            continue
        for side in range(2):  # 0 = long, 1 = short
            entry_open = open_[entry_i]
            if entry_open <= 0.0:
                continue
            entry_price = entry_open * (1.0 + slip_eff) if side == 0 else entry_open * (1.0 - slip_eff)
            if side == 0:
                tp_level = entry_price * (1.0 + tp_move)
                sl_level = entry_price * (1.0 - sl_move)
            else:
                tp_level = entry_price * (1.0 - tp_move)
                sl_level = entry_price * (1.0 + sl_move)
            reason = 2
            resolve_i = end_i
            for j in range(entry_i, end_i + 1):
                hi = high[j]
                lo = low[j]
                if side == 0:
                    hit_sl = lo <= sl_level
                    hit_tp = hi >= tp_level
                else:
                    hit_sl = hi >= sl_level
                    hit_tp = lo <= tp_level
                if hit_sl:
                    reason = 0
                    resolve_i = j
                    break
                if hit_tp:
                    reason = 1
                    resolve_i = j
                    break
            if reason == 1:
                exit_price = tp_level
            elif reason == 0:
                exit_price = sl_level
            else:
                exit_price = close[resolve_i]
            if side == 0:
                raw = (exit_price * (1.0 - slip_eff) - entry_price) / entry_price
            else:
                raw = (entry_price - exit_price * (1.0 + slip_eff)) / entry_price
            net = raw * notional - 2.0 * fee_eff * notional
            bars_held = resolve_i - entry_i
            if side == 0:
                long_net[i] = net
                long_bars[i] = bars_held
                long_reason[i] = reason
            else:
                short_net[i] = net
                short_bars[i] = bars_held
                short_reason[i] = reason
    return long_net, short_net, long_bars, short_bars, long_reason, short_reason


def _build_dense_quality_liveatr_label(
    frame: pd.DataFrame,
    *,
    atr_cfg: dict[str, float],
    max_horizon_bars: int,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"dense quality liveatr label missing columns: {missing}")
    n = len(frame)
    open_ = pd.to_numeric(frame["open"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    atr_pct = atr_eval._atr_pct(frame, int(atr_cfg["atr_window"]))
    notional = float(omega.BASE_TEMPLATE["notional"])
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)

    t0 = time.time()
    long_net, short_net, long_bars, short_bars, long_reason, short_reason = _simulate_dense_dual_side_liveatr_barrier(
        open_, high, low, close, atr_pct, n,
        float(atr_cfg["tp_mult"]), float(atr_cfg["sl_mult"]),
        float(atr_cfg["min_tp"]), float(atr_cfg["min_sl"]), float(atr_cfg["max_tp"]), float(atr_cfg["max_sl"]),
        int(max_horizon_bars), notional, fee_eff, slip_eff,
    )
    elapsed = time.time() - t0

    label = np.zeros(n, dtype=np.int64)
    best_util = np.maximum(long_net, short_net)
    long_better = long_net >= short_net
    positive = best_util > MIN_UTILITY
    label[positive & long_better] = 1
    label[positive & ~long_better] = 2

    def _reason_counts(reason_arr: np.ndarray, active_mask: np.ndarray) -> dict[str, int]:
        names = {0: "sl", 1: "tp", 2: "timeout"}
        vals, counts = np.unique(reason_arr[active_mask], return_counts=True)
        return {names[int(v)]: int(c) for v, c in zip(vals, counts)}

    # length-n mask (not n-1) to match label/long_reason/short_reason -- the simulate loop itself
    # only ever visits i in [0, n-2], so index n-1 is forced False here to mirror that exactly.
    valid = np.zeros(n, dtype=bool)
    if n > 1:
        valid[: n - 1] = atr_pct[: n - 1] > 0.0
    counts = {int(k): int(v) for k, v in zip(*np.unique(label, return_counts=True))}
    diag = {
        "rows": int(n),
        "valid_atr_bars": int(valid.sum()),
        "class_counts": {"0_cash": counts.get(0, 0), "1_long": counts.get(1, 0), "2_short": counts.get(2, 0)},
        "class_ratios": {k: float(v) / max(n, 1) for k, v in {"0_cash": counts.get(0, 0), "1_long": counts.get(1, 0), "2_short": counts.get(2, 0)}.items()},
        "long_reason_counts_valid_bars": _reason_counts(long_reason, valid) if n > 1 else {},
        "short_reason_counts_valid_bars": _reason_counts(short_reason, valid) if n > 1 else {},
        "long_bars_held_mean_when_labeled": float(long_bars[label == 1].mean()) if int((label == 1).sum()) else 0.0,
        "short_bars_held_mean_when_labeled": float(short_bars[label == 2].mean()) if int((label == 2).sum()) else 0.0,
        "min_utility": float(MIN_UTILITY),
        "max_horizon_bars": int(max_horizon_bars),
        "atr_cfg": dict(atr_cfg),
        "build_elapsed_sec": float(elapsed),
    }
    if diag["class_counts"]["1_long"] == 0 or diag["class_counts"]["2_short"] == 0:
        raise RuntimeError(f"dense quality liveatr label degenerate (a side never wins): {diag['class_counts']}")
    return label, diag


def _ce_quality(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, 3),
        target[:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
        reduction="none",
    ).reshape(-1, int(parent.CFG.k)).mean(dim=1)


def _fit_quality_head_only(
    baseline_payload: dict[str, Any],
    x_dir: pd.DataFrame,
    y_qual: np.ndarray,
    route_frame: pd.DataFrame,
    *,
    expert_idx: int,
    seed: int,
    epochs: int,
    device: torch.device,
    model_path: Path,
) -> dict[str, Any]:
    """Mirrors `train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622._fit_exit_head_only`
    exactly, aimed at `quality_head` instead of `exit_head`: loads the baseline (currently-deployed)
    per-expert payload, freezes every parameter, then unfreezes ONLY `model.quality_head` before
    training. Encoder, direction_head, and exit_head weights are untouched (identical tensors to the
    live bundle after `load_state_dict`, since their `requires_grad_(False)` params never receive an
    optimizer step)."""
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    if list(x_dir.columns) != list(baseline_payload["scaler"]["columns"]):
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} feature column contract mismatch for quality-only retrain")
    x_np = parent._standardize_apply(x_dir, baseline_payload["scaler"])
    y_np = np.asarray(y_qual, dtype=np.int64)
    classes = sorted(np.unique(y_np).astype(int).tolist())
    if not set(classes).issubset({0, 1, 2}):
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} quality labels must be subset of {{0,1,2}}, got {classes}")
    route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    weights = compute_sample_weight(class_weight="balanced", y=y_np).astype(np.float32) * route_w
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid quality-only sample weights")

    n = len(y_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    ds = TensorDataset(torch.from_numpy(x_np[train_idx]), torch.from_numpy(y_np[train_idx]), torch.from_numpy(weights[train_idx]))
    dl = DataLoader(ds, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)

    model = parent.ThreeHeadTabM(int(baseline_payload["n_features"]), cfg=parent.CFG).to(device)
    model.load_state_dict(baseline_payload["state_dict"])
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.quality_head.parameters():
        param.requires_grad_(True)
    opt = torch.optim.AdamW(model.quality_head.parameters(), lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        for xb, yb, wb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            out = model(xb)
            loss = (_ce_quality(out["quality"], yb) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.quality_head.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_np[val_idx]).to(device)
            vy = torch.from_numpy(y_np[val_idx]).to(device)
            vw = torch.from_numpy(weights[val_idx]).to(device)
            vo = model(vx)
            val_loss = float(((_ce_quality(vo["quality"], vy) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
        if val_loss + 1.0e-6 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(parent.CFG.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    payload = {
        **baseline_payload,
        "model_id": MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "best_quality_validation_loss": float(best_loss),
        "quality_epochs_ran": int(last_epoch),
        "frozen_contract": "encoder_direction_exit_frozen_quality_head_only_retrained",
        "quality_target": "quality_liveatr_barrier_dense_relabel_20260813",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_path)
    return payload


def _retrain_component_quality_head_liveatr(
    component: str,
    frame: pd.DataFrame,
    y_qual: np.ndarray,
    route_frame: pd.DataFrame,
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    baseline_bundle_path = sweep.COMPONENTS[component]["bundle"]
    bundle = torch.load(baseline_bundle_path, map_location=device, weights_only=False)
    baseline_models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    x_dir = parent._base_input(frame, base_cols)

    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        model_path = out_dir / component / "models" / f"{expert}_3head_tabm_quality_liveatr.pt"
        payload = _fit_quality_head_only(
            baseline_models[expert], x_dir, y_qual, route_frame,
            expert_idx=idx, seed=int(seed), epochs=int(epochs), device=device, model_path=model_path,
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(model_path),
            "quality_epochs_ran": int(payload["quality_epochs_ran"]),
            "best_quality_validation_loss": float(payload["best_quality_validation_loss"]),
        }

    bundle_path = out_dir / component / "true_3head_tabm_bundle.pt"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": parent.CFG.__dict__, "model_id": MODEL_ID},
        bundle_path,
    )
    return {"baseline_bundle": str(baseline_bundle_path), "new_bundle": str(bundle_path), "summaries": summaries}


def _fresh_predictions(cfg: dict[str, Any], frame: pd.DataFrame, bundle_path: Path, *, device: torch.device) -> tuple[pd.DataFrame, str]:
    """Recompute direction+quality predictions FRESH from `bundle_path`'s own forward pass, using
    the exact pipeline `train_eval_omega4_3head_parent72_loose_entry_quality_20260620.main()` uses to
    build its own `validation_predictions_qXXX.csv` (`parent._predict_payload`/`_routed`/
    `_prediction_output`). Needed because `sweep.prep_component` takes predictions from a CSV file,
    not from the bundle directly -- see VAL-EVAL CORRECTNESS NOTE in the module docstring."""
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    base_cols = list(bundle["base_cols"])
    models = bundle["models"]
    x = parent._base_input(frame, base_cols)
    route = hard._route_id(frame)
    preds = {expert: parent._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    q_tag = f"q{round(float(cfg['quality_threshold']) * 100):03d}"
    src = parent._prediction_output(frame, direction, quality, threshold=float(cfg["quality_threshold"]), prefix="omega1_regime3_expertdq_oof")
    return src, q_tag


def _evaluate_val_quality(component: str, new_bundle_path: Path, out_dir: Path) -> dict[str, Any]:
    cfg = dict(sweep.COMPONENTS[component])
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    comp_dir = out_dir / component
    comp_dir.mkdir(parents=True, exist_ok=True)

    def _run(bundle_path: Path, tag: str) -> dict[str, Any]:
        src, q_tag = _fresh_predictions(cfg, val_frame, bundle_path, device=sweep.DEVICE)
        pred_path = comp_dir / f"validation_predictions_{tag}_{q_tag}.csv"
        src.to_csv(pred_path, index=False)
        cfg_run = dict(cfg)
        cfg_run["bundle"] = bundle_path
        prepped = sweep.prep_component(component, cfg_run, val_frame, pred_path, oof=True)
        metrics, _ledger = sweep.replay_exit_variant(
            prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
            risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
            exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=prepped["fee"], slip=prepped["slip"],
            cost_mult=sweep.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=sweep.DEVICE,
        )
        return metrics

    # Self-check: baseline evaluated via the ORIGINAL static precomputed CSV (the well-established
    # path, same one `h48cons._evaluate_val`'s baseline branch uses) vs. this script's own
    # fresh-recompute pipeline applied to the SAME unchanged bundle. These should closely agree;
    # a material mismatch flags a bug in `_fresh_predictions`/`_evaluate_val_quality`, not a real
    # finding, and is reported as such rather than silently trusted.
    static_pred_csv = sweep.EXT_PRED_DIR / component / f"validation_predictions_{cfg['q_tag']}.csv"
    baseline_static_prepped = sweep.prep_component(component, cfg, val_frame, static_pred_csv, oof=True)
    baseline_static, _ = sweep.replay_exit_variant(
        baseline_static_prepped["frame"], baseline_static_prepped["x"], baseline_static_prepped["dec"], baseline_static_prepped["loaded"],
        risk_margin_fraction=baseline_static_prepped["margin"], risk_leverage=baseline_static_prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=baseline_static_prepped["fee"], slip=baseline_static_prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=baseline_static_prepped["notional_scaled_sltp"], device=sweep.DEVICE,
    )
    baseline_fresh = _run(Path(cfg["bundle"]), "baseline_fresh")
    new_metrics = _run(new_bundle_path, "new")

    pnl_mismatch = abs(float(baseline_static["pnl"]) - float(baseline_fresh["pnl"]))
    return {
        "baseline_static_precomputed_csv": baseline_static,
        "baseline_fresh_recomputed_from_original_bundle": baseline_fresh,
        "baseline_self_check_pnl_abs_diff": float(pnl_mismatch),
        "baseline_self_check_note": "fresh-recompute pipeline vs static CSV on the SAME unchanged bundle; large abs_diff would indicate a bug in this script's eval path, not a real finding",
        "quality_liveatr_relabel": new_metrics,
        "quality_threshold_held_fixed_at": float(cfg["quality_threshold"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["checkpoint_only", "full"], default="full")
    ap.add_argument("--max-horizon-bars", type=int, default=DEFAULT_MAX_HORIZON_BARS)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260813)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = torch.device("cpu")
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=prepare_frames", flush=True)
    t0 = time.time()
    frames = omega4._prepare_frames(
        disable_tp_sl=False, direction_label_dir=DIRECTION_LABEL_DIR,
        quality_mode="same_as_direction", quality_label_dir=None,
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()
    train_df = frames["train_df"]
    print(f"  train_df rows={len(train_df)} elapsed={time.time() - t0:.1f}s "
          f"avail_gb={_available_memory_gb():.2f} rss_gb={_process_rss_gb():.2f}", flush=True)

    print("stage=build_dense_quality_liveatr_label", flush=True)
    t0 = time.time()
    y_qual, label_diag = _build_dense_quality_liveatr_label(
        train_df, atr_cfg=LIVE_ATR_CFG, max_horizon_bars=int(args.max_horizon_bars),
        fee=fee, slip=slip, cost_mult=float(args.cost_mult),
    )
    print(f"  label build elapsed={time.time() - t0:.1f}s", flush=True)
    (out_dir / "stage0_label_checkpoint.json").write_text(
        json.dumps(label_diag, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    print(json.dumps(label_diag, ensure_ascii=False, indent=2, default=_json_default), flush=True)

    if str(args.stage) == "checkpoint_only":
        print("stage=done (checkpoint_only)", flush=True)
        return 0

    route_frame = train_df
    results: dict[str, Any] = {"label_checkpoint": label_diag, "components": {}}
    for component in ("h48qual", "zig075"):
        avail_gb = _available_memory_gb()
        rss_gb = _process_rss_gb()
        print(f"stage=pre_retrain_memory_check component={component} avail_gb={avail_gb:.2f} rss_gb={rss_gb:.2f}", flush=True)
        if avail_gb < MIN_AVAILABLE_MEMORY_GB or rss_gb > MAX_PROCESS_RSS_GB:
            print(f"  MEMORY_SAFETY_STOP before {component} retrain -- avail_gb={avail_gb:.2f} "
                  f"(floor {MIN_AVAILABLE_MEMORY_GB}) rss_gb={rss_gb:.2f} (cap {MAX_PROCESS_RSS_GB})", flush=True)
            results["components"][component] = {"skipped_for_memory": True}
            continue

        print(f"stage=retrain_quality_head component={component}", flush=True)
        t0 = time.time()
        retrain_info = _retrain_component_quality_head_liveatr(
            component, train_df, y_qual, route_frame,
            seed=int(args.seed), epochs=int(args.epochs), device=device, out_dir=out_dir,
        )
        print(f"  {component} retrain elapsed={time.time() - t0:.1f}s", flush=True)

        print(f"stage=evaluate_val component={component}", flush=True)
        t0 = time.time()
        val_metrics = _evaluate_val_quality(component, Path(retrain_info["new_bundle"]), out_dir)
        print(f"  {component} eval elapsed={time.time() - t0:.1f}s", flush=True)
        print(json.dumps({"component": component, "val": val_metrics}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
        results["components"][component] = {"retrain": retrain_info, "val_metrics": val_metrics}

    report = {
        "model_id": MODEL_ID,
        "predecessor": "eth_omega461_exit_head_liveatr_relabel_20260813 (exit head fix, same night, same barrier)",
        "design": (
            "Dense per-bar (every TRAIN bar, not a candidate subsample) dual-side (long+short) "
            "live-ATR-barrier relabel of h48qual's quality_head target, replacing the h48_conservative "
            "48-bar/ATR96/1.2x/0.8x barrier with the LIVE _ComponentConfig barrier "
            "(atr_window=192, tp_mult=12.0, sl_mult=6.0, floors 0.075/0.040, caps 0.22/0.12), "
            "no fixed time horizon (see HORIZON DESIGN NOTE in the script docstring), computed via a "
            "numba-jitted forward walk (adapted from build_zigzag_action_labels_barrier_matched_20260704.py). "
            "Retrains ONLY quality_head per expert (encoder/direction_head/exit_head frozen from the "
            "currently-deployed live bundle), mirroring _fit_exit_head_only's freeze pattern. "
            "quality_threshold held fixed at the live value (0.50 h48qual / 0.75 zig075), not re-swept."
        ),
        "uses_future_only_for_offline_labeling": True,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "val_window": [sweep.VAL_START, sweep.VAL_END],
        "oos_opened": False,
        **results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={out_dir / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
