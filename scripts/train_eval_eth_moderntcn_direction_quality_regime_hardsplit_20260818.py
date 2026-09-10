#!/usr/bin/env python3
"""RESEARCH ONLY -- ModernTCN direction+quality, TabM training technique + regime hard-split.

User instruction chain (2026-08-18): after establishing that TabM's exit/direction heads already
had sequence-input variants tested and closed (TCN, docs/experiments/eth_h48qual_tcn_sequence_model_
20260812.md and eth_omega461_tcn_exit_head_20260813.md), and that quality_head+sequence-input was
started but abandoned mid-flight (eth_nhits_moderntcn_direction_quality, killed 2026-08-17 to free
GPU for a different priority, only one partial isolation-stage datapoint captured), user asked to
resume the ModernTCN line but with TabM's own training technique ported in, and to compare against
TODAY's regime-hard-split TabM parent (see eth_candidate_hardregime_pilot_seed2559205075/report.json)
rather than the abandoned script's own TabM-control arm.

Reuses train_eval_eth_direction_quality_nhits_moderntcn_20260816.py's architecture code (
ModernTCNBackbone, TwoHeadClassifier, WindowDataset, build_backbone, ARCH_DEFAULT_PARAMS,
_standardize_fit, _valid_indices, _split_with_embargo) UNMODIFIED via import -- that file is not
touched. Everything below it (data loading, training loop, regime routing) is new.

Three deliberate deviations from the abandoned script, each disclosed:

1. Quality label: same_as_direction (y_qual_full = y_dir_full), NOT h48_conservative. The abandoned
   script's own h48_conservative choice does not match today's Phase-1-confirmed-best quality label
   (docs/experiments/eth_candidate_unified_single_component_redesign_20260817.md) used by the
   regime-hard-split TabM parent this is being compared against -- using h48_conservative here would
   confound "architecture" with "quality label" in the comparison.

2. Regime hard-split: 3 separate models (bull/bear/chop), each trained ONLY on rows where
   argmax(regime3_current_sensitive_wide24_{bull,bear,chop}_prob) == that regime (weight=0 rows are
   literally excluded from the row-index array, not soft-weighted) -- mirrors exactly how the
   --hard-regime-filter flag works in train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py
   (route_probs.argmax(axis=1) == expert_idx). The abandoned script trained one unified model; this
   was the axis needed to make it comparable to the hard-split TabM baseline.

3. TRAIN_START moved from 2024-06-01 to 2025-01-01 (boundary-change disclosure): the regime3 route
   CSVs (data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/) only cover
   2025-01-01 onward -- 2024-06..2024-12 rows have no regime label and would be unusable for hard
   splitting. Root cause verified 2026-08-18: the regime3 HMM itself was fit+selected on 2024 data
   (experiment_regime3_current_hmm_wide24_20260529.py, fit_source=training_features_2024.csv,
   validation_policy="2024Q4 validation; 2025/2026 forward tests") -- 2024 regime labels would be
   in-sample for the classifier itself even if they existed, so there was never a usable 2024 route
   file to begin with, independent of the data-availability question.

4. TRAIN_END extended from 2025-09-30 to 2026-02-28, and VAL/OOS redefined as 2 months each
   (2026-08-18, user instruction: data first, then 2mo/2mo VAL/OOS) -- VAL=2026-03-01..04-30,
   OOS=2026-05-01..06-30, using up every bit of regime3-labeled data through the 2026-06-30 route
   ceiling with TRAIN taking everything before VAL.
   direction_label_dir now points at a freshly regenerated zigzag_action_labels_20260531_extended_
   20260818 (built via scripts/build_wave3_action_labels_20260531.py, unmodified, same params,
   pointed at the current data/splits/year_oos/training_features_2026_rebuilt.csv which now extends
   to 2026-07-20) instead of the registered zigzag_action_labels_20260531 (truncated 2026-02-28,
   because the label build's OWN 2026-05-31 run only had price data through 2026-02-28 at the time --
   not an algorithm limit). Verified before use, not trusted blind: 2024/2025 label counts reproduce
   the original registry's manifest.json exactly (10335/50512/44533 and 12332/49149/43620); the 42/
   16897 (0.25%) rows that differ from the old file in the Jan-Feb 2026 overlap are all in the old
   file's last ~11 days (2026-02-17..02-28) -- the zigzag boundary-uncertainty effect (the old file
   couldn't see past 2026-02-28 to confirm the in-progress swing; the new one, seeing through
   2026-07-20, resolves it correctly), not a reproduction bug.
   2025Q1-2026Q1 all fall inside the new TRAIN range and are no longer meaningful eval windows (this
   mirrors the finding, also made today, that the old 2025Q1-Q3 numbers were in-sample and
   uninformative for both models). Neither VAL nor OOS overlaps today's earlier oos_q2 (2026-04-01..
   06-30) exactly, so this run's numbers are not a like-for-like drop-in replacement for today's
   earlier comparison table -- report both windows on their own terms.

Training technique -- TabM's ACTUAL current recipe (train_eval_omega1_2_tabm_3head_20260603.py's
_fit_expert_3head/_fit_expert_omega4), not the abandoned script's own EMA/warmup/label-smoothing/
GCE/ELR/mixup additions (those are that script's own author's generic-DL-checklist bolt-ons, not
literally what TabM does -- user asked to inherit TabM's technique, so this file does NOT carry
those over): AdamW, plain class-balanced weighted cross-entropy (quality weight = direction weight,
since quality target == direction target here), quality_loss_weight=0.80 (matches CFG.quality_loss_
weight exactly). ONE deliberate upgrade over TabM's own patience=8 counter, per explicit user
instruction to use "the validated best" rather than a blind copy: Prechelt UP_4 strip-based early
stopping (STRIP_LEN=5, UP_S=4) + cosine LR (2e-4->2e-6), both already validated better in this repo
(feedback_modern_dl_training_checklist memory; research_eth_candidate_faithful_tabm_batchensemble_
baseline_grid_prechelt_20260816.py is the reference implementation this ports from, byte-identical
strip logic).

Staged: --stage pilot (1 seed x 3 regimes, ARCH_DEFAULT_PARAMS/ARCH_DEFAULT_TRAIN, sanity + first
directional read) and --stage final (N>=5 genuinely-random seeds x 3 regimes, FINAL_ARCH_PARAMS/
FINAL_TRAIN_PARAMS -- the winning config from hpsearch_eth_moderntcn_regime_hardsplit_20260818.py's
chop-proxy search, top_candidates.json rank#0: window=48, dropout=0.0555, lr=7.974e-4,
weight_decay=6.988e-6, batch_size=1024, use_revin=False, sel_loss=0.9198 -- that search was stopped
after 5 completed trials by explicit user decision, justified because optuna.samplers.TPESampler
defaults to n_startup_trials=10, so trials 0-9 are ALL pure random sampling regardless of whether the
search runs 5 or 10 trials -- confirmed via inspect.signature before stopping, not assumed).
--stage final resumes per-regime across interruptions: if out_dir/report.json already has a
completed entry (with an on-disk model file) for a given (seed, regime), that regime is skipped and
the prior result reused -- added 2026-08-18 after two same-day server GPU stalls and two dev WSL2/
Windows restarts wiped in-progress work with no persistence.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false (causal window construction reused unmodified from the
base script; VAL/OOS labels/predictions are never joined backward into TRAIN).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_eth_direction_quality_nhits_moderntcn_20260816 as base_nt  # noqa: E402 -- reused unmodified, not copied
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402 -- _try_execution, BASE_TEMPLATE (notional only, see below)
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402 -- _continue_to_barrier_net
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402 -- _atr_pct

MODEL_ID = "eth_moderntcn_direction_quality_regime_hardsplit_20260818"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

WIDE24_2025 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
WIDE24_2026 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
# Freshly regenerated 2026-08-18 (not the registered zigzag_action_labels_20260531, truncated
# 2026-02-28) via scripts/build_wave3_action_labels_20260531.py unmodified, same params, against
# the current training_features_2026_rebuilt.csv (now extends to 2026-07-20). Verified: 2024/2025
# counts reproduce the original manifest exactly; see module docstring point 4 for the full check.
DIRECTION_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531_extended_20260818"

TRAIN_START, TRAIN_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-02-28 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-30 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-30 23:59:59")

STRIP_LEN = 5
UP_S = 4
LR_MAX = 2.0e-4
LR_MIN = 2.0e-6
QUALITY_LOSS_WEIGHT = 0.80  # matches train_eval_omega1_2_tabm_3head_20260603.CFG.quality_loss_weight
PILOT_MAX_EPOCHS = 30
MIN_FIT_ROWS = 500
MIN_ES_ROWS = 100

# 2026-08-18: winning config from hpsearch_eth_moderntcn_regime_hardsplit_20260818.py's chop-proxy
# search (top_candidates.json rank#0, sel_loss=0.9198) -- see module docstring for why the search was
# stopped after 5/10 trials. "lr" deliberately dropped from FINAL_TRAIN_PARAMS: audited 2026-08-18 and
# found _fit_one_regime's AdamW call below uses the module-level LR_MAX constant directly, never
# train_params.get("lr", ...) -- the hpsearch script's "lr" dimension (1e-4..5e-3 log-uniform) was a
# dead key for all 5 completed trials, every one of them actually trained at the same fixed LR_MAX=2e-4
# cosine schedule regardless of what Optuna suggested. Not re-searched (user decision): LR_MAX=2e-4 is
# already this repo's own validated-best value (feedback_modern_dl_training_checklist memory), so the
# bug happens to have landed on a good value anyway -- kept as-is rather than spending more trials on it.
FINAL_TRAIN_PARAMS = {"window": 48, "weight_decay": 6.98841654484495e-06, "batch_size": 1024}
FINAL_ARCH_EXTRA = {"dropout": 0.05553211975846527, "use_revin": False}

# Matches eval_eth_moderntcn_regime_hardsplit_val_oos_20260818.ATR_CFG exactly (not imported --
# importing that eval module here would be circular, since it imports this module).
ATR_CFG = {"atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0, "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12}


def _quality_target_risk_adjusted_barrier_meta_action_atr(
    panel: pd.DataFrame, y_dir_full: np.ndarray, *, fee: float, slip: float, cost_mult: float,
    min_edge: float, max_mae_sl_ratio: float, min_mfe_mae: float, max_hold_bars: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """2026-08-19 -- ported from train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py's
    _quality_target_risk_adjusted_barrier_meta_action (same simulate-to-barrier + MFE/MAE/hold-gate
    logic, same omega._try_execution/exit_head._continue_to_barrier_net calls) with ONE deliberate
    fix: uses THIS session's ATR-adaptive TP/SL (ATR_CFG, matching the live omega4_6_1_live.py
    formula) instead of the original's fixed omega.BASE_TEMPLATE take_profit=2.6%/stop_loss=1.4% --
    CLAUDE.md's Position-Feature Train/Inference Parity Contract flags exactly that fixed-constant
    pattern as a known, previously-fixed-elsewhere bug class; porting it unfixed here would just
    reproduce it in a new file. omega.BASE_TEMPLATE["notional"] (0.45, risk sizing only, unrelated
    to the TP/SL bug) is still used, matching this session's NOTIONAL/LEVERAGE convention throughout.

    NOT fixed here (disclosed, not silently carried over): the barrier check inside
    exit_head._continue_to_barrier_net is CLOSE-PRICE-ONLY (walks forward comparing unrealized PnL
    against close each bar), not the intrabar high/low touch that the live evaluate_exit() actually
    uses for h48qual/zig075 -- reusing the intrabar-correct barrier walker would be a bigger change
    than this exploratory quality-label test calls for; inherited as-is like the original TabM runs
    that used this same function.

    2026-08-19 recalibration (round 2, after round 1's positive_rate_active=0.42% came back degenerate
    -- diag: active=200774, pass=848, net_edge_fail=126746(63%)/hold_fail=31466(16%)/mae_fail=41714
    (21%)): max_mae is now max_mae_sl_ratio * that bar's own ATR-adaptive sl_move (0.71, the original
    fixed thresholds' own implied ratio: 0.01/0.014=0.714) instead of a fixed 0.01 -- scales with the
    now much-wider ATR SL (4-12% vs the original's fixed 1.4%) instead of silently staying tight
    relative to it. max_hold_bars doubled (288->576) since round 1's mae_fail wasn't even the dominant
    rejection reason (net_edge_fail was, at 63%) so hold_fail(16%) likely also needs more room given
    the wider TP takes longer to reach. min_edge/min_mfe_mae unchanged (ratio/small-fixed, not tied to
    absolute TP/SL width). Still check this function's own positive_rate_active/reason_counts
    diagnostic before trusting the result -- not assumed fixed just because it's less naive.
    """
    arrays = {c: pd.to_numeric(panel[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    notional = float(omega.BASE_TEMPLATE["notional"])
    max_hold = int(max_hold_bars)

    atr_pct = atr_eval._atr_pct(panel, ATR_CFG["atr_window"])
    tp_move = np.clip(np.maximum(ATR_CFG["min_tp"], atr_pct * ATR_CFG["tp_mult"]), 0.0, ATR_CFG["max_tp"])
    sl_move = np.clip(np.maximum(ATR_CFG["min_sl"], atr_pct * ATR_CFG["sl_mult"]), 0.0, ATR_CFG["max_sl"])

    out = np.zeros(len(panel), dtype=np.int64)
    active = filled = positive = 0
    reason_counts: dict[str, int] = {}
    net_values: list[float] = []
    for i in range(0, len(panel) - 2):
        a_raw = y_dir_full[i]
        if not np.isfinite(a_raw):
            continue
        a = int(a_raw)
        if a not in (1, 2):
            continue
        active += 1
        side = 1 if a == 1 else -1
        ok, entry_price, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not ok:
            reason_counts["entry_not_filled"] = reason_counts.get("entry_not_filled", 0) + 1
            net_values.append(-1.0)
            continue
        filled += 1
        entry_i = min(int(i) + 1, len(panel) - 1)
        cash_after_entry_fee = 1.0 - 1.0 * float(entry_fee) * notional
        net, final_i, _reason = exit_head._continue_to_barrier_net(
            arrays, start_i=entry_i, side=side, entry_price=float(entry_price),
            cash_after_entry_fee=cash_after_entry_fee, notional=notional,
            take_profit=float(tp_move[i]), stop_loss=float(sl_move[i]),
            fee_eff=fee_eff, slip_eff=slip_eff,
        )
        end_i = max(min(int(final_i), len(panel) - 1), int(entry_i))
        mfe = mae = 0.0
        for row_i in range(int(entry_i), end_i + 1):
            px = float(arrays["close"][int(row_i)])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, float(unreal))
            mae = min(mae, float(unreal))
        mae_abs = abs(float(mae))
        mfe_mae = float(mfe) / max(mae_abs, 1.0e-8)
        hold_bars = max(int(end_i) - int(entry_i), 0)
        max_mae_i = float(max_mae_sl_ratio) * float(sl_move[i])
        net_values.append(float(net))
        if float(net) <= float(min_edge):
            reason_counts["net_edge_fail"] = reason_counts.get("net_edge_fail", 0) + 1
        elif mae_abs > max_mae_i:
            reason_counts["mae_fail"] = reason_counts.get("mae_fail", 0) + 1
        elif mfe_mae < float(min_mfe_mae):
            reason_counts["mfe_mae_fail"] = reason_counts.get("mfe_mae_fail", 0) + 1
        elif max_hold > 0 and hold_bars > max_hold:
            reason_counts["hold_fail"] = reason_counts.get("hold_fail", 0) + 1
        else:
            reason_counts["pass"] = reason_counts.get("pass", 0) + 1
            out[i] = a
            positive += 1
    arr = np.asarray(net_values, dtype=np.float64) if net_values else np.asarray([0.0], dtype=np.float64)
    diag = {
        "mode": "risk_adjusted_barrier_meta_action_atr", "active_rows": int(active), "filled_entries": int(filled),
        "positive_rows": int(positive), "positive_rate_active": float(positive / max(active, 1)),
        "reason_counts": reason_counts, "net_mean": float(arr.mean()), "net_p50": float(np.quantile(arr, 0.50)),
    }
    return out, diag


def log(msg: str) -> None:
    print(f"[moderntcn_hardsplit] {msg}", flush=True)


def load_data_samedir_with_regime(quality_mode: str = "same_as_direction") -> dict[str, Any]:
    log(f"panel + zigzag_action(direction) + regime3 hard route 로딩... quality_mode={quality_mode}")
    panel = pd.read_csv(base_nt.PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    missing = [c for c in base_nt.SEQ_COLS if c not in panel.columns]
    if missing:
        raise RuntimeError(f"SEQ_COLS missing: {missing}")

    dir_labels = pd.concat([
        pd.read_csv(DIRECTION_LABEL_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
        for y in (2024, 2025, 2026)
    ], ignore_index=True).drop_duplicates("timestamp", keep="last")
    dir_map = dir_labels.set_index("timestamp")["zigzag_action"]

    route = pd.concat([pd.read_csv(WIDE24_2025, low_memory=False), pd.read_csv(WIDE24_2026, low_memory=False)], ignore_index=True)
    route["timestamp"] = pd.to_datetime(route["timestamp"])
    route = route.drop_duplicates("timestamp", keep="last")
    route_map = route.set_index("timestamp")[hard.ROUTE_COLS]

    raw = panel[base_nt.SEQ_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y_dir_full = dir_map.reindex(panel["timestamp"]).to_numpy()
    if quality_mode == "same_as_direction":
        y_qual_full = y_dir_full.copy()  # quality target IS the direction label
    elif quality_mode == "risk_adjusted_barrier_meta_action_atr":
        y_qual_full, qual_diag = _quality_target_risk_adjusted_barrier_meta_action_atr(
            panel, y_dir_full, fee=omega.FEE_RATE, slip=omega.SLIP_RATE, cost_mult=3.0,
            min_edge=0.001, max_mae_sl_ratio=0.71, min_mfe_mae=1.2, max_hold_bars=576,
        )
        log(f"  risk_adjusted_barrier_meta_action_atr diag: {qual_diag}")
    else:
        raise RuntimeError(f"unknown quality_mode: {quality_mode}")

    route_aligned = route_map.reindex(panel["timestamp"]).to_numpy(dtype=np.float64)
    route_valid = np.isfinite(route_aligned).all(axis=1)
    route_id_full = np.full(len(panel), -1, dtype=np.int64)
    route_id_full[route_valid] = np.argmax(route_aligned[route_valid], axis=1)

    log(f"  패널 {len(panel)}행. direction 결측={np.isnan(y_dir_full.astype(np.float64)).sum()} regime3 결측={(~route_valid).sum()}")
    return {"panel": panel, "raw": raw, "y_dir_full": y_dir_full, "y_qual_full": y_qual_full, "route_id_full": route_id_full}


def _valid_indices_regime(mask: np.ndarray, window: int, y_dir_full: np.ndarray, y_qual_full: np.ndarray, route_id_full: np.ndarray, regime_idx: int) -> np.ndarray:
    idx = base_nt._valid_indices(mask, window, y_dir_full, y_qual_full)
    return idx[route_id_full[idx] == regime_idx]


def _fit_one_regime(arch: str, arch_params: dict[str, Any], train_params: dict[str, Any], *, seed: int,
                     epochs: int, regime_idx: int | None, data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """regime_idx=None means unified/no-split: trains on ALL valid TRAIN rows, no route filtering."""
    base_nt._seed_everything(seed)
    window = int(train_params.get("window", base_nt.DEFAULT_WINDOW))
    train_mask = ((data["panel"]["timestamp"] >= TRAIN_START) & (data["panel"]["timestamp"] <= TRAIN_END)).to_numpy()
    label = hard.EXPERT_NAMES[regime_idx] if regime_idx is not None else "unified"
    if regime_idx is not None:
        train_idx_all = _valid_indices_regime(train_mask, window, data["y_dir_full"], data["y_qual_full"], data["route_id_full"], regime_idx)
    else:
        train_idx_all = base_nt._valid_indices(train_mask, window, data["y_dir_full"], data["y_qual_full"])
    fit_idx, es_idx = base_nt._split_with_embargo(train_idx_all, window)
    if len(fit_idx) < MIN_FIT_ROWS or len(es_idx) < MIN_ES_ROWS:
        raise RuntimeError(f"{label}: too few rows for training (fit={len(fit_idx)}, es={len(es_idx)})")

    log(f"    {label}: fit_rows={len(fit_idx)} es_rows={len(es_idx)} -- standardizing+building datasets...")
    raw_std, _ = base_nt._standardize_fit(data["raw"], fit_idx, window)
    ds_fit = base_nt.WindowDataset(raw_std, window, fit_idx, data["y_dir_full"], data["y_qual_full"])
    ds_es = base_nt.WindowDataset(raw_std, window, es_idx, data["y_dir_full"], data["y_qual_full"])
    batch_size = int(train_params.get("batch_size", 512))
    dl_fit = DataLoader(ds_fit, batch_size=batch_size, shuffle=True)
    dl_es = DataLoader(ds_es, batch_size=1024, shuffle=False)
    log(f"    datasets built, steps_per_epoch={max(1, len(fit_idx) // batch_size)} -- building model+starting training loop...")

    backbone = base_nt.build_backbone(arch, len(base_nt.SEQ_COLS), window, arch_params).to(device)
    model = base_nt.TwoHeadClassifier(backbone, backbone.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=float(train_params.get("weight_decay", 2.0e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs), eta_min=LR_MIN)

    y_dir_fit = data["y_dir_full"][fit_idx].astype(np.int64)
    dir_w_all = compute_sample_weight("balanced", y_dir_fit).astype(np.float32)

    def selection_val_loss_and_bacc() -> tuple[float, float]:
        model.eval()
        losses, preds, trues = [], [], []
        with torch.no_grad():
            for xb, yb_dir, _yb_qual, _ridb in dl_es:
                xb, yb_dir = xb.to(device), yb_dir.to(device)
                out = model(xb)
                k = out["direction"].shape[1]
                ce = torch.nn.functional.cross_entropy(
                    out["direction"].reshape(-1, 3), yb_dir[:, None].expand(-1, k).reshape(-1), reduction="none",
                ).reshape(-1, k).mean(dim=1)
                losses.append(ce.cpu().numpy())
                preds.append(torch.softmax(out["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy())
                trues.append(yb_dir.cpu().numpy())
        all_loss = np.concatenate(losses)
        bacc = balanced_accuracy_score(np.concatenate(trues), np.concatenate(preds))
        return float(all_loss.mean()), float(bacc)

    curve: list[dict[str, Any]] = []
    best_sel_loss = float("inf")
    best_epoch = 0
    best_bacc_at_selection = None
    best_state: dict[str, torch.Tensor] | None = None
    last_strip_val = None
    bad_strips = 0
    for epoch in range(int(epochs)):
        epoch_t0 = time.time()
        model.train()
        for xb, yb_dir, _yb_qual, ridb in dl_fit:
            xb, yb_dir = xb.to(device), yb_dir.to(device)
            wb = torch.from_numpy(dir_w_all[ridb.numpy()]).to(device)
            out = model(xb)
            k = out["direction"].shape[1]
            loss_dir_k = torch.nn.functional.cross_entropy(out["direction"].reshape(-1, 3), yb_dir[:, None].expand(-1, k).reshape(-1), reduction="none").reshape(-1, k)
            loss_qual_k = torch.nn.functional.cross_entropy(out["quality"].reshape(-1, 3), yb_dir[:, None].expand(-1, k).reshape(-1), reduction="none").reshape(-1, k)
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss = loss_dir + QUALITY_LOSS_WEIGHT * loss_qual
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        scheduler.step()

        sel_loss, bacc = selection_val_loss_and_bacc()
        curve.append({"epoch": epoch + 1, "lr": round(float(scheduler.get_last_lr()[0]), 8), "sel_loss": round(sel_loss, 5), "bacc": round(bacc, 5)})
        log(f"    epoch={epoch + 1} sel_loss={sel_loss:.5f} bacc={bacc:.4f} bad_strips={bad_strips} elapsed={time.time() - epoch_t0:.1f}s")

        if sel_loss < best_sel_loss:
            best_sel_loss = sel_loss
            best_epoch = epoch + 1
            best_bacc_at_selection = bacc
            best_state = {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}

        strip_epoch = epoch + 1
        if strip_epoch % STRIP_LEN == 0:
            if last_strip_val is not None:
                bad_strips = bad_strips + 1 if sel_loss > last_strip_val else 0
            last_strip_val = sel_loss
            if bad_strips >= UP_S:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "regime": label, "fit_rows": int(len(fit_idx)), "es_rows": int(len(es_idx)),
        "window": window, "epochs_ran": len(curve), "selected_epoch": best_epoch,
        "selected_sel_loss": best_sel_loss, "selected_bacc": best_bacc_at_selection, "curve": curve,
        "model_state": {k2: v.cpu() for k2, v in model.state_dict().items()}, "n_features": len(base_nt.SEQ_COLS),
        "arch_params": arch_params, "train_params": train_params,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "final"], default="pilot")
    ap.add_argument("--arch", choices=["moderntcn", "nhits"], default="moderntcn")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=PILOT_MAX_EPOCHS)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--out-suffix", default="")
    # 2026-08-19: --split unified added per explicit user request -- a single model trained on ALL
    # TRAIN rows (no regime3 route filtering), to check whether the N=3-seed hard-split's broadly
    # negative VAL/OOS result (eval_eth_moderntcn_regime_hardsplit_val_oos_20260818.py, 8/9 threshold
    # cells negative) is specific to the hard-split axis or a more basic ModernTCN/architecture issue.
    # Same tuned hyperparameters as --stage final (trial1 from the chop-proxy hpsearch) -- no separate
    # HP search was run for the unified case, disclosed rather than silently reused as if validated.
    ap.add_argument("--split", choices=["hard", "unified"], default="hard")
    # 2026-08-19: risk_adjusted_barrier_meta_action_atr added per explicit user request, after
    # confirming (a) the current same_as_direction quality target is literally a second classifier
    # re-predicting the SAME zigzag_action label direction predicts, not any profitability signal,
    # and (b) this exact quality target IS what the TabM hard-split parent uses too (report.json
    # confirmed), so this isn't a ModernTCN-specific mislabeling -- and (c) this repo already tried
    # risk_adjusted_barrier_meta_action twice on TabM with mixed results (best-of-5 OOS PnL once,
    # but also a bacc regression from "multitask interference" once) -- see this function's own
    # _quality_target_risk_adjusted_barrier_meta_action_atr docstring for the full history + the ATR
    # fix applied when porting.
    ap.add_argument("--quality-mode", choices=["same_as_direction", "risk_adjusted_barrier_meta_action_atr"], default="same_as_direction")
    args = ap.parse_args()

    device = base_nt._device(args.device)
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    expert_names = list(hard.EXPERT_NAMES) if args.split == "hard" else ["unified"]
    report_path = out_dir / "report.json"
    results: dict[str, Any] = {}
    if report_path.exists():
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        for expert, r in prior.get("results", {}).items():
            if Path(r.get("model_path", "")).exists():
                results[expert] = r
        if results:
            log(f"resuming from {report_path}: {sorted(results)} already done, skipping")

    remaining_experts = [e for e in expert_names if e not in results]
    if not remaining_experts:
        log("all regimes already done, nothing to do")
        return 0

    t0 = time.time()
    data = load_data_samedir_with_regime(quality_mode=args.quality_mode)
    log(f"data loaded elapsed={time.time()-t0:.1f}s")

    if args.stage == "final":
        arch_params = {**base_nt.ARCH_DEFAULT_PARAMS[args.arch], **FINAL_ARCH_EXTRA}
        train_params = dict(FINAL_TRAIN_PARAMS)
    else:
        arch_params = base_nt.ARCH_DEFAULT_PARAMS[args.arch]
        train_params = dict(base_nt.ARCH_DEFAULT_TRAIN)

    for regime_idx, expert in enumerate(expert_names) if args.split == "hard" else [(None, "unified")]:
        if expert not in remaining_experts:
            continue
        log(f"=== stage={args.stage} split={args.split} regime={expert} seed={args.seed} epochs<={args.epochs} ===")
        t0 = time.time()
        r = _fit_one_regime(args.arch, arch_params, train_params, seed=args.seed, epochs=args.epochs, regime_idx=regime_idx, data=data, device=device)
        elapsed = time.time() - t0
        log(f"  {expert}: fit_rows={r['fit_rows']} es_rows={r['es_rows']} epochs_ran={r['epochs_ran']} "
            f"selected_epoch={r['selected_epoch']} selected_bacc={r['selected_bacc']:.4f} elapsed={elapsed:.1f}s")
        model_path = out_dir / "models" / f"{expert}_moderntcn_regime_hardsplit.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": r.pop("model_state"), "n_features": r["n_features"], "window": r["window"],
                    "arch": args.arch, "arch_params": arch_params}, model_path)
        r["elapsed_sec"] = elapsed
        r["model_path"] = str(model_path)
        results[expert] = r

        report = {
            "model_id": MODEL_ID, "arch": args.arch, "seed": int(args.seed), "stage": args.stage,
            "quality_mode": args.quality_mode, "regime_hard_split": args.split == "hard",
            "train_start": str(TRAIN_START), "train_end": str(TRAIN_END),
            "val_start": str(VAL_START), "val_end": str(VAL_END), "oos_start": str(OOS_START), "oos_end": str(OOS_END),
            "results": results,
            "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        log(f"  report checkpoint saved (regime={expert})")

    log(f"report={report_path}")
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
