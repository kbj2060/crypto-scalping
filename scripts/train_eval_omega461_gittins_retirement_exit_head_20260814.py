#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 post-entry literature scouting (#6) rank-5 candidate: Deep RL for the
Gittins index via the "retirement formulation" (Dhankhar, Mishra, Bodas, "Tabular and Deep
Reinforcement Learning for Gittins Index", arXiv:2405.01157, v1 2024-05-02 / v4 2025-08-25).
QGI = tabular Q-learning for the Gittins index; DGN = Deep Gittins Network, the neural-network
variant this script implements. See docs/experiments/eth_omega461_gittins_index_exit_head_20260814.md
for the full paper-mechanism writeup and this project's reformulation design/justification -- this
docstring summarizes only what the code does.

=== What DGN computes (paper Section III/IV, verified by direct WebFetch of the HTML full text) ===
Retirement formulation: for a state x, Vr(x,M) = max{Q_M(x,1), Q_M(x,0)=M} where Q_M(x,1) = r(x,1) +
gamma * E[max{Q_M(j,1), M}] is "the value of continuing, given you may bail out for a fixed prize M
at any future point". The indifference point M(x) = inf{M : Vr(x,M) = M} is the retirement value;
the Gittins index is G(x) = M(x)*(1-gamma). DGN learns, for a NEURAL-NETWORK-approximated continuous
state space, a function Q_theta(s, x) (paper Eq 9: two states as input -- the state s being valued
and a REFERENCE state x whose own indifference point is being tracked) via:
  target(s_k, x) = r(s_k) + gamma * max(Q_theta'(s'_k, x), M_n(x))          [paper Eq 9]
  loss = mean_k mean_x (target(s_k,x) - Q_theta(s_k,x))^2                   [paper Eq 10]
  M_{n+1}(x) = M_n(x) + beta(n) * (Q_theta(x,x) - M_n(x))                   [paper Eq 11, slow timescale]
Q_theta' is a Polyak-soft-updated target network (standard DQN device, needed for the max(...)
bootstrap to be stable -- the paper's own tabular/DGN convergence relies on the two-timescale
condition beta(n) = o(alpha(n))).

=== This script's ENGINEERING SIMPLIFICATIONS vs the paper (documented, not silent) ===
1. Reference-state set X is not a fixed enumerated table (paper's own experiments use small
   discrete/quantized state spaces, e.g. job "ages" 1..N_max) -- this project's state is a
   continuous 115-dim position-conditioned feature vector, so X is instead resampled every
   minibatch AS the minibatch's own visited states (a standard, well-established extension of
   index-style methods to continuous domains via function approximation: every transition still
   updates Q^x for many different x across training, just not the SAME x set every step).
2. M_n(x) (paper Eq 11, a separately-tracked slow EMA) is realized here via the SAME Polyak target
   network Q_theta' already required for the Eq-9 bootstrap, read at its own diagonal:
   M_n(x) := Q_theta'(x, x). This merges two "slow-tracked copy of a self-consistency quantity"
   mechanisms (DQN target network + Eq-11 M-update) into one, rather than maintaining a third
   network -- both are conceptually the same kind of device (a delayed, more stable estimate of the
   online network's own self-evaluation).
3. Training is OFFLINE/batch (fitted-Q-iteration style) over the SAME live-ATR-barrier candidate
   dataset research_eth_omega461_exit_head_liveatr_relabel_20260813 / train_eval_omega461_gbdt_
   exit_head_liveatr_20260813 already built and validated for GBDT (#4) and TCN (#5), not online
   environment interaction -- the paper's own scheduling experiments are online, but every bar
   within one sampled candidate's forward-simulated barrier path is a genuine sequential
   (s_t, r_t, s_{t+1}) transition (verified via frame_exit's exit_path_entry_i/exit_path_hold_bars
   ordering), so fitted-Q over these logged trajectories is a standard, not ad hoc, offline-RL
   adaptation.
Both are called out again, with full justification, in the companion doc -- this is not a silent
downgrade of the paper's method.

=== Reward / state / dataset (reused, not reinvented) ===
State = the SAME 115-dim (102 base_cols + 13 pos_cols) position-conditioned vector every h48qual
exit_head variant (TabM baseline, GBDT #4, TCN #5) already uses (parent._exit_input_from_position_
rows). Reward r_t = bar-to-bar delta of frame_exit["exit_path_unrealized"] (mark-to-market PnL
against the SAME fixed BASE_TEMPLATE notional the dataset builder already computed it with) --
r_0 (first bar of a candidate's path) = its own unrealized value minus 0. Terminal bar of each
candidate's forward-simulated path (TP/SL/timeout resolution) is a true episode end: target = r_T
only, no bootstrap. Dataset is train_eval_omega461_gbdt_exit_head_liveatr_20260813._build_dataset
(seed=260813, max_candidates=1500) imported UNCHANGED -- identical rows/positive_count/
used_candidates to the TabM/GBDT/TCN baseline dataset, reference-checked below exactly as those two
scripts already do.

Per-regime (bull/bear/chop) experts, each trained on ALL transitions weighted by that expert's soft
Regime3 route probability at the transition-owner bar (parent._route_probs) -- same routed-expert
STRUCTURE as every other exit_head variant in this lineage (integration point
train_eval_omega4_2_risk_sidecar_20260622._prepare_exit_runtime expects one model per
hard.EXPERT_NAMES), though the weighting itself is necessarily different from GBDT/TCN's
compute_sample_weight(balanced)*route_prob (regression has no class-imbalance concept to correct;
only the route-probability factor carries over, applied to BOTH the sample and reference axes of the
(B,B) loss matrix as an outer product).

fresh_forward_bar_by_bar=true (dataset build is the same causal forward barrier simulation the
TabM/GBDT/TCN runs used, unmodified; DGN training is offline fitted-Q over that dataset's own
already-causal per-candidate trajectories, not a new bar-by-bar simulation). trade_ledgers_used_as_
input=false. saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false. Training
uses only the pre-2025-10-01 TRAIN split (same frame as TabM/GBDT/TCN). Does NOT touch
trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env. Does NOT touch
zig075 in any way.

=== Server memory/GPU safety (per the coordinator's explicit warning: a prior exit-head retraining
job on this SAME shared server caused a full outage via memory exhaustion, and 3 shadow bots
(eth_exithead_asymmetric_shadow, eth_regime_aware_exit_guard_shadow, eth-jmlam4-shadow.service)
run continuously here) ===
Sequential per-expert training (never parallel), free -h memory logged before/after each expert via
_mem_check with a 4GB floor (same pattern as scripts/run_jm_full_retrain_seed_robustness_20260813.sh),
and GPU memory logged via torch.cuda.mem_get_info if --device cuda. Aborts (not "pushes through") if
available memory drops below the floor.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import research_eth_omega461_exit_head_h48cons_relabel_20260813 as h48cons  # noqa: E402
import train_eval_omega461_gbdt_exit_head_liveatr_20260813 as gbdt_train  # noqa: E402

MODEL_ID = "eth_omega461_gittins_retirement_exit_head_20260814"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
REFERENCE_REPORT = gbdt_train.REFERENCE_REPORT
SEED = 260813  # SAME seed as the TabM/GBDT/TCN baseline dataset build -- identical dataset, not a new one.
MAX_CANDIDATES_DEFAULT = 1500

GAMMA = 0.999  # baseline avg_hold_bars ~= 551-670 (asymmetric_tabm_liveatr VAL/OOS-Q1); 0.999**600 ~= 0.549
# -- decays meaningfully but not so sharply the network ignores multi-hundred-bar continuation value.
TARGET_TAU = 0.01  # Polyak soft-update factor for the target net (also serves as the Eq-11 M(x) reader).
ARCH_HIDDEN = (64, 128, 64)  # paper Section IV: "three hidden layers with (64,128,64) neurons, ReLU".
VAL_HOLDOUT_FRAC = 0.10  # candidate-level (not row-level) holdout, avoids leaking a candidate's own path across split.

MIN_AVAILABLE_MEMORY_GB = 4.0  # matches run_jm_full_retrain_seed_robustness_20260813.sh's floor.


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _available_memory_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return float("inf")


def _mem_check(label: str, floor_gb: float = MIN_AVAILABLE_MEMORY_GB) -> dict[str, Any]:
    avail_gb = _available_memory_gb()
    gpu_gb = None
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        gpu_gb = {"free_gb": free_b / 1.0e9, "total_gb": total_b / 1.0e9}
    print(f"[MEM][{label}] available_gb={avail_gb:.2f} floor_gb={floor_gb:.2f} gpu={gpu_gb}", flush=True)
    if avail_gb < floor_gb:
        raise RuntimeError(
            f"[MEM][CRITICAL] available memory {avail_gb:.2f}GB < {floor_gb:.2f}GB floor -- "
            "stopping to protect the shared live-trading server (per the documented prior outage)."
        )
    return {"label": label, "available_gb": avail_gb, "gpu": gpu_gb, "timestamp": time.time()}


class DGN(nn.Module):
    """Deep Gittins Network (paper Section IV): input = concat(s, x) (state being valued, reference
    state whose indifference point is being tracked), 3 hidden layers (64,128,64), ReLU, scalar
    output Q^x(s, continue). forward(s, x) supports either matched batches (s,x same leading dim) or
    the caller pre-broadcasting to a (B,B,2*D) shape before flattening -- this module only does the
    linear algebra, batching/broadcasting is the training loop's responsibility (kept explicit rather
    than hidden in a *args-shaped forward, per CLAUDE.md simplicity guidance)."""

    def __init__(self, state_dim: int, hidden: tuple[int, int, int] = ARCH_HIDDEN) -> None:
        super().__init__()
        h1, h2, h3 = hidden
        self.state_dim = int(state_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, 1),
        )

    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, x], dim=-1)).squeeze(-1)

    def pairwise(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """s: (B,D), x: (B,D) -> (B,B) matrix of Q(s_k, x_j) for all k,j pairs."""
        b = s.shape[0]
        s_exp = s.unsqueeze(1).expand(b, b, s.shape[1]).reshape(b * b, -1)
        x_exp = x.unsqueeze(0).expand(b, b, x.shape[1]).reshape(b * b, -1)
        return self.forward(s_exp, x_exp).reshape(b, b)

    def diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """M(x) := Q(x,x) for a batch of reference states -- the Eq-11 retirement-value read."""
        return self.forward(x, x)


def _build_transitions(frame_exit: pd.DataFrame) -> dict[str, np.ndarray]:
    """Reconstruct per-candidate SEQUENTIAL (s_t -> s_{t+1}) trajectories from the live-ATR-barrier
    dataset's own bookkeeping columns (exit_path_entry_i uniquely identifies one candidate's forward-
    simulated path, exit_path_hold_bars is that path's own bar offset -- both produced, unmodified,
    by research_eth_omega461_exit_head_liveatr_relabel_20260813._build_exit_dataset_entry_label_live_
    atr_barrier). EVERY row becomes exactly one transition (either non-terminal with a valid next-row
    index within the same candidate, or terminal at that candidate's own barrier resolution bar).
    Reward is the bar's own incremental contribution to exit_path_unrealized (already computed by the
    SAME dataset builder using the fixed BASE_TEMPLATE notional -- not re-derived here)."""
    n = len(frame_exit)
    entry_i = frame_exit["exit_path_entry_i"].to_numpy(dtype=np.int64)
    hold_bars = frame_exit["exit_path_hold_bars"].to_numpy(dtype=np.int64)
    unreal = frame_exit["exit_path_unrealized"].to_numpy(dtype=np.float64)

    order = np.lexsort((hold_bars, entry_i))  # sort by candidate, then within-candidate by hold_bars
    s_idx = np.empty(n, dtype=np.int64)
    r = np.empty(n, dtype=np.float64)
    s_next_idx = np.empty(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)

    prev_unreal = 0.0
    prev_entry = None
    group_start = 0
    for pos in range(n):
        row = order[pos]
        cur_entry = entry_i[row]
        if cur_entry != prev_entry:
            prev_unreal = 0.0
            group_start = pos
        s_idx[pos] = row
        r[pos] = unreal[row] - prev_unreal
        prev_unreal = unreal[row]
        prev_entry = cur_entry
        # terminal iff this is the last row of its candidate group (next order-position belongs to a
        # different candidate, or this is the very last row overall)
        is_last = (pos + 1 >= n) or (entry_i[order[pos + 1]] != cur_entry)
        if is_last:
            done[pos] = True
            s_next_idx[pos] = row  # dummy, unused (masked by done)
        else:
            s_next_idx[pos] = order[pos + 1]
    del group_start  # bookkeeping only, not needed downstream

    n_candidates = int(np.sum(done))
    return {
        "s_idx": s_idx, "reward": r.astype(np.float32), "s_next_idx": s_next_idx, "done": done,
        "entry_i": entry_i, "n_candidates": n_candidates,
    }


def _candidate_split(entry_i: np.ndarray, *, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Candidate-level (not row-level) train/val split -- every transition from one candidate's path
    stays entirely on one side, so the held-out TD-loss diagnostic never leaks a partially-seen
    trajectory. Returns boolean masks over the ROW/transition axis (same length as entry_i)."""
    uniq = np.unique(entry_i)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(round(len(uniq) * float(val_frac))))
    val_candidates = set(uniq[perm[:n_val]].tolist())
    is_val = np.array([e in val_candidates for e in entry_i], dtype=bool)
    return ~is_val, is_val


def _fit_expert(
    *,
    state_mat: torch.Tensor,
    transitions: dict[str, np.ndarray],
    route_w: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    seed: int,
    steps: int,
    batch_size: int,
    log_every: int,
    device: torch.device,
) -> tuple[DGN, DGN, dict[str, Any]]:
    torch.manual_seed(int(seed))
    state_dim = state_mat.shape[1]
    online = DGN(state_dim).to(device)
    target = copy.deepcopy(online).to(device)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=1.0e-3)

    s_idx_t = torch.from_numpy(transitions["s_idx"]).to(device)
    s_next_idx_t = torch.from_numpy(transitions["s_next_idx"]).to(device)
    reward_t = torch.from_numpy(transitions["reward"]).to(device)
    done_t = torch.from_numpy(transitions["done"]).to(device)
    route_w_t = torch.from_numpy(route_w.astype(np.float32)).to(device)

    train_idx = np.nonzero(train_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
    rng = np.random.default_rng(int(seed) + 1)

    def _step_loss(batch_row: np.ndarray, *, train: bool) -> torch.Tensor:
        b = torch.from_numpy(batch_row).to(device)
        s = state_mat.index_select(0, s_idx_t.index_select(0, b))
        s_next = state_mat.index_select(0, s_next_idx_t.index_select(0, b))
        r = reward_t.index_select(0, b)
        d = done_t.index_select(0, b)
        w = route_w_t.index_select(0, b)
        x = s  # diagonal reference-state choice: this minibatch's own states (see module docstring #1)
        with torch.no_grad():
            m_x = target.diagonal(x)  # (B,) -- Eq-11 M(x) read via the target net
            q_next = target.pairwise(s_next, x)  # (B,B) -- Q_theta'(s'_k, x_j)
            bootstrap = torch.maximum(q_next, m_x.unsqueeze(0))  # broadcast M(x_j) across rows k
            not_done = (~d).float().unsqueeze(1)
            tgt = r.unsqueeze(1) + GAMMA * bootstrap * not_done  # (B,B), Eq-9 target
        pred = online.pairwise(s, x) if train else online.pairwise(s, x).detach()
        weight = torch.outer(w, w)
        denom = torch.clamp(weight.sum(), min=1.0e-8)
        return (weight * (pred - tgt) ** 2).sum() / denom

    train_losses: list[float] = []
    val_losses: list[dict[str, float]] = []
    t0 = time.time()
    for step in range(int(steps)):
        batch_row = rng.choice(train_idx, size=min(int(batch_size), len(train_idx)), replace=len(train_idx) < batch_size)
        opt.zero_grad(set_to_none=True)
        loss = _step_loss(batch_row, train=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p_online, p_target in zip(online.parameters(), target.parameters()):
                p_target.mul_(1.0 - TARGET_TAU).add_(p_online, alpha=TARGET_TAU)
        train_losses.append(float(loss.detach().cpu()))
        if (step + 1) % int(log_every) == 0 or step == 0:
            val_loss = None
            if len(val_idx) > 0:
                with torch.no_grad():
                    vb = rng.choice(val_idx, size=min(int(batch_size), len(val_idx)), replace=len(val_idx) < batch_size)
                    val_loss = float(_step_loss(vb, train=False).detach().cpu())
                val_losses.append({"step": step + 1, "val_loss": val_loss})
            elapsed = time.time() - t0
            print(
                f"    step={step + 1}/{steps} train_loss={train_losses[-1]:.6f} "
                f"val_loss={val_loss} elapsed={elapsed:.1f}s steps_per_sec={(step + 1) / max(elapsed, 1e-6):.2f}",
                flush=True,
            )

    diag = {
        "steps": int(steps), "batch_size": int(batch_size), "elapsed_sec": time.time() - t0,
        "final_train_loss": float(np.mean(train_losses[-max(1, int(log_every)):])),
        "final_val_loss": val_losses[-1]["val_loss"] if val_losses else None,
        "val_loss_curve": val_losses,
        "n_train_transitions": int(len(train_idx)), "n_val_transitions": int(len(val_idx)),
    }
    return online, target, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES_DEFAULT)
    ap.add_argument("--steps", type=int, default=6000, help="gradient steps PER regime expert")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--skip-reference-check", action="store_true")
    args = ap.parse_args()

    device = parent._device(str(args.device))
    print(f"stage=start device={device}", flush=True)
    _mem_check("start")

    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    (out_dir / "h48qual").mkdir(parents=True, exist_ok=True)

    print("stage=build_dataset (reusing train_eval_omega461_gbdt_exit_head_liveatr_20260813._build_dataset unchanged)", flush=True)
    x_exit_raw, y_exit, frame_exit, exit_diag = gbdt_train._build_dataset(int(args.max_candidates))
    del y_exit  # binary exit label not used by this regression-target design; kept only via exit_diag for provenance

    reference_check: dict[str, Any] | None = None
    if int(args.max_candidates) == MAX_CANDIDATES_DEFAULT and REFERENCE_REPORT.exists() and not args.skip_reference_check:
        ref = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))["dataset"]
        reference_check = {
            "rows_match": int(exit_diag["rows"]) == int(ref["rows"]),
            "positive_count_match": int(exit_diag["positive_count"]) == int(ref["positive_count"]),
            "used_candidates_match": int(exit_diag["used_candidates"]) == int(ref["used_candidates"]),
            "rebuilt_rows": int(exit_diag["rows"]), "reference_rows": int(ref["rows"]),
        }
        print(f"stage=dataset_reference_check {reference_check}", flush=True)
        if not (reference_check["rows_match"] and reference_check["positive_count_match"]):
            print("WARNING: rebuilt dataset does NOT match the original full1500 TabM run's report.json.", flush=True)

    baseline_bundle_path = h48cons.sweep.COMPONENTS["h48qual"]["bundle"]
    base_cols = list(torch.load(baseline_bundle_path, map_location="cpu", weights_only=False)["base_cols"])
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)  # columns = base_cols + POS_COLS
    all_cols = list(base_cols) + list(parent.POS_COLS)
    route_probs = parent._route_probs(frame_exit)  # (n,3), bull/bear/chop
    _mem_check("after_dataset_build")

    print("stage=build_transitions", flush=True)
    transitions = _build_transitions(frame_exit)
    print(
        f"  transitions={len(transitions['s_idx'])} candidates={transitions['n_candidates']} "
        f"terminal_rows={int(transitions['done'].sum())}",
        flush=True,
    )
    train_mask, val_mask = _candidate_split(transitions["entry_i"], val_frac=VAL_HOLDOUT_FRAC, seed=SEED)
    print(f"  candidate-level split: train_transitions={int(train_mask.sum())} val_transitions={int(val_mask.sum())}", flush=True)

    print("stage=standardize", flush=True)
    x_arr = x_exit.to_numpy(dtype=np.float64)
    train_rows = np.nonzero(train_mask)[0]
    mean = x_arr[train_rows].mean(axis=0)
    std = x_arr[train_rows].std(axis=0)
    std[std < 1.0e-6] = 1.0
    x_std = ((x_arr - mean) / std).astype(np.float32)
    if not np.isfinite(x_std).all():
        raise RuntimeError("non-finite standardized Gittins state matrix")
    state_mat = torch.from_numpy(x_std).to(device)
    print(f"  state_mat shape={tuple(state_mat.shape)}", flush=True)
    _mem_check("after_standardize")

    results: dict[str, Any] = {}
    models_state: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        print(f"stage=fit_expert expert={expert}", flush=True)
        route_w = route_probs[:, idx]
        online, target_net, diag = _fit_expert(
            state_mat=state_mat, transitions=transitions, route_w=route_w,
            train_mask=train_mask, val_mask=val_mask, seed=SEED + idx,
            steps=int(args.steps), batch_size=int(args.batch_size), log_every=int(args.log_every), device=device,
        )
        print(f"  {expert}: {diag['final_train_loss']=:.6f} {diag['final_val_loss']=} elapsed={diag['elapsed_sec']:.1f}s", flush=True)
        # Deploy the TARGET net (the slow-tracked, more stable M(x) estimator -- see module docstring
        # simplification #2), not the faster-moving online net.
        models_state[expert] = {k: v.detach().cpu() for k, v in target_net.state_dict().items()}
        results[expert] = diag
        _mem_check(f"after_fit_expert_{expert}")

    bundle_path = out_dir / "h48qual" / "gittins_retirement_bundle.pt"
    torch.save(
        {
            "model_id": MODEL_ID, "models": models_state,
            "base_cols": base_cols, "pos_cols": list(parent.POS_COLS), "all_cols": all_cols,
            "scaler": {"columns": all_cols, "mean": mean.astype(np.float32), "std": std.astype(np.float32)},
            "arch": {"hidden": list(ARCH_HIDDEN), "state_dim": int(state_mat.shape[1])},
            "gamma": GAMMA, "target_tau": TARGET_TAU,
            "baseline_bundle_base_cols_source": str(baseline_bundle_path),
        },
        bundle_path,
    )
    print(f"bundle={bundle_path}", flush=True)

    report = {
        "model_id": MODEL_ID,
        "paper_citation": "Dhankhar, Mishra, Bodas, arXiv:2405.01157 (v1 2024-05-02 / v4 2025-08-25), retirement formulation, QGI (tabular)/DGN (deep) algorithms",
        "design": (
            "DGN (paper Eq 9-11) adapted to this project's offline live-ATR-barrier candidate dataset "
            "(train_eval_omega461_gbdt_exit_head_liveatr_20260813._build_dataset, seed=260813, "
            "max_candidates=1500, UNCHANGED). State = 115-dim (102 base_cols + 13 pos_cols), identical "
            "contract to TabM/GBDT/TCN exit_head. Reward = bar-to-bar delta of exit_path_unrealized "
            "(same fixed-notional mark-to-market PnL already computed by the dataset builder). "
            "Reference-state set X = each minibatch's own states (diagonal choice, documented "
            "simplification #1). M_n(x) realized via a Polyak target network's own diagonal read "
            "(documented simplification #2, merges Eq-11's M-update with the standard DQN target-net "
            "mechanism). Per-regime (bull/bear/chop) experts trained on all transitions weighted by "
            "route probability (outer product over the (B,B) loss matrix's sample and reference axes). "
            "Deployed model = the TARGET net per expert (slow-tracked, stable M(x) estimator), not the "
            "online net. Full mechanism + reformulation justification: "
            "docs/experiments/eth_omega461_gittins_index_exit_head_20260814.md."
        ),
        "gamma": GAMMA, "target_tau": TARGET_TAU, "arch_hidden": list(ARCH_HIDDEN),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "device": str(device),
        "dataset": exit_diag,
        "dataset_reference_check": reference_check,
        "transitions": {
            "n_transitions": int(len(transitions["s_idx"])), "n_candidates": int(transitions["n_candidates"]),
            "n_terminal_rows": int(transitions["done"].sum()),
            "val_holdout_frac_candidate_level": VAL_HOLDOUT_FRAC,
            "n_train_transitions": int(train_mask.sum()), "n_val_transitions": int(val_mask.sum()),
        },
        "expert_diagnostics": results,
        "bundle_path": str(bundle_path),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={out_dir / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
