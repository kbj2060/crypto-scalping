#!/usr/bin/env python3
"""RESEARCH ONLY -- C3 (Odyssey4 layer/parameter improvement proposal 20260816), the largest item.

CANDIDATE, not canonical: this script is a standalone architecture experiment. It does NOT modify
scripts/train_eval_omega1_2_tabm_3head_20260603.py (the canonical script -- A1's GCE port was
tested and REVERTED, see docs/experiments/eth_odyssey4_gce_canonical_port_20260816.md; the
canonical training loss is plain cross_entropy, unchanged) or trading_bot_modules/
odyssey_tabm_core.py (the live-deployed vendored copy, untouched). This candidate uses plain CE
throughout (direction/quality/exit) to match the real canonical baseline fairly.

Current architecture (canonical, "independent trunk" baseline): 3 completely separate
ThreeHeadTabM instances (bull/bear/chop), each with its OWN in_proj/blocks/norms trunk, trained one
at a time in main()'s `for idx, expert in enumerate(hard.EXPERT_NAMES)` loop via
_fit_expert_3head. B1's measurement (diagnose_odyssey4_expert_effective_sample_size_20260816.py)
found each expert's trunk sees the SAME raw row count (78568) but a much smaller EFFECTIVE (route_w
soft-weighted) sample count -- bull 28.6%, bear 28.0%, chop 43.4% -- i.e. every trunk is trained
mostly on down-weighted noise for its own regime, while carrying 3x the total trunk parameters of
a single model. This candidate is capacity-neutral-per-trunk and data-efficient (SAME trunk sees
ALL 78568 rows' gradient, just re-weighted 3 ways per batch instead of split into 3 totally
separate models) -- this data-efficiency argument stands on its own. (NOTE, 2026-08-16 correction:
an earlier draft framed this as "the reverse of the already-closed R+S+B completion failure", on
the theory that R+S+B failed from a capacity/memorization phase transition. A parallel session's
N>=5-seed reproduction of that axis found the true mechanism was NOT capacity/memorization --
baseline_R_only and full_R_S_B_embed's true accuracy ceilings differ by only 0.003-0.009; the real
gap was the embedding architecture's noisier early val_loss triggering premature epoch-1 stopping
~80% of the time vs ~20% for the plain architecture, i.e. a training-reliability gap, not a
capacity-vs-data gap. See feedback_modern_dl_training_checklist memory, 2026-08-16. This candidate
does not rest on that now-superseded framing.)

DESIGN CHOSEN: soft-routing-weighted multi-head loss over ONE shared trunk, not regime-embedding-
conditioned heads. Rationale: route_w (Regime3 HMM's continuous bull/bear/chop probability) is
ALREADY used as a per-bar sample weight multiplying each regime's classification loss --
_fit_expert_3head literally does `dir_w = balanced_weight * route_w[:, expert_idx]` today, just
inside 3 separate training calls. The most natural single-model generalization is to keep that
exact same route_w-weighted-loss mechanism, but compute encode(x) ONCE per batch and apply all 3
regimes' head sets to the SAME hidden representation h, backpropagating a single combined loss
(summed across regimes) through the shared trunk in one step. This requires zero change to what
route_w means or how it's computed -- only WHERE the loss terms land (one trunk instead of three).
The regime-embedding-conditioned-head alternative (concat route probs into head input) was not
chosen because it would require re-deriving a *new* routing mechanism instead of reusing the one
already validated live (Odyssey3/4's CONFIRMED entry-veto results all rest on this same Regime3
hard-argmax route_id -- eth_odyssey3_zig075_short_entry_veto_uptrend_confirmed / eth_odyssey4_..
_long_entry_veto_downtrend_confirmed), and because _routed()'s existing hard-argmax-select-by-regime
inference code (used everywhere downstream, including this script's own eval) already assumes
"one head-set is authoritative per bar" rather than a continuously blended head output -- the
soft-weighted-loss design lets training match that same semantics (route_w down-weights a regime's
own head loss on bars that aren't really that regime, while _routed() still hard-selects exactly
one head-set's PREDICTION per bar at eval/live time, unchanged from today).

Maximal reuse of the canonical script's already-correct downstream code, unmodified via import:
canon._routed (dict-based, doesn't care whether preds came from 3 models or 1 shared model),
canon._prediction_output, canon._to_decisions, canon.omega._metrics, canon._metrics_with_shared_exit
and canon._predict_loaded_exit (both take a plain `(model, scaler)` pair and only ever call
`model(x)["exit"]` -- a _RegimeView wrapper around the shared model duck-types as a drop-in
ThreeHeadTabM for exactly this purpose, so the entire PnL/MDD backtest loop is reused byte for byte,
not re-derived).

fresh_forward_bar_by_bar=n/a for the internal-split direction_balanced_accuracy comparison
(classifier training, matches this session's A1/C1 methodology); the val_raw/oos_raw PnL/MDD
backtest below IS a bar-by-bar walk (reuses canon._metrics_with_shared_exit's own causal loop) over
this project's standard VAL(2025-09-01..2025-12-31)/OOS(2026-01-01..2026-03-31)-adjacent
val_raw/oos_raw split as already defined by canon.SPLIT_TS/omega's frame split -- no saved trade
ledger or future row is used as model input; trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816 as gate  # noqa: E402
import eth_odyssey4_true_feature_pipeline_20260816 as truepipe  # noqa: E402

canon = gate.base  # train_eval_omega1_2_tabm_3head_20260603 (plain CE; A1's GCE port was reverted)
hard = gate.hard
CFG = canon.CFG

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_shared_trunk_regime_experts_20260816"
N_REGIMES = 3


def log(msg: str) -> None:
    print(f"[shared_trunk_regime_experts] {msg}", flush=True)


class SharedTrunkThreeHeadTabM(nn.Module):
    """Same encode() as canon.ThreeHeadTabM (in_proj/blocks/norms/BatchEnsemble k=8 gates,
    unchanged), but N_REGIMES separate direction/quality/exit head sets sharing that one trunk."""

    def __init__(self, n_features: int, *, cfg=CFG, n_regimes: int = N_REGIMES) -> None:
        super().__init__()
        self.k = int(cfg.k)
        self.n_features = int(n_features)
        self.n_regimes = int(n_regimes)
        self.input_scale = nn.Parameter(torch.randn(self.k, self.n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(self.k, self.n_features))
        self.in_proj = nn.Linear(self.n_features, int(cfg.hidden))
        self.blocks = nn.ModuleList(nn.Linear(int(cfg.hidden), int(cfg.hidden)) for _ in range(max(0, int(cfg.layers) - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(self.k, int(cfg.hidden)) * 0.03 + 1.0) for _ in range(max(0, int(cfg.layers) - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(int(cfg.hidden)) for _ in range(max(0, int(cfg.layers))))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.direction_heads = nn.ModuleList(nn.Linear(int(cfg.hidden), 3) for _ in range(self.n_regimes))
        self.quality_heads = nn.ModuleList(nn.Linear(int(cfg.hidden), 3) for _ in range(self.n_regimes))
        self.exit_heads = nn.ModuleList(nn.Linear(int(cfg.hidden), 2) for _ in range(self.n_regimes))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return h

    def forward_regime(self, x: torch.Tensor, regime_idx: int) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        return {
            "direction": self.direction_heads[regime_idx](h),
            "quality": self.quality_heads[regime_idx](h),
            "exit": self.exit_heads[regime_idx](h),
        }


class _RegimeView(nn.Module):
    """Duck-types as a ThreeHeadTabM for exactly one regime, so canon._predict_loaded_exit /
    canon._metrics_with_shared_exit (which only ever call `model(x)["exit"]`) work unmodified."""

    def __init__(self, shared: SharedTrunkThreeHeadTabM, regime_idx: int) -> None:
        super().__init__()
        self.shared = shared
        self.regime_idx = int(regime_idx)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.shared.forward_regime(x, self.regime_idx)


def _fit_shared_trunk(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, seed: int, epochs: int, device: torch.device, model_path: Path) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = canon._standardize_fit(x_all)
    x_dir_np = canon._standardize_apply(x_dir, scaler)
    x_exit_np = canon._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_probs = canon._route_probs(route_frame).astype(np.float32)  # (n, 3)
    exit_route_probs = canon._route_probs(exit_route_frame).astype(np.float32)
    dir_balanced = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32)
    ex_balanced = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32)
    dir_w = dir_balanced[:, None] * route_probs  # (n, 3) -- per-regime weight, same formula _fit_expert_3head uses per-expert
    ex_w = ex_balanced[:, None] * exit_route_probs

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = SharedTrunkThreeHeadTabM(x_dir_np.shape[1], cfg=CFG, n_regimes=N_REGIMES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    best_state = None
    best_loss = float("inf")
    best_per_regime = None
    stale = 0
    last_epoch = 0
    t0 = time.time()
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        for xb, yb, wb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)  # wb: (batch, 3)
            xe, ye, we = xe.to(device), ye.to(device), we.to(device)
            h_dir = model.encode(xb)  # shared trunk forward, ONCE per batch
            h_exit = model.encode(xe)
            loss_dir = torch.zeros((), device=device)
            loss_qual = torch.zeros((), device=device)
            loss_exit = torch.zeros((), device=device)
            for r in range(N_REGIMES):
                logits_dir_r = model.direction_heads[r](h_dir)
                logits_qual_r = model.quality_heads[r](h_dir)
                logits_exit_r = model.exit_heads[r](h_exit)
                l_dir_r = torch.nn.functional.cross_entropy(logits_dir_r.reshape(-1, 3), yb[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
                l_qual_r = torch.nn.functional.cross_entropy(logits_qual_r.reshape(-1, 3), yb[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
                l_exit_r = torch.nn.functional.cross_entropy(logits_exit_r.reshape(-1, 2), ye[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
                wr = wb[:, r]
                wer = we[:, r]
                loss_dir = loss_dir + (l_dir_r.mean(dim=1) * wr).sum() / torch.clamp(wr.sum(), min=1.0)
                loss_qual = loss_qual + (l_qual_r.mean(dim=1) * wr).sum() / torch.clamp(wr.sum(), min=1.0)
                loss_exit = loss_exit + (l_exit_r.mean(dim=1) * wer).sum() / torch.clamp(wer.sum(), min=1.0)
            loss = loss_dir + float(CFG.quality_loss_weight) * loss_qual + float(CFG.exit_loss_weight) * loss_exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            h_v = model.encode(vx)
            h_ve = model.encode(ve)
            vloss_total = 0.0
            per_regime = {}
            for r, expert in enumerate(hard.EXPERT_NAMES):
                vo_dir = model.direction_heads[r](h_v)
                vo_qual = model.quality_heads[r](h_v)
                vo_exit = model.exit_heads[r](h_ve)
                vdir = torch.nn.functional.cross_entropy(vo_dir.reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
                vqual = torch.nn.functional.cross_entropy(vo_qual.reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
                vex = torch.nn.functional.cross_entropy(vo_exit.reshape(-1, 2), vey[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
                wr = vw[:, r]
                wer = vew[:, r]
                vdir_loss = float(((vdir.mean(dim=1) * wr).sum() / torch.clamp(wr.sum(), min=1.0)).detach().cpu())
                vqual_loss = float(((vqual.mean(dim=1) * wr).sum() / torch.clamp(wr.sum(), min=1.0)).detach().cpu())
                vex_loss = float(((vex.mean(dim=1) * wer).sum() / torch.clamp(wer.sum(), min=1.0)).detach().cpu())
                dir_pred = torch.softmax(vo_dir, dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
                bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred))
                per_regime[expert] = {"direction_val_loss": vdir_loss, "quality_val_loss": vqual_loss, "exit_val_loss": vex_loss, "direction_balanced_accuracy": bacc}
                vloss_total += vdir_loss + float(CFG.quality_loss_weight) * vqual_loss + float(CFG.exit_loss_weight) * vex_loss
        if vloss_total + 1.0e-6 < best_loss:
            best_loss = vloss_total
            best_state = {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}
            best_per_regime = per_regime
            stale = 0
        else:
            stale += 1
            if stale >= int(CFG.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "model_id": "eth_odyssey4_shared_trunk_regime_experts_20260816",
        "config": CFG.__dict__,
        "state_dict": {k2: v.detach().cpu() for k2, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_dir_np.shape[1]),
        "n_params": int(n_params),
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(last_epoch),
        "per_regime_best_components": best_per_regime,
        "train_seconds": round(time.time() - t0, 1),
    }
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_shared_payload(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, dict[str, np.ndarray]]:
    model = SharedTrunkThreeHeadTabM(int(payload["n_features"]), cfg=CFG, n_regimes=N_REGIMES).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = canon._standardize_apply(x, payload["scaler"])
    chunks = {expert: {"direction": [], "quality": [], "exit": []} for expert in hard.EXPERT_NAMES}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        h = model.encode(xb)
        for r, expert in enumerate(hard.EXPERT_NAMES):
            chunks[expert]["direction"].append(torch.softmax(model.direction_heads[r](h), dim=-1).mean(dim=1).detach().cpu().numpy())
            chunks[expert]["quality"].append(torch.softmax(model.quality_heads[r](h), dim=-1).mean(dim=1).detach().cpu().numpy())
            chunks[expert]["exit"].append(torch.softmax(model.exit_heads[r](h), dim=-1).mean(dim=1).detach().cpu().numpy())
    return {expert: {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in d.items()} for expert, d in chunks.items()}


def _direction_val_bacc(payload: dict[str, Any], x_all: pd.DataFrame, y_all: np.ndarray, *, device: torch.device) -> float:
    """canon._fit_expert_3head doesn't track direction_balanced_accuracy itself (only
    best_validation_loss) -- compute it post-hoc on the SAME internal 85/15 val slice that
    function used internally (n=len(y_all), split=max(0.85n, min(n-1,512)), val=[split:]),
    reusing canon._predict_payload/payload['scaler'] unmodified."""
    n = len(y_all)
    split = max(int(n * 0.85), min(n - 1, 512))
    val_idx = np.arange(split, n)
    x_val = x_all.iloc[val_idx].reset_index(drop=True)
    preds = canon._predict_payload(payload, x_val, device=device)
    pred_dir = preds["direction"].argmax(axis=1)
    return float(balanced_accuracy_score(y_all[val_idx], pred_dir))


def _bacc_summary(baseline_bacc: dict[str, float] | None, payload_shared: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if baseline_bacc is not None:
        out["baseline_independent_trunk"] = baseline_bacc
    if payload_shared is not None:
        out["shared_trunk"] = {e: payload_shared["per_regime_best_components"][e]["direction_balanced_accuracy"] for e in hard.EXPERT_NAMES}
    return out


def run_one_seed(seed: int, *, epochs: int, device: torch.device, frames: dict[str, Any], base_cols: list[str], x_train, y_train, train_raw, x_exit, y_exit, frame_exit, fee: float, slip: float, run_baseline: bool, run_shared: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"seed": int(seed)}
    baseline_payloads: dict[str, Any] | None = None
    baseline_bacc: dict[str, float] | None = None
    shared_payload: dict[str, Any] | None = None

    if run_baseline:
        baseline_payloads = {}
        baseline_bacc = {}
        for idx, expert in enumerate(hard.EXPERT_NAMES):
            payload = canon._fit_expert_3head(
                x_train, y_train, train_raw, x_exit, y_exit, frame_exit,
                expert_idx=idx, seed=seed, epochs=epochs, device=device,
                model_path=OUT_DIR / "models" / f"seed{seed}_baseline_{expert}.pt",
            )
            baseline_payloads[expert] = payload
            baseline_bacc[expert] = _direction_val_bacc(payload, x_train, y_train, device=device)
            log(f"  seed={seed} baseline expert={expert}: epochs_ran={payload['epochs_ran']} best_val_loss={payload['best_validation_loss']:.4f} dir_bacc={baseline_bacc[expert]:.4f}")
        result["baseline_n_params_total"] = sum(sum(p.numel() for p in canon.ThreeHeadTabM(x_train.shape[1], cfg=CFG).parameters()) for _ in hard.EXPERT_NAMES)

    if run_shared:
        shared_payload = _fit_shared_trunk(
            x_train, y_train, train_raw, x_exit, y_exit, frame_exit,
            seed=seed, epochs=epochs, device=device,
            model_path=OUT_DIR / "models" / f"seed{seed}_shared_trunk.pt",
        )
        log(f"  seed={seed} shared_trunk: epochs_ran={shared_payload['epochs_ran']} best_val_loss={shared_payload['best_validation_loss']:.4f} n_params={shared_payload['n_params']}")
        result["shared_n_params_total"] = shared_payload["n_params"]

    result["direction_balanced_accuracy"] = _bacc_summary(baseline_bacc, shared_payload)

    # PnL/MDD via the canonical script's existing backtest metrics -- reused unmodified.
    if run_baseline and run_shared:
        val_raw, oos_raw = frames["val_raw"], frames["oos_raw"]
        loaded_baseline = canon._load_payloads(baseline_payloads, device=device)
        shared_model = SharedTrunkThreeHeadTabM(int(shared_payload["n_features"]), cfg=CFG, n_regimes=N_REGIMES).to(device)
        shared_model.load_state_dict(shared_payload["state_dict"])
        shared_model.eval()
        loaded_shared = {expert: (_RegimeView(shared_model, idx), shared_payload["scaler"]) for idx, expert in enumerate(hard.EXPERT_NAMES)}

        # canon.main()'s own predict_frame always uses the "..._oof" prefix (_to_decisions'
        # _tabm_prefix(oof) hardcodes the column-name prefix regardless of what string is passed
        # to _prediction_output's `prefix` kwarg) then, for oos, RENAMES the columns to strip
        # "_oof_" before calling _to_decisions(..., oof=False) -- mirrored exactly here so
        # _to_decisions finds the columns it expects for both val (oof=True) and oos (oof=False).
        PREFIX = "omega1_regime3_expertdq_oof"

        def predict_frame(frame: pd.DataFrame, payloads: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
            x = canon._base_input(frame, base_cols)
            preds = {expert: canon._predict_payload(payloads[expert], x, device=device) for expert in hard.EXPERT_NAMES}
            route = hard._route_id(frame)
            direction = canon._routed(preds, route, "direction", 3)
            quality = canon._routed(preds, route, "quality", 3)
            out = canon._prediction_output(frame, direction, quality, threshold=0.45, prefix=PREFIX)
            return x, out

        def predict_frame_shared(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            x = canon._base_input(frame, base_cols)
            preds = _predict_shared_payload(shared_payload, x, device=device)
            route = hard._route_id(frame)
            direction = canon._routed(preds, route, "direction", 3)
            quality = canon._routed(preds, route, "quality", 3)
            out = canon._prediction_output(frame, direction, quality, threshold=0.45, prefix=PREFIX)
            return x, out

        def _decisions(out: pd.DataFrame, *, is_val: bool) -> pd.DataFrame:
            if is_val:
                return canon._to_decisions(out, oof=True)
            renamed = out.rename(columns={c: c.replace(f"{PREFIX}_", "omega1_regime3_expertdq_") for c in out.columns})
            return canon._to_decisions(renamed, oof=False)

        backtests: dict[str, Any] = {}
        for split_name, frame in (("val", val_raw), ("oos", oos_raw)):
            is_val = split_name == "val"
            x_b, out_b = predict_frame(frame, baseline_payloads)
            x_s, out_s = predict_frame_shared(frame)
            dec_b = _decisions(out_b, is_val=is_val)
            dec_s = _decisions(out_s, is_val=is_val)
            m_b = canon._metrics_with_shared_exit(frame, x_b, dec_b, loaded_baseline, threshold=0.60, fee=fee, slip=slip, cost_mult=3.0, device=device)
            m_s = canon._metrics_with_shared_exit(frame, x_s, dec_s, loaded_shared, threshold=0.60, fee=fee, slip=slip, cost_mult=3.0, device=device)
            backtests[split_name] = {"baseline_independent_trunk": m_b, "shared_trunk": m_s}
            log(f"  seed={seed} {split_name}: baseline pnl={m_b['pnl']:.2f}% mdd={m_b['mdd']:.2f}% trades={m_b['trades']} | shared pnl={m_s['pnl']:.2f}% mdd={m_s['mdd']:.2f}% trades={m_s['trades']}")
        result["backtests"] = backtests

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--seeds", default="", help="comma-separated seeds; default draws N_SEEDS random seeds via secrets.randbelow")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--mode", choices=["baseline", "shared", "both"], default="both")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument(
        "--feature-pipeline", choices=["light", "true"], default="true",
        help="'true' (default) = eth_odyssey4_true_feature_pipeline_20260816.prepare_frames_true, "
        "the real live 102-base(+13pos)=115-feature contract (recovered 2026-08-16, supersedes the "
        "proxy). 'light' = gate._prepare_frames_light's 185-feature vsnlstm/chronos-bypass proxy "
        "(what this session's earlier B1/A1/C1/C2 diagnostics and this script's own local sanity "
        "check used, before the true pipeline existed) -- kept only for exact reproducibility of "
        "those earlier numbers, not recommended for new runs.",
    )
    args = ap.parse_args()

    global OUT_DIR
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{OUT_DIR.name}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR = out_dir

    if str(args.seeds).strip():
        seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    else:
        seeds = sorted(secrets.randbelow(900_000_000) + 100_000_000 for _ in range(int(args.n_seeds)))

    device = canon._device(str(args.device))
    log(f"=== stage=prepare_frames (feature_pipeline={args.feature_pipeline}) seeds={seeds} epochs={args.epochs} mode={args.mode} device={device} ===")
    frames = truepipe.prepare_frames_true() if args.feature_pipeline == "true" else gate._prepare_frames_light()
    fee, slip = canon.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = canon._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = gate.exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=60000,  # capped: dev box has 15GB RAM, unbounded build used ~13-14GB and tripped OOM once
    )
    x_exit = canon._exit_input_from_position_rows(x_exit_raw, base_cols)
    log(f"  n_train={len(x_train)} n_exit={len(x_exit)} n_features={x_train.shape[1]}")

    run_baseline = args.mode in ("baseline", "both")
    run_shared = args.mode in ("shared", "both")

    report: dict[str, Any] = {
        "design": "C3 -- shared trunk (encode()) + per-regime head sets, soft-routing-weighted combined loss, vs independent-trunk baseline.",
        "feature_pipeline": args.feature_pipeline,
        "seeds": seeds,
        "seed_source": "secrets.randbelow (genuinely random, not fixed-increment) unless --seeds given explicitly",
        "epochs_budget": int(args.epochs),
        "mode": args.mode,
        "results": [],
    }
    t_start = time.time()
    for seed in seeds:
        res = run_one_seed(seed, epochs=int(args.epochs), device=device, frames=frames, base_cols=base_cols, x_train=x_train, y_train=y_train, train_raw=train_raw, x_exit=x_exit, y_exit=y_exit, frame_exit=frame_exit, fee=fee, slip=slip, run_baseline=run_baseline, run_shared=run_shared)
        report["results"].append(res)
        log(f"  seed={seed} done, elapsed={time.time()-t_start:.0f}s")

    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
