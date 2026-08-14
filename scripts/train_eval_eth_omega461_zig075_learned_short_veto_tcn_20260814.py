#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey4: LEARNED zig075 SHORT entry veto (TCN, deep learning).

=== Task framing (user instruction, 2026-08-14) ===
The rule-based sustained-uptrend entry veto (Odyssey4 baseline, CONFIRMED) solves the 2025-Q3
zig075 SHORT damage with a hand-designed threshold rule. The user now asks for a LEARNED
(RL / deep-learning) solution instead of a fixed rule. This script trains that model and evaluates
it in the exact same veto slot, against both the no-veto (Odyssey3) baseline and the rule-veto
(Odyssey4) reference.

=== Why this formulation and not "learn direction" (which failed 29 times) ===
Odyssey1 settled (N>=5 seeds, 7+ model families incl. TCN) that direction skill is not learnable
from this feature universe. We therefore do NOT learn direction. We learn the VETO GATE:
  label[i] = "if a zig075-template SHORT were entered at bar i, does SL fire before TP?"
Because zig075's TP/SL are constants in every observed quarter (ATR floors saturate at
tp=0.075 / sl=0.040 price moves), this counterfactual outcome is a pure price-path property,
computable for EVERY bar (~100k dense labels/year) with no model in the loop. Decision-wise this
is the Q-function of a one-step contextual bandit (action = enter-short vs skip, fixed template):
supervised outcome prediction + greedy thresholding IS the RL solution of this decision problem,
with full counterfactual labels -- strictly better-posed than sparse-reward policy gradient on
13-35 trades/window (and the one prior Deep-RL attempt, the Gittins exit head, already failed).

=== Anti-overfit / anti-selection discipline (pre-registered) ===
- TRAIN = 2024 ONLY (data/splits/year_oos/training_features_2024.csv -- unused by the whole
  Omega4.6.1 lineage). ALL of 2025 (Q1/Q2/Q3/VAL) and 2026 (OOS) are genuine holdout. Q3 is
  therefore excluded from training/calibration BY CONSTRUCTION.
- Labels are computed strictly within 2024 prices; bars whose barrier does not resolve by
  2024-12-31 are censored and dropped => automatic embargo at the 2024/2025 boundary (no training
  label ever looks at 2025 prices).
- NO hyperparameter search: fixed HPs, reusing the repo's already-validated TCN architecture
  (verify_eth_h48qual_tcn_sequence_model_20260812) with a binary head. Fixed epochs, no early
  stopping (nothing is selected on any outcome anywhere before the gate).
- Veto threshold is NOT a free parameter: p_star = (TP - roundtrip_fee) / (TP + SL), the
  break-even probability of the fixed payoff structure ("veto iff E[trade return] < 0").
- N=5 truly random, pre-registered seeds. Deployed score = 5-seed ensemble mean probability;
  per-seed masks additionally evaluated on 2025q3+val for sign consistency (repo seed gate).
- Verdict: same multiwindow single-touch gate as every Odyssey candidate (VAL gate then
  OOS-Q1+OOS-Q2 together, with_gate PnL/MDD non-worse vs the Odyssey3 no-veto baseline).
  2025 quarters stay context tier. The rule-veto (Odyssey4) numbers are cited as a reference
  comparison row, not as the gate baseline.

fresh_forward_bar_by_bar=true for all replay evaluation (model probabilities are computed from
causal features only -- multi-scale trailing returns/vol + existing causal columns; the
forward-looking barrier labels are used for TRAINING on 2024 and for diagnostic AUC reporting on
holdout years, never as an input to any decision). trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/* / .env. Imports existing modules read-only.
Stages: --stage train (GPU-friendly; run on server via handoff.sh), --stage evaluate (CPU, dev),
--stage all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_zig075_learned_short_veto_tcn_20260814"
BASE_2024 = ROOT / "data/splits/year_oos/training_features_2024.csv"

# ---- fixed trade template (zig075 SHORT, ATR floors saturated in every observed quarter) ----
TP_MOVE = 0.075
SL_MOVE = 0.040

# ---- pre-registered training protocol (NO search) ----
SEEDS = [733421, 90817, 428003, 615229, 187561]  # truly scattered, fixed before any result
WINDOW = 288          # 1 day of 5m bars fed to the TCN
HIDDEN = 32
DILATIONS = (1, 2, 4, 8, 16)
EPOCHS = 12
BATCH = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
HOLDOUT_2024_START = "2024-11-01"  # diagnostic AUC only -- never used for stopping/selection

CHANNELS = [
    "log_ret_1", "log_ret_12", "log_ret_96", "log_ret_288", "log_ret_1008", "log_ret_2016",
    "rel_ret_2016", "dual_momentum", "rv_96", "rv_2016", "chop_index", "regime_persistence",
]


def log(msg: str) -> None:
    print(f"[learned_short_veto] {msg}", flush=True)


# =====================================================================================================
# Labels: counterfactual fixed-template SHORT barrier outcome per bar. Mirrors the replay's exit
# arithmetic exactly (close-based checks, slippage on both legs, TP checked before SL on the same
# bar). Censored (no barrier before end of array) -> NaN -> dropped.
# =====================================================================================================


def compute_short_barrier_labels(open_: np.ndarray, close: np.ndarray, slip_eff: float) -> np.ndarray:
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float64)
    entry_price = np.empty(n)
    entry_price[: n - 1] = open_[1:] * (1 - slip_eff)
    entry_price[n - 1] = np.nan
    # move_t = (entry - close_t*(1+slip)) / entry ; TP: move >= TP_MOVE ; SL: move <= -SL_MOVE
    lvl_tp = entry_price * (1 - TP_MOVE) / (1 + slip_eff)   # close_t <= lvl_tp  -> take_profit
    lvl_sl = entry_price * (1 + SL_MOVE) / (1 + slip_eff)   # close_t >= lvl_sl  -> stop_loss
    CHUNK = 4096
    for i in range(n - 2):
        t = i + 1
        while t < n:
            seg = close[t: min(t + CHUNK, n)]
            tp_hit = seg <= lvl_tp[i]
            sl_hit = seg >= lvl_sl[i]
            any_hit = tp_hit | sl_hit
            if any_hit.any():
                j = int(np.argmax(any_hit))
                # same-bar tie: replay checks take_profit first
                labels[i] = 0.0 if tp_hit[j] else 1.0
                break
            t += CHUNK
    return labels  # 1.0 = SL-first (short-hostile), 0.0 = TP-first, NaN = censored


# =====================================================================================================
# Features: multi-scale causal trend/vol channels. Generic ingredients (trailing log returns at
# several horizons, ETH-vs-BTC relative return, realized vol, plus existing causal columns) -- the
# model learns its own aggregation/threshold surface; the hand rule's exact rolling-fraction
# aggregation is deliberately NOT provided as an input.
# =====================================================================================================


def build_channels(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    lc = np.log(pd.to_numeric(df["close"], errors="raise"))
    lb = np.log(pd.to_numeric(df["close_btc"], errors="raise"))
    r1 = lc.diff(1)
    for k in (1, 12, 96, 288, 1008, 2016):
        out[f"log_ret_{k}"] = lc.diff(k)
    out["rel_ret_2016"] = lc.diff(2016) - lb.diff(2016)
    out["dual_momentum"] = pd.to_numeric(df["dual_momentum"], errors="raise")
    out["rv_96"] = r1.rolling(96, min_periods=96).std()
    out["rv_2016"] = r1.rolling(2016, min_periods=2016).std()
    out["chop_index"] = pd.to_numeric(df["chop_index"], errors="raise")
    out["regime_persistence"] = pd.to_numeric(df["regime_persistence"], errors="raise")
    return out[CHANNELS]


def load_year(csv_path: Path) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "close_btc", "dual_momentum", "chop_index", "regime_persistence"]
    df = pd.read_csv(csv_path, low_memory=False, usecols=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return df


# =====================================================================================================
# Model: repo-validated causal TCN (verify_eth_h48qual_tcn_sequence_model_20260812), binary head.
# =====================================================================================================


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.pad)

    def forward(self, x):
        y = self.conv(x)
        return y[:, :, : -self.pad] if self.pad else y


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, 3, dilation)
        self.conv2 = CausalConv1d(ch, ch, 3, dilation)
        self.norm1 = nn.GroupNorm(1, ch)
        self.norm2 = nn.GroupNorm(1, ch)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.act(self.norm2(self.conv2(y)))
        return x + y


class TCNVeto(nn.Module):
    def __init__(self, in_ch: int, hidden: int = HIDDEN, dilations=DILATIONS):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList([TCNBlock(hidden, d) for d in dilations])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B, C, T)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h[:, :, -1]).squeeze(-1)  # logit at the last (current) bar


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, window: int):
        self.X, self.y, self.idx, self.window = X, y, idx, window

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        return (torch.from_numpy(self.X[:, i - self.window + 1: i + 1].copy()).float(),
                torch.tensor(self.y[i], dtype=torch.float32))


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    rank = np.empty(len(score)); rank[order] = np.arange(1, len(score) + 1)
    pos = y_true > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((rank[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# =====================================================================================================
# Stage: train (2024 only)
# =====================================================================================================


def stage_train(device: torch.device) -> dict[str, Any]:
    import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
    import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

    fee, slip = omega._load_fee_slip()
    slip_eff = float(slip) * float(sweep.COST_MULT)

    log("=== stage=train: loading 2024 ===")
    df = load_year(BASE_2024)
    log(f"  2024 rows={len(df)} range={df['timestamp'].iloc[0]}..{df['timestamp'].iloc[-1]}")

    log("=== labels (counterfactual SHORT barrier, censored-at-year-end dropped) ===")
    labels = compute_short_barrier_labels(
        pd.to_numeric(df["open"], errors="raise").to_numpy(np.float64),
        pd.to_numeric(df["close"], errors="raise").to_numpy(np.float64), slip_eff)
    n_lab = int(np.isfinite(labels).sum())
    base_rate = float(np.nanmean(labels))
    log(f"  labeled={n_lab}/{len(labels)} censored={int(len(labels) - n_lab)} SL-first base rate={base_rate:.4f}")
    monthly = pd.Series(labels, index=df["timestamp"]).groupby(pd.Grouper(freq="MS")).mean()
    for ts, v in monthly.items():
        log(f"    {ts:%Y-%m} SL-first={v:.3f}")

    log("=== features ===")
    ch = build_channels(df)
    scaler_mean = ch.mean(skipna=True)
    scaler_std = ch.std(skipna=True).replace(0.0, 1.0)
    Xdf = (ch - scaler_mean) / scaler_std
    X = Xdf.to_numpy(np.float64).T  # (C, N)
    finite_row = np.isfinite(X).all(axis=0)
    valid = np.zeros(len(df), dtype=bool)
    # window fully finite AND label finite
    csum = np.cumsum(finite_row.astype(np.int64))
    for i in range(WINDOW - 1, len(df)):
        lo = i - WINDOW + 1
        if np.isfinite(labels[i]) and (csum[i] - (csum[lo - 1] if lo > 0 else 0)) == WINDOW:
            valid[i] = True
    idx_all = np.where(valid)[0]
    holdout_ts = pd.Timestamp(HOLDOUT_2024_START)
    is_holdout = df["timestamp"].to_numpy()[idx_all] >= np.datetime64(holdout_ts)
    idx_train, idx_hold = idx_all[~is_holdout], idx_all[is_holdout]
    log(f"  train samples={len(idx_train)}  holdout(diagnostic-only) samples={len(idx_hold)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Xf = X.astype(np.float32)
    seed_reports: dict[str, Any] = {}
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed % (2**32 - 1))
        model = TCNVeto(len(CHANNELS)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        lossf = nn.BCEWithLogitsLoss()
        ds = WindowDataset(Xf, labels, idx_train, WINDOW)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
        model.train()
        for ep in range(EPOCHS):
            tot, nb = 0.0, 0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward(); opt.step()
                tot += float(loss.item()); nb += 1
            log(f"  seed={seed} epoch={ep + 1}/{EPOCHS} loss={tot / max(nb, 1):.4f}")
        # diagnostic holdout AUC (Nov-Dec 2024) -- reported, never selected on
        model.eval()
        with torch.no_grad():
            probs = []
            hd = DataLoader(WindowDataset(Xf, labels, idx_hold, WINDOW), batch_size=BATCH)
            for xb, _ in hd:
                probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
        p_hold = np.concatenate(probs) if probs else np.array([])
        auc = _auc(labels[idx_hold], p_hold) if len(p_hold) else float("nan")
        log(f"  seed={seed} holdout(2024-11+) AUC={auc:.4f}")
        torch.save(model.state_dict(), OUT_DIR / f"tcn_veto_seed{seed}.pt")
        seed_reports[str(seed)] = {"holdout_auc_diagnostic": auc}

    train_report = {
        "train_year": "2024", "seeds": SEEDS, "window": WINDOW, "hidden": HIDDEN,
        "dilations": list(DILATIONS), "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "channels": CHANNELS, "label_base_rate_2024": base_rate,
        "label_monthly_sl_first": {f"{ts:%Y-%m}": (None if pd.isna(v) else float(v)) for ts, v in monthly.items()},
        "n_train_samples": int(len(idx_train)), "n_holdout_samples": int(len(idx_hold)),
        "scaler_mean": {c: float(scaler_mean[c]) for c in CHANNELS},
        "scaler_std": {c: float(scaler_std[c]) for c in CHANNELS},
        "hp_search_performed": False, "early_stopping": False,
        "per_seed": seed_reports,
    }
    (OUT_DIR / "train_report.json").write_text(json.dumps(train_report, indent=2) + "\n", encoding="utf-8")
    log(f"train_report={OUT_DIR / 'train_report.json'}")
    return train_report


# =====================================================================================================
# Stage: evaluate (dev, CPU) -- ensemble mask -> same veto replay slot, same gate.
# =====================================================================================================


def _predict_year(csv_path: Path, scaler_mean: pd.Series, scaler_std: pd.Series, device: torch.device) -> pd.DataFrame:
    df = load_year(csv_path)
    ch = build_channels(df)
    Xf = ((ch - scaler_mean) / scaler_std).to_numpy(np.float64).T.astype(np.float32)
    finite_row = np.isfinite(Xf).all(axis=0)
    csum = np.cumsum(finite_row.astype(np.int64))
    valid = np.zeros(len(df), dtype=bool)
    for i in range(WINDOW - 1, len(df)):
        lo = i - WINDOW + 1
        if (csum[i] - (csum[lo - 1] if lo > 0 else 0)) == WINDOW:
            valid[i] = True
    idx = np.where(valid)[0]
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    for seed in SEEDS:
        model = TCNVeto(len(CHANNELS))
        model.load_state_dict(torch.load(OUT_DIR / f"tcn_veto_seed{seed}.pt", map_location="cpu"))
        model.to(device).eval()
        probs = np.full(len(df), np.nan)
        with torch.no_grad():
            for lo in range(0, len(idx), 4096):
                sel = idx[lo: lo + 4096]
                xb = torch.stack([torch.from_numpy(Xf[:, i - WINDOW + 1: i + 1].copy()) for i in sel]).float().to(device)
                probs[sel] = torch.sigmoid(model(xb)).cpu().numpy()
        out[f"p_seed{seed}"] = probs
    out["p_ensemble"] = out[[f"p_seed{s}" for s in SEEDS]].mean(axis=1)
    return out


def stage_evaluate(device: torch.device) -> int:
    import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
    import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
    import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
    import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
    import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
    import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
    import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
    import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as vetomod  # noqa: E402

    fee, slip = omega._load_fee_slip()
    fee_eff = float(fee) * float(sweep.COST_MULT)
    slip_eff = float(slip) * float(sweep.COST_MULT)
    # break-even veto threshold from the payoff structure (pre-registered formula, no free param)
    p_star = (TP_MOVE - 2.0 * fee_eff) / (TP_MOVE + SL_MOVE)
    log(f"=== stage=evaluate: p_star(break-even)={p_star:.4f} (TP={TP_MOVE}, SL={SL_MOVE}, roundtrip fee={2 * fee_eff:.5f}) ===")

    train_report = json.loads((OUT_DIR / "train_report.json").read_text(encoding="utf-8"))
    scaler_mean = pd.Series(train_report["scaler_mean"])
    scaler_std = pd.Series(train_report["scaler_std"])

    log("=== inference over holdout years (2025, 2026) ===")
    preds = {
        sweep.BASE_2025: _predict_year(sweep.BASE_2025, scaler_mean, scaler_std, device),
        sweep.BASE_2026: _predict_year(sweep.BASE_2026, scaler_mean, scaler_std, device),
    }

    # diagnostic-only: barrier labels on holdout years for AUC reporting (forward-looking; never
    # used in any decision -- masks below are pure functions of causal probabilities)
    log("=== diagnostic labels on holdout years (report-only) ===")
    diag_auc: dict[str, Any] = {}
    for base_csv, name in ((sweep.BASE_2025, "2025"), (sweep.BASE_2026, "2026")):
        dfy = load_year(base_csv)
        lab = compute_short_barrier_labels(
            pd.to_numeric(dfy["open"], errors="raise").to_numpy(np.float64),
            pd.to_numeric(dfy["close"], errors="raise").to_numpy(np.float64), slip_eff)
        p = preds[base_csv]["p_ensemble"].to_numpy()
        ok = np.isfinite(lab) & np.isfinite(p)
        diag_auc[name] = {"n": int(ok.sum()), "auc_ensemble": _auc(lab[ok], p[ok]), "sl_first_rate": float(np.nanmean(lab))}
        # quarterly split for 2025
        if name == "2025":
            q = dfy["timestamp"].dt.quarter.to_numpy()
            for qq in (1, 2, 3, 4):
                sel = ok & (q == qq)
                diag_auc[f"2025q{qq}"] = {"n": int(sel.sum()), "auc_ensemble": _auc(lab[sel], p[sel]), "sl_first_rate": float(np.nanmean(lab[np.isfinite(lab) & (q == qq)]))}
        log(f"  {name}: {diag_auc[name]}")

    def mask_for_frame(aligned_frame: pd.DataFrame, wname: str, col: str) -> np.ndarray:
        base_csv = gate.WINDOW_DEFS[wname]["base_csv"]
        merged = aligned_frame[["timestamp"]].merge(preds[base_csv][["timestamp", col]], on="timestamp", how="left")
        if len(merged) != len(aligned_frame) or not merged["timestamp"].equals(aligned_frame["timestamp"]):
            raise RuntimeError(f"{wname}: prediction merge failed")
        return (merged[col] >= p_star).fillna(False).to_numpy(dtype=bool)

    report: dict[str, Any] = {
        "design": "Odyssey4 learned zig075 SHORT entry veto -- TCN trained on 2024-only counterfactual barrier labels, deployed at break-even threshold in the validated veto slot.",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "p_star": p_star, "train_report": train_report, "diagnostic_auc_holdout": diag_auc,
    }

    log("=== load windows + detector-free baseline (G0b re-run) ===")
    windows = gate.load_all_windows()
    score_by_base, _th, threshold = guard.build_detector()
    prepared: dict[str, tuple] = {}
    baseline_runs: dict[str, dict[str, Any]] = {}
    g0_ok = True
    for wname in gate.ALL_WINDOWS:
        prepared[wname] = guard.prepare_regime_aware_components(wname, windows, score_by_base, threshold, OUT_DIR, gate.DEVICE)
        aligned_frame, components, prep_diag = prepared[wname]
        diag, ledger = vetomod.greedy_replay_entry_veto(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=gate.DEVICE)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        ref_ng, ref_wg = vetomod.G0_ODYSSEY3[wname]
        ok = vetomod._close(no_gate, ref_ng) and vetomod._close(with_gate, ref_wg)
        g0_ok = g0_ok and ok
        baseline_runs[wname] = {"no_gate": no_gate, "with_gate": with_gate, "ledger": ledger}
        log(f"  G0 {wname:8s} no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d} with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d} match={ok}")
    report["g0_baseline_reproduction_pass"] = bool(g0_ok)
    if not g0_ok:
        report["gate_pass"] = False
        _write(report)
        log("ABORT: G0 baseline reproduction failed")
        return 1

    log("=== candidate: learned veto (ensemble) on all 6 windows ===")
    comparison: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        aligned_frame, components, _ = prepared[wname]
        m = mask_for_frame(aligned_frame, wname, "p_ensemble")
        cand_components = vetomod._attach_veto_mask(components, m)
        diag, ledger = vetomod.greedy_replay_entry_veto(aligned_frame, cand_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=gate.DEVICE)
        ledger.to_csv(OUT_DIR / f"portfolio_ledger_{wname}_learned_veto_ensemble.csv", index=False)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        diff = vetomod._ledger_diff(baseline_runs[wname]["ledger"], ledger)
        comparison[wname] = {
            "tier": gate.WINDOW_DEFS[wname]["tier"],
            "odyssey3_baseline": {"no_gate": baseline_runs[wname]["no_gate"], "with_gate": baseline_runs[wname]["with_gate"]},
            "learned_veto_ensemble": {"no_gate": no_gate, "with_gate": with_gate},
            "veto_bars": diag["veto_bars"], "mask_active_frac": float(m.mean()),
            "ledger_diff": {k: v for k, v in diff.items() if k in ("n_removed", "n_added", "removed_return_sum", "added_return_sum", "removed_trades", "added_trades")},
        }
        log(f"  {wname:8s} learned  no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d} with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d} mask_active={m.mean() * 100:5.1f}% veto_bars={diag['veto_bars']} removed={diff['n_removed']} added={diff['n_added']}")
    report["comparison"] = comparison

    log("=== per-seed sign consistency (2025q3 + val) ===")
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for wname in ("2025q3", "val"):
            aligned_frame, components, _ = prepared[wname]
            m = mask_for_frame(aligned_frame, wname, f"p_seed{seed}")
            cand_components = vetomod._attach_veto_mask(components, m)
            _diag, ledger = vetomod.greedy_replay_entry_veto(aligned_frame, cand_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=gate.DEVICE)
            wg = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
            per_seed[str(seed)][wname] = {"with_gate": wg, "mask_active_frac": float(m.mean())}
            log(f"  seed={seed} {wname:8s} with_gate={wg['pnl']:7.2f}%/{wg['mdd']:7.2f}%/{wg['trades']:3d} mask_active={m.mean() * 100:5.1f}%")
    report["per_seed_q3_val"] = per_seed
    q3_base_wg = baseline_runs["2025q3"]["with_gate"]["pnl"]
    seeds_improving_q3 = sum(1 for s in SEEDS if per_seed[str(s)]["2025q3"]["with_gate"]["pnl"] > q3_base_wg)
    val_base_wg = baseline_runs["val"]["with_gate"]["pnl"]
    seeds_nonworse_val = sum(1 for s in SEEDS if per_seed[str(s)]["val"]["with_gate"]["pnl"] >= val_base_wg - 1e-9)
    report["seed_consistency"] = {"seeds_improving_2025q3_with_gate": int(seeds_improving_q3), "seeds_nonworse_val_with_gate": int(seeds_nonworse_val), "n_seeds": len(SEEDS)}
    log(f"  seeds improving Q3 with_gate: {seeds_improving_q3}/{len(SEEDS)}; seeds non-worse VAL with_gate: {seeds_nonworse_val}/{len(SEEDS)}")

    log("=== verdict (vs Odyssey3 no-veto baseline) ===")
    baseline_tuples = {w: (baseline_runs[w]["no_gate"], baseline_runs[w]["with_gate"]) for w in gate.ALL_WINDOWS}
    candidate_tuples = {w: (comparison[w]["learned_veto_ensemble"]["no_gate"], comparison[w]["learned_veto_ensemble"]["with_gate"]) for w in gate.ALL_WINDOWS}
    summary_strict = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=0.0)
    summary_relaxed = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=3.0)
    # reference row: the CONFIRMED rule veto (Odyssey4 baseline), cited from its report for context
    rule_reference = {
        "2025q3": {"with_gate": {"pnl": 20.17, "mdd": -19.72, "trades": 17}},
        "2025q2": {"with_gate": {"pnl": 5.62, "mdd": -23.59, "trades": 19}},
        "note": "docs/experiments/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.md (CONFIRMED)",
    }
    report["summary"] = {
        "val_gate_pass_strict": bool(summary_strict["rows"]["val"]["with_gate_pass"]),
        "multiwindow_strict_mdd0pp": summary_strict,
        "multiwindow_relaxed_mdd3pp": summary_relaxed,
        "rule_veto_reference": rule_reference,
    }
    report["gate_pass"] = True
    _write(report)
    log(f"stage=done strict={summary_strict['final_verdict']} relaxed={summary_relaxed['final_verdict']} val_strict={report['summary']['val_gate_pass_strict']}")
    return 0


def _write(report: dict[str, Any]) -> None:
    import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    if args.stage in ("train", "all"):
        stage_train(device)
    if args.stage in ("evaluate", "all"):
        return stage_evaluate(torch.device("cpu") if device.type == "cuda" and args.stage == "all" else device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
