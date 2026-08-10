"""Zigzag-SEQUENCE transformer for the theta=0.5% regime nowcaster (2026-08-08).

Contract: docs/experiments/btc_regime_theta005_zigzag_sequence_transformer_20260808.json
(pre-registered before the first run; read it, do not re-derive the gates from this file).

QUESTION.  The frozen detector `btc_regime_theta005_zigzagonly_S2fine5_lam05` feeds 5 causal
zigzag DIRECTION states to LightGBM ONE BAR AT A TIME.  LightGBM has no temporal receptive field,
so it cannot represent "the 0.1% zigzag has flipped against the 0.5% one three times in the last
20 bars".  This script asks whether a small causal transformer over a WINDOW of the same states
extracts anything the per-bar view cannot.  Inputs stay pure zigzag geometry: no panel features,
no returns, no volume -- the panel was measured at -1.2pp OOS on this task.

WHY THIS IS NOT THE 2026-08-06 DEEPFEAT LINE.  That line ran a transformer over the 113-feature
PANEL against the 4%-scale zigzag soft label.  Here the inputs are 5-15 zigzag geometry channels
and the target is the 0.5% wave, a 16-bar nowcasting question in the one frame where supervised
learning has already beaten the no-learning baseline (czz05 61.3 -> lgbm_jm 66.0 VAL).

STRUCTURE.
  Stage 0   regression gate: reproduce the frozen LightGBM incumbent on identical rows/splits.
            If it does not come back at 70.1 / 68.0 (+/-0.3pp) the run HALTS -- without a
            reproduced incumbent no comparison in this file means anything.
  Stage 1   12-cell screen (window x n_layers x mode) on VAL at the inherited decode lambda=0.5.
  Stage 1b  the Stage-1 winner retrained over 5 randomly drawn seeds, probabilities averaged.
  Stage 2   lambda sweep on the bagged probabilities (post-processing, no retraining).
  OOS       exactly one read.  ADOPT only if VAL > 70.1 AND OOS >= 68.0.

Eligibility (coverage >= 50%, median run >= 8 bars) is the floor that has rejected every
confetti candidate this project produced -- notably the 1D-CNN zoo entry at 46k flips.  It is not
lowered after seeing results.

Early stopping monitors an INNER validation split carved from the end of TRAIN, never the project
VAL window: the incumbent trains a fixed 700 trees with no early stopping, so checkpoint-picking
on VAL would tilt the comparison toward the transformer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_btc_regime_classifier_lag_20260808 import (  # noqa: E402
    by_quintile, detection_lag, dir_of, wave_position,
)
from refine_btc_regime_classifier_theta005_20260808 import (  # noqa: E402
    MIN_COVERAGE, MIN_MEDIAN_RUN, PANEL_PATH, PURGE, SCORE_SCALES,
    jump_decode_proba, summarize, to_named,
)
from reselect_btc_regime_classifier_zigzag_only_20260808 import zigzag_geometry  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    SEED, TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/regime_zigzag_seqformer_20260808"
OUT_PARQUET = ROOT / "data/research/btc_regime_theta005_seqformer_20260808.parquet"
CONTRACT = ROOT / "docs/experiments/btc_regime_theta005_zigzag_sequence_transformer_20260808.json"

THETA = 0.005
THRESHOLDS = [0.001, 0.002, 0.0035, 0.005, 0.008]     # S2_fine5, the frozen config's set
INCUMBENT_VAL, INCUMBENT_OOS, REPRO_TOL = 70.1, 68.0, 0.3
STAGE1_LAM = 0.5
LAMBDAS = [0.25, 0.5, 1.0, 2.0]
WINDOWS = [32, 64, 128]
LAYERS = [1, 2]
MODES = ["state", "geo"]
D_MODEL = {1: 64, 2: 32}
TRAIN_STRIDE = 4
INNER_VAL_FRAC = 0.10
N_SEEDS = 5
SCREEN_SEED = 20260808
MAX_EPOCHS, PATIENCE, BATCH, LR, WD, DROPOUT, HEADS, CLIP = 20, 4, 512, 3e-4, 5e-4, 0.2, 4, 1.0

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ZigzagSeqFormer(nn.Module):
    """Pre-LN causal transformer encoder over a window of zigzag-geometry channels.

    Causality is already guaranteed by construction -- the window ends at bar t and every channel
    is an online state machine -- so the attention mask is an inductive bias (each position sees
    only its own past), not a leak guard.  Last-token pooling reads the representation at bar t.
    """

    def __init__(self, n_ch: int, window: int, d_model: int, n_layers: int):
        super().__init__()
        self.inp = nn.Linear(n_ch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, HEADS, d_model * 2, dropout=DROPOUT, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer("mask", torch.triu(torch.ones(window, window, dtype=torch.bool), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(self.inp(x) + self.pos, mask=self.mask)
        return self.head(self.norm(h[:, -1])).squeeze(-1)


def make_windows(feat: np.ndarray, window: int) -> np.ndarray:
    """(n, C) -> (n, C, W) strided VIEW where row t is the window ending at bar t (edge-padded)."""
    pad = np.repeat(feat[:1], window - 1, axis=0)
    arr = np.concatenate([pad, feat], axis=0)
    return np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)


def batch_of(wins: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(wins[idx].transpose(0, 2, 1)))


def train_one(wins, y_all, tr_idx, inner_idx, window, d_model, n_layers, seed) -> np.ndarray:
    """Train on tr_idx, early-stop on inner_idx, return p(bull) for every bar."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ZigzagSeqFormer(wins.shape[1], window, d_model, n_layers).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss()
    y_t = torch.from_numpy(y_all.astype(np.float32))

    xb_inner = batch_of(wins, inner_idx)
    yb_inner = y_t[inner_idx]
    best, best_state, bad = float("inf"), None, 0
    rng = np.random.default_rng(seed)
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = rng.permutation(len(tr_idx))
        for s in range(0, len(perm), BATCH):
            b = tr_idx[perm[s: s + BATCH]]
            xb = batch_of(wins, b).to(DEV, non_blocking=True)
            loss = lossf(model(xb), y_t[b].to(DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(inner_idx), 4096):
                xb = xb_inner[s: s + 4096].to(DEV)
                yb = yb_inner[s: s + 4096].to(DEV)
                tot += float(lossf(model(xb), yb)) * len(yb)
                cnt += len(yb)
        vl = tot / max(cnt, 1)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    out = np.empty(len(wins), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(wins), 4096):
            xb = batch_of(wins, np.arange(s, min(s + 4096, len(wins)))).to(DEV)
            out[s: s + 4096] = torch.sigmoid(model(xb)).float().cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "full"], default="full")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(json.dumps({"device": str(DEV), "contract": str(CONTRACT.relative_to(ROOT))}), flush=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    n = len(close)

    oracles = {t: zigzag_oracle(close, threshold=t)[0] for t in SCORE_SCALES}
    y_dir, pivots = zigzag_oracle(close, threshold=THETA)
    geo = {t: zigzag_geometry(close, t) for t in THRESHOLDS}

    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_all = np.flatnonzero(train_mask)[:-PURGE]
    tr_all = tr_all[y_dir[tr_all] != 0]
    v_idx = np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy())
    o_idx = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())
    windows_idx = {"val_2025Q4": v_idx, "oos_2026Q1": o_idx}
    y_all = (y_dir == 1).astype(np.int64)
    print(json.dumps({"n_bars": n, "train_rows": len(tr_all),
                      "bull_frac": round(float(y_all[tr_all].mean()), 3)}), flush=True)

    def decode(p: np.ndarray, lam: float) -> np.ndarray:
        return to_named(jump_decode_proba(p, lam))          # w=1.0: no vote blend

    # ---------------- Stage 0: reproduce the frozen incumbent ----------------
    lgb_seeds = sorted(int(s) for s in
                       np.random.default_rng(SEED + 1).choice(1_000_000, size=N_SEEDS, replace=False))
    x_state = np.column_stack([geo[t][0] for t in THRESHOLDS]).astype(np.float32)
    ps = []
    for s in lgb_seeds:
        clf = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05,
                                 num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                                 bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                                 random_state=s, n_jobs=-1, verbosity=-1)
        clf.fit(x_state[tr_all], y_all[tr_all])
        ps.append(clf.predict_proba(x_state)[:, 1])
    p_incumbent = np.mean(ps, axis=0)
    inc_state = decode(p_incumbent, STAGE1_LAM)
    inc = {w: summarize(inc_state, oracles, idx) for w, idx in windows_idx.items()}
    inc_val = inc["val_2025Q4"]["agree"]["0.005"]
    inc_oos = inc["oos_2026Q1"]["agree"]["0.005"]
    repro_ok = (abs(inc_val - INCUMBENT_VAL) <= REPRO_TOL) and (abs(inc_oos - INCUMBENT_OOS) <= REPRO_TOL)
    print(json.dumps({"STAGE0_incumbent": {"seeds": lgb_seeds, "val": inc_val, "oos": inc_oos,
                                           "expected": [INCUMBENT_VAL, INCUMBENT_OOS],
                                           "reproduced": repro_ok}}, indent=2), flush=True)
    if not repro_ok:
        (OUT_DIR / "results.json").write_text(json.dumps(
            {"halted": "STAGE 0 regression gate failed", "incumbent_measured": inc}, indent=2))
        return 1

    # ---------------- Stage 1: 12-cell screen ----------------
    feats = {
        "state": np.column_stack([geo[t][0] for t in THRESHOLDS]).astype(np.float32),
        "geo": np.column_stack([geo[t][0] for t in THRESHOLDS]
                               + [geo[t][1] for t in THRESHOLDS]
                               + [geo[t][2] for t in THRESHOLDS]).astype(np.float32),
    }
    for k, v in feats.items():                     # standardize on TRAIN rows only
        mu, sd = v[tr_all].mean(0), v[tr_all].std(0)
        feats[k] = ((v - mu) / np.where(sd < 1e-8, 1.0, sd)).astype(np.float32)

    cut = int(len(tr_all) * (1 - INNER_VAL_FRAC))
    inner_idx = tr_all[cut:]
    fit_idx = tr_all[:cut]
    fit_idx = fit_idx[fit_idx <= inner_idx[0] - PURGE][::TRAIN_STRIDE]
    print(json.dumps({"fit_windows": len(fit_idx), "inner_val_windows": len(inner_idx)}), flush=True)

    win_cache: dict[tuple[str, int], np.ndarray] = {}
    stage1: dict[str, dict] = {}
    for mode in MODES:
        for W in WINDOWS:
            key = (mode, W)
            if key not in win_cache:
                win_cache[key] = make_windows(feats[mode], W)
            for L in LAYERS:
                name = f"{mode}|W{W}|L{L}|d{D_MODEL[L]}"
                p = train_one(win_cache[key], y_all, fit_idx, inner_idx, W, D_MODEL[L], L, SCREEN_SEED)
                s = summarize(decode(p, STAGE1_LAM), oracles, v_idx)
                stage1[name] = {"val_agree": s["agree"]["0.005"], "coverage_pct": s["coverage_pct"],
                                "median_run_bars": s["median_run_bars"], "n_channels": feats[mode].shape[1]}
                print(f"  {name:22} VAL {s['agree']['0.005']}  cov {s['coverage_pct']}  "
                      f"run {s['median_run_bars']}  [{time.time() - t0:.0f}s]", flush=True)

    elig1 = {k: v for k, v in stage1.items()
             if v["coverage_pct"] >= MIN_COVERAGE and v["median_run_bars"] >= MIN_MEDIAN_RUN
             and v["val_agree"] is not None}
    win1 = max(elig1, key=lambda k: (elig1[k]["val_agree"], elig1[k]["coverage_pct"])) if elig1 else None
    print(json.dumps({"STAGE1_WINNER": win1, "n_eligible": len(elig1)}, indent=2), flush=True)

    out: dict = {"contract": str(CONTRACT.relative_to(ROOT)), "device": str(DEV),
                 "incumbent_reproduced": {"val": inc_val, "oos": inc_oos, "seeds": lgb_seeds},
                 "stage1": stage1, "stage1_winner": win1, "n_val_cells_scored": len(stage1)}
    if win1 is None or args.stage == "screen":
        out["halted"] = "no eligible Stage-1 cell" if win1 is None else "screen-only run"
        (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
        return 0 if win1 else 1

    # ---------------- Stage 1b: seed bag on the winner ----------------
    mode, Ws, Ls, _ = win1.split("|")
    W, L = int(Ws[1:]), int(Ls[1:])
    seeds = sorted(int(s) for s in
                   np.random.default_rng(SEED + 7).choice(1_000_000, size=N_SEEDS, replace=False))
    print(json.dumps({"stage1b_seeds": seeds}), flush=True)
    bag = [train_one(win_cache[(mode, W)], y_all, fit_idx, inner_idx, W, D_MODEL[L], L, s) for s in seeds]
    p_seq = np.mean(bag, axis=0)
    out["stage1b"] = {"seeds": seeds, "per_seed_val_agree": [
        summarize(decode(b, STAGE1_LAM), oracles, v_idx)["agree"]["0.005"] for b in bag]}
    print(json.dumps(out["stage1b"], indent=2), flush=True)

    # ---------------- Stage 2: lambda sweep ----------------
    stage2 = {}
    for lam in LAMBDAS:
        s = summarize(decode(p_seq, lam), oracles, v_idx)
        stage2[f"lam{lam:g}"] = {"val_agree": s["agree"]["0.005"], "coverage_pct": s["coverage_pct"],
                                 "median_run_bars": s["median_run_bars"]}
        print(f"  lam{lam:g}: VAL {s['agree']['0.005']}  cov {s['coverage_pct']}  run {s['median_run_bars']}",
              flush=True)
    elig2 = {k: v for k, v in stage2.items()
             if v["coverage_pct"] >= MIN_COVERAGE and v["median_run_bars"] >= MIN_MEDIAN_RUN
             and v["val_agree"] is not None}
    win2 = max(elig2, key=lambda k: (elig2[k]["val_agree"], elig2[k]["median_run_bars"])) if elig2 else None
    out.update({"stage2": stage2, "stage2_winner": win2,
                "n_val_cells_scored": len(stage1) + len(stage2)})
    if win2 is None:
        out["halted"] = "no eligible Stage-2 cell"
        (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
        return 1

    # ---------------- single OOS read ----------------
    lam_sel = float(win2[3:])
    final = decode(p_seq, lam_sel)
    rep = {w: summarize(final, oracles, idx) for w, idx in windows_idx.items()}
    val_a, oos_a = rep["val_2025Q4"]["agree"]["0.005"], rep["oos_2026Q1"]["agree"]["0.005"]
    adopt = bool(val_a > INCUMBENT_VAL and oos_a >= INCUMBENT_OOS
                 and rep["val_2025Q4"]["coverage_pct"] >= MIN_COVERAGE
                 and rep["val_2025Q4"]["median_run_bars"] >= MIN_MEDIAN_RUN)

    pos = wave_position(y_dir, pivots, n)
    secondary = {}
    for w, idx in windows_idx.items():
        secondary[w] = {
            "seqformer": {"quintile": by_quintile(dir_of(final), y_dir, pos, idx),
                          "detection_lag": detection_lag(dir_of(final), y_dir, pivots, idx[0], idx[-1])},
            "incumbent": {"quintile": by_quintile(dir_of(inc_state), y_dir, pos, idx),
                          "detection_lag": detection_lag(dir_of(inc_state), y_dir, pivots, idx[0], idx[-1])},
        }

    out.update({"final": {"mode": mode, "window": W, "n_layers": L, "d_model": D_MODEL[L],
                          "lambda": lam_sel, "thresholds": THRESHOLDS},
                "measured": rep, "secondary_not_selected_on": secondary,
                "adopt_rule": f"val > {INCUMBENT_VAL} AND oos >= {INCUMBENT_OOS}",
                "val_agreement_pct": val_a, "oos_agreement_pct": oos_a, "adopt": adopt,
                "runtime_sec": round(time.time() - t0)})
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"FINAL": out["final"], "val": val_a, "oos": oos_a,
                      "incumbent": {"val": inc_val, "oos": inc_oos}, "ADOPT": adopt}, indent=2), flush=True)

    pd.DataFrame({"timestamp": ts, "close": close, "oracle005": y_dir,
                  "p_seqformer": p_seq, "seqformer_final": final,
                  "p_incumbent": p_incumbent, "incumbent_final": inc_state}).to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
