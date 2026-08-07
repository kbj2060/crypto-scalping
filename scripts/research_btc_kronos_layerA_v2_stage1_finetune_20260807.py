"""Kronos layerA v2 -- Stage 1: FINE-TUNED Kronos transition detector (2026-08-07).

Escalation from the closed Stage 0 (frozen embeddings: VAL +0.0095 / OOS +0.0006, redundant with
causalfix features). The Kronos paper's own recommended workflow is domain-adapted fine-tuning;
the frozen emb-only AUC 0.71 showed the representation carries real transition structure.

PRE-REGISTERED DESIGN (fixed before any result was seen):
- Backbone: Kronos-small, tokenizer FROZEN, decoder fine-tuned end-to-end with a linear
  classification head on the last-position hidden state (512-bar causal 5m windows, exact
  Stage 0 preprocessing: per-window z-norm, clip +-5, quote_volume as amount, time stamps).
- Label: transition_soon (same as layerA). Loss: BCE with pos_weight from the gradient-train
  base rate.
- Splits: gradient-train < 2025-07-01; internal checkpoint-selection tail 2025-07-01..2025-08-31
  (12k-row fixed random subset, AUC) -- the REAL VAL (2025-09..12) and OOS (2026Q1) are never
  touched during training and appear only in the final gate. The paired LGBM baseline keeps its
  full train (< 2025-09-01); the FT model accepting a 2-month data disadvantage is deliberate
  (no selection leakage).
- Budget: max 3 epochs (~7,500 steps at batch 64), AdamW backbone lr 3e-5 / head lr 1e-3,
  wd 0.01, 100-step linear warmup then cosine, fp16 autocast, best-internal-tail-AUC checkpoint.
- Deployment-shaped GATE (same margins as Stage 0): layerA v2 = LGBM on
  [110 existing features + kronos_ft_prob] vs paired LGBM [110] on identical rows.
  ACCEPT iff VAL AUC and OOS AUC BOTH improve >= +0.005. ft_prob-alone AUC is a diagnostic,
  not a selectable outcome. Single seed (20260807) for this gate; if it passes, an N>=3 seed
  stability check is required BEFORE any downstream chain rebuild (Stage 2, which has its own
  pre-registered worst-quarter rule).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tmp/kronos_vendor_20260807"))

from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from research_btc_kronos_layerA_v2_stage0_20260807 import (  # noqa: E402
    RAW_SOURCES,
    WINDOW,
    build_layerA_dataset,
)

OUT_DIR = ROOT / "tmp/btc_kronos_layerA_20260807"
CKPT_PATH = OUT_DIR / "kronos_ft_transition_best.pt"
PROB_PATH = OUT_DIR / "kronos_ft_prob.parquet"
SEED = 20260807
GRAD_TRAIN_END = "2025-07-01"
TAIL_END = "2025-09-01"
VAL_END = "2026-01-01"
OOS_END = "2026-04-01"
ACCEPT_MARGIN = 0.005
MAX_EPOCHS = 3
BATCH = 64
EVAL_EVERY = 800
TAIL_SUBSET = 12000
BACKBONE_LR = 3e-5
HEAD_LR = 1e-3


def load_raw():
    from model.kronos import calc_time_stamps

    raw = pd.concat(
        [pd.read_csv(p, usecols=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]) for p in RAW_SOURCES],
        ignore_index=True,
    )
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    x_all = raw[["open", "high", "low", "close", "volume", "quote_volume"]].to_numpy(dtype=np.float32)
    stamp_all = calc_time_stamps(raw["timestamp"]).values.astype(np.float32)
    pos_map = {t: i for i, t in enumerate(raw["timestamp"])}
    return x_all, stamp_all, pos_map


def make_batch(x_all, stamp_all, pos_b, window=WINDOW):
    win = np.stack([x_all[p - window + 1:p + 1] for p in pos_b])
    mean = win.mean(axis=1, keepdims=True)
    std = win.std(axis=1, keepdims=True)
    win = np.clip((win - mean) / (std + 1e-5), -5.0, 5.0)
    stamp = np.stack([stamp_all[p - window + 1:p + 1] for p in pos_b])
    return torch.from_numpy(win), torch.from_numpy(stamp)


class KronosTransitionFT(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.d_model, 1)

    def hidden(self, s1, s2, stamp):
        x = self.backbone.embedding([s1, s2]) + self.backbone.time_emb(stamp)
        for layer in self.backbone.transformer:
            x = layer(x)
        return self.backbone.norm(x)

    def forward(self, s1, s2, stamp):
        return self.head(self.hidden(s1, s2, stamp)[:, -1, :]).squeeze(-1)


@torch.no_grad()
def predict_probs(model, tok, x_all, stamp_all, positions, device, batch_size=96, window=WINDOW):
    model.eval()
    out = np.empty(len(positions), dtype=np.float32)
    for start in range(0, len(positions), batch_size):
        pos_b = positions[start:start + batch_size]
        x_t, s_t = make_batch(x_all, stamp_all, pos_b, window)
        x_t, s_t = x_t.to(device), s_t.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            s1, s2 = tok.encode(x_t, half=True)
            logit = model(s1, s2, s_t)
        out[start:start + len(pos_b)] = torch.sigmoid(logit.float()).cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true", help="reuse checkpoint, redo extraction+gate")
    ap.add_argument("--skip-extract", action="store_true", help="reuse prob parquet, redo gate only")
    # Context window. 512 was the original pre-registered choice (Kronos-small's max_context); the
    # 2026-08-07 sweep (research_btc_kronos_layerA_window_sweep_20260807.py) selects a window on the
    # INTERNAL TAIL ONLY and hands exactly one winner here -- the gate rule below is unchanged.
    ap.add_argument("--window", type=int, default=WINDOW)
    args = ap.parse_args()
    window = int(args.window)
    from model.kronos import Kronos, KronosTokenizer

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    dfA, feature_cols = build_layerA_dataset()
    x_all, stamp_all, pos_map = load_raw()
    ts = pd.to_datetime(dfA["timestamp"])
    positions = np.array([pos_map[t] for t in ts], dtype=np.int64)
    # Row set stays pinned to 512 bars of history regardless of --window, so the paired LGBM
    # baseline and the Stage 0 numbers remain row-for-row comparable across every window tried.
    keep = positions >= (WINDOW - 1)
    dfA = dfA.loc[keep].reset_index(drop=True)
    ts = ts[keep].reset_index(drop=True)
    positions = positions[keep]
    y = dfA["transition_soon"].astype(int).to_numpy()

    grad_mask = (ts < GRAD_TRAIN_END).to_numpy()
    tail_mask = ((ts >= GRAD_TRAIN_END) & (ts < TAIL_END)).to_numpy()
    print(f"rows: grad_train={grad_mask.sum()} internal_tail={tail_mask.sum()} total={len(dfA)}", flush=True)

    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    if not args.skip_train and not args.skip_extract:
        backbone = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
        model = KronosTransitionFT(backbone).to(device)
        used_backbone = [backbone.embedding, backbone.time_emb, backbone.transformer, backbone.norm]
        backbone_params = [p for m in used_backbone for p in m.parameters()]
        opt = torch.optim.AdamW(
            [{"params": backbone_params, "lr": BACKBONE_LR},
             {"params": model.head.parameters(), "lr": HEAD_LR}],
            weight_decay=0.01,
        )
        scaler = torch.amp.GradScaler("cuda")
        pos_rate = float(y[grad_mask].mean())
        pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)], device=device)
        print(f"grad-train pos_rate={pos_rate:.4f} pos_weight={float(pos_weight):.3f}", flush=True)

        rng = np.random.default_rng(SEED)
        tail_idx = np.flatnonzero(tail_mask)
        tail_sub = rng.choice(tail_idx, size=min(TAIL_SUBSET, len(tail_idx)), replace=False)
        grad_idx = np.flatnonzero(grad_mask)
        steps_per_epoch = math.ceil(len(grad_idx) / BATCH)
        total_steps = steps_per_epoch * MAX_EPOCHS
        warmup = 100
        best_tail_auc = -1.0
        step = 0
        print(f"steps_per_epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

        for epoch in range(MAX_EPOCHS):
            order = rng.permutation(grad_idx)
            model.train()
            for start in range(0, len(order), BATCH):
                idx_b = order[start:start + BATCH]
                x_t, s_t = make_batch(x_all, stamp_all, positions[idx_b], window)
                x_t, s_t = x_t.to(device), s_t.to(device)
                y_t = torch.from_numpy(y[idx_b].astype(np.float32)).to(device)
                lr_scale = min(1.0, (step + 1) / warmup) * 0.5 * (1.0 + math.cos(math.pi * step / max(total_steps, 1)))
                for g, base_lr in zip(opt.param_groups, (BACKBONE_LR, HEAD_LR)):
                    g["lr"] = base_lr * lr_scale
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    with torch.no_grad():
                        s1, s2 = tok.encode(x_t, half=True)
                    logit = model(s1, s2, s_t)
                    loss = F.binary_cross_entropy_with_logits(logit, y_t, pos_weight=pos_weight)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                step += 1
                if step % 200 == 0:
                    print(f"epoch={epoch} step={step}/{total_steps} loss={float(loss):.4f}", flush=True)
                if step % EVAL_EVERY == 0 or step == total_steps:
                    p_tail = predict_probs(model, tok, x_all, stamp_all, positions[tail_sub], device, window=window)
                    auc = float(roc_auc_score(y[tail_sub], p_tail))
                    marker = ""
                    if auc > best_tail_auc:
                        best_tail_auc = auc
                        torch.save({"state_dict": model.state_dict(), "step": step, "tail_auc": auc}, CKPT_PATH)
                        marker = " *saved*"
                    print(f"  internal_tail AUC={auc:.4f} (best={best_tail_auc:.4f}){marker}", flush=True)
                    model.train()
        print(f"training done; best internal tail AUC={best_tail_auc:.4f}", flush=True)

    if not args.skip_extract:
        backbone = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
        model = KronosTransitionFT(backbone).to(device)
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"loaded checkpoint step={ckpt['step']} tail_auc={ckpt['tail_auc']:.4f}", flush=True)
        probs = predict_probs(model, tok, x_all, stamp_all, positions, device, window=window)
        pd.DataFrame({"timestamp": ts, "kronos_ft_prob": probs}).to_parquet(PROB_PATH, index=False)
        print(f"saved {PROB_PATH}", flush=True)

    ft = pd.read_parquet(PROB_PATH)
    df = dfA.merge(ft, on="timestamp", how="inner").reset_index(drop=True)
    yy = df["transition_soon"].astype(int)
    tss = pd.to_datetime(df["timestamp"])
    tr = tss < TAIL_END  # LGBM baseline keeps its full train (< 2025-09-01)
    val = (tss >= TAIL_END) & (tss < VAL_END)
    oos = (tss >= VAL_END) & (tss < OOS_END)

    def run(cols, label):
        clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05,
                             min_child_samples=100, class_weight="balanced", verbosity=-1)
        clf.fit(df.loc[tr, cols].astype(np.float32), yy[tr])
        p = clf.predict_proba(df[cols].astype(np.float32))[:, 1]
        res = {f"{n}_auc": float(roc_auc_score(yy[m], p[m])) for n, m in [("VAL", val), ("OOS", oos)]}
        print(f"{label}: VAL AUC={res['VAL_auc']:.4f} | OOS AUC={res['OOS_auc']:.4f}", flush=True)
        return res

    base = run(feature_cols, "paired baseline [110]")
    aug = run(feature_cols + ["kronos_ft_prob"], "augmented [110 + kronos_ft_prob]")
    diag = {f"{n}_auc": float(roc_auc_score(yy[m], df.loc[m, "kronos_ft_prob"])) for n, m in [("VAL", val), ("OOS", oos)]}
    print(f"diagnostic ft_prob alone: VAL AUC={diag['VAL_auc']:.4f} | OOS AUC={diag['OOS_auc']:.4f}", flush=True)

    verdict = (aug["VAL_auc"] >= base["VAL_auc"] + ACCEPT_MARGIN and aug["OOS_auc"] >= base["OOS_auc"] + ACCEPT_MARGIN)
    report = {
        "design": "fine-tuned Kronos-small (frozen tokenizer) + linear head, kronos_ft_prob as 111th layerA feature",
        "seed": SEED,
        "accept_rule": f"VAL and OOS AUC both >= paired baseline + {ACCEPT_MARGIN}",
        "paired_baseline": base,
        "augmented": aug,
        "diagnostic_ft_prob_alone": diag,
        "verdict_pass": bool(verdict),
        "next_if_pass": "N>=3 seed stability check BEFORE Stage 2 downstream rebuild",
    }
    (OUT_DIR / "stage1_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("VERDICT:", "PASS -> seed stability check next" if verdict else "FAIL -> fine-tune axis closed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
