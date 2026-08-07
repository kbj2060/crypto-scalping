"""Kronos fine-tune CONTEXT-WINDOW sweep (2026-08-07).

Motivation: the Stage 1 design fixed window=512 bars (42.7h) purely because that is Kronos-small's
max_context -- but the label is "zigzag pivot within 24 bars" (2h ahead). A 512-bar context may be
mostly irrelevant history diluting the signal. Sweep shorter contexts to find out.

SELECTION DISCIPLINE (this is the whole point -- pre-registered before running):
- The window is chosen on the INTERNAL TAIL ONLY (2025-07-01..2025-08-31, the same 12k fixed subset
  Stage 1 uses for checkpoint selection). The real VAL (2025-09..12) and OOS (2026Q1) are NOT
  computed here at all -- this script never even loads them into a metric. Sweeping 5 windows and
  then reading VAL/OOS for each would be a 5x multiple-comparisons hole of exactly the kind that
  produced this repo's 5 VAL/OOS decorrelation reproductions.
- Only the single tail-best window is then handed to the untouched Stage 1 gate
  (research_btc_kronos_layerA_v2_stage1_finetune_20260807.py, --window), which keeps its original
  pre-registered rule: LGBM[110 + ft_prob] vs paired LGBM[110], VAL AND OOS AUC both >= +0.005.

EQUAL-BUDGET FAIRNESS:
- Every window trains for exactly 1 epoch over the SAME row set (rows with >=512 bars of history,
  so shorter windows get no extra rows), same batch 64, same LRs, same seed, same schedule.
- The window=512 reference point comes from the already-completed first epoch of the Stage 1 run
  (best internal tail AUC 0.6553 at step 800; 0.6382/0.6397 after) -- recorded in
  tmp/btc_kronos_layerA_20260807/stage1_w512_1epoch_partial.log, not re-run.
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

from sklearn.metrics import roc_auc_score  # noqa: E402

from research_btc_kronos_layerA_v2_stage0_20260807 import build_layerA_dataset  # noqa: E402
from research_btc_kronos_layerA_v2_stage1_finetune_20260807 import (  # noqa: E402
    BACKBONE_LR,
    BATCH,
    GRAD_TRAIN_END,
    HEAD_LR,
    SEED,
    TAIL_END,
    TAIL_SUBSET,
    KronosTransitionFT,
    load_raw,
)

OUT_DIR = ROOT / "tmp/btc_kronos_layerA_20260807"
MIN_HISTORY = 512  # pin the row set to the 512 run's, so every window sees identical rows
WINDOWS = [256, 128, 64, 32]
EVAL_EVERY = 800
W512_REFERENCE = {"window": 512, "best_tail_auc": 0.6553, "evals": [0.6553, 0.6382, 0.6397],
                  "source": "stage1_w512_1epoch_partial.log (first epoch of the Stage 1 run)"}


def make_batch(x_all, stamp_all, pos_b, window):
    win = np.stack([x_all[p - window + 1:p + 1] for p in pos_b])
    mean = win.mean(axis=1, keepdims=True)
    std = win.std(axis=1, keepdims=True)
    win = np.clip((win - mean) / (std + 1e-5), -5.0, 5.0)
    stamp = np.stack([stamp_all[p - window + 1:p + 1] for p in pos_b])
    return torch.from_numpy(win), torch.from_numpy(stamp)


@torch.no_grad()
def tail_auc(model, tok, x_all, stamp_all, positions, y, device, window, batch_size=128):
    model.eval()
    probs = np.empty(len(positions), dtype=np.float32)
    for start in range(0, len(positions), batch_size):
        pos_b = positions[start:start + batch_size]
        x_t, s_t = make_batch(x_all, stamp_all, pos_b, window)
        with torch.autocast("cuda", dtype=torch.float16):
            s1, s2 = tok.encode(x_t.to(device), half=True)
            logit = model(s1, s2, s_t.to(device))
        probs[start:start + len(pos_b)] = torch.sigmoid(logit.float()).cpu().numpy()
    model.train()
    return float(roc_auc_score(y, probs))


def train_one_window(window, *, x_all, stamp_all, positions, y, grad_idx, tail_sub, device, tok):
    from model.kronos import Kronos

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    backbone = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
    model = KronosTransitionFT(backbone).to(device)
    used = [backbone.embedding, backbone.time_emb, backbone.transformer, backbone.norm]
    backbone_params = [p for m in used for p in m.parameters()]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": BACKBONE_LR}, {"params": model.head.parameters(), "lr": HEAD_LR}],
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda")
    pos_rate = float(y[grad_idx].mean())
    pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)], device=device)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(grad_idx)
    total_steps = math.ceil(len(order) / BATCH)
    warmup = 100
    best = -1.0
    evals: list[float] = []
    model.train()
    for step0, start in enumerate(range(0, len(order), BATCH)):
        step = step0 + 1
        idx_b = order[start:start + BATCH]
        x_t, s_t = make_batch(x_all, stamp_all, positions[idx_b], window)
        y_t = torch.from_numpy(y[idx_b].astype(np.float32)).to(device)
        lr_scale = min(1.0, step / warmup) * 0.5 * (1.0 + math.cos(math.pi * step0 / max(total_steps, 1)))
        for g, base_lr in zip(opt.param_groups, (BACKBONE_LR, HEAD_LR)):
            g["lr"] = base_lr * lr_scale
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            with torch.no_grad():
                s1, s2 = tok.encode(x_t.to(device), half=True)
            logit = model(s1, s2, s_t.to(device))
            loss = F.binary_cross_entropy_with_logits(logit, y_t, pos_weight=pos_weight)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % 400 == 0:
            print(f"  w={window} step={step}/{total_steps} loss={float(loss.detach()):.4f}", flush=True)
        if step % EVAL_EVERY == 0 or step == total_steps:
            auc = tail_auc(model, tok, x_all, stamp_all, positions[tail_sub], y[tail_sub], device, window)
            evals.append(auc)
            mark = ""
            if auc > best:
                best = auc
                torch.save({"state_dict": model.state_dict(), "step": step, "tail_auc": auc, "window": window},
                           OUT_DIR / f"kronos_ft_w{window}_best.pt")
                mark = " *saved*"
            print(f"  w={window} internal_tail AUC={auc:.4f} (best={best:.4f}){mark}", flush=True)
    del model, backbone, opt
    torch.cuda.empty_cache()
    return {"window": window, "best_tail_auc": best, "evals": evals, "steps": total_steps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="*", default=WINDOWS)
    args = ap.parse_args()
    from model.kronos import KronosTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dfA, _ = build_layerA_dataset()
    x_all, stamp_all, pos_map = load_raw()
    ts = pd.to_datetime(dfA["timestamp"])
    positions = np.array([pos_map[t] for t in ts], dtype=np.int64)
    keep = positions >= (MIN_HISTORY - 1)
    dfA = dfA.loc[keep].reset_index(drop=True)
    ts = ts[keep].reset_index(drop=True)
    positions = positions[keep]
    y = dfA["transition_soon"].astype(int).to_numpy()

    grad_idx = np.flatnonzero((ts < GRAD_TRAIN_END).to_numpy())
    tail_idx = np.flatnonzero(((ts >= GRAD_TRAIN_END) & (ts < TAIL_END)).to_numpy())
    tail_sub = np.random.default_rng(SEED).choice(tail_idx, size=min(TAIL_SUBSET, len(tail_idx)), replace=False)
    print(f"pinned row set: grad_train={len(grad_idx)} tail_subset={len(tail_sub)}", flush=True)

    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    results = [W512_REFERENCE]
    for w in args.windows:
        print(f"=== window {w} ===", flush=True)
        results.append(train_one_window(w, x_all=x_all, stamp_all=stamp_all, positions=positions, y=y,
                                        grad_idx=grad_idx, tail_sub=tail_sub, device=device, tok=tok))
        (OUT_DIR / "window_sweep_report.json").write_text(
            json.dumps({"selection_metric": "internal tail AUC only (VAL/OOS never computed here)",
                        "equal_budget": "1 epoch, batch 64, identical pinned row set, same seed",
                        "results": results}, indent=2) + "\n", encoding="utf-8")

    best = max(results, key=lambda r: r["best_tail_auc"])
    print("\n=== SWEEP SUMMARY (internal tail only) ===", flush=True)
    for r in sorted(results, key=lambda r: -r["best_tail_auc"]):
        print(f"  window={r['window']:>4}  best_tail_AUC={r['best_tail_auc']:.4f}  evals={[round(e,4) for e in r['evals']]}", flush=True)
    print(f"TAIL-BEST WINDOW = {best['window']} (AUC {best['best_tail_auc']:.4f}) -> hand this ONE to the Stage 1 gate", flush=True)
    report = {
        "selection_metric": "internal tail AUC only (VAL/OOS never computed here)",
        "equal_budget": "1 epoch, batch 64, identical pinned row set, same seed",
        "results": results,
        "tail_best_window": best["window"],
        "next": "run stage1 gate with --window <tail_best_window>; gate rule unchanged (VAL and OOS AUC both >= paired baseline + 0.005)",
    }
    (OUT_DIR / "window_sweep_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
