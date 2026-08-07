"""Regime-Curriculum Meta-Transformer: 3-stage training combining this session's two
oracle-validated BTC 5m labels (triple_barrier: 100% oracle win rate/44.3x OOS equity; zigzag:
73-75% oracle win rate) plus a 2025 finding (AEDL, Chen et al., MDPI Applied Sciences 15(24):13204)
that triple-barrier ALONE is a near-zero-Sharpe label across 16 assets/25 years, and that
multi-scale representation + meta-learning (MAML) for regime adaptation is what actually closes
that gap -- not more architecture complexity (their own ablation found REMOVING a causal-inference
component improved results, matching this session's repeated "simpler wins" pattern).

Two prior attempts THIS session already failed and inform this design:
- Joint multi-task training (one shared encoder, both heads trained together from scratch) hurt
  both heads via loss interference (train_btc_multitask_tb_zigzag_ensemble_20260806.py).
- Injecting raw zigzag pivot-tracker features into the triple-barrier input hurt backtest PnL
  (build_btc_5m_zigzag_state_causal_features_20260806.py).
This design avoids both failure modes: zigzag knowledge is transferred via SEQUENTIAL curriculum
pretraining (representation learning, not raw features or joint loss), and regime adaptation is
handled by a lightweight Reptile (first-order MAML) meta-learner over rolling weekly tasks, not a
single global weight set trained across BTC's wildly different train (+155%) / VAL (-18%) / OOS
(-24%) regimes.

Stage 1 (curriculum, easy): pretrain the shared SupervisedTransformerEncoder + a zigzag head on
the zigzag label (TRAIN split only).
Stage 2 (curriculum, hard): freeze the encoder's input projection, positional embedding, and first
transformer layer; fine-tune the remaining layer(s) + a fresh triple-barrier head on the
triple-barrier label (TRAIN split only).
Stage 3 (regime meta-learning): Reptile-style meta-training of ONLY the tb_head (kept small and
stable per common last-layer-only meta-adaptation practice) over rolling weekly tasks (2-week
support / 1-week query) drawn from TRAIN. At VAL/OOS evaluation, walk forward week by week: before
predicting/trading each week, adapt tb_head on the preceding 2 weeks (still strictly past data --
causal), then predict/trade that week with the adapted head.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402
from ensemble.deep_features.btc_deepfeat_encoders_20260806 import SupervisedTransformerEncoder  # noqa: E402
from core.causal_futures_backtest import simulate_single_position  # noqa: E402
from train_btc_deepfeat_encoders_20260806 import _prepare_target, _soft_ce_loss  # noqa: E402

TB_LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet"
ZZ_LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_DIR = ROOT / "tmp/btc_regime_curriculum_meta_transformer_20260807"

TB_CASH_WEIGHT = 0.9
ZZ_CASH_WEIGHT = 1.0
D_MODEL, N_HEADS, N_LAYERS, DROPOUT, EMBED_DIM = 96, 4, 3, 0.25, 32

WEEK_BARS = 7 * 24 * 12  # 2016 5m bars/week
SUPPORT_WEEKS = 2
INNER_STEPS = 5
INNER_LR = 5e-3
META_LR = 0.3
META_ITERS_PER_EPOCH = 60
META_EPOCHS = 6

TP_MULT, SL_MULT, HORIZON_BARS = 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_zigzag_aligned(ds) -> tuple[np.ndarray, np.ndarray]:
    zz = pd.read_parquet(ZZ_LABEL_PATH, columns=["timestamp", "zigzag_action", "zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"])
    zz = zz.sort_values("timestamp").reset_index(drop=True)
    if not (zz["timestamp"].to_numpy() == ds.timestamps_all).all():
        raise RuntimeError("zigzag label timestamps don't match the panel used by build_dataset")
    y_hard = zz["zigzag_action"].to_numpy(dtype=np.int64)
    y_soft = zz[["zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"]].to_numpy(dtype=np.float32)
    return y_hard, y_soft


def _iterate_batches(row_idx, batch_size, rng):
    idx = row_idx.copy()
    rng.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]


def _batch_loss(model, head, ds, y_hard_all, y_soft_all, chunk, device, cash_weight):
    x = torch.from_numpy(ds.get_batch(chunk)).to(device)
    soft = torch.from_numpy(y_soft_all[chunk]).to(device)
    hard = torch.from_numpy(y_hard_all[chunk]).to(device)
    target, weight = _prepare_target(soft, hard, 1.0, cash_weight)
    emb = model(x)
    logits = head(emb)
    return _soft_ce_loss(logits, target, weight), logits, hard


@torch.no_grad()
def _evaluate_head(model, head, ds, y_hard_all, y_soft_all, split, device, batch_size, cash_weight):
    model.eval()
    head.eval()
    row_idx = ds.end_idx[split]
    total_loss, correct, n = 0.0, 0, 0
    for i in range(0, len(row_idx), batch_size):
        chunk = row_idx[i : i + batch_size]
        loss, logits, hard = _batch_loss(model, head, ds, y_hard_all, y_soft_all, chunk, device, cash_weight)
        total_loss += float(loss.item()) * len(chunk)
        correct += int((logits.argmax(dim=-1) == hard).sum().item())
        n += len(chunk)
    return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1), "n": n}


def _train_stage(model, head, ds, y_hard_all, y_soft_all, device, epochs, patience, lr, wd, batch_size, seed, cash_weight, label):
    rng = np.random.default_rng(seed)
    params = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    best_loss, best_model_state, best_head_state, since_best = float("inf"), None, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        for chunk in _iterate_batches(ds.end_idx["train"], batch_size, rng):
            opt.zero_grad()
            loss, _, _ = _batch_loss(model, head, ds, y_hard_all, y_soft_all, chunk, device, cash_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        val_m = _evaluate_head(model, head, ds, y_hard_all, y_soft_all, "val", device, batch_size, cash_weight)
        scheduler.step(val_m["loss"])
        print(f"[{label}] epoch {epoch}: val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.4f}")
        if val_m["loss"] < best_loss - 1e-4:
            best_loss = val_m["loss"]
            best_model_state = copy.deepcopy(model.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                print(f"[{label}] early stop at epoch {epoch}")
                break

    model.load_state_dict(best_model_state)
    head.load_state_dict(best_head_state)
    return best_loss


def _week_task_rows(ds, start_row: int, end_row: int) -> np.ndarray:
    """Rows from ds.end_idx['train'] falling within [start_row, end_row)."""
    train_rows = ds.end_idx["train"]
    return train_rows[(train_rows >= start_row) & (train_rows < end_row)]


def _reptile_meta_train(model, tb_head, ds, device, seed):
    """Only tb_head is meta-adapted (kept small/stable). model (encoder) stays frozen during
    stage 3 -- it already carries the curriculum-learned representation from stages 1-2."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    rng = np.random.default_rng(seed)
    train_rows = ds.end_idx["train"]
    n_min, n_max = int(train_rows.min()), int(train_rows.max())
    support_bars = SUPPORT_WEEKS * WEEK_BARS

    task_starts = list(range(n_min + support_bars, n_max - WEEK_BARS, WEEK_BARS))
    print(f"[meta] {len(task_starts)} candidate weekly tasks in TRAIN")

    for epoch in range(1, META_EPOCHS + 1):
        rng.shuffle(task_starts)
        meta_loss_sum = 0.0
        n_tasks_done = 0
        for task_start in task_starts[:META_ITERS_PER_EPOCH]:
            support_rows = _week_task_rows(ds, task_start - support_bars, task_start)
            query_rows = _week_task_rows(ds, task_start, task_start + WEEK_BARS)
            if len(support_rows) < 32 or len(query_rows) < 8:
                continue

            fast_head = copy.deepcopy(tb_head)
            fast_opt = torch.optim.SGD(fast_head.parameters(), lr=INNER_LR)
            for _ in range(INNER_STEPS):
                chunk = rng.choice(support_rows, size=min(128, len(support_rows)), replace=False)
                fast_opt.zero_grad()
                with torch.no_grad():
                    x = torch.from_numpy(ds.get_batch(chunk)).to(device)
                    emb = model(x)
                soft = torch.from_numpy(ds.y_soft_all[chunk]).to(device)
                hard = torch.from_numpy(ds.y_hard_all[chunk]).to(device)
                target, weight = _prepare_target(soft, hard, 1.0, TB_CASH_WEIGHT)
                logits = fast_head(emb)
                loss = _soft_ce_loss(logits, target, weight)
                loss.backward()
                fast_opt.step()

            with torch.no_grad():
                q_chunk = query_rows[: min(256, len(query_rows))]
                x = torch.from_numpy(ds.get_batch(q_chunk)).to(device)
                emb = model(x)
                soft = torch.from_numpy(ds.y_soft_all[q_chunk]).to(device)
                hard = torch.from_numpy(ds.y_hard_all[q_chunk]).to(device)
                target, weight = _prepare_target(soft, hard, 1.0, TB_CASH_WEIGHT)
                q_logits = fast_head(emb)
                q_loss = _soft_ce_loss(q_logits, target, weight)
                meta_loss_sum += float(q_loss.item())
                n_tasks_done += 1

            # Reptile update: move meta-params toward the task-adapted params
            with torch.no_grad():
                for meta_p, fast_p in zip(tb_head.parameters(), fast_head.parameters()):
                    meta_p.add_(META_LR * (fast_p - meta_p))

        print(f"[meta] epoch {epoch}: mean_query_loss={meta_loss_sum / max(n_tasks_done, 1):.4f} n_tasks={n_tasks_done}")

    for p in model.parameters():
        p.requires_grad_(True)
    return tb_head


@torch.no_grad()
def _predict_with_head(model, head, ds, chunk, device):
    x = torch.from_numpy(ds.get_batch(chunk)).to(device)
    emb = model(x)
    logits = head(emb)
    return logits.argmax(dim=-1).cpu().numpy()


def _walk_forward_predict(model, meta_head, ds, split, device, rng):
    """For each week in `split`, adapt a COPY of meta_head on the preceding 2 weeks (causal --
    always strictly past data relative to the predicted week), predict/trade that week with the
    adapted head, then discard the adaptation (start fresh from meta_head next week)."""
    row_idx = ds.end_idx[split]
    if len(row_idx) == 0:
        return row_idx, np.zeros(0, dtype=np.int64)
    row_min, row_max = int(row_idx.min()), int(row_idx.max())
    support_bars = SUPPORT_WEEKS * WEEK_BARS

    all_train_and_split_rows = np.concatenate([ds.end_idx["train"], ds.end_idx["val"], ds.end_idx["oos"]])
    all_train_and_split_rows.sort()

    preds = np.zeros(len(row_idx), dtype=np.int64)
    week_starts = list(range(row_min, row_max + 1, WEEK_BARS))
    for w_start in week_starts:
        w_end = min(w_start + WEEK_BARS, row_max + 1)
        support_rows = all_train_and_split_rows[(all_train_and_split_rows >= w_start - support_bars) & (all_train_and_split_rows < w_start)]
        query_mask = (row_idx >= w_start) & (row_idx < w_end)
        query_rows = row_idx[query_mask]
        if len(query_rows) == 0:
            continue

        fast_head = copy.deepcopy(meta_head)
        if len(support_rows) >= 32:
            fast_opt = torch.optim.SGD(fast_head.parameters(), lr=INNER_LR)
            for _ in range(INNER_STEPS):
                chunk = rng.choice(support_rows, size=min(128, len(support_rows)), replace=False)
                fast_opt.zero_grad()
                with torch.no_grad():
                    x = torch.from_numpy(ds.get_batch(chunk)).to(device)
                    emb = model(x)
                soft = torch.from_numpy(ds.y_soft_all[chunk]).to(device)
                hard = torch.from_numpy(ds.y_hard_all[chunk]).to(device)
                target, weight = _prepare_target(soft, hard, 1.0, TB_CASH_WEIGHT)
                logits = fast_head(emb)
                loss = _soft_ce_loss(logits, target, weight)
                loss.backward()
                fast_opt.step()

        pred = _predict_with_head(model, fast_head, ds, query_rows, device)
        preds[query_mask] = pred

    return row_idx, preds


def _fresh_entry_mask(side_state: np.ndarray) -> np.ndarray:
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh


def _backtest(row_idx, pred, panel, tp_moves_all, sl_moves_all):
    side_state = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
    fresh = _fresh_entry_mask(side_state)
    idx, side = row_idx[fresh], side_state[fresh]
    tp, sl = tp_moves_all[idx], sl_moves_all[idx]
    finite = np.isfinite(tp) & np.isfinite(sl)
    idx, side, tp, sl = idx[finite], side[finite], tp[finite], sl[finite]
    if len(idx) == 0:
        return {"n_trades": 0}
    result = simulate_single_position(
        timestamps=panel["timestamp"], open_px=panel["open"].to_numpy(dtype=np.float64),
        high=panel["high"].to_numpy(dtype=np.float64), low=panel["low"].to_numpy(dtype=np.float64),
        close=panel["close"].to_numpy(dtype=np.float64), decision_indices=idx, scores=side.astype(np.float64),
        tp_moves=tp, sl_moves=sl, upper_threshold=0.0, lower_threshold=0.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    ledger = result.ledger
    if len(ledger) == 0:
        return {"n_trades": 0}
    equity = result.equity
    running_max = np.maximum.accumulate(equity)
    mdd = float(((equity - running_max) / running_max).min() * 100)
    return {
        "n_trades": int(len(ledger)), "win_rate": float((ledger["trade_return"] > 0).mean()),
        "sum_ret_pct": float(ledger["trade_return"].sum() * 100), "final_equity": float(equity[-1]), "mdd_pct": mdd,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--stage1-epochs", type=int, default=25)
    p.add_argument("--stage2-epochs", type=int, default=25)
    args = p.parse_args()

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(
        window=48, train_stride=4, label_path=TB_LABEL_PATH, hard_col="trade_outcome_action",
        soft_cols=["trade_outcome_soft_cash", "trade_outcome_soft_long", "trade_outcome_soft_short"],
    )
    zz_hard_all, zz_soft_all = _load_zigzag_aligned(ds)

    encoder = SupervisedTransformerEncoder(len(ds.feature_columns), d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT, embed_dim=EMBED_DIM).to(device)

    # ---- Stage 1: pretrain on zigzag ----
    zz_head = nn.Linear(EMBED_DIM, 3).to(device)
    print("=== Stage 1: zigzag pretraining ===")
    _train_stage(encoder, zz_head, ds, zz_hard_all, zz_soft_all, device, args.stage1_epochs, 6, 3e-4, 5e-4, args.batch_size, args.seed, ZZ_CASH_WEIGHT, "stage1-zigzag")

    # ---- Stage 2: freeze early layers, fine-tune on triple-barrier ----
    for p_ in encoder.input_proj.parameters():
        p_.requires_grad_(False)
    encoder.pos_embed.requires_grad_(False)
    for p_ in encoder.encoder.layers[0].parameters():
        p_.requires_grad_(False)
    tb_head = nn.Linear(EMBED_DIM, 3).to(device)
    print("=== Stage 2: triple-barrier fine-tuning (early layers frozen) ===")
    _train_stage(encoder, tb_head, ds, ds.y_hard_all, ds.y_soft_all, device, args.stage2_epochs, 6, 1.5e-4, 5e-4, args.batch_size, args.seed + 1, TB_CASH_WEIGHT, "stage2-tripbarrier")

    stage2_val = _evaluate_head(encoder, tb_head, ds, ds.y_hard_all, ds.y_soft_all, "val", device, args.batch_size, TB_CASH_WEIGHT)
    stage2_oos = _evaluate_head(encoder, tb_head, ds, ds.y_hard_all, ds.y_soft_all, "oos", device, args.batch_size, TB_CASH_WEIGHT)
    print(f"[stage2 final] val_acc={stage2_val['acc']:.4f} oos_acc={stage2_oos['acc']:.4f}")

    # ---- Stage 3: Reptile meta-learning of tb_head over rolling weekly regime tasks ----
    print("=== Stage 3: Reptile regime meta-learning ===")
    meta_head = _reptile_meta_train(encoder, copy.deepcopy(tb_head), ds, device, args.seed + 2)

    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)
    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(12).sum().to_numpy()
    vol = pd.Series(cumret).rolling(288, min_periods=288).std().to_numpy()
    tp_moves_all, sl_moves_all = TP_MULT * vol, SL_MULT * vol

    rng = np.random.default_rng(args.seed + 3)
    results = {}
    for split in ("val", "oos"):
        # Stage-2-only (no meta-adaptation) baseline for comparison
        row_idx = ds.end_idx[split]
        pred_stage2 = np.concatenate([
            _predict_with_head(encoder, tb_head, ds, row_idx[i : i + args.batch_size], device)
            for i in range(0, len(row_idx), args.batch_size)
        ])
        results[f"{split}_stage2_no_meta"] = _backtest(row_idx, pred_stage2, panel, tp_moves_all, sl_moves_all)
        print(f"{split}/stage2_no_meta:", json.dumps(results[f"{split}_stage2_no_meta"]))

        row_idx_meta, pred_meta = _walk_forward_predict(encoder, meta_head, ds, split, device, rng)
        results[f"{split}_meta_adapted"] = _backtest(row_idx_meta, pred_meta, panel, tp_moves_all, sl_moves_all)
        print(f"{split}/meta_adapted:", json.dumps(results[f"{split}_meta_adapted"]))

    torch.save({"encoder_state": encoder.state_dict(), "tb_head_state": tb_head.state_dict(), "meta_head_state": meta_head.state_dict()}, OUT_DIR / "bundle.pt")
    (OUT_DIR / "results.json").write_text(json.dumps({"stage2_val": stage2_val, "stage2_oos": stage2_oos, "backtest": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
