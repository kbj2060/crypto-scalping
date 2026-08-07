"""Multi-task ensemble: one shared transformer encoder with TWO direction heads, one trained on
the triple-barrier oracle label, one on the zigzag oracle label (both independently oracle-
validated per [[project-btc-oracle-label-selection-protocol-20260806]]: triple_barrier 100% win
rate/44.3x OOS equity, zigzag 73-75% win rate/~4.6-4.9x OOS equity). A trade is only taken when
BOTH heads' argmax AGREE on direction (both LONG or both SHORT) -- this is prediction-time
agreement gating, not label-construction-time combination (the latter was already tried in this
repo's history via `build_btc_cusum_trendscan_zigzag_hybrid_20260803.py` and failed; prediction-
time agreement between two independently-oracle-validated, imperfect trained models is a
genuinely different and untested hypothesis).

Architecture matches the best single-label triple-barrier config
(window=48, d_model=96, n_layers=3, dropout=0.25). Loss = soft-CE(tb, cash_weight=0.9) +
soft-CE(zigzag, cash_weight=1.0) -- tb's cash_weight carries over from its own tuning; zigzag's
class balance (~5% CASH) is very different so its default weighting is tried first.
"""
from __future__ import annotations

import argparse
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
OUT_DIR = ROOT / "tmp/btc_multitask_tb_zigzag_ensemble_20260806"

TB_CASH_WEIGHT = 0.9
ZZ_CASH_WEIGHT = 1.0
TP_MULT, SL_MULT, HORIZON_BARS = 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010


class MultiTaskModel(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.tb_head = nn.Linear(embed_dim, 3)
        self.zz_head = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        h = self.dropout(emb)
        return self.tb_head(h), self.zz_head(h)


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


@torch.no_grad()
def _evaluate(model, ds, zz_hard_all, zz_soft_all, split, device, batch_size):
    model.eval()
    row_idx = ds.end_idx[split]
    tb_correct = zz_correct = agree_correct = 0
    tb_loss_sum = zz_loss_sum = 0.0
    n = 0
    for i in range(0, len(row_idx), batch_size):
        chunk = row_idx[i : i + batch_size]
        x = torch.from_numpy(ds.get_batch(chunk)).to(device)
        tb_logits, zz_logits = model(x)
        tb_soft = torch.from_numpy(ds.y_soft_all[chunk]).to(device)
        tb_hard = torch.from_numpy(ds.y_hard_all[chunk]).to(device)
        zz_soft = torch.from_numpy(zz_soft_all[chunk]).to(device)
        zz_hard = torch.from_numpy(zz_hard_all[chunk]).to(device)
        tb_target, tb_w = _prepare_target(tb_soft, tb_hard, 1.0, TB_CASH_WEIGHT)
        zz_target, zz_w = _prepare_target(zz_soft, zz_hard, 1.0, ZZ_CASH_WEIGHT)
        tb_loss_sum += float(_soft_ce_loss(tb_logits, tb_target, tb_w).item()) * len(chunk)
        zz_loss_sum += float(_soft_ce_loss(zz_logits, zz_target, zz_w).item()) * len(chunk)
        tb_pred = tb_logits.argmax(dim=-1)
        zz_pred = zz_logits.argmax(dim=-1)
        tb_correct += int((tb_pred == tb_hard).sum().item())
        zz_correct += int((zz_pred == zz_hard).sum().item())
        agree = tb_pred == zz_pred
        agree_correct += int((agree & (tb_pred == tb_hard)).sum().item())
        n += len(chunk)
    return {
        "n": n, "tb_acc": tb_correct / n, "zz_acc": zz_correct / n,
        "tb_loss": tb_loss_sum / n, "zz_loss": zz_loss_sum / n,
        "agree_and_tb_correct_acc": agree_correct / n,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--train-stride", type=int, default=4)
    args = p.parse_args()

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(
        window=48, train_stride=args.train_stride, label_path=TB_LABEL_PATH, hard_col="trade_outcome_action",
        soft_cols=["trade_outcome_soft_cash", "trade_outcome_soft_long", "trade_outcome_soft_short"],
    )
    zz_hard_all, zz_soft_all = _load_zigzag_aligned(ds)

    encoder = SupervisedTransformerEncoder(len(ds.feature_columns), d_model=96, n_heads=4, n_layers=3, dropout=0.25, embed_dim=32)
    model = MultiTaskModel(encoder, embed_dim=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    rng = np.random.default_rng(args.seed)
    best_val_loss, best_state, epochs_since_best = float("inf"), None, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = _iterate_batches(ds.end_idx["train"], args.batch_size, rng)
        train_loss_sum, n_seen = 0.0, 0
        for chunk in batches:
            x = torch.from_numpy(ds.get_batch(chunk)).to(device)
            tb_soft = torch.from_numpy(ds.y_soft_all[chunk]).to(device)
            tb_hard = torch.from_numpy(ds.y_hard_all[chunk]).to(device)
            zz_soft = torch.from_numpy(zz_soft_all[chunk]).to(device)
            zz_hard = torch.from_numpy(zz_hard_all[chunk]).to(device)
            tb_target, tb_w = _prepare_target(tb_soft, tb_hard, 1.0, TB_CASH_WEIGHT)
            zz_target, zz_w = _prepare_target(zz_soft, zz_hard, 1.0, ZZ_CASH_WEIGHT)

            opt.zero_grad()
            tb_logits, zz_logits = model(x)
            loss = _soft_ce_loss(tb_logits, tb_target, tb_w) + _soft_ce_loss(zz_logits, zz_target, zz_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += float(loss.item()) * len(chunk)
            n_seen += len(chunk)

        val_metrics = _evaluate(model, ds, zz_hard_all, zz_soft_all, "val", device, args.batch_size)
        val_total_loss = val_metrics["tb_loss"] + val_metrics["zz_loss"]
        scheduler.step(val_total_loss)
        row = {"epoch": epoch, "train_loss": train_loss_sum / n_seen, "val_total_loss": val_total_loss, **val_metrics}
        history.append(row)
        print(json.dumps(row))

        if val_total_loss < best_val_loss - 1e-4:
            best_val_loss = val_total_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save({"model_state": best_state, "mean": ds.mean, "std": ds.std, "feature_columns": ds.feature_columns}, OUT_DIR / "multitask_bundle.pt")

    val_final = _evaluate(model, ds, zz_hard_all, zz_soft_all, "val", device, args.batch_size)
    oos_final = _evaluate(model, ds, zz_hard_all, zz_soft_all, "oos", device, args.batch_size)
    print("FINAL val:", json.dumps(val_final))
    print("FINAL oos:", json.dumps(oos_final))

    # ---- Agreement-gated backtest ----
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)
    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(12).sum().to_numpy()
    vol = pd.Series(cumret).rolling(288, min_periods=288).std().to_numpy()
    tp_moves_all, sl_moves_all = TP_MULT * vol, SL_MULT * vol

    def fresh_mask(side_state):
        fresh = np.zeros(len(side_state), dtype=bool)
        fresh[0] = side_state[0] != 0
        fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
        return fresh

    @torch.no_grad()
    def predict_side(split):
        model.eval()
        row_idx = ds.end_idx[split]
        tb_preds, zz_preds = [], []
        for i in range(0, len(row_idx), args.batch_size):
            chunk = row_idx[i : i + args.batch_size]
            x = torch.from_numpy(ds.get_batch(chunk)).to(device)
            tb_logits, zz_logits = model(x)
            tb_preds.append(tb_logits.argmax(dim=-1).cpu().numpy())
            zz_preds.append(zz_logits.argmax(dim=-1).cpu().numpy())
        tb_pred = np.concatenate(tb_preds)
        zz_pred = np.concatenate(zz_preds)
        agree_side = np.where((tb_pred == 1) & (zz_pred == 1), 1, np.where((tb_pred == 2) & (zz_pred == 2), -1, 0))
        tb_side = np.where(tb_pred == 1, 1, np.where(tb_pred == 2, -1, 0))
        return row_idx, agree_side, tb_side

    backtest_results = {}
    for split in ("val", "oos"):
        row_idx, agree_side, tb_side = predict_side(split)
        for name, side_state in [("agreement_gated", agree_side), ("tb_only", tb_side)]:
            fresh = fresh_mask(side_state)
            idx, side = row_idx[fresh], side_state[fresh]
            tp, sl = tp_moves_all[idx], sl_moves_all[idx]
            finite = np.isfinite(tp) & np.isfinite(sl)
            idx, side, tp, sl = idx[finite], side[finite], tp[finite], sl[finite]
            if len(idx) == 0:
                backtest_results[f"{split}_{name}"] = {"n_trades": 0}
                continue
            result = simulate_single_position(
                timestamps=panel["timestamp"], open_px=panel["open"].to_numpy(dtype=np.float64),
                high=panel["high"].to_numpy(dtype=np.float64), low=panel["low"].to_numpy(dtype=np.float64),
                close=close, decision_indices=idx, scores=side.astype(np.float64), tp_moves=tp, sl_moves=sl,
                upper_threshold=0.0, lower_threshold=0.0, horizon_bars=HORIZON_BARS,
                margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
            )
            ledger = result.ledger
            if len(ledger) == 0:
                backtest_results[f"{split}_{name}"] = {"n_trades": 0}
                continue
            equity = result.equity
            running_max = np.maximum.accumulate(equity)
            mdd = float(((equity - running_max) / running_max).min() * 100)
            summary = {
                "n_trades": int(len(ledger)),
                "win_rate": float((ledger["trade_return"] > 0).mean()),
                "sum_ret_pct": float(ledger["trade_return"].sum() * 100),
                "final_equity": float(equity[-1]),
                "mdd_pct": mdd,
            }
            backtest_results[f"{split}_{name}"] = summary
            print(f"{split}/{name}:", json.dumps(summary))

    (OUT_DIR / "results.json").write_text(
        json.dumps({"history": history, "val_final": val_final, "oos_final": oos_final, "backtest": backtest_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
