"""G5 retest -- does adding the 4 causal zigzag-state features to the triple-barrier entry
model's input change its OOS gross edge, under G2's purge+embargo+uniqueness hygiene and N=5
genuinely random seeds?

Paired design: the SAME 5 seeds run twice, hygiene and architecture frozen to G2's D_purge_uniq
config (the only variable that changes between conditions is the feature set):
  A_causalfix_only     -- causalfix_final 113 cols (G2's own D_purge_uniq control, rerun on fresh
                           seeds so the comparison is paired, not borrowed from a different seed set)
  B_plus_zigzag_state  -- causalfix_final + zz_trend_state/zz_bars_since_pivot/
                           zz_move_from_pivot_pct/zz_dist_to_threshold
                           (scripts/build_btc_5m_zigzag_state_causal_features_20260806.py)

Contract: docs/experiments/btc_tb_entry_zigzag_state_feature_retest_20260807.json
Registry line: btc_zigzag_as_entry_model_component (docs/model_contracts/research_line_registry.json)

Prior informal result (uncontrolled hygiene, seed count unknown) claimed raw zigzag-feature
injection "hurt backtest PnL" -- this predates G2's discovery that single/few-seed evaluation on
this label is unreliable (effective_n ~4,058 vs 43,798 nominal). This script is the controlled
retest.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.causal_futures_backtest import simulate_single_position  # noqa: E402
from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402
from ensemble.deep_features.btc_deepfeat_encoders_20260806 import build_model  # noqa: E402
from train_btc_deepfeat_encoders_20260806 import _prepare_target, _soft_ce_loss  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet"
SPAN_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_label_span_20260807.parquet"
ZIGZAG_STATE_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_state_causal_features_20260806.parquet"
ZIGZAG_STATE_COLS = ["zz_trend_state", "zz_bars_since_pivot", "zz_move_from_pivot_pct", "zz_dist_to_threshold"]
OUT_DIR = ROOT / "tmp/btc_g5_zigzag_feature_retest_20260807"

SOFT_COLS = ["trade_outcome_soft_cash", "trade_outcome_soft_long", "trade_outcome_soft_short"]
# frozen to G2's D_purge_uniq (== the shipped checkpoint's) config; only the feature set varies
ARCH_CFG = dict(arch="transformer", window=48, embed_dim=32, d_model=96, n_heads=4, n_layers=3,
                ffn_mult=2, dropout=0.25, head_type="linear")
TRAIN_CFG = dict(batch_size=512, lr=3e-4, weight_decay=5e-4, grad_clip_norm=1.0, epochs=30,
                 patience=8, min_delta=1e-4, train_stride=4, cash_weight=0.9, label_sharpen=1.0)
EMBARGO_BARS = 288
HYGIENE = dict(purge=True, uniqueness_weights=True, embargo_bars=EMBARGO_BARS)

CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS = 12, 288, 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
ACCOUNT_COST = ROUNDTRIP_COST_RATE * MARGIN_FRACTION * LEVERAGE

CONDITIONS = {
    "A_causalfix_only": dict(extra_feature_path=None, extra_feature_cols=None),
    "B_plus_zigzag_state": dict(extra_feature_path=ZIGZAG_STATE_PATH, extra_feature_cols=ZIGZAG_STATE_COLS),
}

# G0' reference (naive direction baselines, gate_g0prime_btc_naive_direction_baselines_20260807.py)
ALWAYS_SHORT_OOS_GROSS_BPS = 2.72


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _evaluate(model, ds, split, device, batch_size):
    model.eval()
    row_idx = ds.end_idx[split]
    total_loss, total_correct, total_n = 0.0, 0, 0
    for i in range(0, len(row_idx), batch_size):
        chunk = row_idx[i : i + batch_size]
        x = torch.from_numpy(ds.get_batch(chunk)).to(device)
        y_soft = torch.from_numpy(ds.y_soft_all[chunk]).to(device)
        y_hard = torch.from_numpy(ds.y_hard_all[chunk]).to(device)
        target, weight = _prepare_target(y_soft, y_hard, TRAIN_CFG["label_sharpen"], TRAIN_CFG["cash_weight"])
        logits, _, _ = model(x)
        total_loss += float(_soft_ce_loss(logits, target, weight).item()) * len(chunk)
        total_correct += int((logits.argmax(dim=-1) == y_hard).sum().item())
        total_n += len(chunk)
    return {"n": total_n, "soft_ce_loss": total_loss / max(total_n, 1),
            "hard_top1_acc": total_correct / max(total_n, 1)}


def _train(ds, seed, device):
    _seed_everything(seed)
    model = build_model(
        ARCH_CFG["arch"], len(ds.feature_columns), ds.category_sizes, embed_dim=ARCH_CFG["embed_dim"],
        d_model=ARCH_CFG["d_model"], n_heads=ARCH_CFG["n_heads"], n_layers=ARCH_CFG["n_layers"],
        ffn_mult=ARCH_CFG["ffn_mult"], dropout=ARCH_CFG["dropout"], quality_head=False,
        head_type=ARCH_CFG["head_type"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    rng = np.random.default_rng(seed)

    train_idx = ds.end_idx["train"]
    weight_lookup = None
    if ds.train_weight is not None:
        weight_lookup = np.zeros(ds.feat_std.shape[0], dtype=np.float32)
        weight_lookup[train_idx] = ds.train_weight

    best_loss, best_state, since_best, history = float("inf"), None, 0, []
    for epoch in range(1, TRAIN_CFG["epochs"] + 1):
        model.train()
        order = train_idx.copy()
        rng.shuffle(order)
        loss_sum, n_seen = 0.0, 0
        for i in range(0, len(order), TRAIN_CFG["batch_size"]):
            chunk = order[i : i + TRAIN_CFG["batch_size"]]
            x = torch.from_numpy(ds.get_batch(chunk)).to(device)
            y_soft = torch.from_numpy(ds.y_soft_all[chunk]).to(device)
            y_hard = torch.from_numpy(ds.y_hard_all[chunk]).to(device)
            target, weight = _prepare_target(y_soft, y_hard, TRAIN_CFG["label_sharpen"], TRAIN_CFG["cash_weight"])
            if weight_lookup is not None:
                weight = weight * torch.from_numpy(weight_lookup[chunk]).to(device)
            opt.zero_grad()
            logits, _, _ = model(x)
            loss = _soft_ce_loss(logits, target, weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip_norm"])
            opt.step()
            loss_sum += float(loss.item()) * len(chunk)
            n_seen += len(chunk)

        val_m = _evaluate(model, ds, "val", device, TRAIN_CFG["batch_size"])
        sched.step(val_m["soft_ce_loss"])
        history.append({"epoch": epoch, "train_loss": loss_sum / max(n_seen, 1), **{f"val_{k}": v for k, v in val_m.items()}})
        if val_m["soft_ce_loss"] < best_loss - TRAIN_CFG["min_delta"]:
            best_loss, since_best = val_m["soft_ce_loss"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= TRAIN_CFG["patience"]:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def _predict(model, ds, split, device, batch_size=1024):
    model.eval()
    row_idx = ds.end_idx[split]
    out = []
    for i in range(0, len(row_idx), batch_size):
        x = torch.from_numpy(ds.get_batch(row_idx[i : i + batch_size])).to(device)
        logits, _, _ = model(x)
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _fresh_entry_mask(side_state):
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh


def _backtest(row_idx, side_state, tp_all, sl_all, panel):
    mask = _fresh_entry_mask(side_state)
    idx, side = row_idx[mask], side_state[mask]
    tp, sl = tp_all[idx], sl_all[idx]
    finite = np.isfinite(tp) & np.isfinite(sl)
    idx, side, tp, sl = idx[finite], side[finite], tp[finite], sl[finite]
    if len(idx) == 0:
        return None
    return simulate_single_position(
        timestamps=panel["timestamp"], open_px=panel["open"].to_numpy(dtype=np.float64),
        high=panel["high"].to_numpy(dtype=np.float64), low=panel["low"].to_numpy(dtype=np.float64),
        close=panel["close"].to_numpy(dtype=np.float64), decision_indices=idx,
        scores=side.astype(np.float64), tp_moves=tp, sl_moves=sl, upper_threshold=0.0,
        lower_threshold=0.0, horizon_bars=HORIZON_BARS, margin_fraction=MARGIN_FRACTION,
        leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )


def _summarize(result):
    if result is None or len(result.ledger) == 0:
        return {"n_trades": 0}
    rets = result.ledger["trade_return"].to_numpy(dtype=np.float64)
    equity = result.equity
    running_max = np.maximum.accumulate(equity)
    std_bps = float(rets.std(ddof=1) * 10000.0) if len(rets) > 1 else float("nan")
    gross_bps = float((rets.mean() + ACCOUNT_COST) * 10000.0)
    return {
        "n_trades": int(len(rets)), "win_rate": float((rets > 0).mean()),
        "gross_mean_ret_bps": gross_bps,
        "t_stat_gross": float(gross_bps / (std_bps / np.sqrt(len(rets)))) if std_bps and len(rets) > 1 else float("nan"),
        "sum_ret_pct": float(rets.sum() * 100.0), "final_equity": float(equity[-1]),
        "mdd_pct": float(((equity - running_max) / running_max).min() * 100.0),
        "long_trades": int((result.ledger["side"] == 1).sum()),
        "short_trades": int((result.ledger["side"] == -1).sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="481937,1029384756,6271,88420193,3305577")
    p.add_argument("--conditions", type=str, default=",".join(CONDITIONS))
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]
    if len(seeds) < 5:
        raise ValueError("seed-diversity gate: this line's contract declares seed_ensemble_claim=true, "
                          "which requires >=5 seeds (CLAUDE.md)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_all, sl_all = TP_MULT * vol, SL_MULT * vol

    rows, hygiene_report = [], {}
    for cname in conditions:
        spec = CONDITIONS[cname]
        ds = build_dataset(
            window=ARCH_CFG["window"], train_stride=TRAIN_CFG["train_stride"],
            label_path=LABEL_PATH, hard_col="trade_outcome_action", soft_cols=SOFT_COLS,
            label_span_path=SPAN_PATH, extra_feature_path=spec["extra_feature_path"],
            extra_feature_cols=spec["extra_feature_cols"], **HYGIENE,
        )
        hygiene_report[cname] = {
            "n_features": len(ds.feature_columns), "n_train": int(len(ds.end_idx["train"])),
            "n_val": int(len(ds.end_idx["val"])), "n_oos": int(len(ds.end_idx["oos"])), **(ds.hygiene or {}),
        }
        print(f"[{cname}] n_features={len(ds.feature_columns)} {json.dumps(hygiene_report[cname])}")

        for seed in seeds:
            model, history = _train(ds, seed, device)
            row = {"condition": cname, "seed": seed, "epochs_run": len(history), "n_features": len(ds.feature_columns)}
            for split in ("val", "oos"):
                m = _evaluate(model, ds, split, device, TRAIN_CFG["batch_size"])
                probs = _predict(model, ds, split, device)
                pred = probs.argmax(axis=1)
                side_state = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
                bt = _summarize(_backtest(ds.end_idx[split], side_state, tp_all, sl_all, panel))
                row[f"{split}_acc"] = m["hard_top1_acc"]
                row.update({f"{split}_{k}": v for k, v in bt.items()})
            rows.append(row)
            print(json.dumps(row))

    df = pd.DataFrame(rows)
    agg_cols = ["oos_win_rate", "oos_sum_ret_pct", "oos_gross_mean_ret_bps", "oos_n_trades",
                "oos_mdd_pct", "val_sum_ret_pct", "val_win_rate", "oos_acc"]
    agg = df.groupby("condition")[agg_cols].agg(["mean", "std"])

    print("\n=== per-seed results ===")
    hdr = (f"{'condition':<22}{'seed':>12}{'oos_acc':>9}{'oos_tr':>8}{'oos_win%':>10}"
           f"{'oos_gross':>11}{'oos_t':>7}{'oos_sum%':>10}{'val_sum%':>10}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"{r['condition']:<22}{r['seed']:>12}{r['oos_acc']*100:>9.1f}{r['oos_n_trades']:>8.0f}"
              f"{r['oos_win_rate']*100:>10.1f}{r['oos_gross_mean_ret_bps']:>11.2f}"
              f"{r['oos_t_stat_gross']:>7.2f}{r['oos_sum_ret_pct']:>10.1f}{r['val_sum_ret_pct']:>10.1f}")

    print("\n=== condition means (std) ===")
    hdr2 = f"{'condition':<22}{'oos_win%':>18}{'oos_gross_bps':>20}{'oos_sum%':>20}{'val_sum%':>20}"
    print(hdr2)
    print("-" * len(hdr2))
    for c in agg.index:
        def cell(col):
            return f"{agg.loc[c, (col, 'mean')]:.2f} ({agg.loc[c, (col, 'std')]:.2f})"
        print(f"{c:<22}{cell('oos_win_rate'):>18}{cell('oos_gross_mean_ret_bps'):>20}"
              f"{cell('oos_sum_ret_pct'):>20}{cell('val_sum_ret_pct'):>20}")

    # G5 pass criterion: condition B's seed-mean OOS gross edge significantly (t>=2) above always_short
    if "B_plus_zigzag_state" in agg.index:
        b_mean = agg.loc["B_plus_zigzag_state", ("oos_gross_mean_ret_bps", "mean")]
        b_std = agg.loc["B_plus_zigzag_state", ("oos_gross_mean_ret_bps", "std")]
        n = len(seeds)
        t_vs_always_short = (b_mean - ALWAYS_SHORT_OOS_GROSS_BPS) / (b_std / np.sqrt(n)) if b_std else float("nan")
        print(f"\nG5 check: condition B seed-mean OOS gross={b_mean:.2f}bps (sd={b_std:.2f}) "
              f"vs always_short={ALWAYS_SHORT_OOS_GROSS_BPS:.2f}bps -> t={t_vs_always_short:.2f} "
              f"({'PASS' if t_vs_always_short >= 2 else 'FAIL'}, need t>=2)")

    payload = {"arch_cfg": ARCH_CFG, "train_cfg": TRAIN_CFG, "hygiene_spec": HYGIENE, "seeds": seeds,
               "hygiene": hygiene_report, "per_seed": rows,
               "aggregate": json.loads(agg.to_json()),
               "always_short_oos_gross_bps_reference": ALWAYS_SHORT_OOS_GROSS_BPS,
               "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
               "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
    (OUT_DIR / "g5_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    df.to_csv(OUT_DIR / "g5_per_seed.csv", index=False)
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
