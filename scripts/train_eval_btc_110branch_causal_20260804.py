"""BTC-110: causalfix market state + CURRENT Regime3 + DVOL + on-chain branch model.

Feature contract: 94 causalfix market fields, 4 CURRENT Regime3 outputs, 6 causal Deribit
DVOL fields, and 6 daily CoinMetrics fields.  Regime3 PRED fields are forbidden.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.backtest_metrics import bar_level_performance  # noqa: E402
from core.causal_futures_backtest import fit_tail_thresholds, purged_decision_mask, simulate_single_position  # noqa: E402

BASE = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
DVOL = ROOT / "data/splits/year_oos/btc_dvol_features_20260804.parquet"
ONCHAIN = ROOT / "data/splits/year_oos/btc_onchain_features_20260804.parquet"
CKPT = ROOT / "data/ensemble/supervised/btc_110branch_causal_20260804.pt"
OUT = ROOT / "tmp/btc_110branch_causal_20260804"

TRAIN_END = pd.Timestamp("2025-09-01")
VAL_END, CAL_END, TEST_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01"), pd.Timestamp("2026-08-01")
HORIZON = 48
MARGIN, LEVERAGE, COST = 0.30, 3.0, 0.0014
QUANTILES = np.array([.05, .10, .25, .50, .75, .90, .95], dtype=np.float32)
TP_SL = [("moderate", .75, .25), ("wide", .90, .10)]
TAILS = [(.80, .20), (.90, .10), (.95, .05)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXCLUDE = {"timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "close_btc", "volume_btc", "quote_volume_btc", "mtf1h_ts_t_value", "mtf1h_ts_opt_L"}
REGIME = ["regime3_current_sensitive_wide24_bull_prob", "regime3_current_sensitive_wide24_bear_prob", "regime3_current_sensitive_wide24_chop_prob", "regime3_current_sensitive_wide24_confidence"]
DVOL_COLS = ["dvol_btc", "dvol_eth", "dvol_btc_eth_spread", "dvol_btc_pctrank_720h", "dvol_btc_roc_24h", "dvol_btc_roc_168h"]
ONCHAIN_COLS = ["mvrv", "mvrv_pctrank_90d", "net_exchange_flow_pct_supply", "sply_ex_roc_7d", "active_addr_roc_7d", "active_addr_pctrank_90d"]


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(BASE).sort_values("timestamp").reset_index(drop=True)
    dvol, onchain = pd.read_parquet(DVOL), pd.read_parquet(ONCHAIN)
    frame = frame.merge(dvol[["timestamp", *DVOL_COLS]], on="timestamp", how="left", validate="one_to_one")
    frame = frame.merge(onchain[["timestamp", *ONCHAIN_COLS]], on="timestamp", how="left", validate="one_to_one")
    market = [c for c in frame.columns if c not in EXCLUDE and c not in REGIME + DVOL_COLS + ONCHAIN_COLS]
    if len(market) != 94:
        raise RuntimeError(f"Expected 94 causalfix market fields, got {len(market)}")
    forbidden = [c for c in frame if c.startswith("regime3_pred_")]
    if forbidden:
        raise RuntimeError(f"Forbidden Regime3 PRED fields in contract: {forbidden}")
    cols = market + REGIME + DVOL_COLS + ONCHAIN_COLS
    if len(cols) != 110 or frame[cols].columns.tolist() != cols:
        raise RuntimeError("BTC-110 feature contract mismatch")
    return frame, cols


class BTC110Branch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        def branch(n: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(n, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(.10))
        self.market, self.regime, self.dvol, self.onchain = branch(94), branch(4), branch(6), branch(6)
        self.fuse = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(.10))
        self.direction, self.quantile = nn.Linear(64, 1), nn.Linear(64, len(QUANTILES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([self.market(x[:, :94]), self.regime(x[:, 94:98]), self.dvol(x[:, 98:104]), self.onchain(x[:, 104:])], 1)
        z = self.fuse(z)
        return self.direction(z).squeeze(1), torch.sort(self.quantile(z), 1).values


def pinball(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    e, levels = y[:, None] - q, torch.as_tensor(QUANTILES, device=q.device)[None]
    return torch.maximum(levels * e, (levels - 1) * e).mean()


def epoch(model, loader, optimizer=None) -> dict:
    model.train(optimizer is not None); total = np.zeros(3); n = 0
    for x, y, d in loader:
        x, y, d = x.to(DEVICE), y.to(DEVICE), d.to(DEVICE)
        if optimizer: optimizer.zero_grad()
        logit, q = model(x); dl, ql = nn.functional.binary_cross_entropy_with_logits(logit, d), pinball(q, y)
        loss = dl + 20 * ql
        if optimizer:
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        total += [loss.item(), dl.item(), ql.item()]; n += 1
    return dict(zip(("loss", "direction_bce", "pinball"), (total / max(n, 1)).tolist()))


def predict(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); ds = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=1024)
    scores, quantiles = [], []
    with torch.no_grad():
        for (batch,) in ds:
            score, q = model(batch.to(DEVICE)); scores.append(torch.sigmoid(score).cpu().numpy()); quantiles.append(q.cpu().numpy())
    return np.concatenate(scores), np.concatenate(quantiles)


def moves(q: np.ndarray, hi: float, lo: float):
    idx = {round(float(v), 2): i for i, v in enumerate(QUANTILES)}; a, b = idx[round(hi, 2)], idx[round(lo, 2)]
    return np.maximum(.006, q[:, a]), np.maximum(.004, -q[:, b]), np.maximum(.006, -q[:, b]), np.maximum(.004, q[:, a])


def evaluate(frame, indices, scores, q, upper, lower, hi, lo):
    ltp, lsl, stp, ssl = moves(q, hi, lo); long = scores >= upper
    result = simulate_single_position(timestamps=frame.timestamp, open_px=frame.open.to_numpy(), high=frame.high.to_numpy(), low=frame.low.to_numpy(), close=frame.close.to_numpy(), decision_indices=indices, scores=scores, tp_moves=np.where(long, ltp, stp), sl_moves=np.where(long, lsl, ssl), upper_threshold=upper, lower_threshold=lower, horizon_bars=HORIZON, margin_fraction=MARGIN, leverage=LEVERAGE, roundtrip_cost_rate=COST)
    metric = bar_level_performance(result.equity, result.ledger); metric["mean_trade_return_pct"] = float(result.ledger.trade_return.mean() * 100) if len(result.ledger) else 0.; metric["skipped_while_open"] = result.skipped_while_open
    return metric, result.ledger


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); frame, cols = load_frame(); ts = pd.DatetimeIndex(frame.timestamp)
    raw = frame[cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    target = np.log(frame.close.shift(-HORIZON) / frame.open.shift(-1)).to_numpy(np.float32)
    valid = np.isfinite(target) & np.isfinite(raw).all(1)
    masks = {"train": purged_decision_mask(ts, start=ts[0], end=TRAIN_END, horizon_bars=HORIZON), "val": purged_decision_mask(ts, start=TRAIN_END, end=VAL_END, horizon_bars=HORIZON), "cal": purged_decision_mask(ts, start=VAL_END, end=CAL_END, horizon_bars=HORIZON), "test": purged_decision_mask(ts, start=CAL_END, end=TEST_END, horizon_bars=HORIZON)}
    train_rows = np.flatnonzero(masks["train"] & valid)[::8]; mean, std = raw[train_rows].mean(0), raw[train_rows].std(0); std[std < 1e-6] = 1
    x = np.clip((raw - mean) / std, -10, 10).astype(np.float32)
    idx = {name: np.flatnonzero(mask & valid)[::(8 if name == "train" else 4 if name == "val" else 1)] for name, mask in masks.items()}
    model = BTC110Branch().to(DEVICE); train = DataLoader(TensorDataset(torch.from_numpy(x[idx["train"]]), torch.from_numpy(target[idx["train"]]), torch.from_numpy((target[idx["train"]] > 0).astype(np.float32))), batch_size=256, shuffle=True); val = DataLoader(TensorDataset(torch.from_numpy(x[idx["val"]]), torch.from_numpy(target[idx["val"]]), torch.from_numpy((target[idx["val"]] > 0).astype(np.float32))), batch_size=512)
    opt, best, bad = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4), float("inf"), 0
    for e in range(1, 13):
        tr, va = epoch(model, train, opt), epoch(model, val)
        print(f"epoch={e} train={tr} val={va}", flush=True)
        if va["loss"] < best - 1e-5:
            best, bad = va["loss"], 0; torch.save({"state": model.state_dict(), "mean": mean, "std": std, "feature_cols": cols, "validation": va}, CKPT)
        else:
            bad += 1
            if bad >= 3: break
    saved = torch.load(CKPT, map_location=DEVICE, weights_only=False); model.load_state_dict(saved["state"])
    cal_s, cal_q = predict(model, x[idx["cal"]])
    test_s, test_q = predict(model, x[idx["test"]])
    choices, rows = [], []
    for name, hi, lo in TP_SL:
        for uq, lq in TAILS:
            t = fit_tail_thresholds(cal_s, upper_quantile=uq, lower_quantile=lq); m, ledger = evaluate(frame, idx["cal"], cal_s, cal_q, t.upper, t.lower, hi, lo); row = {"tpsl": name, "tp_quantile": hi, "sl_quantile": lo, "upper_quantile": uq, "lower_quantile": lq, "upper_threshold": t.upper, "lower_threshold": t.lower, **m}; rows.append(row); choices.append((m["pnl"], row, ledger))
    pd.DataFrame(rows).to_csv(OUT / "calibration_candidates.csv", index=False); _, chosen, ledger = max(choices, key=lambda v: v[0]); ledger.to_csv(OUT / "selected_calibration_ledger.csv", index=False)
    test_m, test_ledger = evaluate(frame, idx["test"], test_s, test_q, chosen["upper_threshold"], chosen["lower_threshold"], chosen["tp_quantile"], chosen["sl_quantile"]); test_ledger.to_csv(OUT / "test_ledger.csv", index=False)
    report = {"architecture": "btc_110_multibranch_market_regime_dvol_onchain", "feature_contract": {"market_causalfix": 94, "regime3_current_outputs": REGIME, "dvol": DVOL_COLS, "onchain": ONCHAIN_COLS, "total": len(cols)}, "model_validation": saved["validation"], "selected_config": chosen, "test_metrics": test_m, "contracts": {"fresh_forward_bar_by_bar": True, "thresholds_fit_on_calibration_only": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False, "split_targets_purged": True, "single_position": True, "bar_level_mark_to_market": True, "regime3_pred_inputs_forbidden": True, "dvol_available_at_plus_1h": True, "onchain_available_at_plus_1d": True, "margin_fraction": MARGIN, "leverage": LEVERAGE, "notional": MARGIN * LEVERAGE}, "promotion_eligible": False, "promotion_blockers": ["test period previously inspected", "current artifact lineage is research-only"]}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n"); print(json.dumps({"selected": chosen, "test": test_m}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
