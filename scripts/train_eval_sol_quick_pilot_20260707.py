"""Phase A pilot (SOL), quick first-pass validation (per user's explicit scope choice: skip a full
risk-sidecar retrain and robust exit-head, use fixed-fraction sizing and a static TP/SL barrier
matching zig075's own config, to get a fast read on whether the SOL signal itself is worth the full
build). Reuses the CORE architecture (ThreeHeadTabM class, 3 regime-expert mixture, standardization)
forked from train_eval_omega1_2_tabm_3head_20260603.py, but with SOL's own regime3/labels/features
(all built in prior pilot steps) and WITHOUT the ETH-specific m7/ATR-safety/exit-baseline-routing
machinery (m7 is confirmed NOT part of zig075/h48qual's own base_cols contract, so dropping it here
matches the live model, not a shortcut relative to it).

Protocol: TRAIN 2025-01-01..09-30 (train the parent), VAL 2025-10-01..12-31 (quality-threshold
selection only), OOS 2026-01-01..06-30 (one-shot). Static TP=7.5%/SL=4% (zig075's own barrier),
fixed margin_fraction=0.30/leverage=3x (no learned sizing), single component (no h48qual analog),
no duration gate. Diagnostic/quick-check only -- not a live-promotion claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL_ID = "sol_quick_pilot_3head_tabm_20260707_v2_102cols"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ROOT / "data/splits/year_oos"
REGIME3_DIR = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707"
LABELS_DIR = ROOT / "tmp/causal_regen_20260516/sol_zigzag_action_labels_20260707"

TRAIN_START, TRAIN_END = "2025-01-01", "2025-09-30 23:59:59"
VAL_START, VAL_END = "2025-10-01", "2025-12-31 23:59:59"
OOS_START, OOS_END = "2026-01-01", "2026-06-30 23:59:59"
QUALITY_THRESHOLDS = [0.60, 0.70, 0.75, 0.80]
TAKE_PROFIT, STOP_LOSS = 0.075, 0.040  # zig075's own static barrier
MARGIN_FRACTION, LEVERAGE = 0.30, 3.0
FEE, SLIP = 0.0005, 0.0003  # matching this project's typical taker fee+slip assumption
EXPERT_NAMES = ["bull", "bear", "chop"]
ROUTE_COLS = ["regime3_current_sensitive_wide24_bull_prob", "regime3_current_sensitive_wide24_bear_prob",
              "regime3_current_sensitive_wide24_chop_prob"]
POS_COLS = ["pos_side", "pos_hold_bars", "pos_unrealized", "pos_mfe", "pos_mae", "pos_giveback",
            "pos_dist_to_tp", "pos_dist_to_sl", "pos_notional", "pos_leverage", "pos_exposure",
            "pos_tp", "pos_sl"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ThreeHeadConfig:
    k = 8
    hidden = 192
    layers = 3
    dropout = 0.08
    batch_size = 2048
    lr = 2.0e-3
    weight_decay = 2.0e-4
    patience = 8
    quality_loss_weight = 0.80


CFG = ThreeHeadConfig()


class ThreeHeadTabM(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.k = CFG.k
        self.n_features = n_features
        self.input_scale = nn.Parameter(torch.randn(self.k, n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(self.k, n_features))
        self.in_proj = nn.Linear(n_features, CFG.hidden)
        self.blocks = nn.ModuleList(nn.Linear(CFG.hidden, CFG.hidden) for _ in range(CFG.layers - 1))
        self.expert_scale = nn.ParameterList(nn.Parameter(torch.randn(self.k, CFG.hidden) * 0.03 + 1.0) for _ in range(CFG.layers - 1))
        self.norms = nn.ModuleList(nn.LayerNorm(CFG.hidden) for _ in range(CFG.layers))
        self.dropout = nn.Dropout(CFG.dropout)
        self.direction_head = nn.Linear(CFG.hidden, 3)
        self.quality_head = nn.Linear(CFG.hidden, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return {"direction": self.direction_head(h), "quality": self.quality_head(h)}


def load_split(year_files: list[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for yf in year_files:
        feat = pd.read_csv(SPLITS / yf, low_memory=False)
        feat["timestamp"] = pd.to_datetime(feat["timestamp"])
        year = yf.split("_")[-1].split(".")[0]
        regime = pd.read_csv(REGIME3_DIR / f"sol_features_{year}_regime3_current_sensitive_hmm_wide24.csv", low_memory=False)
        regime["timestamp"] = pd.to_datetime(regime["timestamp"])
        label = pd.read_csv(LABELS_DIR / f"sol_zigzag_action_labels_{year}.csv", low_memory=False)
        label["timestamp"] = pd.to_datetime(label["timestamp"])
        regime_cols = [c for c in regime.columns if c != "timestamp"]
        merged = feat.merge(regime[["timestamp", *regime_cols]], on="timestamp", how="inner")
        merged = merged.merge(label[["timestamp", "zigzag_action"]], on="timestamp", how="inner")
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out[(out["timestamp"] >= start) & (out["timestamp"] <= end)].reset_index(drop=True)


def base_input(frame: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    x = frame.reindex(columns=base_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c in POS_COLS:
        x[c] = 0.0
    return x.astype(np.float32)


def standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict]:
    arr = x.to_numpy(dtype=np.float32)
    mean, std = np.nanmean(arr, axis=0).astype(np.float32), np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    out = (arr - mean) / std
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized SOL training matrix")
    return out.astype(np.float32), {"mean": mean, "std": std, "columns": list(x.columns)}


def standardize_apply(x: pd.DataFrame, scaler: dict) -> np.ndarray:
    if list(x.columns) != list(scaler["columns"]):
        raise RuntimeError("SOL feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - scaler["mean"]) / scaler["std"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized SOL inference matrix")
    return out.astype(np.float32)


def fit_expert(x_np: np.ndarray, y: np.ndarray, route_w: np.ndarray, *, seed: int, epochs: int = 28) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    w = compute_sample_weight(class_weight="balanced", y=y).astype(np.float32) * route_w
    if float(w.sum()) <= 0.0:
        raise RuntimeError("invalid sample weights (route has zero mass)")
    n = len(y)
    split = max(int(n * 0.85), min(n - 1, 512))
    model = ThreeHeadTabM(x_np.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    ds = TensorDataset(torch.from_numpy(x_np[:split]), torch.from_numpy(y[:split]), torch.from_numpy(w[:split]))
    dl = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True)
    vx, vy, vw = (torch.from_numpy(x_np[split:]).to(DEVICE), torch.from_numpy(y[split:]).to(DEVICE), torch.from_numpy(w[split:]).to(DEVICE))
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            out = model(xb)
            k = CFG.k
            loss_dir = torch.nn.functional.cross_entropy(out["direction"].reshape(-1, 3), yb[:, None].expand(-1, k).reshape(-1), reduction="none").reshape(-1, k)
            loss_qual = torch.nn.functional.cross_entropy(out["quality"].reshape(-1, 3), yb[:, None].expand(-1, k).reshape(-1), reduction="none").reshape(-1, k)
            loss = ((loss_dir.mean(1) * wb).sum() / wb.sum().clamp(min=1.0)) + CFG.quality_loss_weight * ((loss_qual.mean(1) * wb).sum() / wb.sum().clamp(min=1.0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vo = model(vx)
            k = CFG.k
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, k).reshape(-1), reduction="none").reshape(-1, k)
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vy[:, None].expand(-1, k).reshape(-1), reduction="none").reshape(-1, k)
            vloss = float(((vdir.mean(1) * vw).sum() / vw.sum().clamp(min=1.0) + CFG.quality_loss_weight * (vqual.mean(1) * vw).sum() / vw.sum().clamp(min=1.0)).cpu())
        if vloss + 1e-6 < best_loss:
            best_loss, best_state, stale = vloss, {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= CFG.patience:
                break
    model.load_state_dict(best_state)
    return {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "n_features": x_np.shape[1], "best_val_loss": best_loss}


@torch.no_grad()
def predict(payload: dict, x: pd.DataFrame, scaler: dict) -> tuple[np.ndarray, np.ndarray]:
    model = ThreeHeadTabM(payload["n_features"]).to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = standardize_apply(x, scaler)
    dir_probs, qual_probs = [], []
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start:start + 8192]).to(DEVICE)
        out = model(xb)
        dir_probs.append(torch.softmax(out["direction"], dim=-1).mean(1).cpu().numpy())
        qual_probs.append(torch.softmax(out["quality"], dim=-1).mean(1).cpu().numpy())
    return np.concatenate(dir_probs), np.concatenate(qual_probs)


def greedy_replay(frame: pd.DataFrame, side: np.ndarray, quality_for_side: np.ndarray, threshold: float) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    n = len(frame)
    pos, entry_price, entry_i = 0, 0.0, 0
    rows = []
    for i in range(n - 2):
        if pos != 0:
            move = (close[i] * (1 - SLIP) - entry_price) / entry_price if pos > 0 else (entry_price - close[i] * (1 + SLIP)) / entry_price
            if move >= TAKE_PROFIT or move <= -STOP_LOSS:
                exit_px = close[i] * (1 - SLIP if pos > 0 else 1 + SLIP)
                raw = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                notional = MARGIN_FRACTION * LEVERAGE
                trade_return = raw * notional - 2 * FEE * notional
                rows.append({"entry_i": entry_i, "exit_i": i, "side": int(pos), "trade_return": float(trade_return),
                             "reason": "take_profit" if move >= TAKE_PROFIT else "stop_loss"})
                pos = 0
            continue
        s = int(side[i])
        if s == 0 or quality_for_side[i] < threshold:
            continue
        pos = s
        entry_price = close[min(i + 1, n - 1)] * (1 + SLIP if s > 0 else 1 - SLIP)
        entry_i = min(i + 1, n - 1)
    return pd.DataFrame(rows)


def metrics(ledger: pd.DataFrame) -> dict:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    r = ledger["trade_return"].to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {"pnl": float((curve[-1] - 1) * 100), "mdd": float(dd.min() * 100), "trades": int(len(r)), "wr": float((r > 0).mean())}


def main() -> int:
    print("Loading SOL TRAIN/VAL/OOS splits...", flush=True)
    train = load_split(["sol_features_2025.csv"], TRAIN_START, TRAIN_END)
    val = load_split(["sol_features_2025.csv"], VAL_START, VAL_END)
    oos = load_split(["sol_features_2026.csv"], OOS_START, OOS_END)
    print(f"train={len(train)} val={len(val)} oos={len(oos)}", flush=True)

    zig075_bundle = torch.load(
        ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt",
        map_location="cpu", weights_only=False)
    base_cols = list(zig075_bundle["base_cols"])
    missing = [c for c in base_cols if c not in train.columns]
    if missing:
        raise RuntimeError(f"SOL frame missing zig075's base_cols: {missing}")
    print(f"base_cols (restricted to zig075's exact 102): {len(base_cols)}", flush=True)

    x_all = base_input(train, base_cols)
    x_np, scaler = standardize_fit(x_all)
    y = pd.to_numeric(train["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    route = train[ROUTE_COLS].to_numpy(dtype=np.float64)

    models = {}
    for idx, expert in enumerate(EXPERT_NAMES):
        print(f"training expert {expert}...", flush=True)
        payload = fit_expert(x_np, y, route[:, idx].astype(np.float32), seed=707 + idx)
        models[expert] = payload
        print(f"  {expert}: best_val_loss={payload['best_val_loss']:.4f}", flush=True)

    torch.save({"models": models, "base_cols": base_cols, "scaler": scaler, "config": CFG.__dict__},
               OUT_DIR / "sol_quick_pilot_bundle.pt")

    def decide(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = base_input(frame, base_cols)
        route_local = frame[ROUTE_COLS].to_numpy(dtype=np.float64)
        route_id = np.argmax(route_local, axis=1)
        side = np.zeros(len(frame), dtype=np.int64)
        qual_for_side = np.zeros(len(frame), dtype=np.float64)
        for idx, expert in enumerate(EXPERT_NAMES):
            mask = route_id == idx
            if not mask.any():
                continue
            dir_p, qual_p = predict(models[expert], x.loc[mask], scaler)
            local_side = np.where(dir_p[:, 1] > dir_p[:, 2], 1, np.where(dir_p[:, 2] > dir_p[:, 1], -1, 0))
            local_side = np.where(dir_p.argmax(axis=1) == 0, 0, local_side)
            local_qual = np.where(local_side > 0, qual_p[:, 1], np.where(local_side < 0, qual_p[:, 2], 0.0))
            idxs = np.where(mask)[0]
            side[idxs] = local_side
            qual_for_side[idxs] = local_qual
        return side, qual_for_side

    print("\nRunning VAL threshold sweep...", flush=True)
    val_side, val_qual = decide(val)
    grid = []
    for th in QUALITY_THRESHOLDS:
        lg = greedy_replay(val, val_side, val_qual, th)
        m = metrics(lg)
        grid.append({"threshold": th, **m})
        print(f"  threshold={th:.2f} -> pnl={m['pnl']:+7.2f}% mdd={m['mdd']:+6.2f}% n={m['trades']:3d} wr={m['wr']:.3f}", flush=True)
    grid.sort(key=lambda r: r["pnl"], reverse=True)
    best = grid[0]
    print(f"\nBest VAL threshold: {best}", flush=True)

    print("\nRunning OOS one-shot confirm...", flush=True)
    oos_side, oos_qual = decide(oos)
    lg_oos = greedy_replay(oos, oos_side, oos_qual, best["threshold"])
    m_oos = metrics(lg_oos)
    print(f"OOS frozen (threshold={best['threshold']}): {m_oos}", flush=True)

    result = {"model_id": MODEL_ID, "quick_pilot_simplifications": [
        "static TP7.5%/SL4% barrier (zig075's own config, not re-derived via ATR-safety)",
        "fixed margin_fraction=0.30/leverage=3x (no learned risk sidecar)",
        "no exit-head (matches live model's effectively-inert 0.95 threshold)",
        "no duration gate", "single component only (no h48qual analog)"],
        "val_grid": grid, "val_best": best, "oos_oneshot": m_oos}
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT_DIR / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
