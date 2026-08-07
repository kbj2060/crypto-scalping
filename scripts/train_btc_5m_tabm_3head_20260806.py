"""BTC 5m: 3-head TabM, shared backbone, same architecture family as ETH's live zig075
(ThreeHeadTabM / TabMQualityHead in scripts/train_eval_btc_v3_tabm_quality_stage1events_20260720.py),
per user's suggestion to mirror that design instead of separate single-task models.

Heads (share one TabM encoder, k=8 input-scale/bias ensemble):
  A: transition_soon (binary)      -- Layer A, this session's strongest single result (AUC 0.77)
  B: zigzag_action (3-class)       -- Layer B direction, this session's diagnosed bottleneck (acc 63%)
  C: net_ret_sim (regression)      -- oracle barrier-simulated net return, active bars only (masked
                                       elsewhere); auxiliary signal only, not a live filter -- the
                                       goal is to see whether sharing gradient with A's clean signal
                                       and C's magnitude signal helps B specifically, not to replace
                                       B with a hard quality gate (that already failed once).

Standalone TabM (single-head) was worse than LightGBM on both A and B individually with default
hyperparameters -- this test asks whether multi-task sharing changes that, not whether TabM alone
does.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
PIVOT_PATH = ROOT / "data/splits/year_oos/btc_5m_pivot_transition_labels_20260806.parquet"
ZIGZAG_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
QUALITY_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_quality_oracle_20260806.parquet"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"

DROP_RAW = {"timestamp", "open", "high", "low", "close", "close_btc", "volume_btc", "quote_volume_btc"}
VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_C_WEIGHT = 0.5


@dataclass(frozen=True)
class TabMConfig:
    k: int = 8
    hidden: int = 96
    layers: int = 3
    dropout: float = 0.15
    batch_size: int = 512
    lr: float = 1.5e-3
    weight_decay: float = 3.0e-4
    max_epochs: int = 80
    patience: int = 10
    seed: int = 20260806


class ThreeHeadTabM(nn.Module):
    def __init__(self, n_features: int, *, cfg: TabMConfig) -> None:
        super().__init__()
        self.k = cfg.k
        self.input_scale = nn.Parameter(torch.randn(cfg.k, n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(cfg.k, n_features))
        self.in_proj = nn.Linear(n_features, cfg.hidden)
        self.blocks = nn.ModuleList(nn.Linear(cfg.hidden, cfg.hidden) for _ in range(max(0, cfg.layers - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(cfg.k, cfg.hidden) * 0.03 + 1.0) for _ in range(max(0, cfg.layers - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.hidden) for _ in range(max(0, cfg.layers)))
        self.dropout = nn.Dropout(cfg.dropout)
        self.head_a = nn.Linear(cfg.hidden, 1)
        self.head_b = nn.Linear(cfg.hidden, 3)
        self.head_c = nn.Linear(cfg.hidden, 1)

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return h  # (batch, k, hidden)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        a = self.head_a(h).squeeze(-1).mean(dim=1)
        b = self.head_b(h).mean(dim=1)
        c = self.head_c(h).squeeze(-1).mean(dim=1)
        return a, b, c


def _standardize_fit(x: np.ndarray) -> dict:
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def _standardize_apply(x: np.ndarray, scaler: dict) -> np.ndarray:
    return np.nan_to_num(((x - scaler["mean"]) / scaler["std"]), nan=0.0).astype(np.float32)


def build_dvol_features() -> pd.DataFrame:
    df = pd.read_csv(DVOL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["available_at"] = df["timestamp"] + pd.Timedelta(hours=1)
    df = df[["available_at", "close"]].rename(columns={"available_at": "timestamp", "close": "dvol_btc"}).sort_values("timestamp")
    df["dvol_btc_roc_24h"] = df["dvol_btc"].pct_change(24)
    df["dvol_btc_roc_168h"] = df["dvol_btc"].pct_change(168)
    df["dvol_btc_pctrank_720h"] = df["dvol_btc"].rolling(720, min_periods=180).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    return df


def main() -> int:
    cfg = TabMConfig()
    torch.manual_seed(cfg.seed)
    print(f"device={DEVICE}")

    panel = pd.read_parquet(PANEL_PATH)
    piv = pd.read_parquet(PIVOT_PATH, columns=["timestamp", "transition_soon"])
    zz = pd.read_parquet(ZIGZAG_PATH, columns=["timestamp", "zigzag_action"])
    qual = pd.read_parquet(QUALITY_PATH, columns=["timestamp", "net_ret_sim"])
    dvol = build_dvol_features()

    df = panel.merge(piv, on="timestamp", how="inner").merge(zz, on="timestamp", how="inner").merge(qual, on="timestamp", how="inner")
    df = pd.merge_asof(df.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    df = df.dropna(subset=["transition_soon", "zigzag_action"]).reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c not in DROP_RAW] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]
    X_raw = df[feature_cols].to_numpy(dtype=np.float64)
    yA = df["transition_soon"].to_numpy(dtype=np.float32)
    yB = df["zigzag_action"].to_numpy(dtype=np.int64)
    yC_raw = df["net_ret_sim"].to_numpy(dtype=np.float32)
    maskC = np.isfinite(yC_raw)
    # scale net_ret_sim to a saner regression range (it's ~1e-3 in raw units)
    C_SCALE = 100.0
    yC = np.nan_to_num(yC_raw * C_SCALE, nan=0.0)

    train_mask = (df["timestamp"] < VAL_START).to_numpy()
    val_mask = ((df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)).to_numpy()
    oos_mask = ((df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)).to_numpy()
    print(f"train={train_mask.sum()} val={val_mask.sum()} oos={oos_mask.sum()}")
    print(f"maskC coverage (train): {maskC[train_mask].mean():.4f}")

    scaler = _standardize_fit(X_raw[train_mask])
    X = _standardize_apply(X_raw, scaler)

    rng = np.random.default_rng(cfg.seed)
    train_idx = np.flatnonzero(train_mask)
    perm = rng.permutation(len(train_idx))
    n_holdout = max(int(0.10 * len(train_idx)), 1000)
    holdout_idx, fit_idx = train_idx[perm[:n_holdout]], train_idx[perm[n_holdout:]]

    pos_weight = torch.tensor([(1 - yA[fit_idx].mean()) / max(yA[fit_idx].mean(), 1e-6)], device=DEVICE)
    class_counts = np.bincount(yB[fit_idx], minlength=3).astype(np.float32)
    class_weight = torch.tensor(class_counts.sum() / (3 * class_counts), device=DEVICE, dtype=torch.float32)

    model = ThreeHeadTabM(len(feature_cols), cfg=cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def to_t(arr, idx, dtype=torch.float32):
        return torch.as_tensor(arr[idx], dtype=dtype, device=DEVICE)

    xt, xh = to_t(X, fit_idx), to_t(X, holdout_idx)
    yAt, yAh = to_t(yA, fit_idx), to_t(yA, holdout_idx)
    yBt, yBh = to_t(yB, fit_idx, torch.int64), to_t(yB, holdout_idx, torch.int64)
    yCt, yCh = to_t(yC, fit_idx), to_t(yC, holdout_idx)
    mCt, mCh = to_t(maskC, fit_idx), to_t(maskC, holdout_idx)

    def joint_loss(a, b, c, ya, yb, yc, mc):
        la = nn.functional.binary_cross_entropy_with_logits(a, ya, pos_weight=pos_weight)
        lb = nn.functional.cross_entropy(b, yb, weight=class_weight)
        if mc.sum() > 0:
            lc = (nn.functional.mse_loss(c, yc, reduction="none") * mc).sum() / mc.sum()
        else:
            lc = torch.tensor(0.0, device=DEVICE)
        return la + lb + LOSS_C_WEIGHT * lc, (la.item(), lb.item(), float(lc))

    best_loss, best_state, patience_left = float("inf"), None, cfg.patience
    train_rng = np.random.default_rng(cfg.seed + 1)
    for epoch in range(cfg.max_epochs):
        model.train()
        fit_perm = train_rng.permutation(len(xt))
        for start in range(0, len(fit_perm), cfg.batch_size):
            idx = fit_perm[start:start + cfg.batch_size]
            opt.zero_grad()
            a, b, c = model(xt[idx])
            loss, _ = joint_loss(a, b, c, yAt[idx], yBt[idx], yCt[idx], mCt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            a, b, c = model(xh)
            hold_loss, parts = joint_loss(a, b, c, yAh, yBh, yCh, mCh)
            hold_loss = float(hold_loss.item())
        if epoch % 5 == 0 or epoch == cfg.max_epochs - 1:
            print(f"epoch {epoch}: hold_loss={hold_loss:.4f} (A={parts[0]:.4f} B={parts[1]:.4f} C={parts[2]:.4f})")
        if hold_loss < best_loss - 1e-4:
            best_loss, best_state, patience_left = hold_loss, {k: v.clone() for k, v in model.state_dict().items()}, cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop at epoch {epoch}, best_hold_loss={best_loss:.4f}")
                break
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        Xfull = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        aa, bb, cc = model(Xfull)
        probA = torch.sigmoid(aa).cpu().numpy()
        predB = bb.argmax(dim=1).cpu().numpy()
        predC = (cc / C_SCALE).cpu().numpy()

    df["probA"] = probA
    df["predB"] = predB
    df["predC"] = predC

    print("\n" + "=" * 20 + " HEAD A (transition) " + "=" * 20)
    for name, mask in [("VAL", val_mask), ("OOS", oos_mask)]:
        yt_np, p = yA[mask], probA[mask]
        auc = roc_auc_score(yt_np, p)
        ap = average_precision_score(yt_np, p)
        base_rate = yt_np.mean()
        thresh = np.quantile(p, 0.90)
        prec = yt_np[p >= thresh].mean()
        print(f"{name}: AUC={auc:.4f} AP={ap:.4f} (base={base_rate:.4f}) top-decile precision={prec:.4f}")

    print("\n" + "=" * 20 + " HEAD B (direction) " + "=" * 20)
    maj_baseline = pd.Series(yB[train_mask]).value_counts(normalize=True).max()
    for name, mask in [("VAL", val_mask), ("OOS", oos_mask)]:
        yt_np, yp = yB[mask], predB[mask]
        acc = (yt_np == yp).mean()
        f1m = f1_score(yt_np, yp, average="macro")
        print(f"{name}: baseline={maj_baseline:.4f} acc={acc:.4f} macro-F1={f1m:.4f}")
        print(classification_report(yt_np, yp, target_names=["CASH", "LONG", "SHORT"], digits=3))

    print("\n" + "=" * 20 + " HEAD C (quality/magnitude) " + "=" * 20)
    for name, mask in [("VAL", val_mask), ("OOS", oos_mask)]:
        m = mask & maskC
        if m.sum() == 0:
            continue
        corr = np.corrcoef(predC[m], yC_raw[m])[0, 1]
        print(f"{name}: n={m.sum()} corr(pred, actual net_ret_sim)={corr:.4f}")

    df[["timestamp", "probA", "predB", "predC"]].to_parquet(
        ROOT / "tmp/btc_1h_volregime_20260805/btc5m_3head_tabm_pred.parquet", index=False)
    print("\nwrote predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
