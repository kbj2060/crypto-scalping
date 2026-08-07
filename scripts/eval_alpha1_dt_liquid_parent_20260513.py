#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT, prepare_features, predict_policy_frame  # noqa: E402
from scripts import eval_alpha1_rl_exit_and_sizing_20260513 as alpha1  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha1_dt_liquid_parent_20260513"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_dt_liquid_parent_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_dt_liquid_parent_20260513_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_dt_liquid_parent_20260513_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_dt_liquid_parent_20260513_grid.csv"

SEQ_LEN = 48
HORIZONS = (12, 24, 48, 96)
NOTIONAL_BUCKETS = np.asarray([0.575, 0.8625, 1.2075, 1.6675, 2.3], dtype=np.float32)


@dataclass(frozen=True)
class DTConfig:
    name: str
    prob_th: float
    edge_th: float
    margin_th: float
    max_notional: float
    fixed_leverage: float
    take_profit: float
    stop_loss: float
    max_hold: int
    cooldown: int


def _grid() -> list[DTConfig]:
    rows: list[DTConfig] = []
    for p in (0.44, 0.50, 0.56, 0.62):
        for edge in (0.0015, 0.0030, 0.0050):
            rows.append(DTConfig(f"dt_liquid_p{p:.2f}_e{edge:.4f}", p, edge, 0.0010, 2.3, 2.0, 0.040, 0.018, 48, 12))
    return rows


class LiquidTimeGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.tau = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.mix = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = torch.zeros_like(x[:, 0])
        outs: list[torch.Tensor] = []
        for t in range(x.shape[1]):
            inp = self.mix(x[:, t])
            gate = self.tau(torch.cat([inp, torch.abs(inp - state)], dim=-1))
            state = state + gate * (inp - state)
            outs.append(state)
        return torch.stack(outs, dim=1)


class DTLiquidParent(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 96, n_notional: int = len(NOTIONAL_BUCKETS)) -> None:
        super().__init__()
        self.state_proj = nn.Linear(input_dim, d_model)
        self.prev_action = nn.Embedding(3, d_model)
        self.cond_proj = nn.Linear(3, d_model)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        self.liquid = LiquidTimeGate(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 3,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, 3)
        self.notional_head = nn.Linear(d_model, n_notional)
        self.edge_head = nn.Linear(d_model, 2)

    def forward(self, seq: torch.Tensor, prev_action: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.state_proj(seq) + self.prev_action(prev_action.long()) + self.cond_proj(cond) + self.pos[:, -seq.shape[1] :]
        h = self.liquid(h)
        mask = torch.triu(torch.ones(h.shape[1], h.shape[1], device=h.device, dtype=torch.bool), diagonal=1)
        z = self.encoder(h, mask=mask)
        ctx = self.norm(z[:, -1])
        return self.action_head(ctx), self.notional_head(ctx), self.edge_head(ctx)


class SeqDataset(Dataset):
    def __init__(self, features: np.ndarray, action: np.ndarray, notional: np.ndarray, edge: np.ndarray, cond: np.ndarray, indices: np.ndarray) -> None:
        self.features = features.astype(np.float32)
        self.action = action.astype(np.int64)
        self.notional = notional.astype(np.int64)
        self.edge = edge.astype(np.float32)
        self.cond = cond.astype(np.float32)
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        end = int(self.indices[idx])
        start = end - SEQ_LEN + 1
        x = self.features[start : end + 1]
        c = self.cond[start : end + 1]
        prev = np.zeros(SEQ_LEN, dtype=np.int64)
        if start > 0:
            prev[:] = self.action[start - 1 : end]
        else:
            prev[1:] = self.action[start:end]
        return (
            torch.from_numpy(x),
            torch.from_numpy(prev),
            torch.from_numpy(c),
            torch.tensor(self.action[end], dtype=torch.long),
            torch.tensor(self.notional[end], dtype=torch.long),
            torch.from_numpy(self.edge[end]),
        )


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_targets(df: pd.DataFrame, *, fee: float, slip: float, cost_mult: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = _close(df)
    n = len(df)
    action = np.zeros(n, dtype=np.int64)
    notional = np.zeros(n, dtype=np.int64)
    edge = np.zeros((n, 2), dtype=np.float32)
    cond = np.zeros((n, 3), dtype=np.float32)
    cost = 2.0 * float(fee + slip) * float(cost_mult)
    for i in range(SEQ_LEN, max(SEQ_LEN, n - max(HORIZONS) - 2)):
        entry_i = min(i + 1, n - 1)
        le = _fill_price(df, entry_i, 1, slip * cost_mult, entry=True)
        se = _fill_price(df, entry_i, -1, slip * cost_mult, entry=True)
        long_rs: list[float] = []
        short_rs: list[float] = []
        long_path: list[float] = []
        short_path: list[float] = []
        for h in HORIZONS:
            exit_i = min(i + h, n - 1)
            lx = _fill_price(df, exit_i, 1, slip * cost_mult, entry=False)
            sx = _fill_price(df, exit_i, -1, slip * cost_mult, entry=False)
            long_rs.append((lx - le) / max(le, 1e-12) - cost)
            short_rs.append((se - sx) / max(se, 1e-12) - cost)
        fut = close[i + 1 : min(i + max(HORIZONS) + 1, n)]
        if len(fut):
            long_path = list(fut / max(le, 1e-12) - 1.0)
            short_path = list(se / np.maximum(fut, 1e-12) - 1.0)
        long_best = float(max(long_rs))
        short_best = float(max(short_rs))
        long_cvar = float(np.quantile(long_path, 0.10)) if long_path else 0.0
        short_cvar = float(np.quantile(short_path, 0.10)) if short_path else 0.0
        edge[i] = [long_best, short_best]
        best = max(long_best, short_best)
        worst_cvar = long_cvar if long_best >= short_best else short_cvar
        cond[i] = [max(best, 0.0), abs(min(worst_cvar, 0.0)), cost_mult / 3.0]
        if best < 0.003 or worst_cvar < -0.040:
            action[i] = ACTION_CASH
            notional[i] = 0
        elif long_best >= short_best:
            action[i] = ACTION_LONG
        else:
            action[i] = ACTION_SHORT
        if action[i] != ACTION_CASH:
            score = max(best, 0.0) / max(abs(worst_cvar), 0.006)
            notional[i] = int(np.clip(np.searchsorted([0.35, 0.80, 1.40, 2.20], score, side="right"), 0, len(NOTIONAL_BUCKETS) - 1))
    return action, notional, edge, cond


def _feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    feats = prepare_features(df, side_hint=0, close=_close(df), feature_cols=feature_cols)
    x = feats.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = (np.nanstd(x, axis=0) + 1e-6).astype(np.float32)
    return ((x - mean) / std).astype(np.float32), {"mean": mean, "std": std, "feature_cols": list(feats.columns)}


def _apply_feature_matrix(df: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    feats = prepare_features(df, side_hint=0, close=_close(df), feature_cols=list(norm["feature_cols"]))
    x = feats.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    return ((x - norm["mean"]) / norm["std"]).astype(np.float32)


def _train_model(x: np.ndarray, action: np.ndarray, notional: np.ndarray, edge: np.ndarray, cond: np.ndarray, *, epochs: int = 8) -> DTLiquidParent:
    torch.manual_seed(20260513)
    idx = np.arange(SEQ_LEN, len(x) - max(HORIZONS) - 2, 3, dtype=np.int64)
    rng = np.random.default_rng(20260513)
    if len(idx) > 42000:
        idx = rng.choice(idx, size=42000, replace=False)
    ds = SeqDataset(x, action, notional, edge, cond, np.sort(idx))
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)
    device = _device()
    model = DTLiquidParent(x.shape[1]).to(device)
    counts = np.bincount(action[idx], minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights[0] *= 0.55
    weights = weights / max(weights.mean(), 1e-6)
    ce_action = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    ce_notional = nn.CrossEntropyLoss()
    huber = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    for ep in range(int(epochs)):
        total = 0.0
        for xb, prev, cb, ya, yn, ye in loader:
            xb, prev, cb, ya, yn, ye = xb.to(device), prev.to(device), cb.to(device), ya.to(device), yn.to(device), ye.to(device)
            logits, nlogits, epred = model(xb, prev, cb)
            active = ya != ACTION_CASH
            loss = ce_action(logits, ya) + 0.35 * huber(epred, ye)
            if active.any():
                loss = loss + 0.25 * ce_notional(nlogits[active], yn[active])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu())
        print(f"[{MODEL_ID}] epoch={ep+1} loss={total/max(len(loader),1):.5f}", flush=True)
    return model.cpu().eval()


def _predict_parent(model: DTLiquidParent, x: np.ndarray, cond: np.ndarray, cfg: DTConfig) -> pd.DataFrame:
    device = _device()
    model = model.to(device).eval()
    n = len(x)
    actions = np.zeros(n, dtype=np.int64)
    sides = np.zeros(n, dtype=np.int64)
    notionals = np.zeros(n, dtype=np.float64)
    leverages = np.ones(n, dtype=np.float64) * float(cfg.fixed_leverage)
    confidence = np.zeros(n, dtype=np.float64)
    quality = np.zeros(n, dtype=np.float64)
    with torch.no_grad():
        for start in range(SEQ_LEN, n, 4096):
            ends = np.arange(start, min(n, start + 4096), dtype=np.int64)
            seqs = np.stack([x[e - SEQ_LEN + 1 : e + 1] for e in ends])
            conds = np.stack([cond[e - SEQ_LEN + 1 : e + 1] for e in ends])
            prev = np.zeros((len(ends), SEQ_LEN), dtype=np.int64)
            logits, nlogits, epred = model(torch.from_numpy(seqs).to(device), torch.from_numpy(prev).to(device), torch.from_numpy(conds).to(device))
            prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            nb = torch.argmax(nlogits, dim=-1).detach().cpu().numpy()
            ev = epred.detach().cpu().numpy()
            best = np.argmax(prob, axis=1)
            for j, i in enumerate(ends):
                act = int(best[j])
                side = 1 if act == ACTION_LONG else -1 if act == ACTION_SHORT else 0
                edge_val = float(ev[j, 0] if side > 0 else ev[j, 1] if side < 0 else 0.0)
                margin = float(abs(ev[j, 0] - ev[j, 1]))
                if act == ACTION_CASH or prob[j, act] < cfg.prob_th or edge_val < cfg.edge_th or margin < cfg.margin_th:
                    continue
                actions[i] = act
                sides[i] = side
                confidence[i] = float(prob[j, act])
                quality[i] = edge_val
                notionals[i] = min(float(NOTIONAL_BUCKETS[int(nb[j])]), float(cfg.max_notional))
    return pd.DataFrame(
        {
            "action": actions,
            "side": sides,
            "notional_exposure": notionals,
            "leverage": leverages,
            "position_fraction": np.where(leverages > 0, notionals / leverages, 0.0),
            "take_profit": np.where(actions != ACTION_CASH, float(cfg.take_profit), 0.0),
            "stop_loss": np.where(actions != ACTION_CASH, float(cfg.stop_loss), 0.0),
            "max_hold_bars": np.where(actions != ACTION_CASH, int(cfg.max_hold), 0).astype(np.int64),
            "cooldown_bars": np.where(actions != ACTION_CASH, int(cfg.cooldown), 0).astype(np.int64),
            "quality_score": quality,
            "confidence": confidence,
        }
    )


def _predict_v27_fast(model: torch.nn.Module, df: pd.DataFrame, seq_cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    device = _device()
    model = model.to(device).eval()
    arr = df.loc[:, seq_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    pad = np.zeros((v31.SEQ_LEN - 1, arr.shape[1]), dtype=np.float32)
    padded = np.vstack([pad, arr])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=v31.SEQ_LEN, axis=0)
    if windows.shape[1] == arr.shape[1]:
        windows = windows.transpose(0, 2, 1)
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(df), 4096):
            seqs = np.ascontiguousarray(windows[start : start + 4096])
            xx = ((seqs - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)
            outs.append(model(torch.from_numpy(xx).to(device)).detach().cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.40 * c2["pnl"] + 0.20 * c3["pnl"] - 0.30 * abs(c1["mdd"]))


def main() -> int:
    print(f"[{MODEL_ID}] loading artifacts", flush=True)
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base = dict(parent["config"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    print(f"[{MODEL_ID}] building CVaR/cost-stressed targets", flush=True)
    x_train, feat_norm = _feature_matrix(train, feature_cols)
    a_train, n_train, e_train, c_train = _build_targets(train, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0)
    model = _train_model(x_train, a_train, n_train, e_train, c_train, epochs=8)
    print(f"[{MODEL_ID}] preparing validation/eval matrices", flush=True)
    x_val = _apply_feature_matrix(val, feat_norm)
    _, _, _, c_val = _build_targets(val, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0)
    x_eval = _apply_feature_matrix(eval_df, feat_norm)
    _, _, _, c_eval = _build_targets(eval_df, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0)
    val_q = _predict_v27_fast(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = _predict_v27_fast(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_alpha1_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    grid_rows: list[dict[str, Any]] = []
    selected: DTConfig | None = None
    best_score = -1e18
    for cfg in _grid():
        val_dec = _predict_parent(model, x_val, c_val, cfg)
        v1 = alpha1.backtest_alpha1(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec)
        v2 = alpha1.backtest_alpha1(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec)
        v3 = alpha1.backtest_alpha1(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec)
        score = _score(v1, v2, v3)
        grid_rows.append({**asdict(cfg), "score": score, "val_pnl": v1["pnl"], "val_mdd": v1["mdd"], "val_trades": v1["trades"], "val_c2_pnl": v2["pnl"], "val_c3_pnl": v3["pnl"]})
        if score > best_score:
            best_score = score
            selected = cfg
    assert selected is not None
    experiments = []
    for name, dec in (
        ("alpha1", eval_alpha1_dec),
        (f"dt_liquid_parent::{selected.name}", _predict_parent(model, x_eval, c_eval, selected)),
    ):
        metrics = {
            f"cost{mult}": alpha1.backtest_alpha1(eval_df, parent, jackpot_model, add_cfg, eval_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=dec)
            for mult in (1, 2, 3)
        }
        experiments.append({"name": name, "metrics": metrics, "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])})
        print(f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "dt_liquid_parent.pt"
    torch.save({"model_id": MODEL_ID, "state_dict": model.state_dict(), "feature_norm": feat_norm, "selected_config": asdict(selected), "seq_len": SEQ_LEN, "notional_buckets": NOTIONAL_BUCKETS, "target": "cost3_cvar_future_utility"}, model_path)
    pd.DataFrame(grid_rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)
    best = max(experiments, key=lambda x: x["score"])
    blocking = list(parent_audit.get("blocking", []))
    warnings = list(parent_audit.get("warnings", []))
    if best["name"] != "alpha1" and best["metrics"]["cost1"]["pnl"] <= alpha1.ALPHA1_BASELINE["cost1"]["pnl"]:
        warnings.append("dt_liquid_parent_did_not_beat_alpha1_cost1")
    if best["metrics"]["cost2"]["pnl"] <= 0.0:
        warnings.append("best_cost2_not_survived")
    if best["metrics"]["cost3"]["pnl"] <= 0.0:
        warnings.append("best_cost3_not_survived")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best["name"] != "alpha1" and best["metrics"]["cost1"]["pnl"] > alpha1.ALPHA1_BASELINE["cost1"]["pnl"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "parent_replacement": True,
        "architecture": "Decision Transformer style sequence policy + liquid time gate + CVaR/cost3 future-utility targets",
        "selected_config": asdict(selected),
        "parent_audit": parent_audit,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Experimental alpha1 parent replacement. DT-style sequence policy conditions on desired return/downside/cost, uses a liquid time-constant gate before a causal Transformer, and predicts CASH/LONG/SHORT plus notional bucket. TP/SL/hold are fixed for this first low-interference parent replacement test.",
        "selected_config": asdict(selected),
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"model": str(model_path), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
