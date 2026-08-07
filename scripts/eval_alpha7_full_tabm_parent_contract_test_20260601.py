#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import FullyLearnedGovernorConfig, build_training_set, prepare_features  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default  # noqa: E402
from scripts.eval_hf_v13_deep_tabular_parent_mdd_20260514 import (  # noqa: E402
    ParentDataset,
    RuntimeConfig,
    _decisions_from_outputs,
    _normalise_apply,
    _normalise_fit,
    _predict_outputs,
)
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import BASE_CLEAN_DIR, _flatten, _score  # noqa: E402


MODEL_ID = "alpha7_full_tabm_parent_contract_test_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_full_tabm_parent_contract_test_20260601"


class BatchEnsembleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int) -> None:
        super().__init__()
        self.ensemble_size = int(ensemble_size)
        self.weight = nn.Parameter(torch.empty(int(out_features), int(in_features)))
        self.bias = nn.Parameter(torch.empty(int(out_features)))
        self.r = nn.Parameter(torch.empty(self.ensemble_size, int(in_features)))
        self.s = nn.Parameter(torch.empty(self.ensemble_size, int(out_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.r, mean=1.0, std=0.05)
        nn.init.normal_(self.s, mean=1.0, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0) * self.r[:, None, :]
        elif x.ndim == 3:
            if x.shape[0] != self.ensemble_size:
                raise ValueError(f"expected first dim ensemble_size={self.ensemble_size}, got {tuple(x.shape)}")
            x = x * self.r[:, None, :]
        else:
            raise ValueError(f"BatchEnsembleLinear expects [B,D] or [K,B,D], got {tuple(x.shape)}")
        return F.linear(x, self.weight, self.bias) * self.s[:, None, :]


class TabMParent(nn.Module):
    def __init__(
        self,
        n_features: int,
        cfg: FullyLearnedGovernorConfig,
        *,
        hidden: int = 192,
        ensemble_size: int = 5,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.ensemble_size = int(ensemble_size)
        self.norm = nn.LayerNorm(int(n_features))
        self.tabm1 = BatchEnsembleLinear(int(n_features), int(hidden), self.ensemble_size)
        self.tabm2 = BatchEnsembleLinear(int(hidden), int(hidden), self.ensemble_size)
        self.dropout = nn.Dropout(float(dropout))
        self.action_head = nn.Linear(int(hidden), 3)
        self.quality_head = nn.Linear(int(hidden), 1)
        self.bucket_heads = nn.ModuleDict(
            {
                "notional": nn.Linear(int(hidden), len(cfg.notional_buckets)),
                "leverage": nn.Linear(int(hidden), len(cfg.leverage_buckets)),
                "take_profit": nn.Linear(int(hidden), len(cfg.take_profit_buckets)),
                "stop_loss": nn.Linear(int(hidden), len(cfg.stop_loss_buckets)),
                "max_hold": nn.Linear(int(hidden), len(cfg.max_hold_buckets)),
                "cooldown": nn.Linear(int(hidden), len(cfg.cooldown_buckets)),
            }
        )
        self.loss_log_vars = nn.ParameterDict(
            {
                "action": nn.Parameter(torch.zeros(())),
                "quality": nn.Parameter(torch.zeros(())),
                "bucket": nn.Parameter(torch.zeros(())),
            }
        )

    def _encode_members(self, x_tab: torch.Tensor) -> torch.Tensor:
        z = self.norm(x_tab)
        z = self.tabm1(z)
        z = F.gelu(z)
        z = self.dropout(z)
        z = self.tabm2(z)
        z = F.gelu(z)
        z = self.dropout(z)
        return z

    def forward(self, x_tab: torch.Tensor, x_seq: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        members = self._encode_members(x_tab)
        out = {
            "action": self.action_head(members).mean(dim=0),
            "quality": self.quality_head(members).squeeze(-1).mean(dim=0),
        }
        for key, head in self.bucket_heads.items():
            out[key] = head(members).mean(dim=0)
        return out


def _balanced(model: nn.Module, name: str, term: torch.Tensor) -> torch.Tensor:
    log_vars = getattr(model, "loss_log_vars", None)
    if log_vars is None or name not in log_vars:
        return term
    s = log_vars[name].clamp(-3.0, 3.0)
    return torch.exp(-s) * term + 0.5 * s


def _trade_biased_loss(model: nn.Module, outputs: dict[str, torch.Tensor], y: dict[str, torch.Tensor]) -> torch.Tensor:
    action = y["action"]
    active = action != 0
    action_weight = torch.ones(3, device=action.device)
    action_weight[0] = 0.12
    action_loss = F.cross_entropy(outputs["action"], action, weight=action_weight)
    q_weight = torch.where(active, torch.tensor(1.25, device=action.device), torch.tensor(0.15, device=action.device))
    quality_loss = (F.smooth_l1_loss(outputs["quality"], y["quality"], reduction="none") * q_weight).mean()
    bucket_loss = torch.zeros((), device=action.device)
    if bool(active.any()):
        for key in ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"):
            bucket_loss = bucket_loss + F.cross_entropy(outputs[key][active], y[key][active])
        bucket_loss = bucket_loss / 6.0
    return 1.75 * _balanced(model, "action", action_loss) + _balanced(model, "quality", quality_loss) + 0.75 * _balanced(model, "bucket", bucket_loss)


def _train_tabm_model(
    model: nn.Module,
    train_ds: ParentDataset,
    val_ds: ParentDataset,
    *,
    epochs: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, xs, yb in train_loader:
            xb = xb.to(device)
            yb = {k: v.to(device) for k, v in yb.items()}
            opt.zero_grad(set_to_none=True)
            loss = _trade_biased_loss(model, model(xb, None), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        model.eval()
        vtotal = 0.0
        vcount = 0
        with torch.no_grad():
            for xb, xs, yb in val_loader:
                xb = xb.to(device)
                yb = {k: v.to(device) for k, v in yb.items()}
                loss = _trade_biased_loss(model, model(xb, None), yb)
                vtotal += float(loss.item()) * len(xb)
                vcount += len(xb)
        tr = total / max(count, 1)
        va = vtotal / max(vcount, 1)
        history.append({"epoch": float(epoch), "train_loss": tr, "val_loss": va})
        print(f"[{MODEL_ID}] full_tabm_parent_tradebiased epoch={epoch:02d} train_loss={tr:.5f} val_loss={va:.5f}", flush=True)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return {"best_val_loss": float(best_val), "history": history, "loss": "trade_biased_cash_weight_0.12"}


def _runtime_grid(model_key: str) -> list[RuntimeConfig]:
    rows: list[RuntimeConfig] = []
    for conf in (0.20, 0.35, 0.50):
        for q_floor in (-0.010, 0.000, 0.010):
            for scale, cap in ((0.85, 2.10), (1.00, 3.00)):
                for unc in (0.070, 0.120):
                    rows.append(
                        RuntimeConfig(
                            name=f"{model_key}_c{conf:.2f}_q{q_floor:.3f}_s{scale:.2f}_cap{cap:.2f}_u{unc:.3f}",
                            model_key=model_key,
                            mode="replace",
                            confidence=float(conf),
                            quality_floor=float(q_floor),
                            notional_scale=float(scale),
                            max_notional=float(cap),
                            uncertainty_max=float(unc),
                        )
                    )
    return rows


def _score_contract(costs: dict[str, Any]) -> float:
    c3 = costs["cost3"]
    if int(c3["trades"]) < 30:
        return -1e9 + float(c3["pnl"])
    return float(_score(costs))


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Full TabM tabular parent contract test on current Alpha7 lifecycle labels.")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(6060104)
    torch.manual_seed(6060104)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    train_all, eval_df, overlay = _load_frames_with_risk()
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    primary = joblib.load(BASE_CLEAN_DIR / "primary_no_tp/parent.pkl")
    cfg = FullyLearnedGovernorConfig(**dict(primary["config"]))
    feature_cols = list(primary["feature_cols"])

    x_train, y_train, train_meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=feature_cols)
    x_val_label, y_val, val_meta = build_training_set(val_df, cfg=cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=feature_cols)
    x_train_norm, norm = _normalise_fit(x_train)
    x_val_label_norm = _normalise_apply(x_val_label, norm)
    x_val_full = prepare_features(val_df, side_hint=0, feature_cols=feature_cols, strict=True)
    x_oos_full = prepare_features(eval_df, side_hint=0, feature_cols=feature_cols, strict=True)
    x_val_full_norm = _normalise_apply(x_val_full, norm)
    x_oos_full_norm = _normalise_apply(x_oos_full, norm)

    train_ds = ParentDataset(x_train_norm.to_numpy(dtype=np.float32), y_train)
    val_ds = ParentDataset(x_val_label_norm.astype(np.float32), y_val)
    model = TabMParent(len(feature_cols), cfg, hidden=int(args.hidden), ensemble_size=int(args.ensemble_size))
    training = _train_tabm_model(model, train_ds, val_ds, epochs=int(args.epochs), device=device, batch_size=int(args.batch_size))
    val_out = _predict_outputs(model, x_val_full_norm, None, device, int(args.batch_size), mc_passes=3, temperature=1.20)
    oos_out = _predict_outputs(model, x_oos_full_norm, None, device, int(args.batch_size), mc_passes=3, temperature=1.20)

    rows: list[dict[str, Any]] = []
    payload: dict[str, pd.DataFrame] = {}
    for rt in _runtime_grid("full_tabm_parent"):
        dec = _decisions_from_outputs(val_out, cfg, rt, val_df.index)
        costs = _combo_metrics(val_df, dec)
        rows.append({"runtime": asdict(rt), "score": float(_score_contract(costs)), "validation": costs})
        payload[rt.name] = dec
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    selected_rt = RuntimeConfig(**selected["runtime"])
    val_dec = payload[selected_rt.name]
    oos_dec = _decisions_from_outputs(oos_out, cfg, selected_rt, eval_df.index)
    selected["oos"] = _combo_metrics(eval_df, oos_dec)

    val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame(
        [
            {
                "runtime": r["runtime"]["name"],
                "score": r["score"],
                **_flatten("val", r["validation"]),
            }
            for r in rows
        ]
    ).to_csv(OUT_DIR / "ranking.csv", index=False)
    torch.save(
        {
            "model_id": MODEL_ID,
            "architecture": "Full TabM tabular parent: BatchEnsemble hidden layers with shared heads averaged across ensemble members.",
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "normalizer": norm,
            "config": dict(primary["config"]),
            "training": training,
            "selected_runtime": selected["runtime"],
        },
        OUT_DIR / "full_tabm_parent.pt",
    )
    report = {
        "model_id": MODEL_ID,
        "design": "Full TabM tabular parent contract test. Unlike the previous TabM-CryptoMamba frontend test, this BatchEnsemble model owns the lifecycle parent heads directly. Existing Alpha7 feature contract, lifecycle labels, and _combo_metrics are unchanged. OOS is evaluated once after validation selection.",
        "feature_cols": feature_cols,
        "train_meta": train_meta,
        "val_meta": val_meta,
        "overlay": overlay,
        "device": str(device),
        "training": training,
        "selected": selected,
        "top_grid": rows[:10],
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "model": str(OUT_DIR / "full_tabm_parent.pt"),
            "validation_decisions": str(OUT_DIR / "validation_decisions.csv"),
            "oos_decisions": str(OUT_DIR / "oos_2026_decisions.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
