#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import FullyLearnedGovernorConfig, build_training_set, prepare_features  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default  # noqa: E402
from scripts.eval_hf_v13_deep_tabular_parent_mdd_20260514 import (  # noqa: E402
    FTTransformerParent,
    ParentDataset,
    RuntimeConfig,
    _decisions_from_outputs,
    _normalise_apply,
    _normalise_fit,
    _predict_outputs,
    _train_model,
)
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import BASE_CLEAN_DIR, _flatten, _score  # noqa: E402


MODEL_ID = "alpha7_shared_backbone_ft_contract_test_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_shared_backbone_ft_contract_test_20260601"


class SharedMLPParent(nn.Module):
    def __init__(self, n_features: int, cfg: FullyLearnedGovernorConfig, hidden: int = 192, dropout: float = 0.15) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action_head = nn.Linear(hidden, 3)
        self.quality_head = nn.Linear(hidden, 1)
        self.bucket_heads = nn.ModuleDict(
            {
                "notional": nn.Linear(hidden, len(cfg.notional_buckets)),
                "leverage": nn.Linear(hidden, len(cfg.leverage_buckets)),
                "take_profit": nn.Linear(hidden, len(cfg.take_profit_buckets)),
                "stop_loss": nn.Linear(hidden, len(cfg.stop_loss_buckets)),
                "max_hold": nn.Linear(hidden, len(cfg.max_hold_buckets)),
                "cooldown": nn.Linear(hidden, len(cfg.cooldown_buckets)),
            }
        )
        self.loss_log_vars = nn.ParameterDict(
            {
                "action": nn.Parameter(torch.zeros(())),
                "quality": nn.Parameter(torch.zeros(())),
                "bucket": nn.Parameter(torch.zeros(())),
            }
        )

    def forward(self, x_tab: torch.Tensor, x_seq: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z = self.backbone(x_tab)
        out = {
            "action": self.action_head(z),
            "quality": self.quality_head(z).squeeze(-1),
        }
        out.update({k: head(z) for k, head in self.bucket_heads.items()})
        return out


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


def _train_one(
    name: str,
    model: nn.Module,
    *,
    train_ds: ParentDataset,
    val_ds: ParentDataset,
    x_val: np.ndarray,
    x_oos: np.ndarray,
    cfg: FullyLearnedGovernorConfig,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    epochs: int,
) -> dict[str, Any]:
    training = _train_model(name, model, train_ds, val_ds, epochs=int(epochs), device=device, batch_size=int(batch_size))
    val_out = _predict_outputs(model, x_val, None, device, int(batch_size), mc_passes=3, temperature=1.20)
    oos_out = _predict_outputs(model, x_oos, None, device, int(batch_size), mc_passes=3, temperature=1.20)
    rows: list[dict[str, Any]] = []
    payload: dict[str, pd.DataFrame] = {}
    for rt in _runtime_grid(name):
        dec = _decisions_from_outputs(val_out, cfg, rt, val_df.index)
        costs = _combo_metrics(val_df, dec)
        rows.append({"runtime": asdict(rt), "score": float(_score_contract(costs)), "validation": costs})
        payload[rt.name] = dec
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    rt = RuntimeConfig(**selected["runtime"])
    val_dec = payload[rt.name]
    oos_dec = _decisions_from_outputs(oos_out, cfg, rt, eval_df.index)
    selected["oos"] = _combo_metrics(eval_df, oos_dec)
    val_dec.to_csv(OUT_DIR / f"{name}_validation_decisions.csv", index=False)
    oos_dec.to_csv(OUT_DIR / f"{name}_oos_2026_decisions.csv", index=False)
    torch.save(
        {
            "model_id": MODEL_ID,
            "model_key": name,
            "state_dict": model.state_dict(),
            "training": training,
            "selected_runtime": selected["runtime"],
        },
        OUT_DIR / f"{name}.pt",
    )
    pd.DataFrame(
        [
            {
                "runtime": r["runtime"]["name"],
                "score": r["score"],
                **_flatten("val", r["validation"]),
            }
            for r in rows
        ]
    ).to_csv(OUT_DIR / f"{name}_ranking.csv", index=False)
    return {"model_key": name, "training": training, "selected": selected, "top_grid": rows[:10]}


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Small contract test for report options 3/4 on current Alpha7 data.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(6060103)
    torch.manual_seed(6060103)
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
    results = []
    results.append(
        _train_one(
            "shared_mlp",
            SharedMLPParent(len(feature_cols), cfg),
            train_ds=train_ds,
            val_ds=val_ds,
            x_val=x_val_full_norm,
            x_oos=x_oos_full_norm,
            cfg=cfg,
            val_df=val_df,
            eval_df=eval_df,
            device=device,
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
        )
    )
    results.append(
        _train_one(
            "ft_transformer",
            FTTransformerParent(len(feature_cols), cfg, d_model=64, n_layers=2),
            train_ds=train_ds,
            val_ds=val_ds,
            x_val=x_val_full_norm,
            x_oos=x_oos_full_norm,
            cfg=cfg,
            val_df=val_df,
            eval_df=eval_df,
            device=device,
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
        )
    )
    best = sorted(results, key=lambda r: float(r["selected"]["score"]), reverse=True)[0]
    report = {
        "model_id": MODEL_ID,
        "design": "Project-adapted report options 3/4. Trains standalone PyTorch shared-backbone and FT-Transformer lifecycle parents with the existing Alpha7 feature contract, lifecycle labels, and _combo_metrics backtest. No OOS selection.",
        "feature_cols": feature_cols,
        "train_meta": train_meta,
        "val_meta": val_meta,
        "overlay": overlay,
        "device": str(device),
        "results": results,
        "best_by_validation": best,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "best_by_validation": best["model_key"], "selected": best["selected"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
