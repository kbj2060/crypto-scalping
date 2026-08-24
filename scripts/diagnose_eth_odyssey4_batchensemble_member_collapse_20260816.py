#!/usr/bin/env python3
"""BatchEnsemble k=8 member-collapse diagnostic for the deployed ThreeHeadTabM (R-only) architecture.

Loads a real trained 3-head TabM bundle (bull/bear/chop experts) and measures, on real OOS feature
data, how correlated/diverse the k=8 ensemble members' per-member logits actually are. This is the
"cheap, concrete, actionable diagnostic" flagged in feedback_modern_dl_training_checklist as never
measured for this project's TabM (only surveyed as a literature concern, arXiv:2601.16936).

Analysis-only: no training, no GPU needed. Read-only against an existing checkpoint + real OOS data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eth_odyssey4_true_feature_pipeline_20260816 as truepipe  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as base3head  # noqa: E402

DEFAULT_BUNDLE = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_3head_parent72_loose_entry_quality_20260620_pinned102_zig075_formal5seed_20260815_seed946043153"
    / "true_3head_tabm_bundle.pt"
)


def _rebuild_model(expert_payload: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg_dict = expert_payload["config"]
    cfg = base3head.ThreeHeadConfig(
        k=int(cfg_dict["k"]),
        hidden=int(cfg_dict["hidden"]),
        layers=int(cfg_dict["layers"]),
        dropout=float(cfg_dict["dropout"]),
        batch_size=int(cfg_dict["batch_size"]),
        lr=float(cfg_dict["lr"]),
        weight_decay=float(cfg_dict["weight_decay"]),
        patience=int(cfg_dict["patience"]),
        exit_loss_weight=float(cfg_dict["exit_loss_weight"]),
        quality_loss_weight=float(cfg_dict["quality_loss_weight"]),
    )
    model = base3head.ThreeHeadTabM(int(expert_payload["n_features"]), cfg=cfg)
    model.load_state_dict(expert_payload["state_dict"])
    model.eval()
    return model, cfg_dict


def _member_stats(logits_k: torch.Tensor) -> dict[str, Any]:
    """logits_k: (n, k, C). Returns pairwise-correlation and agreement diagnostics."""
    probs_k = torch.softmax(logits_k, dim=-1)  # (n, k, C)
    n, k, c = probs_k.shape
    pred_k = probs_k.argmax(dim=-1)  # (n, k)

    # Pairwise Pearson correlation of each member's max-class probability vector (flattened over n,
    # per class channel averaged) -- a cheap scalar proxy for "do members move together".
    top_prob = probs_k.max(dim=-1).values  # (n, k) -- confidence trace per member
    top_np = top_prob.detach().cpu().numpy()
    corr = np.corrcoef(top_np.T)  # (k, k)
    iu = np.triu_indices(k, k=1)
    pairwise_corr = corr[iu]

    # Unanimity: fraction of rows where all k members agree on argmax class.
    pred_np = pred_k.detach().cpu().numpy()
    unanimous = (pred_np == pred_np[:, [0]]).all(axis=1).mean()

    # Expected unanimity under a null of k INDEPENDENT copies of the same marginal class
    # distribution (upper-bound-ish baseline: how much unanimity comes "for free" just from a
    # shared skewed marginal, vs from members being literally the same function).
    class_freq = np.stack([(pred_np == cls).mean(axis=0) for cls in range(c)], axis=0)  # (C, k)
    mean_class_freq = class_freq.mean(axis=1)  # (C,) marginal averaged across members
    independent_unanimity = float((mean_class_freq**k).sum())

    # Per-class-column std across members of predicted probability, averaged over samples/classes --
    # near 0 means members literally collapsed to the same function.
    cross_member_std = probs_k.std(dim=1).mean().item()

    return {
        "n_samples": int(n),
        "k": int(k),
        "n_classes": int(c),
        "mean_pairwise_corr_of_top_confidence": float(np.mean(pairwise_corr)),
        "min_pairwise_corr_of_top_confidence": float(np.min(pairwise_corr)),
        "max_pairwise_corr_of_top_confidence": float(np.max(pairwise_corr)),
        "argmax_unanimity_rate": float(unanimous),
        "independent_baseline_unanimity_rate": independent_unanimity,
        "unanimity_excess_over_independent_baseline": float(unanimous - independent_unanimity),
        "mean_cross_member_prob_std": cross_member_std,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--split", choices=["val", "oos"], default="oos")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"[diag] loading bundle: {args.bundle}", file=sys.stderr)
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    base_cols = list(bundle["base_cols"])

    print("[diag] preparing real true-feature-pipeline frames (this reads real OHLCV/label CSVs)...", file=sys.stderr)
    frames = truepipe.prepare_frames_true(disable_tp_sl=True)
    raw = frames["val_raw"] if args.split == "val" else frames["oos_raw"]
    x_df = base3head._base_input(raw, base_cols)

    results: dict[str, Any] = {"bundle": str(args.bundle), "split": args.split}
    for expert_name in ("bull", "bear", "chop"):
        payload = bundle["models"][expert_name]
        model, cfg_dict = _rebuild_model(payload)
        scaler = payload["scaler"]
        x_np = base3head._standardize_apply(x_df, scaler)
        with torch.no_grad():
            h = model.encode(torch.from_numpy(x_np))
            dir_logits_k = model.direction_head(h)  # (n, k, 3)
            qual_logits_k = model.quality_head(h)  # (n, k, 3)
        dir_stats = _member_stats(dir_logits_k)
        qual_stats = _member_stats(qual_logits_k)
        results[expert_name] = {
            "n_features": int(payload["n_features"]),
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
            "direction_head": dir_stats,
            "quality_head": qual_stats,
        }
        print(
            f"[diag] {expert_name}: dir mean_pairwise_corr={dir_stats['mean_pairwise_corr_of_top_confidence']:.4f} "
            f"unanimity={dir_stats['argmax_unanimity_rate']:.4f} "
            f"(independent_baseline={dir_stats['independent_baseline_unanimity_rate']:.4f}) "
            f"cross_member_prob_std={dir_stats['mean_cross_member_prob_std']:.4f}",
            file=sys.stderr,
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"[diag] wrote {args.out}", file=sys.stderr)
    else:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
