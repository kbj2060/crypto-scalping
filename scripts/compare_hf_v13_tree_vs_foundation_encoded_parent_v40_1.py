#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronos import Chronos2Pipeline  # noqa: E402
from tsfm.model.kairos import AutoModel as KairosAutoModel  # noqa: E402

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    FullyLearnedGovernorConfig,
    build_training_set,
    prepare_features,
    train_policy,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _audit_contract,
    _close,
    _feature_cols,
    _json_default,
    _read,
    backtest_policy_frame,
)
from scripts.train_eval_hf_v13_multitrack_foundation_parent_v40 import (  # noqa: E402
    CHRONOS_MODEL,
    KAIROS_MODEL,
    MICRO_PRIORITY,
    QuantFastKANLayer,
    _extract_macro_embeddings,
    _extract_micro_embeddings,
    _parent_cfg,
    _state_cols,
)


MODEL_ID = "hf_v13_tree_vs_foundation_encoded_parent_v40_1_20260512"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_encoded_parent_v40_1_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_encoded_parent_v40_1_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_encoded_parent_v40_1_20260512_audit.json"


class StateKANTeacher(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.enc1 = QuantFastKANLayer(in_dim, 64, grid_size=8)
        self.enc2 = QuantFastKANLayer(64, hidden_dim, grid_size=8)
        self.action_head = nn.Linear(hidden_dim, 3)
        self.quality_head = nn.Linear(hidden_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc2(torch.nn.functional.gelu(self.enc1(x)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.action_head(z), self.quality_head(z).squeeze(-1)


def _fit_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = (np.nanstd(x, axis=0) + 1e-6).astype(np.float32)
    return mean, std


def _apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :]) / std[None, :]).astype(np.float32)


def _fit_state_kan(x: np.ndarray, action: np.ndarray, quality: np.ndarray, *, epochs: int, seed: int) -> tuple[StateKANTeacher, dict[str, np.ndarray]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    mean, std = _fit_norm(x)
    x_n = _apply_norm(x, mean, std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StateKANTeacher(x.shape[1]).to(device)
    counts = np.bincount(action.astype(np.int64), minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights[0] *= 0.40
    weights = weights / max(weights.mean(), 1e-6)
    ce = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    huber = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_n), torch.from_numpy(action.astype(np.int64)), torch.from_numpy(quality.astype(np.float32))),
        batch_size=256,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    model.train()
    for _ in range(int(epochs)):
        for xb, ab, qb in loader:
            xb, ab, qb = xb.to(device), ab.to(device), qb.to(device)
            logits, qhat = model(xb)
            loss = ce(logits, ab) + 1.5 * huber(qhat, qb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu().eval(), {"mean": mean, "std": std}


def _encode_state(model: StateKANTeacher, x: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    x_n = _apply_norm(x, norm["mean"], norm["std"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_n), 1024):
            z = model.encode(torch.from_numpy(x_n[start : start + 1024])).numpy().astype(np.float32)
            out.append(z)
    return np.vstack(out)


def _add_embedding_cols(base: pd.DataFrame, prefix: str, values: np.ndarray) -> pd.DataFrame:
    cols = [f"{prefix}_{j:03d}" for j in range(values.shape[1])]
    extra = pd.DataFrame(values, columns=cols, index=base.index)
    return pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare base HGB parent vs Chronos/Kairos/KAN encoded HGB parent.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--train-stride", type=int, default=48)
    p.add_argument("--kan-epochs", type=int, default=25)
    p.add_argument("--embed-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=2041)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading data", flush=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    audit = _audit_contract(train_all, eval_df, feature_cols)
    cfg: FullyLearnedGovernorConfig = _parent_cfg()

    print(f"[{MODEL_ID}] building baseline labels", flush=True)
    x_train_base, y, meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.train_stride), batch_size=512, feature_cols=feature_cols)
    valid = np.arange(0, max(0, len(train_df) - cfg.max_train_horizon_bars - 1), max(1, int(args.train_stride)), dtype=np.int64)
    if len(valid) != len(x_train_base):
        raise RuntimeError(f"valid/train mismatch: {len(valid)} vs {len(x_train_base)}")

    print(f"[{MODEL_ID}] preparing full features", flush=True)
    train_feat = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    val_feat = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_feat = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)

    print(f"[{MODEL_ID}] training baseline tree parent", flush=True)
    baseline_bundle = train_policy(x_train_base, y, cfg=cfg, random_state=int(args.seed), feature_cols=list(x_train_base.columns))

    state_cols = _state_cols(feature_cols, train_feat)
    micro_cols = [c for c in MICRO_PRIORITY if c in train_feat.columns]
    if not micro_cols:
        raise RuntimeError("no microstructure columns available for Kairos track")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{MODEL_ID}] loading Chronos-2 and Kairos_23m on {device}", flush=True)
    chronos = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map=device)
    kairos = KairosAutoModel.from_pretrained(KAIROS_MODEL, trust_remote_code=True).to(device).eval()

    emb_dir = ROOT / "data/ensemble/supervised/hf_v13_multitrack_foundation_parent_v40_20260512" / "embeddings"
    print(f"[{MODEL_ID}] extracting foundation embeddings", flush=True)
    train_macro = _extract_macro_embeddings(chronos, train_df, valid, cache_path=emb_dir / f"tree_train_macro_s{args.train_stride}.npy", batch_size=args.embed_batch)
    val_macro = _extract_macro_embeddings(chronos, val_df, np.arange(len(val_df), dtype=np.int64), cache_path=emb_dir / "val_macro.npy", batch_size=args.embed_batch)
    eval_macro = _extract_macro_embeddings(chronos, eval_df, np.arange(len(eval_df), dtype=np.int64), cache_path=emb_dir / "eval_macro.npy", batch_size=args.embed_batch)
    train_micro = _extract_micro_embeddings(kairos, train_feat, valid, micro_cols, cache_path=emb_dir / f"tree_train_micro_s{args.train_stride}.npy", batch_size=args.embed_batch)
    val_micro = _extract_micro_embeddings(kairos, val_feat, np.arange(len(val_feat), dtype=np.int64), micro_cols, cache_path=emb_dir / "val_micro.npy", batch_size=args.embed_batch)
    eval_micro = _extract_micro_embeddings(kairos, eval_feat, np.arange(len(eval_feat), dtype=np.int64), micro_cols, cache_path=emb_dir / "eval_micro.npy", batch_size=args.embed_batch)

    print(f"[{MODEL_ID}] fitting PCA + QuantFastKAN state encoder", flush=True)
    macro_pca = PCA(n_components=128, random_state=int(args.seed))
    micro_pca = PCA(n_components=32, random_state=int(args.seed))
    train_macro_128 = macro_pca.fit_transform(train_macro).astype(np.float32)
    val_macro_128 = macro_pca.transform(val_macro).astype(np.float32)
    eval_macro_128 = macro_pca.transform(eval_macro).astype(np.float32)
    train_micro_32 = micro_pca.fit_transform(train_micro).astype(np.float32)
    val_micro_32 = micro_pca.transform(val_micro).astype(np.float32)
    eval_micro_32 = micro_pca.transform(eval_micro).astype(np.float32)

    train_state = train_feat.iloc[valid].reindex(columns=state_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    val_state = val_feat.reindex(columns=state_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    eval_state = eval_feat.reindex(columns=state_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    state_model, state_norm = _fit_state_kan(train_state, np.asarray(y["action"]), np.asarray(y["quality"]), epochs=int(args.kan_epochs), seed=int(args.seed))
    train_state_32 = _encode_state(state_model, train_state, state_norm)
    val_state_32 = _encode_state(state_model, val_state, state_norm)
    eval_state_32 = _encode_state(state_model, eval_state, state_norm)

    print(f"[{MODEL_ID}] training encoded tree parent", flush=True)
    x_train_enc = x_train_base.reset_index(drop=True).copy()
    x_train_enc = _add_embedding_cols(x_train_enc, "macro_ctx", train_macro_128)
    x_train_enc = _add_embedding_cols(x_train_enc, "micro_dyn", train_micro_32)
    x_train_enc = _add_embedding_cols(x_train_enc, "state_kan", train_state_32)
    encoded_feature_cols = list(x_train_enc.columns)
    encoded_bundle = train_policy(x_train_enc, y, cfg=cfg, random_state=int(args.seed), feature_cols=encoded_feature_cols)

    val_enc = val_df.reset_index(drop=True).copy()
    val_enc = _add_embedding_cols(val_enc, "macro_ctx", val_macro_128)
    val_enc = _add_embedding_cols(val_enc, "micro_dyn", val_micro_32)
    val_enc = _add_embedding_cols(val_enc, "state_kan", val_state_32)
    eval_enc = eval_df.reset_index(drop=True).copy()
    eval_enc = _add_embedding_cols(eval_enc, "macro_ctx", eval_macro_128)
    eval_enc = _add_embedding_cols(eval_enc, "micro_dyn", eval_micro_32)
    eval_enc = _add_embedding_cols(eval_enc, "state_kan", eval_state_32)

    print(f"[{MODEL_ID}] backtesting baseline vs encoded", flush=True)
    baseline_val = {f"cost{k}": backtest_policy_frame(val_df, baseline_bundle, fee=cfg.fee * k, slip=cfg.slip * k) for k in (1, 2, 3)}
    baseline_eval = {f"cost{k}": backtest_policy_frame(eval_df, baseline_bundle, fee=cfg.fee * k, slip=cfg.slip * k) for k in (1, 2, 3)}
    encoded_val = {f"cost{k}": backtest_policy_frame(val_enc, encoded_bundle, fee=cfg.fee * k, slip=cfg.slip * k) for k in (1, 2, 3)}
    encoded_eval = {f"cost{k}": backtest_policy_frame(eval_enc, encoded_bundle, fee=cfg.fee * k, slip=cfg.slip * k) for k in (1, 2, 3)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "baseline_tree_parent.pkl").open("wb") as f:
        pickle.dump(baseline_bundle, f)
    with (args.out_dir / "encoded_tree_parent.pkl").open("wb") as f:
        pickle.dump(encoded_bundle, f)
    with (args.out_dir / "state_kan_teacher.pt").open("wb") as f:
        torch.save({"state_dict": state_model.state_dict(), "state_cols": state_cols, "norm": state_norm}, f)

    comparison = {
        "validation": {
            "baseline": baseline_val,
            "encoded": encoded_val,
            "delta_cost1_pnl": float(encoded_val["cost1"]["pnl"] - baseline_val["cost1"]["pnl"]),
            "delta_cost1_mdd": float(encoded_val["cost1"]["mdd"] - baseline_val["cost1"]["mdd"]),
        },
        "oos_2026": {
            "baseline": baseline_eval,
            "encoded": encoded_eval,
            "delta_cost1_pnl": float(encoded_eval["cost1"]["pnl"] - baseline_eval["cost1"]["pnl"]),
            "delta_cost1_mdd": float(encoded_eval["cost1"]["mdd"] - baseline_eval["cost1"]["mdd"]),
            "delta_cost2_pnl": float(encoded_eval["cost2"]["pnl"] - baseline_eval["cost2"]["pnl"]),
            "delta_cost3_pnl": float(encoded_eval["cost3"]["pnl"] - baseline_eval["cost3"]["pnl"]),
        },
    }

    blocking = list(audit.get("blocking", []))
    warnings = list(audit.get("warnings", []))
    final_audit = {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "feature_audit": audit,
        "comparison": comparison,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Fair tree-vs-tree comparison. Both models use the same HistGradientBoosting parent policy and same 2025 Jan-Sep training split. The challenger augments the base feature matrix with Chronos-2 macro PCA-128, Kairos_23m micro PCA-32, and QuantFastKAN state-32 encodings.",
        "split_policy": "train=2025 Jan-Sep, validation=2025 Q4, OOS=2026 fixed",
        "chronos_model": CHRONOS_MODEL,
        "kairos_model": KAIROS_MODEL,
        "macro_cols": ["open", "high", "low", "close", "volume"],
        "micro_cols": micro_cols,
        "state_cols": state_cols,
        "train_stride": int(args.train_stride),
        "training_meta": meta,
        "comparison": comparison,
        "audit": final_audit,
        "artifacts": {
            "baseline_bundle": str(args.out_dir / "baseline_tree_parent.pkl"),
            "encoded_bundle": str(args.out_dir / "encoded_tree_parent.pkl"),
            "state_kan_teacher": str(args.out_dir / "state_kan_teacher.pt"),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(final_audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "comparison": comparison}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
