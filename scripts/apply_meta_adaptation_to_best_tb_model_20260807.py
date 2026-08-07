"""Apply ONLY stage 3 (Reptile regime meta-learning) from
train_btc_regime_curriculum_meta_transformer_20260807.py directly to this session's actual best
triple-barrier model (tmp/btc_deepfeat_tripbarrier_20260806/flatsmooth_cw_0.9/, trained from
scratch with no zigzag curriculum: OOS win rate 35.5%/sum_ret -9.5%) -- skipping stages 1-2, since
the zigzag-curriculum pretraining was shown to hurt (stage2-alone OOS win rate 33.0%/-46.9%,
worse than training from scratch). This isolates whether Reptile regime adaptation helps on top of
the actually-best starting point, rather than on top of the curriculum-damaged one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402
from ensemble.deep_features.btc_deepfeat_encoders_20260806 import SupervisedTransformerEncoder  # noqa: E402
import train_btc_regime_curriculum_meta_transformer_20260807 as meta_mod  # noqa: E402

BEST_CHECKPOINT = ROOT / "tmp/btc_deepfeat_tripbarrier_20260806/flatsmooth_cw_0.9/deepfeat_bundle.pt"
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_DIR = ROOT / "tmp/btc_meta_adaptation_on_best_model_20260807"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_mod._seed_everything(20260806)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
    config = bundle["config"]

    ds = build_dataset(
        window=config["window"], train_stride=config["train_stride"],
        label_path=config["label_path"], hard_col=config["hard_col"],
        soft_cols=config["soft_cols"].split(","),
    )

    encoder = SupervisedTransformerEncoder(
        config["n_features"], d_model=config["d_model"], n_heads=config["n_heads"],
        n_layers=config["n_layers"], dropout=config["dropout"], embed_dim=config["embed_dim"],
    ).to(device)
    encoder_state = {k[len("encoder."):]: v for k, v in bundle["model_state"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)

    tb_head = nn.Linear(config["embed_dim"], 3).to(device)
    tb_head.load_state_dict({"weight": bundle["model_state"]["head.weight"], "bias": bundle["model_state"]["head.bias"]})

    baseline_val = meta_mod._evaluate_head(encoder, tb_head, ds, ds.y_hard_all, ds.y_soft_all, "val", device, 512, meta_mod.TB_CASH_WEIGHT)
    baseline_oos = meta_mod._evaluate_head(encoder, tb_head, ds, ds.y_hard_all, ds.y_soft_all, "oos", device, 512, meta_mod.TB_CASH_WEIGHT)
    print(f"[loaded best model] val_acc={baseline_val['acc']:.4f} oos_acc={baseline_oos['acc']:.4f}")

    import copy
    print("=== Stage 3 only: Reptile regime meta-learning on the best pretrained model ===")
    meta_head = meta_mod._reptile_meta_train(encoder, copy.deepcopy(tb_head), ds, device, 20260808)

    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)
    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(12).sum().to_numpy()
    vol = pd.Series(cumret).rolling(288, min_periods=288).std().to_numpy()
    tp_moves_all, sl_moves_all = meta_mod.TP_MULT * vol, meta_mod.SL_MULT * vol

    rng = np.random.default_rng(20260809)
    results = {}
    for split in ("val", "oos"):
        row_idx = ds.end_idx[split]
        pred_no_meta = np.concatenate([
            meta_mod._predict_with_head(encoder, tb_head, ds, row_idx[i : i + 512], device)
            for i in range(0, len(row_idx), 512)
        ])
        results[f"{split}_no_meta_baseline"] = meta_mod._backtest(row_idx, pred_no_meta, panel, tp_moves_all, sl_moves_all)
        print(f"{split}/no_meta_baseline:", json.dumps(results[f"{split}_no_meta_baseline"]))

        row_idx_meta, pred_meta = meta_mod._walk_forward_predict(encoder, meta_head, ds, split, device, rng)
        results[f"{split}_meta_adapted"] = meta_mod._backtest(row_idx_meta, pred_meta, panel, tp_moves_all, sl_moves_all)
        print(f"{split}/meta_adapted:", json.dumps(results[f"{split}_meta_adapted"]))

    (OUT_DIR / "results.json").write_text(
        json.dumps({"baseline_val_acc": baseline_val, "baseline_oos_acc": baseline_oos, "backtest": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
