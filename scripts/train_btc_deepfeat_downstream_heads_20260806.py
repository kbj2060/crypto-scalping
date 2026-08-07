"""Replace the plain nn.Linear direction/quality heads (baked into DeepFeatModel, trained
end-to-end with the encoder) with two separate downstream predictors trained on the FROZEN
transformer embedding: a TabM-style ensemble MLP and a tree-based model (HistGradientBoosting).

Base encoder: tmp/btc_deepfeat_sharpen_sweep/cw_0.2_quality/deepfeat_bundle.pt (the final tuned
transformer, window=48/d_model=96/n_layers=3, label_sharpen=0.7/cash_weight=0.2). Its own linear
heads reached val 67.1%/OOS 64.8% direction accuracy -- baseline for this comparison.

Embeddings are extracted once (encoder frozen, no fine-tuning) for train (train_stride=4, same
redundancy-cutting subsample the encoder itself was trained with -- consecutive-window embeddings
are as autocorrelated as the raw windows were), val, oos.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402
from ensemble.deep_features.btc_deepfeat_encoders_20260806 import build_model  # noqa: E402
from ensemble.deep_features.btc_deepfeat_tabm_head_20260806 import TabMEnsembleHead  # noqa: E402
from train_btc_deepfeat_encoders_20260806 import _prepare_target, _soft_ce_loss  # noqa: E402

CHECKPOINT = ROOT / "tmp/btc_deepfeat_sharpen_sweep/cw_0.2_quality/deepfeat_bundle.pt"
OUT_DIR = ROOT / "tmp/btc_deepfeat_downstream_heads_20260806"

LABEL_SHARPEN = 0.7
CASH_WEIGHT = 0.2
SEED = 20260806


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _extract_embeddings(encoder, ds, split: str, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    encoder.eval()
    row_idx = ds.end_idx[split]
    out = []
    for i in range(0, len(row_idx), batch_size):
        chunk = row_idx[i : i + batch_size]
        x = torch.from_numpy(ds.get_batch(chunk)).to(device)
        out.append(encoder(x).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def _train_tabm(emb_train, y_soft_train, y_hard_train, y_quality_train, emb_val, y_soft_val, y_hard_val, y_quality_val, emb_oos, y_hard_oos, y_quality_oos, device):
    model = TabMEnsembleHead(in_dim=emb_train.shape[1], n_experts=8, hidden=64, n_layers=2, dropout=0.1, n_classes=3, quality_head=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    et, ys, yh, yq = (torch.from_numpy(a) for a in (emb_train, y_soft_train, y_hard_train, y_quality_train))
    ev, ysv, yhv, yqv = (torch.from_numpy(a) for a in (emb_val, y_soft_val, y_hard_val, y_quality_val))
    n = et.shape[0]
    batch_size = 512
    rng = np.random.default_rng(SEED)

    best_val_loss, best_state, epochs_since_best = float("inf"), None, 0
    for epoch in range(1, 41):
        model.train()
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            chunk = idx[i : i + batch_size]
            xb, ysb, yhb, yqb = et[chunk].to(device), ys[chunk].to(device), yh[chunk].to(device), yq[chunk].to(device)
            target, weight = _prepare_target(ysb, yhb, LABEL_SHARPEN, CASH_WEIGHT)
            opt.zero_grad()
            logits, quality_pred = model(xb)
            loss = _soft_ce_loss(logits, target, weight) + 0.15 * F.mse_loss(quality_pred, yqb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            logits_v, quality_v = model(ev.to(device))
            target_v, weight_v = _prepare_target(ysv.to(device), yhv.to(device), LABEL_SHARPEN, CASH_WEIGHT)
            val_loss = float(_soft_ce_loss(logits_v, target_v, weight_v).item())
            val_acc = float((logits_v.argmax(dim=-1).cpu() == yhv).float().mean().item())
        scheduler.step(val_loss)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss, best_state, epochs_since_best = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= 8:
                break
        print(f"  tabm epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_v, quality_v = model(ev.to(device))
        logits_o, quality_o = model(torch.from_numpy(emb_oos).to(device))
        val_acc = float((logits_v.argmax(dim=-1).cpu().numpy() == y_hard_val).mean())
        oos_acc = float((logits_o.argmax(dim=-1).cpu().numpy() == y_hard_oos).mean())
        val_quality_mse = float(F.mse_loss(quality_v.cpu(), torch.from_numpy(y_quality_val)).item())
        oos_quality_mse = float(F.mse_loss(quality_o.cpu(), torch.from_numpy(y_quality_oos)).item())
    return {
        "model": "tabm_ensemble_head",
        "val_hard_top1_acc": val_acc,
        "oos_hard_top1_acc": oos_acc,
        "val_quality_mse": val_quality_mse,
        "oos_quality_mse": oos_quality_mse,
        "best_epoch_val_loss": best_val_loss,
    }, model


def _train_tree(emb_train, y_hard_train, y_quality_train, cash_weight, emb_val, y_hard_val, y_quality_val, emb_oos, y_hard_oos, y_quality_oos):
    sample_weight = np.where(y_hard_train == 0, cash_weight, 1.0)
    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=SEED)
    clf.fit(emb_train, y_hard_train, sample_weight=sample_weight)
    reg = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=SEED)
    reg.fit(emb_train, y_quality_train)

    def _eval(emb, y_hard, y_quality):
        pred = clf.predict(emb)
        proba = clf.predict_proba(emb)
        q_pred = reg.predict(emb)
        return {
            "hard_top1_acc": float((pred == y_hard).mean()),
            "log_loss": float(log_loss(y_hard, proba, labels=[0, 1, 2])),
            "quality_mse": float(np.mean((q_pred - y_quality) ** 2)),
        }

    val_m = _eval(emb_val, y_hard_val, y_quality_val)
    oos_m = _eval(emb_oos, y_hard_oos, y_quality_oos)
    return {
        "model": "hist_gradient_boosting",
        "val_hard_top1_acc": val_m["hard_top1_acc"],
        "oos_hard_top1_acc": oos_m["hard_top1_acc"],
        "val_log_loss": val_m["log_loss"],
        "oos_log_loss": oos_m["log_loss"],
        "val_quality_mse": val_m["quality_mse"],
        "oos_quality_mse": oos_m["quality_mse"],
    }, clf, reg


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = bundle["config"]
    full_model = build_model(
        config["arch"], config["n_features"], config["category_sizes"], embed_dim=config["embed_dim"],
        d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"],
        ffn_mult=config["ffn_mult"], dropout=config["dropout"], quality_head=config["quality_head"],
    )
    full_model.load_state_dict(bundle["model_state"])
    encoder = full_model.encoder.to(device)

    ds = build_dataset(window=config["window"], train_stride=config.get("train_stride", 4))

    emb_train = _extract_embeddings(encoder, ds, "train", device)
    emb_val = _extract_embeddings(encoder, ds, "val", device)
    emb_oos = _extract_embeddings(encoder, ds, "oos", device)

    y_soft_train = ds.y_soft_all[ds.end_idx["train"]]
    y_hard_train = ds.y_hard_all[ds.end_idx["train"]]
    y_quality_train = ds.y_quality_all[ds.end_idx["train"]]
    y_soft_val = ds.y_soft_all[ds.end_idx["val"]]
    y_hard_val = ds.y_hard_all[ds.end_idx["val"]]
    y_quality_val = ds.y_quality_all[ds.end_idx["val"]]
    y_hard_oos = ds.y_hard_all[ds.end_idx["oos"]]
    y_quality_oos = ds.y_quality_all[ds.end_idx["oos"]]

    print(f"embedding dims: train={emb_train.shape} val={emb_val.shape} oos={emb_oos.shape}")
    print("baseline (plain linear head, DeepFeatModel end-to-end): val_acc=0.6711 oos_acc=0.6477")

    print("=== training TabM ensemble head ===")
    tabm_result, tabm_model = _train_tabm(
        emb_train, y_soft_train, y_hard_train, y_quality_train,
        emb_val, y_soft_val, y_hard_val, y_quality_val,
        emb_oos, y_hard_oos, y_quality_oos, device,
    )
    print(json.dumps(tabm_result, indent=2))

    print("=== training tree-based (HistGradientBoosting) head ===")
    tree_result, clf, reg = _train_tree(
        emb_train, y_hard_train, y_quality_train, CASH_WEIGHT,
        emb_val, y_hard_val, y_quality_val,
        emb_oos, y_hard_oos, y_quality_oos,
    )
    print(json.dumps(tree_result, indent=2))

    torch.save({"model_state": tabm_model.state_dict(), "encoder_checkpoint": str(CHECKPOINT), "config": config}, OUT_DIR / "tabm_head_bundle.pt")
    import joblib
    joblib.dump({"clf": clf, "reg": reg, "encoder_checkpoint": str(CHECKPOINT), "config": config}, OUT_DIR / "tree_head_bundle.joblib")

    summary = {
        "baseline_linear_head": {"val_hard_top1_acc": 0.6711, "oos_hard_top1_acc": 0.6477},
        "tabm": tabm_result,
        "tree": tree_result,
    }
    (OUT_DIR / "downstream_heads_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
