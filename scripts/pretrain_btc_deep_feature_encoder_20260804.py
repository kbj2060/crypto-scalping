"""Stage B (pretrain) of the BTC deep-feature-encoder plan: unsupervised JEPA
pretraining of ensemble/deep_features/tabular_jepa_encoder.TabularJEPAEncoder on the
unified raw panel (scripts/build_btc_unified_raw_panel_20260804.py).

Causal by construction: pretraining only sees rows before VAL_START (no label ever
touched, no future info -- each window looks backward only), consistent with the
Fresh-Forward causal-availability rule in CLAUDE.md. After pretraining, the encoder
is frozen and used to emit embeddings for the *entire* timeline (train/VAL/OOS) --
each embedding still only depends on that bar's own backward-looking window, so this
does not leak VAL/OOS information into pretraining.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble.deep_features.tabular_jepa_encoder import JEPAConfig, TabularJEPAEncoder, WindowDataset  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_unified_raw_panel_20260804.parquet"
CKPT_PATH = ROOT / "data/ensemble/supervised/btc_deepfeat_jepa_encoder_20260804.pt"
EMB_OUT_PATH = ROOT / "data/splits/year_oos/btc_deepfeat_embeddings_20260804.parquet"

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
EXCLUDE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "close_btc",
    "volume_btc", "quote_volume_btc",
    "mtf1h_ts_t_value", "mtf1h_ts_opt_L",
}
WINDOW = 32
BATCH_SIZE = 256
EPOCHS = 8
LR = 2e-4
EMBED_DIM = 24


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]
    n_feat = len(feat_cols)
    print(f"{len(frame)} rows, {n_feat} feature cols")

    train_mask = frame["timestamp"] < VAL_START
    raw = frame[feat_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = raw.loc[train_mask].median()
    raw = raw.fillna(medians).fillna(0.0)
    q25 = raw.loc[train_mask].quantile(0.25)
    q75 = raw.loc[train_mask].quantile(0.75)
    scale = (q75 - q25).replace(0, 1.0)
    center = raw.loc[train_mask].median()
    standardized = ((raw - center) / scale).clip(-8.0, 8.0).to_numpy(dtype=np.float32)

    train_features = standardized[train_mask.to_numpy()]
    train_timestamps = frame.loc[train_mask, "timestamp"].to_numpy()
    print(f"pretrain rows (< VAL_START only): {len(train_features)}")

    train_ds = WindowDataset(train_features, train_timestamps, window=WINDOW)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    cfg = JEPAConfig(n_features=n_feat, window=WINDOW, embed_dim=EMBED_DIM)
    model = TabularJEPAEncoder(cfg).to(device)
    opt = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()) + list(model.readout.parameters()),
        lr=LR, weight_decay=1e-4,
    )

    model.train()
    for epoch in range(EPOCHS):
        t0 = time.time()
        tot_jepa, tot_con, n_batches = 0.0, 0.0, 0
        for x, _end in train_loader:
            x = x.to(device)
            loss_jepa, loss_con = model.forward_pretrain(x)
            loss = loss_jepa + cfg.contrastive_weight * loss_con
            opt.zero_grad()
            loss.backward()
            opt.step()
            model._update_target_ema()
            tot_jepa += loss_jepa.item()
            tot_con += loss_con.item()
            n_batches += 1
        print(f"epoch {epoch}: jepa={tot_jepa/n_batches:.4f} contrastive={tot_con/n_batches:.4f} "
              f"({time.time()-t0:.1f}s, {n_batches} batches)")

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": cfg,
        "feat_cols": feat_cols,
        "center": center.to_dict(),
        "scale": scale.to_dict(),
    }, CKPT_PATH)
    print(f"saved checkpoint -> {CKPT_PATH}")

    # Emit frozen embeddings for the ENTIRE timeline (train+VAL+OOS). Each window is
    # still backward-looking only -- no VAL/OOS info leaks into the (already-finished)
    # pretraining above.
    model.eval()
    full_ds = WindowDataset(standardized, frame["timestamp"].to_numpy(), window=WINDOW)
    full_loader = DataLoader(full_ds, batch_size=1024, shuffle=False, num_workers=0)
    all_embs, all_ends = [], []
    with torch.no_grad():
        for x, end in full_loader:
            x = x.to(device)
            emb = model.encode(x).cpu().numpy()
            all_embs.append(emb)
            all_ends.append(end.numpy())
    embs = np.concatenate(all_embs, axis=0)
    ends = np.concatenate(all_ends, axis=0)

    emb_cols = [f"deepfeat_{i}" for i in range(EMBED_DIM)]
    out = pd.DataFrame(embs, columns=emb_cols)
    out.insert(0, "timestamp", frame["timestamp"].to_numpy()[ends])
    out.to_parquet(EMB_OUT_PATH, index=False)
    print(f"wrote {len(out)} embedding rows -> {EMB_OUT_PATH} "
          f"(first {WINDOW - 1} rows of the timeline have no window, so are absent)")


if __name__ == "__main__":
    main()
