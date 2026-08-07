"""Kronos layerA v2 -- Stage 0 cheap gate (2026-08-07).

The one 2025/26-paper axis that qualifies as a "genuinely different representation" under the
2026-08-07 arc rule: a finance-specific pretrained K-line foundation model (Kronos,
arXiv:2508.02739, ~12B bars / 45 exchanges) as a FROZEN feature extractor for the transition
detector (layerA, the promoted swingtransition model's source signal -- the project's only
component with real OOS AUC). Explicitly NOT an entry/direction model: self-trained transformer
embeddings on this repo's own data are a closed axis (embedding ceiling, 2026-08-06), and the
2026 literature agrees zero-shot TSFM return forecasting loses to GBDT.

PRE-REGISTERED DESIGN (fixed before any result was seen):
- Kronos-small + Tokenizer-base, FROZEN. Window = 512 consecutive 5m bars ending AT row t
  (positional, causal). Channels: open/high/low/close/volume/amount(:=quote_volume).
  Per-window z-norm + clip +-5 (KronosPredictor's exact scheme). Embedding = last-position
  hidden state after the final RMSNorm (512-d).
- Classifier: the exact layerA LGBM recipe (400 trees, num_leaves 31, lr 0.05,
  min_child_samples 100, balanced) on [110 existing features + 512 emb] vs a PAIRED baseline
  [110 features] recomputed on the identical row set (rows with <512 bars of history dropped).
- ACCEPT RULE: VAL AUC and OOS AUC must BOTH improve by >= +0.005 over the paired baseline.
  Pass -> Stage 1 (rebuild swing_transition_prob v2 + full downstream chain, judged by its own
  pre-registered worst-quarter non-degradation rule). Fail -> axis closed, no fine-tuning
  escalation without a separate decision.
- Embedding-only LGBM is reported as a diagnostic; it is NOT a selectable outcome.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tmp/kronos_vendor_20260807"))

from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from eval_btc_5m_layerA_layerB_20260806 import (  # noqa: E402
    DROP_RAW,
    OOS_END,
    OOS_START,
    PANEL_PATH,
    PIVOT_PATH,
    VAL_START,
    build_dvol_features,
)

RAW_SOURCES = [ROOT / f"data/splits/year_oos/btc_features_{y}.csv" for y in (2024, 2025, 2026)]
OUT_DIR = ROOT / "tmp/btc_kronos_layerA_20260807"
EMB_PATH = OUT_DIR / "kronos_small_emb_512.parquet"
WINDOW = 512
ACCEPT_MARGIN = 0.005


def build_layerA_dataset() -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_parquet(PANEL_PATH)
    dvol = build_dvol_features()
    panel = pd.merge_asof(panel.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    feature_cols = [c for c in panel.columns if c not in DROP_RAW]
    piv = pd.read_parquet(PIVOT_PATH, columns=["timestamp", "transition_soon"])
    dfA = panel.merge(piv, on="timestamp", how="inner").dropna(subset=["transition_soon"]).reset_index(drop=True)
    return dfA, feature_cols


def extract_embeddings(timestamps: pd.Series, *, batch_size: int, limit: int | None) -> pd.DataFrame:
    from model.kronos import Kronos, KronosTokenizer, calc_time_stamps

    raw = pd.concat(
        [pd.read_csv(p, usecols=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]) for p in RAW_SOURCES],
        ignore_index=True,
    )
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    x_all = raw[["open", "high", "low", "close", "volume", "quote_volume"]].to_numpy(dtype=np.float32)
    stamp_all = calc_time_stamps(raw["timestamp"]).values.astype(np.float32)
    pos_map = {t: i for i, t in enumerate(raw["timestamp"])}

    wanted = pd.to_datetime(timestamps)
    positions = np.array([pos_map.get(t, -1) for t in wanted], dtype=np.int64)
    keep = positions >= (WINDOW - 1)
    if (positions < 0).any():
        raise SystemExit(f"{int((positions < 0).sum())} layerA timestamps missing from raw 5m series")
    kept_ts = wanted[keep].reset_index(drop=True)
    kept_pos = positions[keep]
    if limit is not None:
        kept_ts = kept_ts.iloc[:limit].reset_index(drop=True)
        kept_pos = kept_pos[:limit]
    print(f"embedding rows: {len(kept_pos)} (dropped {int((~keep).sum())} early-history rows)", flush=True)

    device = torch.device("cuda")
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device).eval()
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device).eval()

    out = np.empty((len(kept_pos), mdl.d_model), dtype=np.float16)
    with torch.no_grad():
        for start in range(0, len(kept_pos), batch_size):
            pos_b = kept_pos[start:start + batch_size]
            win = np.stack([x_all[p - WINDOW + 1:p + 1] for p in pos_b])  # [B, 512, 6]
            mean = win.mean(axis=1, keepdims=True)
            std = win.std(axis=1, keepdims=True)
            win = np.clip((win - mean) / (std + 1e-5), -5.0, 5.0)
            stamp = np.stack([stamp_all[p - WINDOW + 1:p + 1] for p in pos_b])
            x_t = torch.from_numpy(win).to(device)
            s_t = torch.from_numpy(stamp).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                s1, s2 = tok.encode(x_t, half=True)
                h = mdl.embedding([s1, s2]) + mdl.time_emb(s_t)
                for layer in mdl.transformer:
                    h = layer(h)
                h = mdl.norm(h)
            out[start:start + len(pos_b)] = h[:, -1, :].float().cpu().numpy().astype(np.float16)
            if (start // batch_size) % 50 == 0:
                print(f"  batch {start // batch_size}/{len(kept_pos) // batch_size}", flush=True)

    emb = pd.DataFrame(out, columns=[f"kronos_emb_{i}" for i in range(out.shape[1])])
    emb.insert(0, "timestamp", kept_ts)
    return emb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N rows")
    ap.add_argument("--skip-extract", action="store_true", help="reuse cached embeddings parquet")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dfA, feature_cols = build_layerA_dataset()
    print(f"layerA dataset rows={len(dfA)} features={len(feature_cols)}", flush=True)

    if args.skip_extract and EMB_PATH.exists():
        emb = pd.read_parquet(EMB_PATH)
    else:
        emb = extract_embeddings(dfA["timestamp"], batch_size=args.batch_size, limit=args.limit)
        emb.to_parquet(EMB_PATH, index=False)
        print(f"saved embeddings {EMB_PATH} shape={emb.shape}", flush=True)

    df = dfA.merge(emb, on="timestamp", how="inner").reset_index(drop=True)
    emb_cols = [c for c in df.columns if c.startswith("kronos_emb_")]
    y = df["transition_soon"].astype(int)
    tr = df["timestamp"] < VAL_START
    val = (df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)
    oos = (df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)
    print(f"paired rows: train={tr.sum()} val={val.sum()} oos={oos.sum()}", flush=True)

    def run(cols: list[str], label: str) -> dict[str, float]:
        clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05,
                             min_child_samples=100, class_weight="balanced", verbosity=-1)
        clf.fit(df.loc[tr, cols].astype(np.float32), y[tr])
        p = clf.predict_proba(df[cols].astype(np.float32))[:, 1]
        res = {}
        for name, mask in [("VAL", val), ("OOS", oos)]:
            res[f"{name}_auc"] = float(roc_auc_score(y[mask], p[mask]))
            res[f"{name}_ap"] = float(average_precision_score(y[mask], p[mask]))
        print(f"{label}: VAL AUC={res['VAL_auc']:.4f} AP={res['VAL_ap']:.4f} | OOS AUC={res['OOS_auc']:.4f} AP={res['OOS_ap']:.4f}", flush=True)
        return res

    base = run(feature_cols, "paired baseline [110 feats]")
    aug = run(feature_cols + emb_cols, "augmented [110 + 512 kronos]")
    diag = run(emb_cols, "diagnostic emb-only")

    verdict = (
        aug["VAL_auc"] >= base["VAL_auc"] + ACCEPT_MARGIN
        and aug["OOS_auc"] >= base["OOS_auc"] + ACCEPT_MARGIN
    )
    report = {
        "design": "frozen Kronos-small last-pos hidden (512d) as extra layerA features",
        "accept_rule": f"VAL and OOS AUC both >= baseline + {ACCEPT_MARGIN}",
        "paired_baseline": base,
        "augmented": aug,
        "diagnostic_emb_only": diag,
        "verdict_pass": bool(verdict),
        "rows": {"train": int(tr.sum()), "val": int(val.sum()), "oos": int(oos.sum())},
    }
    (OUT_DIR / "stage0_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("VERDICT:", "PASS -> Stage 1 allowed" if verdict else "FAIL -> axis closed at Stage 0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
