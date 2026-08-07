"""Causal windowed dataset for the new BTC deep-feature (CNN/Transformer) architecture line.

Joins the causalfix_final 5m panel (113 feature cols, category-grouped per
scripts/build_btc_feature_categories_20260806.py) with the existing zigzag risk-adjusted soft
labels (scripts/build_btc_5m_zigzag_and_pivot_labels_20260806.py ->
data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet) used as the teacher target.

Window t covers bars [t-window+1, t] (causal, inputs never look ahead). The soft label at t is
an offline-computed target (it legitimately uses future bars to know the eventual pivot/return
of the active wave) -- that is expected for a training target, not a live input feature.

Fresh-Forward split (per CLAUDE.md): VAL 2025-09-01..2025-12-31, OOS 2026-01-01..2026-03-31,
everything strictly before VAL_START is train.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_btc_feature_categories_20260806 as cats  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"

VAL_START = pd.Timestamp("2025-09-01")
VAL_END = pd.Timestamp("2025-12-31")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-03-31")

SOFT_LABEL_COLS = ["zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"]
HARD_LABEL_COL = "zigzag_action"
QUALITY_LABEL_COL = "zigzag_path_calmar"  # ret/max(mae,risk_floor); 0 on CASH/inactive bars, always >=0


@dataclass
class BTCDeepFeatDataset:
    """Lazy-window dataset: only the (n_rows, F) standardized feature matrix is materialized
    (~120MB for this panel). Windows are sliced on demand via `get_window(row_idx)` so nothing
    scales with window length until a batch is actually pulled -- a dense (N, window, F) tensor
    for this panel/window size would be ~5.5GB, unnecessary to hold in memory at once.
    """

    feature_columns: list[str]
    category_order: list[str]
    category_sizes: list[int]
    window: int
    feat_std: np.ndarray  # (n_rows, F) standardized, full series
    y_soft_all: np.ndarray  # (n_rows, 3)
    y_hard_all: np.ndarray  # (n_rows,)
    y_quality_all: np.ndarray  # (n_rows,) log1p(zigzag_path_calmar), 0 on CASH/inactive bars
    timestamps_all: np.ndarray  # (n_rows,)
    end_idx: dict[str, np.ndarray]  # split name -> row indices with a full causal window available
    mean: np.ndarray  # (F,) train-only standardization stats
    std: np.ndarray  # (F,)
    train_weight: np.ndarray | None = None  # (len(end_idx["train"]),) average-uniqueness weights
    hygiene: dict | None = None  # purge/embargo/uniqueness bookkeeping, None when all disabled

    def get_window(self, row_idx: int) -> np.ndarray:
        return self.feat_std[row_idx - self.window + 1 : row_idx + 1]

    def get_batch(self, row_indices: np.ndarray) -> np.ndarray:
        out = np.empty((len(row_indices), self.window, self.feat_std.shape[1]), dtype=np.float32)
        for i, t in enumerate(row_indices):
            out[i] = self.get_window(int(t))
        return out


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def build_dataset(
    window: int = 48,
    train_stride: int = 1,
    *,
    label_path: Path | None = None,
    hard_col: str | None = None,
    soft_cols: list[str] | None = None,
    quality_col: str | None = None,
    extra_feature_path: Path | None = None,
    extra_feature_cols: list[str] | None = None,
    label_span_path: Path | None = None,
    purge: bool = False,
    embargo_bars: int = 0,
    uniqueness_weights: bool = False,
) -> BTCDeepFeatDataset:
    """`train_stride` subsamples the train split's window-end row indices (every Nth row).
    Consecutive 5m windows overlap in window-1 of their window bars, so at stride=1 the training
    set is dominated by near-duplicate samples that let a model memorize very fast without
    learning anything that generalizes -- this is the direct cause of best-val-loss landing at
    epoch 1 in every architecture on this dataset. VAL/OOS stay dense (stride=1) since those are
    evaluation, not training, and must reflect the true bar-by-bar distribution.

    `label_path`/`hard_col`/`soft_cols`/`quality_col` default to the zigzag wave label; pass the
    causal triple-barrier trade-outcome label's path/columns (see
    scripts/build_btc_5m_tripbarrier_tradeoutcome_labels_20260806.py) to train against that
    instead. `quality_col=None` (the triple-barrier label has no quality target) fills y_quality_all
    with zeros -- fine as long as the quality head/loss stay disabled for that run.

    `label_span_path`/`purge`/`embargo_bars`/`uniqueness_weights` are gate-G2 training hygiene, all
    off by default so existing callers are unaffected. The triple-barrier label at bar i is only
    resolved at bar `i + label_span_bars[i]` (median 51, p90 189 bars -- see
    scripts/build_btc_5m_tripbarrier_label_span_20260807.py), which has two consequences the
    original split construction ignored:

    - `purge=True` drops train rows whose label window reaches into VAL, and VAL rows whose label
      window reaches into OOS. Without it the tail of train is supervised by VAL-period price
      action, and VAL -- which selects the checkpoint -- is partly supervised by OOS price action.
      `embargo_bars` drops a further fixed buffer beyond each purge boundary.
    - `uniqueness_weights=True` weights each train sample by its average uniqueness (mean of
      1/concurrency over the bars its label window covers, concurrency counted over the actual
      train sample set). Overlapping labels mean the nominal sample count massively overstates the
      independent information available; this is the standard correction for that.

    `extra_feature_path`/`extra_feature_cols`: optionally merge additional CAUSAL feature columns
    (timestamp-aligned to the same panel) onto the standard 113 causalfix_final columns before
    standardization -- e.g. scripts/build_btc_5m_zigzag_state_causal_features_20260806.py's live
    pivot-tracker features. Only affects `feature_columns`/`feat_std`/`n_features`; does not touch
    `category_order`/`category_sizes` (fine for the transformer encoder, which only needs the total
    feature count -- the cnn_category encoder would need category metadata updated too if used with
    extra features, which this session hasn't needed).
    """
    label_path = label_path or LABEL_PATH
    hard_col = hard_col or HARD_LABEL_COL
    soft_cols = soft_cols or SOFT_LABEL_COLS

    panel = pd.read_parquet(PANEL_PATH)
    label_cols = ["timestamp", hard_col, *soft_cols]
    if quality_col:
        label_cols.append(quality_col)
    labels = pd.read_parquet(label_path, columns=label_cols)
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = labels.sort_values("timestamp").reset_index(drop=True)
    if not (panel["timestamp"].to_numpy() == labels["timestamp"].to_numpy()).all():
        raise RuntimeError("panel/label timestamp misalignment")

    feature_columns = cats.feature_columns()
    feat = panel[feature_columns].to_numpy(dtype=np.float64)

    if extra_feature_path and extra_feature_cols:
        extra = pd.read_parquet(extra_feature_path, columns=["timestamp", *extra_feature_cols])
        extra = extra.sort_values("timestamp").reset_index(drop=True)
        if not (extra["timestamp"].to_numpy() == panel["timestamp"].to_numpy()).all():
            raise RuntimeError("extra_feature_path timestamps don't match the panel")
        feat = np.concatenate([feat, extra[extra_feature_cols].to_numpy(dtype=np.float64)], axis=1)
        feature_columns = feature_columns + list(extra_feature_cols)

    valid_row = ~np.isnan(feat).any(axis=1)  # drops mtf1h warmup NaNs at series start

    timestamps = panel["timestamp"].to_numpy()
    y_soft_all = labels[soft_cols].to_numpy(dtype=np.float32)
    y_hard_all = labels[hard_col].to_numpy(dtype=np.int64)
    y_quality_all = (
        np.log1p(labels[quality_col].to_numpy(dtype=np.float32).clip(min=0.0))
        if quality_col
        else np.zeros(len(labels), dtype=np.float32)
    )

    n = len(panel)
    has_full_window = valid_row.copy()
    for k in range(1, window):
        has_full_window[k:] &= valid_row[:-k]
    has_full_window[: window - 1] = False
    candidate_idx = np.flatnonzero(has_full_window)

    ts_series = pd.Series(timestamps)
    train_mask_rows = valid_row & (ts_series < VAL_START).to_numpy()
    train_fit_idx = candidate_idx[train_mask_rows[candidate_idx]]
    if len(train_fit_idx) == 0:
        raise RuntimeError("no train rows available to fit standardization stats")
    mean, std = _standardize_fit(feat[train_fit_idx])

    feat_std = ((feat - mean) / std).astype(np.float32)
    if not np.isfinite(feat_std[candidate_idx]).all():
        raise RuntimeError("non-finite values in standardized feature matrix (candidate rows)")

    val_rows = (ts_series >= VAL_START) & (ts_series <= VAL_END)
    oos_rows = (ts_series >= OOS_START) & (ts_series <= OOS_END)
    train_idx = candidate_idx[train_mask_rows[candidate_idx]]
    if train_stride > 1:
        train_idx = train_idx[::train_stride]
    end_idx = {
        "train": train_idx,
        "val": candidate_idx[val_rows.to_numpy()[candidate_idx]],
        "oos": candidate_idx[oos_rows.to_numpy()[candidate_idx]],
    }

    train_weight, hygiene = None, None
    if purge or uniqueness_weights:
        if label_span_path is None:
            raise ValueError("purge/uniqueness_weights require label_span_path")
        spans_df = pd.read_parquet(label_span_path, columns=["timestamp", "label_span_bars"])
        spans_df = spans_df.sort_values("timestamp").reset_index(drop=True)
        if not (spans_df["timestamp"].to_numpy() == timestamps).all():
            raise RuntimeError("label_span_path timestamps don't match the panel")
        spans = spans_df["label_span_bars"].to_numpy(dtype=np.int64)
        # label at row i is resolved by row i + spans[i] (entry is i+1, span counted from entry)
        label_end = np.arange(n, dtype=np.int64) + spans

        hygiene = {"purge": bool(purge), "embargo_bars": int(embargo_bars),
                   "uniqueness_weights": bool(uniqueness_weights),
                   "n_train_before_purge": int(len(end_idx["train"])),
                   "n_val_before_purge": int(len(end_idx["val"]))}
        if purge:
            val_start_i = int(np.searchsorted(timestamps, np.datetime64(VAL_START)))
            oos_start_i = int(np.searchsorted(timestamps, np.datetime64(OOS_START)))
            keep_train = label_end[end_idx["train"]] < val_start_i - embargo_bars
            keep_val = label_end[end_idx["val"]] < oos_start_i - embargo_bars
            end_idx["train"] = end_idx["train"][keep_train]
            end_idx["val"] = end_idx["val"][keep_val]
            hygiene.update(n_train_after_purge=int(len(end_idx["train"])),
                           n_val_after_purge=int(len(end_idx["val"])))

        if uniqueness_weights:
            t_idx = end_idx["train"]
            starts, ends = t_idx + 1, np.minimum(label_end[t_idx], n - 1)
            concurrency = np.zeros(n + 1, dtype=np.float64)
            np.add.at(concurrency, starts, 1.0)
            np.add.at(concurrency, ends + 1, -1.0)
            concurrency = np.cumsum(concurrency)[:n]
            inv_c = np.where(concurrency > 0, 1.0 / np.maximum(concurrency, 1e-12), 0.0)
            prefix = np.concatenate([[0.0], np.cumsum(inv_c)])
            span_len = np.maximum(ends - starts + 1, 1)
            uniqueness = (prefix[ends + 1] - prefix[starts]) / span_len
            train_weight = (uniqueness / uniqueness.mean()).astype(np.float32)
            hygiene.update(
                uniqueness_mean=float(uniqueness.mean()),
                uniqueness_min=float(uniqueness.min()),
                uniqueness_max=float(uniqueness.max()),
                # nominal samples x mean uniqueness -- the independent-information-equivalent count
                effective_sample_size=float(uniqueness.sum()),
            )

    return BTCDeepFeatDataset(
        feature_columns=feature_columns,
        category_order=cats.CATEGORY_ORDER,
        category_sizes=[len(cats.CATEGORY_MAP[c]) for c in cats.CATEGORY_ORDER],
        window=window,
        feat_std=feat_std,
        y_soft_all=y_soft_all,
        y_hard_all=y_hard_all,
        y_quality_all=y_quality_all,
        timestamps_all=timestamps,
        end_idx=end_idx,
        mean=mean,
        std=std,
        train_weight=train_weight,
        hygiene=hygiene,
    )


def main() -> int:
    ds = build_dataset()
    for name in ("train", "val", "oos"):
        print(f"{name}: n={len(ds.end_idx[name])}")
    print(f"feature dim={len(ds.feature_columns)} window={ds.window} categories={ds.category_order}")
    print(f"category_sizes={ds.category_sizes} sum={sum(ds.category_sizes)}")
    batch = ds.get_batch(ds.end_idx["train"][:4])
    print(f"sample batch shape={batch.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
