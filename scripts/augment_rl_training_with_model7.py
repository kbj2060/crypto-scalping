#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import logging

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.seven_model_ensemble import SevenModelEnsemble


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DROP_M7_COLUMNS = {
    "m7_prob_dn",
    "m7_prob_fl",
    "m7_prob_up",
    "m7_direction",
    "m7_hdb_label",
    "m7_hdb_prob",
    "m7_vae_threshold",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Append 7-model ensemble outputs as numeric columns to rl_training_data_full.csv"
    )
    p.add_argument("--rl-path", default="data/rl_training_data_full.csv")
    p.add_argument("--feature-path", default="data/training_features_5m.csv")
    p.add_argument("--output-path", default="")
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--limit", type=int, default=0, help="debug: only first N rows (0=all)")
    p.add_argument(
        "--startup-check-only",
        action="store_true",
        help="Validate imports/arguments and exit without writing files",
    )
    return p.parse_args()


def _load_frames(rl_path: str, feature_path: str, timestamp_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(rl_path):
        raise FileNotFoundError(f"RL csv not found: {rl_path}")

    rl_df = pd.read_csv(rl_path)
    if timestamp_col not in rl_df.columns:
        raise ValueError(f"timestamp col missing in RL csv: {timestamp_col}")
    rl_df[timestamp_col] = pd.to_datetime(rl_df[timestamp_col], errors="coerce")

    work_df = rl_df.copy()
    if os.path.exists(feature_path):
        feat_df = pd.read_csv(feature_path)
        if timestamp_col in feat_df.columns:
            feat_df[timestamp_col] = pd.to_datetime(feat_df[timestamp_col], errors="coerce")
            extra_cols = [c for c in feat_df.columns if c not in work_df.columns and c != timestamp_col]
            if extra_cols:
                work_df = work_df.merge(feat_df[[timestamp_col] + extra_cols], on=timestamp_col, how="left")
                logger.info("Merged feature frame: +%d columns", len(extra_cols))
        else:
            logger.warning("feature csv has no timestamp column: %s", feature_path)
    else:
        logger.warning("feature csv not found, continue without merge: %s", feature_path)

    return rl_df, work_df


def main() -> int:
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: augment_rl_training_with_model7")
        return 0

    rl_df, work_df = _load_frames(args.rl_path, args.feature_path, args.timestamp_col)
    if args.limit > 0:
        rl_df = rl_df.iloc[: args.limit].copy()
        work_df = work_df.iloc[: args.limit].copy()
        logger.info("limit mode: first %d rows", len(rl_df))

    logger.info("Rows=%d | RL cols=%d | Work cols=%d", len(work_df), len(rl_df.columns), len(work_df.columns))
    ensemble = SevenModelEnsemble()
    m7 = ensemble.predict_batch(work_df)
    drop_cols = [c for c in DROP_M7_COLUMNS if c in m7.columns]
    if drop_cols:
        m7 = m7.drop(columns=drop_cols)
        logger.info("Dropped deprecated model7 columns: %d", len(drop_cols))
    logger.info("Generated model7 columns: %d", len(m7.columns))

    overlap = [c for c in m7.columns if c in rl_df.columns]
    if overlap:
        rl_df = rl_df.drop(columns=overlap)
        logger.info("Dropped existing overlapping columns: %d", len(overlap))

    out_df = pd.concat([rl_df.reset_index(drop=True), m7.reset_index(drop=True)], axis=1)
    output_path = args.output_path.strip() or args.rl_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    logger.info("Saved: %s", output_path)
    logger.info("Final shape: rows=%d cols=%d", len(out_df), len(out_df.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
