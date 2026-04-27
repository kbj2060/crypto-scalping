from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.ensemble_router import EnsembleRouter
from ensemble.seven_model_ensemble import SevenModelEnsemble
from features.registry import M7_PROB_ALIASES, find_missing_columns, get_m7_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AI_COLS = [
    "patchtst_median",
    "patchtst_regime_sim",
    "tide_vol_raw",
    "tide_vol_zscore",
    "timesnet_cycle_sin",
    "timesnet_cycle_cos",
    "timesnet_cycle_delta",
    "dlinear_smf_ema",
    "dlinear_smf_slope",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified 2025 RL dataset with M7 + AI features.")
    p.add_argument("--features-path", default=str(ROOT / "data/splits/year_oos/training_features_2025.csv"))
    p.add_argument("--rl-path", default=str(ROOT / "data/splits/year_oos/rl_base_2025.csv"))
    p.add_argument("--output-path", default=str(ROOT / "data/rl_training_2025_unified.csv"))
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--ai-min-valid-ratio", type=float, default=0.95)
    p.add_argument("--ai-min-std-cols", type=int, default=6)
    p.add_argument("--ai-warmup-bars", type=int, default=256)
    p.add_argument("--startup-check-only", action="store_true")
    return p.parse_args()


def _validate_m7(df: pd.DataFrame) -> None:
    required = get_m7_columns("rl_core", include_entry_price=False)
    missing = find_missing_columns(df.columns, required, aliases=M7_PROB_ALIASES)
    if missing:
        raise ValueError(f"Unified dataset missing M7 required cols: {sorted(missing)}")


def _validate_ai(df: pd.DataFrame) -> None:
    missing = [c for c in AI_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Unified dataset missing AI cols: {missing}")


def _validate_ai_quality(df: pd.DataFrame, min_valid_ratio: float, min_std_cols: int) -> None:
    ai = df[AI_COLS].apply(pd.to_numeric, errors="coerce")
    valid_ratio = ai.notna().mean().to_dict()
    low_valid = {k: v for k, v in valid_ratio.items() if float(v) < float(min_valid_ratio)}
    if low_valid:
        raise ValueError(f"AI feature valid ratio too low (<{min_valid_ratio}): {low_valid}")

    stds = ai.fillna(0.0).std(ddof=0).to_dict()
    nz_std_cols = [k for k, v in stds.items() if float(v) > 1e-8]
    if len(nz_std_cols) < int(min_std_cols):
        raise ValueError(
            f"AI feature variance too low: non-flat cols={len(nz_std_cols)} < required={min_std_cols}. stds={stds}"
        )
    logger.info("[AI-QUALITY] valid_ratio=%s", {k: round(float(v), 4) for k, v in valid_ratio.items()})
    logger.info("[AI-QUALITY] std=%s", {k: round(float(v), 8) for k, v in stds.items()})


def main(args: argparse.Namespace) -> int:
    features_path = Path(args.features_path)
    rl_path = Path(args.rl_path)
    output_path = Path(args.output_path)
    ts_col = str(args.timestamp_col)

    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    if not rl_path.exists():
        raise FileNotFoundError(f"RL base file not found: {rl_path}")

    logger.info("Loading base frames...")
    feat_df = pd.read_csv(features_path)
    rl_df = pd.read_csv(rl_path)
    for df_name, frame in (("features", feat_df), ("rl_base", rl_df)):
        if ts_col not in frame.columns:
            raise KeyError(f"{df_name} missing timestamp col: {ts_col}")
        frame[ts_col] = pd.to_datetime(frame[ts_col], errors="coerce")
        frame.dropna(subset=[ts_col], inplace=True)
        frame.sort_values(ts_col, inplace=True)
        frame.drop_duplicates(subset=[ts_col], keep="last", inplace=True)

    merged = feat_df.merge(rl_df, on=ts_col, how="inner", suffixes=("", "_rl"))
    drop_rl_dup = [c for c in merged.columns if c.endswith("_rl")]
    if drop_rl_dup:
        merged.drop(columns=drop_rl_dup, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    logger.info("Merged base rows=%d cols=%d", len(merged), len(merged.columns))

    logger.info("Running M7 batch inference...")
    m7 = SevenModelEnsemble()
    m7_df = m7.predict_batch(merged).reset_index(drop=True)
    if len(m7_df) != len(merged):
        raise RuntimeError(f"M7 row mismatch: base={len(merged)} m7={len(m7_df)}")

    overlap_m7 = [c for c in m7_df.columns if c in merged.columns]
    if overlap_m7:
        merged = merged.drop(columns=overlap_m7)
    merged = pd.concat([merged, m7_df], axis=1)
    logger.info("M7 cols merged: %d", len(m7_df.columns))

    logger.info("Running AI forecaster batch inference...")
    router = EnsembleRouter()
    ai_df = router.get_refined_features(merged).reset_index(drop=True)
    if len(ai_df) != len(merged):
        raise RuntimeError(f"AI row mismatch: base={len(merged)} ai={len(ai_df)}")

    overlap_ai = [c for c in ai_df.columns if c in merged.columns]
    if overlap_ai:
        merged = merged.drop(columns=overlap_ai)
    merged = pd.concat([merged, ai_df], axis=1)
    logger.info("AI cols merged: %s", list(ai_df.columns))

    _validate_m7(merged)
    _validate_ai(merged)
    _validate_ai_quality(
        merged,
        min_valid_ratio=float(args.ai_min_valid_ratio),
        min_std_cols=int(args.ai_min_std_cols),
    )

    warmup = max(0, int(args.ai_warmup_bars))
    merged["ai_ready"] = 1.0
    if warmup > 0 and len(merged) > 0:
        merged.loc[: min(warmup - 1, len(merged) - 1), "ai_ready"] = 0.0
    logger.info("AI warmup bars=%d (ai_ready=0 for initial segment)", warmup)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info("Saved unified dataset: %s", output_path)
    logger.info("Final rows=%d cols=%d", len(merged), len(merged.columns))
    return 0


if __name__ == "__main__":
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: build_unified_rl_dataset")
        raise SystemExit(0)
    raise SystemExit(main(args))
