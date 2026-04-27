from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    p.add_argument("--resume", action="store_true", default=True, help="Resume from latest checkpoint when available.")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument(
        "--checkpoint-dir",
        default=str(ROOT / "data" / "tmp" / "unified_build_ckpt"),
        help="Directory to store intermediate CSV checkpoints.",
    )
    p.add_argument("--keep-checkpoints", action="store_true", default=True)
    p.add_argument("--no-keep-checkpoints", dest="keep_checkpoints", action="store_false")
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


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _causal_anomaly_flag(s: pd.Series, window: int = 512, z_th: float = 2.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu = x.rolling(window=window, min_periods=max(20, window // 8)).mean()
    sd = x.rolling(window=window, min_periods=max(20, window // 8)).std(ddof=0).replace(0.0, np.nan)
    z = (x - mu) / (sd + 1e-8)
    return (z > float(z_th)).astype(float).fillna(0.0)


def _ensure_m7_compat(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = _num(out, "close", 0.0).clip(lower=1e-8)

    # Prob aliases required by rl_core.
    if "m7_trend_xgb_dn" not in out.columns:
        out["m7_trend_xgb_dn"] = _num(out, "m7_prob_dn", 1.0 / 3.0)
    if "m7_trend_xgb_fl" not in out.columns:
        out["m7_trend_xgb_fl"] = _num(out, "m7_prob_fl", 1.0 / 3.0)
    if "m7_trend_xgb_up" not in out.columns:
        out["m7_trend_xgb_up"] = _num(out, "m7_prob_up", 1.0 / 3.0)

    # Quantile derived width.
    if "m7_qwidth" not in out.columns:
        out["m7_qwidth"] = (_num(out, "m7_q90", 0.0) - _num(out, "m7_q10", 0.0)).abs().clip(lower=0.0)

    # Entry offsets from prices.
    if "m7_entry_long_offset" not in out.columns:
        out["m7_entry_long_offset"] = ((_num(out, "m7_entry_long_price", close) - close) / close).fillna(0.0)
    if "m7_entry_short_offset" not in out.columns:
        out["m7_entry_short_offset"] = ((close - _num(out, "m7_entry_short_price", close)) / close).fillna(0.0)

    # TP/SL offsets from prices (magnitude).
    if "m7_tp_offset" not in out.columns:
        out["m7_tp_offset"] = ((_num(out, "m7_tp_price", close) - close) / close).abs().fillna(0.0)
    if "m7_sl_offset" not in out.columns:
        out["m7_sl_offset"] = ((_num(out, "m7_sl_price", close) - close) / close).abs().fillna(0.0)

    # Vol rank fallback.
    if "m7_gmm_vol_rank" not in out.columns:
        gz = _num(out, "garch_vol_z", 0.0).abs()
        out["m7_gmm_vol_rank"] = np.clip(gz / 3.0, 0.0, 1.0)

    # Anomaly flags fallback (causal rolling z-score).
    if "m7_iso_anom" not in out.columns:
        out["m7_iso_anom"] = _causal_anomaly_flag(_num(out, "m7_iso_score", 0.0), window=512, z_th=2.0)
    if "m7_vae_anom" not in out.columns:
        out["m7_vae_anom"] = _causal_anomaly_flag(_num(out, "m7_vae_error", 0.0), window=512, z_th=2.0)

    # Final finite cleanup for derived cols.
    derived = [
        "m7_trend_xgb_dn", "m7_trend_xgb_fl", "m7_trend_xgb_up",
        "m7_qwidth",
        "m7_entry_long_offset", "m7_entry_short_offset",
        "m7_tp_offset", "m7_sl_offset",
        "m7_gmm_vol_rank", "m7_iso_anom", "m7_vae_anom",
    ]
    for c in derived:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _ckpt_paths(ckpt_dir: Path) -> dict[str, Path]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": ckpt_dir / "01_base_merged.csv",
        "m7": ckpt_dir / "02_after_m7.csv",
        "ai": ckpt_dir / "03_after_ai.csv",
        "failed": ckpt_dir / "zz_failed_snapshot.csv",
    }


def _save_ckpt(df: pd.DataFrame, path: Path, tag: str) -> None:
    t0 = time.time()
    df.to_csv(path, index=False)
    logger.info("[CKPT] saved %-14s rows=%d cols=%d path=%s (%.1fs)", tag, len(df), len(df.columns), path, time.time() - t0)


def main(args: argparse.Namespace) -> int:
    features_path = Path(args.features_path)
    rl_path = Path(args.rl_path)
    output_path = Path(args.output_path)
    ts_col = str(args.timestamp_col)

    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    if not rl_path.exists():
        raise FileNotFoundError(f"RL base file not found: {rl_path}")

    ckpt_dir = Path(args.checkpoint_dir)
    ck = _ckpt_paths(ckpt_dir)

    merged: pd.DataFrame | None = None
    pbar = tqdm(total=7, desc="build_unified_rl_dataset", unit="stage", dynamic_ncols=True)
    try:
        # Stage 1: base merge (or resume)
        if bool(args.resume) and ck["base"].exists():
            logger.info("[RESUME] loading base checkpoint: %s", ck["base"])
            merged = pd.read_csv(ck["base"])
        else:
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
            _save_ckpt(merged, ck["base"], "base")
        pbar.update(1)

        # Stage 2: M7 inference (or resume)
        if bool(args.resume) and ck["m7"].exists():
            logger.info("[RESUME] loading m7 checkpoint: %s", ck["m7"])
            merged = pd.read_csv(ck["m7"])
        else:
            assert merged is not None
            logger.info("Running M7 batch inference...")
            t0 = time.time()
            m7 = SevenModelEnsemble()
            m7_df = m7.predict_batch(merged).reset_index(drop=True)
            if len(m7_df) != len(merged):
                raise RuntimeError(f"M7 row mismatch: base={len(merged)} m7={len(m7_df)}")
            overlap_m7 = [c for c in m7_df.columns if c in merged.columns]
            if overlap_m7:
                merged = merged.drop(columns=overlap_m7)
            merged = pd.concat([merged, m7_df], axis=1)
            merged = _ensure_m7_compat(merged)
            logger.info("M7 cols merged: %d (%.1fs)", len(m7_df.columns), time.time() - t0)
            _save_ckpt(merged, ck["m7"], "after_m7")
        pbar.update(1)

        # Stage 3: AI inference (or resume)
        if bool(args.resume) and ck["ai"].exists():
            logger.info("[RESUME] loading ai checkpoint: %s", ck["ai"])
            merged = pd.read_csv(ck["ai"])
        else:
            assert merged is not None
            logger.info("Running AI forecaster batch inference...")
            t0 = time.time()
            router = EnsembleRouter()
            ai_df = router.get_refined_features(merged).reset_index(drop=True)
            if len(ai_df) != len(merged):
                raise RuntimeError(f"AI row mismatch: base={len(merged)} ai={len(ai_df)}")
            overlap_ai = [c for c in ai_df.columns if c in merged.columns]
            if overlap_ai:
                merged = merged.drop(columns=overlap_ai)
            merged = pd.concat([merged, ai_df], axis=1)
            logger.info("AI cols merged: %s (%.1fs)", list(ai_df.columns), time.time() - t0)
            _save_ckpt(merged, ck["ai"], "after_ai")
        pbar.update(1)

        assert merged is not None
        _validate_m7(merged)
        pbar.update(1)
        _validate_ai(merged)
        pbar.update(1)
        _validate_ai_quality(
            merged,
            min_valid_ratio=float(args.ai_min_valid_ratio),
            min_std_cols=int(args.ai_min_std_cols),
        )
        pbar.update(1)

        warmup = max(0, int(args.ai_warmup_bars))
        merged["ai_ready"] = 1.0
        if warmup > 0 and len(merged) > 0:
            merged.loc[: min(warmup - 1, len(merged) - 1), "ai_ready"] = 0.0
        logger.info("AI warmup bars=%d (ai_ready=0 for initial segment)", warmup)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        logger.info("Saved unified dataset: %s", output_path)
        logger.info("Final rows=%d cols=%d", len(merged), len(merged.columns))
        pbar.update(1)

        if not bool(args.keep_checkpoints):
            for p in ck.values():
                if p.exists():
                    p.unlink(missing_ok=True)
            logger.info("Checkpoint files removed (--no-keep-checkpoints).")

        return 0
    except Exception:
        if merged is not None:
            try:
                _save_ckpt(merged, ck["failed"], "failed_snapshot")
            except Exception as ce:
                logger.warning("Failed to save error snapshot: %s", ce)
        raise
    finally:
        pbar.close()


if __name__ == "__main__":
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: build_unified_rl_dataset")
        raise SystemExit(0)
    raise SystemExit(main(args))
