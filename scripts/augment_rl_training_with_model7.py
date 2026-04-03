#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import logging
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.seven_model_ensemble import SevenModelEnsemble
from features.schema import build_rl_feature_keep
from features.registry import get_m7_columns
from strategies.elite_builder import (
    compute_synthetic_alphas,
    compute_regime,
    compute_volatility_models,
    compute_new_elite_signals,
    EliteSignals,
    row_to_market_row,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_meta(
    *,
    output_path: str,
    rl_path: str,
    feature_path: str,
    row_count: int,
    col_count: int,
    m7_cols: list[str],
    dropped_cols: list[str],
) -> None:
    meta_path = f"{output_path}.meta.json"
    payload = {
        "schema_version": "feature_registry.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_role": "rl_train_with_m7",
        "rows": int(row_count),
        "cols": int(col_count),
        "sources": {
            "rl_path": rl_path,
            "feature_path": feature_path,
            "rl_sha256": _sha256(rl_path) if os.path.exists(rl_path) else None,
            "feature_sha256": _sha256(feature_path) if os.path.exists(feature_path) else None,
        },
        "m7": {
            "generated_cols": sorted(m7_cols),
            "dropped_cols": sorted(dropped_cols),
            "required_rl_core": sorted(get_m7_columns("rl_core", include_entry_price=False)),
            "required_live_strict": sorted(get_m7_columns("live_strict", include_entry_price=False)),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Saved meta: %s", meta_path)


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

    # M7 모델이 요구하는 파생 피처 사전 계산
    # (training_features CSV에 포함되지 않은 volatility / regime / synthetic alpha 보완)
    logger.info("🔧 합성 알파 피처 계산 중...")
    work_df = compute_synthetic_alphas(work_df)
    logger.info("🔧 레짐 라벨 계산 중...")
    work_df = compute_regime(work_df)
    logger.info("🔧 변동성 모델 피처 계산 중 (GARCH/OU/Jump/EVT)...")
    work_df = compute_volatility_models(work_df)
    logger.info("🔧 신규 Elite 시그널 배치 계산 중 (volume_confirm/liquidity_trap/trend_health)...")
    work_df = compute_new_elite_signals(work_df)
    logger.info("🔧 행별 Elite 시그널 계산 중 (sig_whale/sig_oi_divergence 등)...")
    _elite_extractor = EliteSignals()
    # rolling std (look-ahead 방지): 현재 행 기준 과거 576봉(≈2일) std 사용
    if "smart_money_flow" in work_df.columns:
        _smf_rolling_std = (
            work_df["smart_money_flow"]
            .rolling(window=576, min_periods=10)
            .std()
            .fillna(work_df["smart_money_flow"].expanding(min_periods=1).std())
            .fillna(1.0)
        )
    else:
        _smf_rolling_std = pd.Series(1.0, index=work_df.index)
    _elite_keys = [
        "sig_whale", "sig_oi_divergence", "sig_ai_squeeze", "sig_orderblock",
        "sig_liq_squeeze", "sig_net_taker", "sig_hurst_ofi",
        "sig_funding_cascade", "sig_multifractal", "sig_cluster_fib",
        "sig_top_trader_squeeze", "sig_btc_corr_breakout",
        "sig_garch_regime", "sig_ou_mean_rev", "sig_jump_rebound", "sig_evt_tail",
    ]
    for _k in _elite_keys:
        if _k not in work_df.columns:
            work_df[_k] = 0.0
    _records = work_df.to_dict("records")
    for _i in range(len(_records)):
        try:
            _cur = row_to_market_row(_records[_i])
            _prev = row_to_market_row(_records[_i - 1]) if _i > 0 else _cur
            _sigs = _elite_extractor.compute_all(current=_cur, prev=_prev, smf_std=float(_smf_rolling_std.iloc[_i]))
            for _k in _elite_keys:
                if _k in _sigs:
                    work_df.at[_i, _k] = float(_sigs[_k])
        except Exception:
            raise
    logger.info("Work cols after enrichment: %d", len(work_df.columns))

    ensemble = SevenModelEnsemble()
    m7 = ensemble.predict_batch(work_df)
    raw_m7_cols = list(m7.columns)
    drop_cols = [c for c in get_m7_columns("deprecated", include_entry_price=True) if c in m7.columns]
    if drop_cols:
        m7 = m7.drop(columns=drop_cols)
        logger.info("Dropped deprecated model7 columns: %d", len(drop_cols))
    logger.info("Generated model7 columns: %d", len(m7.columns))

    overlap = [c for c in m7.columns if c in rl_df.columns]
    if overlap:
        rl_df = rl_df.drop(columns=overlap)
        logger.info("Dropped existing overlapping columns: %d", len(overlap))

    # RL 학습에 실제로 필요한(keep-set) 원본 피처를 누락 없이 반영
    rl_keep = set(build_rl_feature_keep(include_entry_price=False))
    passthrough_cols = [
        c for c in work_df.columns
        if c in rl_keep and c not in rl_df.columns and c not in m7.columns
    ]
    passthrough = work_df[passthrough_cols].reset_index(drop=True) if passthrough_cols else pd.DataFrame(index=rl_df.index)
    if passthrough_cols:
        logger.info("Added passthrough RL features from feature frame: %d", len(passthrough_cols))

    out_df = pd.concat(
        [rl_df.reset_index(drop=True), passthrough, m7.reset_index(drop=True)],
        axis=1,
    )
    output_path = args.output_path.strip() or args.rl_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    _write_meta(
        output_path=output_path,
        rl_path=args.rl_path,
        feature_path=args.feature_path,
        row_count=len(out_df),
        col_count=len(out_df.columns),
        m7_cols=raw_m7_cols,
        dropped_cols=drop_cols,
    )

    logger.info("Saved: %s", output_path)
    logger.info("Final shape: rows=%d cols=%d", len(out_df), len(out_df.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
