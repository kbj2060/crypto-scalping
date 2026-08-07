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
import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.seven_model_ensemble import SevenModelEnsemble
from features.high_order_state import add_high_order_state_features
from features.schema import build_rl_feature_keep
from features.registry import get_m7_columns
from pipeline.feature_contract import load_feature_contract, rl_passthrough_keep
from strategies.elite_builder import (
    compute_synthetic_alphas,
    compute_regime,
    compute_volatility_models,
    compute_new_elite_signals,
    EliteSignals,
    row_to_market_row,
)
from core.cvp import add_cvp_features


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{context} required column(s) missing: {', '.join(missing)}")


def _derive_prereq_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Base alpha inputs
    if "whale_retail_ratio" not in out.columns:
        _require_columns(out, ["sum_toptrader_long_short_ratio", "count_long_short_ratio"], "whale_retail_ratio")
        denom = pd.to_numeric(out["count_long_short_ratio"], errors="coerce").replace(0, pd.NA)
        num = pd.to_numeric(out["sum_toptrader_long_short_ratio"], errors="coerce")
        out["whale_retail_ratio"] = (num / denom).fillna(0.0)

    if "whale_conviction" not in out.columns:
        _require_columns(out, ["sum_toptrader_long_short_ratio"], "whale_conviction")
        out["whale_conviction"] = pd.to_numeric(out["sum_toptrader_long_short_ratio"], errors="coerce").diff().fillna(0.0)

    if "smart_money_flow" not in out.columns:
        _require_columns(out, ["sum_open_interest_value"], "smart_money_flow")
        out["smart_money_flow"] = (
            pd.to_numeric(out["sum_open_interest_value"], errors="coerce")
            .pct_change()
            .clip(-1, 1)
            .fillna(0.0)
        )

    if "funding_abs" not in out.columns or "funding_pressure" not in out.columns:
        _require_columns(out, ["last_funding_rate"], "funding_abs/funding_pressure")
        _funding = pd.to_numeric(out["last_funding_rate"], errors="coerce")
        if "funding_abs" not in out.columns:
            out["funding_abs"] = _funding.abs()
        if "funding_pressure" not in out.columns:
            out["funding_pressure"] = _funding.rolling(window=288, min_periods=1).sum()

    if (
        "funding_roc_12" not in out.columns
        or "funding_roc_48" not in out.columns
        or "funding_roc_288" not in out.columns
        or "funding_z_score" not in out.columns
    ):
        _require_columns(out, ["last_funding_rate"], "funding_roc/funding_z_score")
        _funding = pd.to_numeric(out["last_funding_rate"], errors="coerce").fillna(0.0)
        _base = _funding.abs().shift(1).replace(0, np.nan)
        if "funding_roc_12" not in out.columns:
            out["funding_roc_12"] = ((_funding - _funding.shift(12)) / (_base + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "funding_roc_48" not in out.columns:
            out["funding_roc_48"] = ((_funding - _funding.shift(48)) / (_base + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "funding_roc_288" not in out.columns:
            out["funding_roc_288"] = ((_funding - _funding.shift(288)) / (_base + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "funding_z_score" not in out.columns:
            _roll_mean = _funding.rolling(window=288, min_periods=20).mean()
            _roll_std = _funding.rolling(window=288, min_periods=20).std().replace(0, np.nan)
            out["funding_z_score"] = ((_funding - _roll_mean) / _roll_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "timestamp" in out.columns and (
        "session_europe" not in out.columns or "session_us" not in out.columns or "is_hour_open" not in out.columns
    ):
        _ts = pd.to_datetime(out["timestamp"], errors="coerce")
        _hour = _ts.dt.hour.fillna(0).astype(int)
        _minute = _ts.dt.minute.fillna(0).astype(int)
        if "session_europe" not in out.columns:
            out["session_europe"] = ((_hour >= 7) & (_hour < 16)).astype(float)
        if "session_us" not in out.columns:
            out["session_us"] = ((_hour >= 13) & (_hour < 22)).astype(float)
        if "is_hour_open" not in out.columns:
            out["is_hour_open"] = (_minute == 0).astype(float)

    if "hurst_48" not in out.columns:
        _require_columns(out, ["close"], "hurst_48")
        _ret = pd.to_numeric(out["close"], errors="coerce").pct_change().fillna(0.0)

        def _rs_hurst(x):
            if len(x) < 10:
                return 0.5
            mean_r = x.mean()
            deviate = (x - mean_r).cumsum()
            r = float(deviate.max() - deviate.min())
            s = float(x.std())
            if s < 1e-10:
                return 0.5
            return float(np.log(r / s + 1e-10) / np.log(len(x)))

        out["hurst_48"] = (
            _ret.rolling(window=48, min_periods=24).apply(_rs_hurst, raw=True).fillna(0.5)
        )

    if "hurst_change" not in out.columns:
        out["hurst_change"] = pd.to_numeric(out["hurst_48"], errors="coerce").diff().fillna(0.0)

    if "regime_break" not in out.columns:
        _require_columns(out, ["breakout_strength"], "regime_break")
        _bs = pd.to_numeric(out["breakout_strength"], errors="coerce")
        out["regime_break"] = (_bs.abs() >= 0.6).astype(float)

    if "vwap_dist" not in out.columns:
        _require_columns(out, ["high", "low", "close", "volume"], "vwap_dist")
        typical_price = (
            pd.to_numeric(out["high"], errors="coerce")
            + pd.to_numeric(out["low"], errors="coerce")
            + pd.to_numeric(out["close"], errors="coerce")
        ) / 3.0
        volume = pd.to_numeric(out["volume"], errors="coerce")
        tp_vol = typical_price * volume
        roll = 288
        cum_tp_vol = tp_vol.rolling(window=roll, min_periods=1).sum()
        cum_vol = volume.rolling(window=roll, min_periods=1).sum()
        vwap = cum_tp_vol / cum_vol.replace(0, pd.NA)
        out["vwap_dist"] = ((pd.to_numeric(out["close"], errors="coerce") - vwap) / (vwap + 1e-8)).fillna(0.0)

    # CVP extended feature for SyntheticAlphaEngine (FDLV/SVPS)
    if "cvp_vah_val_width" not in out.columns:
        _require_columns(out, ["close", "volume"], "cvp_vah_val_width")
        out = add_cvp_features(
            out,
            lookback=200,
            n_clusters=4,
            output_cols=[
                "cvp_poc_dist",
                "cvp_vah_val_width",
                "cvp_cluster_position",
                "cvp_volume_imbalance",
                "cvp_regime",
            ],
        )

    return out

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Append 7-model ensemble outputs as numeric columns to rl_training_data_full.csv"
    )
    p.add_argument("--rl-path", default="data/rl_training_data_full.csv")
    p.add_argument("--feature-path", default="data/training_features_5m.csv")
    p.add_argument("--output-path", default="")
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--feature-manifest", default="docs/feature_contract_manifest.json")
    p.add_argument("--trend-xgb-meta", default="")
    p.add_argument("--entry-price-meta", default="")
    p.add_argument("--multi-target-meta", default="")
    p.add_argument("--quantile-meta", default="")
    p.add_argument("--lightgbm-ensemble-meta", default="")
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
    feature_manifest: str | None,
    passthrough_cols: list[str],
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
        "feature_contract": {
            "manifest_path": feature_manifest,
            "passthrough_cols": sorted(passthrough_cols),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Saved meta: %s", meta_path)


def main() -> int:
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: augment_m7_dataset")
        return 0

    rl_df, work_df = _load_frames(args.rl_path, args.feature_path, args.timestamp_col)
    if args.limit > 0:
        rl_df = rl_df.iloc[: args.limit].copy()
        work_df = work_df.iloc[: args.limit].copy()
        logger.info("limit mode: first %d rows", len(rl_df))

    logger.info("Rows=%d | RL cols=%d | Work cols=%d", len(work_df), len(rl_df.columns), len(work_df.columns))

    logger.info("🔧 합성 알파/레짐 선행 피처 검증 및 파생 계산 중...")
    work_df = _derive_prereq_features(work_df)

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
    work_df = add_high_order_state_features(work_df)
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
    for _i in tqdm(range(len(_records)), desc="elite-signals", unit="row"):
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

    meta_paths = {
        key: path
        for key, path in {
            "trend_xgb": args.trend_xgb_meta,
            "entry_price_model": args.entry_price_meta,
            "multi_target_lgbm": args.multi_target_meta,
            "quantile_forest": args.quantile_meta,
            "lightgbm_ensemble": args.lightgbm_ensemble_meta,
        }.items()
        if str(path).strip()
    }
    ensemble = SevenModelEnsemble(meta_paths=meta_paths or None)
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
    contract = load_feature_contract(args.feature_manifest) if args.feature_manifest else {}
    contract_keep = rl_passthrough_keep(contract)
    passthrough_cols = [
        c for c in work_df.columns
        if c in rl_keep and (not contract_keep or c in contract_keep) and c not in rl_df.columns and c not in m7.columns
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
        feature_manifest=args.feature_manifest,
        passthrough_cols=passthrough_cols,
    )

    logger.info("Saved: %s", output_path)
    logger.info("Final shape: rows=%d cols=%d", len(out_df), len(out_df.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
