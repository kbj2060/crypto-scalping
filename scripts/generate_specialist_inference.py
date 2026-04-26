#!/usr/bin/env python3
"""
2026년 데이터를 rl_training_data_full.csv 에서 분리하고,
롱/숏/프라이머리 DSAC 스페셜리스트 배치 추론 결과를 추가한다.

출력 컬럼 (meta_ 접두사):
  meta_primary_raw, meta_primary_std
  meta_long_logit, meta_long_raw, meta_long_std
  meta_short_logit, meta_short_raw, meta_short_std

사용법:
  python scripts/generate_specialist_inference.py
  python scripts/generate_specialist_inference.py --year 2026 --output data/splits/year_oos/rl_meta_2026.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("spec_infer")


# ─────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────
_DEFAULT_INPUT  = str(_ROOT / "data" / "rl_training_data_full.csv")
_DEFAULT_OUTPUT = str(_ROOT / "data" / "splits" / "year_oos" / "rl_meta_2026.csv")
_DEFAULT_YEAR   = 2026


def _safe(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _row_to_features(row: "pd.Series") -> dict:
    """CSV 한 행을 스페셜리스트 라우터가 요구하는 features dict 로 변환."""
    d = {str(col): _safe(row[col]) for col in row.index}
    # current_spread: 라우터가 요구하는 키, OHLC 로 추산
    close = _safe(row.get("close", 1.0), 1.0)
    high  = _safe(row.get("high",  close))
    low   = _safe(row.get("low",   close))
    d["current_spread"] = float(np.clip((high - low) / max(close, 1e-8), 0.0, 0.05))
    return d


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def run(
    input_csv: str = _DEFAULT_INPUT,
    output_csv: str = _DEFAULT_OUTPUT,
    year: int = _DEFAULT_YEAR,
    long_ckpt: str | None = None,
    short_ckpt: str | None = None,
    single_ckpt: str | None = None,
    log_interval: int = 1000,
) -> None:
    # ── 1. 데이터 로드 & 연도 필터 ──────────────────────────────
    log.info("로드 중: %s", input_csv)
    df = pd.read_csv(input_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df26 = df[df["timestamp"].dt.year == year].reset_index(drop=True)
    log.info("  전체 rows=%d / %d년 rows=%d", len(df), year, len(df26))
    if len(df26) == 0:
        raise ValueError(f"{year}년 데이터 없음: {input_csv}")

    # ── 2. 스페셜리스트 로드 ────────────────────────────────────
    import tempfile, os as _os
    with tempfile.TemporaryDirectory(prefix="spec_inf_") as _tmp:
        _os.environ["DSAC_LIVE_STATE_PATH"]   = _os.path.join(_tmp, "ls.json")
        _os.environ["FUSE_ADAPT_STATE_PATH"]  = _os.path.join(_tmp, "fa.json")
        from trading_bot import DSACSignalRouter

        log.info("스페셜리스트 로드 중...")
        router = DSACSignalRouter(
            model_path=single_ckpt or None,
            long_path=long_ckpt or None,
            short_path=short_ckpt or None,
        )
        long_router    = router.long_router
        short_router   = router.short_router
        primary_router = router.primary_router
        log.info("  로드 완료 (device=%s)", router.device)

    # ── 3. 배치 추론 ─────────────────────────────────────────────
    cols_out = [
        "meta_primary_raw", "meta_primary_std",
        "meta_long_logit",  "meta_long_raw",  "meta_long_std",
        "meta_short_logit", "meta_short_raw", "meta_short_std",
    ]
    results = {c: np.zeros(len(df26), dtype=np.float32) for c in cols_out}

    empty_pos = {}  # 포지션 없음으로 고정 (배치 추론)

    for i, row in enumerate(df26.itertuples(index=False)):
        features = _row_to_features(pd.Series(row._asdict()))

        try:
            _, _, p_info = primary_router.decide(features, empty_pos)
            p_raw = _safe(p_info.get("raw_action", 0.0))
            p_std = _safe(p_info.get("std", 1.0), 1.0)
        except Exception:
            p_raw, p_std = 0.0, 1.0

        try:
            _, _, l_info = long_router.decide(features, empty_pos)
            l_logit = _safe(l_info.get("logit", 0.0))
            l_raw   = _safe(l_info.get("raw_action", 0.0))
            l_std   = _safe(l_info.get("std", 1.0), 1.0)
        except Exception:
            l_logit, l_raw, l_std = 0.0, 0.0, 1.0

        try:
            _, _, s_info = short_router.decide(features, empty_pos)
            s_logit = _safe(s_info.get("logit", 0.0))
            s_raw   = _safe(s_info.get("raw_action", 0.0))
            s_std   = _safe(s_info.get("std", 1.0), 1.0)
        except Exception:
            s_logit, s_raw, s_std = 0.0, 0.0, 1.0

        results["meta_primary_raw"][i] = p_raw
        results["meta_primary_std"][i] = p_std
        results["meta_long_logit"][i]  = l_logit
        results["meta_long_raw"][i]    = l_raw
        results["meta_long_std"][i]    = l_std
        results["meta_short_logit"][i] = s_logit
        results["meta_short_raw"][i]   = s_raw
        results["meta_short_std"][i]   = s_std

        if (i + 1) % log_interval == 0 or i == len(df26) - 1:
            log.info("  추론 %d / %d  (%.1f%%)", i + 1, len(df26), 100.0 * (i + 1) / len(df26))

    # ── 4. 저장 ──────────────────────────────────────────────────
    for col, arr in results.items():
        df26[col] = arr

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df26.to_csv(output_csv, index=False)
    log.info("저장 완료: %s  (rows=%d, cols=%d)", output_csv, len(df26), len(df26.columns))


def main() -> None:
    ap = argparse.ArgumentParser(description="DSAC 스페셜리스트 배치 추론 + 2026 데이터 분리")
    ap.add_argument("--input",        default=_DEFAULT_INPUT)
    ap.add_argument("--output",       default=_DEFAULT_OUTPUT)
    ap.add_argument("--year",         type=int, default=_DEFAULT_YEAR)
    ap.add_argument("--long-ckpt",    default=None)
    ap.add_argument("--short-ckpt",   default=None)
    ap.add_argument("--single-ckpt",  default=None)
    ap.add_argument("--log-interval", type=int, default=1000)
    args = ap.parse_args()

    run(
        input_csv=args.input,
        output_csv=args.output,
        year=args.year,
        long_ckpt=args.long_ckpt,
        short_ckpt=args.short_ckpt,
        single_ckpt=args.single_ckpt,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
