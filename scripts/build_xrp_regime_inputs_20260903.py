#!/usr/bin/env python3
"""XRP 레짐 파이프라인 **상류 입력 생성** — BTC/ETH엔 있고 XRP엔 없던 것들.

BTC Phase 2(`research_btc_regime_label_conditional_lift_20260902.py`)가 요구하는 입력 중
XRP에 없는 것을 만든다. 실사 결과 **3개 전부 부재**였다:

    data/xrp_5m_1year.csv                       ❌ -> klines CSV에서 생성
    data/research/funding_extracted/XRPUSDT/    ❌ -> 펀딩 zip에서 추출
    data/splits/year_oos/xrp_features_2024_2026.csv  ❌ -> **만들지 않는다**(아래 참조)

⭐**캐노니컬 피쳐 파일은 만들지 않는다.** BTC Phase 2에서 그 파일은 `build_btc_pivots()`가
지그재그 피벗을 뽑는 용도로만 쓰인다(OHLC 4컬럼만 읽는다). XRP는 klines CSV 자체가 같은
구간(2024-01~2026-08)을 덮으므로 그걸 직접 쓴다 — 없는 파일을 만들려고 상류 파이프라인을
통째로 돌리는 것보다 정직하고 짧다.

⚠️XRP klines는 2026-08-04까지다(ETH 08-28 대비 24일 뒤처짐). 레짐 평가창(VAL/OOS)은
2026-02-17까지이므로 영향 없다.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KLINES = ROOT / "binance_data/klines/XRPUSDT/XRPUSDT-5m-api.csv"
FUNDING_ZIP_DIR = ROOT / "binance_data/funding_rate_other"
OUT_KLINES = ROOT / "data/xrp_5m_1year.csv"
OUT_FUNDING = ROOT / "data/research/funding_extracted/XRPUSDT"
# BTC/ETH의 5m_1year 파일과 같은 종료 시점(레짐 평가창이 그 안에 있다)
END = pd.Timestamp("2026-02-17 15:00:00")


def log(m): print(f"[xrp-regime-in] {m}", flush=True)


def main() -> int:
    # ── klines ──
    d = pd.read_csv(KLINES)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if d["timestamp"].dt.tz is not None:
        d["timestamp"] = d["timestamp"].dt.tz_localize(None)
    d = d.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    d = d.loc[d["timestamp"] <= END].reset_index(drop=True)
    OUT_KLINES.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT_KLINES, index=False)
    gaps = int((d["timestamp"].diff().dt.total_seconds().dropna() / 300 != 1).sum())
    log(f"klines -> {OUT_KLINES.name}  {len(d):,}행  {d.timestamp.min()} ~ {d.timestamp.max()}  갭 {gaps}")

    # ── funding ──
    OUT_FUNDING.mkdir(parents=True, exist_ok=True)
    n_f = 0
    for p in sorted(FUNDING_ZIP_DIR.glob("XRPUSDT-fundingRate-*.zip")):
        out = OUT_FUNDING / (p.stem + ".csv")
        with zipfile.ZipFile(p) as z:
            with z.open(z.namelist()[0]) as f:
                pd.read_csv(f).to_csv(out, index=False)
        n_f += 1
    log(f"funding -> {OUT_FUNDING}/  {n_f}개월")
    fr = pd.concat([pd.read_csv(p) for p in sorted(OUT_FUNDING.glob("*.csv"))], ignore_index=True)
    log(f"  컬럼 {list(fr.columns)}  {len(fr):,}행")

    (ROOT / "data/research/funding_extracted/XRPUSDT/_build_report.json").write_text(json.dumps(
        {"klines_rows": int(len(d)), "klines_gaps": gaps,
         "klines_range": [str(d.timestamp.min()), str(d.timestamp.max())],
         "funding_months": n_f, "funding_rows": int(len(fr)),
         "canonical_features": "만들지 않음 -- Phase2는 피벗 계산에 OHLC만 쓰므로 klines를 직접 사용",
         }, ensure_ascii=False, indent=2))
    log("완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
