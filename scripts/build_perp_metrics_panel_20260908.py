#!/usr/bin/env python3
"""60종 무기한 5분봉 **메트릭 패널** 캐시 (2026-09-08).

`binance_data/metrics/[SYM]-metrics-YYYY-MM-DD.zip` (종목당 945일, 총 56,700파일)에서
sum_open_interest · sum_open_interest_value · count_toptrader_long_short_ratio ·
sum_toptrader_long_short_ratio · count_long_short_ratio · sum_taker_long_short_vol_ratio
를 가격 패널(`tmp/xsec_perp_screen_20260908/panel.npz`)의 5분 타임스탬프 격자에 맞춰 저장한다.
"""
from __future__ import annotations
import glob, io, os, zipfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MDIR = ROOT / "binance_data/metrics"
PAN = ROOT / "tmp/xsec_perp_screen_20260908/panel.npz"
OUTF = ROOT / "tmp/xsec_perp_screen_20260908/metrics_panel.npz"
COLS = ("sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio")


def main() -> int:
    z = np.load(PAN, allow_pickle=True)
    ts = pd.DatetimeIndex(pd.to_datetime(z["ts"])); syms = list(z["syms"])
    print(f"격자 {len(ts):,}봉 × {len(syms)}종목", flush=True)
    M = {c: np.full((len(ts), len(syms)), np.nan, np.float32) for c in COLS}
    for si, s in enumerate(syms):
        files = sorted(glob.glob(str(MDIR / f"{s}-metrics-*.zip")))
        if not files:
            print(f"  [{si+1}/{len(syms)}] {s}: 파일 없음", flush=True); continue
        parts = []
        for f in files:
            try:
                zf = zipfile.ZipFile(f)
                for n in zf.namelist():
                    if n.endswith(".csv"):
                        parts.append(pd.read_csv(io.BytesIO(zf.read(n))))
            except Exception:
                continue
        if not parts: continue
        d = pd.concat(parts, ignore_index=True)
        d["t"] = pd.to_datetime(d["create_time"], errors="coerce")
        d = d.dropna(subset=["t"]).sort_values("t").drop_duplicates("t", keep="last").set_index("t")
        d = d.reindex(ts)
        for c in COLS:
            if c in d.columns:
                M[c][:, si] = pd.to_numeric(d[c], errors="coerce").to_numpy(np.float32)
        if (si + 1) % 10 == 0:
            print(f"  [{si+1}/{len(syms)}] {s} · 파일 {len(files)} · 커버 "
                  f"{np.isfinite(M['sum_open_interest'][:, si]).mean():.1%}", flush=True)
    np.savez_compressed(OUTF, ts=ts.to_numpy(), syms=np.array(syms), **M)
    print(f"\n저장 {OUTF} ({OUTF.stat().st_size/1e6:.0f}MB)")
    for c in COLS:
        print(f"  {c:>36}: 커버 {np.isfinite(M[c]).mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
