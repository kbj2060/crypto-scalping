#!/usr/bin/env python3
"""확장 패널 — 179종 × 2021-12 ~ 2026-08 (2026-09-10).

## 왜 뒤로 늘리나
09-08 은 *"판정까지 남은 것 = 표본뿐, 비겹침 5일 n=67, 필요 98 ⇒ 추가 5.2개월"* 이라 했다.
**기다리는 대신 뒤로 늘린다.** metrics 아카이브 실제 시작일은 **2021-12-01**(11-01 은 404).
972일 → 1,732일(+78%). 설계가 **동결**돼 더 고를 게 없으므로 저장소 규율상 전 구간이 확인창이다
([[feedback_train_is_the_confirmation_window_for_model_free_rules_20260908]]).
⭐게다가 2022 는 루나·FTX 국면이고 **설계할 때 한 번도 안 본 구간**이라, 5개월 더 기다려
같은 국면 표본을 얻는 것보다 강한 검정이다.

## ⭐모집단은 그 시점 기준으로 잡는다
기존 83종만 뒤로 늘리면 2022 구간의 상위 40위가 **2026년까지 살아남은 종목 중에서만** 뽑힌다 —
[방금 증명한 생존편향](../docs/xsec_crowding_universe_expansion_survivorship_20260910.md)을
확장 구간에 그대로 다시 심는 셈이다. 그래서 **2022-06 시점에 실제 거래되던 142종**을 확인해
현 패널에 없던 96종을 더했다(FTT·SRM·TOMO·HNT·BTCST·BTS 등 그때만 존재한 이름 포함).

⚠️2023 이후 상장분은 확장 구간에서 NaN 이고 유동성 마스크가 알아서 배제한다 — 채우지 않는다.
출력 tmp/xsec_perp_screen_ext_20260910/
"""
from __future__ import annotations

import glob
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
KDIR = ROOT / "binance_data/klines"
MDIR = ROOT / "binance_data/metrics"
OUT = ROOT / ".claude/worktrees/position-exit-monitoring-model-6506ce/tmp/xsec_perp_screen_ext_20260910"
START, END = pd.Timestamp("2021-12-01"), pd.Timestamp("2026-08-28 13:45")
COLS = ("sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio")


def log(m):
    print(f"[ext {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range(START, END, freq="5min")
    files = sorted(glob.glob(str(KDIR / "*/*-5m-api.csv")))
    log(f"격자 {len(ts):,}봉 ({ts[0]} → {ts[-1]}) · klines {len(files)}종")

    op, cl, qv = {}, {}, {}
    for i, f in enumerate(files):
        sym = Path(f).parent.name
        d = pd.read_csv(f, usecols=["timestamp", "open", "close", "quote_volume"],
                        parse_dates=["timestamp"])
        d = d.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
        d = d.reindex(ts)
        if d["close"].notna().sum() < 2000:          # 격자 안 데이터가 거의 없으면 제외
            continue
        op[sym] = d["open"].astype("float32"); cl[sym] = d["close"].astype("float32")
        qv[sym] = d["quote_volume"].astype("float32")
        if (i + 1) % 40 == 0:
            log(f"  klines {i+1}/{len(files)}")
    O = pd.DataFrame(op); C = pd.DataFrame(cl); Q = pd.DataFrame(qv)
    syms = list(O.columns)
    cov = C.notna().mean()
    log(f"가격 패널 {O.shape} · 커버 중앙 {cov.median():.1%} · 전구간 종목 {(cov>0.98).sum()}")
    np.savez_compressed(OUT / "panel.npz", O=O.to_numpy(), C=C.to_numpy(), Q=Q.to_numpy(),
                        ts=ts.to_numpy(), syms=np.array(syms))
    del O, C, Q, op, cl, qv          # ⚠️메트릭 배열(2.1GB)과 동시 보유 회피
    import gc; gc.collect()

    M = {c: np.full((len(ts), len(syms)), np.nan, np.float32) for c in COLS}
    for si, s in enumerate(syms):
        fs = sorted(glob.glob(str(MDIR / f"{s}-metrics-*.zip")))
        parts = []
        for f in fs:
            try:
                zf = zipfile.ZipFile(f)
                for n in zf.namelist():
                    if n.endswith(".csv"):
                        parts.append(pd.read_csv(io.BytesIO(zf.read(n))))
            except Exception:
                continue
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True)
        d["t"] = pd.to_datetime(d["create_time"], errors="coerce")
        d = d.dropna(subset=["t"]).sort_values("t").drop_duplicates("t", keep="last").set_index("t")
        d = d.reindex(ts)
        for c in COLS:
            if c in d.columns:
                M[c][:, si] = pd.to_numeric(d[c], errors="coerce").to_numpy(np.float32)
        if (si + 1) % 20 == 0:
            log(f"  metrics {si+1}/{len(syms)} · 롱숏비 누적커버 "
                f"{np.isfinite(M['count_toptrader_long_short_ratio'][:, :si+1]).mean():.1%}")
    np.savez_compressed(OUT / "metrics_panel.npz", ts=ts.to_numpy(), syms=np.array(syms), **M)
    log(f"저장 {OUT}")
    for c in COLS:
        log(f"  {c:>36}: 커버 {np.isfinite(M[c]).mean():.1%}")
    # 연도별 유효 종목 수 — 확장 구간에서 횡단면이 얼마나 두꺼운지
    yr = pd.DatetimeIndex(ts).year
    log("연도별 롱숏비 유효 종목 수(중앙):")
    R = M["count_toptrader_long_short_ratio"]
    for y in sorted(set(yr)):
        m = yr == y
        log(f"    {y}: {np.median(np.isfinite(R[m]).sum(1)):.0f}종")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
