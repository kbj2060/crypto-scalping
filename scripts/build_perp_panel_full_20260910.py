#!/usr/bin/env python3
"""확대 모집단 패널 빌더 — 가격 + 메트릭 (2026-09-10).

사용자: *"우선 상위 10종으로 진행해줘"* — 60종 → 70종.

## ⚠️기존 패널을 덮어쓰지 않는다
`tmp/xsec_perp_screen_20260908/` 는 09-08 이래 모든 횡단면 결과의 입력이다. 덮어쓰면
**이전 결과들의 근거가 조용히 바뀐다**. 새 디렉터리 `tmp/xsec_perp_screen_70_20260910/` 에 만든다.

## ⭐확대의 진짜 목적은 검정력이 아니라 **모집단 선정의 미래참조 제거**다
기존 60종은 **2026년 현재 유동성**으로 골랐다. 2024년 시점에서는 알 수 없는 정보다.
새로 넣는 10종은 오늘 기준 61~70위지만 2024년엔 상위권이던 이름들이 섞여 있다
(VET·XTZ·SUSHI·AXS 부류). 유동성 순위는 시점마다 다시 계산되므로 그 구간에서 실제로 선택 대상이 된다.
⇒ 09-08 이 한계로 적어둔 *"생존편향: 패널 60종은 현재 상장분만"* 을 직접 공격하는 작업이다.
⚠️따라서 기대 효과는 "표본이 늘어 유의해진다"가 **아니라** "기존 추정치가 낙관이었는지 드러난다" 이다.
   확대 후 성과가 **내려가면 그게 정상**이고, 그때 내려간 값이 더 정직한 값이다.

## 🔴2차 확대(폐지 종목) — 여기가 진짜 생존편향 검정이다
1차(현존 61~70위 10종)는 **생존편향을 못 고친다**. 폐지 종목은 현재 상장 목록에 없어
후보에 들어오지도 않기 때문이다. 아카이브를 훑어 **창 안에서 사라진 11종 + 후속 티커 2종**을 넣었다.
거기 **MATIC(→POL) · RNDR(→RENDER) · EOS** 가 있다 — 2024년 상위권인데 패널에 **통째로 없었다**.
원인: 60종 패널이 사실상 *"전 기간 동일 티커로 연속 데이터가 있는 종목"* 으로 걸러져 있었고,
**티커를 바꾼 자산이 탈락**했다. 이건 우연이 아니라 구조적 누락이다.
⚠️폐지 시점에 보유 중이던 포지션은 `fwd` 가 NaN 이라 자동 제외된다 — 실제로는 강제 청산되므로
   이 처리는 **낙관 쪽**이다. 한계로 기록한다.
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
OUT = ROOT / ".claude/worktrees/position-exit-monitoring-model-6506ce/tmp/xsec_perp_screen_full_20260910"
OLD = ROOT / "tmp/xsec_perp_screen_20260908/panel.npz"
COLS = ("sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio")


def log(m):
    print(f"[panel70 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    old_syms = set(str(s) for s in np.load(OLD, allow_pickle=True)["syms"])
    files = sorted(glob.glob(str(KDIR / "*/*-5m-api.csv")))
    log(f"klines 디렉터리 {len(files)}종 (기존 패널 {len(old_syms)}종)")

    op, cl, qv = {}, {}, {}
    for f in files:
        sym = Path(f).parent.name
        d = pd.read_csv(f, usecols=["timestamp", "open", "close", "quote_volume"],
                        parse_dates=["timestamp"])
        d = d.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
        op[sym] = d["open"].astype("float32"); cl[sym] = d["close"].astype("float32")
        qv[sym] = d["quote_volume"].astype("float32")
    O = pd.DataFrame(op).sort_index(); C = pd.DataFrame(cl).sort_index()
    Q = pd.DataFrame(qv).sort_index()
    # ⚠️격자는 **기존 패널과 같은 구간**으로 자른다 — 구간이 달라지면 확대 효과와 기간 효과가 섞인다
    z0 = np.load(OLD, allow_pickle=True); ts0 = pd.DatetimeIndex(z0["ts"])
    O = O.reindex(ts0); C = C.reindex(ts0); Q = Q.reindex(ts0)
    syms = list(O.columns)
    new = [s for s in syms if s not in old_syms]
    log(f"가격 패널 {O.shape} · 신규 {len(new)}종: {' '.join(new)}")
    for s in new:
        log(f"   {s:14s} 종가 커버 {C[s].notna().mean():6.1%} · "
            f"일평균 달러량 ${Q[s].sum()/max((ts0[-1]-ts0[0]).days,1)/1e6:8.1f}M")
    np.savez_compressed(OUT / "panel.npz", O=O.to_numpy(), C=C.to_numpy(), Q=Q.to_numpy(),
                        ts=ts0.to_numpy(), syms=np.array(syms))

    M = {c: np.full((len(ts0), len(syms)), np.nan, np.float32) for c in COLS}
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
            log(f"  🔴 {s}: metrics 없음"); continue
        d = pd.concat(parts, ignore_index=True)
        d["t"] = pd.to_datetime(d["create_time"], errors="coerce")
        d = d.dropna(subset=["t"]).sort_values("t").drop_duplicates("t", keep="last").set_index("t")
        d = d.reindex(ts0)
        for c in COLS:
            if c in d.columns:
                M[c][:, si] = pd.to_numeric(d[c], errors="coerce").to_numpy(np.float32)
        if s in new:
            log(f"  ✅ 신규 {s:14s} metrics 파일 {len(fs)} · 롱숏비 커버 "
                f"{np.isfinite(M['count_toptrader_long_short_ratio'][:, si]).mean():.1%}")
        elif (si + 1) % 20 == 0:
            log(f"  [{si+1}/{len(syms)}] {s}")
    np.savez_compressed(OUT / "metrics_panel.npz", ts=ts0.to_numpy(),
                        syms=np.array(syms), **M)
    log(f"저장 {OUT}")
    for c in COLS:
        log(f"  {c:>36}: 커버 {np.isfinite(M[c]).mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
