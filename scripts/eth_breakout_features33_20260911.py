#!/usr/bin/env python3
"""돌파 예고/경보 모델의 **33개 피쳐 단일 출처** (2026-09-11).

🔴학습과 라이브가 이 함수 하나를 공유한다. 두 곳에 같은 수식을 두면 조용히 어긋난다
  (저장소의 학습/추론 파리티 계약). 여기 코드는 `alert_base_20260911.build()` 의
  피쳐 구간을 그대로 옮긴 것이고, `--parity` 로 두 경로가 같은 값을 내는지 검증한다.

입력: timestamp · open · high · low · close · quote_volume · trades · taker_buy_quote
출력: 33개 피쳐 DataFrame (열 순서 고정 — 모델 입력 순서가 바뀌면 안 된다)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPRESS = 0.70


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    """봉 t 까지만 쓴다. 어떤 열도 미래를 보지 않는다."""
    d = d.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    op = d.open.to_numpy(float)
    qv = d.quote_volume.to_numpy(float); n = d.trades.to_numpy(float)
    tbq = d.taker_buy_quote.to_numpy(float)
    lr = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(max(c[0], 1e-12)))
    S = pd.Series(lr)
    rv12, rv288 = S.rolling(12).std(), S.rolling(288).std()
    volexp = (rv12 / rv288).to_numpy()
    comp = (volexp < COMPRESS) & np.isfinite(volexp)

    tr = np.maximum.reduce([hi - lo, np.abs(hi - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))])
    atr = pd.Series(tr).rolling(96).mean().to_numpy() / np.maximum(c, 1e-9)
    ats = qv / np.maximum(n, 1)                                  # 평균 체결 크기
    tbr = np.abs(tbq / np.maximum(qv, 1e-9) - 0.5)               # 테이커 |쏠림|

    F: dict[str, np.ndarray] = {}
    for nm, s in (("n", n), ("qv", qv), ("ats", ats), ("tbr", tbr)):
        ss = pd.Series(s)
        for W in (96, 288, 864, 2016):
            z = ((ss - ss.rolling(W).mean()) / ss.rolling(W).std()).to_numpy()
            F[f"z_{nm}_{W}"] = z
            if W in (864, 2016) and nm in ("n", "qv"):
                # 배포 경보의 "3봉 지속" 아이디어 — 한 봉 튀는 것과 유지되는 것을 가른다
                F[f"z_{nm}_{W}_min3"] = pd.Series(z).rolling(3).min().to_numpy()
    F["volexp"] = volexp
    F["volexp_d1"] = np.r_[0.0, np.diff(volexp)]
    F["volexp_d12"] = volexp - np.r_[np.full(12, np.nan), volexp[:-12]]
    F["comp_depth"] = COMPRESS - volexp                           # 얼마나 깊이 눌렸나
    F["atr_pct"] = atr
    F["atr_rank"] = pd.Series(atr).rolling(288).rank(pct=True).to_numpy()
    F["bbw_rank"] = S.rolling(48).std().rolling(288).rank(pct=True).to_numpy()
    F["range_atr"] = (hi - lo) / np.maximum(c * atr, 1e-12)
    F["body_range"] = np.abs(c - op) / np.maximum(hi - lo, 1e-9)
    F["absret_atr"] = np.abs(lr) / np.maximum(atr, 1e-9)
    # 압축이 몇 봉째인가 — 오래 눌릴수록 터질 때 크다는 통념을 모델이 쓸 수 있게 준다
    grp = (~comp).cumsum()
    F["comp_age"] = pd.Series(comp.astype(int)).groupby(grp).cumsum().to_numpy()
    hh = d.timestamp.dt.hour.to_numpy() + d.timestamp.dt.minute.to_numpy() / 60.0
    F["hod_sin"] = np.sin(2 * np.pi * hh / 24); F["hod_cos"] = np.cos(2 * np.pi * hh / 24)

    out = pd.DataFrame(F)
    out.attrs["features"] = list(F.keys())
    return out


def _self_check() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    c = 2000 + np.cumsum(rng.normal(0, 1, n))
    d = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="5min"),
                      "open": c, "high": c + 1, "low": c - 1, "close": c,
                      "quote_volume": rng.normal(5e6, 1e5, n),
                      "trades": rng.normal(5000, 200, n),
                      "taker_buy_quote": rng.normal(2.5e6, 5e4, n)})
    f = build_features(d)
    assert f.shape[1] == 33, f.shape
    # 🔴절단 불변성 -- 봉 t 까지만 줘도 t 의 값이 같아야 한다(미래를 안 본다)
    t = n - 50
    fc = build_features(d.iloc[:t + 1])
    for col in f.columns:
        a, b = float(fc[col].iloc[-1]), float(f[col].iloc[t])
        assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, (col, a, b)
    print(f"self-check OK  (33피쳐 · 절단 불변성 {len(f.columns)}칸)")


if __name__ == "__main__":
    _self_check()
