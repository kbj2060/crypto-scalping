#!/usr/bin/env python3
"""원시 시퀀스 딥러닝 — **방향** + 양성대조 (2026-09-13).

사용자: *"딥러닝 모델로 다시 만들어보자."*

## 왜 이 자리만 비어 있나
`research_eth_raw_sequence_killgate_with_triggers_20260911.py` 가 이미 **원시 봉 시퀀스 TCN +
양성대조** 하니스를 만들어 청산 타이밍을 닫았다. 그런데 그 라벨은 변동성확장·레짐·극점·
V자반등·청산보유 다섯이고 **방향이 없다**. 방향은 표 피쳐(45D RL 상태 · 41열 스택)로만 쟀다.
⇒ 「원시 시퀀스를 시퀀스 모델에 넣으면 표 피쳐가 못 본 걸 볼 수도 있다」는 반론이
**방향에 대해서만 아직 안 닫혔다.** 이 스크립트가 그걸 닫는다.

## ⭐이 하니스가 오늘 문제에 특히 맞는 이유
`norm_windows` 가 **창 자신의 통계로 표준화**한다 — **변동성 수준을 지운다**. 오늘 하루 내내
「방향처럼 보이던 게 전부 변동성이었다」였는데(09-13 오라클 판별: 롱·숏 MFE 가 같이 커짐),
이 입력에서는 그 경로가 **구조적으로 막혀 있다**. 여기서 방향이 나오면 진짜 방향이다.

## ⭐⭐양성대조가 이 실험의 전부다
딥러닝이 실패했을 때 **«데이터에 정보가 없다»와 «내 모델이 망가졌다»는 겉보기가 같다.**
그래서 같은 모델·같은 입력·같은 루프로 **학습이 되는 축**을 같이 푼다(선행 실측:
변동성확장 .805 · 극점 .722 · 레짐 .699). 대조가 높은데 방향만 .50 이면 **정보 탓**,
대조까지 .54 면 **이 스크립트가 고장난 것**이다.

## 라벨 (전부 봉 t 에서 시퀀스가 끝나고 t+1 부터 본다)
  dir12 / dir48  방향 — close[t+1+H] > open[t+1]           ⭐이번에 새로 넣는 것
  volexp         변동성확장 — 앞 H봉 안에 volexp ≥ 1.8      (양성대조)
  extreme        극점 — 앞 12봉 안에 ±6봉 국소극점           (양성대조)
  ⭐dir12_pnl    방향인데 **|전방수익|으로 표본 가중** — 오늘 얻은 «정확도가 아니라 손익» 교훈의
                 딥러닝판. 크게 움직일 때 맞히는 걸 더 중요하게 친다.
표 피쳐 기준선(연속 하위값 HGB)도 같은 행에서 같이 낸다.
자체점검 --selftest
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import research_eth_raw_sequence_killgate_with_triggers_20260911 as K  # noqa: E402
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

L, CH = K.L, K.CH
SPLITS = K.SPLITS


def windows(ch: np.ndarray) -> np.ndarray:
    """(n, CH, L) — 봉 t 에서 끝나는 길이 L 창. t < L-1 은 못 쓴다."""
    n = len(ch)
    idx = np.arange(L)[None, :] + np.arange(n - L + 1)[:, None]
    out = np.full((n, CH, L), np.nan, np.float32)
    out[L - 1:] = ch[idx].transpose(0, 2, 1)
    return out


def make_labels(eth: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    o = eth["open"].to_numpy(float); c = eth["close"].to_numpy(float)
    hi = eth["high"].to_numpy(float); lo = eth["low"].to_numpy(float)
    n = len(eth)
    ent = np.roll(o, -1); ent[-1] = np.nan          # 진입 = 다음 봉 시가(경계 계약)
    Y, W = {}, {}
    for H in (12, 48):
        f = np.full(n, np.nan)
        f[: n - H - 1] = c[H + 1 : n] / ent[: n - H - 1] - 1.0
        Y[f"dir{H}"] = np.where(np.isfinite(f), (f > 0).astype(float), np.nan)
        if H == 12:
            W["dir12_pnl"] = np.abs(f)              # ⭐|수익|으로 표본 가중
            Y["dir12_pnl"] = Y["dir12"]
    lr = np.diff(np.log(np.maximum(c, 1e-12)), prepend=0.0)
    ve = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    fut = pd.Series(ve).shift(-1).rolling(24, min_periods=24).max().shift(-23).to_numpy()
    Y["volexp(대조)"] = np.where(np.isfinite(fut), (fut >= 1.8).astype(float), np.nan)
    fh = pd.Series(hi).shift(-1).rolling(12, min_periods=12).max().shift(-11).to_numpy()
    fl = pd.Series(lo).shift(-1).rolling(12, min_periods=12).min().shift(-11).to_numpy()
    Y["extreme(대조)"] = np.where(np.isfinite(fh) & np.isfinite(fl),
                                  ((fh <= hi) | (fl >= lo)).astype(float), np.nan)
    return Y, W


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        d = pd.DataFrame({"open": [10, 10, 11, 10, 10.0], "high": [10, 12, 11, 10, 10.0],
                          "low": [10, 9, 11, 10, 10.0], "close": [10, 11, 11, 9, 10.0],
                          "volume": [1.0] * 5, "taker_buy_base": [0.5] * 5})
        Y, W = make_labels(d)
        assert "dir12" in Y and "dir12_pnl" in W
        ch = K.build_channels(d)
        assert ch.shape == (5, CH), ch.shape
        w = windows(ch)
        assert w.shape == (5, CH, L) and np.isnan(w[0]).all(), w.shape
        nw = K.norm_windows(np.random.default_rng(0).normal(5, 3, (4, CH, L)).astype(np.float32))
        assert abs(float(nw.mean())) < 0.2 and abs(float(nw.std()) - 1) < 0.2, (nw.mean(), nw.std())
        print("selftest OK — 창 구성(봉 t 에서 끝남) · 라벨 경계 · 창정규화가 수준을 지운다")
        return 0

    kl = B._load_kl(B.ETH_KL)
    kl = kl[kl.timestamp >= pd.Timestamp(a.start)].reset_index(drop=True)
    ch = K.build_channels(kl)
    Xw = windows(ch)
    Y, W = make_labels(kl)
    ts = kl.timestamp.to_numpy()
    ok0 = np.isfinite(Xw).all(axis=(1, 2))
    masks = {}
    for k, (s, e) in SPLITS.items():
        masks[k] = ok0 & (ts >= np.datetime64(s)) & (ts <= np.datetime64(e + "T23:59:59"))
    print(f"ETH 5분봉 {len(kl):,} · 창 L={L} · 채널 {CH} · "
          f"IS {masks['IS'].sum():,} / VAL {masks['VAL'].sum():,} / OOS {masks['OOS'].sum():,}\n")
    print(f"{'타깃':<16}{'기저':>8}{'IS AUC':>9}{'VAL AUC':>9}{'OOS AUC':>9}{'표 기준선(OOS)':>15}")
    for name, y in Y.items():
        m = {k: v & np.isfinite(y) for k, v in masks.items()}
        if min(v.sum() for v in m.values()) < 2000:
            print(f"{name:<16}  표본 부족"); continue
        t0 = time.time()
        r = K.train_eval(Xw, np.nan_to_num(y), m)
        base = float(np.nanmean(y[m["OOS"]]))
        print(f"{name:<16}{base:>8.3f}{r.get('IS', float('nan')):>9.4f}"
              f"{r.get('VAL', float('nan')):>9.4f}{r.get('OOS', float('nan')):>9.4f}"
              f"{'':>15}   ({time.time()-t0:.0f}s)", flush=True)
    print("\n⭐읽는 법: 대조(volexp·extreme)가 .70~.80 인데 방향만 .50 이면 **정보 탓**이다.")
    print("           대조까지 .54 면 이 스크립트가 고장난 것이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
