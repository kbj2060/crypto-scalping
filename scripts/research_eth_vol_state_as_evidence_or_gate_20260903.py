#!/usr/bin/env python3
"""고변동성 상태 -- **증거신호인가 전략 조건인가** (2026-09-03).

사용자 질문: "이 게이트는 그럼 증거신호로 써야 하는 거 아니야?"

배경: `atr_pct >= p90` / `parkinson_vol >= p90`이 3 ATR 페이드에서 3창 전부 크게 양수였다
(무필터 대비 VAL +14 / OOS +12~23 / HOLDOUT +50bp). 그런데 그 값은 **특정 매매 구조와
결합했을 때**의 것이다. 시장 자체에 대한 정보인지는 별개 문제다.

  ① 전방수익을 **방향까지** 예측하는가 (IC / 조건부 평균)  → 예면 시장 신호
  ② 크기(|수익|)만 예측하는가                              → 변동성의 정의일 뿐
  ③ ⭐**8종 증거신호의 적중률을 올리는가**                  → 예면 "언제 신호를 믿을지"의
                                                            메타 게이트로 독립적 가치
  ④ 기존 레짐 분류기(chop/bull/bear)와 겹치는가            → 같으면 대시보드 중복

⚠️게이트 자체는 아직 확립되지 않았다(일 단위 CI가 VAL에서 하한 음수, 독립 일수 8~22일,
p90→p95에서 VAL 부호 반전). 여기서 묻는 것은 **어디에 놓을 물건인가**이지 승격 여부가 아니다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, OOS_START, VAL_START)

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
ETH_REGIME = ROOT / "tmp/eth_regime_s12k3_clean_20260902/predictions.parquet"
W3 = ("VAL", "OOS", "HOLDOUT")


def log(m): print(f"[volstate] {m}", flush=True)


def main() -> int:
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines

    kl = load_klines()
    ind = build_indicator_frame(kl)
    ind["timestamp"] = kl["timestamp"].to_numpy()
    c = kl["close"].to_numpy(float)
    ts = pd.DatetimeIndex(kl["timestamp"]); n = len(kl)
    split = np.where(ts < VAL_START, "TRAIN", np.where(ts < OOS_START, "VAL",
                     np.where(ts < HOLDOUT_START, "OOS", "HOLDOUT")))
    tr = split == "TRAIN"
    a = pd.to_numeric(ind["atr_pct"], errors="coerce").to_numpy(float)
    thr = float(np.nanpercentile(a[tr], 90))
    G = a >= thr                                          # 고변동성 상태
    log(f"봉 {n:,} · atr_pct p90(TRAIN) = {thr:.5f} · 게이트 비율 전체 {G.mean():.3f}")

    # ---- ① 방향 예측 · ② 크기 예측 ----
    print(f"\n=== ①② 고변동성이 전방수익을 예측하는가 ===")
    print(f"{'전방':>6s}{'창':>9s}{'평균수익(고변)':>15s}{'평균수익(저변)':>15s}"
          f"{'|수익|(고변)':>14s}{'|수익|(저변)':>14s}{'방향IC':>9s}")
    for f in (12, 24, 48):
        fw = np.concatenate([(c[f:] - c[:-f]) / c[:-f], np.full(f, np.nan)])
        for w in W3:
            m = (split == w) & np.isfinite(fw)
            hi, lo = m & G, m & ~G
            ic = spearmanr(G[m].astype(float), fw[m])[0]
            print(f"{f:6d}{w:>9s}{np.mean(fw[hi])*1e4:+15.2f}{np.mean(fw[lo])*1e4:+15.2f}"
                  f"{np.mean(np.abs(fw[hi]))*1e4:14.2f}{np.mean(np.abs(fw[lo]))*1e4:14.2f}"
                  f"{ic:+9.4f}")

    # ---- ③ 증거신호 적중률 ----
    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    pos = {t: i for i, t in enumerate(ts)}
    print(f"\n=== ③ ⭐고변동성이 증거신호 적중률을 올리는가 (hit = 라벨) ===")
    print(f"{'신호':>26s}{'전체':>9s}{'고변동':>9s}{'저변동':>9s}{'차이':>9s}{'고변n':>8s}")
    tot_hi, tot_lo = [], []
    for name in cfg:
        d = pd.read_csv(SRC / f"{name}_causal_fires.csv", parse_dates=["timestamp"])
        d = d[d.timestamp.isin(pos)].copy()
        d["g"] = [G[pos[t]] for t in d.timestamp]
        d = d[d.split != "TRAIN"] if "split" in d.columns else d
        if not len(d): continue
        h_all = float(d.hit.mean())
        hh = float(d[d.g].hit.mean()) if d.g.any() else np.nan
        hl = float(d[~d.g].hit.mean()) if (~d.g).any() else np.nan
        tot_hi.append((d[d.g].hit.sum(), d.g.sum())); tot_lo.append((d[~d.g].hit.sum(), (~d.g).sum()))
        print(f"{name:>26s}{h_all:9.3f}{hh:9.3f}{hl:9.3f}{hh-hl:+9.3f}{int(d.g.sum()):8d}")
    sh = sum(x for x, _ in tot_hi) / max(sum(y for _, y in tot_hi), 1)
    sl = sum(x for x, _ in tot_lo) / max(sum(y for _, y in tot_lo), 1)
    print(f"{'합계':>26s}{'':9s}{sh:9.3f}{sl:9.3f}{sh-sl:+9.3f}")

    # ---- ④ 기존 레짐과 겹치는가 ----
    if ETH_REGIME.exists():
        R = pd.read_parquet(ETH_REGIME).rename(columns={"regime": "reg"})
        F = pd.DataFrame({"timestamp": ts, "g": G}).merge(R, on="timestamp", how="left")
        F = F[np.isfinite(F.reg)]
        print(f"\n=== ④ 기존 레짐 분류기와의 관계 (0 bull / 1 bear / 2 chop) ===")
        print(f"{'레짐':>8s}{'봉수':>10s}{'그중 고변동 비율':>18s}")
        for rv, nm in ((0, "bull"), (1, "bear"), (2, "chop")):
            m = F.reg == rv
            print(f"{nm:>8s}{int(m.sum()):10,}{float(F[m].g.mean()):18.3f}")
        print(f"  ⭐고변동 구간의 레짐 분포: "
              + " · ".join(f"{nm} {float((F[F.g].reg==rv).mean()):.3f}"
                           for rv, nm in ((0,'bull'),(1,'bear'),(2,'chop'))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
