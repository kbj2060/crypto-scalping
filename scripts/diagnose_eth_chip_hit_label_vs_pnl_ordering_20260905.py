#!/usr/bin/env python3
"""증거신호 칩 — **"정확도는 높은데 왜 페이드가 손해인가"** 직접 진단 (2026-09-05).

사용자 질문: *"증거신호 각각은 정확도가 높은데 어떻게 (반대로 하는 게 수익이라는 게) 성립하지?"*

칩의 **실제 hit 라벨**(`tmp/eth_entry_oof_metalabel_20260903/<sig>_oof.csv`, 신호별 자체 H/K)을
**실제 F0 손익**(sim_exit 5.0/1.5/0.1, 200봉, −10bp)과 교차시켜 두 가지를 분리한다.

  (1) 기저율   칩 라벨은 손익을 정확히 가른다(hit=1 페이드 +44bp / hit=0 페이드 −33bp).
               문제는 **발동 봉에서의 hit률이 0.42~0.45로 50% 미만**이라는 것 —
               보상이 대칭(±40bp)이므로 기대값이 페이드 −1, 지속 +4로 갈린다.
               AUC가 높다는 건 발동들 사이의 **순위**를 잘 매긴다는 뜻이지 사건이 절반 넘게
               일어난다는 뜻이 아니다(K는 신호별 자기 모집단에서 50/50 보정 — 8종 합집합
               GAP12 첫발동 모집단에서는 내려앉는다).
  (2) 순서     칩 라벨은 "H봉 안 **어느 시점에든** K×ATR 터치"다. 매매는 "**먼저** 터치"를 요구한다.
               신호 방향 +2ATR 터치는 200봉 안 81% 일어나지만, 그게 반대 방향 1.5ATR보다
               먼저 오는 건 43%뿐이다. 나머지가 통째로 손실이다(§5.7 "순수 MFE-터치 라벨의 맹점").

연구/개발 점수. HOLDOUT 미접촉(프레임이 이미 배제).
Usage: python scripts/diagnose_eth_chip_hit_label_vs_pnl_ordering_20260905.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


C1 = _load("c1_diag", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
OOFD = ROOT / "tmp/eth_entry_oof_metalabel_20260903"
K_TOUCH, ARM, SL, BIG = 2.0, 1.5, 5.0, 10 ** 6      # 예시 터치 문턱 + F0 셀의 무장/손절선
WINDOWS = ("TRAIN", "VAL", "OOS")


def main():
    B = C1.build()
    pos, sd, split = B["pos"], B["sd"], B["split"]
    cont_bp, fade_bp, bidx = B["cont_bp"], B["fade_bp"], B["bidx"]
    entry, atr_pct = B["entry"], B["atr"] / B["entry"]
    fs = np.where(sd == 1, 1.0, -1.0)                                  # 페이드 부호: 바닥 발동 → 롱
    ix = (bidx + 1)[:, None] + np.arange(C1.FWD)
    H, L = B["h"][ix], B["l"][ix]
    fav = np.where(fs[:, None] > 0, H, L); adv = np.where(fs[:, None] > 0, L, H)
    mv_f = fs[:, None] * (fav - entry[:, None]) / entry[:, None]       # 페이드 방향 유리 폭
    mv_a = fs[:, None] * (adv - entry[:, None]) / entry[:, None]       # 페이드 방향 불리 폭
    A = atr_pct[:, None]

    def first(cond):
        i = np.argmax(cond, axis=1); return np.where(cond.any(axis=1), i, BIG)

    t_fav = first(mv_f >= K_TOUCH * A); t_arm = first(mv_a <= -ARM * A); t_stop = first(mv_a <= -SL * A)

    rows = []
    for s in C1.SIGNALS:
        d = pd.read_csv(OOFD / f"{s}_oof.csv", usecols=["pos", "side", "hit"])
        d["is_downside"] = (d["side"] == "bottom").astype(int)
        rows.append(d[["pos", "is_downside", "hit"]])
    O = pd.concat(rows).drop_duplicates(["pos", "is_downside"])
    hit = pd.DataFrame({"pos": pos, "is_downside": sd}).merge(O, on=["pos", "is_downside"], how="left")["hit"].to_numpy(float)
    print(f"발동 {len(pos):,} · hit 라벨 매칭 {int(np.isfinite(hit).sum()):,}\n")

    print("=== (1) 칩 자체 hit 라벨 × 실제 손익 (bp, 10bp 차감) ===")
    for w in WINDOWS:
        m = (split == w) & np.isfinite(hit)
        for hv, nm in ((1, "hit=1 (칩이 맞은 발동)"), (0, "hit=0 (칩이 틀린 발동)")):
            mm = m & (hit == hv)
            print(f"  {w:5s} {nm:22s} n={int(mm.sum()):>6}  페이드 {fade_bp[mm].mean():+7.2f}   지속 {cont_bp[mm].mean():+7.2f}")
        p = float((hit[m] == 1).mean())
        print(f"  {w:5s} {'hit률':22s} {p:.3f}   ⇒ 기대 페이드 {p*fade_bp[m & (hit==1)].mean() + (1-p)*fade_bp[m & (hit==0)].mean():+.2f}bp")
    print(f"\n=== (2) 순서: 신호 방향 +{K_TOUCH}ATR 터치가 반대 {ARM}ATR보다 **먼저** 오는가 ===")
    for w in WINDOWS:
        m = split == w
        print(f"  {w:5s} n={int(m.sum()):>6}  200봉 안 터치 {(t_fav[m] < BIG).mean():.3f}  |  "
              f"반대 {ARM}ATR보다 먼저 {(t_fav[m] < t_arm[m]).mean():.3f}  |  페이드 손절({SL}ATR)이 먼저 {(t_stop[m] < t_fav[m]).mean():.3f}")
    print("\n=== (3) 세 갈래 손익 ===")
    for w in WINDOWS:
        m = split == w
        for mm, nm in ((m & (t_fav < BIG) & (t_fav < t_arm), "유리 터치가 먼저"),
                       (m & (t_fav < BIG) & (t_fav > t_arm), "터치하나 역행이 먼저"),
                       (m & (t_fav == BIG), "터치 못함")):
            print(f"  {w:5s} {nm:18s} n={int(mm.sum()):>6} ({mm.sum()/m.sum():.3f})  페이드 {fade_bp[mm].mean():+8.2f}  지속 {cont_bp[mm].mean():+8.2f}")


if __name__ == "__main__":
    main()
