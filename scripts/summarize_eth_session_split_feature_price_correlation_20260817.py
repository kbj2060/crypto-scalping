#!/usr/bin/env python3
"""analyze_eth_session_split_feature_price_correlation_20260817.py 산출물 요약.

판정 기준 (사전 등록):
  1) 세션 안에서 TRAIN / VAL / OOS IC 부호가 모두 같아야 한다 (부호 일관성).
  2) 그 세션-피처-호라이즌 셀의 실제 IC 가 hour-rotation 귀무분포 대비 |z| >= 2 여야 한다.
     귀무분포는 VAL 과 OOS 각각에서 따로 본다 (TRAIN 은 in-sample 이라 참고용).
  3) 세션 분할이 실제로 무언가를 만들었는지 보려면 pooled IC 보다 |IC| 가 커야 한다.
  4) close_level 과의 |Spearman| >= 0.5 인 피처는 price-trend contamination 으로 별도 표시
     (repo 관행: 0.5~0.6 이상이면 실격 후보).
다중비교: 102 feature x 6 horizon x 4 session = 2448 셀/split 이므로 1)+2)+3) 을 모두
통과해야만 후보로 본다. 통과 개수를 rotation-null 이 만들어내는 통과 개수와도 비교한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "tmp/session_split_20260817"
SESSIONS = ["us", "europe", "asia", "none"]
FWD = [f"fwd{h}" for h in [1, 3, 6, 12, 24, 72]]
BAR_LABEL = {"fwd1": "5m", "fwd3": "15m", "fwd6": "30m", "fwd12": "1h", "fwd24": "2h", "fwd72": "6h"}


def main() -> None:
    ic = pd.read_parquet(OUTDIR / "session_ic_with_null.parquet")
    pd.set_option("display.width", 200)

    # ---------------------------------------------------------------- 표본 크기
    sizes = (
        ic[ic.target == "fwd1"].groupby(["split", "session"])["n"].first().unstack("session")
    )
    print("=== 세션별 바 수 ===")
    print(sizes.to_string())

    # ---------------------------------------------------------------- contamination
    lvl = ic[ic.target == "close_level"].pivot_table(
        index="feature", columns=["split", "session"], values="ic"
    )
    lvl_max = lvl.abs().max(axis=1).sort_values(ascending=False)
    contaminated = set(lvl_max[lvl_max >= 0.5].index)
    print(f"\n=== close_level |Spearman| >= 0.5 (price-trend contamination) : {len(contaminated)}개 ===")
    print(lvl_max.head(20).round(3).to_string())

    # ---------------------------------------------------------------- 세션별 IC
    fwd = ic[ic.target.isin(FWD)].copy()
    wide = fwd.pivot_table(index=["session", "target", "feature"], columns="split", values="ic")
    zw = fwd.pivot_table(index=["session", "target", "feature"], columns="split", values="z_vs_null")
    zw.columns = [f"z_{c}" for c in zw.columns]
    tab = wide.join(zw).reset_index()

    pooled = (
        fwd[fwd.session == "pooled"]
        .pivot_table(index=["target", "feature"], columns="split", values="ic")
        .rename(columns=lambda c: f"pooled_{c}")
        .reset_index()
    )
    tab = tab.merge(pooled, on=["target", "feature"], how="left")
    tab = tab[tab.session.isin(SESSIONS)].copy()

    tab["sign_ok"] = (
        (np.sign(tab["TRAIN"]) == np.sign(tab["VAL"]))
        & (np.sign(tab["VAL"]) == np.sign(tab["OOS"]))
        & tab["TRAIN"].notna() & tab["VAL"].notna() & tab["OOS"].notna()
    )
    tab["min_abs_ic"] = tab[["TRAIN", "VAL", "OOS"]].abs().min(axis=1)
    tab["null_ok"] = (tab["z_VAL"].abs() >= 2) & (tab["z_OOS"].abs() >= 2)
    tab["beats_pooled"] = (
        (tab["VAL"].abs() > tab["pooled_VAL"].abs()) & (tab["OOS"].abs() > tab["pooled_OOS"].abs())
    )
    tab["contaminated"] = tab.feature.isin(contaminated)
    tab["pass_all"] = tab.sign_ok & tab.null_ok & tab.beats_pooled

    print("\n=== 기준별 통과 셀 수 (세션 x 기준) ===")
    crit = tab.groupby("session")[["sign_ok", "null_ok", "beats_pooled", "pass_all"]].sum()
    crit["cells"] = tab.groupby("session").size()
    print(crit.to_string())

    print("\n=== pooled 대비 세션별 최고 |IC| (OOS) ===")
    best = (
        tab.assign(abs_oos=tab["OOS"].abs())
        .sort_values("abs_oos", ascending=False)
        .groupby("session")
        .head(3)[["session", "target", "feature", "TRAIN", "VAL", "OOS", "pooled_OOS", "z_OOS", "contaminated"]]
    )
    print(best.round(4).to_string(index=False))

    print("\n=== 3기준 모두 통과 (multi-test 통과 후보) ===")
    cand = tab[tab.pass_all].sort_values("min_abs_ic", ascending=False)
    if cand.empty:
        print("없음")
    else:
        show = cand[
            ["session", "target", "feature", "TRAIN", "VAL", "OOS", "pooled_VAL", "pooled_OOS",
             "z_VAL", "z_OOS", "min_abs_ic", "contaminated"]
        ].copy()
        show["horizon"] = show.target.map(BAR_LABEL)
        print(show.head(40).round(4).to_string(index=False))
        print(f"\n총 {len(cand)}개 / {len(tab)}셀  (오염 제외 {int((~cand.contaminated).sum())}개)")

    tab.to_csv(OUTDIR / "session_summary.csv", index=False)
    print(f"\nWROTE {OUTDIR / 'session_summary.csv'}")


if __name__ == "__main__":
    main()
