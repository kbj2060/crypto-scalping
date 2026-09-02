#!/usr/bin/env python3
"""B12: TabPFN **분류** 후보 -- 2x2의 마지막 빈 칸 (2026-09-03, 서버 GPU).

  |          | 회귀            | 분류            |
  | HGB      | 현행 (B9)       | B11             |
  | TabPFN   | B10 (회귀기)    | **여기**        |

B10은 `TabPFNRegressor`로 돌렸는데 TabPFN의 주력은 **분류기**다(이 저장소의 증거신호 8종
메타라벨도 전부 TabPFNClassifier). 회귀기 결과(3창 −0.08~−2.39bp)가 TabPFN을 과소평가했을 수 있다.

라벨은 B11에서 경쟁력이 확인된 `y > 40bp` (TRAIN 균형 28.2%)를 쓴다 -- τ와 정확히 일치.
컨텍스트 제약은 그대로다: TRAIN 81,168 중 18,000씩 4멤버 = **멤버당 22%**.

⭐사전등록(동일): VAL·OOS 양 창 승리 못 하면 현행 회귀 동결. 이기면 5시드 중 4개 이상 개별 승리
+ 대조군 3종 재통과. ⚠️9번째 후보라 근소한 승리는 채택 근거가 못 된다.
"""
from __future__ import annotations

import json
import os
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
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
OUT = ROOT / "tmp/eth_entry_b12_tabpfn_cls_20260903"
DEPTH, WAIT, TAU0, NSLOT = 3.0, 6, 0.0040, 4
LABEL_THR = 0.0040
SUBSAMPLE, N_MEMBERS = 18000, 4


def log(m): print(f"[b12] {m}", flush=True)


def main() -> int:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier

    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    excl = set(base + ["arm", "sig_id", "atr_pct", "depth", "y", "split", "timestamp", "i",
                       "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"])
    R = [c for c in D.columns if c.endswith("_r136")] + \
        [c for c in D.columns if c not in excl and not c.endswith("_r136")]
    R = list(dict.fromkeys([c for c in R if D[c].dtype.kind in "fiub"]))
    FEATS = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R))
    X = D[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (D.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    y = D["y"].to_numpy()
    lab = (y > LABEL_THR).astype(int)
    dsel = ((D.depth == DEPTH) & (D.btf <= WAIT)).to_numpy()
    log(f"행 {len(D):,} · TRAIN {int(tr.sum()):,} · 피쳐 {len(FEATS)} · "
        f"라벨 y>{LABEL_THR*1e4:.0f}bp 균형 {float(lab[tr].mean()):.1%}")

    # 현행 회귀 기준선
    reg = {s: HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
           .fit(X[tr], y[tr]).predict(X) for s in SEEDS}
    p_reg = np.mean([reg[s] for s in SEEDS], axis=0)
    fracs, cur = {}, {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == wn).to_numpy()
        fracs[wn] = float((p_reg[m] > TAU0).mean())
        cur[wn] = stat(slotN(D[m][p_reg[m] > TAU0], NSLOT))[1]
    log("현행 회귀  " + " ".join(f"{k} {v:+.2f}bp(유지 {fracs[k]:.1%})" for k, v in cur.items()))

    pred_rows = np.flatnonzero(dsel)
    Xp = X.iloc[pred_rows].to_numpy()
    itr = np.flatnonzero(tr)
    log(f"TabPFN 분류: {SUBSAMPLE:,}×{N_MEMBERS}멤버 (멤버당 {SUBSAMPLE/len(itr):.0%}) · 예측 {len(pred_rows):,}행")
    members = []
    for k in range(N_MEMBERS):
        rs = np.random.default_rng(SEEDS[k]).choice(itr, size=min(SUBSAMPLE, len(itr)), replace=False)
        clf = TabPFNClassifier(device="cuda", random_state=SEEDS[k])
        clf.fit(X.iloc[rs].to_numpy(), lab[rs])
        members.append(clf.predict_proba(Xp)[:, 1])
        log(f"  멤버 {k+1}/{N_MEMBERS} 완료")
    pt = np.full(len(D), np.nan); pt[pred_rows] = np.mean(members, axis=0)

    tab = {}
    log("\n=== 비교 (동일 유지비율) ===")
    print(f"{'구간':9s} {'현행 회귀':>13s} {'TabPFN 분류':>15s} {'차이':>9s}")
    for wn in ("VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == wn).to_numpy()
        pv = pt[m]; thr = np.quantile(pv, 1 - fracs[wn])
        nn, mm = stat(slotN(D[m][pv > thr], NSLOT))[:2]
        tab[wn] = mm
        print(f"{wn:9s} {cur[wn]:+10.2f}bp {mm:+12.2f}bp {mm-cur[wn]:+8.2f}  (n={nn})")
    win = (tab["VAL"] > cur["VAL"]) and (tab["OOS"] > cur["OOS"])
    log(f"\n⭐사전등록 판정: TabPFN 분류가 양 창에서 현행을 "
        + ("**이김 → 시드별 개별 검정 필요**" if win else "**못 이김 → 회귀(HGB squared) 동결 유지**"))
    if win:
        for wn in ("VAL", "OOS"):
            m = dsel & (D.split == wn).to_numpy()
            cv = [stat(slotN(D[m][reg[s][m] > TAU0], NSLOT))[1] for s in SEEDS]
            sv = [stat(slotN(D[m][members[k][ (dsel[pred_rows]) if False else slice(None)][
                 np.isin(pred_rows, np.flatnonzero(m))] > np.quantile(
                 members[k][np.isin(pred_rows, np.flatnonzero(m))], 1 - fracs[wn])], NSLOT))[1]
                 for k in range(N_MEMBERS)]
            log(f"  {wn:5s} 멤버별 {['%+.1f' % v for v in sv]} vs 회귀시드 {['%+.1f' % v for v in cv]}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"current_reg": cur, "tabpfn_cls": tab, "keep_fracs": fracs, "wins": bool(win),
               "label_thr_bp": LABEL_THR * 1e4, "label_balance": float(lab[tr].mean()),
               "subsample": SUBSAMPLE, "members": N_MEMBERS,
               "data_per_member": SUBSAMPLE / len(itr)}, open(OUT / "b12_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
