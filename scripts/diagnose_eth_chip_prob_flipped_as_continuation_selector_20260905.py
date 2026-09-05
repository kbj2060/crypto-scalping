#!/usr/bin/env python3
"""칩 확률을 **뒤집어** 지속 선택기로 쓸 수 있는가 (2026-09-05).

사용자 아이디어: *"증거신호의 라벨 0을 지속 신호의 라벨 1로 두고 학습하면 지속 증거신호가 되는 거 아니야?"*

⚠️**먼저 짚을 것 — 라벨을 뒤집어 학습해도 같은 분류기다.** 이진 분류에서 y→1−y로 학습하면
확률이 p→1−p로 뒤집힐 뿐 순위가 동일하고 AUC(1−p, 1−y) = AUC(p, y)다. 즉 "지속 증거신호"를
새로 학습하는 것은 **이미 서빙 중인 칩 확률을 반대로 읽는 것**과 정확히 같다. 새 정보는 없다.
⇒ 진짜 질문은 "그 확률로 **고르면** 지속 규칙(cont_all)보다 버는가"이고, 이 스크립트가 그걸 잰다.

측정
  A  칩 확률(OOF)이 "지속이 이익인가"(net_bp_flip > 0)를 얼마나 가르는가 — AUC, 분위별 손익
  B  hit 라벨과 지속 수익성의 관계 — P(지속 이익 | hit=0) vs P(지속 이익 | hit=1)
  C  **선택 팔**: 칩 확률 하위 q%(신호별 **인과 확장분위**, 번인 50 — 라이브 구현 가능 형태)일 때만
     지속 진입 → cont_all 대비 일별 짝비교 CI (§5.27 표준). q ∈ {20, 30, 50}
  D  **사이징 팔**: w ∝ (1 − 인과분위), 평균 1 정규화 — 건수를 안 줄이는 형태
판정: VAL·OOS 두 창 모두 짝비교 CI 하한 > 0. HOLDOUT 미접촉.
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


C1 = _load("c1_flip", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
OOFD = ROOT / "tmp/eth_entry_oof_metalabel_20260903"
BURN, WINDOWS = 50, ("TRAIN", "VAL", "OOS")


def causal_pct(F, p):
    """신호별 **확장 분위**: 각 발동에서 같은 신호의 이전 발동들 대비 백분위. 번인 전에는 NaN."""
    out = np.full(len(p), np.nan)
    for s, g in F.groupby("signal", sort=False):
        idx = g.index.to_numpy(); q = p[idx]
        for j in range(BURN, len(q)):
            past = q[:j][np.isfinite(q[:j])]
            if len(past) >= BURN and np.isfinite(q[j]):
                out[idx[j]] = (past < q[j]).mean()
    return out


def main():
    B = C1.build()
    pos, sd, split, ts = B["pos"], B["sd"], B["split"], B["ts"]
    cont_bp, fade_bp, cont_ex = B["cont_bp"], B["fade_bp"], B["cont_ex"]
    Fp = B["Fp"].reset_index(drop=True)
    rows = []
    for s in C1.SIGNALS:
        d = pd.read_csv(OOFD / f"{s}_oof.csv", usecols=["pos", "side", "hit", "proba_oof"])
        d["is_downside"] = (d["side"] == "bottom").astype(int); d["signal"] = s
        rows.append(d[["pos", "is_downside", "signal", "hit", "proba_oof"]])
    O = pd.concat(rows).drop_duplicates(["pos", "is_downside", "signal"])
    key = pd.DataFrame({"pos": pos, "is_downside": sd, "signal": Fp["signal"].to_numpy()}).merge(
        O, on=["pos", "is_downside", "signal"], how="left")
    p = key["proba_oof"].to_numpy(float); hit = key["hit"].to_numpy(float)
    fin = np.isfinite(p)
    print(f"발동 {len(pos):,} · 칩 확률 유효 {fin.mean():.3f} (창별 {[round(float(fin[split==w].mean()),3) for w in WINDOWS]})\n")

    from sklearn.metrics import roc_auc_score
    print("=== A. 칩 확률이 '지속이 이익인가'를 가르는가 (1−p 기준 AUC) ===")
    for w in WINDOWS:
        m = (split == w) & fin
        y = (cont_bp[m] > 0).astype(int)
        a_cont = roc_auc_score(y, 1.0 - p[m]); a_hit = roc_auc_score(hit[m], p[m])
        print(f"  {w:5s} n={int(m.sum()):>6}  AUC(1−p → 지속 이익) {a_cont:.4f}   [참고] AUC(p → hit) {a_hit:.4f}   지속 이익률 {y.mean():.3f}")
    print("\n=== B. hit 라벨 자체와 지속 수익성 ===")
    for w in WINDOWS:
        m = (split == w) & np.isfinite(hit)
        for hv in (0, 1):
            mm = m & (hit == hv)
            print(f"  {w:5s} hit={hv}  n={int(mm.sum()):>6}  P(지속 이익) {(cont_bp[mm]>0).mean():.3f}  지속 {cont_bp[mm].mean():+7.2f}bp")
    print("\n=== C/D. 인과 분위 기반 선택·사이징 팔 vs cont_all ===")
    cp = causal_pct(Fp, p)
    base = {w: C1.pf(C1.cand_of(ts[split == w], pos[split == w] + 1, pos[split == w] + 1 + cont_ex[split == w], cont_bp[split == w])) for w in WINDOWS}
    for w in WINDOWS:
        print(f"  {w:5s} cont_all: n={base[w]['stats']['n']} exp={base[w]['stats']['exp_bp']}bp 일평균={base[w]['stats']['daily_mean_bp']} 샤프={base[w]['stats']['daily_sharpe_ann']}")
    arms = {}
    for q in (0.20, 0.30, 0.50):
        arms[f"select_p하위{int(q*100)}%"] = ("filter", np.isfinite(cp) & (cp <= q))
    arms["size_w∝(1−분위)"] = ("size", None)
    for nm, (kind, sel) in arms.items():
        line = f"  {nm:20s}"
        for w in WINDOWS:
            m = split == w
            if kind == "filter":
                mm = m & sel
                if mm.sum() < 100:
                    line += f" | {w} n<100"; continue
                r = C1.pf(C1.cand_of(ts[mm], pos[mm] + 1, pos[mm] + 1 + cont_ex[mm], cont_bp[mm]))
            else:
                wt = np.where(np.isfinite(cp[m]), 1.0 - cp[m], 0.5); wt = wt / wt.mean()
                r = C1.pf(C1.cand_of(ts[m], pos[m] + 1, pos[m] + 1 + cont_ex[m], cont_bp[m] * wt))
            d = C1.day_paired(r["pnl"], r["ts"], base[w]["pnl"], base[w]["ts"])
            line += f" | {w} n={r['stats']['n']:>5} exp={r['stats']['exp_bp']:>6} Δ={d['diff_bp_day']:>7}{str(d['ci95']):>18} 이긴날={d['win_day_frac']}"
        print(line)
    print("\n=== 참고: 칩 확률 5분위별 지속 손익 (TRAIN 분위 경계) ===")
    tr = (split == "TRAIN") & fin
    edges = np.quantile(p[tr], [0.2, 0.4, 0.6, 0.8]); qi = np.where(fin, np.digitize(p, edges), -1)
    for w in WINDOWS:
        m = (split == w) & fin
        print(f"  {w:5s} " + "  ".join(f"Q{k+1}(p낮음~높음) {cont_bp[m & (qi==k)].mean():+7.2f}(n{int((m&(qi==k)).sum())})" for k in range(5)))


if __name__ == "__main__":
    main()
