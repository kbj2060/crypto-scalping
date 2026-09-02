#!/usr/bin/env python3
"""B11: 분류 재구성 -- 회귀 후 임계 vs 결정 경계 직접 학습 (2026-09-03).

B9는 "아키텍처 스윕"이라 했지만 **회귀 안에서 손실함수만** 바꿨다(squared/absolute/quantile/
winsorize) + B10의 TabPFN 회귀. **분류로 재구성하는 축은 안 돌렸다.** 여기서 채운다.

현행은 연속 수익을 회귀한 뒤 τ=40bp로 자른다 -- 즉 **크기는 학습에 쓰고 결정에는 버린다.**
분류는 결정 경계를 직접 최적화하되 −150bp와 +30bp를 같은 음성으로 뭉갠다. 어느 쪽이 나은지는
선험적으로 안 정해진다.

후보
  E  분류 y > 40bp  -- τ와 정확히 일치하는 라벨 (TRAIN p75=+43.3bp이므로 대략 73:27 균형)
  F  분류 y > 0     -- 참고용. 양수 84.9%라 불균형이 실제로 문제인지 확인

⚠️예측 스케일이 다르므로(확률 vs bp) **동일 유지비율**로 비교한다.
⚠️7·8번째 후보라 순차 검정 부담이 크다 -- 근소한 승리는 채택 근거가 못 된다.

⭐사전등록: 양 창 승리 못 하면 현행 동결. 이기면 **5시드 중 4개 이상 개별 승리** + 대조군 3종 재통과.
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
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              HistGradientBoostingRegressor)

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
OUT = ROOT / "tmp/eth_entry_b11_cls_20260903"
DEPTH, WAIT, TAU0, NSLOT = 3.0, 6, 0.0040, 4
B_RND = 150
RNG = np.random.default_rng(20260903)


def log(m): print(f"[b11] {m}", flush=True)


def main() -> int:
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
    dsel = ((D.depth == DEPTH) & (D.btf <= WAIT)).to_numpy()
    log(f"행 {len(D):,} · TRAIN {int(tr.sum()):,} · 피쳐 {len(FEATS)}")
    log(f"라벨 균형(TRAIN): y>40bp {float((y[tr] > 0.0040).mean()):.1%} · y>0 {float((y[tr] > 0).mean()):.1%}")

    def policy(p, wn, frac=None, tau=None):
        m = dsel & (D.split == wn).to_numpy()
        w = D[m]; pv = p[m]
        thr = tau if tau is not None else np.quantile(pv, 1 - frac)
        v = slotN(w[pv > thr], NSLOT)
        return stat(v)[:2]

    # 현행 회귀
    reg = {s: HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
           .fit(X[tr], y[tr]).predict(X) for s in SEEDS}
    p_reg = np.mean([reg[s] for s in SEEDS], axis=0)
    fracs, cur = {}, {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == wn).to_numpy()
        fracs[wn] = float((p_reg[m] > TAU0).mean())
        cur[wn] = policy(p_reg, wn, tau=TAU0)[1]
    log("A 회귀(현행)  " + " ".join(f"{k} {v:+.2f}bp(유지 {fracs[k]:.1%})" for k, v in cur.items()))

    CAND = {"E 분류 y>40bp": 0.0040, "F 분류 y>0": 0.0}
    res, preds = {}, {}
    for name, thr_lab in CAND.items():
        lab = (y > thr_lab).astype(int)
        ps = {s: HistGradientBoostingClassifier(random_state=s, **HP)
              .fit(X[tr], lab[tr]).predict_proba(X)[:, 1] for s in SEEDS}
        p = np.mean([ps[s] for s in SEEDS], axis=0)
        preds[name] = (p, ps)
        res[name] = {wn: policy(p, wn, frac=fracs[wn])[1] for wn in ("VAL", "OOS", "HOLDOUT")}
        log(f"{name:14s} " + " ".join(f"{k} {v:+.2f}bp" for k, v in res[name].items()))

    log("\n=== 비교 (동일 유지비율) ===")
    print(f"{'후보':16s} {'VAL':>10s} {'OOS':>10s} {'HOLDOUT':>10s} | 양 창 승리")
    print(f"{'A 회귀(현행)':16s} {cur['VAL']:+10.2f} {cur['OOS']:+10.2f} {cur['HOLDOUT']:+10.2f} | —")
    winner = None
    for name, r in res.items():
        w = (r["VAL"] > cur["VAL"]) and (r["OOS"] > cur["OOS"])
        print(f"{name:16s} {r['VAL']:+10.2f} {r['OOS']:+10.2f} {r['HOLDOUT']:+10.2f} | {'✅' if w else '❌'}")
        if w and (winner is None or r["VAL"] > res[winner]["VAL"]):
            winner = name

    if winner is None:
        log("\n⭐사전등록 판정: **양 창에서 현행을 이긴 분류 후보 없음 → 회귀(squared_error) 동결 유지**")
    else:
        log(f"\n⭐{winner}가 양 창 승리 → 추가 기준 검정")
        p, ps = preds[winner]
        for wn in ("VAL", "OOS"):
            sv = [policy(ps[s], wn, frac=fracs[wn])[1] for s in SEEDS]
            cv = [policy(reg[s], wn, tau=TAU0)[1] for s in SEEDS]
            beat = sum(a > b for a, b in zip(sv, cv))
            log(f"  {wn:5s} 시드별 개별 승리 {beat}/5 " +
                "(" + ", ".join(f"{a:+.1f}vs{b:+.1f}" for a, b in zip(sv, cv)) + ")")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"current": cur, "candidates": res, "keep_fracs": fracs,
               "winner": winner, "label_balance_train":
               {"gt40bp": float((y[tr] > 0.0040).mean()), "gt0": float((y[tr] > 0).mean())}},
              open(OUT / "b11_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
