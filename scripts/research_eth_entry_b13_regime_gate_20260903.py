#!/usr/bin/env python3
"""B13: 정책 수준 레짐 게이트 검정 (2026-09-03).

동결 모델에는 레짐 **피쳐**가 20개 들어 있지만(er_24 · chop_index · regime_trending ·
regime_persistence · hurst · state7/12_* 등), 정책 수준 **게이트**("chop일 때만 진입")는 없다.
레짐 분류기 **예측값**은 B1에서 기여 0으로 확인돼 제외됐다(A+B+C OOS 0.0998 → +D 0.0952).

여기서 검정하는 건 세 번째 것 -- **하드 게이트**다. 피쳐 축이 아니라 정책 축이라 별개 질문이다.

⚠️사전 예상은 "별로 안 들을 것"이다. 게이트는 모델이 이미 부드럽게 학습할 수 있는 걸 딱딱하게
강제하는 것이고, 레짐 원재료가 이미 피쳐로 들어가 있다. 그래도 싼 검정이라 확인한다.

레짐은 **OOF 확장창 예측**(tmp/eth_entry_oof_regime_20260903)을 쓴다 -- 누수 없음.
게이트 후보: ETH chop만 / BTC chop만 / 둘 다 chop. (0 bull / 1 bear / 2 chop)

⭐사전등록(10번째 선택 축이라 엄격): 게이트가 **VAL·OOS 양 창**에서 무게이트를 이기고
**5시드 중 4개 이상 개별 승리**해야 채택. 못 넘으면 무게이트 동결.
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

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_eth_entry_b1_cumulative_arms_20260903 import stat  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

ART = ROOT / "tmp/eth_entry_limit_fade_v1_20260903/model.joblib"
B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
OOFR = ROOT / "tmp/eth_entry_oof_regime_20260903"
OUT = ROOT / "tmp/eth_entry_b13_regime_gate_20260903"


def log(m): print(f"[b13] {m}", flush=True)


def main() -> int:
    P = joblib.load(ART)
    pol = P["policy"]
    DEPTH, WAIT, TAU, NSLOT = pol["depth_atr"], pol["wait_bars"], pol["tau"], pol["slots"]
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    X = D[P["feature_cols"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(pd.Series(P["feature_medians"]))
    per_seed = [m.predict(X) for m in P["models"]]
    D["pred"] = np.mean(per_seed, axis=0)
    log(f"동결 모델 재적재 · 행 {len(D):,} · 정책 d{DEPTH}/w{WAIT}/τ{TAU*1e4:.0f}bp/{NSLOT}슬롯")

    for k in ("eth", "btc"):
        r = pd.read_parquet(OOFR / f"regime_oof_{k}.parquet").rename(
            columns={"regime_oof": f"reg_{k}"})[["timestamp", f"reg_{k}"]]
        D = D.merge(r, on="timestamp", how="left")
        D[f"reg_{k}"] = D[f"reg_{k}"].fillna(-1).astype(int)
    dsel = ((D.depth == DEPTH) & (D.btf <= WAIT)).to_numpy()
    log(f"레짐 OOF 조인 · 유효비율 ETH {float((D.reg_eth>=0).mean()):.1%} / BTC {float((D.reg_btc>=0).mean()):.1%}")
    for wn in ("VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == wn).to_numpy()
        log(f"  {wn:8s} chop 비중 ETH {float((D[m].reg_eth==2).mean()):.1%} · "
            f"BTC {float((D[m].reg_btc==2).mean()):.1%} · 둘다 {float(((D[m].reg_eth==2)&(D[m].reg_btc==2)).mean()):.1%}")

    GATES = {
        "무게이트": lambda w: np.ones(len(w), bool),
        "ETH chop": lambda w: (w.reg_eth == 2).to_numpy(),
        "BTC chop": lambda w: (w.reg_btc == 2).to_numpy(),
        "둘 다 chop": lambda w: ((w.reg_eth == 2) & (w.reg_btc == 2)).to_numpy(),
    }
    log("\n=== 게이트 비교 (동결 모델·동결 τ) ===")
    print(f"{'게이트':12s} | " + " | ".join(f"{w:>20s}" for w in ("VAL", "OOS", "HOLDOUT")))
    res, rows = {}, []
    for gname, gf in GATES.items():
        cells, r = [], {"gate": gname}
        for wn in ("VAL", "OOS", "HOLDOUT"):
            m = dsel & (D.split == wn).to_numpy()
            w = D[m]
            keep = (w.pred.to_numpy() > TAU) & gf(w)
            v = slotN(w[keep], NSLOT); nn, bp, pf = stat(v)
            r[f"{wn}_bp"] = round(bp, 2); r[f"{wn}_n"] = nn
            cells.append(f"{bp:+7.2f}bp n={nn:4d}")
        res[gname] = r; rows.append(r)
        print(f"{gname:12s} | " + " | ".join(f"{c:>20s}" for c in cells))

    ng = res["무게이트"]
    winner = None
    for g, r in res.items():
        if g == "무게이트": continue
        if r["VAL_bp"] > ng["VAL_bp"] and r["OOS_bp"] > ng["OOS_bp"]:
            if winner is None or r["VAL_bp"] > res[winner]["VAL_bp"]:
                winner = g
    if winner is None:
        log("\n⭐사전등록 판정: **양 창에서 무게이트를 이긴 레짐 게이트 없음 → 무게이트 동결**")
    else:
        log(f"\n⭐{winner}가 양 창 승리 → 시드별 개별 검정")
        gf = GATES[winner]
        for wn in ("VAL", "OOS"):
            m = dsel & (D.split == wn).to_numpy()
            w = D[m]; g = gf(w)
            gv, nv = [], []
            for ps in per_seed:
                p = ps[m]
                gv.append(stat(slotN(w[(p > TAU) & g], NSLOT))[1])
                nv.append(stat(slotN(w[p > TAU], NSLOT))[1])
            beat = sum(a > b for a, b in zip(gv, nv))
            log(f"  {wn:5s} 개별 승리 {beat}/5 " + "(" +
                ", ".join(f"{a:+.1f}vs{b:+.1f}" for a, b in zip(gv, nv)) + ")")
            if beat < 4:
                log(f"    → {wn}에서 4/5 미달, 채택 불가")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "gate_compare.csv", index=False)
    json.dump({"winner": winner, "results": res, "policy": {k: v for k, v in pol.items()
               if not isinstance(v, dict)}}, open(OUT / "b13_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
