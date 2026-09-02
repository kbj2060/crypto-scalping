#!/usr/bin/env python3
"""B9: 모델 아키텍처 스윕 (2026-09-03).

타깃 분포 실측: 평균 +20.92bp · 중앙값 +24.07 · **왜도 −0.84(음수)** · 첨도 +14.54 ·
p1 −154.3bp · 양수 84.9%. 트레일링스톱의 전형 -- 대부분 작게 이기고 가끔 크게 잃는다.
squared error는 소수의 큰 손실을 맞히는 데 용량을 쓰는데, 슬롯 배분에 필요한 건 **순위**다.

후보
  A  HGB squared_error (현행, 기준)
  B  HGB absolute_error        -- 중앙값 지향, 왼쪽 꼬리에 덜 끌림
  C  HGB quantile(α=0.5, 0.6)  -- 하방 이상치 제거, 순위 직결
  D  winsorize(p1/p99) + squared_error -- 가장 단순한 꼬리 대응
  (TabFM 제외: 가중치가 non-commercial 라이선스라 라이브 승격 불가. 성능도 1승1패로 미결)

⚠️방법론 보정: 손실이 다르면 예측 **스케일**이 달라 고정 τ=40bp는 비교 불가다.
   현행 모델이 τ=40bp에서 남기는 **유지비율**을 구해, 모든 후보에 같은 비율을 적용한다.

⭐사전등록: 대안이 현행을 **VAL·OOS 양 창에서** 못 이기면 현행(squared_error)을 동결한다.
   승자는 무작위 필터 대조군·5시드·시간블록 부트스트랩을 다시 통과해야 한다.
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
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
OUT = ROOT / "tmp/eth_entry_b9_arch_20260903"
DEPTH, WAIT, TAU0, NSLOT = 3.0, 6, 0.0040, 4
B_RND = 150
RNG = np.random.default_rng(20260903)


def log(m): print(f"[b9] {m}", flush=True)


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

    ytr = y[tr]
    lo, hi = np.quantile(ytr, [0.01, 0.99])
    ARCH = {
        "A squared (현행)": dict(loss="squared_error", wins=False),
        "B absolute":       dict(loss="absolute_error", wins=False),
        "C quantile α=0.5": dict(loss="quantile", quantile=0.5, wins=False),
        "C quantile α=0.6": dict(loss="quantile", quantile=0.6, wins=False),
        "D winsor+squared": dict(loss="squared_error", wins=True),
    }

    # 현행 기준 유지비율 산출
    hp0 = {k: v for k, v in HP.items()}
    p0 = np.mean([HistGradientBoostingRegressor(random_state=s, loss="squared_error", **hp0)
                  .fit(X[tr], ytr).predict(X) for s in SEEDS], axis=0)
    fracs = {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == wn).to_numpy()
        fracs[wn] = float((p0[m] > TAU0).mean())
    log(f"현행 τ={TAU0*1e4:.0f}bp 유지비율 → " + " ".join(f"{k} {v:.1%}" for k, v in fracs.items()))

    res, preds = [], {}
    for name, spec in ARCH.items():
        kw = {k: v for k, v in spec.items() if k != "wins"}
        tgt = np.clip(ytr, lo, hi) if spec["wins"] else ytr
        ps = {}
        for s in SEEDS:
            mdl = HistGradientBoostingRegressor(random_state=s, **kw, **hp0)
            ps[s] = mdl.fit(X[tr], tgt).predict(X)
        p = np.mean([ps[s] for s in SEEDS], axis=0)
        preds[name] = (p, ps)
        row = {"arch": name}
        cells = []
        for wn in ("VAL", "OOS", "HOLDOUT"):
            m = dsel & (D.split == wn).to_numpy()
            w = D[m]; pv = p[m]
            thr = np.quantile(pv, 1 - fracs[wn])       # ⭐동일 유지비율로 맞춤
            v = slotN(w[pv > thr], NSLOT); nn, mm, _ = stat(v)
            row[f"{wn}_bp"] = round(mm, 2); row[f"{wn}_n"] = nn
            cells.append(f"{mm:+7.2f}bp n={nn:4d}")
        res.append(row)
        log(f"{name:20s} | " + " | ".join(f"{c:>20s}" for c in cells))

    r = pd.DataFrame(res)
    cur = r[r.arch.str.startswith("A ")].iloc[0]
    alt = r[~r.arch.str.startswith("A ")]
    winner = None
    for _, a in alt.iterrows():
        if a.VAL_bp > cur.VAL_bp and a.OOS_bp > cur.OOS_bp:
            if winner is None or a.VAL_bp > winner.VAL_bp:
                winner = a
    log(f"\n⭐사전등록 판정: " + (f"**{winner.arch}가 양 창에서 현행을 이김 → 후보 채택**"
                              if winner is not None else "**양 창에서 현행을 이긴 대안 없음 → squared_error 동결**"))

    chosen = winner.arch if winner is not None else cur.arch
    p, ps = preds[chosen]
    log(f"\n=== 대조군 재검정 ({chosen}) ===")
    ctrl = {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == wn).to_numpy()
        w = D[m]; pv = p[m]
        thr = np.quantile(pv, 1 - fracs[wn])
        keep = pv > thr
        real = stat(slotN(w[keep], NSLOT))[1]
        allm = stat(slotN(w, NSLOT))[1]
        rr = np.array([stat(slotN(w[RNG.random(len(w)) < fracs[wn]], NSLOT))[1] for _ in range(B_RND)])
        sv = [stat(slotN(w[ps[s][m] > np.quantile(ps[s][m], 1 - fracs[wn])], NSLOT))[1] for s in SEEDS]
        sub = w[keep].sort_values("fi"); v = slotN(w[keep], NSLOT)
        s2 = sub.iloc[:len(v)].copy(); s2["y2"] = v; s2["day"] = (s2.fi // 288).astype(int)
        days = s2.day.unique()
        bs = np.array([np.concatenate([s2.loc[s2.day == dd, "y2"].to_numpy()
                       for dd in RNG.choice(days, len(days), replace=True)]).mean() * 1e4
                       for _ in range(2000)])
        ctrl[wn] = {"real": real, "keep_all": allm, "rnd": float(rr.mean()),
                    "p": float((rr >= real).mean()), "seeds": int(sum(x > allm for x in sv)),
                    "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
        log(f"  {wn:8s} 실제 {real:+6.2f} | 무필터 {allm:+6.2f} | 무작위 {rr.mean():+6.2f} "
            f"p={float((rr>=real).mean()):.3f} | 시드 {sum(x>allm for x in sv)}/5 | "
            f"CI [{np.quantile(bs,.025):+.2f},{np.quantile(bs,.975):+.2f}]")

    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "arch_sweep.csv", index=False)
    json.dump({"chosen": chosen, "keep_fracs": fracs, "controls": ctrl,
               "tabfm_excluded": "non-commercial pretrained weights license blocks live use"},
              open(OUT / "b9_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
