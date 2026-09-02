#!/usr/bin/env python3
"""B7: 확장 격자에서 τ 상향 + 대조군 재검정 (2026-09-03).

B6에서 체결이 11,311 -> 138,005로 늘어 표본 문제가 풀렸다(깊이 4종 x 대기 2종).
VAL 최고가 τ=30bp(격자 끝)였으므로 위로 넓히고, 대조군을 이 격자에서 다시 통과시킨다.

  ① τ 상향 {30,40,50,75,100}bp -- 봉우리가 격자 안에 있는가
  ② ⭐무작위 필터 대조군 -- 같은 유지비율로 아무거나 남기기 (슬롯 비우기 효과 분리)
  ③ 무작위 5시드 부호 일치
  ④ 시간블록(일) 군집 부트스트랩
  ⑤ DSR / PBO -- 훑은 조합 전체를 trial 수로

B6의 fills.csv를 재사용한다(재시뮬 불필요). 모델은 5시드 재학습.
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

from core.selection_stats import deflated_sharpe_ratio, pbo_cscv, sharpe  # noqa: E402
from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN, DEPTHS, WAITS  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
OUT = ROOT / "tmp/eth_entry_b7_20260903"
TAUS = [0.0, 0.0010, 0.0020, 0.0030, 0.0040, 0.0050, 0.0075, 0.0100]
SLOTS = [2, 4, 6]
B_RND = 150
RNG = np.random.default_rng(20260903)


def log(m): print(f"[b7] {m}", flush=True)


def main() -> int:
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    R136 = [c for c in D.columns if c.endswith("_r136")] + \
           [c for c in D.columns if c not in base + ["arm", "sig_id", "atr_pct", "depth", "y",
            "split", "timestamp", "i", "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"]
            and not c.endswith("_r136")]
    R136 = list(dict.fromkeys([c for c in R136 if D[c].dtype.kind in "fiub"]))
    FEATS = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R136))
    X = D[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (D.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    per_seed = {}
    for s in SEEDS:
        per_seed[s] = HistGradientBoostingRegressor(random_state=s, **HP).fit(X[tr], D.loc[tr, "y"]).predict(X)
        log(f"  시드 {s} 학습 완료")
    D["pred"] = np.mean([per_seed[s] for s in SEEDS], axis=0)
    log(f"행 {len(D):,} · TRAIN {int(tr.sum()):,} · 피쳐 {len(FEATS)}")

    # ---- ① τ 상향 격자 ----
    rows = []
    for depth in DEPTHS:
        for wait in WAITS:
            sel = (D.depth == depth) & (D.btf <= wait)
            for tau in TAUS:
                for ns in SLOTS:
                    r = {"depth": depth, "wait": wait, "tau_bp": tau * 1e4, "slots": ns}
                    for wn in ("VAL", "OOS", "HOLDOUT"):
                        w = D[sel & (D.split == wn) & (D.pred > tau)]
                        v = slotN(w, ns); nn, m, _ = stat(v)
                        r[f"{wn}_bp"] = round(m, 2); r[f"{wn}_n"] = nn
                    rows.append(r)
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True); R.to_csv(OUT / "grid_extended.csv", index=False)
    ok = R[(R.VAL_n >= 150) & (R.OOS_n >= 120)]
    log(f"\n=== τ 상향 격자, 표본조건 충족 {len(ok)}/{len(R)} · VAL 상위 10 ===")
    pd.set_option("display.width", 220)
    print(ok.sort_values("VAL_bp", ascending=False).head(10)[
        ["depth", "wait", "tau_bp", "slots", "VAL_bp", "VAL_n", "OOS_bp", "OOS_n",
         "HOLDOUT_bp", "HOLDOUT_n"]].to_string(index=False))
    log("\n=== τ별 최고 (봉우리가 격자 안인가) ===")
    for tau in TAUS:
        t = ok[ok.tau_bp == tau * 1e4]
        if not len(t): 
            log(f"  τ={tau*1e4:6.1f}bp  표본조건 충족 0"); continue
        b = t.loc[t.VAL_bp.idxmax()]
        log(f"  τ={tau*1e4:6.1f}bp  VAL {b.VAL_bp:+6.2f}(n{int(b.VAL_n)}) "
            f"OOS {b.OOS_bp:+6.2f}(n{int(b.OOS_n)}) HOLD {b.HOLDOUT_bp:+6.2f}(n{int(b.HOLDOUT_n)}) "
            f"@ d{b.depth}/w{int(b.wait)}/{int(b.slots)}슬롯")

    best = ok.loc[ok.VAL_bp.idxmax()]
    dsel = (D.depth == best.depth) & (D.btf <= best.wait)
    tau = best.tau_bp / 1e4; ns = int(best.slots)
    log(f"\n⭐VAL 최적: d{best.depth}/w{int(best.wait)}/τ{best.tau_bp:.0f}bp/{ns}슬롯")

    # ---- ② 무작위 필터 대조군 ----
    log(f"\n=== ② 무작위 필터 대조군 (같은 유지비율, B={B_RND}) ===")
    ctrl = {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        w = D[dsel & (D.split == wn)]
        k = (w.pred > tau).to_numpy(); frac = float(k.mean())
        real = stat(slotN(w[k], ns))[1]
        allm = stat(slotN(w, ns))[1]
        rr = np.array([stat(slotN(w[RNG.random(len(w)) < frac], ns))[1] for _ in range(B_RND)])
        ctrl[wn] = {"real": real, "keep_all": allm, "rnd": float(rr.mean()),
                    "lo": float(np.quantile(rr, .025)), "hi": float(np.quantile(rr, .975)),
                    "p": float((rr >= real).mean())}
        log(f"  {wn:8s} 실제 {real:+6.2f} | 무필터 {allm:+6.2f} | 무작위필터 {rr.mean():+6.2f} "
            f"[{np.quantile(rr,.025):+.2f},{np.quantile(rr,.975):+.2f}] → p={float((rr>=real).mean()):.3f}")

    # ---- ③ 시드 ----
    log("\n=== ③ 시드별 ===")
    for wn in ("VAL", "OOS", "HOLDOUT"):
        w = D[dsel & (D.split == wn)]; msk = (dsel & (D.split == wn)).to_numpy()
        vals = [stat(slotN(w[per_seed[s][msk] > tau], ns))[1] for s in SEEDS]
        log(f"  {wn:8s} " + ", ".join(f"{v:+.2f}" for v in vals) +
            f" → 무필터({ctrl[wn]['keep_all']:+.2f}) 초과 {sum(v>ctrl[wn]['keep_all'] for v in vals)}/5")

    # ---- ④ 시간블록 부트스트랩 ----
    log("\n=== ④ 시간블록(일) 군집 부트스트랩 ===")
    for wn in ("VAL", "OOS", "HOLDOUT"):
        w = D[dsel & (D.split == wn) & (D.pred > tau)].sort_values("fi")
        v = slotN(w, ns)
        s2 = w.iloc[:len(v)].copy(); s2["y2"] = v; s2["day"] = (s2.fi // 288).astype(int)
        days = s2.day.unique()
        bs = np.array([np.concatenate([s2.loc[s2.day == dd, "y2"].to_numpy()
                       for dd in RNG.choice(days, len(days), replace=True)]).mean() * 1e4
                       for _ in range(2000)])
        log(f"  {wn:8s} {ctrl[wn]['real']:+6.2f}bp 95%CI [{np.quantile(bs,.025):+.2f},{np.quantile(bs,.975):+.2f}] "
            f"블록 {len(days)}일 · 무필터 초과확률 {float((bs>ctrl[wn]['keep_all']).mean()):.3f}")

    # ---- ⑤ DSR / PBO ----
    log(f"\n=== ⑤ DSR / PBO (trial 수 = 격자 {len(R)}조합) ===")
    days_all = pd.date_range(D[D.split == "VAL"].timestamp.min().floor("D"),
                             D[D.split == "OOS"].timestamp.max().floor("D"), freq="D")
    cols, names = [], []
    for _, r in R.iterrows():
        s2 = (D.depth == r.depth) & (D.btf <= r.wait) & (D.split.isin(["VAL", "OOS"])) & (D.pred > r.tau_bp / 1e4)
        w = D[s2].sort_values("fi")
        v = slotN(w, int(r.slots))
        if len(v) < 40: continue
        t = w.iloc[:len(v)].copy(); t["y2"] = v
        ser = t.groupby(t.timestamp.dt.floor("D"))["y2"].sum().reindex(days_all, fill_value=0.0)
        cols.append(ser.to_numpy()); names.append(f"d{r.depth}_w{int(r.wait)}_t{int(r.tau_bp)}_s{int(r.slots)}")
    M = np.column_stack(cols)
    sh = np.array([sharpe(M[:, i]) for i in range(M.shape[1])])
    bi = int(np.argmax(sh))
    d1 = deflated_sharpe_ratio(M[:, bi], sh)
    log(f"  행렬 {M.shape[0]}일 x {M.shape[1]}조합 | 최고 {names[bi]} 일Sharpe {sh[bi]:.4f}")
    log(f"  DSR: " + ", ".join(f"{k}={round(float(v),4) if isinstance(v,(int,float)) else v}" for k, v in d1.items()))
    for nsp in (10, 6):
        try:
            p = pbo_cscv(M, n_splits=nsp)
            log(f"  PBO n_splits={nsp}: " + ", ".join(f"{k}={round(float(v),4)}" for k, v in p.items()
                                                      if isinstance(v, (int, float))))
        except Exception as e:
            log(f"  PBO n_splits={nsp}: 실패 {type(e).__name__}")

    json.dump({"best": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                        for k, v in best.items()}, "controls": ctrl,
               "n_trials": int(len(R)), "dsr": {k: (float(v) if isinstance(v, (int, float)) else v)
                                                for k, v in d1.items()}},
              open(OUT / "b7_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
