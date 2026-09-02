#!/usr/bin/env python3
"""B4: 필터 정책 대조군 -- 모델이 무작위 필터를 이기는가 (2026-09-03).

B3에서 기계+트리거는 통과했다(모멘텀 뒤집기 3창 손실, 무작위봉/순환이동 백분위 100%·p=0.000).
그런데 기여 분해를 보면 기계 +17~21bp / 트리거 +5~7bp / **모델 필터 +19~25bp**로
가장 큰 몫이 아직 미검정이다. 여기서 검정한다.

  ⭐핵심 대조군: **무작위 필터** -- 같은 비율로 아무 팔이나 버린다.
     1슬롯에서는 버리는 것 자체가 슬롯을 비워 뒤의 거래를 열어주므로, 무작위로 버려도
     성과가 오를 수 있다. 모델이 그걸 넘어야 진짜다.
  + τ는 **VAL에서만** 고르고 OOS를 보고한다 (앞서 스윕을 보고 읽은 절차를 교정)
  + 시드별 부호 일치, 시간블록 군집 부트스트랩

모델: A+B+F (Tier0 23 + 팔·신호메타 4 + 레짐패널 136 = 162피쳐), HGB 5시드 평균.
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
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat, slot_sim  # noqa: E402
from research_eth_regime_s12k3_label_train_20260902 import GBM3_MODEL_PATH, load_frame  # noqa: E402

OUT = ROOT / "tmp/eth_entry_b4_filtercontrol_20260903"
TAUS = [0.0, 0.0002, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030]
B_RND = 200
RNG = np.random.default_rng(20260903)


def log(m): print(f"[b4] {m}", flush=True)


def main() -> int:
    d = pd.read_csv(ROOT / "tmp/eth_entry_b1_20260903/arm_rows.csv", parse_dates=["ts"])
    src = joblib.load(GBM3_MODEL_PATH); cols = src["feature_cols"]; med = src["feature_medians"]
    rf = load_frame()
    x = rf[["timestamp"] + cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med.get(c, 0.0))
    x = x.rename(columns={"timestamp": "ts"})
    dup = [c for c in cols if c in d.columns]
    x = x.rename(columns={c: c + "_r136" for c in dup})
    R136 = [(c + "_r136" if c in dup else c) for c in cols]
    d = d.merge(x, on="ts", how="left").dropna(subset=R136).reset_index(drop=True)

    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    FEATS = base + ["arm", "sig_id", "sig_dir", "atr"] + R136
    X = d[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (d.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    per_seed = {s: HistGradientBoostingRegressor(random_state=s, **HP)
                .fit(X[tr], d.loc[tr, "y"]).predict(X) for s in SEEDS}
    d["pred"] = np.mean([per_seed[s] for s in SEEDS], axis=0)
    log(f"행 {len(d):,} | TRAIN {int(tr.sum()):,} | 피쳐 {len(FEATS)} (A+B+F)")

    def pol(w, keep):
        v = slot_sim(w, keep, 1)
        _, m, _ = stat(v)
        return len(v), m

    # ---- τ 스윕: VAL에서만 고름 ----
    log("\n=== τ 스윕 (선택은 VAL만) ===")
    print(f"{'τ(bp)':>7s} {'유지비율':>8s} | {'VAL 1슬롯':>18s} | {'OOS 1슬롯':>18s} | {'HOLDOUT':>18s}")
    sweep = []
    for tau in TAUS:
        cells, row = [], {"tau_bp": tau * 1e4}
        kf = None
        for wn in ("VAL", "OOS", "HOLDOUT"):
            w = d[d.split == wn]
            k = (w.pred > tau).to_numpy()
            if wn == "VAL": kf = float(k.mean())
            nn, m = pol(w, k)
            row[f"{wn}_bp"] = round(m, 2); row[f"{wn}_n"] = nn
            cells.append(f"{m:+7.2f}bp (n{nn})")
        row["keep_val"] = round(kf, 3)
        sweep.append(row)
        print(f"{row['tau_bp']:7.1f} {kf:8.1%} | " + " | ".join(f"{c:>18s}" for c in cells))
    sw = pd.DataFrame(sweep)
    best = sw.loc[sw.VAL_bp.idxmax()]
    tau = best.tau_bp / 1e4
    log(f"\n⭐VAL 최적 τ = {best.tau_bp:.1f}bp (유지 {best.keep_val:.1%}) "
        f"→ OOS {best.OOS_bp:+.2f}bp (n{int(best.OOS_n)}) · HOLDOUT {best.HOLDOUT_bp:+.2f}bp")

    # ---- ⭐무작위 필터 대조군 ----
    log(f"\n=== ⭐무작위 필터 대조군 (같은 유지비율, B={B_RND}) ===")
    res = {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        w = d[d.split == wn]
        k = (w.pred > tau).to_numpy()
        frac = float(k.mean())
        _, real = pol(w, k)
        _, allm = pol(w, np.ones(len(w), bool))
        rr = []
        for _ in range(B_RND):
            rk = RNG.random(len(w)) < frac
            _, m = pol(w, rk); rr.append(m)
        rr = np.array(rr)
        res[wn] = {"real": real, "keep_all": allm, "rnd_mean": float(rr.mean()),
                   "rnd_lo": float(np.quantile(rr, .025)), "rnd_hi": float(np.quantile(rr, .975)),
                   "pct": float((rr < real).mean()), "p": float((rr >= real).mean())}
        log(f"  {wn:8s} 실제 {real:+7.2f} | 다잡기 {allm:+7.2f} | 무작위필터 {rr.mean():+7.2f} "
            f"[{np.quantile(rr,.025):+.2f},{np.quantile(rr,.975):+.2f}] → 백분위 {float((rr<real).mean()):.0%} p={float((rr>=real).mean()):.3f}")

    # ---- 시드별 ----
    log("\n=== 시드별 (같은 τ) ===")
    for wn in ("VAL", "OOS"):
        w = d[d.split == wn]; m = (d.split == wn).to_numpy()
        vals = []
        for s in SEEDS:
            _, v = pol(w, (per_seed[s][m] > tau))
            vals.append(v)
        log(f"  {wn:5s} " + ", ".join(f"{v:+.2f}" for v in vals) +
            f"  → 다잡기({res[wn]['keep_all']:+.2f}) 초과 {sum(v>res[wn]['keep_all'] for v in vals)}/5")

    # ---- 시간블록 부트스트랩 (필터 정책) ----
    log("\n=== 시간블록(일) 군집 부트스트랩, 필터 정책 ===")
    for wn in ("VAL", "OOS"):
        w = d[d.split == wn]
        k = (w.pred > tau).to_numpy()
        sub = w[k & w.filled.astype(bool)].sort_values("fill_i")
        v = slot_sim(w, k, 1)
        s2 = sub.iloc[:len(v)].copy(); s2["y2"] = v
        s2["day"] = (s2.fill_i // 288).astype(int)
        days = s2.day.unique()
        bs = []
        for _ in range(2000):
            pick = RNG.choice(days, size=len(days), replace=True)
            bs.append(np.concatenate([s2.loc[s2.day == dd, "y2"].to_numpy() for dd in pick]).mean() * 1e4)
        bs = np.array(bs)
        log(f"  {wn:5s} {res[wn]['real']:+7.2f}bp 95%CI [{np.quantile(bs,.025):+.2f},{np.quantile(bs,.975):+.2f}] "
            f"블록 {len(days)}일 · 다잡기({res[wn]['keep_all']:+.2f}) 초과확률 {float((bs>res[wn]['keep_all']).mean()):.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    sw.to_csv(OUT / "tau_sweep.csv", index=False)
    json.dump({"best_tau_bp": float(best.tau_bp), "controls": res, "features": len(FEATS)},
              open(OUT / "b4_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
