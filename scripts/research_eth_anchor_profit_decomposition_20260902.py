#!/usr/bin/env python3
"""앵커 수익의 구조 분해 -- 인과적으로 도달 가능한 자리에 수익이 있는가 (2026-09-02).

인과 대안 3종 + 복구 2종 + 학습형 클라이맥스 예측기까지 전부 실패했다. 마지막으로,
**수익이 클러스터의 어느 자리에 있는지** 직접 분해한다. 이게 축의 개폐를 결정한다:

  - 단일봉 클러스터의 앵커 = 고립 트리거. **직전 GAP봉에 트리거가 없다**는 건 인과적으로
    알 수 있다(다만 이후 GAP봉에 안 온다는 보장은 없다 -> 그래도 절반은 인과).
  - 다봉 클러스터에서 **첫 봉**이 앵커 = 인과적으로 잡을 수 있다(변형 B가 정확히 이것).
  - 다봉 클러스터에서 **중간/마지막** 봉이 앵커 = 그 봉에서는 알 수 없다. 구조적으로 도달 불가.

수익이 셋째 칸에 몰려 있으면 이 축은 닫힌다. 첫째/둘째 칸에 있으면 인과 규칙이 존재한다.
포지션 중첩 효과를 빼기 위해 **1슬롯 순차가 아닌 건별(per-fire) 수익**으로 본다.
"""
from __future__ import annotations

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

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, OOS_START, VAL_START)
from research_eth_evidence_signal_ensemble_pnl_20260902 import per_fire_outcomes  # noqa: E402
from research_eth_causal_climax_predictor_20260902 import clusters_of  # noqa: E402

OUT_DIR = ROOT / "tmp/eth_anchor_profit_decomposition_20260902"
START = pd.Timestamp("2024-01-01")
SPEC = {"short_term_return_z": {"gap": 3, "sl": 3.0, "arm": 1.0, "trail": 0.1, "horizon": 12},
        "demarker_extreme": {"gap": 12, "sl": 2.0, "arm": 1.5, "trail": 0.1, "horizon": 8}}


def log(m): print(f"[decomp] {m}", flush=True)


def main() -> int:
    from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame

    src = load_klines(); ind = build_indicator_frame(src)
    ret3_z = ind["ret3_z"].to_numpy(); dem = compute_demarker(src["high"], src["low"]).to_numpy()
    atr_src = ind["atr_pct"].to_numpy(); src_ts = pd.DatetimeIndex(src["timestamp"])
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = kl["timestamp"]; o, h, l, c = (kl[k].to_numpy() for k in ("open", "high", "low", "close"))
    pos_of = {t: i for i, t in enumerate(pd.DatetimeIndex(ts))}
    raw = {"short_term_return_z": {"bottom": ret3_z <= -2.5, "top": ret3_z >= 2.5, "ex": ret3_z},
           "demarker_extreme": {"bottom": dem <= 0.10, "top": dem >= 0.90, "ex": dem}}

    out = []
    for name, spec in SPEC.items():
        gap = spec["gap"]; r = raw[name]; recs = []
        for side in ("bottom", "top"):
            mneg = side == "bottom"
            idx = np.flatnonzero(np.nan_to_num(r[side].astype(float), nan=0.0).astype(bool))
            idx = idx[(idx < len(src) - spec["horizon"] - gap - 1) &
                      (src_ts[idx].to_numpy() >= np.datetime64(START))]
            ex = r["ex"]
            for cl in clusters_of(idx, gap):
                vals = [ex[j] for j in cl]
                k = int(np.argmin(vals)) if mneg else int(np.argmax(vals))
                a = cl[k]
                if len(cl) == 1:
                    bucket = "①단일봉(고립)"
                elif k == 0:
                    bucket = "②다봉-첫봉"
                elif k == len(cl) - 1:
                    bucket = "③다봉-마지막"
                else:
                    bucket = "④다봉-중간"
                recs.append({"pos_src": a, "side": side, "bucket": bucket, "csize": len(cl)})
        d = pd.DataFrame(recs)
        d["timestamp"] = src_ts[d["pos_src"].to_numpy()]
        d["atr"] = atr_src[d["pos_src"].to_numpy()]
        d = d[d["timestamp"].isin(pos_of) & np.isfinite(d["atr"]) & (d["atr"] > 0)].copy()
        d["pos"] = [pos_of[t] for t in d["timestamp"]]
        d = d.sort_values("pos").reset_index(drop=True)

        t = per_fire_outcomes(ts, o, h, l, c, d["pos"].to_numpy(np.int64),
                              np.where(d["side"] == "bottom", 1.0, -1.0),
                              d["atr"].to_numpy(float), spec["horizon"],
                              spec["sl"], spec["arm"], spec["trail"])
        t = t.merge(d[["timestamp", "bucket", "csize"]].rename(columns={"timestamp": "decision_ts"}),
                    on="decision_ts", how="left")
        for wn, lo, hi in (("VAL", VAL_START, OOS_START), ("OOS", OOS_START, HOLDOUT_START),
                           ("HOLDOUT", HOLDOUT_START, ts.max() + pd.Timedelta(minutes=5))):
            w = t[(t.decision_ts >= lo) & (t.decision_ts < hi)]
            for b, g in w.groupby("bucket"):
                v = g["trade_return"].to_numpy()
                out.append({"signal": name, "window": wn, "bucket": b, "n": len(v),
                            "mean_bp": round(float(v.mean() * 1e4), 2),
                            "total_bp": round(float(v.sum() * 1e4), 1),
                            "win": round(float((v > 0).mean()), 3)})
    res = pd.DataFrame(out)
    OUT_DIR.mkdir(parents=True, exist_ok=True); res.to_csv(OUT_DIR / "decomposition.csv", index=False)

    for name in SPEC:
        log(f"\n=== {name} -- 앵커 자리별 건별수익 (1슬롯 아님) ===")
        r = res[res.signal == name]
        print(r.pivot_table(index="bucket", columns="window",
                            values=["n", "mean_bp", "total_bp"])
              .reindex(columns=["VAL", "OOS", "HOLDOUT"], level=1).round(2).to_string())

    log("\n=== 인과 도달 가능성 요약 (총수익 기준, 3창 합) ===")
    agg = res.groupby(["signal", "bucket"])["total_bp"].sum().unstack()
    agg["인과도달"] = ["△절반(직전GAP만 확인가능)" if b.startswith("①") else
                    "O 가능(변형B)" if b.startswith("②") else "X 불가"
                    for b in agg.columns[:0]] if False else None
    print(agg.round(0).to_string())
    tot = res.groupby("bucket")["total_bp"].sum()
    share = (tot / tot.abs().sum() * 100).round(1)
    print("\n자리별 총수익 비중(%):"); print(share.to_string())
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
