#!/usr/bin/env python3
"""발동 앵커 선택의 라이브 파리티 감사 v2 -- **raw 트리거**에서 출발 (2026-09-02).

v1은 이미 앵커 dedup된 CSV에 인과 규칙을 덧씌워 무변화(100%)가 나왔다. 그건 검정이 아니다.
이번엔 raw 트리거(str_z: |ret3_z|>=2.5 / demarker: dem<=0.10 또는 >=0.90)에서 시작해 두 규칙을
갈라 비교한다.

  연구(앵커선택) -- cluster_dedup: GAP 안 연속발동을 묶고 그 클러스터의 **최극단 봉**을 남긴다.
                    ⚠️어느 봉이 최극단인지는 클러스터가 끝나야 알 수 있으므로 라이브 불가.
  인과(첫발동)   -- 클러스터의 **첫 발동**을 취하고 GAP 봉 재무장 금지. 완전히 인과적.

같은 청산·비용·1슬롯 순차로 VAL/OOS/HOLDOUT 3창 전부 재평가한다.
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
from research_eth_evidence_signal_ensemble_pnl_20260902 import (  # noqa: E402
    per_fire_outcomes, sequential_portfolio, summarize)
from research_eth_kalman_demarker_gridscreen_20260831 import cluster_dedup  # noqa: E402

OUT_DIR = ROOT / "tmp/eth_causal_anchor_parity_v2_20260902"
START = pd.Timestamp("2024-01-01")
SPEC = {   # 트리거 규칙 + GAP + 동결 청산설정
    "short_term_return_z": {"gap": 3, "sl": 3.0, "arm": 1.0, "trail": 0.1, "horizon": 12},
    "demarker_extreme":    {"gap": 12, "sl": 2.0, "arm": 1.5, "trail": 0.1, "horizon": 8},
}
TOP2 = ["short_term_return_z", "demarker_extreme"]


def log(m: str) -> None:
    print(f"[parity_v2] {m}", flush=True)


def causal_first(idx: np.ndarray, gap: int) -> np.ndarray:
    keep, last = [], -10**9
    for i in np.sort(idx):
        if i - last > gap:
            keep.append(i); last = i
    return np.array(keep, dtype=np.int64)


def main() -> int:
    from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame

    src = load_klines()
    ind = build_indicator_frame(src)
    ret3_z = ind["ret3_z"].to_numpy()
    dem = compute_demarker(src["high"], src["low"]).to_numpy()
    src_ts = pd.DatetimeIndex(src["timestamp"])
    log(f"raw 프레임 {len(src):,}봉")

    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = kl["timestamp"]
    o, h, l, c = (kl[k].to_numpy() for k in ("open", "high", "low", "close"))
    pos_of = {t: i for i, t in enumerate(pd.DatetimeIndex(ts))}
    hold_end = ts.max()
    atr_src = ind["atr_pct"].to_numpy()

    raw = {
        "short_term_return_z": {"bottom": ret3_z <= -2.5, "top": ret3_z >= 2.5, "ex": ret3_z},
        "demarker_extreme":    {"bottom": dem <= 0.10, "top": dem >= 0.90, "ex": dem},
    }

    tabs = {}
    for name, spec in SPEC.items():
        r, gap = raw[name], SPEC[name]["gap"]
        sets = {"연구(앵커선택)": [], "인과(첫발동)": []}
        for side in ("bottom", "top"):
            m = np.nan_to_num(r[side].astype(float), nan=0.0).astype(bool)
            idx = np.flatnonzero(m)
            idx = idx[(idx < len(src) - spec["horizon"]) &
                      (src_ts[idx].to_numpy() >= np.datetime64(START))]
            a = cluster_dedup(idx, r["ex"][idx], most_negative=(side == "bottom"), gap=gap)
            b = causal_first(idx, gap)
            sets["연구(앵커선택)"].append((side, a))
            sets["인과(첫발동)"].append((side, b))
            log(f"{name}/{side}: raw {len(idx):,} -> 앵커 {len(a):,} / 인과 {len(b):,} "
                f"(겹침 {len(np.intersect1d(a,b)):,} = 앵커의 {len(np.intersect1d(a,b))/max(len(a),1):.1%})")

        for vn, parts in sets.items():
            rows = []
            for side, idx in parts:
                keep = [(int(i), side) for i in idx if src_ts[i] in pos_of and
                        np.isfinite(atr_src[i]) and atr_src[i] > 0]
                for i, sd in keep:
                    rows.append({"pos": pos_of[src_ts[i]], "side": sd, "atr_pct": atr_src[i]})
            f = pd.DataFrame(rows).sort_values("pos").reset_index(drop=True)
            dec = f["pos"].to_numpy(np.int64)
            sc = np.where(f["side"].to_numpy() == "bottom", 1.0, -1.0)
            a_ = f["atr_pct"].to_numpy(float)
            for sgn, lb in ((1.0, "real"), (-1.0, "flip")):
                t = per_fire_outcomes(ts, o, h, l, c, dec, sc * sgn, a_, spec["horizon"],
                                      spec["sl"], spec["arm"], spec["trail"])
                t["signal"] = name
                t["decision_pos"] = [pos_of[x] for x in t["decision_ts"]]
                tabs[(vn, name, lb)] = t

    prio = {n: i for i, n in enumerate(TOP2)}
    rows = []
    for vn in ("연구(앵커선택)", "인과(첫발동)"):
        for lb in ("real", "flip"):
            allc = pd.concat([tabs[(vn, n, lb)] for n in TOP2], ignore_index=True)
            allc["prio"] = allc["signal"].map(prio)
            for wn, (lo, hi) in (("VAL", (VAL_START, OOS_START)), ("OOS", (OOS_START, HOLDOUT_START)),
                                 ("HOLDOUT", (HOLDOUT_START, hold_end + pd.Timedelta(minutes=5)))):
                w = allc[(allc.decision_ts >= lo) & (allc.decision_ts < hi)]
                for k in (1, 2):
                    s = summarize(sequential_portfolio(w[w["signal"].isin(TOP2[:k])], prio), f"top{k}")
                    s.update({"window": wn, "kind": lb, "variant": vn}); rows.append(s)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "anchor_parity_v2.csv", index=False)

    for arm in ("top1", "top2"):
        log(f"\n=== {arm} (real) 3창 ===")
        r = df[(df.kind == "real") & (df.arm == arm)]
        print(r.pivot_table(index="variant", columns="window",
                            values=["n", "mean_bp", "pf", "total_bp"])
              .reindex(columns=["VAL", "OOS", "HOLDOUT"], level=1).round(2).to_string())

    log("\n=== 인과본 방향뒤집기 ===")
    cf = df[df.variant == "인과(첫발동)"].pivot_table(index="arm", columns=["window", "kind"], values="total_bp")
    for arm in cf.index:
        ok = all(cf.loc[arm, (w, "real")] > max(cf.loc[arm, (w, "flip")], 0) for w in ("VAL", "OOS", "HOLDOUT"))
        print(f"  {arm}: {'O 3창 통과' if ok else 'X'} " + " ".join(
            f"{w} {cf.loc[arm,(w,'real')]:+.0f}/{cf.loc[arm,(w,'flip')]:+.0f}" for w in ("VAL", "OOS", "HOLDOUT")))
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
