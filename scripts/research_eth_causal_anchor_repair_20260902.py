#!/usr/bin/env python3
"""앵커 미래참조의 복구 시도 -- 인과적으로 클라이맥스를 잡을 방법이 있는가 (2026-09-02).

인과 대안 3종(첫발동/후행최극단/지연확정)이 전부 실패했으므로, 그대로면 이 계열 전체의
경제성게이트가 무효다. 결론 내리기 전에 정당한 복구 두 가지를 시도한다.

  진단 1 -- 앵커는 클러스터의 몇 번째인가?
      앵커가 대개 **마지막** 트리거라면 "트리거 후 1봉 기다려 더 큰 게 안 오면 진입"이라는
      1봉 지연 규칙으로 거의 재현할 수 있다. 앵커가 중간이면 그 길은 막힌다.

  복구 R1 -- 절대 임계 강화(dedup 없음)
      클러스터 상대 비교 대신 **절대적으로 더 극단적인** 트리거만 쓴다. 완전히 인과적이다.
      |ret3_z| >= {3.0, 3.5, 4.0}, dem <= {0.05, 0.03} / >= {0.95, 0.97}.

  복구 R2 -- 1봉 지연 확정
      트리거 봉 i에서 바로 안 들어가고, i+1에 더 극단적인 동측 트리거가 없으면 i+1 종가 기준으로
      진입한다(= i+1이 결정봉). 클러스터 전체를 안 기다리므로 지연이 1봉뿐이다.
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

OUT_DIR = ROOT / "tmp/eth_causal_anchor_repair_20260902"
START = pd.Timestamp("2024-01-01")
SPEC = {"short_term_return_z": {"gap": 3, "sl": 3.0, "arm": 1.0, "trail": 0.1, "horizon": 12},
        "demarker_extreme": {"gap": 12, "sl": 2.0, "arm": 1.5, "trail": 0.1, "horizon": 8}}
TOP2 = ["short_term_return_z", "demarker_extreme"]
R1_THRESH = {"short_term_return_z": [2.5, 3.0, 3.5, 4.0],
             "demarker_extreme": [0.10, 0.05, 0.03, 0.02]}


def log(m): print(f"[repair] {m}", flush=True)


def evaluate(name, items, spec, ts, o, h, l, c, pos_of, atr_src, src_ts, hold_end, tag):
    rows = [{"pos": pos_of[src_ts[i]], "side": sd, "atr_pct": atr_src[ai]}
            for i, sd, ai in items if i < len(atr_src) and src_ts[i] in pos_of
            and np.isfinite(atr_src[ai]) and atr_src[ai] > 0]
    if not rows:
        return None
    f = pd.DataFrame(rows).sort_values("pos").reset_index(drop=True)
    out = {}
    for sgn, lb in ((1.0, "real"), (-1.0, "flip")):
        t = per_fire_outcomes(ts, o, h, l, c, f["pos"].to_numpy(np.int64),
                              np.where(f["side"] == "bottom", 1.0, -1.0) * sgn,
                              f["atr_pct"].to_numpy(float), spec["horizon"],
                              spec["sl"], spec["arm"], spec["trail"])
        t["signal"] = name; t["prio"] = 0
        t["decision_pos"] = [pos_of[x] for x in t["decision_ts"]]
        out[lb] = t
    return out


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
    hold_end = ts.max()
    W = (("VAL", VAL_START, OOS_START), ("OOS", OOS_START, HOLDOUT_START),
         ("HOLDOUT", HOLDOUT_START, hold_end + pd.Timedelta(minutes=5)))

    def trig(name, thr):
        if name == "short_term_return_z":
            return {"bottom": ret3_z <= -thr, "top": ret3_z >= thr, "ex": ret3_z}
        return {"bottom": dem <= thr, "top": dem >= 1.0 - thr, "ex": dem}

    # ---------- 진단 1: 앵커는 클러스터의 몇 번째인가 ----------
    log("=== 진단: 앵커의 클러스터 내 위치 ===")
    for name, spec in SPEC.items():
        gap = spec["gap"]; r = trig(name, R1_THRESH[name][0])
        pos_stats = []
        for side in ("bottom", "top"):
            mneg = side == "bottom"
            idx = np.flatnonzero(np.nan_to_num(r[side].astype(float), nan=0.0).astype(bool))
            idx = idx[(idx < len(src) - spec["horizon"] - gap - 1) &
                      (src_ts[idx].to_numpy() >= np.datetime64(START))]
            exm = {int(i): float(r["ex"][i]) for i in idx}
            cur = [int(idx[0])]
            for i in idx[1:]:
                if int(i) - cur[-1] > gap:
                    a = min(cur, key=lambda j: exm[j]) if mneg else max(cur, key=lambda j: exm[j])
                    pos_stats.append((cur.index(a), len(cur))); cur = [int(i)]
                else:
                    cur.append(int(i))
        ps = pd.DataFrame(pos_stats, columns=["rank", "size"])
        single = (ps["size"] == 1).mean()
        multi = ps[ps["size"] > 1]
        log(f"  {name}: 클러스터 {len(ps):,}개, 단일봉 {single:.1%} | "
            f"다봉 클러스터에서 앵커가 첫봉 {(multi['rank']==0).mean():.1%} / "
            f"마지막봉 {(multi['rank']==multi['size']-1).mean():.1%} / "
            f"중간 {((multi['rank']>0)&(multi['rank']<multi['size']-1)).mean():.1%} "
            f"(평균 크기 {multi['size'].mean():.1f}봉)")

    # ---------- R1: 절대 임계 강화, dedup 없음 ----------
    log("\n=== R1: 절대 임계 강화 (dedup 없음, 완전 인과) -- top1 = str_z 단독 ===")
    res = []
    for name, spec in SPEC.items():
        for thr in R1_THRESH[name]:
            r = trig(name, thr); items = []
            for side in ("bottom", "top"):
                idx = np.flatnonzero(np.nan_to_num(r[side].astype(float), nan=0.0).astype(bool))
                idx = idx[(idx < len(src) - spec["horizon"] - 1) &
                          (src_ts[idx].to_numpy() >= np.datetime64(START))]
                items += [(int(i), side, int(i)) for i in idx]
            ev = evaluate(name, items, spec, ts, o, h, l, c, pos_of, atr_src, src_ts, hold_end, "R1")
            if ev is None: continue
            row = {"signal": name, "thr": thr, "n_raw": len(items)}
            okall = True
            for wn, lo, hi in W:
                rr = ev["real"]; ff = ev["flip"]
                a = summarize(sequential_portfolio(rr[(rr.decision_ts >= lo) & (rr.decision_ts < hi)], {name: 0}), "x")
                b = summarize(sequential_portfolio(ff[(ff.decision_ts >= lo) & (ff.decision_ts < hi)], {name: 0}), "x")
                row[f"{wn}_mean"] = a.get("mean_bp"); row[f"{wn}_pf"] = a.get("pf"); row[f"{wn}_n"] = a.get("n")
                okall &= (a.get("total_bp", -1) or -1) > max(b.get("total_bp", 0) or 0, 0)
            row["flip3창"] = "O" if okall else "X"
            res.append(row)
    rdf = pd.DataFrame(res)
    print(rdf.round(2).to_string(index=False))
    OUT_DIR.mkdir(parents=True, exist_ok=True); rdf.to_csv(OUT_DIR / "r1_absolute_threshold.csv", index=False)

    # ---------- R2: 1봉 지연 확정 ----------
    log("\n=== R2: 1봉 지연 확정 (i+1에 더 극단 동측 트리거 없으면 i+1을 결정봉) ===")
    res2 = []
    for name, spec in SPEC.items():
        r = trig(name, R1_THRESH[name][0]); items = []
        for side in ("bottom", "top"):
            mneg = side == "bottom"
            m = np.nan_to_num(r[side].astype(float), nan=0.0).astype(bool)
            idx = np.flatnonzero(m)
            idx = idx[(idx < len(src) - spec["horizon"] - 2) &
                      (src_ts[idx].to_numpy() >= np.datetime64(START))]
            ex = r["ex"]
            sel = set(int(i) for i in idx)
            for i in idx:
                nxt = int(i) + 1
                better = (nxt in sel) and ((ex[nxt] < ex[i]) if mneg else (ex[nxt] > ex[i]))
                if not better:
                    items.append((int(i) + 1, side, int(i)))   # 결정봉 i+1, atr는 앵커 i
        ev = evaluate(name, items, spec, ts, o, h, l, c, pos_of, atr_src, src_ts, hold_end, "R2")
        row = {"signal": name, "n": len(items)}
        okall = True
        for wn, lo, hi in W:
            rr = ev["real"]; ff = ev["flip"]
            a = summarize(sequential_portfolio(rr[(rr.decision_ts >= lo) & (rr.decision_ts < hi)], {name: 0}), "x")
            b = summarize(sequential_portfolio(ff[(ff.decision_ts >= lo) & (ff.decision_ts < hi)], {name: 0}), "x")
            row[f"{wn}_mean"] = a.get("mean_bp"); row[f"{wn}_pf"] = a.get("pf"); row[f"{wn}_n"] = a.get("n")
            okall &= (a.get("total_bp", -1) or -1) > max(b.get("total_bp", 0) or 0, 0)
        row["flip3창"] = "O" if okall else "X"
        res2.append(row)
    r2df = pd.DataFrame(res2)
    print(r2df.round(2).to_string(index=False))
    r2df.to_csv(OUT_DIR / "r2_one_bar_delay.csv", index=False)
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
