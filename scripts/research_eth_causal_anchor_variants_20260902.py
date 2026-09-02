#!/usr/bin/env python3
"""앵커 선택의 인과 대안 3종 비교 -- 엣지가 정말 미래참조에서만 나오는가 (2026-09-02).

v2에서 "첫 발동" 인과 규칙이 엣지를 전멸시켰다(top2 3창 −5.73/−2.68/−3.98, 방향뒤집기 3창 실패).
하지만 "첫 발동"은 가능한 인과 규칙 중 하나일 뿐이라 그것만으로 단정할 수 없다. 앵커 선택의
본질은 "클러스터의 최극단 봉"이고, 최극단이 곧 반전 클라이맥스라는 게 이 신호들의 논지다.
그렇다면 **미래를 안 보고 클라이맥스를 잡는 방법**이 있는지가 진짜 질문이다.

  A. 연구(앵커선택)   -- cluster_dedup: 클러스터 최극단. ⚠️미래참조.
  B. 인과-첫발동      -- 클러스터 첫 트리거. (v2에서 실패)
  C. 인과-후행최극단  -- 트리거가 나고 그 봉의 극단성이 **직전 GAP봉의 모든 트리거보다** 크면 발동.
                        뒤만 본다. 클러스터 안에서 갱신될 때마다 여러 번 발동할 수 있다.
  D. 인과-지연확정    -- 클러스터가 끝났음이 확정된 봉(마지막 트리거 + GAP)에서 앵커를 확정하고
                        **그 다음 봉 시가로 진입**. 앵커 선택 자체는 연구와 동일하되 진입만 늦다.

D는 연구와 앵커가 100% 같으므로 "앵커 선택이 옳은가"와 "진입을 늦춰도 남는가"를 분리해준다.
D가 살면 엣지는 진짜고 지연만 감수하면 되고, D도 죽으면 엣지는 그 봉에 즉시 들어가야만 존재한다.
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

OUT_DIR = ROOT / "tmp/eth_causal_anchor_variants_20260902"
START = pd.Timestamp("2024-01-01")
SPEC = {
    "short_term_return_z": {"gap": 3, "sl": 3.0, "arm": 1.0, "trail": 0.1, "horizon": 12},
    "demarker_extreme":    {"gap": 12, "sl": 2.0, "arm": 1.5, "trail": 0.1, "horizon": 8},
}
TOP2 = ["short_term_return_z", "demarker_extreme"]


def log(m: str) -> None:
    print(f"[variants] {m}", flush=True)


def v_first(idx, ex, most_neg, gap):
    keep, last = [], -10**9
    for i in np.sort(idx):
        if i - last > gap:
            keep.append(i); last = i
    return np.array(keep, np.int64)


def v_trailing_extreme(idx, ex, most_neg, gap):
    """트리거 봉의 극단성이 직전 gap봉 안의 모든 트리거보다 크면 발동 (뒤만 봄)."""
    idx = np.sort(idx)
    exm = {int(i): float(e) for i, e in zip(idx, ex)}
    keep = []
    for i in idx:
        prev = [exm[int(j)] for j in idx[(idx < i) & (idx >= i - gap)]]
        v = exm[int(i)]
        if not prev or (v < min(prev) if most_neg else v > max(prev)):
            keep.append(int(i))
    return np.array(keep, np.int64)


def v_delayed_anchor(idx, ex, most_neg, gap):
    """클러스터 확정 시점(마지막 트리거+gap)에서 결정. 반환은 (decision_idx, anchor_idx)."""
    idx = np.sort(idx)
    if len(idx) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    exm = {int(i): float(e) for i, e in zip(idx, ex)}
    out_dec, out_anc = [], []
    cur = [int(idx[0])]
    for i in idx[1:]:
        if int(i) - cur[-1] > gap:
            a = min(cur, key=lambda j: exm[j]) if most_neg else max(cur, key=lambda j: exm[j])
            out_anc.append(a); out_dec.append(cur[-1] + gap)
            cur = [int(i)]
        else:
            cur.append(int(i))
    a = min(cur, key=lambda j: exm[j]) if most_neg else max(cur, key=lambda j: exm[j])
    out_anc.append(a); out_dec.append(cur[-1] + gap)
    return np.array(out_dec, np.int64), np.array(out_anc, np.int64)


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
    raw = {"short_term_return_z": {"bottom": ret3_z <= -2.5, "top": ret3_z >= 2.5, "ex": ret3_z},
           "demarker_extreme": {"bottom": dem <= 0.10, "top": dem >= 0.90, "ex": dem}}

    tabs = {}
    for name, spec in SPEC.items():
        r, gap = raw[name], spec["gap"]
        built = {k: [] for k in ("A_연구앵커", "B_첫발동", "C_후행최극단", "D_지연확정")}
        for side in ("bottom", "top"):
            mneg = side == "bottom"
            m = np.nan_to_num(r[side].astype(float), nan=0.0).astype(bool)
            idx = np.flatnonzero(m)
            idx = idx[(idx < len(src) - spec["horizon"] - gap - 1) &
                      (src_ts[idx].to_numpy() >= np.datetime64(START))]
            ex = r["ex"][idx]
            for tag, fn in (("A_연구앵커", cluster_dedup), ("B_첫발동", v_first),
                            ("C_후행최극단", v_trailing_extreme)):
                a = (fn(idx, ex, most_negative=mneg, gap=gap) if tag == "A_연구앵커"
                     else fn(idx, ex, mneg, gap))
                built[tag] += [(int(i), side, int(i)) for i in a]
            d, anc = v_delayed_anchor(idx, ex, mneg, gap)
            built["D_지연확정"] += [(int(dd), side, int(aa)) for dd, aa in zip(d, anc)]
            log(f"{name}/{side}: raw {len(idx):,} | A {sum(1 for x in built['A_연구앵커'] if x[1]==side):,} "
                f"B {sum(1 for x in built['B_첫발동'] if x[1]==side):,} "
                f"C {sum(1 for x in built['C_후행최극단'] if x[1]==side):,} "
                f"D {sum(1 for x in built['D_지연확정'] if x[1]==side):,}")

        for tag, items in built.items():
            rows = [{"pos": pos_of[src_ts[i]], "side": sd, "atr_pct": atr_src[ai]}
                    for i, sd, ai in items
                    if i < len(src) and src_ts[i] in pos_of
                    and np.isfinite(atr_src[ai]) and atr_src[ai] > 0]
            f = pd.DataFrame(rows).sort_values("pos").reset_index(drop=True)
            dec = f["pos"].to_numpy(np.int64)
            sc = np.where(f["side"].to_numpy() == "bottom", 1.0, -1.0)
            a_ = f["atr_pct"].to_numpy(float)
            for sgn, lb in ((1.0, "real"), (-1.0, "flip")):
                t = per_fire_outcomes(ts, o, h, l, c, dec, sc * sgn, a_, spec["horizon"],
                                      spec["sl"], spec["arm"], spec["trail"])
                t["signal"] = name
                t["decision_pos"] = [pos_of[x] for x in t["decision_ts"]]
                tabs[(tag, name, lb)] = t

    prio = {n: i for i, n in enumerate(TOP2)}
    rows = []
    for tag in ("A_연구앵커", "B_첫발동", "C_후행최극단", "D_지연확정"):
        for lb in ("real", "flip"):
            allc = pd.concat([tabs[(tag, n, lb)] for n in TOP2], ignore_index=True)
            allc["prio"] = allc["signal"].map(prio)
            for wn, (lo, hi) in (("VAL", (VAL_START, OOS_START)), ("OOS", (OOS_START, HOLDOUT_START)),
                                 ("HOLDOUT", (HOLDOUT_START, hold_end + pd.Timedelta(minutes=5)))):
                w = allc[(allc.decision_ts >= lo) & (allc.decision_ts < hi)]
                for k in (1, 2):
                    s = summarize(sequential_portfolio(w[w["signal"].isin(TOP2[:k])], prio), f"top{k}")
                    s.update({"window": wn, "kind": lb, "variant": tag}); rows.append(s)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True); df.to_csv(OUT_DIR / "variants.csv", index=False)

    for arm in ("top1", "top2"):
        log(f"\n=== {arm} mean_bp / PF ===")
        r = df[(df.kind == "real") & (df.arm == arm)]
        p = r.pivot_table(index="variant", columns="window", values=["mean_bp", "pf", "n"])
        print(p.reindex(columns=["VAL", "OOS", "HOLDOUT"], level=1).round(2).to_string())

    log("\n=== 방향뒤집기 3창 통과 여부 ===")
    for tag in ("A_연구앵커", "B_첫발동", "C_후행최극단", "D_지연확정"):
        cf = df[df.variant == tag].pivot_table(index="arm", columns=["window", "kind"], values="total_bp")
        for arm in cf.index:
            ok = all(cf.loc[arm, (w, "real")] > max(cf.loc[arm, (w, "flip")], 0) for w in ("VAL", "OOS", "HOLDOUT"))
            print(f"  {tag:14s} {arm}: {'O' if ok else 'X'}  " + " ".join(
                f"{w[:3]} {cf.loc[arm,(w,'real')]:+.0f}/{cf.loc[arm,(w,'flip')]:+.0f}"
                for w in ("VAL", "OOS", "HOLDOUT")))
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
