#!/usr/bin/env python3
"""Zeus Baseline v2 — **임계값 곡선** (2026-09-17, dev/CPU · 학습 없음)

동결된 v2(balnobb 라우팅 × zigzag 부모 2개 앙상블 × 더블배리어 TP1.5%/SL1%)의 숫자는
**건수 3,700 맞춤**에서 나왔다. 그건 팔 간 비교를 위한 인위적 제약이고, 실제 임계값은 아니다.

🔴**가장 약한 고리가 빈도다** — 3,700 맞춤에서 1.23건/일로 「최소 1건/일」을 간신히 넘는다.
임계값을 실제로 정하면 그게 유지되는지가 이 실험의 질문이다.

한 칸을 고르면 그게 또 선택이므로 **곡선 전체**를 낸다. 같이 내는 것:
 · 통과 건수 · 건/일(=288/평균보유) · 건당bp · 함축 p · 날짜블록 CI · 순bp/일
 · ⭐**「최소 1건/일」을 만족하는 q 구간**과 그 구간에서의 성과 범위

캐시된 확률(stageP_probs.npz)만 쓴다 -- 재학습 0.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_side_skill_decomposition_20260917 as K  # noqa: E402

SEEDS = [613042, 27851, 904377, 155690, 488213]
FOLDS = [f for f in K.FOLDS if f[0] in ("F1", "F2", "F3", "CAND")]
CACHE = E.OUT / "stageP_probs.npz"
TPB, SLB, COST = K.BASE_TP * 1e4, K.BASE_SL * 1e4, 1.02
GRID = np.round(np.arange(0.34, 0.96, 0.02), 4)
MIN_PER_DAY = 1.0


def log(*a): print(*a, flush=True)


def main() -> int:
    cache = dict(np.load(CACHE, allow_pickle=True))
    df, _ = E.load()
    EN = ("bull", "bear", "chop")
    segs = []
    for name, _t0, _t1, v0, v1 in FOLDS:
        te = df[(df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")].reset_index(drop=True)
        ev = tabm._route_probs(te).argmax(1)
        # N0x2: 같은 폴드의 시드 i·i+1 을 평균 -> 시드 슬롯마다 하나씩
        per_slot = []
        for i, sd in enumerate(SEEDS):
            sd2 = SEEDS[(i + 1) % len(SEEDS)]
            D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
            for ei, en in enumerate(EN):
                z1 = dict(cache[f"{name}|N0{en}s{sd}"].item())
                z2 = dict(cache[f"{name}|N0{en}s{sd2}"].item())
                m = ev == ei
                D[m] = (z1["D"][m] + z2["D"][m]) / 2.0
                Q[m] = (z1["Q"][m] + z2["Q"][m]) / 2.0
            per_slot.append((D, Q))
        segs.append((name, te, per_slot))
        log(f"  {name}: {len(te):,}봉 · N0x2 재조합 {len(per_slot)} 시드슬롯")

    log(f"\n{'q':>6}{'통과':>8}{'통과율':>8}{'건/일':>7}{'중앙보유':>9}{'건당bp':>9}{'함축p':>8}"
        f"{'CI(건당)':>20}{'순/일':>8}{'1건/일':>7}")
    rows = []
    for q in GRID:
        per = []
        for si in range(len(SEEDS)):
            pnl, hold, days = [], [], []
            for name, te, per_slot in segs:
                D, Q = per_slot[si]
                da = D.argmax(1)
                qf = np.where(da > 0, Q[np.arange(len(Q)), da], Q[:, 0])
                side = np.where((da == 1) & (qf >= q), 1.0, np.where((da == 2) & (qf >= q), -1.0, 0.0))
                idx = np.where(side != 0)[0]
                if len(idx) < 20:
                    continue
                hi = pd.to_numeric(te["high"]).to_numpy(float)
                lo = pd.to_numeric(te["low"]).to_numpy(float)
                cl = pd.to_numeric(te["close"]).to_numpy(float)
                r, h, _a, _b, _c = K._first_touch_open(idx, side, hi, lo, cl,
                                                       K.BASE_TP, K.BASE_SL, K.MAXBARS)
                pnl.append(r * 1e4 - COST); hold.append(h.astype(float))
                days.append(te.timestamp.dt.floor("D").to_numpy()[idx])
            if not pnl:
                continue
            pnl = np.concatenate(pnl); hold = np.concatenate(hold); days = np.concatenate(days)
            lo_, hi_, _nd = E.block_ci(pnl, days)
            per.append((len(pnl), float(pnl.mean()), lo_, hi_, 288.0 / max(hold.mean(), 1e-9),
                        float(np.median(hold))))
        if not per:
            log(f"{q:>6.2f}  (통과 없음)"); continue
        a = np.array(per, dtype=float)
        n, g, clo, chi, pdy, mh = a.mean(0)
        tot = sum(len(te) for _n, te, _p in segs)
        ok = "✅" if pdy >= MIN_PER_DAY else "🔴"
        rows.append({"q": float(q), "n": n, "pass_rate": n / tot, "gross_bp": g,
                     "p": (g + COST + SLB) / (TPB + SLB), "ci95": [clo, chi],
                     "per_day": pdy, "median_hold": mh, "net_day": (g - 0.0) * pdy,
                     "meets_min": bool(pdy >= MIN_PER_DAY)})
        log(f"{q:>6.2f}{int(n):>8,}{n/tot*100:>7.2f}%{pdy:>7.2f}{mh:>9.0f}{g:>+9.2f}"
            f"{(g+COST+SLB)/(TPB+SLB)*100:>7.2f}%  [{clo:+7.2f},{chi:+7.2f}]{g*pdy:>8.1f}{ok:>7}")

    ok_rows = [r for r in rows if r["meets_min"]]
    if ok_rows:
        best_q = max(ok_rows, key=lambda r: r["net_day"])
        log(f"\n⭐「최소 1건/일」 만족 q 구간: {ok_rows[0]['q']:.2f} ~ {ok_rows[-1]['q']:.2f} "
            f"({len(ok_rows)}칸)")
        log(f"  그 구간 순/일 범위 {min(r['net_day'] for r in ok_rows):.1f} ~ "
            f"{max(r['net_day'] for r in ok_rows):.1f} · 건당bp {min(r['gross_bp'] for r in ok_rows):+.2f} ~ "
            f"{max(r['gross_bp'] for r in ok_rows):+.2f}")
        log(f"  ⚠️최고 칸(q={best_q['q']:.2f}, 순/일 {best_q['net_day']:.1f})을 «고르면» 그게 선택이다. "
            f"실제 임계값은 TRAIN 꼬리에서 정한다.")
    else:
        log("\n🔴「최소 1건/일」을 만족하는 q 가 없다.")
    (E.OUT / "stageS_v2_threshold_curve.json").write_text(json.dumps(rows, indent=2, default=float))
    log(f"저장: {E.OUT}/stageS_v2_threshold_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
