"""**ETH 스트래들 조이기** — 선택 강건성 · 경로 · 비용 (2026-09-14, 사용자 *"이더리움 스트래들 더 조여줘"*).

지금까지 나온 것: 익절 +10% / 손절 −3% 양측 동시진입을 **예측변동성 최저 5분위**에서만 하면
5창(2022~23 · TRAIN · VAL · OOS · TEST) 전부 쌍당 손익이 양수다. 게이트 4종에서 재현되고 펀딩과 무관하다.
약한 곳은 넷이고 이 스크립트가 그 넷을 친다:

① **(익절,손절) 주변 강건성** — 10%/3% 는 내가 격자를 훑어 고른 칸이다. 이웃 칸에서 매끄럽게
   좋아야 진짜고, 그 칸만 뾰족하면 격자 운이다.
② **게이트 컷 민감도** — 「최저 20%」도 눈으로 골랐다. 10~50% 로 움직이며 **단조**인지 본다.
③ 🔴**경로** — 지금까지 평균만 봤다. 한 번에 한 쌍씩 **순차로** 굴려 자산곡선·MDD·최악 연속손실·
   연도별 손익을 낸다. 쌍당 −322bp 가 연달아 오면 평균이 양수여도 못 굴린다.
④ **비용·체결 민감도** — 손절 슬리피지를 중앙 14bp 로 고정했다. 꼬리(99분위 227bp)와
   「익절 지정가가 닿아도 일부는 못 먹는다」를 넣어도 남는지 본다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_direction_barrier_label_20260914 as B  # noqa: E402
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

OUT = ROOT / "data/research/eth_direction_barrier_label_20260914"
ART = OUT / "vol_model_holdout2223_ETHUSDT.joblib"
WINS = ("BACK22_23", "TRAIN", "VAL", "OOS", "TEST")


def gate_series(d, c, hi, lo) -> np.ndarray:
    import joblib
    import live_eth_sizing_vol_model_20260912 as svm
    art = joblib.load(ART)
    X = svm.build_features(d.timestamp, c, d.quote_volume.to_numpy(float),
                           d.trades.to_numpy(float), hi, lo)
    return np.log(np.clip(svm.predict_vol(art["models"], X), 1e-9, None))


def cell(c, hi, lo, ok, win, u, dn, stride, g, edge):
    """게이트 통과 앵커의 쌍당 손익·블록 t·해결시간. 창마다."""
    res = {}
    for w in WINS:
        L = B.label_window(c, hi, lo, ok, *win[w], stride, u, dn)
        if len(L["idx"]) < 100:
            continue
        sel = g[L["idx"]] <= edge
        if sel.sum() < 30:
            continue
        m, bars = L["m"][sel], L["bars"][sel]
        bb = float(np.median(bars))
        blk = (L["idx"][sel] // max(int(bb), 1)).astype(np.int64)
        bm = np.array([m[blk == k].mean() for k in np.unique(blk)])
        t = float(bm.mean() / (bm.std(ddof=1) / np.sqrt(len(bm)))) if len(bm) > 2 and bm.std(ddof=1) > 0 else np.nan
        res[w] = {"n": int(sel.sum()), "pair_bp": float(m.mean()), "block_t": t,
                  "hours": bb * 5 / 60, "blocks": int(len(bm))}
    return res


def sequential(c, hi, lo, ok, g, edge, lo_i, hi_i, u, dn, lev: float = 1.0):
    """🔴한 번에 **한 쌍**만. 두 다리가 모두 끝나야 다음 쌍을 연다 -- 겹치는 앵커로 평균을 내는 것과
    달리 이건 실제로 굴릴 수 있는 경로다. 자본배수는 쌍당 수익 × lev 로 복리."""
    eq, peak, mdd = 1.0, 1.0, 0.0
    i, pairs, curve, streak, worst_streak = lo_i, [], [], 0.0, 0.0
    while i < hi_i:
        if not (ok[i] and np.isfinite(g[i]) and g[i] <= edge):
            i += 1; continue
        a = B.leg(c, hi, lo, i, 1, u, dn); b = B.leg(c, hi, lo, i, 2, u, dn)
        if a is None or b is None:
            i += 1; continue
        m = 0.5 * (a[0] + b[0])                       # 쌍당 수익률(비용 포함, size-free)
        end = i + int(max(a[1], b[1]))
        eq *= (1.0 + lev * m)
        peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
        streak = min(0.0, streak + m) if m < 0 else 0.0
        worst_streak = min(worst_streak, streak)
        pairs.append({"i": i, "end": end, "bp": 1e4 * m, "hours": (end - i) * 5 / 60})
        curve.append(eq)
        i = end + 1
        if eq <= 0:
            break
    bps = np.array([p["bp"] for p in pairs]) if pairs else np.zeros(1)
    return {"n_pairs": len(pairs), "equity": eq, "mdd": mdd, "mean_bp": float(bps.mean()),
            "median_bp": float(np.median(bps)), "win_rate": float((bps > 0).mean()),
            "worst_pair_bp": float(bps.min()), "worst_streak_bp": 1e4 * worst_streak,
            "median_hours": float(np.median([p["hours"] for p in pairs])) if pairs else np.nan,
            "t": float(bps.mean() / (bps.std(ddof=1) / np.sqrt(len(bps)))) if len(bps) > 2 else np.nan,
            "pairs": pairs}


def concurrent(c, hi, lo, ok, g, edge, lo_i, hi_i, u, dn, kmax: int, stride: int = 1):
    """동시 **최대 kmax 쌍**. 자격 봉마다 빈 슬롯이 있으면 자본의 1/kmax 로 연다.

    🔴순차(k=1)와 겹침평균의 차이는 «진입 시점 표집»이다 -- 순차는 직전 쌍이 끝난 직후로 몰리고
    겹침평균은 균등하다. 그 둘이 부호가 다르면 결과가 **진입 시점에 민감**하다는 뜻이므로,
    실제로 굴릴 수 있는 중간(k=4·8)이 어느 쪽인지가 판정이다."""
    eq, peak, mdd = 1.0, 1.0, 0.0
    open_until = []                      # 열린 쌍의 종료 봉
    bps = []
    pend = {}                            # 종료봉 -> [수익률…]
    for i in range(lo_i, hi_i, stride):
        for e in sorted([k for k in pend if k <= i]):
            for m in pend.pop(e):
                eq *= (1.0 + m / kmax)
                peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
        open_until = [e for e in open_until if e > i]
        if len(open_until) >= kmax or not (ok[i] and np.isfinite(g[i]) and g[i] <= edge):
            continue
        a = B.leg(c, hi, lo, i, 1, u, dn); b = B.leg(c, hi, lo, i, 2, u, dn)
        if a is None or b is None:
            continue
        m = 0.5 * (a[0] + b[0]); end = i + int(max(a[1], b[1]))
        open_until.append(end); pend.setdefault(end, []).append(m); bps.append(1e4 * m)
    for e in sorted(pend):
        for m in pend[e]:
            eq *= (1.0 + m / kmax); peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
    b_ = np.array(bps) if bps else np.zeros(1)
    return {"n_pairs": len(bps), "equity": eq, "mdd": mdd, "mean_bp": float(b_.mean()),
            "t": float(b_.mean() / (b_.std(ddof=1) / np.sqrt(len(b_)))) if len(b_) > 2 else np.nan}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--tag", default="tighten")
    a = ap.parse_args()
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    ok = sm["ok"]
    g = gate_series(d, c, hi, lo)
    tl, th = win["TRAIN"]
    edge20 = float(np.nanpercentile(g[tl:th], 20))
    rep = {}

    print("① (익절, 손절) 주변 강건성 — 게이트 최저20% 적용, 창별 쌍당bp")
    print(f"{'익절/손절':>11} " + " ".join(f"{w:>10}" for w in WINS) + f" {'5창양수':>7} {'중앙h':>6}")
    grid = {}
    for u in (0.06, 0.08, 0.10, 0.12, 0.15):
        for dn in (0.02, 0.025, 0.03, 0.04):
            r = cell(c, hi, lo, ok, win, u, dn, a.stride, g, edge20)
            if len(r) < len(WINS):
                continue
            v = [r[w]["pair_bp"] for w in WINS]
            grid[f"{u*100:g}%/{dn*100:g}%"] = r
            print(f"{u*100:g}%/{dn*100:g}%".rjust(11) + " " +
                  " ".join(f"{x:>+10.1f}" for x in v) +
                  f" {'✅' if min(v) > 0 else '  ':>6} {r['OOS']['hours']:>6.0f}")
    npass = sum(1 for r in grid.values() if min(r[w]["pair_bp"] for w in WINS) > 0)
    print(f"  ⇒ {npass}/{len(grid)} 칸이 5창 전부 양수 (우연 기대 {len(grid)/32:.1f}칸)")
    rep["grid"] = grid

    print("\n② 게이트 컷 민감도 — 10%/3%")
    print(f"{'컷':>6} " + " ".join(f"{w:>10}" for w in WINS) + f" {'n(OOS)':>8}")
    cuts = {}
    for pct in (10, 15, 20, 25, 30, 40, 50, 100):
        e = float(np.nanpercentile(g[tl:th], pct))
        r = cell(c, hi, lo, ok, win, 0.10, 0.03, a.stride, g, e)
        cuts[pct] = r
        print(f"{pct:>5}% " + " ".join(f"{r[w]['pair_bp']:>+10.1f}" if w in r else f"{'-':>10}"
                                       for w in WINS) + f" {r['OOS']['n']:>8,}")
    rep["cuts"] = cuts

    print("\n③ 순차 진입 경로 — 한 번에 한 쌍 (10%/3%, 게이트 최저20%)")
    print(f"{'창':>10} {'쌍수':>5} {'평균bp':>8} {'중앙bp':>8} {'승률':>6} {'t':>6} "
          f"{'최악1쌍':>9} {'최악연속':>9} {'자산배수':>8} {'MDD':>7} {'중앙h':>6}")
    seq = {}
    for w in WINS:
        s = sequential(c, hi, lo, ok, g, edge20, *win[w], 0.10, 0.03)
        seq[w] = {k: v for k, v in s.items() if k != "pairs"}
        print(f"{w:>10} {s['n_pairs']:>5} {s['mean_bp']:>+8.1f} {s['median_bp']:>+8.1f} "
              f"{s['win_rate']:>6.1%} {s['t']:>+6.2f} {s['worst_pair_bp']:>+9.1f} "
              f"{s['worst_streak_bp']:>+9.1f} {s['equity']:>8.3f} {s['mdd']:>7.1%} {s['median_hours']:>6.0f}")
    rep["sequential"] = seq

    print("\n④ 비용·체결 민감도 (순차, 전 창 합산 쌍당bp)")
    base_slip = B.H.STOP_SLIP_MED_BP
    print(f"{'가정':>28} " + " ".join(f"{w:>9}" for w in WINS))
    sens = {}
    for name, slip, fee_extra in (("기준(슬립14bp)", 14.0, 0.0), ("슬립 50bp", 50.0, 0.0),
                                  ("슬립 100bp", 100.0, 0.0), ("슬립 227bp(99분위)", 227.0, 0.0),
                                  ("슬립14 + 익절 추가5bp", 14.0, 5.0)):
        B.H.STOP_SLIP_MED_BP = slip
        old_peg = B.H.PEG_EXIT_BP; B.H.PEG_EXIT_BP = old_peg + fee_extra
        row = {}
        for w in WINS:
            s = sequential(c, hi, lo, ok, g, edge20, *win[w], 0.10, 0.03)
            row[w] = s["mean_bp"]
        B.H.PEG_EXIT_BP = old_peg
        sens[name] = row
        print(f"{name:>28} " + " ".join(f"{row[w]:>+9.1f}" for w in WINS))
    B.H.STOP_SLIP_MED_BP = base_slip
    rep["sensitivity"] = sens

    print("\n⑤ 동시 보유 쌍수 — 겹침평균이 실제로 굴릴 수 있는 값인가 (10%/3%, 컷 20%)")
    print(f"{'창':>10} " + " ".join(f"{'k='+str(k):>22}" for k in (1, 4, 8, 32)))
    conc = {}
    for w in WINS:
        row = {}
        for k in (1, 4, 8, 32):
            r = concurrent(c, hi, lo, ok, g, edge20, *win[w], 0.10, 0.03, k, stride=3)
            row[k] = r
        conc[w] = row
        print(f"{w:>10} " + " ".join(
            f"{row[k]['mean_bp']:>+7.1f}bp n{row[k]['n_pairs']:<4} x{row[k]['equity']:.2f}"
            for k in (1, 4, 8, 32)))
    rep["concurrent"] = {w: {str(k): {kk: vv for kk, vv in v.items()} for k, v in r.items()}
                         for w, r in conc.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(f"\n저장: {OUT/(a.tag+'.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
