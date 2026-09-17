#!/usr/bin/env python3
"""Zeus — **N0x2 베이스라인의 TP/SL 격자** (2026-09-17, 학습 0)

TP1.5%/SL1% 는 **배포 부모(zig075) 추론** 위에서 고른 값이다 —
`docs/experiments/omega461_exit_barrier_design_20260917.md` §6 이 남긴 구멍:
「이 숫자는 전부 배포 방향머리 기준이다」. N0x2 는 balnobb 라우팅으로 새로 학습한
6개 모델이라 발화하는 봉이 다르므로 같은 배리어가 최적이라는 보장이 없다.

⭐**재학습이 필요 없다.** 더블배리어는 학습 라벨이 아니라 **청산 규칙**이고 각 팔의 두
머리는 이미 학습돼 캐시에 있다. TP/SL 을 바꿔도 **그 팔의 진입 집합은 한 건도 안 변한다**
— 팔마다 «동일 3,700건» 위에서 청산만 갈아끼운 비교가 된다.

🔴**그리고 순환이 하나 있다**(사용자 지적, 2026-09-17): TP1.5%/SL1% 자체를 고른 격자의
대상이 **zig075 = zigzag 라벨 모델**이었다. 그 배리어로 라벨 서열을 쟀으니 출전 선수
하나에 맞춘 청산으로 전원을 채점한 셈이다. 그래서 이 스크립트는 **라벨 3계열 각각에**
같은 격자를 돌려 **각자 최적 배리어에서** 비교한다:
  N0   zigzag 양두 (3모델)      N5  h48 양두 (3모델)      N7  더블배리어 양두 (3모델)
  N0x2 zigzag 양두 ×2 앙상블 (6모델, 현행 Baseline v2)
⭐용량이 섞이지 않게 서열 판정은 **N0 · N5 · N7 (전부 3모델)** 로 한다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm            # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_side_skill_decomposition_20260917 as K    # noqa: E402

SEEDS = [613042, 27851, 904377, 155690, 488213]
FOLDS = [f for f in K.FOLDS if f[0] in ("F1", "F2", "F3", "CAND")]
CACHE = Path(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cache=")),
                  str(E.OUT / "stageP_probs.npz")))
TARGET, COST, PEG = 3700, 1.02, 5.52
EN = ("bull", "bear", "chop")
ARMS = (next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--arms=")), None)
        or ["N0", "N5", "N7", "N0x2"])
# 팔 이름은 `N7@<라벨태그>` 형태를 받는다 -- 같은 구조를 «다른 라벨»로 재학습한 캐시를
# 가리키기 위해서다(캐시 키에 라벨 태그가 붙는다, 2026-09-17 수정).
_KNOWN = {"N0", "N5", "N7", "N0x2"}
_base = lambda a: a.split("@", 1)[0]
assert {_base(a) for a in ARMS} <= _KNOWN, f"모르는 팔: {sorted({_base(a) for a in ARMS} - _KNOWN)}"
TPS = [0.010, 0.015, 0.020, 0.025, 0.030]
SLS = [0.005, 0.007, 0.010, 0.013]


def log(*a): print(*a, flush=True)


def ckey(arm, fold, ei, sd):
    """캐시 키 규약이 팔마다 다르다 -- N0 는 전문가 «이름», N5/N7 은 «정수» 인덱스.
    `N7@tag` 면 키 끝에 `@tag` 가 붙는다(라벨별로 갈린 캐시)."""
    b, _, tag = arm.partition("@")
    if b in ("N0", "N0x2"):
        return f"{fold}|N0{EN[ei]}s{sd}"
    return f"{fold}|{b}{ei}s{sd}" + (f"@{tag}" if tag else "")


def main() -> int:
    z = np.load(CACHE, allow_pickle=True)
    df, _ = E.load()
    log(f"캐시 {CACHE} · 시드 {SEEDS} · 폴드 {[f[0] for f in FOLDS]} · 격자 {len(TPS)}×{len(SLS)}")

    bars, SEGS = {}, {a: [] for a in ARMS}      # SEGS[arm] = [(name, te, hi, lo, cl, [(D,Q)×5])]
    for name, _t0, _t1, v0, v1 in FOLDS:
        te = df[(df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")].reset_index(drop=True)
        ev = tabm._route_probs(te).argmax(1)
        bars[name] = (te, pd.to_numeric(te["high"]).to_numpy(np.float64),
                      pd.to_numeric(te["low"]).to_numpy(np.float64),
                      pd.to_numeric(te["close"]).to_numpy(np.float64))
        for arm in ARMS:
            per_seed = []
            for sd in SEEDS:
                D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
                for ei in range(3):             # balnobb 하드 라우팅(argmax)
                    c = dict(z[ckey(arm, name, ei, sd)].item()); m = ev == ei
                    D[m], Q[m] = c["D"][m], c["Q"][m]
                per_seed.append((D, Q))
            if _base(arm) == "N0x2":                   # 같은 폴드의 시드 i, i+1 앙상블
                slots = [((per_seed[i][0] + per_seed[(i + 1) % len(SEEDS)][0]) / 2.0,
                          (per_seed[i][1] + per_seed[(i + 1) % len(SEEDS)][1]) / 2.0)
                         for i in range(len(SEEDS))]
            else:
                slots = per_seed
            SEGS[arm].append((name, *bars[name], slots))
        log(f"  {name}: {len(te):,}봉 · {len(ARMS)}팔")

    # ── 진입 집합은 TP/SL 과 무관하므로 팔·슬롯당 «한 번»만 정한다 ──
    def pick(arm):
        out = []
        for si in range(len(SEEDS)):
            allq = []
            for _n, _te, _h, _l, _c, slots in SEGS[arm]:
                D, Q = slots[si]; da = D.argmax(1)
                qf = np.where(da > 0, Q[np.arange(len(Q)), da], Q[:, 0])
                allq.append(qf[da != 0])
            allq = np.concatenate(allq)
            thr = float(np.sort(allq)[::-1][min(TARGET, len(allq)) - 1])
            per_fold = []
            for name, te, h, l, c, slots in SEGS[arm]:
                D, Q = slots[si]; da = D.argmax(1)
                qf = np.where(da > 0, Q[np.arange(len(Q)), da], Q[:, 0])
                side = np.where((da == 1) & (qf >= thr), 1.0,
                                np.where((da == 2) & (qf >= thr), -1.0, 0.0))
                per_fold.append((te, h, l, c, side, np.where(side != 0)[0]))
            out.append((thr, per_fold))
            log(f"  {arm} 슬롯{si}: q={thr:.4f} · 진입 {sum(len(f[5]) for f in per_fold):,}건")
        return out

    def cell(entries, tp, sl):
        tpb, slb = tp * 1e4, sl * 1e4
        acc = []
        for _thr, per_fold in entries:
            pnl, hold, days, rsn = [], [], [], []
            for te, h, l, c, side, idx in per_fold:
                if len(idx) < 20:
                    continue
                r, hh, _res, rn, _m = K._first_touch_open(idx, side, h, l, c, tp, sl, K.MAXBARS)
                pnl.append(r * 1e4 - COST); hold.append(hh.astype(float)); rsn.append(rn)
                days.append(te.timestamp.dt.floor("D").to_numpy()[idx])
            pnl = np.concatenate(pnl); hold = np.concatenate(hold)
            days = np.concatenate(days); rsn = np.concatenate(rsn)
            lo_, hi_, nd = E.block_ci(pnl, days)
            g = float(pnl.mean()); pdy = 288.0 / max(hold.mean(), 1e-9)
            acc.append({"n": len(pnl), "g": g, "lo": lo_, "hi": hi_, "indep": nd,
                        "med": float(np.median(hold)), "per_day": pdy,
                        "net_day": g * pdy, "net_day_peg": (g + COST - PEG) * pdy,
                        "sl_r": float((rsn == 0).mean()), "tp_r": float((rsn == 1).mean()),
                        "un_r": float((rsn == 2).mean())})
        A = pd.DataFrame(acc); g = A.g.mean()
        return {"tp": tp, "sl": sl, "n": int(A.n.mean()), "gross_bp": g,
                "p": (g + COST + slb) / (tpb + slb), "p_star": (slb + COST) / (tpb + slb),
                "ci95": [A.lo.mean(), A.hi.mean()], "indep_days": A.indep.mean(),
                "median_hold": A.med.mean(), "per_day": A.per_day.mean(),
                "net_day": A.net_day.mean(), "net_day_peg": A.net_day_peg.mean(),
                "sl_rate": A.sl_r.mean(), "tp_rate": A.tp_r.mean(), "un_rate": A.un_r.mean(),
                "seed_spread": A.g.max() - A.g.min()}

    LBL = {"N0": "zigzag 양두 ×3", "N5": "h48 양두 ×3", "N7": "더블배리어 양두 ×3",
           "N0x2": "zigzag 양두 ×6(Baseline v2)"}
    LBL = {a: LBL[_base(a)] + (f" [{a.split('@')[1]}]" if "@" in a else "") for a in ARMS}
    rows, bests = [], {}
    for arm in ARMS:
        entries = pick(arm)
        R = pd.DataFrame([dict(arm=arm, **cell(entries, tp, sl)) for tp in TPS for sl in SLS])
        # ⭐한 팔 안에서 건수가 같아야 이 표가 「청산 비교」다. 다르면 선별성이 섞인 것이다.
        assert R.n.nunique() == 1, f"{arm}: 셀마다 건수가 다르다 {sorted(R.n.unique())}"
        rows += R.to_dict("records")
        log(f"\n{'='*118}\n■ {arm} — {LBL[arm]} · 동일 진입 {R.n.iloc[0]:,}건 · 5슬롯 평균 · "
            f"시간청산 없음(최대 {K.MAXBARS}봉)\n{'='*118}")
        log(f"{'TP':>5}{'SL':>5}{'건당bp':>9}{'함축p':>8}{'손익분기':>9}{'CI95':>20}"
            f"{'SL%':>7}{'TP%':>7}{'미해소':>7}{'중앙보유':>8}{'건/일':>7}{'순/일':>8}{'peg':>8}{'시드폭':>8}")
        for _, r in R.iterrows():
            star = " 🔴" if r.per_day < 1.0 else ("  ✅" if r.ci95[0] > 0 else "   ")
            log(f"{r.tp*100:>4.1f}%{r.sl*100:>4.1f}%{r.gross_bp:>+9.2f}{r.p*100:>7.2f}%"
                f"{r.p_star*100:>8.2f}%  [{r.ci95[0]:+7.2f},{r.ci95[1]:+7.2f}]"
                f"{r.sl_rate*100:>6.1f}%{r.tp_rate*100:>6.1f}%{r.un_rate*100:>6.1f}%"
                f"{r.median_hold:>8.0f}{r.per_day:>7.2f}{r.net_day:>8.1f}{r.net_day_peg:>8.1f}"
                f"{r.seed_spread:>8.2f}{star}")
        ok = R[R.per_day >= 1.0]                       # 🔴«최소 1건/일» 을 못 지키면 후보가 아니다
        bests[arm] = (R[(R.tp == 0.015) & (R.sl == 0.010)].iloc[0],
                      (ok if len(ok) else R).loc[(ok if len(ok) else R).net_day.idxmax()])

    log(f"\n{'='*118}\n■ 라벨 서열 — «각자 최적 배리어»에서 다시 (건/일 ≥ 1 제약)\n{'='*118}")
    log(f"{'팔':<6}{'라벨':<26}{'현행1.5/1.0':>13}{'자기최적':>12}{'배리어':>12}"
        f"{'건/일':>7}{'순/일':>8}{'시드폭':>8}")
    for arm in ARMS:
        cur, bst = bests[arm]
        log(f"{arm:<6}{LBL[arm]:<26}{cur.gross_bp:>+13.2f}{bst.gross_bp:>+12.2f}"
            f"  TP{bst.tp*100:>3.1f}%/SL{bst.sl*100:.1f}%{bst.per_day:>7.2f}"
            f"{bst.net_day:>8.1f}{bst.seed_spread:>8.2f}")
    log("⭐읽는 법: 현행 대비 «자기최적»이 크게 오르는 팔이 있으면 기존 서열이 배리어 정렬의")
    log("  산물이었다는 뜻이다. 순위가 그대로면 배리어는 서열의 원인이 아니다.")
    log("⚠️팔마다 20칸 중 최고를 고른 값이므로 «위로» 편향돼 있다 — 시드폭·CI 폭과 함께 읽는다.")

    out = E.OUT / "stageT_sltp_grid_by_label.json"
    out.write_text(json.dumps(rows, indent=2, default=float))
    log(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
