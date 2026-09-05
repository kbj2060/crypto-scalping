#!/usr/bin/env python3
"""물타기 사다리의 **증거금 기회비용** + 복리 계좌 경로 (2026-09-05).

§5.30 1~4편과 `..._avgdown_upgrade_on_verified_edge_...`는 전부 **건당** 기대값이다. 건당으로는
안 보이는 비용이 하나 있다: **물타기를 하려면 사다리분 증거금을 미리 비워둬야 한다.**

  트랜치 20% x (1 + 물타기 4회) = **계좌 100%** -> 동시 보유 가능 포지션 **1개**.
  물타기 없음 · 트랜치 20%            -> 동시 5개.

지속 규칙은 하루 22건 발동한다. 슬롯이 1개면 그 중 대부분을 못 받는다. 즉 물타기의 대가는
"평균단가"가 아니라 **받지 못한 거래**다. 이 스크립트는 그 교환을 계좌 곡선으로 잰다.

모델(M1, 예약 사다리):
  · 진입 시 (1+max_adds)xtranche 를 **예약**한다. 그래야 물타기가 증거금 부족으로 실패하지 않는다.
    (예약하지 않는 M2는 물타기가 실패할 수 있고, 그러면 경로 자체가 달라져 단독 시뮬과 어긋난다.)
  · 자유 증거금 >= 예약분일 때만 신규 진입. 슬롯 상한도 함께 건다.
  · 청산 시 계좌에 손익을 더하고 예약을 푼다. **복리**(다음 거래 크기는 갱신된 계좌 기준).
  · 손익은 단독 경로 시뮬 결과를 그대로 쓴다(경로는 증거금과 독립).

HOLDOUT(>=2026-04-01) 미접촉.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

UPG = None
TRAIN_END, VAL_END, OOS_END = (pd.Timestamp(x) for x in ("2025-09-01", "2026-01-01", "2026-04-01"))
OUT = ROOT / "data/research/eth_avgdown_portfolio_margin_20260905"
WINDOWS = ("TRAIN", "VAL", "OOS")

# (container, tp_x, add_mode, add_x, max_adds, tranche, leverage, slot_cap, label)
ARMS = (
    # (A) 레버리지 스윕 -- 트레일(검증본)·물타기없음·트랜치 20%. "30배가 왜 안 되나"의 답.
    [("trail", 0.0, "none", 0.0, 0, 0.20, L, 5, f"[A] 트레일·물타기없음·20%x{L:g}배 (명목 {0.20*L:.1f}배/건)")
     for L in (3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0)] +
    # (B) 물타기 사다리 -- 30배 고정, 사다리 예약이 거래 수와 성장에 미치는 영향
    [("trail", 0.0, "price", 1.0, 2, 0.20, 30.0, 5, "[B] +물타기 1.0ATRx2 · 20%x30배 (사다리 60%)"),
     ("trail", 0.0, "price", 1.0, 4, 0.20, 30.0, 5, "[B] +물타기 1.0ATRx4 · 20%x30배 (사다리 100%)"),
     ("trail", 0.0, "price", 2.0, 4, 0.20, 30.0, 5, "[B] +물타기 2.0ATRx4 · 20%x30배 (사다리 100%)"),
     ("trail", 0.0, "price", 1.0, 4, 0.10, 30.0, 5, "[B] +물타기 1.0ATRx4 · 10%x30배 (사다리 50%)"),
     ("trail", 0.0, "signal", 0.0, 2, 0.20, 30.0, 5, "[B] +신호물타기x2 · 20%x30배 (사다리 60%)"),
     ("trail", 0.0, "signal", 0.0, 2, 0.20, 10.0, 5, "[B] +신호물타기x2 · 20%x10배 (사다리 60%)"),
     ("trail", 0.0, "signal", 0.0, 2, 0.10, 10.0, 5, "[B] +신호물타기x2 · 10%x10배 (사다리 30%)"),
     # (C) 물타기를 사다리 예약 없이 같은 명목으로 -- "그 자본을 그냥 처음부터 넣었다면"
     ("trail", 0.0, "none", 0.0, 0, 0.60, 10.0, 5, "[C] 물타기없음·60%x10배 (명목 6.0배 = 사다리 전개와 동일)"),
     ("trail", 0.0, "none", 0.0, 0, 0.50, 10.0, 5, "[C] 물타기없음·50%x10배 (명목 5.0배)"),
     # (D) 사용자 현행 컨테이너
     ("hybrid", 1.5, "none", 0.0, 0, 0.20, 10.0, 5, "[D] 부분익절(1.5ATR 절반)+트레일 · 20%x10배"),
     ("tp", 16.7, "price", 1.0, 4, 0.20, 30.0, 5, "[D] 사용자 현행: 고정TP16.7·손절없음·물타기 1.0ATRx4·20%x30배"),
     ("tp", 16.7, "none", 0.0, 0, 0.20, 30.0, 5, "[D] 고정TP16.7·손절없음·물타기없음·20%x30배"),
     ("tp", 33.3, "price", 1.0, 4, 0.20, 30.0, 5, "[D] 고정TP33.3·손절없음·물타기 1.0ATRx4·20%x30배")]
)
COSTS = [4.0, 7.8, 10.0]


def log(m): print(f"[pf-margin] {m}", flush=True)


def main() -> int:
    global UPG
    t0 = time.time()
    spec = importlib.util.spec_from_file_location(
        "upg", ROOT / "scripts/research_eth_avgdown_upgrade_on_verified_edge_20260905.py")
    UPG = importlib.util.module_from_spec(spec); spec.loader.exec_module(UPG)

    log("신호 프레임 재구성...")
    _s1 = UPG._load("s1_pfm", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
    _s1.VAL_END = OOS_END
    sig, _f, _e = _s1.build_sig()
    ts = pd.to_datetime(sig["timestamp"]).dt.tz_localize(None)
    m = (ts < OOS_END).to_numpy()
    sig, ts = sig.loc[m].reset_index(drop=True), ts.loc[m].reset_index(drop=True)
    o, h, l, c = (sig[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(sig)
    pc = np.r_[np.nan, c[:-1]]
    tr_ = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    atr = pd.Series(tr_).rolling(UPG.ATR_N).mean().to_numpy()

    up = np.zeros(n, bool); dn = np.zeros(n, bool)
    for name in UPG.SIGNALS:
        for side, arr in (("bottom", dn), ("top", up)):
            col = f"{side}_{name}"
            if col not in sig.columns:
                continue
            f = sig[col].fillna(False).to_numpy(bool); last = -10**9
            for i in np.flatnonzero(f):
                if i - last > UPG.GAP:
                    arr[i] = True
                last = i
    both = up & dn
    fire_up, fire_dn = up & ~both, dn & ~both
    P = np.flatnonzero(fire_up | fire_dn)
    S = np.where(fire_up[P], 1.0, -1.0)
    ok = (P + 1 + UPG.FWD < n) & np.isfinite(atr[P]) & (atr[P] > 0)
    P, S = P[ok], S[ok]
    split = np.where(ts < TRAIN_END, "TRAIN", np.where(ts < VAL_END, "VAL", "OOS"))
    log(f"  모집단 {len(P):,}건")

    # ⭐방향 축: 지속(검증본) vs 페이드(=칩 방향, 사용자가 실제로 따라가는 쪽). 자본구조는 동일.
    FADE = [("tp", 16.7, "price", 1.0, 4, 0.20, 30.0, 5, "[E] 페이드(칩방향) 고정TP16.7·물타기·20%x30배"),
            ("trail", 0.0, "none", 0.0, 0, 0.20, 10.0, 5, "[E] 페이드(칩방향) 트레일·물타기없음·20%x10배"),
            ("trail", 0.0, "none", 0.0, 0, 0.20, 3.0, 5, "[E] 페이드(칩방향) 트레일·물타기없음·20%x3배")]

    rows = []
    for (ct, tx, am, ax, ma, tr, lv, cap, label), sgn_mult in (
            [(a, 1.0) for a in ARMS] + [(a, -1.0) for a in FADE]):
        Suse = S * sgn_mult
        res = [UPG.simulate(o, h, l, c, atr, fire_up, fire_dn, p, s, ct, tx, am, ax, ma, lv, n)
               for p, s in zip(P, Suse)]
        au = np.array([r["acct_units"] for r in res], float)
        cu = np.array([r["cost_units"] for r in res], float)
        oc = np.array([r["outcome"] for r in res], object)
        ex = np.array([r["exit_off"] for r in res], int)
        entry_bar, exit_bar = P + 1, P + 1 + ex
        ladder = tr * (1 + ma)                                  # 예약 증거금(계좌 비율)
        for cost in COSTS:
            # 계좌 수익률(계좌 비율). 청산이면 투입 증거금 전액 손실.
            ret = np.where(oc == "liq", au * tr, (au - cost / 1e4 * cu) * tr * lv)
            # ⭐성장최적 명목(켈리): 명목 1배당 수익률 x -> N* = mean(x)/var(x) (계좌 대비 명목 배수)
            x_unit = np.where(oc == "liq", -1.0 / lv, au - cost / 1e4 * cu)
            for sp in WINDOWS:
                idx = np.flatnonzero(split[P] == sp)
                if len(idx) < 30:
                    continue
                order = idx[np.argsort(entry_bar[idx], kind="stable")]
                eq, peak, mdd = 1.0, 1.0, 0.0
                reserved, open_pos = 0.0, []                     # (exit_bar, reserve_frac, ret)
                taken_idx, skipped_margin, skipped_slot = [], 0, 0
                eq_path = []
                for kk in order:
                    eb = entry_bar[kk]
                    still = []
                    for (xb, rf, rr) in open_pos:
                        if xb <= eb:
                            eq *= (1.0 + rr); reserved -= rf
                            peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
                            eq_path.append(eq)
                        else:
                            still.append((xb, rf, rr))
                    open_pos = still
                    if len(open_pos) >= cap:
                        skipped_slot += 1; continue
                    if reserved + ladder > 1.0 + 1e-12:
                        skipped_margin += 1; continue
                    open_pos.append((exit_bar[kk], ladder, ret[kk])); reserved += ladder
                    taken_idx.append(kk)
                for (xb, rf, rr) in open_pos:
                    eq *= (1.0 + rr); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
                    eq_path.append(eq)
                taken = len(taken_idx)
                ti = np.array(taken_idx) if taken else np.array([], dtype=int)
                days = (ts.iloc[P[order]].max() - ts.iloc[P[order]].min()).days or 1
                rows.append({"arm": label, "container": ct, "tp_x": tx, "add_mode": am, "add_x": ax,
                             "max_adds": ma, "tranche": tr, "leverage": lv, "slot_cap": cap,
                             "direction": "cont" if sgn_mult > 0 else "fade",
                             "ladder_frac": round(ladder, 3), "cost_bp": cost, "window": sp,
                             "n_fires": int(len(idx)), "n_taken": taken,
                             "take_rate": round(taken / len(idx), 4),
                             "skipped_margin": skipped_margin, "skipped_slot": skipped_slot,
                             "final_equity": round(float(eq), 5),
                             "cagr": round(float(eq ** (365.0 / days) - 1.0), 4) if eq > 0 else -1.0,
                             "max_dd": round(float(mdd), 4),
                             "min_equity": round(float(min(eq_path)) if eq_path else 1.0, 5),
                             "exp_pct_per_taken": round(float(ret[ti].mean() * 100), 4) if taken else None,
                             "log_growth_per_taken": round(
                                 float(np.log1p(np.clip(ret[ti], -0.999999, None)).mean()), 6) if taken else None,
                             "ruin_rate_of_taken": round(float((oc[ti] == "liq").mean()), 5) if taken else None,
                             "unit_mean_bp": round(float(x_unit[idx].mean() * 1e4), 3),
                             "unit_sd_bp": round(float(x_unit[idx].std(ddof=1) * 1e4), 1),
                             "kelly_notional_per_trade": round(
                                 float(x_unit[idx].mean() / max(x_unit[idx].var(ddof=1), 1e-18)), 3),
                             "notional_per_tranche": round(tr * lv, 3),
                             "notional_full_ladder": round(tr * lv * (1 + ma), 3)})
        r10 = [x for x in rows if x["arm"] == label and x["cost_bp"] == 4.0]
        log(f"\n{label}  (사다리 예약 {ladder:.0%})")
        for x in r10:
            log(f"    {x['window']:5s} 체결 {x['n_taken']:5d}/{x['n_fires']:5d} ({x['take_rate']:.1%}) "
                f"| 최종자본 x{x['final_equity']:.3f} · CAGR {x['cagr']:+.1%} · MDD {x['max_dd']:.1%} "
                f"· 최저 x{x['min_equity']:.3f} · 전손 {x['ruin_rate_of_taken']:.2%} "
                f"· **로그성장 {x['log_growth_per_taken']:+.5f}/건**  [비용 4bp]")
            log(f"          명목: 트랜치 {x['notional_per_tranche']:.2f}배 · 사다리 전개 "
                f"{x['notional_full_ladder']:.2f}배 | 켈리 최적 {x['kelly_notional_per_trade']:.2f}배 "
                f"(건당 {x['unit_mean_bp']:+.2f}bp · SD {x['unit_sd_bp']:.0f}bp)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(
        {"holdout_touched": False, "model": "M1 reserved-ladder margin, compounding",
         "n_pop": int(len(P)), "rows": rows}, ensure_ascii=False, indent=2))
    log(f"\n산출: {OUT}/report.json ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
