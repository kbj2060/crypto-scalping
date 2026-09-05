#!/usr/bin/env python3
"""§5.30 격자를 **통과한 셀**을 복리 계좌에 그대로 태우면 어떻게 되나 (2026-09-05).

`..._avgdown_boundary_and_multiplicity_...`는 확장 격자 243셀 중 **126셀이 세 창 전부 계좌 양수**이고
다중성 귀무(평균 41.9 · 95분위 65)의 100 백분위라고 판정했다. 그런데 그 판정은 전부 **건당 기대값**이다.

건당 기대값이 양수여도 복리 계좌는 죽을 수 있다: 명목이 켈리 최적의 몇 배면 분산 드래그(sigma^2/2)가
평균을 넘는다. 30배 x 20% 트랜치 x 사다리 5단 = 명목 30배는 그 영역이다.

이 스크립트는 통과 셀 상위 N개를 **선행 스크립트의 시뮬레이션 함수 그대로** 돌리되(파리티 검증 포함),
동시보유·**사다리 예약 증거금**·복리를 걸어 계좌 곡선을 낸다. 규약은 선행 상속(HOLD 72봉, 비용 4bp,
30배, 봉내 순서 비관=청산 우선). HOLDOUT 미접촉.
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

BND = ROOT / "data/research/eth_chip_avgdown_boundary_multiplicity_20260905/report.json"
OUT = ROOT / "data/research/eth_chip_avgdown_passing_cells_compounding_20260905"
TOP_N, SLOT_CAP = 10, 5
WINDOWS = ("TRAIN", "VAL", "OOS")


def log(m): print(f"[pass-cmp] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    spec = importlib.util.spec_from_file_location(
        "bnd", ROOT / "scripts/research_eth_chip_avgdown_boundary_and_multiplicity_20260905.py")
    BM = importlib.util.module_from_spec(spec); spec.loader.exec_module(BM)
    HOLD, LEV, COST, LIQ = BM.HOLD, BM.LEVERAGE, BM.COST_BP, BM.LIQ_BP

    def sim2(o, h, l, c, pos, sgn, tp_bp, add_bp, max_adds, sl_bp, tranche, n):
        """BM.simulate와 **동일 로직** + 청산봉 오프셋. 아래 파리티 검사가 동일성을 확인한다."""
        if pos + 1 >= n:
            return np.nan, "", 0, 0
        entry = o[pos + 1]; avg, k = entry, 1
        end = min(pos + 1 + HOLD, n - 1); scale = tranche * LEV * 100.0
        for j in range(pos + 1, end + 1):
            hi, lo = h[j], l[j]
            adv = (avg - lo) / avg if sgn > 0 else (hi - avg) / avg
            fav = (hi - avg) / avg if sgn > 0 else (avg - lo) / avg
            if adv * 1e4 >= LIQ:
                return -tranche * k * 100.0, "liq", k, j - pos - 1
            while k <= max_adds and adv * 1e4 >= add_bp:
                add_px = avg * (1 - sgn * add_bp / 1e4)
                avg = (avg * k + add_px) / (k + 1); k += 1
                adv = (avg - lo) / avg if sgn > 0 else (hi - avg) / avg
                fav = (hi - avg) / avg if sgn > 0 else (avg - lo) / avg
                if adv * 1e4 >= LIQ:
                    return -tranche * k * 100.0, "liq", k, j - pos - 1
            if sl_bp is not None and adv * 1e4 >= sl_bp:
                return (-sl_bp * k - COST * k) / 1e4 * scale, "sl", k, j - pos - 1
            if fav * 1e4 >= tp_bp:
                return (tp_bp * k - COST * k) / 1e4 * scale, "tp", k, j - pos - 1
        move = sgn * (c[end] / avg - 1) * 1e4
        return (move * k - COST * k) / 1e4 * scale, "timeout", k, end - pos - 1

    rep = json.loads(BND.read_text())
    valid = [x for x in rep["cells"] if not x["degenerate"] and len(x["splits"]) == 3]
    passing = [x for x in valid if all(x["splits"][s]["account_pct"] > 0 for s in x["splits"])]
    passing.sort(key=lambda z: -z["splits"]["TRAIN"]["excess_vs_null"])
    log(f"통과 셀 {len(passing)} / 유효 {len(valid)} -> 상위 {TOP_N} 복리 검정")

    log("신호 프레임 재구성...")
    _s1 = BM._load("s1_pc", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
    _s1.VAL_END = BM.OOS_END
    sig, _f, _e = _s1.build_sig()
    ts = pd.to_datetime(sig["timestamp"]).dt.tz_localize(None)
    m = (ts < BM.OOS_END).to_numpy()
    sig, ts = sig.loc[m].reset_index(drop=True), ts.loc[m].reset_index(drop=True)
    o, h, l, c = (sig[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(sig)
    split = np.where(ts < BM.TRAIN_END, "TRAIN", np.where(ts < BM.VAL_END, "VAL", "OOS"))
    P, S = [], []
    for name, hz in BM.CHIP.items():
        for side in ("bottom", "top"):
            col = f"{side}_{name}"
            if col not in sig.columns:
                continue
            for pos in BM.first_fire_positions(sig[col].fillna(False).to_numpy(bool), hz):
                P.append(int(pos)); S.append(1.0 if side == "bottom" else -1.0)
    P, S = np.asarray(P), np.asarray(S)
    ok = P + 1 + HOLD < n
    P, S = P[ok], S[ok]
    log(f"  칩 발동 {len(P):,}건 (칩 방향 = 페이드, 선행과 동일)")

    # ── 파리티: sim2가 BM.simulate와 같은 값을 내는가 (무작위 300건) ──
    rng = np.random.default_rng(7)
    idx = rng.choice(len(P), 300, replace=False)
    tp0, add0, ma0, sl0, tr0 = (passing[0]["tp_bp"], passing[0]["add_bp"], passing[0]["max_adds"],
                                passing[0]["sl_bp"], passing[0]["tranche"])
    d = max(abs(sim2(o, h, l, c, P[i], S[i], tp0, add0, ma0, sl0, tr0, n)[0]
                - BM.simulate(o, h, l, c, P[i], S[i], tp0, add0, ma0, sl0, tr0, n)[0]) for i in idx)
    log(f"  ⭐파리티 |Δ| = {d:.3e} (sim2 vs 선행 simulate, 300건)")
    assert d < 1e-9, "시뮬레이션 파리티 실패"

    rows = []
    for rank, cell in enumerate(passing[:TOP_N], 1):
        tp, add, ma, sl, tr = (cell["tp_bp"], cell["add_bp"], cell["max_adds"],
                               cell["sl_bp"], cell["tranche"])
        r = [sim2(o, h, l, c, p, s, tp, add, ma, sl, tr, n) for p, s in zip(P, S)]
        acct = np.array([x[0] for x in r], float) / 100.0        # 계좌 비율
        oc = np.array([x[1] for x in r], object)
        ex = np.array([x[3] for x in r], int)
        entry_bar, exit_bar = P + 1, P + 1 + ex
        ladder = tr * (1 + ma)
        sl_txt = "없음" if sl is None else f"{sl:.0f}"
        label = (f"#{rank} TP{tp:.1f} 물타기{add:.0f}bpx{ma} 손절{sl_txt} 트랜치{tr:.0%} "
                 f"(사다리 {ladder:.0%} · 명목 {tr*LEV*(1+ma):.1f}배)")
        out = {"rank": rank, "tp_bp": tp, "add_bp": add, "max_adds": ma, "sl_bp": sl,
               "tranche": tr, "ladder_frac": round(ladder, 3),
               "notional_full_ladder": round(tr * LEV * (1 + ma), 2), "windows": {}}
        log(f"\n{label}")
        for sp in WINDOWS:
            ii = np.flatnonzero(split[P] == sp)
            if len(ii) < 30:
                continue
            order = ii[np.argsort(entry_bar[ii], kind="stable")]
            eq, peak, mdd, reserved, open_pos, taken_idx = 1.0, 1.0, 0.0, 0.0, [], []
            lo_eq = 1.0
            for kk in order:
                eb = entry_bar[kk]; still = []
                for (xb, rf, rr) in open_pos:
                    if xb <= eb:
                        eq *= (1.0 + rr); reserved -= rf
                        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0); lo_eq = min(lo_eq, eq)
                    else:
                        still.append((xb, rf, rr))
                open_pos = still
                if len(open_pos) >= SLOT_CAP or reserved + ladder > 1.0 + 1e-12:
                    continue
                open_pos.append((exit_bar[kk], ladder, acct[kk])); reserved += ladder
                taken_idx.append(kk)
            for (xb, rf, rr) in open_pos:
                eq *= (1.0 + rr); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1.0)
                lo_eq = min(lo_eq, eq)
            taken = len(taken_idx)
            per = acct[np.array(taken_idx)] if taken else np.array([0.0])
            g = float(np.log1p(np.clip(per, -0.999999, None)).mean())
            out["windows"][sp] = {
                "n_fires": int(len(ii)), "n_taken": int(taken),
                "take_rate": round(taken / len(ii), 4),
                "exp_pct_per_trade": round(float(per.mean() * 100), 4),
                "log_growth_per_trade": round(g, 6),
                "final_equity": round(float(eq), 6), "max_dd": round(float(mdd), 4),
                "min_equity": round(float(lo_eq), 6),
                "ruin_rate": round(float((oc[np.array(taken_idx)] == "liq").mean()), 5) if taken else None}
            w = out["windows"][sp]
            log(f"    {sp:5s} 체결 {w['n_taken']:5d}/{w['n_fires']:5d} ({w['take_rate']:.1%}) | "
                f"건당 {w['exp_pct_per_trade']:+.3f}% · **로그성장 {w['log_growth_per_trade']:+.5f}/건** | "
                f"최종 x{w['final_equity']:.4f} · MDD {w['max_dd']:.1%} · 전손 {w['ruin_rate']:.2%}")
        rows.append(out)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(
        {"holdout_touched": False, "n_passing": len(passing), "n_valid": len(valid),
         "slot_cap": SLOT_CAP, "model": "M1 reserved-ladder margin, compounding",
         "cells": rows}, ensure_ascii=False, indent=2))
    log(f"\n산출: {OUT}/report.json ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
