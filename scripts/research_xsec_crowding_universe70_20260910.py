#!/usr/bin/env python3
"""쏠림 페이드 — **모집단 60 vs 70 vs 83종** 비교 (2026-09-10).

사용자: *"우선 상위 10종으로 진행해줘"*.

## ⭐사전등록: 기대 방향을 결과 전에 못박는다
확대의 목적은 검정력이 아니라 **모집단 선정의 미래참조 제거**다.
기존 60종은 **2026년 현재 유동성**으로 골랐고 그건 2024년엔 알 수 없는 정보다.
⇒ **성과가 내려가는 것이 정상이고, 내려간 값이 더 정직한 값이다.**
   올라가면 오히려 설명이 필요하다(신규 종목이 우연히 좋았던 것인지 확인해야 한다).
결과를 보고 해석을 고르지 않기 위해 여기 적어둔다.

## 🔴1차 확대는 생존편향을 못 고친다 (자기정정)
현존 61~70위 10종을 더한 것으로는 생존편향이 안 잡힌다 — **폐지 종목은 현재 상장 목록에 없어
후보에 들어오지도 않기 때문**이다. 아카이브를 훑어 창 안에서 사라진 11종 + 후속 티커 2종을 더했다.
거기 **MATIC(→POL) · RNDR(→RENDER) · EOS** 가 있고, 이들은 2024년 상위권인데 패널에 **통째로 없었다**.
60종 패널이 사실상 *"전 기간 동일 티커로 연속 데이터가 있는 종목"* 으로 걸러져 티커 변경 자산이 탈락한 것이다.
⇒ **83종 팔에서는 성과 하락을 실제로 예상한다.** MATIC 은 2024년 내내 유동성 상위 40위라 선택됐을 종목이다.

## ⭐선행 진단: 추가 종목이 **실제로 선택되는가**
안 뽑히면 확대는 무효과이고 두 패널 비교는 같은 것을 두 번 재는 셈이다.
그래서 성과보다 **선택 빈도를 먼저** 낸다. 이게 0에 가까우면 나머지 숫자를 해석하지 않는다.

규약은 09-08 헤드라인 그대로: 1일 보유 · 시가 진입/청산 · NU 유동성 상위 · k 롱/숏 · 표본외 비겹침.
출력 tmp/xsec_universe70_20260910/report.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P60 = Path("/home/kbj20/crypto-scalping/tmp/xsec_perp_screen_20260908")
P70 = ROOT / "tmp/xsec_perp_screen_70_20260910"
P83 = ROOT / "tmp/xsec_perp_screen_full_20260910"
OUT = ROOT / "tmp/xsec_universe70_20260910"
H, LIQW, FEE_RT = 288, 288, 10.0
NU_GRID, K_GRID = (30, 40, 50), (3, 5)
OOS_A, OOS_B = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-07-31 23:59:59")
BOOT, SEED = 4000, 20260910


def log(m):
    print(f"[u70 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def block_ci(x, block=5, b=BOOT, seed=SEED):
    if len(x) < 20:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed); n = len(x); o = []
    for _ in range(b):
        st = rng.integers(0, n, int(np.ceil(n / block)))
        o.append(x[np.concatenate([np.arange(s, s + block) % n for s in st])[:n]].mean())
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


def run(pdir: Path, tag: str, new_syms: set, rep: dict, window: str = "OOS"):
    """window: OOS = 2025-09~2026-07 · IS = 그 이전.
    ⚠️**폐지 검정은 IS 에서만 성립한다** — 폐지가 전부 2024년에 일어나
    MATIC·RNDR·GAL 등은 OOS 창에 아예 존재하지 않는다."""
    z = np.load(pdir / "panel.npz", allow_pickle=True)
    ts = pd.DatetimeIndex(z["ts"]); O = z["O"].astype(np.float64); Q = z["Q"]
    syms = [str(s) for s in z["syms"]]
    RAW = np.load(pdir / "metrics_panel.npz")["count_toptrader_long_short_ratio"]
    S = np.where(RAW > 0, np.log(np.maximum(RAW, 1e-9)), np.nan)
    assert np.nanstd(S) > 0.01, "신호가 상수/불리언 — := 우선순위 함정"
    Qr = pd.DataFrame(Q).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    fwd = np.full_like(O, np.nan); fwd[:-(H + 1)] = O[H + 1:] / O[1:-H] - 1.0
    tid = np.arange(LIQW + 1, len(ts) - H - 2, H)
    tt = ts[tid]
    oos = ((tt >= OOS_A) & (tt <= OOS_B)) if window == "OOS" else (tt < OOS_A)
    is_new = np.array([s in new_syms for s in syms])
    log(f"[{tag}] {O.shape} · 종목 {len(syms)} (추가 {int(is_new.sum())}) · {window} 시점 {int(oos.sum())}")

    for NU in NU_GRID:
        el = (liq[tid] < NU) & np.isfinite(S[tid]) & np.isfinite(fwd[tid])
        sa = np.where(el, S[tid], np.nan); F = np.where(el, fwd[tid], np.nan)
        nval = np.isfinite(sa).sum(1)
        order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), 1)
        # 신규 종목이 **선택 가능 모집단**에 드는 비율(유동성 컷 통과)
        elig_new = float(el[:, is_new].mean()) if is_new.any() else 0.0
        for k in K_GRID:
            rr = np.flatnonzero((nval >= 2 * k + 2) & oos)
            if len(rr) < 50:
                continue
            lo_i = order[rr][:, :k]
            hi_i = order[rr][np.arange(len(rr))[:, None],
                             (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
            g = (np.nanmean(np.take_along_axis(F[rr], lo_i, 1), 1)
                 - np.nanmean(np.take_along_axis(F[rr], hi_i, 1), 1)) / 2 * 1e4
            m = np.isfinite(g); g = g[m]
            if len(g) < 50:
                continue
            picked = np.concatenate([lo_i, hi_i], axis=1)[m]
            new_pick = float(is_new[picked].mean()) if is_new.any() else 0.0
            net = g - FEE_RT
            rep[f"{tag}|NU{NU}|k{k}"] = {
                "n": int(len(g)), "gross_bp": float(g.mean()), "net_bp": float(net.mean()),
                "net_ci95": block_ci(net), "gross_ci95": block_ci(g),
                "top5_removed_gross": float(g[g < np.percentile(g, 95)].mean()),
                "new_sym_eligible_share": elig_new, "new_sym_picked_share": new_pick}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s60 = set(str(s) for s in np.load(P60 / "panel.npz", allow_pickle=True)["syms"])
    s70 = set(str(s) for s in np.load(P70 / "panel.npz", allow_pickle=True)["syms"])
    new = s70 - s60
    log(f"신규 {len(new)}종: {' '.join(sorted(new))}")
    rep = {"prereg": "확대 목적 = 모집단 선정 미래참조 제거. **성과 하락이 정상**", "new_syms": sorted(new),
           "cells": {}}
    s83 = set(str(x) for x in np.load(P83 / "panel.npz", allow_pickle=True)["syms"])
    delisted = s83 - s70
    rep["delisted_syms"] = sorted(delisted)
    log(f"🔴폐지/후속 {len(delisted)}종: {' '.join(sorted(delisted))}")
    run(P60, "u60", new, rep["cells"])
    run(P70, "u70", new, rep["cells"])
    run(P83, "u83", delisted, rep["cells"])   # ⭐신규 표시는 **폐지 종목** 기준
    # ⭐진짜 생존편향 검정 — 폐지 종목이 살아 있던 IS 창에서
    run(P60, "is60", delisted, rep["cells"], window="IS")
    run(P83, "is83", delisted, rep["cells"], window="IS")

    log("=" * 104)
    log("⭐선행 진단 — 추가 종목이 실제로 뽑히는가 (u70=현존 하위10 · u83=폐지13)")
    for key, c in rep["cells"].items():
        if key.startswith("u70") or key.startswith("u83"):
            log(f"  {key:14s} 유동성컷 통과 {c['new_sym_eligible_share']:6.2%} · "
                f"**선택 비중 {c['new_sym_picked_share']:6.2%}**")
    log("=" * 104)
    log(f"{'구성':>10} {'n':>4} {'60종':>9} {'70종':>9} {'83종':>9} {'83−60':>8} "
        f"{'83종 순 CI95':>22} {'83상5%제거':>10}")
    for NU in NU_GRID:
        for k in K_GRID:
            a = rep["cells"].get(f"u60|NU{NU}|k{k}"); b = rep["cells"].get(f"u70|NU{NU}|k{k}")
            c = rep["cells"].get(f"u83|NU{NU}|k{k}")
            if not (a and b and c):
                continue
            log(f"{'NU'+str(NU)+' k'+str(k):>10} {c['n']:>4} {a['gross_bp']:>+9.1f} "
                f"{b['gross_bp']:>+9.1f} {c['gross_bp']:>+9.1f} {c['gross_bp']-a['gross_bp']:>+8.1f} "
                f"[{c['net_ci95'][0]:>+7.1f},{c['net_ci95'][1]:>+7.1f}] {c['top5_removed_gross']:>+10.1f}")
    log("=" * 104)
    log("⭐⭐진짜 생존편향 검정 — 폐지 종목이 살아 있던 **표본내(2024~2025-08)** 창")
    log(f"{'구성':>10} {'n':>4} {'60종':>9} {'83종':>9} {'차이':>8} {'폐지종목 선택비중':>16}")
    for NU in NU_GRID:
        for k in K_GRID:
            a = rep["cells"].get(f"is60|NU{NU}|k{k}"); b = rep["cells"].get(f"is83|NU{NU}|k{k}")
            if not (a and b):
                continue
            log(f"{'NU'+str(NU)+' k'+str(k):>10} {b['n']:>4} {a['gross_bp']:>+9.1f} "
                f"{b['gross_bp']:>+9.1f} {b['gross_bp']-a['gross_bp']:>+8.1f} "
                f"{b['new_sym_picked_share']:>15.2%}")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
