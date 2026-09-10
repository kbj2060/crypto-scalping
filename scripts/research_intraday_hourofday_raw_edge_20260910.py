#!/usr/bin/env python3
"""**시각대별 일중 모멘텀/되돌림** 원시 엣지 사전점검 (2026-09-10). 모델 없음. 지평 1시간.

사용자: *"주간 말고 1시간 정도면 돼. 다시 증거신호 후보를 가져와."*
근거 문헌: NAJEF 2022 「Intraday return predictability in the cryptocurrency markets: Momentum, reversal, or both」
`10.1016/j.najef.2022.101733`(피인용 33). ⚠️초록을 Crossref·S2 어디서도 못 받아 **제목까지만 확인**했다 —
따라서 방향(모멘텀이냐 되돌림이냐)을 문헌으로 고정할 수 없어 **양방향을 대칭으로 검정**하고
다중검정 보정을 그만큼 강하게 건다.

## 왜 이 후보인가 (제외 목록 통과)
현행 8종은 전부 **가격·오실레이터·플로우 극단값**이고 **시각 구조를 트리거로 쓰지 않는다**
(`hour_utc`/`weekday` 는 피쳐로만 있고, orthogonal_combo ablation 에서는 오히려 **빼는 게 OOS 개선**이었다).
2026-09-02 기각 5종(Lee-Mykland·VPIN·Corwin-Schultz·라운드넘버·복합AND)과도 겹치지 않는다.
비추천 목록의 Quarter-Hour Effect 는 **10초 해상도 알고리즘 주기성**이라 다른 것이다.
데이터는 klines 만 필요하고, 시간봉 관측이 **23,000개** 라 주간 횡단면을 막았던 검정력 문제가 없다.

## 사전등록 (결과 보기 전 고정)
신호   시각 h 에서 직전 L시간 수익 `r(h-L..h)` 의 부호/크기
타깃   다음 1시간 수익 `r(h..h+1)`
격자   L ∈ {1,2,3,6} × 시각 h ∈ 0..23 × 문턱 {상위/하위 20%, 10%} = **192셀**
방향   문헌 미확인이라 **모멘텀·되돌림 양방향 모두** 보고. 어느 한쪽을 고른 뒤 그것만 보고하지 않는다.
귀무   ① 순환이동(군집·시각구조 보존) ② **격자 통과수 귀무**(무작위 부분표집 B=200) — 192셀 다중검정 보정
비용   10bp 테이커 왕복
1차 기준 **격자 통과수가 무작위 귀무의 95분위를 넘고**, 통과 셀의 순수익(비용 후)이 양수
출력 tmp/intraday_hod_20260910/report.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_exit_synth_dataset_20260910 as BD  # noqa: E402

OUT = ROOT / "tmp/intraday_hod_20260910"
LOOKBACKS = [1, 2, 3, 6]
QS = [0.10, 0.20]
COST_BP = 10.0
B_CYC = 400
B_GRID = 200
SPLIT = pd.Timestamp("2025-09-01")


def log(m):
    print(f"[hod {time.strftime('%H:%M:%S')}] {m}", flush=True)


def hourly(sym_short: str, sym: str):
    kl = BD.load_klines(sym_short, sym)
    t = pd.DatetimeIndex(kl["timestamp"])
    g = pd.Series(kl["close"].to_numpy(float), index=t).resample("1h").last().dropna()
    return g


def two_sided(tgt: np.ndarray, hi: np.ndarray, lo: np.ndarray) -> float:
    """양측 평균 bp. 상승분위 롱 − 하락분위 롱 = 모멘텀 팔. 되돌림은 정확히 부호 반전."""
    return float((tgt[hi].mean() - tgt[lo].mean()) / 2.0 * 1e4)


def cyc_null(tgt: np.ndarray, hi: np.ndarray, lo: np.ndarray, b=B_CYC, seed=0):
    """순환이동 귀무: **전체 수익 배열은 온전**하고 발동 인덱스만 같은 양 s 만큼 민다.
    ⚠️두 팔에 **같은 s** 를 써야 팔 간 시점 간격(=시각 구조)이 보존된다."""
    n = len(tgt); rng = np.random.default_rng(seed); o = []
    for s_ in rng.integers(1, n, b):
        o.append(two_sided(tgt, (hi + s_) % n, (lo + s_) % n))
    return np.array(o)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    px = hourly("eth", "ETHUSDT")
    tgt = px.pct_change().shift(-1).to_numpy()        # 다음 1시간 수익(타깃)
    hours = px.index.hour.to_numpy()
    fin = np.isfinite(tgt)
    tgt_f = np.where(fin, tgt, 0.0)                   # 순환이동용(비유한은 0, 발동은 유한만 씀)
    log(f"시간봉 {len(px):,} · {px.index[0]} → {px.index[-1]}")
    rep = {"lookbacks": LOOKBACKS, "qs": QS, "cost_bp": COST_BP,
           "note": "문헌 초록 미확보 → 모멘텀·되돌림 대칭(rev = −mom) + 격자 통과수 귀무",
           "n_hours": int(len(px)), "cells": {}, "summary": {}}
    is_m = np.asarray(px.index < SPLIT)
    n_pass = 0
    cells_hi_lo = []                                   # 격자 귀무 재사용
    for L in LOOKBACKS:
        past = (px / px.shift(L) - 1.0).to_numpy()
        for h in range(24):
            hm = (hours == h) & np.isfinite(past) & fin
            if hm.sum() < 200:
                continue
            base = np.flatnonzero(hm); pv = past[hm]
            for q in QS:
                hi = base[pv >= np.quantile(pv, 1 - q)]
                lo = base[pv <= np.quantile(pv, q)]
                if min(len(hi), len(lo)) < 60:
                    continue
                cells_hi_lo.append((hi, lo))
                obs = two_sided(tgt_f, hi, lo)         # 모멘텀 팔. 되돌림은 −obs
                nulls = cyc_null(tgt_f, hi, lo, seed=(L * 100 + h) * 10 + int(q * 100))
                nm = float(nulls.mean())
                p_hi = float(np.percentile(nulls, 97.5)); p_lo = float(np.percentile(nulls, 2.5))
                mom_ok, rev_ok = bool(obs > p_hi), bool(obs < p_lo)
                n_pass += int(mom_ok) + int(rev_ok)
                hi_i, lo_i = hi[is_m[hi]], lo[is_m[lo]]
                hi_o, lo_o = hi[~is_m[hi]], lo[~is_m[lo]]
                rep["cells"][f"L{L}|h{h:02d}|q{int(q*100)}"] = {
                    "n_hi": int(len(hi)), "n_lo": int(len(lo)),
                    "mom_bp": obs, "rev_bp": -obs,
                    "excess_bp": obs - nm, "null_mean_bp": nm, "null_ci95": [p_lo, p_hi],
                    "mom_beats_null": mom_ok, "rev_beats_null": rev_ok,
                    "best_net_bp": max(abs(obs - nm) - COST_BP, -999.0) if (mom_ok or rev_ok)
                                   else abs(obs - nm) - COST_BP,
                    "is_mom_bp": two_sided(tgt_f, hi_i, lo_i) if min(len(hi_i), len(lo_i)) >= 30 else None,
                    "oos_mom_bp": two_sided(tgt_f, hi_o, lo_o) if min(len(hi_o), len(lo_o)) >= 30 else None}
    n_cells = len(rep["cells"])
    # 격자 통과수 귀무: 같은 셀 구조에 **무작위 순환이동**을 걸어 통과수 분포를 만든다
    rng = np.random.default_rng(7)
    grid_null = []
    for _ in range(B_GRID):
        sh = int(rng.integers(1, len(tgt_f)))
        cnt = 0
        for hi, lo in cells_hi_lo:
            hi2, lo2 = (hi + sh) % len(tgt_f), (lo + sh) % len(tgt_f)
            o = two_sided(tgt_f, hi2, lo2)
            nl = cyc_null(tgt_f, hi2, lo2, b=120, seed=int(rng.integers(1e6)))
            cnt += int(o > np.percentile(nl, 97.5)) + int(o < np.percentile(nl, 2.5))
        grid_null.append(cnt)
    p95 = float(np.percentile(grid_null, 95))
    stable = sum(1 for v in rep["cells"].values()
                 if v["is_mom_bp"] is not None and v["oos_mom_bp"] is not None
                 and np.sign(v["is_mom_bp"]) == np.sign(v["oos_mom_bp"]))
    pc = [v for v in rep["cells"].values() if (v["mom_beats_null"] or v["rev_beats_null"])
          and v["is_mom_bp"] is not None and v["oos_mom_bp"] is not None]
    pc_agree = sum(1 for v in pc if np.sign(v["is_mom_bp"]) == np.sign(v["oos_mom_bp"]))
    rep["summary"] = {"n_cells": n_cells, "pass_total": n_pass,
                      "passing_cells_with_both_windows": len(pc),
                      "passing_cells_is_oos_sign_agree": pc_agree,
                      "grid_null_mean": float(np.mean(grid_null)), "grid_null_p95": p95,
                      "grid_beats_null": bool(n_pass > p95),
                      "cells_net_pos": sum(abs(v["excess_bp"]) > COST_BP for v in rep["cells"].values()),
                      "cells_is_oos_sign_agree": stable}
    log(f"셀 {n_cells} · 귀무 통과(양방향) {n_pass} (우연 기대 {0.05*n_cells:.1f})")
    log(f"격자 통과수 귀무: 평균 {np.mean(grid_null):.1f} · 95분위 {p95:.1f} "
        f"→ {'초과' if rep['summary']['grid_beats_null'] else '미달'}")
    log(f"|초과|>10bp {rep['summary']['cells_net_pos']}/{n_cells} · IS/OOS 부호일치 전체 {stable}/{n_cells} "
        f"· **통과셀 {pc_agree}/{len(pc)}**")
    top = sorted(rep["cells"].items(), key=lambda kv: -abs(kv[1]["excess_bp"]))[:8]
    for k, v in top:
        io_ = (f"IS {v['is_mom_bp']:+6.1f} OOS {v['oos_mom_bp']:+6.1f}"
               if v["is_mom_bp"] is not None and v["oos_mom_bp"] is not None else "IS/OOS 표본부족")
        log(f"  {k:16s} n={v['n_hi']:4d}/{v['n_lo']:4d} 모멘텀 {v['mom_bp']:+7.1f} 초과 {v['excess_bp']:+7.1f} "
            f"귀무CI[{v['null_ci95'][0]:+6.1f},{v['null_ci95'][1]:+6.1f}] "
            f"{'모멘텀통과' if v['mom_beats_null'] else ('되돌림통과' if v['rev_beats_null'] else '—')} {io_}")
    # ── 통과 셀 강건성: 윈저10 · 블록부트 CI (저장소 반복 함정 = 손익이 꼬리에 몰림) ──
    for k, v in rep["cells"].items():
        if not (v["mom_beats_null"] or v["rev_beats_null"]):
            continue
        L, h, q = int(k[1:k.index("|")]), int(k.split("|")[1][1:]), int(k.split("|")[2][1:]) / 100
        past = (px / px.shift(L) - 1.0).to_numpy()
        hm = (hours == h) & np.isfinite(past) & fin
        base = np.flatnonzero(hm); pv = past[hm]
        hi = base[pv >= np.quantile(pv, 1 - q)]; lo = base[pv <= np.quantile(pv, q)]
        pnl = np.concatenate([tgt_f[hi], -tgt_f[lo]]) * 1e4      # 팔별 부호를 맞춘 건당 손익
        w = np.clip(pnl, *np.percentile(pnl, [10, 90]))
        rng2 = np.random.default_rng(11); n = len(pnl); bs = []
        for _ in range(2000):                                     # 블록 8건(≈8거래일 간격 군집)
            st = rng2.integers(0, n, int(np.ceil(n / 8)))
            bs.append(pnl[np.concatenate([np.arange(x, x + 8) % n for x in st])[:n]].mean())
        v["wins10_bp"] = float(w.mean()); v["boot_ci95"] = [float(np.percentile(bs, 2.5)),
                                                            float(np.percentile(bs, 97.5))]
        v["ci_excludes_zero"] = bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0)
        v["tail_share"] = float(1 - abs(w.mean()) / max(abs(pnl.mean()), 1e-9))
    pcs = [v for v in rep["cells"].values() if "boot_ci95" in v]
    rep["summary"]["passing_ci_excludes_zero"] = sum(v["ci_excludes_zero"] for v in pcs)
    rep["summary"]["passing_wins10_keeps_sign"] = sum(
        1 for v in pcs if np.sign(v["wins10_bp"]) == np.sign(v["mom_bp"]) and abs(v["wins10_bp"]) > COST_BP)
    log(f"통과셀 {len(pcs)} · 블록부트 CI 0제외 {rep['summary']['passing_ci_excludes_zero']} "
        f"· 윈저10 후에도 부호유지&>10bp {rep['summary']['passing_wins10_keeps_sign']}")
    for k, v in sorted(((k, v) for k, v in rep["cells"].items() if "boot_ci95" in v),
                       key=lambda kv: -abs(kv[1]["excess_bp"]))[:5]:
        log(f"  {k:16s} 초과 {v['excess_bp']:+6.1f} 윈저10 {v['wins10_bp']:+6.1f} "
            f"꼬리비중 {v['tail_share']:+.2f} CI[{v['boot_ci95'][0]:+6.1f},{v['boot_ci95'][1]:+6.1f}]")

    # ── 대조군: 격자가 쪼갠 두 축(시각 · 직전수익) 중 무엇이 일하는가 ─────────────
    # 통과가 "시각 구조"가 아니라 "직전수익 조건" 때문이면 시각을 빼도 그대로 나온다.
    ctrl = {}
    for L in LOOKBACKS:
        past = (px / px.shift(L) - 1.0).to_numpy()
        pm = np.isfinite(past) & fin
        base = np.flatnonzero(pm); pv = past[pm]
        for q in QS:                                    # A) 시각 무시, 직전수익만
            hi = base[pv >= np.quantile(pv, 1 - q)]; lo = base[pv <= np.quantile(pv, q)]
            o = two_sided(tgt_f, hi, lo)
            nl = cyc_null(tgt_f, hi, lo, seed=L * 7 + int(q * 100))
            ctrl[f"pooled|L{L}|q{int(q*100)}"] = {
                "n_hi": int(len(hi)), "mom_bp": o, "excess_bp": o - float(nl.mean()),
                "beats": bool(o > np.percentile(nl, 97.5) or o < np.percentile(nl, 2.5))}
    for h in range(24):                                 # B) 직전수익 무시, 시각만(방향 롱)
        m = np.flatnonzero((hours == h) & fin)
        if len(m) < 200:
            continue
        o = float(tgt_f[m].mean() * 1e4)
        nl = np.array([float(tgt_f[(m + int(x)) % len(tgt_f)].mean() * 1e4)
                       for x in np.random.default_rng(h).integers(1, len(tgt_f), B_CYC)])
        ctrl[f"houronly|h{h:02d}"] = {
            "n": int(len(m)), "long_bp": o, "excess_bp": o - float(nl.mean()),
            "beats": bool(o > np.percentile(nl, 97.5) or o < np.percentile(nl, 2.5))}
    rep["controls"] = ctrl
    pooled = {k: v for k, v in ctrl.items() if k.startswith("pooled")}
    honly = {k: v for k, v in ctrl.items() if k.startswith("houronly")}
    rep["summary"]["ctrl_pooled_beats"] = sum(v["beats"] for v in pooled.values())
    rep["summary"]["ctrl_houronly_beats"] = sum(v["beats"] for v in honly.values())
    log(f"대조군 A 시각무시(직전수익만) 통과 {rep['summary']['ctrl_pooled_beats']}/{len(pooled)} "
        f"· 최대 |초과| {max(abs(v['excess_bp']) for v in pooled.values()):.1f}bp")
    log(f"대조군 B 직전수익무시(시각만) 통과 {rep['summary']['ctrl_houronly_beats']}/{len(honly)} "
        f"(우연 {0.05*len(honly):.1f}) · 최대 |초과| {max(abs(v['excess_bp']) for v in honly.values()):.1f}bp")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
