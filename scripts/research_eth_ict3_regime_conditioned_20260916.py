#!/usr/bin/env python3
"""ICT 3종을 레짐·변동성으로 깎으면 경제성이 나오는가 (2026-09-16, 같은 세션 2라운드).

사용자 *"이걸 잘 깎으면 경제성이 있을 것 같은데 레짐, 변동성 등으로 분류해서 테스트도 해봐"*.

1라운드(research_eth_ict_killzone_fvg_eqh_20260916)의 결론이 이 라운드의 가설을 정한다:
막고 있는 것은 정확도 a 가 아니라 **이긴 판의 크기 b** 다(적중 우위 +3~5pp·z 5.5 인데
평균이익 87 < 평균손실 108). 그러므로 물어야 할 것은 "어느 레짐에서 더 잘 맞는가"가 아니라

    ⭐ **어느 레짐에서 W/L 비대칭이 뒤집히는가** (손익분기적중 a* = (L+비용)/(W+L) 이 적중 아래로)

09-14 가 같은 종류의 질문에서 밟은 지뢰 2개를 설계에 미리 넣는다:
  · 조건부 셀의 귀무는 **그 조건으로 매칭**한다(비매칭이면 구간 표류를 신호로 오인 — 고ATR
    스윕 +8.97 → 매칭 후 +2.89). 여기서는 층 전체 봉의 gross 평균으로 **정확히** 계산한다.
  · 층 격자는 다중검정이다. **발동 인덱스 순환이동 귀무의 통과셀 수 95분위**가 유일한 방어.

배포 wide24 HMM 레짐은 쓸 수 없다 — 피쳐에 oi_change_rate 가 있고 바이낸스 5분 OI 는 최근
41.7시간뿐이다(reference_binance_futures_metrics_history_sources_20260909). OHLCV 파생으로 간다.

사전등록 판정(실행 전 고정):
  셀 PASS = 순손익(gross-5.52) > 0 AND 조건매칭 초과 > 0 AND |t_블록(매칭초과)| >= 2
            AND 전·후반 gross 부호 일치 AND 블록 n >= 50
  축 판정 = 층 간 단조성(레짐 효과라면 단조여야 한다)
  전체 판정 = 통과셀 수 > 순환이동 귀무 격자의 95분위

출력: tmp/eth_ict3_regime_20260916/*.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))

from research_eth_ict_killzone_fvg_eqh_20260916 import (  # noqa: E402  식 두 벌 금지
    gross_bp, load, CSV, BTC, day_boot, bh_fdr, prior_day_levels, level_reclaim,
    fvg_touch, equal_level_count, EQ_TOL_ATR, COST_PEG)

OUT = ROOT / "tmp/eth_ict3_regime_20260916"
HORIZONS = [12, 48, 144]
WARMUP = 900
RANKWIN = 288          # 라이브 dalton_atr_pctile 규약(rolling 288) 그대로
NSHIFT = 200
RNG = np.random.default_rng(20260916)


def tercile(x: pd.Series) -> np.ndarray:
    """rolling 288 분위 -> 0/1/2 층, NaN 은 -1. 현재봉은 이미 마감된 봉이라 인과적이다."""
    r = x.rolling(RANKWIN, min_periods=RANKWIN // 2).rank(pct=True).to_numpy()
    out = np.full(len(r), -1, np.int8)
    out[r <= 1 / 3] = 0; out[(r > 1 / 3) & (r < 2 / 3)] = 1; out[r >= 2 / 3] = 2
    return out


def grid_thin(idx: np.ndarray, lo: int, H: int) -> np.ndarray:
    """고정격자 비겹침: 보유기간 H 길이의 블록당 첫 건만. greedy 와 목적 동일하되 벡터화된다."""
    if len(idx) == 0: return idx
    b = (idx - lo) // H
    _, first = np.unique(b, return_index=True)
    return idx[np.sort(first)]


def cell_stats(g: np.ndarray, nul: float, idx: np.ndarray, lo: int, H: int,
               ts_half: np.datetime64, ts: np.ndarray) -> dict:
    kept = grid_thin(idx, lo, H)
    pos = np.searchsorted(idx, kept)
    ex = g[pos] - nul
    t = float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))) if len(ex) > 2 else np.nan
    w = float(g[g > 0].mean()) if (g > 0).any() else np.nan
    l = float(-g[g <= 0].mean()) if (g <= 0).any() else np.nan
    a_need = (l + COST_PEG) / (w + l) if np.isfinite(w) and np.isfinite(l) else np.nan
    h1 = g[ts[idx] < ts_half]; h2 = g[ts[idx] >= ts_half]
    return dict(n=len(idx), n_block=len(kept), gross=float(g.mean()), null=nul,
                excess=float(g.mean()) - nul, t_block=t, net=float(g.mean()) - COST_PEG,
                hit=float((g > 0).mean()), win=w, loss=l, wl=w / l if l else np.nan,
                acc_need=a_need, room_pp=100 * (float((g > 0).mean()) - a_need),
                h1=float(h1.mean()) if len(h1) else np.nan,
                h2=float(h2.mean()) if len(h2) else np.nan)


def passes(r: dict) -> bool:
    return bool(r["n_block"] >= 50 and r["net"] > 0 and r["excess"] > 0
                and abs(r["t_block"]) >= 2 and np.sign(r["h1"]) == np.sign(r["h2"]))


def selftest() -> None:
    assert list(grid_thin(np.array([0, 5, 10, 48, 60, 96]), 0, 48)) == [0, 48, 96]
    assert list(grid_thin(np.array([47, 49]), 0, 48)) == [47, 49]     # 격자 경계는 갈린다
    t = tercile(pd.Series(np.arange(600, dtype=float)))
    assert t[-1] == 2 and (t[:RANKWIN // 2 - 1] == -1).all()
    # 조건매칭 귀무: 층 전체 평균을 빼면 층 안의 무작위 부분집합은 기대 0
    g = np.array([1., 2., 3., 4.]); assert abs((g - g.mean()).mean()) < 1e-12
    # passes: 전·후반 부호 불일치는 탈락
    base = dict(n_block=100, net=1.0, excess=1.0, t_block=3.0, h1=1.0, h2=1.0)
    assert passes(base) and not passes({**base, "h2": -1.0}) and not passes({**base, "net": -0.1})
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest: selftest(); return 0
    selftest(); OUT.mkdir(parents=True, exist_ok=True)
    import live_evidence_signal_dashboard_20260823 as EV  # noqa: E402
    from features.engineering import FeatureEngineer      # noqa: E402

    print("[1/5] 데이터·신호 …", flush=True)
    kl = load(CSV); btc = load(BTC) if BTC.exists() else None
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=None)
    ts_s = sig["timestamp"]; ts = ts_s.to_numpy()
    op, cl = sig["open"].to_numpy(float), sig["close"].to_numpy(float)
    hi, lo_ = sig["high"].to_numpy(float), sig["low"].to_numpy(float)
    atr = sig["atr_pct"].to_numpy(float)
    day = ts_s.dt.floor("D").astype("int64").to_numpy()
    n = len(sig); lo, hi_i = WARMUP, n - max(HORIZONS) - 2
    ts_half = ts[(lo + hi_i) // 2]
    span = (ts_s.iloc[hi_i] - ts_s.iloc[lo]).total_seconds() / 86400

    lv = prior_day_levels(sig)
    fvg_b, fvg_s = fvg_touch(hi, lo_, cl, atr, 0.5)
    W = EV.SWEEP_LOOKBACK
    lvl_hi = sig["high"].rolling(W, min_periods=W).max().shift(1).to_numpy(float)
    lvl_lo = sig["low"].rolling(W, min_periods=W).min().shift(1).to_numpy(float)
    tol = np.nan_to_num(atr) * EQ_TOL_ATR
    sw_b = sig["bottom_liquidity_sweep"].fillna(False).to_numpy(bool)
    sw_t = sig["top_liquidity_sweep"].fillna(False).to_numpy(bool)
    eq_b = sw_b & (equal_level_count(lo_, lvl_lo, tol, W, upper=False) >= 2)
    eq_t = sw_t & (equal_level_count(hi, lvl_hi, tol, W, upper=True) >= 2)
    ARMS = [("pivot_pdhl", level_reclaim(hi, lo_, cl, lv["pdl"], below=True),
                            level_reclaim(hi, lo_, cl, lv["pdh"], below=False)),
            ("sweep_plain", sw_b, sw_t),
            ("eqhl_2touch", eq_b, eq_t),
            ("fvg_ge05atr", fvg_b, fvg_s)]

    print("[2/5] 레짐·변동성 축 …", flush=True)
    fe = FeatureEngineer.__new__(FeatureEngineer)
    chop = fe._calc_chop(sig["high"], sig["low"], sig["close"], length=14)
    atr_s = pd.Series(atr)
    expand = atr_s / atr_s.rolling(RANKWIN, min_periods=RANKWIN // 2).mean()
    trend = sig["close"] / sig["close"].shift(RANKWIN) - 1.0
    AX = {"V_변동성": (tercile(atr_s), ["저변동", "중간", "고변동"]),
          "C_횡보도": (tercile(chop), ["추세장", "중간", "횡보장"]),
          "E_확장수축": (tercile(expand), ["수축", "중립", "확장"]),
          "D_추세정합": (tercile(trend), ["하락추세", "횡보", "상승추세"])}
    yr = ts_s.dt.year.to_numpy()
    AX["Y_연도"] = (np.where((yr >= 2022) & (yr <= 2026), yr - 2022, -1).astype(np.int8),
                   ["2022", "2023", "2024", "2025", "2026"])

    print("[3/5] 조건매칭 귀무 …", flush=True)
    # 층 전체 봉의 gross 평균 = 같은 층·같은 측면 무작위 진입의 기댓값 (정확, O(n))
    G = {(H, s): gross_bp(op, cl, np.arange(lo, hi_i + 1, dtype=np.int64), H, s) for H in HORIZONS for s in (True, False)}
    NUL: dict = {}
    for ax, (lay, names) in AX.items():
        sub = lay[lo:hi_i + 1]
        for H in HORIZONS:
            for s in (True, False):
                g = G[(H, s)]
                for k in range(len(names)):
                    m = sub == k
                    NUL[(ax, k, H, s)] = float(g[m].mean()) if m.any() else np.nan
                NUL[(ax, -9, H, s)] = float(g.mean())     # 전체(무조건) 기준선

    print("[4/5] 셀 …", flush=True)
    rows = []
    for arm, f_long, f_short in ARMS:
        for side, fire, long in (("bottom", f_long, True), ("top", f_short, False)):
            all_idx = np.flatnonzero(np.nan_to_num(fire).astype(bool))
            all_idx = all_idx[(all_idx >= lo) & (all_idx <= hi_i)]
            for H in HORIZONS:
                g_all = gross_bp(op, cl, all_idx, H, long)
                for ax, (lay, names) in AX.items():
                    for k, nm in enumerate(names):
                        sel = lay[all_idx] == k
                        if sel.sum() < 60: continue
                        idx = all_idx[sel]
                        r = cell_stats(g_all[sel], NUL[(ax, k, H, long)], idx, lo, H, ts_half, ts)
                        clo, chi, p = day_boot(g_all[sel] - NUL[(ax, k, H, long)], day[idx])
                        r.update(arm=arm, side=side, H=H, axis=ax, layer=nm, ci_lo=clo, ci_hi=chi, p=p)
                        r["PASS"] = passes(r)
                        rows.append(r)
    D = pd.DataFrame(rows)
    D["q"] = bh_fdr(D.p.to_numpy())
    for c in ("gross", "null", "excess", "net", "win", "loss", "wl", "room_pp", "h1", "h2",
              "ci_lo", "ci_hi", "t_block"):
        D[c] = D[c].round(2)
    D["hit"] = D.hit.round(4); D["acc_need"] = D.acc_need.round(4)
    D.to_csv(OUT / "cells.csv", index=False)
    obs = int(D.PASS.sum())

    print(f"[5/5] 순환이동 격자 귀무 {NSHIFT}회 …", flush=True)
    span_i = hi_i - lo + 1
    cnt = np.zeros(NSHIFT, int)
    for b in range(NSHIFT):
        off = int(RNG.integers(1, span_i))
        for arm, f_long, f_short in ARMS:
            for side, fire, long in (("bottom", f_long, True), ("top", f_short, False)):
                a0 = np.flatnonzero(np.nan_to_num(fire).astype(bool))
                a0 = a0[(a0 >= lo) & (a0 <= hi_i)]
                idxs = np.sort(lo + (a0 - lo + off) % span_i)
                for H in HORIZONS:
                    gg = gross_bp(op, cl, idxs, H, long)
                    for ax, (lay, names) in AX.items():
                        for k in range(len(names)):
                            sel = lay[idxs] == k
                            if sel.sum() < 60: continue
                            r = cell_stats(gg[sel], NUL[(ax, k, H, long)], idxs[sel], lo, H, ts_half, ts)
                            cnt[b] += passes(r)
        if (b + 1) % 50 == 0: print(f"    {b+1}/{NSHIFT}", flush=True)
    p95 = float(np.percentile(cnt, 95))
    pd.DataFrame({"pass_count": cnt}).to_csv(OUT / "shift_null.csv", index=False)

    print(f"\n{'='*150}")
    print(f"셀 {len(D)}개 · 평가 {span:.0f}일 · 비용선 {COST_PEG}bp(양다리 peg)")
    print(f"⭐ 통과셀 관측 {obs} vs 순환이동 귀무 평균 {cnt.mean():.1f} · 95분위 {p95:.1f} · "
          f"백분위 {(cnt < obs).mean()*100:.1f}%")
    print("=" * 150)
    top = D.nlargest(15, "room_pp")
    print("■ 여유pp(적중 − 손익분기적중) 상위 15")
    print(f"{'팔':<12}{'측':>4}{'H':>5}{'축':<10}{'층':<8}{'건수':>7}{'블록':>6}{'gross':>8}{'매칭귀무':>9}"
          f"{'초과':>8}{'순손익':>8}{'적중':>7}{'W':>7}{'L':>7}{'W/L':>6}{'a*':>7}{'여유pp':>7}{'t':>6}{'q':>7}")
    for r in top.itertuples():
        print(f"{r.arm:<12}{'바닥' if r.side=='bottom' else '천장':>4}{r.H:>5}{r.axis:<10}{r.layer:<8}"
              f"{r.n:>7}{r.n_block:>6}{r.gross:>8.2f}{r.null:>9.2f}{r.excess:>+8.2f}{r.net:>+8.2f}"
              f"{r.hit*100:>6.1f}%{r.win:>7.1f}{r.loss:>7.1f}{r.wl:>6.2f}{r.acc_need*100:>6.1f}%"
              f"{r.room_pp:>+7.2f}{r.t_block:>6.2f}{r.q:>7.3f}")
    print(f"\n■ 통과셀 {obs}개")
    for r in D[D.PASS].itertuples():
        print(f"  {r.arm} {r.side} H={r.H} {r.axis}/{r.layer}  n={r.n} 순손익 {r.net:+.2f} "
              f"초과 {r.excess:+.2f} t {r.t_block:.2f} 여유 {r.room_pp:+.2f}pp q={r.q:.3f}")
    print("\n■ W/L 비대칭이 레짐에 따라 움직이는가 (전 팔 평균, H=48)")
    d48 = D[D.H == 48]
    for ax in AX:
        s = d48[d48.axis == ax].groupby("layer", sort=False)[["wl", "hit", "acc_need", "room_pp"]].mean()
        print(f"  {ax}: " + " | ".join(f"{i} W/L {v.wl:.2f} 적중 {v.hit*100:.1f}% a* {v.acc_need*100:.1f}% "
                                       f"여유 {v.room_pp:+.1f}" for i, v in s.iterrows()))
    # ---- 왜 레짐으로 안 깎이는가: 무작위 진입 대비 gross 차이를 정확도/크기로 정확 분해 ----
    print("\n■ ⭐무작위 진입 대비 gross 차이의 정확 분해  Δ = Δa·(W_r+L_r) + [a·ΔW − (1−a)·ΔL]")
    dec = []
    for H in HORIZONS:
        for lgs, nm_s in ((True, "bottom"), (False, "top")):
            gr = G[(H, lgs)]
            a_r, W_r, L_r = float((gr > 0).mean()), float(gr[gr > 0].mean()), float(-gr[gr <= 0].mean())
            dec.append(dict(H=H, side=nm_s, arm="(무작위 대조)", n=len(gr), hit=a_r, win=W_r,
                            loss=L_r, wl=W_r / L_r, acc_term=0.0, size_term=0.0))
            for arm, f_long, f_short in ARMS:
                fire = f_long if lgs else f_short
                ii = np.flatnonzero(np.nan_to_num(fire).astype(bool))
                ii = ii[(ii >= lo) & (ii <= hi_i)]
                g = gross_bp(op, cl, ii, H, lgs)
                a, w, l = float((g > 0).mean()), float(g[g > 0].mean()), float(-g[g <= 0].mean())
                dec.append(dict(H=H, side=nm_s, arm=arm, n=len(ii), hit=a, win=w, loss=l, wl=w / l,
                                acc_term=(a - a_r) * (W_r + L_r),
                                size_term=a * (w - W_r) - (1 - a) * (l - L_r)))
    DEC = pd.DataFrame(dec).round(3); DEC.to_csv(OUT / "decomposition.csv", index=False)
    for r in DEC[DEC.H == 48].itertuples():
        print(f"  H=48 {r.side:>6} {r.arm:<14} n={r.n:>6}  W/L {r.wl:>5.2f}  "
              f"정확도기여 {r.acc_term:>+7.2f}  크기기여 {r.size_term:>+7.2f}  합 {r.acc_term+r.size_term:>+7.2f}")
    m = DEC[DEC.arm != "(무작위 대조)"]
    print(f"  ⇒ 전 H·전 팔: 정확도기여 평균 {m.acc_term.mean():+.2f}bp · 크기기여 평균 "
          f"{m.size_term.mean():+.2f}bp · W/L 이 무작위보다 낮은 셀 "
          f"{int((m.wl.to_numpy() < DEC[DEC.arm=='(무작위 대조)'].wl.mean()).sum())}/{len(m)}")
    print("=" * 150)
    print(json.dumps({"cells": len(D), "pass": obs, "null_p95": p95,
                      "pct": round(float((cnt < obs).mean()) * 100, 1)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
