#!/usr/bin/env python3
"""pivot_pdhl(전일 고저 되찾기) 단독 심층 — ICT 3종 중 유일한 생존 후보 (2026-09-16, 4라운드).

사용자 *"pivot_pdhl 만 따로 더 파봐"*.

왜 이것만 남았나: 3라운드 오라클 검정에서 **거울상이 아닌 유일한 팔**이었다(3개 H 전부 양측
동부호). 1라운드에서는 적중 우위 +5.84pp(z 3.70)로 전 팔 최고였다.
🔴그런데 1라운드 표 자체가 이미 경고를 하고 있다 — **8셀 중 6셀이 전반 양수 → 후반 음수**.

사전등록 6검정(실행 전 고정, 변형 탐색 금지):
  A 안정성   전·후반 / 연도별 / 롤링 1년      — 통과: 전후반 부호일치 AND 연도 4/5 동부호
  B 정의절제 되찾기가 일을 하는가(6변형)      — 통과: D1(현행) > D0(터치만) AND D1 > D3(되찾기실패)
  C 거리매칭 "전일 저가 근처" 표류 제거        — 통과: 매칭 초과 CI95 0 배제
  D 독립일수 고유 발동일 · 일블록 t            — 보고(판정 아님, 검정력 진술용)
  E TP/SL   W/L 을 고칠 유일한 레버(12조합)   — 통과: 순손익>0 AND 전후반 부호일치
  F 타자산  BTC/SOL/XRP 에 정의 그대로 이식    — 통과: 3/3 부호일치 (⚠️상관자산은 완전 표본외 아님)

배리어 판정은 라이브 컨벤션대로 **intrabar 고가/저가**, 한 봉에서 TP·SL 동시 터치는 **SL 우선**
(보수적). 진입 open[i+1], 시간상한 H.

출력: tmp/eth_pivot_pdhl_20260916/*.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))

from research_eth_ict_killzone_fvg_eqh_20260916 import (  # noqa: E402  식 두 벌 금지
    gross_bp, load, day_boot, prior_day_levels, level_reclaim, COST_PEG)
from research_eth_ict3_regime_conditioned_20260916 import grid_thin, tercile  # noqa: E402

DATA = ROOT / "binance_data/klines"
OUT = ROOT / "tmp/eth_pivot_pdhl_20260916"
H_MAIN = 48
HORIZONS = [12, 48, 144]
WARMUP = 900
DEEP = 0.25            # 깊은/얕은 관통 경계 (x atr_pct)
TPS = [0.5, 1.0, 1.5, 2.0]
SLS = [0.5, 1.0, 1.5]


def variants(hi, lo_, cl, atr, pdl, pdh):
    """사전등록 6변형. 바텀(롱)/탑(숏) 쌍으로 돌려준다."""
    prev = np.roll(cl, 1); prev[0] = np.nan
    pierce_lo = (pdl - lo_) / np.maximum(cl, 1e-12)      # 아래로 관통한 깊이(비율)
    pierce_hi = (hi - pdh) / np.maximum(cl, 1e-12)
    d1_b = (lo_ <= pdl) & (cl > pdl) & (prev > pdl)      # 현행 정의
    d1_t = (hi >= pdh) & (cl < pdh) & (prev < pdh)
    deep_b, deep_t = pierce_lo >= DEEP * atr, pierce_hi >= DEEP * atr
    return {
        "D0_터치만":       ((lo_ <= pdl) & (prev > pdl), (hi >= pdh) & (prev < pdh)),
        "D1_되찾기(현행)": (d1_b, d1_t),
        "D2_접근무시":     ((lo_ <= pdl) & (cl > pdl), (hi >= pdh) & (cl < pdh)),
        "D3_되찾기실패":   ((lo_ <= pdl) & (cl <= pdl) & (prev > pdl),
                            (hi >= pdh) & (cl >= pdh) & (prev < pdh)),
        "D4_깊은되찾기":   (d1_b & deep_b, d1_t & deep_t),
        "D5_얕은되찾기":   (d1_b & ~deep_b, d1_t & ~deep_t),
    }


def barrier_pnl(op, hi, lo_, cl, idx, H, tp, sl, long: bool) -> np.ndarray:
    """intrabar TP/SL + 시간상한. 한 봉에서 둘 다 닿으면 SL 우선(보수적). bp."""
    e = op[idx + 1]
    out = np.empty(len(idx))
    for k, i in enumerate(idx):
        s, t = i + 1, min(i + H, len(cl) - 1)
        up = (hi[s:t + 1] - e[k]) / e[k]
        dn = (lo_[s:t + 1] - e[k]) / e[k]
        fav, adv = (up, dn) if long else (-dn, -up)
        h_sl = np.flatnonzero(adv <= -sl)
        h_tp = np.flatnonzero(fav >= tp)
        j_sl = h_sl[0] if len(h_sl) else 10**9
        j_tp = h_tp[0] if len(h_tp) else 10**9
        if j_sl <= j_tp and j_sl < 10**9:   out[k] = -sl
        elif j_tp < 10**9:                  out[k] = tp
        else:
            r = (cl[t] - e[k]) / e[k]
            out[k] = r if long else -r
    return out * 1e4


def dist_matched_null(g_all: np.ndarray, dist: np.ndarray, valid: np.ndarray,
                      idx: np.ndarray, nq: int = 20) -> float:
    """«전일 저가/고가에서 얼마나 가까운가» 분위로 층화 매칭한 귀무 gross.
    09-14 규율: 조건부 셀의 귀무는 그 조건으로 매칭한다."""
    d = dist[valid]
    ed = np.quantile(d[np.isfinite(d)], np.linspace(0, 1, nq + 1)[1:-1])
    qb = np.full(len(dist), -1, np.int16); qb[valid] = np.searchsorted(ed, dist[valid])
    bm = np.bincount(qb[valid], weights=g_all, minlength=nq) / \
        np.maximum(np.bincount(qb[valid], minlength=nq), 1)
    w = np.bincount(qb[idx], minlength=nq).astype(float)
    return float((bm * w).sum() / w.sum())


def block_t(g: np.ndarray, idx: np.ndarray, lo: int, H: int, nul: float) -> float:
    kept = grid_thin(idx, lo, H)
    d = g[np.searchsorted(idx, kept)] - nul
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 else np.nan


def selftest() -> None:
    # barrier_pnl: TP 먼저 닿으면 +tp, SL 먼저면 -sl, 동시봉은 SL 우선
    op = np.array([0., 100., 0., 0., 0.]); cl = np.array([0., 100., 100., 100., 100.])
    hi = np.array([0., 100., 105., 100., 100.]); lo = np.array([0., 100., 100., 100., 100.])
    assert abs(barrier_pnl(op, hi, lo, cl, np.array([0]), 3, 0.04, 0.02, True)[0] - 400) < 1e-9
    lo2 = np.array([0., 100., 97., 100., 100.])
    assert abs(barrier_pnl(op, hi, lo2, cl, np.array([0]), 3, 0.04, 0.02, True)[0] + 200) < 1e-9
    # 아무것도 안 닿으면 시간청산(종가)
    hi3 = np.full(5, 100.); lo3 = np.full(5, 100.)
    assert abs(barrier_pnl(op, hi3, lo3, cl, np.array([0]), 3, 0.04, 0.02, True)[0]) < 1e-9
    # dist_matched_null: 층이 하나뿐이면 그 층 평균
    g = np.array([1., 3., 5., 7.]); dist = np.array([1., 1., 1., 1.]); v = np.ones(4, bool)
    assert abs(dist_matched_null(g, dist, v, np.array([0, 1]), nq=2) - 4.0) < 1e-9
    # variants: 되찾기와 되찾기실패는 서로 배타
    a = variants(np.array([10., 10.]), np.array([8., 8.]), np.array([9.5, 8.5]),
                 np.array([0.01, 0.01]), np.array([9., 9.]), np.array([11., 11.]))
    assert not (a["D1_되찾기(현행)"][0] & a["D3_되찾기실패"][0]).any()
    print("selftest OK")


def run_asset(sym: str, path: Path) -> pd.DataFrame:
    kl = load(path)
    op, hi = kl.open.to_numpy(float), kl.high.to_numpy(float)
    lo_, cl = kl.low.to_numpy(float), kl.close.to_numpy(float)
    tr = np.maximum(hi - lo_, np.maximum(np.abs(hi - np.roll(cl, 1)), np.abs(lo_ - np.roll(cl, 1))))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy() / np.maximum(cl, 1e-12)
    lv = prior_day_levels(kl)
    n = len(kl); lo, hi_i = WARMUP, n - max(HORIZONS) - 2
    V = variants(hi, lo_, cl, atr, lv["pdl"], lv["pdh"])
    rows = []
    for side, k, long in (("bottom", 0, True), ("top", 1, False)):
        idx = np.flatnonzero(np.nan_to_num(V["D1_되찾기(현행)"][k]).astype(bool))
        idx = idx[(idx >= lo) & (idx <= hi_i)]
        if len(idx) < 60: continue
        g = gross_bp(op, cl, idx, H_MAIN, long)
        nul = gross_bp(op, cl, np.arange(lo, hi_i + 1, dtype=np.int64), H_MAIN, long).mean()
        rows.append(dict(sym=sym, side=side, n=len(idx), gross=round(float(g.mean()), 2),
                         null=round(float(nul), 2), excess=round(float(g.mean() - nul), 2),
                         hit=round(float((g > 0).mean()), 4), t=round(block_t(g, idx, lo, H_MAIN, nul), 2)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest: selftest(); return 0
    selftest(); OUT.mkdir(parents=True, exist_ok=True)

    print("[1/6] ETH …", flush=True)
    kl = load(DATA / "ETHUSDT/ETHUSDT-5m-api.csv")
    ts = kl.timestamp
    op, hi = kl.open.to_numpy(float), kl.high.to_numpy(float)
    lo_, cl = kl.low.to_numpy(float), kl.close.to_numpy(float)
    tr = np.maximum(hi - lo_, np.maximum(np.abs(hi - np.roll(cl, 1)), np.abs(lo_ - np.roll(cl, 1))))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy() / np.maximum(cl, 1e-12)
    lv = prior_day_levels(kl)
    day = ts.dt.floor("D").astype("int64").to_numpy(); yr = ts.dt.year.to_numpy()
    n = len(kl); lo, hi_i = WARMUP, n - max(HORIZONS) - 2
    span = (ts.iloc[hi_i] - ts.iloc[lo]).total_seconds() / 86400
    V = variants(hi, lo_, cl, atr, lv["pdl"], lv["pdh"])
    FIRE = {s: np.flatnonzero(np.nan_to_num(V["D1_되찾기(현행)"][k]).astype(bool)) for s, k in
            (("bottom", 0), ("top", 1))}
    FIRE = {s: v[(v >= lo) & (v <= hi_i)] for s, v in FIRE.items()}
    NUL = {(H, lg): float(gross_bp(op, cl, np.arange(lo, hi_i + 1, dtype=np.int64), H, lg).mean())
           for H in HORIZONS for lg in (True, False)}

    print("[2/6] A 안정성 …", flush=True)
    arows = []
    for side, long in (("bottom", True), ("top", False)):
        idx = FIRE[side]
        for H in HORIZONS:
            g = gross_bp(op, cl, idx, H, long); nul = NUL[(H, long)]
            hr = (g > 0).mean()
            for lab, m in ([("전체", np.ones(len(idx), bool)),
                            ("전반", ts.iloc[idx].to_numpy() < ts.iloc[(lo + hi_i) // 2]),
                            ("후반", ts.iloc[idx].to_numpy() >= ts.iloc[(lo + hi_i) // 2])]
                           + [(str(y), yr[idx] == y) for y in range(2022, 2027)]):
                if m.sum() < 50: continue
                gm = g[m]
                arows.append(dict(side=side, H=H, seg=lab, n=int(m.sum()),
                                  gross=round(float(gm.mean()), 2),
                                  excess=round(float(gm.mean() - nul), 2),
                                  hit=round(float((gm > 0).mean()), 4)))
    A = pd.DataFrame(arows); A.to_csv(OUT / "A_stability.csv", index=False)

    print("[3/6] B 정의절제 …", flush=True)
    brows = []
    for name, (fb, ft) in V.items():
        for side, f, long in (("bottom", fb, True), ("top", ft, False)):
            idx = np.flatnonzero(np.nan_to_num(f).astype(bool))
            idx = idx[(idx >= lo) & (idx <= hi_i)]
            if len(idx) < 60: continue
            g = gross_bp(op, cl, idx, H_MAIN, long); nul = NUL[(H_MAIN, True if long else False)]
            w = g[g > 0].mean() if (g > 0).any() else np.nan
            l = -g[g <= 0].mean() if (g <= 0).any() else np.nan
            clo, chi, p = day_boot(g - nul, day[idx])
            brows.append(dict(var=name, side=side, n=len(idx), per_day=round(len(idx) / span, 2),
                              gross=round(float(g.mean()), 2), excess=round(float(g.mean() - nul), 2),
                              hit=round(float((g > 0).mean()), 4), wl=round(w / l, 3),
                              t=round(block_t(g, idx, lo, H_MAIN, nul), 2),
                              ci_lo=round(clo, 2), ci_hi=round(chi, 2), p=round(p, 4)))
    B = pd.DataFrame(brows); B.to_csv(OUT / "B_ablation.csv", index=False)

    print("[4/6] C 거리매칭 + D 독립일수 …", flush=True)
    valid = np.zeros(n, bool); valid[lo:hi_i + 1] = True
    crows = []
    for side, long, lvl in (("bottom", True, lv["pdl"]), ("top", False, lv["pdh"])):
        idx = FIRE[side]
        dist = np.abs(cl - lvl) / np.maximum(cl, 1e-12)
        for H in HORIZONS:
            g_all = gross_bp(op, cl, np.arange(lo, hi_i + 1, dtype=np.int64), H, long)
            g = gross_bp(op, cl, idx, H, long)
            nm = dist_matched_null(g_all, dist, valid, idx)
            clo, chi, p = day_boot(g - nm, day[idx])
            uniq = len(np.unique(day[idx]))
            crows.append(dict(side=side, H=H, n=len(idx), 고유일=uniq,
                              건당일=round(len(idx) / uniq, 2),
                              gross=round(float(g.mean()), 2), 무조건귀무=round(NUL[(H, long)], 2),
                              거리매칭귀무=round(nm, 2), 매칭초과=round(float(g.mean() - nm), 2),
                              ci_lo=round(clo, 2), ci_hi=round(chi, 2), p=round(p, 4),
                              t=round(block_t(g, idx, lo, H, nm), 2)))
    C = pd.DataFrame(crows); C.to_csv(OUT / "C_distmatch.csv", index=False)

    print("[5/6] E TP/SL 격자 …", flush=True)
    erows = []
    half = ts.iloc[(lo + hi_i) // 2]
    for side, long in (("bottom", True), ("top", False)):
        idx = FIRE[side]
        a = np.nan_to_num(atr[idx])
        m1 = ts.iloc[idx].to_numpy() < half
        for tpm in TPS:
            for slm in SLS:
                pnl = np.array([barrier_pnl(op, hi, lo_, cl, np.array([i]), H_MAIN,
                                            tpm * a[k], slm * a[k], long)[0]
                                for k, i in enumerate(idx)])
                w = pnl[pnl > 0].mean() if (pnl > 0).any() else np.nan
                l = -pnl[pnl <= 0].mean() if (pnl <= 0).any() else np.nan
                clo, chi, p = day_boot(pnl - COST_PEG, day[idx])
                erows.append(dict(side=side, tp=tpm, sl=slm, n=len(idx),
                                  gross=round(float(pnl.mean()), 2),
                                  net=round(float(pnl.mean()) - COST_PEG, 2),
                                  hit=round(float((pnl > 0).mean()), 4), wl=round(w / l, 3),
                                  h1=round(float(pnl[m1].mean()), 2),
                                  h2=round(float(pnl[~m1].mean()), 2),
                                  ci_lo=round(clo, 2), ci_hi=round(chi, 2), p=round(p, 4)))
        print(f"    {side} 완료", flush=True)
    E = pd.DataFrame(erows); E.to_csv(OUT / "E_barrier.csv", index=False)

    print("[6/6] F 타자산 …", flush=True)
    F = pd.concat([run_asset(s, DATA / f"{s}USDT/{s}USDT-5m-api.csv")
                   for s in ("ETH", "BTC", "SOL", "XRP")], ignore_index=True)
    F.to_csv(OUT / "F_assets.csv", index=False)

    # ------------------------------------------------------------------ 보고
    print(f"\n{'='*120}\npivot_pdhl(전일 고저 되찾기) 심층 — ETH 5분봉 {n:,}봉 / 평가 {span:.0f}일")
    print("=" * 120)
    print("■ A 안정성 (gross bp · 비용 5.52 차감 전)")
    piv = A.pivot_table(index=["side", "H"], columns="seg", values="gross")
    cols = [c for c in ["전체", "전반", "후반", "2022", "2023", "2024", "2025", "2026"] if c in piv]
    print(piv[cols].round(2).to_string())
    sub = A[(A.seg.isin(["전반", "후반"]))].pivot_table(index=["side", "H"], columns="seg", values="gross")
    print(f"  ⇒ 전·후반 부호 일치 {int((np.sign(sub['전반'])==np.sign(sub['후반'])).sum())}/{len(sub)}")
    ys = A[A.seg.str.isdigit()]
    for (s, H), gg in ys.groupby(["side", "H"]):
        pos = int((gg.gross > 0).sum())
        print(f"     {s} H={H}: 연도 양수 {pos}/{len(gg)}")

    print("\n■ B 정의 절제 (H=48) — 되찾기가 실제로 일을 하는가")
    print(f"{'변형':<16}{'측':>4}{'건수':>8}{'건/일':>7}{'gross':>8}{'초과':>8}{'적중':>8}{'W/L':>7}{'t':>6}{'CI95':>18}")
    for r in B.itertuples():
        print(f"{r.var:<16}{'바닥' if r.side=='bottom' else '천장':>4}{r.n:>8}{r.per_day:>7.2f}"
              f"{r.gross:>8.2f}{r.excess:>+8.2f}{r.hit*100:>7.1f}%{r.wl:>7.2f}{r.t:>6.2f}"
              f"{f'[{r.ci_lo:+.2f},{r.ci_hi:+.2f}]':>18}")

    print("\n■ C 거리매칭 귀무 («전일 저가/고가 근처» 표류 제거) + D 독립일수")
    print(C.to_string(index=False))

    print("\n■ E TP/SL 격자 (intrabar, SL 우선, 시간상한 48봉, 순손익 = gross − 5.52)")
    for side in ("bottom", "top"):
        s = E[E.side == side]
        print(f"  [{'바닥' if side=='bottom' else '천장'}]  " +
              " ".join(f"tp{r.tp}/sl{r.sl}:{r.net:+.2f}" for r in s.itertuples()))
        best = s.loc[s.net.idxmax()]
        print(f"     최고 tp{best.tp}/sl{best.sl}  순손익 {best.net:+.2f}  적중 {best.hit*100:.1f}%  "
              f"W/L {best.wl:.2f}  전반 {best.h1:+.2f} 후반 {best.h2:+.2f}  CI[{best.ci_lo:+.2f},{best.ci_hi:+.2f}]")
    ok = E[(E.net > 0) & (np.sign(E.h1) == np.sign(E.h2)) & (E.ci_lo > 0)]
    print(f"  ⇒ 사전등록 통과(순손익>0 & 전후반 부호일치 & CI 0배제): {len(ok)}/{len(E)}")

    print("\n■ ⭐검정력 — 이 사건빈도로 «무엇을 잴 수 있는가»  (MDE = 2.8 x SE, 80% 검정력)")
    for r in C.itertuples():
        se = (r.ci_hi - r.ci_lo) / (2 * 1.96)
        print(f"  {'바닥' if r.side=='bottom' else '천장'} H={r.H:>3}  n={r.n} 고유일={r.고유일}  "
              f"일군집 SE {se:>6.2f}bp  **MDE {2.8*se:>6.2f}bp**  (비용선 {COST_PEG}bp 의 "
              f"{2.8*se/COST_PEG:>4.1f}배)")
    print("  ⇒ 비용선 미만의 참엣지는 이 표본에서 **원리적으로 확인 불가**다. "
          "따라서 «0 임을 증명»한 게 아니라 «안정성이 없고 상한이 낮다»가 정확한 진술.")

    print("\n■ F 타자산 이식 (정의 그대로, H=48) ⚠️상관자산은 완전 표본외가 아니다")
    print(F.to_string(index=False))
    print("=" * 120)
    print(json.dumps({"barrier_pass": len(ok), "barrier_cells": len(E),
                      "half_consistent": int((np.sign(sub['전반']) == np.sign(sub['후반'])).sum()),
                      "days": round(span, 1)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
