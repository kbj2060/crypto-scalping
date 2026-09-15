#!/usr/bin/env python3
"""ICT 3종 — 청산을 오라클(정답)로 두고 **진입만** 본다 (2026-09-16, 3라운드).

사용자 *"청산을 오라클 정답으로 할 경우로 계산해줘. 난 진입 증거가 필요해"*.

🔴 이 프레이밍은 이 저장소에서 **두 번 닫혔다**(09-11 · 09-13). 두 번 다 같은 이유로 무효였다:
오라클 청산 = MFE(창 안 최댓값)이고 **최댓값은 방향이 아니라 레인지(=변동성)를 잰다.**
09-13 에는 «MFE 회귀 상위1% 초과 +47.7bp» 가 나왔는데 롱 146.4 / 숏 154.8 로 **틀린 방향이 더
컸다** — 합은 3배인데 차는 그대로였다.

그래서 요청대로 오라클로 계산하되, **변동성이 답을 만들 수 없는 형태**로 만든다:

    ⭐ **MFE 점유율 = MFE / (MFE + MAE) = 내 쪽으로 간 전방 레인지의 비율**

이 통계량은 (1) [0,1] 유계, (2) 변동성 수준에 대해 **차수 무관**(레인지로 나눴다),
(3) **역방향 대조군이 정확히 1−share** 라서 "틀린 방향이 더 크다" 함정이 **구조적으로 불가능**하다.
09-13 의 판별식(합/차)도 여기서는 자동이다: 합 ≡ 레인지, 차 ≡ (2·share−1)·레인지.
게다가 귀무를 **전방 레인지 20분위로 매칭**한다 — 09-13 이 "ATR 매칭은 진입 봉만 통제하고
MFE 는 전방 변동성으로 만들어진다"로 걸렸던 자리를 정면으로 막는다.

오라클 정의: 롱 진입 open[i+1], 창 [i+1, i+H] 의 **고가 최대**에 청산(체결가능 — resting TP 는
닿는 즉시 체결, 이미 확정된 봉만 사용). 숏은 저가 최소. MAE 는 반대편 극단.

사전등록 판정: 진입 증거 = 전방레인지 매칭 귀무 대비 **점유율 초과 > 0** AND 일군집 CI95 0 배제
AND |t_블록| >= 2 AND 전·후반 부호 일치. 경제성은 별도로 (점유율초과 × 평균레인지)bp 로 환산.

출력: tmp/eth_ict3_oracle_20260916/*.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))

from research_eth_ict_killzone_fvg_eqh_20260916 import (  # noqa: E402  식 두 벌 금지
    load, CSV, BTC, day_boot, bh_fdr, prior_day_levels, level_reclaim, fvg_touch,
    equal_level_count, killzone_mask, EQ_TOL_ATR, COST_PEG)
from research_eth_ict3_regime_conditioned_20260916 import grid_thin  # noqa: E402

OUT = ROOT / "tmp/eth_ict3_oracle_20260916"
HORIZONS = [12, 48, 144]
WARMUP = 900
NQ = 20                 # 전방 레인지 매칭 분위 수(09-13 의 "전방 변동성 20분위")


def fwd_extremes(high: pd.Series, low: pd.Series, H: int) -> tuple[np.ndarray, np.ndarray]:
    """봉 i 기준 창 [i+1, i+H] 의 고가 최대 / 저가 최소. rolling(H) 뒤 shift(-H) 로 인과 창 정렬."""
    mx = high.rolling(H, min_periods=H).max().shift(-H).to_numpy(float)
    mn = low.rolling(H, min_periods=H).min().shift(-H).to_numpy(float)
    return mx, mn


def oracle(entry: np.ndarray, mx: np.ndarray, mn: np.ndarray, long: bool):
    """(MFE bp, MAE bp, 레인지 bp, 점유율). 롱이면 고가가 유리·저가가 불리, 숏은 반대."""
    up = (mx - entry) / entry * 1e4
    dn = (entry - mn) / entry * 1e4
    mfe, mae = (up, dn) if long else (dn, up)
    rng = mfe + mae
    share = np.divide(mfe, rng, out=np.full_like(rng, 0.5), where=rng > 1e-9)
    return mfe, mae, rng, share


def bin_means(share_all: np.ndarray, qbin: np.ndarray, nq: int) -> np.ndarray:
    """전방 레인지 분위별 귀무 점유율(그 분위 안 모든 봉의 평균)."""
    s = np.bincount(qbin, weights=share_all, minlength=nq)
    c = np.bincount(qbin, minlength=nq).astype(float)
    return s / np.maximum(c, 1)


def matched_null(bm: np.ndarray, b: np.ndarray) -> float:
    """층화 매칭 귀무 = 팔의 분위 분포로 가중한 분위별 귀무 점유율."""
    w = np.bincount(b, minlength=len(bm)).astype(float)
    return float((bm * w).sum() / w.sum())


def selftest() -> None:
    # oracle: 진입 100, 창 고가 110 / 저가 95 -> 롱 MFE 1000bp MAE 500bp 점유율 2/3
    e = np.array([100.]); mx = np.array([110.]); mn = np.array([95.])
    m, a, r, s = oracle(e, mx, mn, True)
    assert abs(m[0] - 1000) < 1e-6 and abs(a[0] - 500) < 1e-6 and abs(s[0] - 2 / 3) < 1e-9
    # ⭐역방향 대조군은 정확히 1-share (오라클 함정이 구조적으로 불가능하다는 근거)
    m2, a2, r2, s2 = oracle(e, mx, mn, False)
    assert abs(s[0] + s2[0] - 1.0) < 1e-12 and abs(r[0] - r2[0]) < 1e-9
    # fwd_extremes: 창 [i+1, i+H] 만 본다 (자기 봉 제외)
    h = pd.Series([1., 9., 2., 3.]); l = pd.Series([1., 0., 2., 3.])
    mx3, mn3 = fwd_extremes(h, l, 2)
    assert mx3[0] == 9.0 and mn3[0] == 0.0 and np.isnan(mx3[2])
    # matched_null: 층 분포가 같으면 전체 평균과 일치
    sa = np.array([0.1, 0.9, 0.2, 0.8]); qb = np.array([0, 0, 1, 1])
    bm = bin_means(sa, qb, 2)
    assert np.allclose(bm, [0.5, 0.5]) and abs(matched_null(bm, qb[[0, 2]]) - 0.5) < 1e-12
    # 층 분포가 한쪽에 쏠리면 그 층의 값이 나온다
    sa2 = np.array([0.1, 0.3, 0.7, 0.9]); bm2 = bin_means(sa2, qb, 2)
    assert abs(matched_null(bm2, np.array([1, 1])) - 0.8) < 1e-12
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest: selftest(); return 0
    selftest(); OUT.mkdir(parents=True, exist_ok=True)
    import live_evidence_signal_dashboard_20260823 as EV  # noqa: E402

    print("[1/4] 데이터·신호 …", flush=True)
    kl = load(CSV); btc = load(BTC) if BTC.exists() else None
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=None)
    ts_s = sig["timestamp"]; ts = ts_s.to_numpy()
    op = sig["open"].to_numpy(float)
    hi, lo_ = sig["high"].to_numpy(float), sig["low"].to_numpy(float)
    atr = sig["atr_pct"].to_numpy(float)
    day = ts_s.dt.floor("D").astype("int64").to_numpy()
    n = len(sig); lo, hi_i = WARMUP, n - max(HORIZONS) - 2
    ts_half = ts[(lo + hi_i) // 2]
    span = (ts_s.iloc[hi_i] - ts_s.iloc[lo]).total_seconds() / 86400
    entry = np.roll(op, -1)                       # 진입 = open[i+1]

    lv = prior_day_levels(sig)
    fvg_b, fvg_s = fvg_touch(hi, lo_, sig["close"].to_numpy(float), atr, 0.5)
    W = EV.SWEEP_LOOKBACK
    lvl_hi = sig["high"].rolling(W, min_periods=W).max().shift(1).to_numpy(float)
    lvl_lo = sig["low"].rolling(W, min_periods=W).min().shift(1).to_numpy(float)
    tol = np.nan_to_num(atr) * EQ_TOL_ATR
    sw_b = sig["bottom_liquidity_sweep"].fillna(False).to_numpy(bool)
    sw_t = sig["top_liquidity_sweep"].fillna(False).to_numpy(bool)
    cnt_lo = equal_level_count(lo_, lvl_lo, tol, W, upper=False)
    cnt_hi = equal_level_count(hi, lvl_hi, tol, W, upper=True)
    kz = killzone_mask(ts_s)["any"]
    ARMS = [("pivot_pdhl", level_reclaim(hi, lo_, sig["close"].to_numpy(float), lv["pdl"], below=True),
                            level_reclaim(hi, lo_, sig["close"].to_numpy(float), lv["pdh"], below=False)),
            ("pivot_s1r1", level_reclaim(hi, lo_, sig["close"].to_numpy(float), lv["s1"], below=True),
                            level_reclaim(hi, lo_, sig["close"].to_numpy(float), lv["r1"], below=False)),
            ("sweep_plain", sw_b, sw_t),
            ("eqhl_2touch", sw_b & (cnt_lo >= 2), sw_t & (cnt_hi >= 2)),
            ("eqhl_3touch", sw_b & (cnt_lo >= 3), sw_t & (cnt_hi >= 3)),
            ("fvg_ge05atr", fvg_b, fvg_s),
            ("kz_x_eqhl", sw_b & (cnt_lo >= 2) & kz, sw_t & (cnt_hi >= 2) & kz),
            ("all3_교집합", sw_b & (cnt_lo >= 2) & fvg_b & kz, sw_t & (cnt_hi >= 2) & fvg_s & kz)]

    print("[2/4] 오라클 청산(MFE/MAE) + 전방레인지 20분위 …", flush=True)
    rows = []
    POOL: dict = {}      # (arm,H) -> [(초과배열, 일, 절대인덱스), ...]  양측 합산 검정용
    QUART: list = []     # 레인지 사분위별 점유율 초과(H=48)
    for H in HORIZONS:
        mx, mn = fwd_extremes(sig["high"], sig["low"], H)
        valid = np.zeros(n, bool); valid[lo:hi_i + 1] = True
        valid &= np.isfinite(mx) & np.isfinite(mn) & np.isfinite(entry)
        for long in (True, False):
            mfe, mae, rng, share = oracle(entry, mx, mn, long)
            qbin = np.full(n, -1, np.int16)
            r_ok = rng[valid]
            edges = np.quantile(r_ok, np.linspace(0, 1, NQ + 1)[1:-1])
            qbin[valid] = np.searchsorted(edges, rng[valid])
            BM = bin_means(share[valid], qbin[valid], NQ)
            BMF = bin_means(mfe[valid], qbin[valid], NQ)   # 레인지매칭 귀무의 MFE(bp) 자체
            base_share = float(share[valid].mean()); base_mfe = float(mfe[valid].mean())
            base_rng = float(rng[valid].mean())
            rows.append(dict(arm="(무작위 대조)", side="bottom" if long else "top", H=H,
                             n=int(valid.sum()), per_day=round(valid.sum() / span, 1),
                             mfe=base_mfe, mae=float(mae[valid].mean()), rng=base_rng,
                             null_mfe=base_mfe, mfe_exc=0.0,
                             share=base_share, null_share=base_share, exc_pp=0.0, exc_bp=0.0,
                             t_block=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
                             h1=np.nan, h2=np.nan))
            for arm, f_long, f_short in ARMS:
                fire = np.nan_to_num(f_long if long else f_short).astype(bool) & valid
                idx = np.flatnonzero(fire)
                if len(idx) < 60: continue
                nul = matched_null(BM, qbin[idx]); nul_mfe = matched_null(BMF, qbin[idx])
                d = share[idx] - nul
                kept = grid_thin(idx, lo, H)
                dk = share[kept] - nul
                t = float(dk.mean() / (dk.std(ddof=1) / np.sqrt(len(dk)))) if len(dk) > 2 else np.nan
                POOL.setdefault((arm, H), []).append((d, day[idx], idx, float(rng[idx].mean())))
                if H == 48:   # 점유율 초과가 레인지 어디에 사는가 (돈은 큰 레인지에 있다)
                    qq = pd.qcut(rng[idx], 4, labels=False)
                    QUART.append(dict(arm=arm, side="bottom" if long else "top",
                                      **{f"Q{k+1}": round(100 * d[qq == k].mean(), 2) for k in range(4)},
                                      **{f"R{k+1}": round(float(rng[idx][qq == k].mean()), 0) for k in range(4)}))
                clo, chi, p = day_boot(d, day[idx])
                h1 = d[ts[idx] < ts_half]; h2 = d[ts[idx] >= ts_half]
                rows.append(dict(arm=arm, side="bottom" if long else "top", H=H, n=len(idx),
                                 per_day=round(len(idx) / span, 2),
                                 mfe=float(mfe[idx].mean()), mae=float(mae[idx].mean()),
                                 rng=float(rng[idx].mean()), null_mfe=nul_mfe,
                                 mfe_exc=float(mfe[idx].mean()) - nul_mfe,
                                 share=float(share[idx].mean()),
                                 null_share=nul, exc_pp=100 * float(d.mean()),
                                 exc_bp=float(d.mean()) * float(rng[idx].mean()),
                                 t_block=t, ci_lo=100 * clo, ci_hi=100 * chi, p=p,
                                 h1=100 * float(h1.mean()) if len(h1) else np.nan,
                                 h2=100 * float(h2.mean()) if len(h2) else np.nan))
        print(f"  H={H} 완료", flush=True)

    D = pd.DataFrame(rows)
    real = D.arm != "(무작위 대조)"
    D.loc[real, "q"] = bh_fdr(D.loc[real, "p"].to_numpy())
    D["PASS"] = (real & (D.exc_pp > 0) & (D.ci_lo > 0) & (D.t_block.abs() >= 2)
                 & (np.sign(D.h1) == np.sign(D.h2)))
    for c in ("mfe", "mae", "rng", "null_mfe", "mfe_exc", "exc_bp", "ci_lo", "ci_hi", "h1", "h2", "t_block"):
        D[c] = D[c].round(2)
    for c in ("share", "null_share"): D[c] = D[c].round(4)
    D["exc_pp"] = D.exc_pp.round(3)
    D.to_csv(OUT / "oracle_cells.csv", index=False)

    print(f"\n{'='*152}")
    print(f"오라클 청산(창 안 최적) — 진입만 격리.  점유율 = MFE/(MFE+MAE) = 내 쪽으로 간 전방 레인지 비율")
    print(f"평가 {span:.0f}일 · 귀무 = **전방 레인지 {NQ}분위 매칭** 무작위 진입 · 역방향 대조군은 정확히 1−점유율")
    print("=" * 152)
    print(f"{'팔':<14}{'측':>4}{'H':>5}{'건수':>8}{'건/일':>7}{'MFE':>9}{'MAE':>9}{'레인지':>9}"
          f"{'귀무MFE':>9}{'MFE초과':>9}{'점유율':>8}{'매칭귀무':>9}{'초과pp':>8}{'초과CI95':>18}{'초과bp':>8}{'t':>6}{'q':>7}")
    for H in HORIZONS:
        for r in D[D.H == H].itertuples():
            ci = f"[{r.ci_lo:+.2f},{r.ci_hi:+.2f}]" if np.isfinite(r.ci_lo) else "—"
            q = f"{r.q:.3f}" if np.isfinite(getattr(r, "q", np.nan)) else "—"
            print(f"{r.arm:<14}{'바닥' if r.side=='bottom' else '천장':>4}{r.H:>5}{r.n:>8}{r.per_day:>7.2f}"
                  f"{r.mfe:>9.1f}{r.mae:>9.1f}{r.rng:>9.1f}{r.null_mfe:>9.1f}{r.mfe_exc:>+9.2f}{r.share*100:>7.2f}%{r.null_share*100:>8.2f}%"
                  f"{r.exc_pp:>+8.3f}{ci:>18}{r.exc_bp:>+8.2f}{r.t_block:>6.2f}{q:>7}"
                  + ("  ★PASS" if r.PASS else ""))
        print("-" * 152)
    print(f"■ 통과 {int(D.PASS.sum())} / {int(real.sum())}")
    print("\n■ ⭐⭐거울상 제거 — 양측을 합쳐서 본다 (측면별 초과가 거울상이면 잔존 베타)")
    print(f"{'팔':<14}{'H':>5}{'건수':>8}{'바닥초과pp':>11}{'천장초과pp':>11}{'양측초과pp':>11}"
          f"{'양측CI95':>18}{'t블록':>7}{'오라클bp':>9}{'필요포착률':>10}")
    prows = []
    for (arm, H), parts in sorted(POOL.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if len(parts) != 2: continue
        d = np.concatenate([x[0] for x in parts]); dd = np.concatenate([x[1] for x in parts])
        rmean = float(np.mean([x[3] for x in parts]))
        kept = np.concatenate([grid_thin(x[2], lo, H) - x[2][0] * 0 for x in parts])  # 블록만 추출용
        dk = np.concatenate([x[0][np.searchsorted(x[2], grid_thin(x[2], lo, H))] for x in parts])
        t = float(dk.mean() / (dk.std(ddof=1) / np.sqrt(len(dk)))) if len(dk) > 2 else np.nan
        clo, chi, _p = day_boot(d, dd)
        bp = float(d.mean()) * rmean
        need = COST_PEG / bp if bp > 0 else np.inf
        b_ = [r for r in rows if r["arm"] == arm and r["H"] == H and r["side"] == "bottom"][0]
        t_ = [r for r in rows if r["arm"] == arm and r["H"] == H and r["side"] == "top"][0]
        prows.append(dict(arm=arm, H=H, n=len(d), exc_bottom=b_["exc_pp"], exc_top=t_["exc_pp"],
                          exc_pooled=100 * float(d.mean()), ci_lo=100 * clo, ci_hi=100 * chi,
                          t_block=t, oracle_bp=bp, need_capture=need))
        print(f"{arm:<14}{H:>5}{len(d):>8}{b_['exc_pp']:>+11.3f}{t_['exc_pp']:>+11.3f}"
              f"{100*d.mean():>+11.3f}{f'[{100*clo:+.2f},{100*chi:+.2f}]':>18}{t:>7.2f}"
              f"{bp:>+9.2f}{(f'{need*100:.0f}%' if np.isfinite(need) else '불가'):>10}")
    pd.DataFrame(prows).round(3).to_csv(OUT / "pooled.csv", index=False)
    P = pd.DataFrame(prows)
    print(f"  ⇒ 양측 CI95 가 0 을 배제한 (팔,H): {int(((P.ci_lo>0)|(P.ci_hi<0)).sum())}/{len(P)}"
          f" · 바닥 초과 음수 {int((P.exc_bottom<0).sum())}/{len(P)} · 천장 초과 양수 "
          f"{int((P.exc_top>0).sum())}/{len(P)}  (거울상 지문)")
    print(f"  ⇒ 필요 포착률: 실측 청산 정책의 포착률은 0.00~0.12 (09-11 오라클 상한 연구)")

    Q = pd.DataFrame(QUART); Q.to_csv(OUT / "range_quartile.csv", index=False)
    print("\n■ ⭐점유율 초과는 레인지 어디에 사는가 (H=48) — 돈은 Q4 에 있다")
    for r in Q.itertuples():
        print(f"  {r.arm:<14}{'바닥' if r.side=='bottom' else '천장'}  "
              f"Q1 {r.Q1:+6.2f}pp(레인지{r.R1:>4.0f}) | Q2 {r.Q2:+6.2f}({r.R2:>4.0f}) | "
              f"Q3 {r.Q3:+6.2f}({r.R3:>4.0f}) | Q4 {r.Q4:+6.2f}({r.R4:>4.0f})"
              + ("   ⬅단조 감소" if r.Q1 > r.Q2 > r.Q3 > r.Q4 else ""))
    mono = sum(1 for r in Q.itertuples() if r.Q1 > r.Q2 > r.Q3 > r.Q4)
    print(f"  ⇒ Q1>Q2>Q3>Q4 단조 감소: {mono}/{len(Q)} · Q4 초과가 음수: "
          f"{int((Q.Q4 < 0).sum())}/{len(Q)}")

    print("\n■ 09-13 판별식(합·차) — 합이 커지는데 차가 안 커지면 변동성이다")
    for H in HORIZONS:
        for arm in ["(무작위 대조)"] + [a for a, _, _ in ARMS]:
            b = D[(D.H == H) & (D.arm == arm) & (D.side == "bottom")]
            t_ = D[(D.H == H) & (D.arm == arm) & (D.side == "top")]
            if b.empty or t_.empty: continue
            s = float(b.mfe.iloc[0] + t_.mfe.iloc[0]); df = float(b.mfe.iloc[0] - t_.mfe.iloc[0])
            print(f"  H={H:>3} {arm:<14} 합(변동성) {s:>8.1f}  차(방향) {df:>+7.2f}  차/합 {df/s:>+7.4f}")
    print("=" * 152)
    print(json.dumps({"cells": int(real.sum()), "pass": int(D.PASS.sum()), "days": round(span, 1)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
