#!/usr/bin/env python3
"""**손절의 꼬리 절단이 가우시안 목적함수를 얼마나 틀리게 하나** (2026-09-14).

배포된 보유시간 선택(`live_eth_trade_plan_20260913.growth_per_hour`)은 켈리 근사다:
    g(H) = L·(μ_H − c_H)/1e4 − ½(L·σ_H/1e4)²,   시간당 = g/(H/60)
그런데 실제로는 **3% 손절**이 걸려 있어 (1) 왼쪽 꼬리가 −3%에서 잘리고 (2) 잘린 거래는
H 를 못 채우고 **일찍 끝난다**. 근사는 둘 다 모른다. 2026-09-13 감사에서 비용 항목은
넣었지만 절단은 못 넣고 «지평 순위용»이라고만 적어 뒀다 -- 이 스크립트가 그 한계를 잰다.

## 방법
869일 1분봉 실제 경로. 진입점은 STRIDE 분 간격.
  · **방향은 뽑지 않고 가중한다**: 정확도 a 의 기대값 = a·g(sign(r_H) 쪽) + (1−a)·g(반대 쪽).
    베르누이 추출보다 몬테카를로 잡음이 없고 정확히 같은 양이다.
  · 손절: **봉내** 고가/저가가 −3% 를 지나면 그 자리에서 끝(비용 21.96bp = 왕복+시장가 추가),
    아니면 H 분 종가(비용 5.88bp). 양쪽 다 펀딩을 더한다.
  · 계좌: 1 + L·(move − cost). 0 이하면 **파산**(로그성장 정의 불가라 따로 센다).
  · 시간당은 **실현 보유시간**으로 나눈다: E[g]/E[t] (갱신보상). 손절이 일찍 끝내 주는
    효과가 여기 들어간다 -- 배포 공식은 t = H 로 고정이라 이걸 못 본다.

## 대조군 (CLAUDE.md·호메로스 §5 규율)
  · **무손절**: 같은 경로·같은 L 이라 크기가 자동으로 매칭된다.
  · **a = 0.50 위약**: 실력이 없으면 어떤 지평도 이기면 안 된다.
  · **공식은 같은 표본의 μ·σ 로 먹인다** -- 차이가 파라미터 오차가 아니라 근사 자체에서
    온다는 걸 보장한다(K_HORIZON × atr 를 쓰면 두 오차가 섞인다).
  · 겹치는 창이라 **일(1440분) 블록 부트스트랩**으로만 CI 를 낸다.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.live_eth_trade_plan_20260913 import (  # noqa: E402
    ROUND_TRIP_COST_BP, STOP_EXTRA_COST_BP, funding_cost_bp)

TAPE = next((p for p in (
    pathlib.Path(__file__).resolve().parents[1] / "data/research/eth_tape_1m_20260906.parquet",
    pathlib.Path("/home/kbj20/crypto-scalping/data/research/eth_tape_1m_20260906.parquet"))
    if p.exists()), None)
HOLDS = (60, 120, 240, 480, 1440)
ACCS = (0.50, 0.55, 0.60, 0.66)
LEVS = (3.0, 6.0)
STOP_PCT = 0.03
STRIDE = 15
CHUNK = 4000
B_BOOT = 400
RNG = np.random.default_rng(20260914)


def _first_touch(window: np.ndarray, barrier: np.ndarray, below: bool) -> np.ndarray:
    """배리어에 처음 닿는 분(1-based). 안 닿으면 0. window 는 (진입수, H)."""
    hit = window <= barrier[:, None] if below else window >= barrier[:, None]
    any_hit = hit.any(axis=1)
    return np.where(any_hit, hit.argmax(axis=1) + 1, 0)


def simulate(c, hi, lo, idx, H):
    """진입점마다 롱·숏 두 결과. 돌려주는 건 (move, hold_min) 두 쌍과 종료 수익률."""
    from numpy.lib.stride_tricks import sliding_window_view
    win_lo = sliding_window_view(lo[1:], H)          # 진입 봉 **다음** 분부터
    win_hi = sliding_window_view(hi[1:], H)
    n = len(idx)
    tL = np.empty(n, np.int32); tS = np.empty(n, np.int32)
    for s in range(0, n, CHUNK):
        sl = idx[s:s + CHUNK]
        cc = c[sl]
        tL[s:s + CHUNK] = _first_touch(win_lo[sl], cc * (1 - STOP_PCT), True)
        tS[s:s + CHUNK] = _first_touch(win_hi[sl], cc * (1 + STOP_PCT), False)
    r = c[idx + H] / c[idx] - 1.0                    # 종료 시점 단순수익률
    return r, tL, tS


def legs(r, touch, side_sign, H, funding_bp):
    """한 측면의 (move, cost_bp, hold_min). 손절이면 −3% 에서 끝나고 시간도 거기서 멈춘다."""
    stopped = touch > 0
    move = np.where(stopped, -STOP_PCT, side_sign * r)
    hold = np.where(stopped, touch.astype(float), float(H))
    cost = np.where(stopped, ROUND_TRIP_COST_BP + STOP_EXTRA_COST_BP, ROUND_TRIP_COST_BP)
    # 펀딩은 **실현 보유시간**에 비례한다 -- 일찍 끝나면 덜 낸다.
    cost = cost + funding_bp * (hold / 480.0)
    return move, cost, hold, stopped


def growth(move, cost_bp, L):
    """건당 로그성장. 계좌가 0 이하가 되면 파산이라 로그가 정의되지 않는다."""
    acct = 1.0 + L * (move - cost_bp / 1e4)
    ruin = acct <= 0.0
    return np.log(np.where(ruin, np.nan, acct)), ruin


def run(c, hi, lo, days, idx, H, with_stop: bool):
    r, tL, tS = simulate(c, hi, lo, idx, H)
    fL, fS = funding_cost_bp(H, "LONG"), funding_cost_bp(H, "SHORT")
    if not with_stop:
        tL = np.zeros_like(tL); tS = np.zeros_like(tS)
    mL, cL, hL, sL = legs(r, tL, +1.0, H, fL)
    mS, cS, hS, sS = legs(r, tS, -1.0, H, fS)
    out = {}
    for L in LEVS:
        gL, ruL = growth(mL, cL, L)
        gS, ruS = growth(mS, cS, L)
        right_is_long = r > 0
        # 맞힌 쪽 / 틀린 쪽으로 가른다(정확도 a 의 조작적 정의)
        g_ok = np.where(right_is_long, gL, gS); h_ok = np.where(right_is_long, hL, hS)
        g_no = np.where(right_is_long, gS, gL); h_no = np.where(right_is_long, hS, hL)
        ru_ok = np.where(right_is_long, ruL, ruS); ru_no = np.where(right_is_long, ruS, ruL)
        st_ok = np.where(right_is_long, sL, sS); st_no = np.where(right_is_long, sS, sL)
        out[L] = (g_ok, g_no, h_ok, h_no, ru_ok | ru_no, st_ok, st_no)
    return out, r, days


def per_hour(g_ok, g_no, h_ok, h_no, a, w=None):
    """E[g]/E[t] (갱신보상). 시간당으로 바꿔야 «짧은 걸 여러 번»이 들어간다."""
    if w is None:
        eg = a * np.nanmean(g_ok) + (1 - a) * np.nanmean(g_no)
        et = a * h_ok.mean() + (1 - a) * h_no.mean()
    else:
        eg = a * np.nansum(g_ok * w) / w.sum() + (1 - a) * np.nansum(g_no * w) / w.sum()
        et = a * (h_ok * w).sum() / w.sum() + (1 - a) * (h_no * w).sum() / w.sum()
    return 60.0 * eg / et


def main() -> int:
    if TAPE is None:
        print("테이프 없음 -- 건너뜀", file=sys.stderr)
        return 0
    d = pd.read_parquet(TAPE, columns=["px_last", "px_max", "px_min"])
    c, hi, lo = (d[k].to_numpy(float) for k in ("px_last", "px_max", "px_min"))
    print(f"표본 {len(c):,}분 = {len(c)/1440:.0f}일 · 손절 {STOP_PCT:.0%} · stride {STRIDE}분")
    print(f"블록 부트 B={B_BOOT} (1일 블록) · 대조군: 무손절 · a=0.50 위약\n")

    rows = []
    for H in HOLDS:
        idx = np.arange(0, len(c) - H - 2, STRIDE)
        days = idx // 1440
        res_on, r, _ = run(c, hi, lo, days, idx, H, True)
        res_off, _, _ = run(c, hi, lo, days, idx, H, False)
        # 공식에 먹일 μ·σ 는 **같은 표본**에서 잰다(파라미터 오차 배제)
        b_bp, sd_bp = np.abs(r).mean() * 1e4, r.std(ddof=1) * 1e4
        for L in LEVS:
            g_ok, g_no, h_ok, h_no, ruin, st_ok, st_no = res_on[L]
            o_ok, o_no, oh_ok, oh_no, oruin, _, _ = res_off[L]
            for a in ACCS:
                sim = per_hour(g_ok, g_no, h_ok, h_no, a)
                nostop = per_hour(o_ok, o_no, oh_ok, oh_no, a)
                cost = ROUND_TRIP_COST_BP  # 공식이 쓰는 비용(측면 평균이라 펀딩은 상쇄)
                gf = (L * ((2 * a - 1) * b_bp - cost) / 1e4
                      - 0.5 * (L * sd_bp / 1e4) ** 2)
                rows.append({"H": H, "L": L, "a": a, "formula": 60 * gf / H,
                             "sim_stop": sim, "sim_nostop": nostop,
                             "stop_rate": a * st_ok.mean() + (1 - a) * st_no.mean(),
                             "ruin_nostop": oruin.mean(), "ruin_stop": ruin.mean(),
                             "b_bp": b_bp, "sd_bp": sd_bp,
                             "hold_real": a * h_ok.mean() + (1 - a) * h_no.mean()})
        print(f"  H={H:>5} 완료 (진입 {len(idx):,} · b={b_bp:.1f}bp · sd={sd_bp:.1f}bp)")
    df = pd.DataFrame(rows)
    df.to_csv(pathlib.Path(__file__).resolve().parents[1]
              / "tmp/stop_truncation_vs_gaussian_20260914.csv", index=False)
    print("\n저장: tmp/stop_truncation_vs_gaussian_20260914.csv")
    return df




# ── 2차: 일블록 부트스트랩 + 국면 분할 (2026-09-14) ─────────────────────────
# 1차에서 «순위는 안 바뀌고 격차는 긴 지평에서 양수»가 나왔다. 두 가지를 더 봐야 한다:
#   ① 격차가 표본잡음인가 -- 겹치는 창이라 **1일 블록** 부트로만 잰다.
#   ② 지금 국면(잔잔)에서도 같은가 -- 테이프 전체는 격변기를 포함해 b 가 2.4배 크다.


def boot_gap(g_ok, g_no, h_ok, h_no, days, a, deployed, b=B_BOOT):
    """일(1440분) 블록 부트스트랩으로 (시뮬 − 공식) 격차의 CI. 겹치는 창이라 이것만 유효하다."""
    uniq = np.unique(days)
    idx_by_day = {d: np.flatnonzero(days == d) for d in uniq}
    out = np.empty(b)
    for k in range(b):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_day[d] for d in pick])
        out[k] = per_hour(g_ok[sel], g_no[sel], h_ok[sel], h_no[sel], a) - deployed
    return np.percentile(out, [2.5, 97.5])


def regime_mask(c, idx, window=1440):
    """진입 직전 window 분의 실현변동성. 중앙값으로 잔잔/험함을 가른다(진입 시점까지만 본다)."""
    r = np.diff(np.log(c), prepend=np.log(c[0]))
    v = pd.Series(r).rolling(window, min_periods=window).std().to_numpy()
    vi = v[idx]                      # 진입 봉까지의 값 -- 미래를 안 본다
    ok = ~np.isnan(vi)
    med = np.nanmedian(vi)
    return ok & (vi <= med), ok & (vi > med), vi


def stage2() -> None:
    from scripts.live_eth_trade_plan_20260913 import expected_cost_bp
    d = pd.read_parquet(TAPE, columns=["px_last", "px_max", "px_min"])
    c, hi, lo = (d[k].to_numpy(float) for k in ("px_last", "px_max", "px_min"))
    L = 6.0
    print(f"\n=== 2차: 일블록 부트(B={B_BOOT}) + 국면 분할 · L={L:g}배 ===")
    print(f"{'H':>6} {'국면':>6} {'b(bp)':>7} {'손절률':>7} {'공식':>8} {'시뮬':>8} "
          f"{'격차':>8} {'95% CI':>18}")
    for H in HOLDS:
        idx = np.arange(0, len(c) - H - 2, STRIDE)
        days = idx // 1440
        res, r, _ = run(c, hi, lo, days, idx, H, True)
        g_ok, g_no, h_ok, h_no, _ruin, st_ok, st_no = res[L]
        calm, wild, _ = regime_mask(c, idx)
        for name, m in (("전체", np.ones(len(idx), bool)), ("잔잔", calm), ("험함", wild)):
            a = 0.60
            bb = np.abs(r[m]).mean() * 1e4
            sd = r[m].std(ddof=1) * 1e4
            dep = 60 * (L * ((2 * a - 1) * bb - expected_cost_bp(H, "LONG", 0.0)) / 1e4
                        - 0.5 * (L * sd / 1e4) ** 2) / H
            sim = per_hour(g_ok[m], g_no[m], h_ok[m], h_no[m], a)
            ci = boot_gap(g_ok[m], g_no[m], h_ok[m], h_no[m], days[m], a, dep)
            sr = a * st_ok[m].mean() + (1 - a) * st_no[m].mean()
            print(f"{H:>6} {name:>6} {bb:>7.1f} {sr:>7.3f} {1e4*dep:>8.2f} {1e4*sim:>8.2f} "
                  f"{1e4*(sim-dep):>8.2f} [{1e4*ci[0]:>7.2f},{1e4*ci[1]:>7.2f}]")


def verdict(df) -> None:
    """이 연구의 **결론을 게이트로 고정한다**. 목적함수를 건드리면 여기서 걸려야 한다."""
    from scripts.live_eth_trade_plan_20260913 import expected_cost_bp as _c
    df = df.copy()
    df["deployed"] = df.apply(
        lambda r: 60 * (r.L * ((2 * r.a - 1) * r.b_bp - _c(int(r.H), "LONG", 0.0)) / 1e4
                        - 0.5 * (r.L * r.sd_bp / 1e4) ** 2) / r.H, axis=1)
    # ① **위험한 방향의 오차가 없어야 한다**: 공식이 «번다»는데 실제가 손실인 칸 0.
    danger = df[(df.deployed > 0) & (df.sim_stop < 0)]
    assert len(danger) == 0, danger
    # ② 격차는 압도적으로 **보수적**(공식이 낮게 본다)이어야 한다.
    conservative = (df.sim_stop > df.deployed).mean()
    assert conservative > 0.8, conservative
    # ③ 🔴가장 중요: **최적 지평이 안 바뀌어야 한다**. 바뀌면 근사를 못 쓴다.
    for (L, a), g in df.groupby(["L", "a"]):
        assert (g.loc[g.deployed.idxmax(), "H"] == g.loc[g.sim_stop.idxmax(), "H"]), (L, a)
    # ④ 격차는 지평에 **단조 증가**한다(손절이 자주 걸릴수록 절단 이득이 크다).
    gap = df.groupby("H").apply(lambda g: (g.sim_stop - g.deployed).mean(), include_groups=False)
    assert gap.iloc[-1] > gap.iloc[0], gap
    print(f"\n✅ 게이트 통과 — 위험 오차 0칸 · 보수적 {100*conservative:.0f}% · 최적 지평 불변 ·"
          f" 격차 단조증가({1e4*gap.iloc[0]:+.2f} → {1e4*gap.iloc[-1]:+.2f} ×1e4/시간)")


if __name__ == "__main__":
    # 인자 없이 돌리면 1차(전 격자) + 2차(부트·국면). `stage2` 만 주면 2차만.
    if "stage2" not in sys.argv:
        verdict(main())
    stage2()
