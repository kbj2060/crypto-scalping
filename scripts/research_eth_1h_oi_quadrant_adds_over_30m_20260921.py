"""«1시간 가격×OI 사분면»이 «상황 읽기 · 30분» 엔진의 30분 연료 규칙 위에 무엇을 더하는가 (2026-09-21).

사용자 질문: "1시간 가격×OI 사분면도 현재 상황 분석과 미래 예측하는 데에 필요한가?"

🔴재야 하는 이유: 엔진의 «연료» 규칙(30분 창의 가격 방향 × 창 ΔOI 부호)이 이미 같은 가족이고,
   **30분 창은 1시간 창의 부분집합**이라 두 부호가 기계적으로 상관된다. 그래서 «1시간이 30분 위에
   조건부로 무엇을 더하나»와 «두 부호가 실제로 얼마나 자주 다른가»를 같이 재야 답이 나온다.

두 통화로 잰다:
  (a) 앞 30분 수익 bp  -- 09-20 사분면 연구와 같은 통화(연속성)
  (b) 30분 안 ±배리어 **선착** -- 카드가 실제로 답하는 질문(어느 목표에 먼저 닿나).
      🔴bp 를 정확도로 환산하지 않는다(저장소 계약) -- 선착은 직접 센다.

방향 규칙은 엔진과 같다: |이동| > MOVE_THR_FRAC(0.35) × 창 고저폭.
"""
import numpy as np, pandas as pd

PANEL = "data/binance_vision/panel/ETHUSDT.parquet"
W30, W60, FWD = 6, 12, 6          # 5분봉 단위
MOVE_THR_FRAC = 0.35              # dashboard/situation.py 와 같은 값
BARRIERS_BP = (30.0, 60.0)
B = 2000
rng = np.random.default_rng(20260921)


def boot_ci(vals, day, mask_a, mask_b):
    """두 부분집합 평균차의 일-블록 부트스트랩 CI. 일별 합/개수를 미리 접고 날짜를 재표집한다."""
    d = np.unique(day); di = np.searchsorted(d, day)
    nd = len(d)
    sa = np.bincount(di[mask_a], vals[mask_a], nd); ca = np.bincount(di[mask_a], None, nd)
    sb = np.bincount(di[mask_b], vals[mask_b], nd); cb = np.bincount(di[mask_b], None, nd)
    pick = rng.integers(0, nd, size=(B, nd))
    A = sa[pick].sum(1) / np.maximum(ca[pick].sum(1), 1)
    Bm = sb[pick].sum(1) / np.maximum(cb[pick].sum(1), 1)
    diff = A - Bm
    return float(vals[mask_a].mean() - vals[mask_b].mean()), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def main():
    df = pd.read_parquet(PANEL, columns=["timestamp", "high", "low", "close", "sum_open_interest"]).sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["close", "sum_open_interest"]).reset_index(drop=True)
    c, hi, lo, oi = (df[k].to_numpy(float) for k in ("close", "high", "low", "sum_open_interest"))
    n = len(c)
    day = df["timestamp"].dt.normalize().astype("int64").to_numpy() // 86_400_000_000_000
    year = df["timestamp"].dt.year.to_numpy()

    def frame(win):
        mv = np.full(n, np.nan); rg = np.full(n, np.nan); do = np.full(n, np.nan)
        mv[win:] = (c[win:] - c[:-win]) / c[:-win] * 1e4
        do[win:] = oi[win:] - oi[:-win]
        # 창 고저폭: 같은 창 안의 max(high)-min(low)
        s_hi = pd.Series(hi).rolling(win).max().to_numpy()
        s_lo = pd.Series(lo).rolling(win).min().to_numpy()
        rg = (s_hi - s_lo) / c * 1e4
        d = np.where(mv > MOVE_THR_FRAC * rg, 1, np.where(mv < -MOVE_THR_FRAC * rg, -1, 0))
        return mv, rg, do, d

    mv30, rg30, do30, d30 = frame(W30)
    mv60, rg60, do60, d60 = frame(W60)

    fwd = np.full(n, np.nan)
    fwd[:-FWD] = (c[FWD:] - c[:-FWD]) / c[:-FWD] * 1e4

    # 선착: 앞 FWD 봉 안에서 +bp 와 -bp 중 어느 쪽을 먼저 치나 (+1 상단 / -1 하단 / 0 미접촉)
    def first_touch(bp):
        out = np.zeros(n, np.int8)
        up = c * (1 + bp / 1e4); dn = c * (1 - bp / 1e4)
        for k in range(1, FWD + 1):
            h = np.full(n, np.nan); l = np.full(n, np.nan)
            h[:-k] = hi[k:]; l[:-k] = lo[k:]
            hit_u = (out == 0) & (h >= up); hit_d = (out == 0) & (l <= dn)
            out[hit_u & ~hit_d] = 1; out[hit_d & ~hit_u] = -1
            out[hit_u & hit_d] = 0   # 같은 봉에 양쪽 = 판정 불가, 버린다(봉 안 순서를 모른다)
        return out

    ft = {bp: first_touch(bp) for bp in BARRIERS_BP}

    ok = ~np.isnan(mv60) & ~np.isnan(fwd) & ~np.isnan(do60) & ~np.isnan(do30)
    print(f"패널 {n:,}행 · 유효 {ok.sum():,}행 · {df['timestamp'].min().date()}~{df['timestamp'].max().date()} · 일수 {len(np.unique(day[ok])):,}")

    s30, s60 = np.sign(do30), np.sign(do60)
    agree = (s30 == s60) & ok
    print(f"\n① ΔOI 부호 일치율 (30분 vs 1시간): {agree.sum() / ok.sum():.1%}  ⇐ 두 프레임이 이만큼 같은 말을 한다")
    for lbl, m in (("추세(d30≠0)", ok & (d30 != 0)), ("횡보(d30=0)", ok & (d30 == 0))):
        sub = m.sum()
        print(f"   {lbl}: 일치 {((s30 == s60) & m).sum() / sub:.1%} (n={sub:,})")

    print("\n② 30분 연료 칸 안에서 1시간 ΔOI 부호가 앞 30분 수익을 가르는가  [일-블록 부트스트랩 95%]")
    print(f"{'30분 칸':<22}{'n':>9}{'1h OI↓ bp':>11}{'1h OI↑ bp':>11}{'차이':>9}{'95% CI':>20}{'연도 부호':>10}")
    for d_lbl, dv in (("상승", 1), ("하락", -1), ("횡보", 0)):
        for o_lbl, osign in (("OI↓", -1), ("OI↑", 1)):
            cell = ok & (d30 == dv) & (s30 == osign)
            a = cell & (s60 < 0); b = cell & (s60 > 0)
            if a.sum() < 300 or b.sum() < 300:
                print(f"{d_lbl+' & '+o_lbl:<22}{cell.sum():>9,}  표본 부족"); continue
            diff, lo_, hi_ = boot_ci(fwd, day[ok | True], a, b)
            yrs = [np.sign(fwd[a & (year == y)].mean() - fwd[b & (year == y)].mean())
                   for y in range(2022, 2027) if (a & (year == y)).sum() > 50 and (b & (year == y)).sum() > 50]
            pos = f"{sum(1 for v in yrs if v > 0)}/{len(yrs)}"
            print(f"{d_lbl+' & '+o_lbl:<22}{cell.sum():>9,}{fwd[a].mean():>11.2f}{fwd[b].mean():>11.2f}{diff:>9.2f}{f'[{lo_:+.2f},{hi_:+.2f}]':>20}{pos:>10}")

    print("\n③ 카드의 통화: 30분 안 ±배리어 **선착** — 1시간 ΔOI 부호가 선착을 바꾸는가")
    for bp in BARRIERS_BP:
        t = ft[bp]
        print(f"  배리어 ±{bp:.0f}bp   (미접촉 {np.mean(t[ok] == 0):.0%})")
        for d_lbl, dv in (("상승", 1), ("하락", -1)):
            for o_lbl, osign in (("OI↓", -1), ("OI↑", 1)):
                cell = ok & (d30 == dv) & (s30 == osign) & (t != 0)
                a = cell & (s60 < 0); b = cell & (s60 > 0)
                if a.sum() < 300 or b.sum() < 300:
                    continue
                up = (t == 1).astype(float)
                diff, lo_, hi_ = boot_ci(up, day, a, b)
                print(f"    {d_lbl} & {o_lbl:<5} 1h OI↓ 상단선착 {up[a].mean():.1%} · 1h OI↑ {up[b].mean():.1%} · 차이 {diff:+.1%} [{lo_:+.1%},{hi_:+.1%}]  n={a.sum():,}/{b.sum():,}")

    print("\n④ 참고: 1시간 프레임 **단독** (09-20 재현, 앞 30분 수익) — 하락 안에서 OI 부호로 가름")
    lowq = ok & (mv60 <= np.nanquantile(mv60[ok], 0.25))
    a, b = lowq & (s60 < 0), lowq & (s60 > 0)
    diff, lo_, hi_ = boot_ci(fwd, day, a, b)
    print(f"   하락 & OI↓ {fwd[a].mean():+.2f}bp vs 하락 & OI↑ {fwd[b].mean():+.2f}bp · 차이 {diff:+.2f} [{lo_:+.2f},{hi_:+.2f}]  (앞 30분 기준이라 1h 기준 +3.52 의 절반 규모가 기대값)")


if __name__ == "__main__":
    main()
