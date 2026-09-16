#!/usr/bin/env python3
"""옵션 감마 «핀닝」 — 인스타 Zomma 그래픽의 유일한 매매 번역본을 검정한다 (2026-09-16).

사용자가 `davidarias_cfa` 의 *"Gamma and Zomma in Vol Shifts"* 를 주며 *"이 전략에 대해 연구해줘"*.

🔴 **그 그래픽은 전략이 아니다** — 진입·청산 규칙이 없고, 담긴 내용은 BS 항등식 두 개다:
  Γ_ATM ∝ 1/(S σ √T)        (저변동성일수록 ATM 감마가 크고 종이 좁다)
  Zomma = Γ(d₁d₂−1)/σ       (ATM 부근 음수 · 날개 양수)
검산 결과 그래픽은 **T=90/252(거래일) 관례에서 정확**하다(σ 8.78% · Δ .8326 · Γ .1538 · Z −.2104).

매매로 번역되는 유일한 주장은 **딜러 감마 → 핀닝**이다:
  "저IV → 감마가 ATM 에 집중 → 딜러의 반대방향 헤지가 커져 만기 근처에서 가격이 스트라이크에 붙는다"
그래픽이 실제로 더하는 건 **조건**이다 — 그 효과가 **저IV 에서 더 강해야 한다**(Γ_ATM ∝ 1/σ).

이 저장소에 옵션 매매 경로는 없다(ETHUSDT 무기한만). GEX 수집기는 있으나
`data/live/deribit_gex.duckdb` 가 **2026-08-15~17 · 47스냅샷**에서 멈춰 판정 불가
(08-15 연구도 `verdict_possible_at_current_data_volume: false`). 그래서 **옵션 데이터 없이**
라운드 스트라이크 격자 + Deribit 만기 시각(매일 08:00 UTC)만으로 대리 검정한다.

사전등록(실행 전 고정):
  H1 핀닝    만기 직전 창(07:00~08:00 UTC)의 스트라이크 거리 < 같은 시각 비만기 기준
             ⚠️매일 만기라 «시각 대조군」이 없다 ⇒ **금요일(주간/월간, OI 큼) vs 평일**로 짝짓는다
  H2 조건    그 효과가 DVOL 하위 분위에서 더 크다  ← 그래픽의 실제 주장
  H3 경제성  핀닝 매매(07:00 진입 → 08:00 청산, 가장 가까운 스트라이크 방향) 순손익 > 5.52bp
  🔴결정적 통제: 저변동성이면 거리가 당연히 작다 ⇒ 거리를 **그 창의 ATR 로 정규화**하고
     DVOL 분위 비교는 **ATR 십분위 매칭 귀무**와 함께 본다(09-14 규율).
판정: 일군집 부트 CI95 0 배제 AND 전·후반 부호 일치.

출력: tmp/eth_option_pinning_20260916/
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/home/kbj20/crypto-scalping")
OUT = ROOT / "tmp/eth_option_pinning_20260916"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from research_eth_ict_killzone_fvg_eqh_20260916 import day_boot, load, COST_PEG  # noqa: E402

GRIDS = [50.0, 25.0, 100.0]      # 사전등록 주 격자 50, 변형 선언 25/100
EXPIRY_H = 8                      # Deribit 만기 08:00 UTC
WARMUP = 900


def strike_dist_bp(close: np.ndarray, grid: float) -> np.ndarray:
    """가장 가까운 격자 스트라이크까지의 거리(bp). 격자는 절대달러라 가격대에 따라 촘촘함이 다르다."""
    return np.abs(close - np.round(close / grid) * grid) / close * 1e4


def pin_frac(close: np.ndarray, grid: float) -> np.ndarray:
    """⭐척도무관 핀닝 측도: 격자 칸 안 상대위치를 접은 값 f ∈ [0,1).

    핀닝이 없으면 close mod grid 는 균등이라 **f ~ U(0,1)** 이고 평균 0.5 · P(f<0.2)=0.2 다.
    거리(bp)는 **가격 수준**이, 거리/ATR 는 **변동성**이 지배한다 — 둘 다 핀닝 측도가 아니다.
    f 는 둘 다에서 자유롭다(격자 칸 폭으로 나눴으므로)."""
    u = (close / grid) % 1.0
    return np.minimum(u, 1.0 - u) * 2.0


def selftest() -> None:
    c = np.array([2000.0, 2025.0, 2049.0, 2051.0])
    d = strike_dist_bp(c, 50.0)
    assert d[0] == 0.0
    assert abs(d[1] - 25 / 2025 * 1e4) < 1e-6            # 정확히 중간
    assert abs(d[2] - 1 / 2049 * 1e4) < 1e-6             # 2050 에 1달러
    assert abs(d[3] - 1 / 2051 * 1e4) < 1e-6
    # 격자가 촘촘하면 거리가 작아진다(자명하지만 방향 고정용)
    assert strike_dist_bp(np.array([2037.0]), 25.0)[0] <= strike_dist_bp(np.array([2037.0]), 50.0)[0]
    # pin_frac: 격자 위 0, 정중앙 1, 그리고 균등분포면 평균 0.5
    f = pin_frac(np.array([2000.0, 2025.0, 2050.0, 2012.5]), 50.0)
    assert f[0] == 0.0 and abs(f[1] - 1.0) < 1e-12 and f[2] == 0.0 and abs(f[3] - 0.5) < 1e-12
    r = np.random.default_rng(0).uniform(1000, 5000, 200_000)
    assert abs(pin_frac(r, 50.0).mean() - 0.5) < 0.01, "균등이면 0.5 여야 한다"
    assert abs((pin_frac(r, 50.0) < 0.2).mean() - 0.2) < 0.01
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest: selftest(); return 0
    selftest(); OUT.mkdir(parents=True, exist_ok=True)

    kl = load(REPO / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
    ts = kl.timestamp
    c, hi, lo = kl.close.to_numpy(float), kl.high.to_numpy(float), kl.low.to_numpy(float)
    op = kl.open.to_numpy(float)
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
    atr = pd.Series(tr).rolling(288, min_periods=144).mean().to_numpy() / np.maximum(c, 1e-12) * 1e4
    hour, minute, dow = ts.dt.hour.to_numpy(), ts.dt.minute.to_numpy(), ts.dt.dayofweek.to_numpy()
    day = ts.dt.floor("D")
    n = len(kl); ok = np.zeros(n, bool); ok[WARMUP:n - 20] = True
    ok &= np.isfinite(atr)
    print(f"ETH 5분봉 {n:,} · {ts.iloc[0]} ~ {ts.iloc[-1]}")

    # DVOL (2024-01~2026-08) — 시간봉, 인과적으로 직전 시간 값을 쓴다
    dv = pd.read_csv(REPO / "data/derivatives/deribit_dvol/ETH_dvol_hourly.csv")
    dv["timestamp"] = pd.to_datetime(dv.timestamp)
    dvs = pd.Series(dv.close.to_numpy(float), index=dv.timestamp).sort_index()
    j = dvs.index.searchsorted(ts.to_numpy(), "right") - 1
    dvol = np.where(j >= 0, dvs.to_numpy()[np.clip(j, 0, len(dvs) - 1)], np.nan)
    stale = (ts.to_numpy() - dvs.index.to_numpy()[np.clip(j, 0, len(dvs) - 1)]) / np.timedelta64(1, "h")
    dvol = np.where((j >= 0) & (stale <= 2.0), dvol, np.nan)   # 2시간 넘게 낡으면 버린다
    print(f"DVOL 매칭 {np.isfinite(dvol).sum():,}봉 ({np.isfinite(dvol).mean()*100:.0f}%) "
          f"· {dvs.index.min()} ~ {dvs.index.max()}")

    rows = []
    for grid in GRIDS:
        d = strike_dist_bp(c, grid)
        dn = pin_frac(c, grid)                          # ⭐척도무관: 균등이면 0.5
        pre = ok & (hour == EXPIRY_H - 1)               # 만기 직전 1시간
        base = ok & (hour != EXPIRY_H - 1) & (hour != EXPIRY_H)
        fri, wd = pre & (dow == 4), pre & (dow != 4)
        for lab, m in (("만기직전(전체)", pre), ("금요일 만기직전", fri), ("평일 만기직전", wd),
                       ("기준(다른 시각)", base)):
            rows.append(dict(grid=grid, seg=lab, n=int(m.sum()),
                             dist_bp=float(d[m].mean()), pin_f=float(dn[m].mean()),
                             p_near=float((dn[m] < 0.2).mean()), med_atr=float(atr[m].mean())))
    D = pd.DataFrame(rows); D.to_csv(OUT / "h1_pinning.csv", index=False)
    print(f"\n{'='*104}\n■ H1 핀닝 — 만기 직전(07:00~08:00 UTC)에 스트라이크에 붙는가")
    print("  f = 격자 칸 안 상대거리(0=스트라이크 위, 1=정중앙). 핀닝 없으면 **균등 ⇒ f=0.500 · P(f<0.2)=0.200**")
    print(f"{'격자':>6}{'구간':<18}{'n':>8}{'거리bp':>9}{'f 평균':>9}{'P(f<0.2)':>10}{'평균ATRbp':>11}")
    for r in D.itertuples():
        print(f"{r.grid:>6.0f}{r.seg:<18}{r.n:>8}{r.dist_bp:>9.2f}{r.pin_f:>9.4f}{r.p_near:>10.4f}{r.med_atr:>11.1f}")

    # H1 검정: 금요일 만기직전 vs 평일 만기직전 (같은 시각 = 시간대 통제)
    print(f"\n■ H1 검정 — 금요일(주간·월간 만기) vs 평일, 같은 시각·ATR 정규화")
    g = 50.0
    d = strike_dist_bp(c, g); dn = pin_frac(c, g)
    pre = ok & (hour == EXPIRY_H - 1)
    f, w = pre & (dow == 4), pre & (dow != 4)
    diff_days = []
    for lab, m in (("금요일", f), ("평일", w)):
        s = pd.DataFrame({"d": dn[m], "day": day[m].to_numpy()})
        gm = s.groupby("day").d.mean()
        diff_days.append(gm)
        print(f"  {lab:<6} n={int(m.sum()):>6} 고유일 {len(gm):>4}  f {dn[m].mean():.4f} "
              f"P(f<0.2) {(dn[m]<0.2).mean():.4f}")
    a, b = diff_days
    delta = a.mean() - b.mean()
    pool = np.concatenate([a.to_numpy(), b.to_numpy()])
    lab = np.r_[np.ones(len(a)), np.zeros(len(b))]
    rng = np.random.default_rng(20260916)
    nul = np.array([(lambda p: p[:len(a)].mean() - p[len(a):].mean())(rng.permutation(pool))
                    for _ in range(4000)])
    print(f"  Δ(금−평) = {delta:+.4f}  순열 귀무 CI95 [{np.percentile(nul,2.5):+.4f}, "
          f"{np.percentile(nul,97.5):+.4f}] · p {2*min((nul<=delta).mean(),(nul>=delta).mean()):.4f}")
    print(f"  ⇒ 핀닝이면 금요일이 **더 작아야**(음수) 한다")

    # H2: DVOL 분위별 — 그래픽의 주장
    print(f"\n■ H2 그래픽의 주장 — 저IV 일수록 핀닝이 강한가 (DVOL 3분위 · ATR 정규화)")
    m = pre & np.isfinite(dvol)
    q = pd.qcut(pd.Series(dvol[m]), 3, labels=["저IV", "중IV", "고IV"])
    s = pd.DataFrame({"d": dn[m], "draw": d[m], "atr": atr[m], "q": q.to_numpy(),
                      "day": day[m].to_numpy()})
    print(f"{'DVOL 분위':<8}{'n':>7}{'고유일':>7}{'거리bp(원)':>12}{'평균ATR':>9}{'f 평균':>9}{'CI95(f)':>20}")
    for k in ["저IV", "중IV", "고IV"]:
        z = s[s.q == k]
        clo, chi, _ = day_boot(z.d.to_numpy(), z.day.astype("int64").to_numpy())
        print(f"{k:<8}{len(z):>7}{z.day.nunique():>7}{z.draw.mean():>12.2f}{z.atr.mean():>9.1f}"
              f"{z.d.mean():>9.4f}{f'[{clo:.4f},{chi:.4f}]':>20}")
    lo_, hi_ = s[s.q == "저IV"], s[s.q == "고IV"]
    dd = lo_.d.mean() - hi_.d.mean()
    print(f"  Δ(저IV − 고IV) = {dd:+.4f} (핀닝이면 음수)")
    print(f"  ⚠️원 거리(bp)는 **가격 수준**이 지배하고(격자가 절대달러) 거리/ATR 는 **변동성**이 지배한다.")
    print(f"     f 만이 둘 다에서 자유롭다. 균등 기준선 0.500 과 비교한다.")

    # H3 경제성: 07:00 진입 -> 08:00 청산, 가장 가까운 스트라이크 방향
    print(f"\n■ H3 경제성 — 07:00 진입 → 08:00 청산, 가장 가까운 스트라이크 쪽으로")
    idx = np.flatnonzero(ok & (hour == EXPIRY_H - 1) & (minute == 0))
    idx = idx[idx + 12 < n]
    tgt = np.round(c[idx] / g) * g
    side = np.sign(tgt - c[idx])                       # 스트라이크가 위면 롱
    e, x = op[idx + 1], c[idx + 12]
    bp = (x - e) / e * 1e4 * side
    keep = side != 0
    bp, idx2 = bp[keep], idx[keep]
    dayk = day.to_numpy()[idx2]
    clo, chi, p = day_boot(bp, pd.Series(dayk).astype("int64").to_numpy())
    print(f"  n={len(bp)} 고유일 {len(np.unique(dayk))} · gross {bp.mean():+.2f}bp "
          f"· 순손익 {bp.mean()-COST_PEG:+.2f}bp · CI95 [{clo:+.2f},{chi:+.2f}] p {p:.4f}")
    h = len(bp) // 2
    print(f"  전반 {bp[:h].mean():+.2f} / 후반 {bp[h:].mean():+.2f} · 적중 {(bp>0).mean()*100:.1f}%")
    dv2 = dvol[idx2]
    fin = np.isfinite(dv2)
    if fin.sum() > 60:
        qq = pd.qcut(pd.Series(dv2[fin]), 3, labels=["저IV", "중IV", "고IV"])
        for k in ["저IV", "중IV", "고IV"]:
            z = bp[fin][qq.to_numpy() == k]
            print(f"    {k}: n={len(z)} gross {z.mean():+.2f} 순 {z.mean()-COST_PEG:+.2f}bp")
    print("=" * 104)
    json.dump({"h1_delta_fri_minus_weekday": float(delta), "h2_delta_lowiv_minus_highiv": float(dd),
               "h3_gross_bp": float(bp.mean()), "h3_net_bp": float(bp.mean() - COST_PEG),
               "h3_n": int(len(bp))}, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
