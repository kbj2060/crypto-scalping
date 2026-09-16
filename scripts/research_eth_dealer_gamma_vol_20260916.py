#!/usr/bin/env python3
"""딜러 감마(GEX) → 전방 실현변동성 — 인스타 Zomma 그래픽의 본론 검정 (2026-09-16).

그래픽의 매매 번역본은 «딜러가 롱감마면 헤지가 역방향이라 변동성이 눌리고, 숏감마면 증폭된다»이고
그래픽이 더하는 조건은 «저IV 일수록 감마가 ATM 에 몰려 효과가 커야 한다»(Γ_ATM ∝ 1/σ)다.

2026-08-15 연구는 `min_usable_snapshots_required: 100` 인데 n=4 라 판정 불가였다.
서버 수집기(cron 매시)가 계속 돌아 **ETH 750 스냅샷(2026-08-15~09-16)** 이 쌓였다 — 이제 잰다.
🔴로컬 duckdb 사본은 08-17 에서 멈춰 있었다(47개). 서버에서 export 해 왔다.

사전등록:
  G1 수준  spearman(front_month_GEX, 전방 RV) < 0        (롱감마 → 변동성 억제)
  G2 부호  NEG-gamma 시각의 전방 RV > POS-gamma 시각
  G3 증분  **후행 RV 를 통제한 뒤에도** 남는가 (변동성 자기상관이 전부일 수 있다)
  G4 조건  그래픽의 주장 — 저IV(하위분위)에서 효과가 더 큰가
판정: 일군집 부트 CI95 0 배제 AND 후행 RV 통제 후 부호 유지.
🔴검정력 경고: 독립일 **32일**뿐이다. «유의하지 않음»은 «효과 없음»이 아니다.

출력: tmp/eth_dealer_gamma_20260916/
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/home/kbj20/crypto-scalping")
OUT = ROOT / "tmp/eth_dealer_gamma_20260916"
GEX = REPO / "tmp/eth_gex_summary_export.csv"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from research_eth_ict_killzone_fvg_eqh_20260916 import load, day_boot  # noqa: E402

HORIZONS = {"1h": 12, "4h": 48, "24h": 288}
RNG = np.random.default_rng(20260916)


def rv_bp(c: np.ndarray, i: np.ndarray, h: int, fwd: bool) -> np.ndarray:
    """구간 로그수익 표준편차 × √h (bp). fwd=True 면 [i, i+h], False 면 [i-h, i]."""
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    out = np.empty(len(i))
    for k, x in enumerate(i):
        s = lr[x + 1:x + 1 + h] if fwd else lr[max(0, x - h + 1):x + 1]
        out[k] = s.std(ddof=1) * np.sqrt(h) * 1e4 if len(s) > 2 else np.nan
    return out


def resid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x 로 y 를 선형 통제한 잔차(절편 포함). 후행 RV 를 빼는 데 쓴다."""
    A = np.c_[np.ones(len(x)), x]
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ b


def selftest() -> None:
    c = np.exp(np.cumsum(np.r_[0.0, RNG.normal(0, 1e-3, 500)]))
    f = rv_bp(c, np.array([100]), 48, True); b = rv_bp(c, np.array([100]), 48, False)
    assert np.isfinite(f[0]) and np.isfinite(b[0]) and abs(f[0] / b[0] - 1) < 2.0
    # fwd 는 미래만 본다: 인덱스 이후를 상수로 만들면 fwd RV 가 0 이 되어야 한다
    c2 = c.copy(); c2[201:] = c2[200]
    assert rv_bp(c2, np.array([200]), 48, True)[0] == 0.0
    # resid: 완전 상관이면 잔차 0
    x = RNG.normal(size=200); assert np.abs(resid(3 * x + 1, x)).max() < 1e-9
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest: selftest(); return 0
    selftest(); OUT.mkdir(parents=True, exist_ok=True)

    g = pd.read_csv(GEX)
    g["ts"] = pd.to_datetime(g.recorded_at_utc, utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    g = g.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    kl = load(REPO / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
    c = kl.close.to_numpy(float); kts = kl.timestamp.to_numpy()
    i = np.searchsorted(kts, g.ts.to_numpy(), "right") - 1
    g["bar"] = i
    g = g[(g.bar > 288) & (g.bar < len(c) - max(HORIZONS.values()) - 2)].reset_index(drop=True)
    lag_min = (g.ts.to_numpy() - kts[g.bar.to_numpy()]) / np.timedelta64(1, "m")
    print(f"GEX {len(g)} 스냅샷 · {g.ts.min()} ~ {g.ts.max()} UTC · 봉 매칭 시차 중앙 {np.median(lag_min):.1f}분")
    print(f"front_month_GEX: 음수 {int((g.front_month_gex_usd<0).sum())} / 양수 "
          f"{int((g.front_month_gex_usd>0).sum())} · 중앙 ${g.front_month_gex_usd.median():,.0f}")
    day = g.ts.dt.floor("D").astype("int64").to_numpy()
    print(f"고유일 {len(np.unique(day))}  🔴이게 독립 관측 수다")

    rows = []
    for name, h in HORIZONS.items():
        fwd = rv_bp(c, g.bar.to_numpy(), h, True)
        trl = rv_bp(c, g.bar.to_numpy(), h, False)
        m = np.isfinite(fwd) & np.isfinite(trl)
        for col in ("front_month_gex_usd", "total_gex_usd"):
            x = g[col].to_numpy()[m]; y = fwd[m]; t = trl[m]
            from scipy.stats import spearmanr
            r_raw = spearmanr(x, y).statistic
            r_ctl = spearmanr(x, resid(y, t)).statistic         # G3 후행 RV 통제
            neg, pos = x < 0, x > 0
            d_bp = (y[neg].mean() - y[pos].mean()) if neg.sum() > 5 and pos.sum() > 5 else np.nan
            yr = resid(y, t)
            d_ctl = (yr[neg].mean() - yr[pos].mean()) if neg.sum() > 5 and pos.sum() > 5 else np.nan
            # CI 는 아래 «일 단위 순열 검정」에서 낸다(시간봉은 독립이 아니라 여기서 내면 과신)
            clo = chi = pv = np.nan
            rows.append(dict(H=name, col=col, n=int(m.sum()),
                             rho_raw=round(r_raw, 4), rho_ctl_trailRV=round(r_ctl, 4),
                             n_neg=int(neg.sum()), n_pos=int(pos.sum()),
                             fwdRV_neg=round(float(y[neg].mean()), 1) if neg.sum() else np.nan,
                             fwdRV_pos=round(float(y[pos].mean()), 1) if pos.sum() else np.nan,
                             diff_raw=round(float(d_bp), 2), diff_ctl=round(float(d_ctl), 2),
                             ci_lo=round(clo, 2), ci_hi=round(chi, 2)))
        # 후행 RV 자체의 설명력(대조군) — 이게 크면 GEX 증분이 진짜 질문이다
        from scipy.stats import spearmanr as sp
        rows.append(dict(H=name, col="(대조)후행RV", n=int(m.sum()),
                         rho_raw=round(sp(trl[m], fwd[m]).statistic, 4), rho_ctl_trailRV=0.0,
                         n_neg=0, n_pos=0, fwdRV_neg=np.nan, fwdRV_pos=np.nan,
                         diff_raw=np.nan, diff_ctl=np.nan, ci_lo=np.nan, ci_hi=np.nan))
    D = pd.DataFrame(rows); D.to_csv(OUT / "gex_vol.csv", index=False)
    print(f"\n{'='*118}")
    print("■ G1/G2/G3 — 딜러 감마 → 전방 실현변동성 (RV 는 bp, 구간 표준편차×√h)")
    print(f"{'H':>5}{'변수':<20}{'n':>6}{'ρ(원)':>9}{'ρ(후행RV통제)':>14}"
          f"{'음감마 n':>9}{'양감마 n':>9}{'fwdRV음':>10}{'fwdRV양':>10}{'차(원)':>9}{'차(통제)':>10}")
    for r in D.itertuples():
        print(f"{r.H:>5}{r.col:<20}{r.n:>6}{r.rho_raw:>9.4f}{r.rho_ctl_trailRV:>14.4f}"
              f"{r.n_neg:>9}{r.n_pos:>9}"
              f"{r.fwdRV_neg if np.isfinite(r.fwdRV_neg) else float('nan'):>10.1f}"
              f"{r.fwdRV_pos if np.isfinite(r.fwdRV_pos) else float('nan'):>10.1f}"
              f"{r.diff_raw if np.isfinite(r.diff_raw) else float('nan'):>9.2f}"
              f"{r.diff_ctl if np.isfinite(r.diff_ctl) else float('nan'):>10.2f}")
    print("  G1 통과 = ρ < 0 (롱감마 → 변동성 억제) · G2 통과 = 차 > 0 (음감마에서 변동성 큼)")

    # 일군집 순열 검정: 음감마일 vs 양감마일 (일 단위로 섞는다 — 시간봉은 독립이 아니다)
    print(f"\n■ 일 단위 순열 검정 (같은 날 스냅샷은 독립이 아니다)")
    for name, h in HORIZONS.items():
        fwd = rv_bp(c, g.bar.to_numpy(), h, True); trl = rv_bp(c, g.bar.to_numpy(), h, False)
        m = np.isfinite(fwd) & np.isfinite(trl)
        yr = resid(fwd[m], trl[m]); x = g.front_month_gex_usd.to_numpy()[m]; dd = day[m]
        dfd = pd.DataFrame({"y": yr, "neg": x < 0, "day": dd})
        per = dfd.groupby("day").agg(y=("y", "mean"), neg=("neg", "mean"))
        lab = per.neg > 0.5
        # 음감마는 희귀하다 — 이진 대신 **연속** GEX 로도 일 단위에서 잰다
        perg = pd.DataFrame({"y": yr, "x": x, "day": dd}).groupby("day").mean()
        from scipy.stats import spearmanr as sp2
        rr = sp2(perg.x, perg.y)
        print(f"  {name}: 일 {len(perg)}개 · spearman(일평균 GEX, 통제후 fwdRV) "
              f"rho {rr.statistic:+.3f} p {rr.pvalue:.4f}   (가설은 음수)")
        if lab.sum() < 3 or (~lab).sum() < 3:
            print(f"        음감마일 {int(lab.sum())} / 양감마일 {int((~lab).sum())} "
                  f"— 이진 검정 불가(음감마 상태가 희귀하다)"); continue
        obs = per.y[lab].mean() - per.y[~lab].mean()
        nul = np.array([(lambda p: p[:lab.sum()].mean() - p[lab.sum():].mean())(
            RNG.permutation(per.y.to_numpy())) for _ in range(4000)])
        p = 2 * min((nul <= obs).mean(), (nul >= obs).mean())
        print(f"  {name}: 음감마일 {int(lab.sum())} vs 양감마일 {int((~lab).sum())} · "
              f"Δ(통제 후 fwdRV) {obs:+.2f}bp · 귀무 CI95 [{np.percentile(nul,2.5):+.2f},"
              f"{np.percentile(nul,97.5):+.2f}] · p {p:.4f}")
    print("=" * 118)
    print(json.dumps({"snapshots": len(g), "unique_days": int(len(np.unique(day)))},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
