#!/usr/bin/env python3
"""GEX 를 «크기» 축으로 — 배포 변동성 예측 대비 증분이 있는가 (2026-09-16).

사용자 *"이걸 이용해서 매매 규칙이나 전략을 세울 순 없나?"*.

앞 라운드 결론이 후보를 하나로 좁힌다:
  · 방향은 죽었다 — 핀닝 f=0.49~0.52(균등 0.500) · 경제성 순 −4.12bp
  · 그런데 **GEX 는 전방 실현변동성을 예측한다**(ρ +0.44~+0.51, 후행RV 통제 후에도 +0.25~+0.34)
    부호는 이론과 반대지만 **예측에는 부호가 어느 쪽이든 상관없다**
  · 이 저장소에서 실제로 돈이 나온 축은 방향이 아니라 **크기**다

그래서 물을 것은 하나다: **GEX 가 klines 에서 못 뽑는 정보를 더하는가?**
(DVOL 이 채택됐던 근거와 같은 성질 — perp klines 파생 불가. 🔴DVOL 은 2026-08-04 에 끝나고
 GEX 는 08-15 에 시작해 **겹치는 구간이 없다** ⇒ DVOL 대비 증분은 지금 측정 불가, HAR-RV 대비만 잰다.)

사전등록:
  기저     HAR-RV (후행 RV 1h/4h/24h 로그) → 로그 전방 RV, 선형
  후보     기저 + GEX 피쳐(로그수준·1h변화·front/total 비율)
  판정     **일 블록 LOO 교차검증**의 OOS R² 증분 > 0 AND 일군집 부트 CI95 0 배제
  경제성   변동성 예측 개선 → 사이징. 개선폭을 명목 오차 감소로 환산해 보고한다
🔴검정력: 독립일 **31일**. 이 라운드는 «후보 등록」이지 승격이 아니다.

출력: tmp/eth_gex_vol_increment_20260916/
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/home/kbj20/crypto-scalping")
OUT = ROOT / "tmp/eth_gex_vol_increment_20260916"
GEX = REPO / "tmp/eth_gex_summary_export.csv"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from research_eth_ict_killzone_fvg_eqh_20260916 import load, day_boot  # noqa: E402
from research_eth_dealer_gamma_vol_20260916 import rv_bp  # noqa: E402  식 두 벌 금지

HORIZONS = {"1h": 12, "4h": 48, "24h": 288}
RNG = np.random.default_rng(20260916)


def fit_predict_loo(X: np.ndarray, y: np.ndarray, day: np.ndarray) -> np.ndarray:
    """일 블록 leave-one-day-out 예측. 같은 날 행은 독립이 아니므로 날 단위로 뺀다."""
    out = np.full(len(y), np.nan)
    for d in np.unique(day):
        te = day == d
        tr = ~te
        if tr.sum() < X.shape[1] + 5: continue
        A = np.c_[np.ones(tr.sum()), X[tr]]
        b, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        out[te] = np.c_[np.ones(te.sum()), X[te]] @ b
    return out


def r2(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(p)
    return 1.0 - ((y[m] - p[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()


def selftest() -> None:
    # LOO: 완전 선형이면 OOS 예측이 정확하다
    d = np.repeat(np.arange(8), 10)
    x = RNG.normal(size=(80, 2)); y = 2 * x[:, 0] - x[:, 1] + 3
    p = fit_predict_loo(x, y, d)
    assert np.nanmax(np.abs(p - y)) < 1e-8, np.nanmax(np.abs(p - y))
    # LOO 가 같은 날을 학습에 안 쓴다: 한 날만 라벨을 뒤집어도 다른 날 예측이 안 변한다
    y2 = y.copy(); y2[d == 0] += 100.0
    p2 = fit_predict_loo(x, y2, d)
    assert np.abs(p2[d == 3] - p[d == 3]).max() > 1e-9   # 학습셋이 바뀌니 조금은 변한다
    assert r2(y, y) == 1.0 and r2(y, np.full_like(y, y.mean())) == 0.0
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
    g["bar"] = np.searchsorted(kts, g.ts.to_numpy(), "right") - 1
    g = g[(g.bar > 288) & (g.bar < len(c) - max(HORIZONS.values()) - 2)].reset_index(drop=True)
    bar = g.bar.to_numpy()
    day = g.ts.dt.floor("D").astype("int64").to_numpy()
    print(f"GEX {len(g)} 스냅샷 · 고유일 {len(np.unique(day))} · {g.ts.min()} ~ {g.ts.max()} UTC")

    # 후행 RV (HAR 성분) — 전부 인과적
    trl = {k: rv_bp(c, bar, h, False) for k, h in HORIZONS.items()}
    gx = g.front_month_gex_usd.to_numpy(); tx = g.total_gex_usd.to_numpy()
    # 부호가 섞인 명목값이라 로그 대신 부호보존 로그(asinh)를 쓴다
    f_gex = np.arcsinh(gx / 1e6)
    f_tot = np.arcsinh(tx / 1e6)
    f_rat = np.clip(gx / np.where(np.abs(tx) < 1e3, np.nan, tx), -5, 5)
    f_chg = np.r_[np.nan, np.diff(f_gex)]
    GF = np.c_[f_gex, f_tot, f_rat, f_chg]
    GNAMES = ["asinh(front/1e6)", "asinh(total/1e6)", "front/total", "Δasinh(front)"]

    rows = []
    for name, h in HORIZONS.items():
        fwd = rv_bp(c, bar, h, True)
        y = np.log(np.maximum(fwd, 1e-6))
        Xb = np.c_[[np.log(np.maximum(trl[k], 1e-6)) for k in HORIZONS]].T   # HAR 3성분
        m = np.isfinite(y) & np.isfinite(Xb).all(1) & np.isfinite(GF).all(1)
        Xb, Xg, yy, dd = Xb[m], np.c_[Xb, GF][m], y[m], day[m]
        pb = fit_predict_loo(Xb, yy, dd)
        pg = fit_predict_loo(Xg, yy, dd)
        ok = np.isfinite(pb) & np.isfinite(pg)
        # 제곱오차 차이(기저 − 후보): 양수면 후보가 낫다. 일군집 부트로 CI.
        de = (yy[ok] - pb[ok]) ** 2 - (yy[ok] - pg[ok]) ** 2
        clo, chi, p = day_boot(de, dd[ok])
        rows.append(dict(H=name, n=int(ok.sum()), days=int(len(np.unique(dd))),
                         r2_har=round(r2(yy[ok], pb[ok]), 4), r2_har_gex=round(r2(yy[ok], pg[ok]), 4),
                         d_r2=round(r2(yy[ok], pg[ok]) - r2(yy[ok], pb[ok]), 4),
                         mse_gain=round(float(de.mean()), 5),
                         ci_lo=round(clo, 5), ci_hi=round(chi, 5), p=round(p, 4),
                         rmse_har=round(float(np.sqrt(((yy[ok]-pb[ok])**2).mean())), 4),
                         rmse_gex=round(float(np.sqrt(((yy[ok]-pg[ok])**2).mean())), 4)))
    D = pd.DataFrame(rows); D.to_csv(OUT / "increment.csv", index=False)
    print(f"\n{'='*118}")
    print("■ HAR-RV 기저 대비 GEX 증분 — 일 블록 LOO 교차검증 (타깃 = log 전방 RV)")
    print(f"{'H':>5}{'n':>6}{'일':>5}{'OOS R² 기저':>12}{'+GEX':>9}{'ΔR²':>9}"
          f"{'MSE 이득':>10}{'CI95':>26}{'p':>8}{'RMSE 기저→+GEX':>18}")
    for r in D.itertuples():
        print(f"{r.H:>5}{r.n:>6}{r.days:>5}{r.r2_har:>12.4f}{r.r2_har_gex:>9.4f}{r.d_r2:>+9.4f}"
              f"{r.mse_gain:>+10.5f}{f'[{r.ci_lo:+.5f},{r.ci_hi:+.5f}]':>26}{r.p:>8.4f}"
              f"{f'{r.rmse_har:.4f}→{r.rmse_gex:.4f}':>18}")
    print("  판정 = ΔR² > 0 AND CI95 가 0 배제")

    # 어느 GEX 피쳐가 일하는가 (전체 적합 계수 t값, 참고용)
    print(f"\n■ 어느 GEX 피쳐가 일하는가 (전체 적합 t값 — 참고용, 일군집 보정 전)")
    for name, h in HORIZONS.items():
        fwd = rv_bp(c, bar, h, True); y = np.log(np.maximum(fwd, 1e-6))
        Xb = np.c_[[np.log(np.maximum(trl[k], 1e-6)) for k in HORIZONS]].T
        m = np.isfinite(y) & np.isfinite(Xb).all(1) & np.isfinite(GF).all(1)
        A = np.c_[np.ones(m.sum()), Xb[m], GF[m]]; yy = y[m]
        b, *_ = np.linalg.lstsq(A, yy, rcond=None)
        res = yy - A @ b
        s2 = res @ res / (len(yy) - A.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.pinv(A.T @ A)))
        t = b / se
        print(f"  {name}: " + " · ".join(f"{n} t={t[4+i]:+.2f}" for i, n in enumerate(GNAMES)))

    # 경제성 환산: 변동성 예측 개선 -> 사이징 명목 오차
    print(f"\n■ 경제성 환산 — 사이징은 1/변동성에 비례한다(배포 경로 규약)")
    for r in D.itertuples():
        # log RV 의 RMSE 가 곧 명목 배수의 로그 오차. exp(RMSE) 가 «몇 배 틀리는가»
        print(f"  {r.H}: 명목 배수 오차 exp(RMSE) {np.exp(r.rmse_har):.3f}배 → "
              f"{np.exp(r.rmse_gex):.3f}배 (개선 {100*(np.exp(r.rmse_har)-np.exp(r.rmse_gex))/np.exp(r.rmse_har):+.2f}%)")
    print(f"\n■ ⭐언제 판정 가능한가 — 현재 t 와 필요 독립일 (t≥2 기준, t ∝ √일)")
    import datetime as dt
    start = g.ts.min().date()
    for r in D.itertuples():
        se = (r.ci_hi - r.ci_lo) / (2 * 1.96)
        t = r.mse_gain / se if se > 0 else float("nan")
        need = int(np.ceil(r.days * (2.0 / t) ** 2)) if t > 0 else None
        eta = (start + dt.timedelta(days=need)) if need and need < 3650 else None
        print(f"  {r.H}: 현재 t {t:+.2f} · 필요 독립일 {need if need else '—'} "
              f"(수집 시작 {start} 기준 ETA {eta if eta else '—'})")
    print("  ⚠️이건 «효과가 지금 크기 그대로일 때» 의 계산이다. 줄어들면 더 걸린다.")
    print("=" * 118)
    print(json.dumps({"days": int(D.days.max()), "best_dR2": float(D.d_r2.max())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
