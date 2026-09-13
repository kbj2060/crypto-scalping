"""**학습된 손절폭 vs 고정 손절폭** — 크기 모델을 손절에도 쓴다 (2026-09-13, 사용자 제안).

사용자: *"이것도 사이징 모델처럼 위험도 분석이라 내 데이터가 잘 맞출 수 있을 것 같으니
한 번 모델 만들어서 테스트해줘."*

## 새 모델을 만들지 않는다 — 이미 있다
배포된 MAE 분위 모델(`live_eth_mae_quantile_model_20260913`)이 «이 시장 상태에서 H 분 동안
각오할 역행폭»을 준다. 손절은 그 분포를 **다른 지점에서 읽는 것**뿐이다. 크기는 꼬리(0.1%
초과)를 쓰고, 손절은 몸통(중앙 근처)을 쓴다. 같은 모델, 다른 분위다.
⇒ 손절폭 = `k × 예측 MAE 분위`.  k 를 키우면 덜 잘리고 한 번의 손실이 커진다.

## 🔴대조군이 이 실험의 전부다
«변동성에 맞춘 것»과 «모델이 잘 맞춘 것»은 다르다. 2026-09-12 사이징에서 1/ATR 대조군이
정확히 이 둘을 갈랐다(ATR 만 쓴 모델은 현행과 동점 -- 이득의 출처는 새 정보였다).
그래서 세 팔을 나란히 놓는다:
  · **고정**   : 2% · 3%  (사용자 제안의 단순판)
  · **ATR연동**: k × atr_pct × √H  (모델 없이 변동성만 -- 이게 진짜 대조군이다)
  · **모델**   : k × 예측 MAE 분위
모델이 ATR연동을 못 이기면 «학습이 필요 없다»가 답이다.

## 판정 지표
1년(735건) 복리 중앙 계좌배수 · 파산율 · **손절률의 국면 간 안정성**.
세 번째가 적응형 손절의 존재 이유다 -- 고정 손절은 잔잔하면 안 걸리고 험하면 난타당한다.

⚠️5분봉으로 돈다(모델이 5분봉 피쳐를 쓴다). 배리어는 봉내 고저 -- 라이브 컨벤션과 같다.
진입 시점 무작위 · 갭은 도달 봉 종가로 근사 · 손절은 테이커라 +3bp.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_mae_quantile_model_20260913 as maq  # noqa: E402
import live_eth_sizing_vol_model_20260912 as svm  # noqa: E402

KL = pathlib.Path("/home/kbj20/crypto-scalping/binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
TRIPS = ROOT / "data/live/account_round_trips.jsonl"
HOLD_BARS = 48               # 4시간 = 5분봉 48개 (사용자 고정)
COST_BP, TAKER_EXTRA_BP = 5.88, 3.0
LEV = 6.0                    # 배포 상한
SEED = 20260913


def load_klines() -> pd.DataFrame:
    d = pd.read_csv(KL, usecols=["timestamp", "open", "high", "low", "close",
                                 "quote_volume", "trades"], parse_dates=["timestamp"])
    return d.dropna().sort_values("timestamp").reset_index(drop=True)


def unit_returns() -> np.ndarray:
    rows = [json.loads(l) for l in TRIPS.read_text().splitlines() if l.strip()]
    return np.array([x["net_pnl"] / (x["max_qty"] * x["entry_price"])
                     for x in rows if x.get("closed")])


def predictions(d: pd.DataFrame) -> dict:
    """모델의 4시간 MAE 분위와, 대조군용 atr_pct 를 한 번에 낸다."""
    art = maq.load_model()
    assert art is not None, "모델 아티팩트가 없다"
    X = svm.build_features(d.timestamp, d.close.to_numpy(float), d.quote_volume.to_numpy(float),
                           d.trades.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float))
    out = {}
    for side, sv in (("LONG", 1), ("SHORT", -1)):
        f = X.copy()
        f["log_h"] = np.log(HOLD_BARS * 5.0)
        f["side"] = sv
        pm = maq.predict_mae(art["models"], f)
        out[side] = {"q50": pm[0.5], "q90": pm[0.9]}
    out["atr_pct"] = np.exp(X["atr288"].to_numpy())      # 빌더가 로그로 저장한다
    out["ok"] = np.isfinite(X.to_numpy(float)).all(1)
    return out


def simulate(d: pd.DataFrame, pred: dict, rng, n: int, *, stop_kind: str, k: float,
             acc: float, lev: float = LEV, risk_budget: float | None = None):
    """stop_kind: 'none' | 'fixed' | 'atr' | 'model'.

    🔴`risk_budget` 이 있으면 **계좌 위험을 먼저 고정하고 배수를 손절폭에서 유도**한다
    (L = 위험/손절, 상한 lev). 문헌의 fixed-fractional / 변동성 타게팅이 이 형태다 --
    손절이 넓어야 하는 국면(험함)에서 자동으로 작게 들어간다.
    없으면 배수가 고정이고 손절폭이 계좌 손실을 정한다(= 현행 구조)."""
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    m = len(c)
    ok = np.flatnonzero(pred["ok"][:m - HOLD_BARS - 1])
    ok = ok[ok > svm.WARMUP]
    start = rng.choice(ok, n, replace=True)
    right = rng.random(n) < acc
    ret = np.empty(n); ruin = np.zeros(n, bool); stopped = np.zeros(n, bool)
    stops = np.empty(n); atr_at = np.empty(n); levs = np.full(n, lev)
    for t in range(n):
        a = int(start[t]); end = a + HOLD_BARS
        truth = 1.0 if c[end] >= c[a] else -1.0
        s = truth if right[t] else -truth
        side = "LONG" if s > 0 else "SHORT"
        e = c[a]
        atr_at[t] = pred["atr_pct"][a]
        if stop_kind == "none":
            stop = None
        elif stop_kind == "fixed":
            stop = k
        elif stop_kind == "atr":
            # 모델 없이 변동성만: atr_pct(봉당) × √봉수 × k
            stop = k * pred["atr_pct"][a] * np.sqrt(HOLD_BARS)
        else:
            stop = k * pred[side]["q50"][a] / 100.0
        stops[t] = stop if stop else np.nan
        L = lev if (risk_budget is None or not stop) else min(lev, risk_budget / max(stop, 1e-9))
        levs[t] = L
        seg_lo, seg_hi = lo[a:end + 1], hi[a:end + 1]
        adv = (e - seg_lo) / e if s > 0 else (seg_hi - e) / e
        worst = float(adv.max())
        cost = COST_BP
        if stop is not None and worst >= stop:
            j = int(np.argmax(adv >= stop))
            move = min(-stop, s * (c[a + j] / e - 1))    # 갭이면 더 나쁜 쪽
            stopped[t] = True
            cost += TAKER_EXTRA_BP
        else:
            move = s * (c[end] / e - 1)
            if worst * L >= 1.0:
                ruin[t] = True; ret[t] = -1.0; continue
        r = L * (move - cost / 1e4)
        if r <= -1.0:
            ruin[t] = True; r = -1.0
        ret[t] = r
    return {"ret": ret, "ruin": float(ruin.mean()), "stop_rate": float(stopped.mean()),
            "stops": stops, "atr": atr_at, "stopped": stopped, "levs": levs}


def log_growth(pool: np.ndarray) -> float:
    """건당 기대 로그성장 -- **켈리 목적함수**이자 여기서 유일하게 안정적인 지표다.

    🔴735건 복리 중앙값은 꼬리 경로가 지배해 파라미터를 조금만 흔들어도 5배씩 튄다
    (2026-09-13 실측: 고정 2% 1.74 vs 2.5% 8.64). 13개 팔에서 최댓값을 고르면 잡음을
    고르게 된다 -- 물타기 칸이 표본마다 자리를 옮긴 것과 같은 함정이다.
    로그성장은 같은 표본에서 결정론적이고 n=8,000 이면 표준오차도 작다."""
    x = np.maximum(pool, -0.999999)
    return float(np.mean(np.log1p(x)) if (pool > -1.0).all()
                 else np.mean(np.where(pool <= -1.0, -10.0, np.log1p(x))))


def compound(pool, rng, trades=735, paths=2500):
    out = np.empty(paths); dead = 0
    for p in range(paths):
        w = 1.0
        for r in rng.choice(pool, trades):
            if r <= -1.0:
                w = 0.0; break
            w *= (1 + r)
            if w <= 0:
                w = 0.0; break
        if w == 0.0:
            dead += 1
        out[p] = w
    return float(np.median(out)), dead / paths


def regime_spread(res) -> float:
    """손절률의 국면 간 편차 -- 적응형 손절의 존재 이유다. 작을수록 좋다."""
    q = pd.qcut(res["atr"], 4, labels=False, duplicates="drop")
    rates = [res["stopped"][q == i].mean() for i in range(4) if (q == i).sum() > 30]
    return float(max(rates) - min(rates)) if len(rates) > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--acc", type=float, default=0.60)
    a = ap.parse_args()
    d = load_klines()
    print(f"5분봉 {len(d):,} ({d.timestamp.min().date()}~{d.timestamp.max().date()}) · "
          f"보유 {HOLD_BARS*5}분 · 배수 {LEV} · 정확도 {a.acc}")
    pred = predictions(d)
    rng = np.random.default_rng(SEED)
    print(f"모델 4시간 MAE 중앙 예측: {np.nanmedian(pred['LONG']['q50']):.2f}% "
          f"(q90 {np.nanmedian(pred['LONG']['q90']):.2f}%)\n")
    print(f"{'팔':>20} {'손절중앙':>10} {'평균L':>6} {'손절률':>7} {'국면편차':>8} "
          f"{'파산%':>7} {'건당 로그성장':>12}  (시드3)")
    rows = {}
    # 🔴**폭을 맞춘다.** 좁은 손절과 넓은 손절을 비교하면 «적응형 대 고정»이 아니라
    # «좁음 대 넓음»을 재게 된다 -- 크기 매칭을 안 해 틀린 답을 받은 게 이 저장소에서 세 번째다
    # (물타기 09-06 · 분할 09-13 · 여기). k 를 넓게 훑어 중앙 손절폭이 겹치게 만든다.
    ARMS = [
        # (이름, 손절종류, k, 위험예산)  위험예산 None = 배수 6 고정(현행 구조)
        ("무손절 6배", "none", 0.0, None),
        ("배수고정·가격2%", "fixed", 0.02, None),
        ("배수고정·가격3%", "fixed", 0.03, None),
        ("배수고정·모델k3.5", "model", 3.5, None),
        ("위험고정 6%·모델", "model", 3.5, 0.06),
        ("위험고정 12%·모델", "model", 3.5, 0.12),
        ("위험고정 18%·모델", "model", 3.5, 0.18),
        ("위험고정 12%·가격3%", "fixed", 0.03, 0.12),
        ("위험고정 18%·가격3%", "fixed", 0.03, 0.18),
    ]
    for lab, kind, k, rb in ARMS:
        # 시드를 3개 돌려 안정성을 같이 본다 -- 한 시드의 최댓값은 잡음일 수 있다.
        gs = []
        for sd in (SEED, SEED + 101, SEED + 202):
            rr = simulate(d, pred, np.random.default_rng(sd), a.n,
                          stop_kind=kind, k=k, acc=a.acc, risk_budget=rb)
            gs.append(log_growth(rr["ret"]))
            if sd == SEED:
                r = rr
        g, gsd = float(np.mean(gs)), float(np.std(gs))
        sp = regime_spread(r) if kind != "none" else float("nan")
        rows[lab] = (r, g, gsd, sp)
        sm = np.nanmedian(r["stops"]) * 100 if kind != "none" else float("nan")
        print(f"{lab:>20} {sm:>9.2f}% {np.mean(r['levs']):>6.2f} {100*r['stop_rate']:>6.1f}% "
              f"{100*sp:>7.1f}%p {100*r['ruin']:>6.2f}% {g:>+9.5f} ±{gsd:.5f}")
    best = max(rows.items(), key=lambda kv: kv[1][1])
    print(f"\n⇒ 건당 로그성장 최대: **{best[0]}** ({best[1][1]:+.5f})")
    lev_fixed = max(v[1] for kk, v in rows.items() if kk.startswith("배수고정"))
    risk_fixed = max(v[1] for kk, v in rows.items() if kk.startswith("위험고정"))
    print(f"   군별 최고: 배수고정 {lev_fixed:+.5f} · 위험고정 {risk_fixed:+.5f}")
    print(f"   ⭐위험고정이 배수고정을 이기나: **{'예' if risk_fixed > lev_fixed else '아니오'}**")
    print("   ⚠️시드 편차(±)보다 작은 차이는 읽지 않는다 -- 팔이 9개라 최댓값은 잡음일 수 있다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
