"""**보유시간별 엣지** — 방향·시점은 실제 그대로 두고 보유시간만 고정 H 로 바꾼 반사실 (2026-09-13).

사용자: *"집행·크기·보유시간 모델 … 현재 상황에서 어떻게 가져가는 게 좋을지"*.
크기는 보유시간의 함수로 이미 정해진다(`live_eth_risk_sizing_policy_20260913`). 그런데
**보유시간 자체를 고르는 근거**가 없었다 -- 화면 기본값이 1일이었을 뿐이다.

## 방법
크기 반사실(user_entry_size_decomposition_20260911)과 같은 논리다: 판단은 재현 못 하지만
**진입 방향·시점을 실제로 고정**하면 나머지 축(여기서는 보유시간)은 진짜 거래로 계산된다.
  · 각 왕복의 진입 VWAP 에서 H 분 뒤 1분봉 종가로 청산했다고 치고 단위당 수익을 잰다.
  · 비용은 대시보드 경로(static 진입 + peg 청산) 5.88bp 왕복.
  · **고정 청산**이다 -- MFE 같은 사후 최적 청산은 진입을 평가 못 한다
    (feedback_oracle_exit_cannot_evaluate_entry_20260913).
  · 신뢰구간은 **일 군집 부트스트랩**(같은 날 왕복은 한 덩어리). 고정 H 가 길면 이웃 왕복의
    창이 겹치므로 겹침 수를 같이 찍는다.

## 이 표가 답하는 것
보유시간 H 의 로그성장 근사  g(H) = L(H)·(μ_H − 비용) − ½·(L(H)·σ_H)²
L(H) 는 생존 모델의 허용 배수(검증 중앙값 · 라이브는 그때그때 값). μ_H·σ_H 가 이 표다.
⚠️n=68 · ETH 단일 · 33일 · 사용자 재량 진입 -- **순위**를 읽는 용도이지 절대값이 아니다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRIPS = ROOT / "data/live/account_round_trips.jsonl"
HOLDS = (60, 120, 240, 480, 1440)
COST_BP = 5.88                       # static 진입 2.0/peg 청산 2.0 + 폴백 기대분. 대시보드 실측.
# 검증구간 중앙 안전MAE(%) -- live_eth_mae_quantile_model_20260913 verify 출력. 라이브는 매 5분 갱신.
SAFE_MAE_REF = {60: 3.54, 120: 5.17, 240: 7.67, 480: 11.37, 1440: 21.66}
CAP_X = 8.0                          # 순자산 상한(사용자 결정 2026-09-13)
SEED = 20260913


def fetch_1m(start_ms: int, end_ms: int) -> pd.DataFrame:
    """공개 엔드포인트. 로컬 1분봉 CSV 가 2026-07-31 까지라 왕복 구간을 못 덮는다."""
    rows, t = [], start_ms
    while t < end_ms:
        u = (f"https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval=1m"
             f"&startTime={t}&limit=1500")
        d = json.load(urllib.request.urlopen(u, timeout=15))
        if not d:
            break
        rows += d
        t = d[-1][0] + 60_000
        time.sleep(0.15)
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["ts", "o", "h", "l", "c"]
    return df.drop_duplicates("ts").astype(float).reset_index(drop=True)


def load_trips() -> list[dict]:
    rows = [json.loads(l) for l in TRIPS.read_text().splitlines() if l.strip()]
    return sorted((x for x in rows if x.get("closed") and x.get("side") in ("LONG", "SHORT")),
                  key=lambda x: x["entry_time"])


def fixed_hold_returns(trips: list[dict], kl: pd.DataFrame) -> pd.DataFrame:
    """행 = 왕복, 열 = 보유시간. 값 = 단위당 순수익(bp, 비용 차감). 추가로 실제 청산 열."""
    ts = kl.ts.to_numpy()
    close, hi, lo = kl.c.to_numpy(), kl.h.to_numpy(), kl.l.to_numpy()
    out = []
    for x in trips:
        sgn = 1.0 if x["side"] == "LONG" else -1.0
        e = float(x["entry_price"])
        i0 = int(np.searchsorted(ts, x["entry_time"], side="right"))    # 진입 뒤 첫 완결봉
        row = {"day": pd.Timestamp(x["entry_time"], unit="ms").strftime("%Y-%m-%d"),
               "hold_actual_min": (x["exit_time"] - x["entry_time"]) / 60_000,
               "actual": sgn * (float(x["exit_price"]) / e - 1) * 1e4 - COST_BP}
        for H in HOLDS:
            j = min(len(ts) - 1, i0 + H - 1)
            row[H] = sgn * (close[j] / e - 1) * 1e4 - COST_BP
            adverse = (e - lo[i0:j + 1].min()) / e if sgn > 0 else (hi[i0:j + 1].max() - e) / e
            row[f"mae{H}"] = 100 * max(0.0, adverse)
        out.append(row)
    return pd.DataFrame(out)


def day_cluster_ci(df: pd.DataFrame, col, n_boot: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    groups = [g[col].to_numpy() for _, g in df.groupby("day")]
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(np.concatenate([groups[k] for k in pick]).mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def growth(L: float, mu_bp: float, sd_bp: float) -> float:
    """건당 로그성장 근사. μ 는 이미 비용 차감."""
    r, s = mu_bp / 1e4, sd_bp / 1e4
    return L * r - 0.5 * (L * s) ** 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--klines", help="1분봉 parquet(ts,o,h,l,c). 없으면 API 에서 받는다")
    a = ap.parse_args()
    trips = load_trips()
    t0, t1 = trips[0]["entry_time"] - 60_000, max(x["exit_time"] for x in trips) + 1440 * 60_000
    kl = pd.read_parquet(a.klines) if a.klines else fetch_1m(t0, t1)
    kl = kl[(kl.ts >= t0) & (kl.ts <= t1)].reset_index(drop=True)
    df = fixed_hold_returns(trips, kl)
    n_days = df.day.nunique()
    print(f"왕복 {len(df)} · 독립 일수 {n_days} · 1분봉 {len(kl):,} · 비용 {COST_BP}bp 왕복\n")
    print(f"{'보유':>7} {'μ(bp)':>7} {'σ(bp)':>7} {'t':>5} {'CI95(일군집)':>16} {'승률':>6} "
          f"{'겹침':>4} {'MAE95%':>7} {'L(H)':>6} {'g(H)':>8}")
    table = {}
    best = None
    ents = df.index.to_numpy()
    for H in HOLDS:
        v = df[H].to_numpy()
        mu, sd = v.mean(), v.std(ddof=1)
        lo, hi = day_cluster_ci(df, H)
        t = mu / (sd / np.sqrt(len(v)))
        # 고정 H 창이 다음 왕복 진입과 겹치는 수 -- 길수록 독립성이 준다
        starts = np.array([x["entry_time"] for x in trips])
        overlap = int((starts[1:] < starts[:-1] + H * 60_000).sum())
        L = min(CAP_X, 100.0 / SAFE_MAE_REF[H])
        g = growth(L, mu, sd)
        table[H] = {"mu_bp": round(float(mu), 2), "sd_bp": round(float(sd), 2),
                    "ci95": [round(lo, 2), round(hi, 2)], "win": round(float((v > 0).mean()), 3),
                    "overlap": overlap, "L": round(L, 2), "growth": round(g, 5)}
        if best is None or g > best[1]:
            best = (H, g)
        print(f"{H:>6}분 {mu:>7.2f} {sd:>7.2f} {t:>5.2f} [{lo:>6.2f}, {hi:>6.2f}] "
              f"{100*(v>0).mean():>5.1f}% {overlap:>4} {np.quantile(df[f'mae{H}'], .95):>6.2f}% "
              f"{L:>6.2f} {g:>8.5f}")
    act = df["actual"].to_numpy()
    lo, hi = day_cluster_ci(df, "actual")
    print(f"{'실제청산':>7} {act.mean():>7.2f} {act.std(ddof=1):>7.2f} "
          f"{act.mean()/(act.std(ddof=1)/np.sqrt(len(act))):>5.2f} [{lo:>6.2f}, {hi:>6.2f}] "
          f"{100*(act>0).mean():>5.1f}%   (중앙 보유 {np.median(df.hold_actual_min):.0f}분)")
    print(f"\n⇒ 로그성장 최대 보유시간: **{best[0]}분** (g={best[1]:.5f})")
    print("   L(H) 는 검증 중앙 안전MAE 기준(라이브는 그때그때 값이라 순위가 바뀔 수 있다)")
    print("\n# live_eth_trade_plan_20260913.EDGE_BY_HOLD 에 붙여넣는 값 (μ, σ 는 비용 차감 후 bp)")
    print("EDGE_BY_HOLD = {" + ", ".join(f"{H}: ({t['mu_bp']}, {t['sd_bp']})"
                                          for H, t in table.items()) + "}")
    print(f"EDGE_ACTUAL_BP = {act.mean():.2f}   # 사용자 실제 청산(같은 비용 기준)")

    # 이 스크립트의 구조 주장을 고정한다
    assert len(df) == 68, "원장이 바뀌었다 -- 상단 상수(비용·안전MAE)와 결론을 다시 볼 것"
    assert all(np.isfinite(df[H]).all() for H in HOLDS), "1분봉이 왕복 구간을 못 덮는다"
    assert (df[f"mae1440"] >= df[f"mae60"]).all(), "역행폭은 보유시간에 단조여야 한다"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
