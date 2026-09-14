#!/usr/bin/env python3
"""liquidity_sweep Phase 0 — 09-09 의 유일한 생존 셀을 **비겹침 블록**으로 검정한다 (2026-09-14).

사용자 *"Liquidity 쪽으로 연구해줘"*.

배경: 09-09 수익성 스크린에서 `liquidity_sweep 천장 H=48` 이 1년 두 반기 gross 4.12/4.03bp 로
이 저장소에서 본 가장 안정적인 셀이었고, "살리려면 조건부 필터로 6~7bp 로 올린다"를 미검정
TODO 로 남겼다. 그 TODO 를 하기 **전에** 09-12 가 Donchian 을 죽인 검정을 먼저 건다:

  ⭐6.6건/일 × H=48(4시간 보유) 면 이벤트가 대량 겹친다. 겉보기 n 이 커도 독립 관측은
    보유기간 길이의 블록 개수뿐이다. 블록당 하나씩만 세면 t 가 무너지는지 본다.

09-09 대비 바뀐 것:
  · 표본 365일 → **1,745일**(2021-12~2026-09, 로컬 CSV, API 아님)
  · 비용 차감 전 **gross** 로 보고(비용선은 읽는 쪽에서 5.52/7.8/10bp 로 판단)
  · 비겹침 블록 t + 일군집 부트 + **연도별 분해**(smt 처럼 레짐 하나가 전부인지)

재구현 금지: sweep 정의는 라이브 `live_evidence_signal_dashboard_20260823.compute_signals` 를
그대로 호출한다(식 두 벌 금지 — liqmap 연구 규율).

출력: tmp/eth_liquidity_sweep_20260914/phase0_*.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))

OUT = ROOT / "tmp/eth_liquidity_sweep_20260914"
CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
HORIZONS = [12, 24, 48, 144]        # 1h · 2h · 4h · 12h (09-09 과 동일)
WARMUP = 900                        # 지표 웜업(864봉) 이후만 — 09-09 과 동일
NBOOT = 400
RNG = np.random.default_rng(20260914)


def gross_bp(op: np.ndarray, cl: np.ndarray, idx: np.ndarray, H: int, long: bool) -> np.ndarray:
    """발동 봉 i -> 진입 open[i+1](인과), 청산 close[i+H]. **비용 차감 안 함**(gross)."""
    e = op[idx + 1]
    raw = (cl[idx + H] - e) / e * 1e4
    return raw if long else -raw


def thin_nonoverlap(idx: np.ndarray, H: int) -> np.ndarray:
    """보유기간 H 안에 겹치는 이벤트를 솎아낸다 — 첫 건 채택 후 H봉 건너뛰기(greedy).
    09-12 규율: 블록 길이는 **보유기간 이상**이어야 한다."""
    keep, last = [], -10**9
    for i in idx:
        if i - last >= H:
            keep.append(i); last = i
    return np.asarray(keep, dtype=np.int64)


def day_cluster_boot(vals: np.ndarray, days: np.ndarray, B: int = 2000) -> tuple[float, float]:
    """일 단위로 군집 재표집한 평균의 95% CI. 같은 날 이벤트는 독립이 아니다."""
    uniq = np.unique(days)
    by = {d: vals[days == d] for d in uniq}
    means = np.empty(B)
    for b in range(B):
        pick = RNG.choice(uniq, len(uniq), replace=True)
        means[b] = np.concatenate([by[d] for d in pick]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def same_side_null(op, cl, n: int, H: int, long: bool, lo: int, hi: int) -> np.ndarray:
    """같은 측면·같은 건수의 무작위 진입 귀무(gross 평균 분포). 창의 표류를 흡수한다."""
    pool = np.arange(lo, hi)
    return np.array([gross_bp(op, cl, RNG.choice(pool, n, replace=False), H, long).mean()
                     for _ in range(NBOOT)])


def selftest() -> None:
    # thin_nonoverlap: 간격이 H 미만이면 버린다
    assert list(thin_nonoverlap(np.array([0, 5, 10, 48, 60, 96]), 48)) == [0, 48, 96]
    assert list(thin_nonoverlap(np.array([7]), 12)) == [7]
    # gross_bp: 부호·진입시점. open[i+1]=100 -> close[i+H]=101 이면 롱 +100bp, 숏 -100bp
    op = np.array([0., 100., 0., 0.]); cl = np.array([0., 0., 0., 101.])
    assert abs(gross_bp(op, cl, np.array([0]), 3, True)[0] - 100.0) < 1e-9
    assert abs(gross_bp(op, cl, np.array([0]), 3, False)[0] + 100.0) < 1e-9
    print("selftest OK")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume",
                                    "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return 0
    selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    import live_evidence_signal_dashboard_20260823 as EV  # noqa: E402

    print("[1/3] 로컬 CSV …", flush=True)
    kl = load(CSV)
    btc = load(BTC) if BTC.exists() else None
    print(f"  ETH 5분봉 {len(kl):,}  {kl.timestamp.iloc[0]} ~ {kl.timestamp.iloc[-1]} UTC", flush=True)

    print("[2/3] 라이브 compute_signals 호출(식 두 벌 금지) …", flush=True)
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=None)
    op = sig["open"].to_numpy(float); cl = sig["close"].to_numpy(float)
    ts = sig["timestamp"].to_numpy()
    day = sig["timestamp"].dt.floor("D").astype("int64").to_numpy()
    year = sig["timestamp"].dt.year.to_numpy()
    n = len(sig); lo, hi = WARMUP, n - max(HORIZONS) - 2
    span = (pd.Timestamp(ts[hi]) - pd.Timestamp(ts[lo])).total_seconds() / 86400
    print(f"  평가 {pd.Timestamp(ts[lo]).date()} ~ {pd.Timestamp(ts[hi]).date()} ({span:.0f}일)", flush=True)

    print("[3/3] 전건 vs 비겹침 블록 …", flush=True)
    rows, yrows = [], []
    for side, long in (("top", False), ("bottom", True)):
        f = sig[f"{side}_liquidity_sweep"].fillna(False).to_numpy(bool)
        idx = np.flatnonzero(f); idx = idx[(idx >= lo) & (idx <= hi)]
        for H in HORIZONS:
            g = gross_bp(op, cl, idx, H, long)
            nul = same_side_null(op, cl, len(idx), H, long, lo, hi)
            kept = thin_nonoverlap(idx, H)
            gk = gross_bp(op, cl, kept, H, long)
            t = float(gk.mean() / (gk.std(ddof=1) / np.sqrt(len(gk)))) if len(gk) > 2 else np.nan
            # 비겹침 블록의 초과분 t (같은측면 귀무 평균을 빼고)
            ex = gk - nul.mean()
            t_ex = float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))) if len(ex) > 2 else np.nan
            clo, chi = day_cluster_boot(g - nul.mean(), day[idx])
            rows.append(dict(side=side, H=H, n_all=len(idx), per_day=round(len(idx)/span, 2),
                             n_block=len(kept), gross_all=round(float(g.mean()), 2),
                             null_gross=round(float(nul.mean()), 2),
                             excess_all=round(float(g.mean()-nul.mean()), 2),
                             excess_ci_lo=round(clo, 2), excess_ci_hi=round(chi, 2),
                             gross_block=round(float(gk.mean()), 2),
                             excess_block=round(float(ex.mean()), 2),
                             t_block=round(t, 2), t_block_excess=round(t_ex, 2),
                             hit_all=round(float((g > 0).mean()), 4)))
            if H == 48:
                for y in np.unique(year[idx]):
                    m = year[idx] == y
                    if m.sum() < 50: continue
                    yrows.append(dict(side=side, year=int(y), n=int(m.sum()),
                                      gross=round(float(g[m].mean()), 2),
                                      hit=round(float((g[m] > 0).mean()), 4)))
    D = pd.DataFrame(rows); Y = pd.DataFrame(yrows)
    D.to_csv(OUT / "phase0_block.csv", index=False); Y.to_csv(OUT / "phase0_by_year.csv", index=False)

    print(f"\n{'='*126}\nliquidity_sweep — 전건 vs 비겹침 블록  (gross = 비용 차감 전, 초과 = 같은측면 무작위 진입 대비)")
    print(f"{'측면':>5}{'H':>5}{'건수':>8}{'건/일':>7}{'블록':>7}{'gross':>8}{'귀무':>8}{'초과':>8}"
          f"{'초과CI95':>18}{'블록gross':>10}{'블록초과':>9}{'t블록':>7}{'t초과':>7}{'적중':>7}")
    for r in D.itertuples():
        print(f"{'천장' if r.side=='top' else '바닥':>5}{r.H:>5}{r.n_all:>8}{r.per_day:>7.2f}{r.n_block:>7}"
              f"{r.gross_all:>8.2f}{r.null_gross:>8.2f}{r.excess_all:>+8.2f}"
              f"{f'[{r.excess_ci_lo:+.2f},{r.excess_ci_hi:+.2f}]':>18}"
              f"{r.gross_block:>10.2f}{r.excess_block:>+9.2f}{r.t_block:>7.2f}{r.t_block_excess:>7.2f}"
              f"{r.hit_all*100:>6.1f}%")
    print(f"\n■ H=48 연도별 (레짐 하나가 전부인지)")
    for r in Y.itertuples():
        print(f"  {'천장' if r.side=='top' else '바닥'} {r.year}  n={r.n:>5}  gross {r.gross:>+7.2f}bp  적중 {r.hit*100:>5.1f}%")
    print("="*126)
    print(json.dumps({"done": True, "days": round(span, 1), "bars": int(n)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
