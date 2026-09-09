#!/usr/bin/env python3
"""증거신호 8종: **어느 신호가 잘 맞고 어느 신호가 잘 틀리는가** (2026-09-09).

사용자: *"어떤 게 잘 맞고 어떤 게 잘 틀리는 지 알려주고 어떤 전략이 수익성이 좋을지 연구해줘"*

설계 (이 저장소 규율 준수):
  · **결과로 거르지 않는다** -- 발동 전건을 센다. 커버리지(건/일)를 항상 같이 보고한다.
  · **동일 측면 무작위 진입 귀무**와 함께 읽는다. VAL/OOS 창이 전부 하락장이면 무작위 숏도
    양수다(feedback_side_asymmetry_needs_same_side_null). 원시 bp 격차로 판정하지 않는다.
  · 진입은 **발동 봉의 다음 봉 시가**(인과적, 라벨 규약과 동일). 청산은 고정 홀딩 종가.
  · 비용 10bp 왕복(테이커). 이 저장소의 표준.
  · 창을 **전반/후반으로 쪼개** 두 창에서 같은 부호가 나오는지 본다(단일 창 우연 방지).

출력: tmp/eth_signal_map_20260909/hitrate_evidence.csv + 콘솔 표
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import live_evidence_signal_dashboard_20260823 as EV   # noqa: E402
import build_eth_anchor_label_dataset_20260907 as B    # noqa: E402

OUT = ROOT / "tmp/eth_signal_map_20260909"
KL = "https://fapi.binance.com/fapi/v1/klines"
COST_BP = 10.0
HORIZONS = [12, 24, 48, 144]          # 1h · 2h · 4h · 12h
NBOOT = 400
RNG = np.random.default_rng(20260909)


CACHE = ROOT / "tmp/eth_signal_map_20260909/klcache"


def page(sym: str, start_ms: int, end_ms: int, interval: str = "5m") -> pd.DataFrame:
    """페이징 + 429 백오프 + 디스크 캐시. 1년치는 71페이지라 무보호로 돌리면 레이트리밋에 걸린다."""
    CACHE.mkdir(parents=True, exist_ok=True)
    ck = CACHE / f"{sym}_{interval}_{start_ms//60000}_{end_ms//3600000}.parquet"
    if ck.exists():
        return pd.read_parquet(ck)
    out, cur = [], start_ms
    while cur < end_ms:
        for attempt in range(6):
            r = requests.get(KL, params={"symbol": sym, "interval": interval, "limit": 1500,
                                         "startTime": cur}, timeout=25)
            if r.status_code == 429:
                w = 5 * (attempt + 1)
                print(f"    429 -- {w}s 대기", flush=True); time.sleep(w); continue
            r.raise_for_status(); break
        dd = r.json()
        if not dd: break
        out += dd; cur = dd[-1][0] + 1
        if len(dd) < 1500: break
        time.sleep(0.4)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "trades",
            "taker_buy_base", "tq", "ignore"]
    df = pd.DataFrame(out, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[df["close_time"] < int(time.time() * 1000)]
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(ck); return df


def fwd_bp(entry_open: np.ndarray, close: np.ndarray, idx: np.ndarray, H: int,
           long: bool) -> np.ndarray:
    """발동 봉 i -> 진입 open[i+1], 청산 close[i+H]. 방향 부호 적용 후 비용 차감(bp)."""
    e = entry_open[idx + 1]
    x = close[idx + H]
    raw = (x - e) / e * 1e4
    return (raw if long else -raw) - COST_BP


def boot_null(entry_open, close, n: int, H: int, long: bool, lo: int, hi: int) -> np.ndarray:
    """같은 측면·같은 건수의 무작위 진입 귀무 분포(평균 bp)."""
    pool = np.arange(lo, hi)
    return np.array([fwd_bp(entry_open, close, RNG.choice(pool, n, replace=False), H, long).mean()
                     for _ in range(NBOOT)])


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=float, default=365.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    now = int(time.time() * 1000); t0 = now - int(a.days * 86400 * 1000)
    warm = 1100 * 300_000
    print(f"[1/3] klines {a.days:.0f}일 + 웜업 …", flush=True)
    kl = page("ETHUSDT", t0 - warm, now); btc = page("BTCUSDT", t0 - warm, now)
    try:
        fund = EV.fetch_funding_history(limit=1000)
        fund["calc_time"] = pd.to_datetime(fund["calc_time"]).dt.tz_localize(None)
    except Exception as e:
        print(f"  ⚠️펀딩 실패({e})"); fund = None
    print(f"  5분봉 {len(kl):,} ({kl.timestamp.iloc[0]} ~ {kl.timestamp.iloc[-1]} UTC)", flush=True)

    print("[2/3] 증거신호 8종 …", flush=True)
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=fund)
    ts = sig["timestamp"].to_numpy()
    op = sig["open"].to_numpy(float); cl = sig["close"].to_numpy(float)
    n = len(sig)
    Hmax = max(HORIZONS)
    warm_end = 900                                   # 지표 웜업(864봉) 이후만
    lo, hi = warm_end, n - Hmax - 2                  # 진입 가능 구간
    mid = (lo + hi) // 2
    days_span = (pd.Timestamp(ts[hi]) - pd.Timestamp(ts[lo])).total_seconds() / 86400
    print(f"  평가 구간 {pd.Timestamp(ts[lo])} ~ {pd.Timestamp(ts[hi])} ({days_span:.0f}일)", flush=True)

    print("[3/3] 측면별 적중·수익 + 무작위 귀무 …", flush=True)
    rows = []
    for s in B.SIGNALS:
        for side, long in (("bottom", True), ("top", False)):
            f = sig[f"{side}_{s}"].fillna(False).to_numpy(bool)
            idx = np.flatnonzero(f); idx = idx[(idx >= lo) & (idx <= hi)]
            if len(idx) < 30:
                continue
            for H in HORIZONS:
                r = fwd_bp(op, cl, idx, H, long)
                nul = boot_null(op, cl, len(idx), H, long, lo, hi)
                p_hi = float((nul >= r.mean()).mean())          # 단측: 신호가 귀무보다 좋은가
                h1 = idx[idx < mid]; h2 = idx[idx >= mid]
                rows.append(dict(signal=s, side=side, H=H, n=len(idx),
                                 per_day=round(len(idx) / days_span, 2),
                                 hit=round(float((r > 0).mean()), 4),
                                 mean_bp=round(float(r.mean()), 2),
                                 med_bp=round(float(np.median(r)), 2),
                                 null_mean=round(float(nul.mean()), 2),
                                 excess=round(float(r.mean() - nul.mean()), 2),
                                 p_vs_null=round(p_hi, 4),
                                 h1_bp=round(float(fwd_bp(op, cl, h1, H, long).mean()), 2) if len(h1) > 20 else None,
                                 h2_bp=round(float(fwd_bp(op, cl, h2, H, long).mean()), 2) if len(h2) > 20 else None))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "hitrate_evidence.csv", index=False)

    # 참고: 항상롱/항상숏 기준선
    allidx = np.arange(lo, hi)
    print(f"\n{'='*118}\n기준선(전 봉 진입, 비용 {COST_BP:.0f}bp 차감)")
    for H in HORIZONS:
        L = fwd_bp(op, cl, allidx, H, True).mean(); S = fwd_bp(op, cl, allidx, H, False).mean()
        print(f"  H={H:>3}봉({H*5:>3}분)  항상롱 {L:+7.2f}bp · 항상숏 {S:+7.2f}bp")
    print("="*118)
    for H in HORIZONS:
        print(f"\n■ H={H}봉 ({H*5}분 보유)  ─ 초과 = 신호 − 같은측면 무작위 진입, p는 단측 부트스트랩")
        print(f"{'신호':<26}{'측면':>6}{'건수':>7}{'건/일':>7}{'적중':>7}{'평균bp':>9}"
              f"{'귀무bp':>9}{'초과':>8}{'p':>8}{'전반':>8}{'후반':>8}")
        q = D[D.H == H].sort_values("excess", ascending=False)
        for r in q.itertuples():
            star = "⭐" if (r.p_vs_null <= 0.05 and (r.h1_bp or 0) > 0 and (r.h2_bp or 0) > 0) else "  "
            print(f"{star}{r.signal:<24}{'바닥' if r.side=='bottom' else '천장':>6}{r.n:>7}{r.per_day:>7.2f}"
                  f"{r.hit*100:>6.1f}%{r.mean_bp:>9.2f}{r.null_mean:>9.2f}{r.excess:>+8.2f}"
                  f"{r.p_vs_null:>8.3f}{(r.h1_bp if r.h1_bp is not None else float('nan')):>8.2f}"
                  f"{(r.h2_bp if r.h2_bp is not None else float('nan')):>8.2f}")
    print(json.dumps({"done": True, "rows": len(D), "days": round(days_span, 1)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
