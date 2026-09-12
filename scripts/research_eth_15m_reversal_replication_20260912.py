#!/usr/bin/env python3
"""15분 되돌림 재현 검정 — arXiv:2608.21888 (2026-09-12).

논문 주장(*Short-horizon mean reversion in cryptocurrency markets*, 2026-08):
  · 15분 지평에서 **183개 바이낸스 페어 중 90%**가 유의한 방향 되돌림(미국주식 187개 중 2.7%)
  · **"신호는 크기가 아니라 부호에 있다"** — lag-1 수익 자기상관은 0에 가까운데
    직전 봉 반대로 베팅하면 효과 대부분이 잡힌다
  · 되돌림은 **공격적 테이커 흐름 뒤에 집중**되고 **흐름 강도와 함께 커진다**
  · 호가 **깊이는 아무것도 조건짓지 않는다**
⚠️논문은 **비용 차감 전** 통계 유의성이다. 이 저장소의 벽은 그 다음이므로 둘 다 따로 낸다.

규약: 봉 i 가 **닫힌 뒤** 부호를 알고, 봉 i+1 **시가 진입 → 종가 청산**(체결 가능한 값).
      비용은 왕복 테이커 10bp / 메이커 7.8bp. 필요 정확도 a* = (1 + 비용/E|r|)/2.
귀무: 순환이동(B=400) — 봉 부호의 군집·빈도를 보존한다.

실행  python3 scripts/research_eth_15m_reversal_replication_20260912.py [--symbols ETHUSDT,BTCUSDT,...]
자체점검 --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
OUT = ROOT / "tmp/eth_15m_reversal_20260912"
COST = {"테이커": 10.0, "메이커": 7.8}
NSHIFT = 400
RNG = np.random.default_rng(20260912)


def fetch_15m(symbol: str, start: str, end: str) -> pd.DataFrame:
    """공개 klines. tmp 캐시 — 같은 심볼을 두 번 받지 않는다."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{symbol}_15m.parquet"
    if path.exists():
        return pd.read_parquet(path)
    cur = int(pd.Timestamp(start).timestamp() * 1000)
    stop = int(pd.Timestamp(end).timestamp() * 1000)
    rows = []
    while cur < stop:
        url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}"
               f"&interval=15m&startTime={cur}&limit=1500")
        with urllib.request.urlopen(url, timeout=25) as resp:
            chunk = json.loads(resp.read())
        if not chunk:
            break
        rows += [[int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]),
                  float(b[5]), float(b[9])] for b in chunk]
        cur = int(chunk[-1][0]) + 900_000
        if len(chunk) < 1500:
            break
        time.sleep(0.15)
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "tbb"])
    d["timestamp"] = pd.to_datetime(d.ts, unit="ms")
    d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    d.to_parquet(path, index=False)
    return d


def series(d: pd.DataFrame) -> dict[str, np.ndarray]:
    o = d.open.to_numpy(float); c = d.close.to_numpy(float)
    vol = d.vol.to_numpy(float); tbb = d.tbb.to_numpy(float)
    r = c / o - 1.0                                   # 봉 안 수익(시가→종가)
    imb = np.where(vol > 0, (2 * tbb - vol) / np.maximum(vol, 1e-12), 0.0)   # 테이커 매수 불균형
    return {"r": r, "imb": imb, "vol": vol, "ts": d.timestamp.to_numpy()}


def hit_stats(sig: np.ndarray, nxt: np.ndarray) -> tuple[float, int, float, float]:
    """적중률 · 표본 · 다음봉 평균 |수익|bp · **건당 실제 기대수익 bp**.

    🔴gross 를 `(2a−1)·E|r|` 로 내면 안 된다 — 그 식은 **이긴 봉과 진 봉의 크기가 같다**고
    가정한다. 되돌림 베팅에서는 그게 깨진다(ETH 95~99% 대역 실측: 이긴 봉 36.66bp vs
    진 봉 **47.89bp**). 자주 이기고 **크게 진다**. 그 식을 쓰면 ETH 대역이 6.24bp 로 보이는데
    실제 평균 부호수익은 **0.76bp**, 8배 과대평가다(2026-09-12 첫 판이 이걸 밟았다).
    대칭 배리어(TP=SL)라면 그 식이 맞다 — 고정 지평 수익에는 안 맞는다.
    """
    m = sig != 0
    if m.sum() == 0:
        return float("nan"), 0, float("nan"), float("nan")
    win = np.sign(nxt[m]) == sig[m]
    a = float(win.mean())
    e = float(np.abs(nxt[m]).mean()) * 1e4
    signed = float((np.where(win, 1.0, -1.0) * np.abs(nxt[m])).mean()) * 1e4
    return a, int(m.sum()), e, signed


def shift_null(sig: np.ndarray, nxt: np.ndarray, b: int = NSHIFT) -> tuple[float, float]:
    """순환이동 귀무 — 신호 부호열을 통째로 굴린다(군집·빈도 보존)."""
    obs = hit_stats(sig, nxt)[0]
    n = len(sig)
    nul = np.empty(b)
    for i in range(b):
        nul[i] = hit_stats(np.roll(sig, int(RNG.integers(100, n - 100))), nxt)[0]
    return float(nul.mean()), float((nul >= obs).mean())


def run(symbol: str, start: str, end: str, verbose: bool = True) -> dict:
    d = fetch_15m(symbol, start, end)
    s = series(d)
    r, imb, ts = s["r"], s["imb"], s["ts"]
    sig = -np.sign(r[:-1])                            # 직전 봉 반대
    nxt = r[1:]
    a, n, e, g = hit_stats(sig, nxt)
    nul, p = shift_null(sig, nxt)
    win = np.sign(nxt[sig != 0]) == sig[sig != 0]
    ew = float(np.abs(nxt[sig != 0])[win].mean()) * 1e4
    el = float(np.abs(nxt[sig != 0])[~win].mean()) * 1e4
    # 손익 비대칭을 반영한 손익분기 적중률: a·E_win − (1−a)·E_loss = 비용
    need = {k: (c + el) / (ew + el) for k, c in COST.items()}
    res = {"symbol": symbol, "n": n, "hit": a, "null": nul, "p": p, "abs_bp": e, "gross_bp": g,
           "need_taker": need["테이커"], "net_taker": g - COST["테이커"],
           "e_win_bp": ew, "e_loss_bp": el,
           "lag1_ret_corr": float(np.corrcoef(r[:-1], r[1:])[0, 1]),
           "lag1_sign_corr": float(np.corrcoef(np.sign(r[:-1]), np.sign(r[1:]))[0, 1]),
           "start": str(ts[0]), "end": str(ts[-1])}
    if verbose:
        print(f"{symbol:<12}n={n:>7,} 적중 {a:.4f} (귀무 {nul:.4f}, p={p:.3f}) · "
              f"이긴봉 {ew:>6.2f} / 진봉 {el:>6.2f}bp · **건당 {g:>5.2f}bp** · "
              f"필요 {need['테이커']:.3f} · 순익 {g - 10:>7.2f}")
    return res


def eth_conditioning(start: str, end: str) -> None:
    """논문의 조건부 주장 — «테이커 흐름 뒤에 집중되고 강도와 함께 커진다»."""
    d = fetch_15m("ETHUSDT", start, end)
    s = series(d)
    r, imb = s["r"], s["imb"]
    sig, nxt = -np.sign(r[:-1]), r[1:]
    flow = np.abs(imb[:-1])                    # 직전 봉의 테이커 불균형 «강도»
    mag = np.abs(r[:-1])                       # 직전 봉 «크기»
    for name, cond in (("테이커 불균형 |imb| 십분위", flow), ("직전 봉 크기 |r| 십분위", mag)):
        print(f"\n--- ETH · {name} ---")
        print(f"{'십분위':>7}{'n':>8}{'적중':>8}{'E|r|bp':>9}{'건당bp':>8}{'순익peg':>8}{'순익10':>9}")
        q = pd.qcut(cond, 10, labels=False, duplicates="drop")
        for k in range(int(np.nanmax(q)) + 1):
            m = q == k
            a, n, e, g = hit_stats(sig[m], nxt[m])
            print(f"{k+1:>7}{n:>8,}{a:>8.4f}{e:>9.2f}{g:>8.2f}{g-5.52:>8.2f}{g-10:>9.2f}")


def band_test(symbols: list[str], start: str, end: str, lo_q: float = 0.95, hi_q: float = 0.99) -> None:
    """가장 그럴듯한 셀(직전 봉 크기 상위 대역)에 이 저장소 규율을 건다.

    십분위가 단조라 «더 조이면 되지 않나» 가 자연스러운 다음 질문인데, 상위 0.5% 는 심볼마다
    엇갈린다(ETH 0.4913 붕괴 · XRP 0.6012, n=173). 그래서 표본이 남는 95~99% 대역으로 본다.
    ⚠️건당 손익은 **가장 싼 비용(양다리 peg 5.52bp)** 을 빼고 **일블록 부트**로 하한을 낸다 --
    평균만 보면 «간신히 양수» 로 보이는 셀이 여기서 갈린다.
    """
    rng = np.random.default_rng(20260912)
    print(f"\n{'='*112}\n직전 봉 크기 {lo_q:.0%}~{hi_q:.0%} 대역 — 순환이동 귀무 · 반기 안정성 · 일블록 부트(peg 5.52bp 차감)")
    print("=" * 112)
    print(f"{'심볼':<9}{'n':>6}{'건/일':>7}{'적중':>8}{'귀무':>8}{'p':>7}{'전반':>8}{'후반':>8}"
          f"{'gross':>8}{'평균순익':>9}{'부트95%하한':>12}")
    for sym in symbols:
        d = fetch_15m(sym, start, end)
        st = series(d)
        r, ts = st["r"], st["ts"][:-1]
        sig, nxt, mag = -np.sign(r[:-1]), r[1:], np.abs(r[:-1])
        lo, hi = np.quantile(mag, [lo_q, hi_q])
        m = (mag >= lo) & (mag < hi)
        a, n, e, g = hit_stats(sig[m], nxt[m])
        days = (pd.Timestamp(ts[-1]) - pd.Timestamp(ts[0])).days
        nul = np.array([hit_stats(np.roll(sig, int(rng.integers(100, len(sig) - 100)))[m], nxt[m])[0]
                        for _ in range(NSHIFT)])
        half = len(ts) // 2
        ar = np.arange(len(sig))
        h1 = hit_stats(sig[m & (ar < half)], nxt[m & (ar < half)])[0]
        h2 = hit_stats(sig[m & (ar >= half)], nxt[m & (ar >= half)])[0]
        idx = np.flatnonzero(m)
        v = np.where(np.sign(nxt[idx]) == sig[idx], 1, -1) * np.abs(nxt[idx]) * 1e4 - 5.52
        day = pd.to_datetime(ts[idx]).floor("D").to_numpy()
        ud = np.unique(day)
        by = {u: v[day == u] for u in ud}
        bs = np.array([np.concatenate([by[u] for u in rng.choice(ud, len(ud))]).mean() for _ in range(2000)])
        print(f"{sym:<9}{n:>6,}{n/days:>7.2f}{a:>8.4f}{nul.mean():>8.4f}{float((nul>=a).mean()):>7.3f}"
              f"{h1:>8.4f}{h2:>8.4f}{g:>8.2f}{v.mean():>9.2f}{np.percentile(bs,2.5):>12.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,BNBUSDT,DOGEUSDT,ADAUSDT,"
                                         "AVAXUSDT,LINKUSDT,LTCUSDT,DOTUSDT,TRXUSDT")
    ap.add_argument("--start", default="2025-09-01")
    ap.add_argument("--end", default="2026-08-20")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        r = np.array([0.01, -0.01, 0.01, -0.01, 0.01])          # 완전 되돌림
        sig, nxt = -np.sign(r[:-1]), r[1:]
        assert hit_stats(sig, nxt)[0] == 1.0, hit_stats(sig, nxt)
        r2 = np.array([0.01, 0.01, 0.01, 0.01, 0.01])           # 완전 지속
        assert hit_stats(-np.sign(r2[:-1]), r2[1:])[0] == 0.0
        a_, n_, e_, g_ = hit_stats(np.array([1.0, 1.0]), np.array([0.01, -0.01]))
        assert n_ == 2 and a_ == 0.5 and abs(e_ - 100.0) < 1e-9 and abs(g_) < 1e-9, (a_, e_, g_)
        # 🔴비대칭에서 두 식이 갈린다: 이겨서 +1bp 두 번, 져서 −10bp 한 번
        a2, _, e2, g2 = hit_stats(np.array([1.0, 1.0, 1.0]), np.array([1e-4, 1e-4, -1e-3]))
        assert abs(a2 - 2/3) < 1e-9 and abs(g2 - (1 + 1 - 10) / 3) < 1e-6, (a2, g2)
        naive = (2 * a2 - 1) * e2                      # +1.33bp
        assert abs(naive - g2) > 2.0 and naive > 0 > g2, (naive, g2)   # 실제는 −2.67bp: 부호까지 뒤집힌다
        print("selftest OK — 되돌림/지속 극단 · 건당 기대수익은 실제 부호수익 평균(비대칭 반영)")
        return 0

    print(f"=== 15분 되돌림 «직전 봉 반대» · {a.start} ~ {a.end} ===")
    print("규약: 봉 i 닫힘 → 봉 i+1 시가 진입 → 종가 청산 · 왕복 테이커 10bp\n")
    rows = []
    for sym in a.symbols.split(","):
        try:
            rows.append(run(sym.strip(), a.start, a.end))
        except Exception as exc:
            print(f"{sym}: 실패 {type(exc).__name__}: {exc}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "breadth.csv", index=False)
    d = pd.DataFrame(rows)
    sig_n = int(((d.p < 0.05) & (d.hit > d.null)).sum())
    print(f"\n{'='*104}")
    print(f"폭(breadth): 되돌림 유의 **{sig_n}/{len(d)}** 심볼 (논문: 183쌍 중 90%)")
    print(f"비용선 통과: 테이커 10bp 후 순익>0 **{int((d.net_taker > 0).sum())}/{len(d)}** · "
          f"메이커 7.8bp 후 **{int((d.gross_bp > 7.8).sum())}/{len(d)}**")
    print(f"lag-1 수익 자기상관 중앙 {d.lag1_ret_corr.median():+.5f} (논문: ≈0) · "
          f"부호 자기상관 중앙 {d.lag1_sign_corr.median():+.5f} (음수면 되돌림)")
    eth_conditioning(a.start, a.end)
    band_test([r["symbol"] for r in rows], a.start, a.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
