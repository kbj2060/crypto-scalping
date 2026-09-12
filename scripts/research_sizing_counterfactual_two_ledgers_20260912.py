#!/usr/bin/env python3
"""크기 반사실 — 오메가4.6.1 섀도우 원장과 사용자 실계좌 원장에 같은 질문 (2026-09-12).

질문: **방향·시점·청산을 전부 그대로 두고 크기만 바꾸면** 손익이 어떻게 달라지나.
판단은 재현 불가지만 크기는 반사실이 진짜 거래로 계산된다(2026-09-11 사용자 제안).

비교하는 크기 규칙 (전부 **평균 노출을 실제와 같게 정규화** -- 안 맞추면 그냥 레버리지 비교다):
  실제      원장에 찍힌 명목/수량 그대로
  균일      모든 건에 같은 크기
  역변동성  1/atr_pct. 09-11 무작위진입 82,167건에서 SD −18%·50배 청산율 13.9→9.1%
            로 검증된 규칙이라 여기서 새로 맞추지 않는다 -- 적용만 한다.
  역변동성+상한  위에 중앙값의 2배 상한. 두 원장 모두 «한 건»이 손실을 지배했다.

원장 둘 다 n<25 다. 여기서 **새 규칙을 고르면** 결과선택 편향이다
(feedback_outcome_selected_subset_inflates_metrics_20260908). 이미 검증된 규칙이
이 표본에서 어떻게 나오는지만 본다.

실행: python scripts/research_sizing_counterfactual_two_ledgers_20260912.py
"""
from __future__ import annotations

import json
import statistics as st
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "tmp" / "sizing_counterfactual_20260912"
OMEGA_LEDGER = ROOT / "data" / "live" / "trade_journal.jsonl"
USER_LEDGER = ROOT / "data" / "live" / "account_round_trips.jsonl"
ATR_BARS = 24          # 1시간봉 24개 = 24시간. 09-11 사이징 작업의 atr_pct(5분×288)와 같은 창.
CAP_MULT = 2.0


def klines(symbol: str, start_ms: int, end_ms: int) -> np.ndarray:
    """1시간봉 [ms, open, high, low, close]. 공개 엔드포인트, 인증 불필요. tmp 에 캐시."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{symbol}_1h.json"
    if path.exists():
        return np.array(json.loads(path.read_text()))
    bars, cur = [], start_ms
    while cur < end_ms:
        url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}"
               f"&interval=1h&startTime={cur}&limit=1000")
        with urllib.request.urlopen(url, timeout=20) as response:
            chunk = json.loads(response.read())
        if not chunk:
            break
        bars += [[int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])] for b in chunk]
        cur = int(chunk[-1][0]) + 3_600_000
        if len(chunk) < 1000:
            break
        time.sleep(0.2)
    path.write_text(json.dumps(bars))
    return np.array(bars)


def atr_pct_at(bars: np.ndarray, entry_ms: int) -> float | None:
    """진입 **직전** ATR_BARS 봉의 평균 트루레인지 / 종가. 진입 봉은 뺀다(그 봉의 움직임이
    진입을 만들었을 수 있고, 그걸 크기 입력으로 쓰면 같은 봉을 두 번 쓰는 셈이다)."""
    i = int(np.searchsorted(bars[:, 0], entry_ms)) - 1
    if i < ATR_BARS:
        return None
    window = bars[i - ATR_BARS:i]
    high, low, prev_close = window[:, 2], window[:, 3], bars[i - ATR_BARS - 1:i - 1, 4]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    return float(tr.mean() / bars[i, 4])


def summarize(moves, sizes, fee_rates):
    """손익 = 변동×크기 − 수수료×크기. **덧셈**으로 쌓는다(복리 아님).

    두 원장의 크기 단위가 다르다 -- 오메가는 계좌 대비 비율, 실계좌는 USDT 명목이다.
    복리식 `equity *= (1+x)` 는 x 가 비율일 때만 뜻이 있는데, 실계좌는 명목이 자산의
    6~30배라 그 식에 넣으면 지수가 폭발한다(실제로 10^26% 가 찍혔다). 게다가 그 배율에서
    한 건이 자산의 50% 를 날렸으므로 복리 곡선은 애초에 청산으로 끊긴다.
    덧셈 누적은 단위에 무관하고, 낙폭도 같은 곡선에서 잰다."""
    pnl = [m * s - f * s for m, s, f in zip(moves, sizes, fee_rates)]
    cum, peak, mdd = 0.0, 0.0, 0.0
    for x in pnl:
        cum += x
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return dict(per=st.mean(pnl), total=sum(pnl), mdd=mdd,
                sd=st.pstdev(pnl), worst=min(pnl))


def normalized(raw, target_mean):
    """평균 노출을 실제와 맞춘다. 이걸 안 하면 규칙 비교가 아니라 레버리지 비교가 된다."""
    scale = target_mean / st.mean(raw)
    return [x * scale for x in raw]


def rules(actual_sizes, atrs):
    """네 규칙의 건별 크기. atr 이 없는 건은 중앙값으로 대체(크기 축의 결측을 0으로 두면
    그 건이 통째로 사라져 비교 대상이 달라진다)."""
    target = st.mean(actual_sizes)
    med_atr = st.median([a for a in atrs if a])
    atrs = [a if a else med_atr for a in atrs]
    inv = normalized([1.0 / a for a in atrs], target)
    cap = CAP_MULT * st.median(inv)
    return {
        "실제": list(actual_sizes),
        "균일": [target] * len(actual_sizes),
        "역변동성": inv,
        f"역변동성+상한({CAP_MULT:g}×중앙)": normalized([min(x, cap) for x in inv], target),
    }


def report(name: str, moves, actual_sizes, fee_rates, atrs, unit: str, scale: float = 1.0) -> None:
    print(f"\n{'='*80}\n{name}  (n={len(moves)}, 단위 {unit})\n{'='*80}")
    print(f"{'크기 규칙':24s} {'건당':>9s} {'누적':>10s} {'최대낙폭':>10s} {'건당SD':>9s} {'최악':>10s} {'최대/중앙':>8s}")
    for label, sizes in rules(actual_sizes, atrs).items():
        s = summarize(moves, sizes, fee_rates)
        ratio = max(sizes) / st.median(sizes)
        print(f"{label:24s} {s['per']*scale:>+8.2f} {s['total']*scale:>+9.1f} {s['mdd']*scale:>9.1f} "
              f"{s['sd']*scale:>8.2f} {s['worst']*scale:>+9.2f} {ratio:>7.1f}x")
    corr = np.corrcoef(actual_sizes, moves)[0, 1]
    print(f"\n실제 크기–가격변동 상관 {corr:+.3f}"
          f"   (음수 = 큰 포지션일수록 나빴다)")


def load_omega():
    rows = [json.loads(l) for l in OMEGA_LEDGER.open() if l.strip()]
    trips = [r for r in rows if r.get("pnl_frac") is not None]
    moves, sizes, fees, atrs = [], [], [], []
    for r in trips:
        symbol = r.get("symbol") or "ETHUSDT"
        entry = datetime.fromisoformat(str(r["actual_opened_at"])).astimezone(timezone.utc)
        bars = klines(symbol, 1783000000000, int(time.time() * 1000))
        moves.append(r["gross_return_frac"])
        sizes.append(float(r.get("notional_exposure") or 1.0))
        fees.append(float(r.get("roundtrip_fee_rate") or 0.001))
        atrs.append(atr_pct_at(bars, int(entry.timestamp() * 1000)))
    return moves, sizes, fees, atrs


def load_user():
    """실계좌는 수량(ETH)이 크기다. 가격변동은 순손익 ÷ (수량 × 진입가) 로 되돌린다 --
    수수료가 이미 빠진 값이라 여기서 또 빼지 않는다(fee_rate = 0)."""
    rows = [json.loads(l) for l in USER_LEDGER.open() if l.strip()]
    bars = klines("ETHUSDT", 1783000000000, int(time.time() * 1000))
    moves, sizes, fees, atrs = [], [], [], []
    for r in rows:
        notional = r["max_qty"] * r["entry_price"]
        moves.append(r["net_pnl"] / notional)
        sizes.append(notional)
        fees.append(0.0)
        atrs.append(atr_pct_at(bars, int(r["entry_time"])))
    return moves, sizes, fees, atrs


def main() -> int:
    print(f"크기 반사실 — 방향·시점·청산 고정, 크기만 교체 (ATR {ATR_BARS}시간)")
    report("오메가4.6.1 섀도우 원장", *load_omega(), unit="계좌 대비 %", scale=100.0)
    report("사용자 실계좌 원장", *load_user(), unit="USDT")
    print("\n⚠️ 두 원장 다 n<25 다. 규칙을 여기서 고르면 결과선택 편향이므로,"
          "\n   역변동성은 09-11 무작위진입 82,167건에서 이미 검증된 것을 그대로 적용했다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
