#!/usr/bin/env python3
"""단건 상한 배수를 **68왕복**으로 다시 잰다 (2026-09-12).

배경: `SIZING_CAP_MULT = 2.0` 은 **19왕복**에서 골랐다(`dashboard/server.py:950` 주석).
같은 날 원장을 수리·재구성해 표본이 **28 → 68건**이 됐으므로 다시 잰다.

## 반사실이 성립하는 이유 / 한계
방향·시점·청산을 실제 그대로 두고 **명목만 자른다**. 건당 손익은 명목에 비례하므로
`잘린 손익 = (실제 손익 / 실제 명목) × min(실제 명목, 상한)` 이 **산술**이다(예측이 아니다).
⚠️한계 둘: (1) 슬리피지는 크기에 비선형이라 **큰 건을 자르면 실제로는 더 유리하다** — 이 계산은
보수적이다. (2) 청산 리스크는 안 들어간다(잘랐으면 안 터졌을 건을 «똑같이 터진 채» 축소한다).

## 🔴고르는 순간 결과선택이 된다
68건에서 최고점을 집어 «이게 최적 배수»라 하면 그 건들에 맞춘 값이다. 그래서 세 가지를 본다:
  1. **곡선 전체** — 넓은 고원인가 칼날인가(고원이면 어디를 집든 비슷하다 = 고를 필요가 없다)
  2. **시간 분할** — 앞절반에서 고르고 뒷절반에서 재고, 반대로도. 배수가 이월되는가
  3. **부트스트랩** — 왕복 재표집 2,000회에서 각 배수의 순위가 얼마나 흔들리나
지표는 누적 손익만이 아니라 **MDD 와 건당 t** 도 같이 낸다(자본구조 판정은 손익만으론 안 된다,
2026-09-06 규율).

실행  python3 scripts/research_sizing_cap_mult_68trips_20260912.py
자체점검 --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/live/account_round_trips.jsonl"
MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 1e9]
NBOOT = 2000


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: int(r["entry_time"]))
    notional = np.array([abs(float(r["max_qty"]) * float(r["entry_price"])) for r in rows])
    pnl = np.array([float(r.get("net_pnl") or 0.0) for r in rows])
    ts = np.array([int(r["entry_time"]) for r in rows])
    return notional, pnl, ts


def apply_cap(notional: np.ndarray, pnl: np.ndarray, cap: float, match: bool = False) -> np.ndarray:
    """명목만 자른 뒤의 건당 손익. 손익은 명목에 비례한다.

    ⭐`match=True` 면 **평균 노출을 상한없음과 같게 맞춘 뒤** 비교한다. 안 맞추면 상한을 조일수록
    노출이 줄어 누적 손익이 같이 줄고, 그건 규칙의 우열이 아니라 그냥 «덜 걸었다» 는 뜻이다
    (2026-09-06 «크기 매칭 없는 짝비교 무효» 규율). 자른 뒤 전체에 상수를 곱하는 것이므로
    상한의 **모양**(어느 건을 얼마나 줄이나)만 남고 크기 차이는 사라진다.
    """
    capped = np.minimum(notional, cap)
    unit = pnl / notional
    if match:
        capped = capped * (notional.mean() / capped.mean())
    return unit * capped


def mdd(x: np.ndarray) -> float:
    eq = np.cumsum(x)
    return float(np.max(np.maximum.accumulate(eq) - eq))


def stats(v: np.ndarray) -> dict:
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {"sum": float(v.sum()), "mean": float(v.mean()), "mdd": mdd(v),
            "t": float(v.mean() / (sd / np.sqrt(len(v)))) if sd else float("nan"),
            "worst": float(v.min())}


def table(notional, pnl, label: str) -> dict[float, dict]:
    med = float(np.median(notional))
    out = {}
    print(f"\n=== {label} (n={len(pnl)} · 중앙 명목 {med:,.0f} USDT) ===")
    print(f"{'배수':>7}{'상한(USDT)':>12}{'걸린건':>7}{'누적(원)':>10}{'⭐누적(크기매칭)':>16}"
          f"{'MDD매칭':>10}{'t':>7}{'최악1건':>10}")
    for m in MULTS:
        cap = med * m
        v = apply_cap(notional, pnl, cap)
        vm = apply_cap(notional, pnl, cap, match=True)
        s, sm = stats(v), stats(vm)
        hit = int((notional > cap).sum())
        name = "상한없음" if m > 1e8 else f"{m:.2f}x"
        print(f"{name:>7}{('-' if m > 1e8 else f'{cap:,.0f}'):>12}{hit:>7}"
              f"{s['sum']:>10.2f}{sm['sum']:>16.2f}{sm['mdd']:>10.2f}{sm['t']:>7.2f}{sm['worst']:>10.2f}")
        out[m] = sm                      # 이후 선택·이월은 **크기 매칭본**으로만 한다
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        n = np.array([100.0, 400.0]); p = np.array([10.0, -80.0])
        v = apply_cap(n, p, 200.0)
        assert abs(v[0] - 10.0) < 1e-9, v            # 상한 미만은 그대로
        assert abs(v[1] - (-40.0)) < 1e-9, v         # 400 → 200 이면 손익도 절반
        assert abs(mdd(np.array([1.0, -3.0, 2.0])) - 3.0) < 1e-9, mdd(np.array([1.0, -3.0, 2.0]))
        assert mdd(np.array([1.0, 1.0])) == 0.0
        print("selftest OK — 비례 축소·MDD")
        return 0

    notional, pnl, ts = load()
    full = table(notional, pnl, "전체")

    # 시간 분할 — 앞에서 고르고 뒤에서 재고, 반대로도
    half = len(pnl) // 2
    a_n, a_p = notional[:half], pnl[:half]
    b_n, b_p = notional[half:], pnl[half:]
    first = table(a_n, a_p, "앞 절반(오래된 쪽)")
    second = table(b_n, b_p, "뒤 절반(최근)")

    def best(tab):
        cand = {m: s for m, s in tab.items() if m <= 1e8}
        return max(cand, key=lambda m: cand[m]["sum"])

    print("\n" + "=" * 96)
    print("이월 검정 — 한쪽에서 고른 배수가 다른 쪽에서도 좋은가")
    print("=" * 96)
    for src, dst, sn, dn, dp in (("앞", "뒤", first, b_n, b_p), ("뒤", "앞", second, a_n, a_p)):
        m = best(sn)
        med_d = float(np.median(dn))
        v = apply_cap(dn, dp, med_d * m, match=True)
        no = apply_cap(dn, dp, np.inf, match=True)
        print(f"  {src}절반 최적 {m:.2f}x → {dst}절반에서 누적 {v.sum():>8.2f} "
              f"(그쪽 상한없음 {no.sum():>8.2f}, 그쪽 자체 최적 {best(second if dst=='뒤' else first):.2f}x)")

    # 부트스트랩 — 왕복 재표집에서 각 배수가 «상한없음» 을 이기는 비율
    print("\n" + "=" * 96)
    print(f"부트스트랩 {NBOOT}회 (왕복 재표집) — 상한없음 대비 이기는 비율 · 누적손익 중앙값")
    print("=" * 96)
    rng = np.random.default_rng(20260912)
    idx = rng.integers(0, len(pnl), size=(NBOOT, len(pnl)))
    med = float(np.median(notional))
    base = np.array([apply_cap(notional[i], pnl[i], np.inf, match=True).sum() for i in idx])
    print(f"{'배수':>7}{'이김%':>8}{'누적 중앙':>11}{'2.5%':>10}{'97.5%':>10}")
    for m in MULTS:
        if m > 1e8:
            continue
        got = np.array([apply_cap(notional[i], pnl[i], float(np.median(notional[i])) * m, match=True).sum()
                        for i in idx])
        print(f"{m:>6.2f}x{float((got > base).mean()) * 100:>8.1f}{np.median(got):>11.2f}"
              f"{np.percentile(got, 2.5):>10.2f}{np.percentile(got, 97.5):>10.2f}")
    print(f"\n참고: 상한없음 누적 중앙 {np.median(base):.2f} · 현재 배포값 2.0x · 중앙 명목 {med:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
