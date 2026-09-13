"""**사용자의 분할 진입이 실제로 돈을 잃었나** — 물타기 금지의 근거를 되묻는다 (2026-09-13).

사용자: *"물타기가 왜 금지지? 대부분 물타기로 수익을 냈던 것 같은데 어떤 근거로?"*

## 되묻게 된 경위
2026-09-13 에 나는 «역행 중 추가 금지»를 진입 submit 게이트로 올렸다. 근거로 든 2026-09-06
감사(크기 매칭 후 24/24 전패)는 **칩 신호 위의 시뮬레이션 전략**을 잰 것이지 사용자의 실제
거래가 아니다. **다른 모집단의 결론을 사용자에게 적용했다** -- 이 저장소가 반복해서 데인 실수다.

## 한계 — 이 스크립트가 «물타기»를 직접 못 잰다
거래소 `userTrades` 는 최근 7일만 준다(실측 5건). 왕복 원장은 수리 과정에서 체결 단위를 잃고
`fills`(체결 개수)와 진입 **레그 VWAP** 만 남겼다. 그래서 «추가가 역행 중이었나(물타기)
순행 중이었나(피라미딩)»를 **가를 수 없다**. 잴 수 있는 것은 «분할했나 안 했나»와 그 강도뿐이다.
⇒ 결론은 «물타기가 좋다/나쁘다»가 아니라 **«분할 자체는 수익률과 무관하고 위험은 다른 데 있다»** 이다.

## 규율
크기를 안 맞추면 답이 뒤집힌다(2026-09-06). 그래서 USDT 순손익과 **건당 수익률(명목 대비 bp)**
을 나란히 본다. 전자는 «얼마 벌었나», 후자는 «잘했나»다.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRIPS = ROOT / "data/live/account_round_trips.jsonl"
SEED = 20260913
N_BOOT = 20000


def load() -> pd.DataFrame:
    rows = [json.loads(l) for l in TRIPS.read_text().splitlines() if l.strip()]
    d = pd.DataFrame([x for x in rows if x.get("closed") and x.get("side") in ("LONG", "SHORT")])
    d["notional"] = d.max_qty * d.entry_price
    d["unit"] = d.net_pnl / d.notional * 1e4          # 건당 수익률(bp) -- 크기 매칭된 지표
    d["hold_min"] = (d.exit_time - d.entry_time) / 60_000
    return d


def boot_diff(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float]:
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)]
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def row(lab: str, g: pd.DataFrame) -> None:
    print(f"{lab:>16} n={len(g):>3} 건당 {g.unit.mean():>+8.2f}bp 중앙 {g.unit.median():>+7.2f} "
          f"승률 {100*(g.unit>0).mean():>5.1f}% 최악 {g.unit.min():>+8.2f}bp "
          f"순손익 {g.net_pnl.sum():>+9.2f} USDT")


def main() -> int:
    d = load()
    rng = np.random.default_rng(SEED)
    print(f"왕복 {len(d)} · 독립 일수 "
          f"{pd.to_datetime(d.entry_time, unit='ms').dt.date.nunique()}\n")
    row("전체", d)
    row("단일(fills=2)", d[d.fills == 2])
    row("분할(fills>2)", d[d.fills > 2])
    print()
    for lo, hi, lab in ((3, 4, "가벼움 3-4"), (5, 8, "보통 5-8"), (9, 10**9, "무거움 9+")):
        row(lab, d[(d.fills >= lo) & (d.fills <= hi)])

    sing, scal = d[d.fills == 2].unit.values, d[d.fills > 2].unit.values
    lo, hi = boot_diff(sing, scal, rng)
    verdict = "구분 못함" if lo < 0 < hi else "유의"
    print(f"\n단일 − 분할 건당 차 {sing.mean()-scal.mean():+.2f}bp · 95%CI [{lo:+.2f}, {hi:+.2f}] ⇒ **{verdict}**")
    print(f"fills–건당수익 순위상관 {d.fills.corr(d.unit, method='spearman'):+.3f}  "
          f"(≈0 이면 분할 자체는 «잘했나»와 무관)")
    print(f"fills–명목    순위상관 {d.fills.corr(d.notional, method='spearman'):+.3f}  "
          f"(＞0 이면 **분할하면 포지션이 커진다** -- 위험은 여기 있다)")

    worst = d.net_pnl.idxmin()
    heavy = d[d.fills >= 9]
    heavy_ex = heavy.drop(index=worst, errors="ignore")
    print(f"\n무거운 분할(9+) 순손익 {heavy.net_pnl.sum():+.2f} → **최악 1건 빼면 "
          f"{heavy_ex.net_pnl.sum():+.2f}** (건당 {heavy.unit.mean():+.2f} → {heavy_ex.unit.mean():+.2f}bp)")
    w = d.loc[worst]
    print(f"그 1건: fills {int(w.fills)} · 명목 {w.notional:,.0f} USDT · 보유 {w.hold_min:.0f}분"
          f"({w.hold_min/60:.1f}시간) · 건당 {w.unit:+.1f}bp · {w.net_pnl:+.2f} USDT")

    # ── 이 분석의 결론을 고정한다 ─────────────────────────────────────────────
    assert lo < 0 < hi, "분할 vs 단일 차가 유의해졌다 -- 판정을 다시 볼 것"
    assert abs(d.fills.corr(d.unit, method="spearman")) < 0.2, "분할 강도가 수익률과 붙기 시작했다"
    assert d.fills.corr(d.notional, method="spearman") > 0.3, "분할–크기 결합이 사라졌다"
    assert d[d.fills > 2].net_pnl.sum() > d[d.fills == 2].net_pnl.sum(), \
        "분할 거래가 순손익의 다수를 만든다는 사실이 바뀌었다"
    assert heavy_ex.net_pnl.sum() > 0 > heavy.net_pnl.sum(), \
        "무거운 분할의 손실이 한 건 집중이라는 구조가 바뀌었다"
    print("\n⇒ 분할 자체는 건당 수익률과 무관(상관 −0.04, CI 0 포함)하고 순손익의 다수를 만든다.")
    print("  위험은 «분할»이 아니라 분할이 데려오는 **크기(+0.51)와 보유시간**이다 — 그 둘은 이미 상한이 막는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
