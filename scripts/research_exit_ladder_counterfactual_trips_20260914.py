"""**예산 사다리 반사실 — 사용자 실제 왕복 원장** (2026-09-14, 사용자 요청 «2단계»).

랜덤 진입 시뮬에서 사다리는 세 축(성장·MDD·최악1건) 27/27 로 이겼다. 그런데 2026-09-13 에
물타기 게이트를 **시뮬 근거로 올렸다가 실계좌 69왕복으로 재보고 당일 철회**한 전례가 있다
([[feedback_wrong_population_evidence_applied_to_user_20260913]]). 사용자의 실제 진입은
무작위가 아니므로, 같은 규칙이 **이 모집단에서도** 같은 방향인지 따로 봐야 한다.

## 방법 — 방향·시점·크기를 실제로 고정하고 «사다리만» 켠다
확립된 반사실 논리 그대로다(user_entry_size_decomposition_20260911):
판단은 재현 못 하지만 **진입 방향·시점·수량을 실제로 고정**하면 나머지 축은 진짜 거래로 계산된다.
  · 각 왕복의 진입 VWAP·수량·측면을 그대로 쓰고, 진입~청산 사이를 **1분봉으로 재생**한다.
  · 매 분 배포된 `exit_fraction_required(순자산, 안전MAE, 명목, hard_cap)` 를 **그대로 호출**해
    요구 비율을 받는다. 재구현하지 않는다.
  · 1% 이상이면 그 분의 종가로 그만큼 부분청산한다(peg 청산 비용). 남은 수량은 **실제 청산가**로 닫는다.
  · 대조군은 «사다리 없음» = 실제 거래 그대로.

## 🔴순자산이 원장에 없다 — 그래서 스윕한다
사다리는 **명목/순자산** 비율로 발동하는데 `account_round_trips.jsonl` 에 순자산 필드가 없다.
명목은 646~35,291(중앙 4,557)이고 메모리 실측 순자산이 ~1,100 이라 진입 배수가 0.6~32배로
흩어진다. 순자산을 하나로 찍지 않고 **여러 값에서 재서 부호가 그 가정에 흔들리는지** 본다.
⚠️예치·출금이 있었다면 복리 재구성도 안 되므로 고정값 스윕이 정직한 최대치다.

## ⚠️이 결과의 지위
저장 원장 기반이므로 **diagnostic 전용**이다(CLAUDE.md Fresh-Forward 규칙). 승격·기대 라이브
성과의 근거로 쓰지 않는다. n=70 · ETH 단일 · 사용자 재량 진입이라 **부호와 순위**를 읽는다.
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
from scripts.live_eth_risk_sizing_policy_20260913 import exit_fraction_required  # noqa: E402
import research_hold_time_counterfactual_68trips_20260913 as base  # noqa: E402

PEG_EXIT_BP = 2.93            # 부분청산 한 다리 비용(대시보드 실측). 진입 비용은 양팔 공통이라 상쇄된다.
TAKER_EXIT_BP = 5.0           # 손절은 시장가다
STOP_LOSS_PCT = 0.03          # 배포값(live_manual_peg_entry_20260912)
STOP_SLIP_MED_BP = 14.0       # 시장가 손절 슬리피지 중앙(실측). 봉이 더 나쁘면 봉을 쓴다.
MIN_FRACTION = 0.01           # 섀도우 기록기·시뮬과 **같은 하한**
SAFE_MAE_REF = base.SAFE_MAE_REF
SEED = 20260914


def replay(trip: dict, kl: pd.DataFrame, equity: float, cap_x: float,
           hold_min: int, ladder: bool, stop: bool = False) -> float | None:
    """이 왕복의 **단위 명목당 순손익(bp)**. ladder=False 면 실제 거래 그대로.

    수량·방향·진입시점은 실제 값을 쓴다. 사다리는 매 분 배포 함수에 물어본다."""
    s = 1.0 if trip["side"] == "LONG" else -1.0
    entry, qty0 = float(trip["entry_price"]), float(trip["max_qty"])
    if not (entry > 0 and qty0 > 0):
        return None
    seg = kl[(kl.ts >= trip["entry_time"]) & (kl.ts <= trip["exit_time"])]
    if seg.empty:
        return None
    notional0 = qty0 * entry
    qty, realized = qty0, 0.0                      # realized 는 USDT
    mae = SAFE_MAE_REF[hold_min]
    if ladder or stop:
        cs, hs, ls = (seg.c.to_numpy(float), seg.h.to_numpy(float), seg.l.to_numpy(float))
        for c, hi_, lo_ in zip(cs, hs, ls):
            if qty <= 1e-12:
                break
            # 🔴손절은 **봉내 고가/저가**로 닿는다(라이브 컨벤션: resting STOP_MARKET 은
            # 종가가 아니라 닿는 즉시 체결). 사다리는 종가 기준이다 -- 두 컨벤션이 공존한다.
            if stop:
                adverse = (entry - lo_) / entry if s > 0 else (hi_ - entry) / entry
                if adverse >= STOP_LOSS_PCT:
                    # 손절선에서 체결. 봉이 그 너머에서 마감했으면 더 나쁜 쪽(보수적).
                    fm = min(-STOP_LOSS_PCT - STOP_SLIP_MED_BP / 1e4, s * (c / entry - 1.0))
                    realized += qty * entry * (fm - TAKER_EXIT_BP / 1e4)
                    qty = 0.0
                    break
            if not ladder:
                continue
            move = s * (c / entry - 1.0)           # 부호 있는 가격변동(비율)
            eq_now = equity + qty * entry * move   # 교차 마진: 미실현이 순자산에 들어간다
            if eq_now <= 0:
                break                              # 청산 -- 사다리가 손댈 게 없다
            need = exit_fraction_required(eq_now, mae, qty * c,
                                          hard_cap=cap_x)["required_fraction"]
            if need >= MIN_FRACTION:
                closed = qty * min(1.0, need)
                realized += closed * entry * (move - PEG_EXIT_BP / 1e4)
                qty -= closed
    # 남은 수량은 **실제 청산가**로 닫는다(실제 청산 시점·가격을 바꾸지 않는다)
    if qty > 1e-12:
        move = s * (float(trip["exit_price"]) / entry - 1.0)
        realized += qty * entry * (move - PEG_EXIT_BP / 1e4)
    return 1e4 * realized / notional0              # 단위 명목당 bp


def run(equity: float, cap_x: float, hold_min: int, trips, kl,
        fix_lev: float | None = None) -> pd.DataFrame:
    """`fix_lev` 가 오면 **모든 왕복의 진입배수를 그 값으로 통일**한다.

    🔴왜 필요한가(2026-09-14 사용자 지적): 원장의 상한 안 진입은 배수 중앙이 1.91배인데
    오늘 배포된 진입은 **상한까지 채워 권고**한다. 사다리는 배수에 계단처럼 반응하므로
    (2배면 40% 역행해야 켜지고 6배면 즉시) 1.91배 표본의 «0/49» 를 «오늘 진입에서도 0» 으로
    읽으면 안 된다. 방향·시점·기간은 실제로 두고 **크기만** 바꿔 그 축을 분리한다."""
    rows = []
    for t in trips:
        if fix_lev is not None:
            t = {**t, "max_qty": equity * fix_lev / float(t["entry_price"])}
        a = replay(t, kl, equity, cap_x, hold_min, ladder=False)
        b = replay(t, kl, equity, cap_x, hold_min, ladder=True)
        st = replay(t, kl, equity, cap_x, hold_min, ladder=False, stop=True)
        both = replay(t, kl, equity, cap_x, hold_min, ladder=True, stop=True)
        if a is None or b is None or st is None or both is None:
            continue
        lev0 = t["max_qty"] * t["entry_price"] / equity
        rows.append({"day": pd.to_datetime(t["entry_time"], unit="ms").date(),
                     "side": t["side"], "base_bp": a, "ladder_bp": b, "diff_bp": b - a,
                     "stop_bp": st, "stop_diff": st - a,
                     "both_bp": both, "both_diff": both - a,
                     "lev0": lev0,
                     # 🔴진입 시점에 이미 상한을 넘었나. 넘었다면 사다리는 «예산 난간»이 아니라
                     # **즉시 청산 명령**이다 -- 오늘 상한(6배)은 그런 진입을 애초에 막는다.
                     # 그 건들을 섞어 재면 «사다리 효과»가 아니라 «상한 소급 적용»을 재게 된다.
                     "over_at_entry": lev0 > cap_x,
                     "entry_ms": t["entry_time"]})
    return pd.DataFrame(rows)


def breakdown(df: pd.DataFrame, cap_x: float, stack_ms: int) -> None:
    """사용자 지적: 과거 원장은 **현재 로직 아래 만들어진 거래가 아니다**.
    상한 안/밖과 스택 전/후로 갈라 «무엇을 재고 있었나»를 드러낸다."""
    def line(tag, sub):
        if sub.empty:
            print(f"  {tag:<28} (없음)"); return
        print(f"  {tag:<28} n={len(sub):>3}  배수중앙 {sub.lev0.median():>6.2f}  "
              f"사다리차 {sub.diff_bp.mean():>+8.2f}bp  이긴건 {int((sub.diff_bp>0).sum()):>3}/{len(sub)}")
    print()
    print(f"[상한 {cap_x:.0f}배 기준] 진입 시점에 이미 넘었나")
    line("상한 초과 진입", df[df.over_at_entry])
    line("상한 안 진입", df[~df.over_at_entry])
    print("[현재 스택 배포(09-12) 전/후]")
    line("스택 이전", df[df.entry_ms < stack_ms])
    line("스택 이후", df[df.entry_ms >= stack_ms])
    print("[상한 안 + 스택 이후만 = 오늘 규칙과 맞는 표본]")
    line("둘 다 만족", df[(~df.over_at_entry) & (df.entry_ms >= stack_ms)])


def _self_check() -> None:
    """사다리가 «안 켜지는 조건»과 «켜지는 조건»이 설계대로인지."""
    kl = pd.DataFrame({"ts": [0, 60_000, 120_000], "o": 0.0, "h": 0.0, "l": 0.0,
                       "c": [100.0, 100.0, 100.0]})
    trip = {"side": "LONG", "entry_price": 100.0, "max_qty": 1.0, "exit_price": 100.0,
            "entry_time": 0, "exit_time": 120_000}
    # 명목 100, 순자산 1000 -> 0.1배. 상한 6배 안이라 사다리는 한 번도 안 켜진다 -> 두 팔 동일.
    a = replay(trip, kl, 1000.0, 6.0, 240, ladder=False)
    b = replay(trip, kl, 1000.0, 6.0, 240, ladder=True)
    assert abs(a - b) < 1e-9, f"상한 안인데 사다리가 켜졌다: {a} vs {b}"
    assert abs(a - (-PEG_EXIT_BP)) < 1e-6, f"움직임 0 이면 비용만 남아야 한다: {a}"
    # 🔴가격이 **평평하면** 일찍 닫으나 끝에 닫으나 비용률이 같아 두 팔이 같아진다 --
    # 첫 판의 자체점검이 그렇게 짜여 «안 잘렸다»고 잘못 실패했다. 움직이는 경로로 재야 한다.
    def mk(path):
        return (pd.DataFrame({"ts": [60_000 * i for i in range(len(path))], "o": 0.0,
                              "h": 0.0, "l": 0.0, "c": [float(x) for x in path]}),
                {"side": "LONG", "entry_price": 100.0, "max_qty": 1.0,
                 "exit_price": float(path[-1]), "entry_time": 0,
                 "exit_time": 60_000 * (len(path) - 1)})
    # ① 계속 하락: 일찍 줄이면 덜 잃는다 -> 사다리가 **낫다**
    k2, t2 = mk([100, 95, 90])
    down_base = replay(t2, k2, 10.0, 6.0, 240, ladder=False)
    down_lad = replay(t2, k2, 10.0, 6.0, 240, ladder=True)
    assert down_lad > down_base + 1e-9, f"하락에서 사다리가 안 도왔다: {down_lad} vs {down_base}"
    # ② 내려갔다 회복: 바닥에서 줄였으니 회복을 못 받는다 -> 사다리가 **나쁘다**
    k3, t3 = mk([100, 95, 100])
    up_base = replay(t3, k3, 10.0, 6.0, 240, ladder=False)
    up_lad = replay(t3, k3, 10.0, 6.0, 240, ladder=True)
    assert up_lad < up_base - 1e-9, f"회복에서 사다리가 안 손해봤다: {up_lad} vs {up_base}"
    print(f"통과 — 상한 안 무동작 · 하락 {down_base:.0f}->{down_lad:.0f}bp 개선 · "
          f"회복 {up_base:.0f}->{up_lad:.0f}bp 손해 · 비용 방향")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", default="800,1100,1500",
                    help="순자산 가정(콤마 구분). 원장에 없어서 스윕한다")
    ap.add_argument("--cap-x", type=float, default=6.0, help="실효 상한(배포 순자산 상한)")
    ap.add_argument("--hold", type=int, default=240, help="안전MAE 를 고를 지평(분)")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check(); return 0

    trips = base.load_trips()
    print(f"왕복 {len(trips)}건 · {pd.to_datetime(trips[0]['entry_time'], unit='ms').date()}"
          f" ~ {pd.to_datetime(trips[-1]['exit_time'], unit='ms').date()}")
    kl = base.fetch_1m(min(t["entry_time"] for t in trips) - 60_000,
                       max(t["exit_time"] for t in trips) + 60_000)
    print(f"1분봉 {len(kl):,}\n")
    print("⚠️저장 원장 기반 -- diagnostic 전용(승격 근거 아님)\n")
    print(f"{'순자산':>7} {'진입배수 중앙':>12} {'사다리 없음':>11} {'사다리':>9} {'차(bp)':>9} "
          f"{'95%CI':>18} {'이긴 건':>8}")
    for e in [float(x) for x in a.equity.split(",")]:
        df = run(e, a.cap_x, a.hold, trips, kl)
        if df.empty:
            print(f"{e:>7.0f}  (표본 없음)"); continue
        lo, hi = base.day_cluster_ci(df, "diff_bp")
        print(f"{e:>7.0f} {df.lev0.median():>12.2f} {df.base_bp.mean():>11.2f} "
              f"{df.ladder_bp.mean():>9.2f} {df.diff_bp.mean():>+9.2f} "
              f"[{lo:>+7.2f},{hi:>+7.2f}] {int((df.diff_bp > 0).sum()):>4}/{len(df)}")
    # 🔴진입배수를 통일해 «크기 축»만 분리한다. 오늘 권고는 상한(6배)까지 채운다.
    e0m = float(a.equity.split(",")[len(a.equity.split(",")) // 2])
    print()
    print(f"=== 진입배수를 통일하면 (방향·시점·기간 실제 고정 · 순자산 {e0m:.0f}) ===")
    print(f"{'배수':>5} {'아무것도':>9} {'손절만':>9} {'사다리만':>9} {'둘 다':>9}")
    for L in (2.0, 4.0, 5.0, 6.0):
        df = run(e0m, a.cap_x, a.hold, trips, kl, fix_lev=L)
        print(f"{L:>5.1f} {df.base_bp.mean():>9.2f} {df.stop_bp.mean():>9.2f} "
              f"{df.ladder_bp.mean():>9.2f} {df.both_bp.mean():>9.2f}")
    print()
    print("각 팔의 «아무것도 안 함» 대비 차 + 일군집 95%CI (배수 6.0 = 오늘 권고 크기)")
    df = run(e0m, a.cap_x, a.hold, trips, kl, fix_lev=6.0)
    for tag, col in (("손절만", "stop_diff"), ("사다리만", "diff_bp"), ("둘 다", "both_diff")):
        lo, hi = base.day_cluster_ci(df, col)
        sig = "유의" if (lo > 0 or hi < 0) else "0 포함"
        print(f"  {tag:<8} {df[col].mean():>+8.2f}bp  [{lo:>+7.2f},{hi:>+7.2f}] {sig:<6} "
              f"이긴건 {int((df[col] > 0).sum()):>3}/{len(df)} · "
              f"건드린건 {int((df[col].abs() > 1e-9).sum()):>3}/{len(df)}")
    print(f"  ⭐손절은 **3% 가격변동**에 걸리므로 배수와 무관하다(단위명목당 bp 가 같다). "
          f"사다리는 배수에 계단처럼 반응한다.")
    # 🔴평균으로 손절을 판정하면 안 된다 -- 손절의 일은 평균이 아니라 **꼬리**다
    # ([[eth_stop_cuts_tail_not_growth_20260914]]). 최악 건과 하위 분위를 따로 본다.
    print()
    print("꼬리 (배수 6.0 · 단위명목당 bp) -- 손절을 평가해야 할 잣대")
    print(f"{'팔':>8} {'최악1건':>9} {'하위5%':>9} {'하위10%':>9} {'평균':>9}")
    for tag, col in (("아무것도", "base_bp"), ("손절만", "stop_bp"),
                     ("사다리만", "ladder_bp"), ("둘 다", "both_bp")):
        v = df[col].to_numpy()
        print(f"{tag:>8} {v.min():>9.1f} {np.quantile(v, .05):>9.1f} "
              f"{np.quantile(v, .10):>9.1f} {v.mean():>9.1f}")
    # 실제로 얼마나 역행했나 -- 3% 손절이 «닿을 뻔한» 정도를 본다
    print()
    mae = []
    for t in trips:
        sg = kl[(kl.ts >= t["entry_time"]) & (kl.ts <= t["exit_time"])]
        if sg.empty:
            continue
        e = float(t["entry_price"])
        mae.append(100 * ((e - sg.l.min()) / e if t["side"] == "LONG"
                          else (sg.h.max() - e) / e))
    mae = np.array(mae)
    print(f"왕복별 최대 역행폭(%): 중앙 {np.median(mae):.2f} · 90분위 {np.quantile(mae,.9):.2f} "
          f"· 최대 {mae.max():.2f}")
    for th in (1, 2, 3, 5):
        print(f"  {th}% 넘긴 왕복: {int((mae >= th).sum()):>3}/{len(mae)}")

    # 사용자 지적(2026-09-14): 과거 원장은 현재 로직 아래 만들어진 거래가 아니다.
    e0 = float(a.equity.split(",")[len(a.equity.split(",")) // 2])
    stack_ms = int(pd.Timestamp("2026-09-12", tz="UTC").timestamp() * 1000)
    print(f"\n=== 무엇을 재고 있었나 (순자산 {e0:.0f} 기준) ===")
    breakdown(run(e0, a.cap_x, a.hold, trips, kl), a.cap_x, stack_ms)
    print("\n⚠️순자산은 원장에 없어 가정이다. 부호가 세 값에서 갈리면 «모른다»가 답이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
