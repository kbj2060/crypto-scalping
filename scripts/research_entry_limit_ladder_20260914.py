"""**가격 지정가 사다리 진입 — 3~5분할이 단일 진입을 이기는가** (2026-09-14, 사용자 요청).

사용자: *"3분할에서 5분할 정도로 분할 매수하는거야. 이걸 정해주는 모델을 만들어줘."*

## 왜 이건 이미 닫힌 축이 아닌가
이 저장소는 분할을 세 번 기각했는데 **셋 다 다른 것**을 쟀다:
  · 시간 분할(TWAP, research_entry_tranche_count_20260913) — 그 모델의 한계 줄이 «**체결은 즉시
    가정**」이다. 모든 칸이 바로 채워지면 나누는 건 노출만 모자라 엣지를 덜 먹으므로 k=1 이
    이길 수밖에 없다. 퍼뜨리는 축도 **보유시간 대비 비율**(시간)이었다.
  · 물타기(averaging_down) — **이미 꽉 찬 포지션에 추가**라 노출이 커진다. 크기 효과였다.
  · 예산 추가매수 — 순행으로 한도가 늘면 채우는 것. 역시 노출이 는다.
**가격 지정가 사다리는 셋 다 아니다**: 총 명목을 **고정**해 k 칸으로 쪼개고, 깊은 칸은 가격이
와야 체결된다. 안 오면 **덜 산 채로 끝난다** -- 그게 이 축의 고유한 대가이자 보험이다.

## 모델이 정하는 것
    k (칸 수)  ·  δ (칸 간격, ATR 배수)  ·  칸당 크기(균등)
칸 0 은 즉시(peg), 칸 i 는 `entry × (1 − i·δ·atr_pct)`(롱) 에 지정가. 미체결은 **취소**한다.

## 체결 모델
지정가는 **닿으면 그 가격에** 체결된다(보수적으로 더 좋은 가격은 안 받는다). 창 안의 누적
최저(롱)/최고(숏)로 첫 도달 시각을 찾아 **시간가중 노출**까지 낸다 -- 늦게 채워진 칸은 노출
기여가 작다. 🔴노출을 안 맞추고 팔을 비교하면 크기 효과를 전략 효과로 읽는다(이 저장소 4회 전과).

⚠️한계: 부분체결·큐 우선순위 없음 · 취소 수수료 없음 · 진입 시각 무작위(사용자 재량 미반영) ·
정확도는 주입값 · 1분봉이라 분 안 경로는 모름.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

KL = pathlib.Path("/home/kbj20/crypto-scalping/binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv")
ATR_MIN = 1440                 # 사이징 워커와 같은 창(24시간). 1분봉이므로 1440개.
MAKER_BP, PEG_BP, EXIT_BP = 2.0, 2.95, 2.93
SEED = 20260914
WINDOWS = {"VAL(2025-09~12)": ("2025-09-01", "2025-12-31"),
           "OOS(2026-01~03)": ("2026-01-01", "2026-03-31"),
           "TEST(2026-04~07)": ("2026-04-01", "2026-07-31")}


def load() -> pd.DataFrame:
    d = pd.read_csv(KL, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    d = d.dropna().sort_values("timestamp").reset_index(drop=True)
    c = d.close.to_numpy(float)
    atr = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(ATR_MIN, min_periods=600).mean()
    d["atr_pct"] = (atr / c).to_numpy()
    return d


def simulate(d: pd.DataFrame, idx: np.ndarray, sides: np.ndarray, *, k: int, delta: float,
             hold: int) -> dict:
    """칸 k · 간격 delta(ATR 배수) · 보유 hold 분. 의도 명목 1 단위당 결과.

    반환: net_bp(의도명목당) · expo(시간가중 체결 비율) · fills(평균 체결 칸수)."""
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    atr = d.atr_pct.to_numpy(float)
    net, expo, fills = [], [], []
    w = hold
    for i, s in zip(idx, sides):
        a = atr[i]
        if not (a > 0) or i + w >= len(c):
            continue
        e0 = c[i]
        # 칸 가격: 롱은 아래로, 숏은 위로. 칸 0 은 즉시 체결(= e0).
        px = np.array([e0 * (1 - s * j * delta * a) for j in range(k)])
        seg_lo, seg_hi = lo[i + 1:i + 1 + w], hi[i + 1:i + 1 + w]
        # 누적 극값으로 **첫 도달 시각**을 찾는다(단조라 searchsorted 가능)
        if s > 0:
            run = np.minimum.accumulate(seg_lo)
            t = np.searchsorted(-run, -px, side="left")     # run 은 비증가 -> 부호 반전
        else:
            run = np.maximum.accumulate(seg_hi)
            t = np.searchsorted(run, px, side="left")
        t[0] = 0                                            # 칸 0 은 즉시
        filled = t < w
        if not filled[0]:
            continue                                        # 칸 0 은 정의상 항상 체결
        size = 1.0 / k
        exit_px = c[i + w]
        pnl = 0.0
        for j in range(k):
            if not filled[j]:
                continue
            fee = PEG_BP if j == 0 else MAKER_BP             # 칸 0 만 peg, 나머지는 메이커
            pnl += size * (s * (exit_px / px[j] - 1.0) - (fee + EXIT_BP) / 1e4)
        # 시간가중 노출: 늦게 채워진 칸은 창의 남은 부분만 기여한다
        expo.append(float(sum(size * (w - t[j]) / w for j in range(k) if filled[j])))
        net.append(1e4 * pnl)
        fills.append(int(filled.sum()))
    if not net:
        return {"n": 0}
    net = np.array(net); expo = np.array(expo)
    return {"n": len(net), "net_bp": float(net.mean()), "expo": float(expo.mean()),
            "per_expo": float(net.mean() / expo.mean()) if expo.mean() > 0 else 0.0,
            "fills": float(np.mean(fills)), "worst": float(net.min()),
            "p05": float(np.quantile(net, 0.05))}


def _self_check() -> None:
    """체결 판정과 시간가중 노출 -- 이 둘이 결론을 만든다."""
    n = 600
    d = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
                      "close": 100.0, "high": 100.0, "low": 100.0})
    d["atr_pct"] = 0.01                                    # ATR 1%
    # 가격이 평평하면 깊은 칸은 **절대** 안 채워진다 -> 체결 1칸, 노출 1/k
    r = simulate(d, np.array([10]), np.array([1.0]), k=4, delta=1.0, hold=60)
    assert r["fills"] == 1, f"평평한데 깊은 칸이 채워졌다: {r['fills']}"
    assert abs(r["expo"] - 0.25) < 1e-9, f"노출이 1/k 가 아니다: {r['expo']}"
    # 단일 진입은 항상 전량 -> 노출 1.0
    r1 = simulate(d, np.array([10]), np.array([1.0]), k=1, delta=1.0, hold=60)
    assert abs(r1["expo"] - 1.0) < 1e-9, r1["expo"]
    # 1% 하락이 **창 첫 봉**에 오면 칸1 도 즉시라 노출 1.0 이 맞다 -- 첫 판 테스트가 그렇게
    # 짜여 «시간가중이 안 먹었다»고 잘못 실패했다. 지연을 보려면 하락을 뒤로 옮겨야 한다.
    d2 = d.copy(); d2.loc[11:, ["close", "high", "low"]] = 99.0
    r_now = simulate(d2, np.array([10]), np.array([1.0]), k=2, delta=1.0, hold=60)
    assert r_now["fills"] == 2 and abs(r_now["expo"] - 1.0) < 1e-9, r_now
    # 30 봉 뒤에 내려가면 칸1 은 그때부터만 노출에 기여한다
    d3 = d.copy(); d3.loc[41:, ["close", "high", "low"]] = 99.0
    r2 = simulate(d3, np.array([10]), np.array([1.0]), k=2, delta=1.0, hold=60)
    assert r2["fills"] == 2, f"1% 내려갔는데 칸1 이 안 채워졌다: {r2['fills']}"
    exp = 0.5 + 0.5 * (60 - 30) / 60                       # 칸0 전체 + 칸1 절반
    assert abs(r2["expo"] - exp) < 1e-6, f"시간가중 {r2['expo']} != {exp}"
    print("통과 — 평평하면 미체결 · 단일은 노출 1.0 · 도달하면 체결 · 시간가중 반영")


def run_plan(d, a) -> int:
    """🔴씨드를 짝지어 잰다 -- 방향 동전던지기 잡음이 팔 차이보다 크다
    ([[feedback_paired_seeds_required_when_side_is_injected_20260914]])."""
    ts = d.timestamp.to_numpy(); c = d.close.to_numpy(float)
    use_stop = not a.no_stop
    arms = [("단일 90만(전량)", 1.0, None), ("절반 45만만", 0.5, None)]
    arms += [(f"45/45 · −{p:.1f}%", 0.5, p / 100) for p in (1.0, 1.5, 2.0, 3.0)]
    acc = float(a.acc.split(",")[0])
    print(f"1분봉 {len(d):,} · 보유 {a.hold}분 · 정확도 {acc} · 씨드 {a.seeds} · "
          f"손절 {'평단 3%(재무장)' if use_stop else '없음'}")
    print("⚠️의도 명목(90 만) 1 단위당 bp -- 레버리지에 무관하다. 계좌 %는 배수만큼 곱한다\n")
    for wname, (w0, w1) in WINDOWS.items():
        lo_i = max(int(np.searchsorted(ts, np.datetime64(w0))), ATR_MIN)
        hi_i = int(np.searchsorted(ts, np.datetime64(w1 + "T23:59:59"))) - a.hold - 1
        if hi_i - lo_i < 5000:
            continue
        idx = np.arange(lo_i, hi_i, a.every)
        truth = np.where(c[idx + a.hold] >= c[idx], 1.0, -1.0)
        res = {nm: [] for nm, _, _ in arms}
        for sdd in range(a.seeds):
            rng = np.random.default_rng(SEED + 1000 * sdd)
            sides = np.where(rng.random(len(idx)) < acc, truth, -truth)
            base = None
            for nm, first, pct in arms:
                r = simulate_plan(d, idx, sides, pct=pct, hold=a.hold, first=first,
                                  use_stop=use_stop)
                base = r["net_bp"] if base is None else base
                res[nm].append((r["net_bp"] - base, r["net_bp"], r["expo"], r["fill"],
                                r["stop"], r["p05"], r["worst"]))
        print(f"{wname}  (진입 {len(idx):,}건)")
        print(f"{'팔':>16} {'Δ단일bp':>12} {'노출':>6} {'추가체결':>8} {'손절률':>7} "
              f"{'하위5%':>9} {'최악':>9}")
        for nm, _, _ in arms:
            v = np.array(res[nm]); m = v[:, 0].mean()
            se = v[:, 0].std(ddof=1) / np.sqrt(len(v)) if v[:, 0].std() > 0 else 0.0
            print(f"{nm:>16} {m:>7.2f}±{se:<4.2f} {v[:, 2].mean():>6.2f} "
                  f"{100*v[:, 3].mean():>7.1f}% {100*v[:, 4].mean():>6.1f}% "
                  f"{v[:, 5].mean():>9.1f} {v[:, 6].mean():>9.1f}")
        print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="0.50,0.55,0.60")
    ap.add_argument("--hold", type=int, default=240)
    ap.add_argument("--every", type=int, default=120, help="몇 분마다 한 번 진입을 표집하나")
    ap.add_argument("--plan", action="store_true", help="사용자 계획(45/45 · −pct) 비교")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--no-stop", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check(); _check_plan(); return 0

    d = load()
    if a.plan:
        return run_plan(d, a)
    ts = d.timestamp.to_numpy()
    print(f"1분봉 {len(d):,} ({d.timestamp.min().date()}~{d.timestamp.max().date()}) · "
          f"보유 {a.hold}분 · {a.every}분마다 표집\n")
    rng = np.random.default_rng(SEED)
    c = d.close.to_numpy(float)
    for acc in [float(x) for x in a.acc.split(",")]:
        print(f"=== 정확도 {acc} ===")
        print(f"{'창':>16} {'칸':>3} {'간격':>5} {'의도명목당bp':>12} {'노출':>6} "
              f"{'노출당':>8} {'체결칸':>7} {'최악':>9} {'하위5%':>9}")
        for wname, (w0, w1) in WINDOWS.items():
            lo_i = max(int(np.searchsorted(ts, np.datetime64(w0))), ATR_MIN)
            hi_i = int(np.searchsorted(ts, np.datetime64(w1 + "T23:59:59"))) - a.hold - 1
            if hi_i - lo_i < 5000:
                continue
            idx = np.arange(lo_i, hi_i, a.every)
            truth = np.where(c[np.minimum(idx + a.hold, len(c) - 1)] >= c[idx], 1.0, -1.0)
            sides = np.where(rng.random(len(idx)) < acc, truth, -truth)
            # 🔴간격이 좁으면(0.25~0.5 ATR) 체결칸이 4.8/5 라 «안 오면 안 산다» 가 작동하지
            # 않는다 -- 사다리가 그냥 평단만 조금 낫게 하는 장치가 된다. 선별이 생기는
            # 지점까지 넓혀 본다(2026-09-14 1차 결과에서 확인).
            for k, delta in ((1, 0.0), (3, 0.5), (3, 1.0), (3, 2.0), (3, 3.0),
                             (5, 0.5), (5, 1.0), (5, 2.0), (5, 3.0)):
                r = simulate(d, idx, sides, k=k, delta=delta, hold=a.hold)
                if not r.get("n"):
                    continue
                print(f"{wname:>16} {k:>3} {delta:>5.2f} {r['net_bp']:>12.2f} {r['expo']:>6.2f} "
                      f"{r['per_expo']:>8.2f} {r['fills']:>7.2f} {r['worst']:>9.1f} "
                      f"{r['p05']:>9.1f}")
        print()
    print("⚠️노출이 팔마다 다르다 -- **노출당**이 크기 매칭된 값이고, «의도명목당» 은 "
          "같은 주문을 냈을 때의 절대 성과다. 둘을 같이 읽는다.")
    return 0


# ── 사용자 계획: 「증거금 상한 90 만. 45 만 진입, −1.5% 에 45 만 추가」 (2026-09-14) ───────
# 🔴이 계획은 크기만 나누는 게 아니라 **손절 위치를 옮긴다**. 배포 손절은 평단 3% 이고
# 물타기로 평단이 바뀌면 다시 건다(`live_manual_peg_entry_20260912.build_stop_plan`,
# closePosition=true). 45/45 가 −1.5% 에서 다 차면 평단 −0.75% -> 손절은 원진입가 대비
# **−3.72%** 로 멀어진다. 즉 「덜 잘리지만, 잘리면 같은 크기」다 -- 그걸 같이 재야 한다.
# ⚠️결과는 **의도 명목 1 단위당**이라 레버리지에 무관하다. 계좌 %만 배수만큼 곱해진다.
STOP_PCT, STOP_SLIP_BP, TAKER_BP = 0.03, 14.0, 4.5


def simulate_plan(d, idx, sides, *, pct, hold, first=0.5, use_stop=True):
    """칸0 즉시 `first`, 칸1 은 `-pct` 지정가에 나머지. 미체결이면 **덜 산 채로** 끝난다.

    봉 안 순서: 롱이 내려갈 때 −1.5% 지정가가 −3.72% 손절보다 **먼저** 닿는다(물리적으로
    가까운 쪽이 먼저다) -- 낙관 가정이 아니라 순서가 정해져 있다. 그래서 체결 -> 손절 순.
    `first=1.0` 이면 단일 진입, `pct=None` 이면 칸1 없음(= `first` 크기 단일)."""
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    net, expo, fills, stops = [], [], [], []
    for i, s in zip(idx, sides):
        e0 = c[i]
        if i + hold >= len(c):
            continue
        size, avg, cost = first, e0, first * (PEG_BP + EXIT_BP) / 1e4
        lvl = None if pct is None or first >= 1.0 else e0 * (1 - s * pct)
        stop = avg * (1 - s * STOP_PCT) if use_stop else None
        t_fill, out, texit = None, None, hold
        for t in range(hold):
            b = i + 1 + t
            if lvl is not None and (lo[b] <= lvl if s > 0 else hi[b] >= lvl):
                add = 1.0 - first
                avg = (avg * size + lvl * add) / (size + add)
                size += add; cost += add * (MAKER_BP + EXIT_BP) / 1e4
                t_fill, lvl = t, None
                stop = avg * (1 - s * STOP_PCT) if use_stop else None
            if stop is not None and (lo[b] <= stop if s > 0 else hi[b] >= stop):
                out, texit = stop, t
                cost += size * (TAKER_BP + STOP_SLIP_BP - EXIT_BP) / 1e4   # 지정가청산 -> 시장가
                break
        px = c[i + hold] if out is None else out * (1 - s * STOP_SLIP_BP / 1e4)
        net.append(1e4 * (size * s * (px / avg - 1.0) - cost))
        expo.append(first * texit / hold
                    + (0.0 if t_fill is None else (1 - first) * (texit - t_fill) / hold))
        fills.append(int(t_fill is not None)); stops.append(int(out is not None))
    n = np.array(net); e = np.array(expo)
    return {"net_bp": float(n.mean()), "expo": float(e.mean()), "fill": float(np.mean(fills)),
            "stop": float(np.mean(stops)), "worst": float(n.min()),
            "p05": float(np.quantile(n, 0.05)), "n": len(n)}


def _check_plan() -> None:
    n = 900
    flat = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
                         "close": 100.0, "high": 100.0, "low": 100.0, "atr_pct": 0.01})
    ix, sd = np.array([10]), np.array([1.0])
    # 평평하면 칸1 미체결 · 손절 미발동 · 노출은 first
    r = simulate_plan(flat, ix, sd, pct=0.015, hold=240)
    assert r["fill"] == 0 and r["stop"] == 0 and abs(r["expo"] - 0.5) < 1e-9, r
    assert abs(r["net_bp"] + 0.5 * (PEG_BP + EXIT_BP)) < 1e-6, r   # 절반만 샀으니 비용도 절반
    # 정확히 −1.5% 로 내려가 머물면 칸1 체결, 손절(평단 −3%)은 미발동
    v = np.r_[np.full(12, 100.0), np.full(n - 12, 98.5)]
    d2 = flat.copy(); d2[["close", "high", "low"]] = np.c_[v, v, v]
    r2 = simulate_plan(d2, ix, sd, pct=0.015, hold=240)
    assert r2["fill"] == 1 and r2["stop"] == 0, r2
    # 🔴핵심 -- **−3.5% 까지만 내렸다 회복**하는 길. 단일은 −3% 에서 잘리지만,
    # 사다리는 평단이 99.25 로 내려가 손절이 96.2725(원진입가 −3.73%)라 **안 잘린다**.
    v3 = np.r_[np.full(12, 100.0), np.linspace(100, 96.5, 120), np.linspace(96.5, 101, n - 132)]
    d3 = flat.copy(); d3[["close", "high", "low"]] = np.c_[v3, v3, v3]
    r3 = simulate_plan(d3, ix, sd, pct=0.015, hold=240)
    r4 = simulate_plan(d3, ix, sd, pct=None, hold=240, first=1.0)
    assert r3["fill"] == 1 and r3["stop"] == 0, r3
    assert r4["stop"] == 1, r4
    assert r3["net_bp"] > r4["net_bp"], (r3, r4)
    print("통과 — 미체결/체결/손절 · 평단 재무장이 손절을 −3.0%->−3.73% 로 민다")

if __name__ == "__main__":
    raise SystemExit(main())
