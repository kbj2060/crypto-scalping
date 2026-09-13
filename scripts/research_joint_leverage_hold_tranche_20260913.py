"""**명목배수 × 보유시간 × 분할횟수를 한꺼번에 고른다** (2026-09-13, 사용자 요청).

사용자: *"레버리지 30배 · 보유시간 2시간 · 분할 균등 매수 3회 — 이 매매 방법 자체를 모델링해서
내가 진입할 때마다 픽스해 줬으면. 안전하고 효율적으로 뽑아낼 수 없나?"*

## 세 값은 독립이 아니다
  · **생존**은 (L, H) 만의 함수다: 보유 H 동안의 역행폭이 1/L 에 닿으면 청산.  방향 실력과 무관.
  · **성장**은 (L, H, 정확도 a) 의 함수다.
  · **분할 k** 는 램프 구간의 노출을 낮춘다 -- 파산을 줄이는 대신 엣지를 덜 먹는다.

## 🔴이 스크립트가 실제로 묻는 것 — 분할의 **대조군**
«k 분할» 과 «그냥 작게 들어가기» 를 나란히 놓는다. 노출가중 포착률 c = 1 − S/(2H) 이므로
k 분할의 평균 노출은 c·L 이다. 그러면 **같은 평균 노출을 만드는 단일 진입 c·L** 이 대조군이다.
크기 매칭 없는 분할 비교는 2026-09-06 에 이미 한 번 틀린 답을 줬다(«물타기 이득»은 전부 크기 효과였다).

분할이 «크기를 줄이는 것»보다 나으려면 **진입 가격이 좋아져야** 한다. 그건 진입 타이밍에
되돌림이 있어야 성립하는데 이 저장소는 그 신호를 못 찾았다. 그래서 이 실험의 귀무가설은
**«분할 = 크기 줄이기»** 이고, 진짜 물어보는 건 «분할이 그 이상을 주나»다.

## 방법
869일 1분봉. 각 표본: 시작점 무작위 · 방향은 **정확도 a 로 주입**(확률 a 로 그 구간의 실제
방향, 아니면 반대) · k 칸을 S 분에 걸쳐 균등 진입 · H 분에 청산 · 비용 5.88bp 왕복.
파산 = 평단 대비 역행 × 그 시점 노출 >= 1. 성장은 파산 포함 로그성장.

⚠️한계: 정확도가 **움직임 크기와 독립**이라는 가정(09-12 15분 되돌림에서 이 가정이 8배 과대를
만든 전례) · 체결 즉시 · 방향 주입은 사후 정보라 «그만큼 맞히는 사람»의 상한이다.
읽는 것은 절대값이 아니라 **(L,H,k) 칸 사이의 순위**다.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
# 워크트리는 data/ 를 공유하지 않는다. 본 체크아웃을 폴백으로 둔다.
TAPE_CANDIDATES = (ROOT / "data/research/eth_tape_1m_20260906.parquet",
                   pathlib.Path("/home/kbj20/crypto-scalping/data/research/eth_tape_1m_20260906.parquet"))
COST_BP = 5.88
SEED = 20260913
N_SIM = 20000
SPREAD_FRAC = 0.5                    # 분할한다면 보유시간의 절반에 걸쳐


def load(path: str | None = None):
    tape = pathlib.Path(path) if path else next(
        (p for p in TAPE_CANDIDATES if p.exists()), TAPE_CANDIDATES[0])
    d = pd.read_parquet(tape, columns=["px_last", "px_max", "px_min"])
    return (d.px_last.to_numpy(float), d.px_max.to_numpy(float), d.px_min.to_numpy(float))


def simulate(px, hi, lo, *, L: float, hold: int, k: int, spread: int, acc: float,
             n: int = N_SIM, seed: int = SEED) -> dict:
    """파산율과 건당 로그성장. 방향은 정확도 acc 로 주입한다."""
    rng = np.random.default_rng(seed)
    m = len(px)
    start = rng.integers(0, m - hold - spread - 2, n)
    right = rng.random(n) < acc                      # 방향을 맞혔는가
    gap = 0 if k == 1 else spread // max(1, k - 1)
    ruin = np.zeros(n, dtype=bool)
    ret = np.zeros(n)
    for t in range(n):
        a = start[t]
        end = a + spread + hold
        truth = 1.0 if px[end] >= px[a] else -1.0    # 실제 방향(사후)
        s = truth if right[t] else -truth            # 내가 고른 방향
        qty = 0.0
        notion = 0.0
        worst = 0.0
        for i in range(k):
            j = a + i * gap
            qty += L / k
            notion += px[j] * (L / k)
            vw = notion / qty                        # 평단
            seg_end = end if i == k - 1 else min(end, a + (i + 1) * gap)
            seg_lo, seg_hi = lo[j:seg_end + 1], hi[j:seg_end + 1]
            if len(seg_lo) == 0:
                continue
            adv = (vw - seg_lo.min()) / vw if s > 0 else (seg_hi.max() - vw) / vw
            worst = max(worst, adv * qty)            # 순자산 대비 손실
        if worst >= 1.0:
            ruin[t] = True
            ret[t] = -1.0
            continue
        vw = notion / qty
        move = s * (px[end] / vw - 1.0)
        ret[t] = qty * (move - COST_BP / 1e4)        # 순자산 대비 손익
    growth = np.where(ret <= -1.0, -np.inf, np.log1p(np.maximum(ret, -0.999999)))
    finite = growth[np.isfinite(growth)]
    return {"ruin": float(ruin.mean()),
            # 파산을 -inf 로 두면 평균이 -inf 라 «파산 포함 기대 로그성장»을 못 읽는다.
            # 파산 경로는 계좌가 0 에서 멈추므로 그 지분만큼 성장이 0 이 되는 것으로 친다.
            "log_growth": float(finite.mean() * (1 - ruin.mean()) - 10.0 * ruin.mean()),
            "mean_ret_bp": float(ret.mean() * 1e4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", type=float, nargs="+", default=[0.50, 0.55, 0.60])
    ap.add_argument("--n", type=int, default=N_SIM)
    ap.add_argument("--tape")
    a = ap.parse_args()
    px, hi, lo = load(a.tape)
    print(f"1분봉 {len(px):,} · 표본 {a.n:,}/칸 · 비용 {COST_BP}bp 왕복\n")

    # 라이브 위험표의 생존 최대배수(2026-09-13). L 은 그 안에서만 고른다.
    SAFE = {60: 1.821, 120: 2.365, 240: 3.36, 480: 6.967, 1440: 23.236}
    print("=" * 78)
    print("① 분할 k 의 대조군 — «k 분할» vs «같은 평균 노출의 단일 진입»")
    print("=" * 78)
    print("포착률 c = 1 − S/(2H). k 분할의 평균 노출 = c·L 이므로 대조군은 단일 c·L 이다.\n")
    print(f"{'a':>5} {'H':>6} {'L':>5} {'칸':>3} {'파산%':>7} {'로그성장':>10}  판정")
    verdict = {}
    for acc in a.acc:
        for H in (120, 240):
            L = min(8.0, 100.0 / SAFE[H])
            S = int(H * SPREAD_FRAC)
            c = 1 - S / (2 * H)
            base = simulate(px, hi, lo, L=L, hold=H, k=1, spread=0, acc=acc, n=a.n)
            spl = simulate(px, hi, lo, L=L, hold=H, k=3, spread=S, acc=acc, n=a.n)
            ctrl = simulate(px, hi, lo, L=c * L, hold=H, k=1, spread=0, acc=acc, n=a.n)
            for lab, r, LL in (("일괄", base, L), ("3분할", spl, L), (f"단일{c:.2f}L", ctrl, c * L)):
                print(f"{acc:>5.2f} {H:>5}분 {LL:>5.2f} {lab:>3} {100*r['ruin']:>6.3f}% "
                      f"{r['log_growth']:>+10.5f}")
            win = "분할" if spl["log_growth"] > ctrl["log_growth"] else "크기 줄이기"
            verdict[(acc, H)] = win
            print(f"{'':>5} {'':>6} {'':>5} {'':>3} {'':>7} {'':>10}  ⇒ 이긴 쪽: **{win}**\n")

    print("=" * 78)
    print("② 생존선 위로 올라가면 — 분할이 파산을 실제로 줄이나 (L = 30배)")
    print("=" * 78)
    print(f"{'H':>6} {'칸':>4} {'파산%':>8} {'로그성장':>10}")
    for H in (120, 240, 480):
        for k in (1, 3, 6):
            S = 0 if k == 1 else int(H * SPREAD_FRAC)
            r = simulate(px, hi, lo, L=30.0, hold=H, k=k, spread=S, acc=0.55, n=a.n)
            print(f"{H:>5}분 {k:>4} {100*r['ruin']:>7.3f}% {r['log_growth']:>+10.5f}")
        print()

    # ── 결론 고정 ────────────────────────────────────────────────────────────
    assert all(v == "크기 줄이기" for v in verdict.values()), \
        f"분할이 크기 줄이기를 이겼다 -- 처방을 다시 볼 것: {verdict}"
    print("확인: 생존선 안에서는 **분할이 크기 줄이기를 이기지 못한다**(전 칸).")
    print("      ⇒ 노출을 낮추고 싶으면 칸을 늘리는 게 아니라 L 을 낮춘다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
