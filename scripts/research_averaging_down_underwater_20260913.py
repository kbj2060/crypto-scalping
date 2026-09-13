"""**물렸을 때 물타기** — 신호가 있나, 그리고 크기를 맞춰도 이기나 (2026-09-13, 사용자 요청).

사용자: *"뮬렸을 때 물타기하는 전략에 대해서도 연구해봐."*

## 왜 다시 재나
2026-09-13 오전에 «역행 중 추가 금지»를 게이트로 올렸다가 철회했다. 근거로 쓴 09-06 감사는
칩 신호 위 시뮬레이션이라 모집단이 달랐고, 실계좌 원장은 체결 단위를 잃어 **물타기와
피라미딩을 가를 수 없었다**. 그게 미해결로 남아 있었다. 여기서는 테이프로 직접 가른다.

## 두 층으로 나눈다 — 순서가 중요하다
  · **L1 신호**: «진입 후 d% 물린 시점»에서 앞으로의 수익률이 무조건부와 다른가?
    전략과 무관한 순수 통계다. 여기가 0 이면 물타기는 «무작위 시점의 크기 증가»일 뿐이고,
    파산만 늘리므로 L2 를 볼 필요도 없다.
  · **L2 전략**: 크기(평균 노출)를 맞춘 뒤에도 물타기가 이기는가?
    2026-09-06 에 크기 매칭 없이 비교해 «물타기 이득»이라는 틀린 답을 한 번 받았다.

## 규율
  · **양측면 필수**: 롱만 재면 상승장 드리프트를 실력으로 읽는다. 롱·숏을 짝지어 평균한다.
  · **거울상 점검**: «d% 물림»의 반대인 «d% 순행»도 같이 잰다. 두 값이 거울상이면 그건
    신호가 아니라 잔존 베타다(feedback_side_mirror_excess_is_residual_beta_20260908).
  · **파산 포함**: 물타기는 지고 있을 때 노출을 키운다 -- 파산이 정확히 그때 문다.
  · 비용은 왕복 5.88bp, 추가 칸도 명목 비례로 낸다.

⚠️한계: 진입 시점 무작위(사용자 재량 미반영) · 체결 즉시 · 1분봉 고저(그 안 경로는 모름).
읽는 것은 절대값이 아니라 **칸 사이의 차이**다.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAPE_CANDIDATES = (ROOT / "data/research/eth_tape_1m_20260906.parquet",
                   pathlib.Path("/home/kbj20/crypto-scalping/data/research/eth_tape_1m_20260906.parquet"))
COST_BP = 5.88
SEED = 20260913
DRAWDOWNS = (0.005, 0.01, 0.02)          # 물린 정도
FORWARD = (60, 240, 1440)                # 추가 진입 후 지켜볼 창(분)
WATCH = 1440                             # 물림을 기다리는 최대 시간(사용자 상한 = 1일)


def load(path: str | None = None):
    tape = pathlib.Path(path) if path else next(
        (p for p in TAPE_CANDIDATES if p.exists()), TAPE_CANDIDATES[0])
    d = pd.read_parquet(tape, columns=["px_last", "px_max", "px_min"])
    return (d.px_last.to_numpy(float), d.px_max.to_numpy(float), d.px_min.to_numpy(float))


def first_touch(seq: np.ndarray, level: float, below: bool) -> int:
    """`level` 을 처음 건드리는 상대 인덱스. 없으면 -1."""
    hit = seq <= level if below else seq >= level
    return int(np.argmax(hit)) if hit.any() else -1


def layer1(px, hi, lo, rng, n: int, d: float) -> dict:
    """물린 시점에서의 **앞으로의 수익률**을 무조건부와 견준다. 양측면·거울상 포함."""
    m = len(px)
    starts = rng.integers(0, m - WATCH - max(FORWARD) - 2, n)
    out = {f: {"down": [], "up": [], "base": []} for f in FORWARD}
    for t in range(n):
        a = int(starts[t])
        side = 1.0 if t % 2 == 0 else -1.0        # 롱·숏을 번갈아 -- 드리프트를 상쇄한다
        e = px[a]
        seq_adv = lo[a:a + WATCH] if side > 0 else hi[a:a + WATCH]
        seq_fav = hi[a:a + WATCH] if side > 0 else lo[a:a + WATCH]
        j_down = first_touch(seq_adv, e * (1 - side * d), below=side > 0)
        j_up = first_touch(seq_fav, e * (1 + side * d), below=side < 0)
        for f in FORWARD:
            # 무조건부 기준선: 같은 측면·같은 창, 시점만 무작위
            b = a + WATCH // 2
            out[f]["base"].append(side * (px[b + f] / px[b] - 1) * 1e4)
            if j_down >= 0 and a + j_down + f < m:
                k = a + j_down
                out[f]["down"].append(side * (px[k + f] / px[k] - 1) * 1e4)
            if j_up >= 0 and a + j_up + f < m:
                k = a + j_up
                out[f]["up"].append(side * (px[k + f] / px[k] - 1) * 1e4)
    return {f: {k: np.array(v) for k, v in cell.items()} for f, cell in out.items()}


def boot_ci(v: np.ndarray, rng, n_boot: int = 4000) -> tuple[float, float]:
    if len(v) < 30:
        return float("nan"), float("nan")
    means = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def layer2(px, hi, lo, rng, n: int, *, L: float, hold: int, d: float, acc: float) -> dict:
    """전략 비교. 셋 다 **평균 노출**을 같이 보고해 크기 매칭으로 읽는다.

    A 일괄      : t0 에 L 전량
    B 물타기    : t0 에 L/2, d% 물리면 L/2 추가 (안 물리면 L/2 로 끝)
    C 크기매칭  : t0 에 B 의 **평균 노출**만큼 전량 (물타기의 대조군)
    """
    m = len(px)
    starts = rng.integers(0, m - hold - 2, n)
    right = rng.random(n) < acc
    res = {k: {"ret": [], "expo": [], "ruin": 0} for k in ("A", "B", "C")}
    # 먼저 B 의 평균 노출을 구해야 C 를 만들 수 있다 -- 두 번 돈다.
    for pass_no in (1, 2):
        if pass_no == 2:
            c_size = float(np.mean(res["B"]["expo"]))
        for t in range(n):
            a = int(starts[t]); end = a + hold
            truth = 1.0 if px[end] >= px[a] else -1.0
            s = truth if right[t] else -truth
            e = px[a]
            seq_adv = lo[a:end] if s > 0 else hi[a:end]
            j = first_touch(seq_adv, e * (1 - s * d), below=s > 0)
            if pass_no == 1:
                # A: 전량
                res["A"]["expo"].append(L)
                mae = ((e - lo[a:end + 1].min()) / e if s > 0 else (hi[a:end + 1].max() - e) / e)
                if mae * L >= 1.0:
                    res["A"]["ruin"] += 1; res["A"]["ret"].append(-1.0)
                else:
                    res["A"]["ret"].append(L * (s * (px[end] / e - 1) - COST_BP / 1e4))
                # B: 절반 + 물리면 절반 추가
                if j < 0:
                    qty, vw = L / 2, e
                    expo = L / 2
                else:
                    p2 = e * (1 - s * d)
                    qty = L
                    vw = (e * (L / 2) + p2 * (L / 2)) / L
                    # 노출가중 평균: 앞 구간 절반, 뒤 구간 전량
                    expo = (L / 2) * (j / hold) + L * (1 - j / hold)
                mae_b = ((vw - lo[a:end + 1].min()) / vw if s > 0 else (hi[a:end + 1].max() - vw) / vw)
                res["B"]["expo"].append(expo)
                if mae_b * qty >= 1.0:
                    res["B"]["ruin"] += 1; res["B"]["ret"].append(-1.0)
                else:
                    res["B"]["ret"].append(qty * (s * (px[end] / vw - 1) - COST_BP / 1e4))
            else:
                res["C"]["expo"].append(c_size)
                mae = ((e - lo[a:end + 1].min()) / e if s > 0 else (hi[a:end + 1].max() - e) / e)
                if mae * c_size >= 1.0:
                    res["C"]["ruin"] += 1; res["C"]["ret"].append(-1.0)
                else:
                    res["C"]["ret"].append(c_size * (s * (px[end] / e - 1) - COST_BP / 1e4))
    out = {}
    for k, v in res.items():
        r = np.array(v["ret"]); ex = float(np.mean(v["expo"]))
        out[k] = {"ret_bp": float(r.mean() * 1e4), "per_expo_bp": float(r.mean() * 1e4 / max(ex, 1e-9)),
                  "expo": ex, "ruin": v["ruin"] / len(r),
                  "growth": float(np.mean(np.log1p(np.maximum(r, -0.999999))))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--acc", type=float, default=0.60)
    a = ap.parse_args()
    px, hi, lo = load(a.tape)
    rng = np.random.default_rng(SEED)
    print(f"1분봉 {len(px):,} · 표본 {a.n:,}/칸 · 비용 {COST_BP}bp · 정확도 {a.acc}\n")

    print("=" * 76)
    print("L1 신호 — «물린 시점»에서 앞으로의 수익률이 무조건부와 다른가 (양측면 평균)")
    print("=" * 76)
    print(f"{'물림':>6} {'창':>7} {'물린뒤':>9} {'무조건부':>9} {'초과':>8} {'95%CI(초과)':>18} {'거울상(순행뒤)':>12}")
    signal = {}
    for d in DRAWDOWNS:
        r = layer1(px, hi, lo, rng, a.n, d)
        for f in FORWARD:
            dn, bs, up = r[f]["down"], r[f]["base"], r[f]["up"]
            if len(dn) < 50:
                continue
            exc = dn.mean() - bs.mean()
            lo_ci, hi_ci = boot_ci(dn - bs.mean(), rng)
            signal[(d, f)] = (exc, lo_ci, hi_ci)
            print(f"{100*d:>5.1f}% {f:>6}분 {dn.mean():>+8.2f} {bs.mean():>+8.2f} {exc:>+8.2f} "
                  f"[{lo_ci:>+7.2f},{hi_ci:>+7.2f}] {up.mean():>+11.2f}")
    # 🔴판정은 «유의한가»가 아니라 **«비용을 넘는가»** 다(2026-09-13).
    # 9칸을 95%로 검정하면 우연히 0.45칸이 통과한다. 실제로 n=4,000 에서는 1.0%/1440 이
    # 유의하게 **음수**였는데 n=12,000 에서는 사라지고 1.0%/60 이 유의하게 **양수**가 됐다 --
    # 칸이 표본마다 자리를 옮기면 그건 신호가 아니라 다중검정 잡음이다.
    # 그리고 추가 진입도 왕복 비용을 낸다. 초과가 COST_BP 를 넘지 못하면 실행할 이유가 없다.
    sig = [k for k, (e, l, h) in signal.items() if l > 0]
    paid = [k for k, (e, l, h) in signal.items() if l > COST_BP]
    print(f"\n⇒ 유의하게 양수인 칸 {len(sig)}/{len(signal)} "
          f"(9칸 95% 검정의 우연 기대치 0.45 — {'기대 범위' if len(sig) <= 2 else '초과'})"
          + (f" {sig}" if sig else ""))
    print(f"⇒ **비용({COST_BP}bp)을 넘는 칸: {len(paid)}/{len(signal)}**"
          + (f" {paid}" if paid else " — 물린 뒤 추가 진입이 비용을 낼 만큼 유리해지지 않는다"))

    print("\n" + "=" * 76)
    print("L2 전략 — 크기(평균 노출)를 맞춰도 물타기가 이기나")
    print("=" * 76)
    print(f"{'물림':>6} {'팔':>10} {'평균노출':>9} {'건당bp':>9} {'노출당bp':>10} {'파산%':>7} {'로그성장':>10}")
    verdict = {}
    for d in DRAWDOWNS:
        r = layer2(px, hi, lo, rng, a.n, L=8.0, hold=WATCH, d=d, acc=a.acc)
        for k, lab in (("A", "일괄"), ("B", "물타기"), ("C", "크기매칭")):
            v = r[k]
            print(f"{100*d:>5.1f}% {lab:>10} {v['expo']:>9.2f} {v['ret_bp']:>+9.2f} "
                  f"{v['per_expo_bp']:>+10.2f} {100*v['ruin']:>6.2f}% {v['growth']:>+10.5f}")
        verdict[d] = r["B"]["growth"] > r["C"]["growth"]
        print(f"{'':>6} {'':>10} ⇒ 물타기 vs 크기매칭: "
              f"**{'물타기' if verdict[d] else '크기매칭'}** 승\n")

    # ── 결론 고정 ────────────────────────────────────────────────────────────
    assert not paid, f"초과가 비용선을 넘은 칸이 생겼다 -- 신호 층을 다시 볼 것: {paid}"
    assert len(sig) <= 2, f"유의 칸이 다중검정 기대치를 크게 넘었다: {sig}"
    assert not any(verdict.values()), f"물타기가 크기매칭을 이겼다 -- 재현 확인 필요: {verdict}"
    print("확인: L1 신호 0 · L2 에서 물타기가 크기매칭을 못 이김 ⇒ 물타기는 «무작위 시점의 크기 증가»다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
