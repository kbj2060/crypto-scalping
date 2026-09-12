"""C(TabPFN 진입) 원장의 진입가가 단/중/장기 저점이었나 — 순환이동 귀무 대조.

adverse = 롱이면 창 내 백분위, 숏이면 1-백분위. 0=완벽한 저점매수/고점매도, 0.5=무작위.
귀무는 진입 인덱스 전체를 순환이동(간격 구조·측면 구성 보존)한 B회 재계산.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
retest = importlib.import_module("scripts.retest_omega4_6_1_extended_oos_20260706")

LEDGER_DIR = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
HORIZONS = [("단기 1일", 288), ("중기 7일", 2016), ("장기 30일", 8640)]
B_NULL, NULL_SEED = 500, 615372041


def _pct_ranks(px, w, centered):
    """px[i] 가 창 안에서 몇 분위인가. 0=창 최저, 1=창 최고."""
    n = len(px)
    out = np.empty(n)
    half = w // 2
    for i in range(n):
        lo = max(0, i - half) if centered else max(0, i - w + 1)
        hi = min(n, i + half + 1) if centered else i + 1
        seg = px[lo:hi]
        out[i] = float(np.mean(seg <= px[i]))
    return out


def _adverse(ranks, idx, side):
    r = ranks[idx]
    return np.where(side > 0, r, 1.0 - r)


def _selfcheck():
    px = np.arange(100.0)
    c = _pct_ranks(px, 21, True)
    assert c[50] == 11 / 21 and c[0] == 1 / 11, c[:3]         # 창 21개 중 11번째(자신 포함), 시작은 창 최저
    k = _pct_ranks(px, 21, False)
    assert k[50] == 1.0, k[50]                                 # 인과 창의 마지막이 항상 최고
    a = _adverse(np.array([0.1, 0.9]), np.array([0, 1]), np.array([1, -1]))
    assert np.allclose(a, [0.1, 0.1]), a                       # 저점 롱 == 고점 숏
    return True


def main():
    assert _selfcheck()
    df = retest.load_frame_current("2026-01-01", "2026-08-30")
    px = df["close"].to_numpy(np.float64)
    n = len(px)
    print(f"[프레임] {n:,}봉  {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}\n")

    ranks = {(name, mode): _pct_ranks(px, w, mode == "사후")
             for name, w in HORIZONS for mode in ("사후", "인과")}

    ledgers = {}
    for f in sorted(LEDGER_DIR.glob("replay_ledger_[AC]*.csv")):
        tag = f.stem[len("replay_ledger_"):]
        if tag == "C":
            continue                                            # C_615372041 과 동일 파일
        ledgers[tag] = pd.read_csv(f)
    pooled = pd.concat([v for k, v in ledgers.items() if k.startswith("C")], ignore_index=True)
    ledgers["C_pooled"] = pooled

    rng = np.random.default_rng(NULL_SEED)
    shifts = rng.integers(1, n, size=B_NULL)

    for name, _w in HORIZONS:
        for mode in ("사후", "인과"):
            R = ranks[(name, mode)]
            print(f"=== {name} · {mode} ===")
            print(f"{'원장':12s} {'n':>4s} {'롱%':>5s} {'adverse':>8s} {'귀무':>7s} {'Δ':>7s} {'p':>6s}"
                  f"  {'롱 adv':>7s} {'숏 adv':>7s}")
            for tag, lg in ledgers.items():
                idx = lg["entry_i"].to_numpy(np.int64)
                side = lg["side"].to_numpy(np.int64)
                obs = float(_adverse(R, idx, side).mean())
                null = np.array([_adverse(R, (idx + s) % n, side).mean() for s in shifts])
                p = float((null <= obs).mean())
                lo = _adverse(R, idx[side > 0], side[side > 0])
                sh = _adverse(R, idx[side < 0], side[side < 0])
                print(f"{tag:12s} {len(lg):4d} {(side>0).mean():5.0%} {obs:8.3f} {null.mean():7.3f}"
                      f" {obs-null.mean():+7.3f} {p:6.3f}  {lo.mean():7.3f} {sh.mean():7.3f}")
            print()

    print("=== adverse ↔ 수익률 상관 (C_pooled, 스피어만) ===")
    lg = ledgers["C_pooled"]
    idx, side = lg["entry_i"].to_numpy(np.int64), lg["side"].to_numpy(np.int64)
    ret = lg["trade_return"].to_numpy(np.float64)
    for name, _w in HORIZONS:
        for mode in ("사후", "인과"):
            a = _adverse(ranks[(name, mode)], idx, side)
            rho = pd.Series(a).corr(pd.Series(ret), method="spearman")
            good, bad = ret[a < 0.5].mean(), ret[a >= 0.5].mean()
            print(f"  {name:9s} {mode}  rho {rho:+.3f}   adv<0.5 평균 {good:+.4f} (n={int((a<0.5).sum())})"
                  f"   adv>=0.5 평균 {bad:+.4f} (n={int((a>=0.5).sum())})")

    print("\n=== 진입 직후 1일(288봉) 역행폭 · 종료사유 (C_pooled) ===")
    R1 = ranks[("단기 1일", "사후")]
    adv = _adverse(R1, idx, side)
    mae = np.array([float(((px[i:i + 289] - px[i]) / px[i] * s_).min()) for i, s_ in zip(idx, side)])
    null_mae = np.array([np.mean([float(((px[j:j + 289] - px[j]) / px[j] * s_).min())
                                  for j, s_ in zip((idx + k) % (n - 289), side)]) for k in shifts[:200]])
    print(f"  실제 1일 MAE 평균 {mae.mean():+.4f}   귀무 {null_mae.mean():+.4f}"
          f"   Δ {mae.mean()-null_mae.mean():+.4f}   p {float((null_mae <= mae.mean()).mean()):.3f}")
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        m = (adv >= lo) & (adv < hi)
        if not m.any():
            continue
        sl = (lg["reason"].to_numpy()[m] == "stop_loss").mean()
        print(f"  adverse [{lo:.2f},{hi:.2f})  n={int(m.sum()):3d}  1일MAE {mae[m].mean():+.4f}"
              f"  수익률 {ret[m].mean():+.4f}  손절비율 {sl:.0%}")


if __name__ == "__main__":
    main()
