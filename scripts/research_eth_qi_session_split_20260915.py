"""**큐 불균형의 시간대 분할** — 반감기가 세션마다 다른가 (2026-09-15).

하루치(31.1M 이벤트)로 QI 의 예측력과 비용 벽이 확정됐다: IC 0.1초 +0.646 인데 관측/필요가
2.2~7.1% 로 14~45배 모자란다. 집행 쪽도 다섯 경로가 전부 막혔고, 그 중 ⑤(배치지연)의 근거가
**반감기 0.30초**였다 — 0.5초면 예측력의 17%만 남는다는 표.

남은 한 갈래: **그 반감기가 시간대마다 다르면** 「0.09초급 배치가 필요하다」가 특정 세션에서는
완화될 수 있다. 아시아 새벽처럼 호가가 느린 구간에서는 큐가 오래 서 있을 수 있다.

🔴피어(`crypto-scalping-f9`)의 `research_eth_microprice_second_horizon_20260914.py` 를 **수정하지
않고 import 만** 한다 — 같은 워킹트리를 공유하므로 남의 파일을 고치지 않는다(09-14 사고 교훈).

세션 구분(UTC): 아시아 00~08 · 유럽 08~16 · 미국 16~24. 세션마다:
  · QI 자기상관 반감기(지연 격자에서 ρ 가 0.5 를 지나는 지점을 로그선형 보간)
  · 지평별 IC · E|r| · 관측/필요
  · 이벤트 밀도(초당 행) — 「느린 시간대는 정말 느린가」
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_microprice_second_horizon_20260914 as MP  # noqa: E402

OUT = ROOT / "data/research/eth_direction_barrier_label_20260914"
SESSIONS = (("아시아", 0, 8), ("유럽", 8, 16), ("미국", 16, 24))
LAGS_MS = (50, 100, 200, 300, 500, 1000, 2000, 5000)


def half_life(ts: np.ndarray, x: np.ndarray, lags=LAGS_MS) -> tuple[float, dict]:
    """지연 격자에서 ρ 를 재고 0.5 를 지나는 지점을 **로그선형 보간**한다.
    격자 사이에서 끊기면 그 사실을 그대로 돌려준다(보간 불가 = NaN)."""
    rho = {}
    for L in lags:
        j = np.searchsorted(ts, ts + L)
        ok = (j < len(ts)) & (np.abs(ts[np.clip(j, 0, len(ts) - 1)] - (ts + L)) <= L)
        if ok.sum() < 5000:
            continue
        rho[L] = float(spearmanr(x[ok], x[j[ok]]).statistic)
    ks = sorted(rho)
    hl = float("nan")
    for a, b in zip(ks, ks[1:]):
        if rho[a] >= 0.5 > rho[b]:
            w = (rho[a] - 0.5) / max(rho[a] - rho[b], 1e-12)
            hl = float(np.exp(np.log(a) + w * (np.log(b) - np.log(a))))
            break
    return hl, rho


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="0.1,0.5,1,2,5,30")
    ap.add_argument("--cost-bp", type=float, default=5.88)
    ap.add_argument("--tag", default="qi_session_split")
    a = ap.parse_args()
    ts, bid, ask, qb, qa = MP.load_bt()      # 튜플로 돌려준다(그 파일 시그니처)
    mid = (bid + ask) / 2.0
    lm = np.log(np.maximum(mid, 1e-9))
    qi = (qb - qa) / np.maximum(qb + qa, 1e-9)
    hour = ((ts // 3_600_000) % 24).astype(int)
    span_h = (ts[-1] - ts[0]) / 3.6e6
    print(f"bookTicker {len(ts):,}행 · {span_h:.1f}시간 · 시각 단조 {bool((np.diff(ts) >= 0).all())}")
    rep = {}
    for name, h0, h1 in SESSIONS:
        m = (hour >= h0) & (hour < h1)
        if m.sum() < 200_000:
            print(f"{name}: {m.sum():,}행 — 표본 부족, 건너뜀"); continue
        t_s, x_s, l_s = ts[m], qi[m], lm[m]
        dur = (t_s[-1] - t_s[0]) / 1000
        hl, rho = half_life(t_s, x_s)
        cell = {"n": int(m.sum()), "rate_per_s": float(m.sum() / max(dur, 1)),
                "half_life_ms": hl, "rho": rho, "h": {}}
        print(f"\n=== {name} (UTC {h0:02d}~{h1:02d}) · {m.sum():,}행 · {m.sum()/max(dur,1):.0f}행/초 ===")
        print(f"  자기상관 ρ: " + " · ".join(f"{L}ms {rho[L]:.3f}" for L in sorted(rho)))
        print(f"  ⭐반감기 {hl:.0f}ms" if np.isfinite(hl) else "  반감기 격자 밖")
        print(f"  {'지평':>7} {'IC':>9} {'E|r|bp':>8} {'손익분기IC':>10} {'관측/필요':>8}")
        for H in [float(v) for v in a.horizons.split(",")]:
            fwd = MP.forward_return(t_s, l_s, int(H * 1000))
            ok = np.isfinite(fwd) & np.isfinite(x_s)
            if ok.sum() < 20_000:
                continue
            ic = float(spearmanr(x_s[ok], fwd[ok]).statistic)
            e = float(np.mean(np.abs(fwd[ok])) * 1e4)
            be = a.cost_bp / (e * 0.7979) if e > 0 else np.nan
            cell["h"][str(H)] = {"ic": ic, "e_abs_bp": e, "breakeven_ic": be, "ratio": ic / be,
                                 "n": int(ok.sum())}
            print(f"  {H:>6}s {ic:>+9.4f} {e:>8.3f} {be:>10.2f} {ic/be:>8.1%}")
        rep[name] = cell
    hls = {k: v["half_life_ms"] for k, v in rep.items() if np.isfinite(v.get("half_life_ms", np.nan))}
    if len(hls) >= 2:
        print(f"\n⭐반감기 세션별: " + " · ".join(f"{k} {v:.0f}ms" for k, v in hls.items()) +
              f"  (최대/최소 {max(hls.values())/min(hls.values()):.2f}배)")
    best = {k: max((c["ratio"] for c in v["h"].values()), default=float("nan")) for k, v in rep.items()}
    print("세션별 최대 관측/필요: " + " · ".join(f"{k} {v:.1%}" for k, v in best.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(f"저장: {OUT/(a.tag+'.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
