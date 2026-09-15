"""E|r| 를 «크기」에 쓰는 것 — 세 검정 + 모델 필요성 (2026-09-16).

[[evr_gate_entry_blocking_rejected_20260916]] 이 «차단은 달러를 잃는다」로 닫히며 남긴 질문.

⭐⭐**노출을 맞추지 않은 비교는 전부 무효다.** 09-15 철회본(`49b84bb`)의 `권고수량 × q` 는
노출을 42% 로 줄이는 규칙이라, 그 Δ 에는 «배분을 바꾼 효과」와 «덜 건 효과」가 섞여 있다.
노출을 맞추면 `×q` 와 `×2q` 는 **완전히 같은 규칙**이 된다(상수배). 그래서 여기서는 모든
규칙을 현행과 **같은 총노출**로 정규화한 뒤 짝지어 비교한다.

세 축으로 가른다:
  (1) 플랫   — 신호 없이 명목 편차만 없앤다. «E|r| 의 이득」처럼 보이는 것의 대부분이 여기다.
  (2) 기울기 — 같은 노출을 백분위에 비례해 다시 나눈다.
  (3) 대조군 — 그 백분위가 **E|r| 모델이어야 하나**, 1시간 실현변동성이면 되나.
"""
import importlib.util, sys, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location(
    "gv", ROOT / "scripts/research_evr_gate_entry_validation_20260916.py")
GV = importlib.util.module_from_spec(_s); _s.loader.exec_module(GV)   # load_trips 재사용
RNG = np.random.default_rng(20260916)


def realized_vol_pct(t: pd.DataFrame, bars: int) -> np.ndarray:
    """진입 시점까지의 실현변동성 백분위 — **모델 없음**. E|r| 의 대조군."""
    pan = pd.read_parquet(ROOT / "data/binance_vision/panel/ETHUSDT.parquet",
                          columns=["timestamp", "close"])
    pan["timestamp"] = pd.to_datetime(pan["timestamp"], utc=True)
    rv = np.log(pan["close"]).diff().rolling(bars).std().shift(1).to_numpy()
    j = np.searchsorted(pan["timestamp"].to_numpy(), t["entry_at"].to_numpy(), "left") - 1
    return np.array([float((rv[:k][np.isfinite(rv[:k])] < rv[k]).mean()) for k in j])


def flat_base(t: pd.DataFrame) -> np.ndarray:
    """«늘 같은 크기» — 직전까지 써온 명목의 확장 평균(인과)."""
    n = t["notional"].to_numpy(float)
    return pd.Series(n).expanding().mean().shift(1).fillna(n[0]).to_numpy()


def alloc(base: np.ndarray, score: np.ndarray | None) -> np.ndarray:
    """score 에 비례해 나누되 **총노출은 base 와 같게** 맞춘다."""
    if score is None:
        return base
    x = base * score
    return x * base.sum() / x.sum()


def path_stats(bp: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    d = bp * w / 1e4; cum = d.cumsum()
    return float(d.sum()), float((np.maximum.accumulate(cum) - cum).max())


def paired(t: pd.DataFrame, a: tuple, b: tuple, label: str, n=8000) -> None:
    """(base, score) 두 규칙의 짝지은 달러 차이에 세 검정 + 한 건 의존성."""
    bp = t["net_bp"].to_numpy()
    days = t["entry_at"].dt.floor("D")
    idxs = [np.where(days == d)[0] for d in sorted(days.unique())]
    d = bp * (alloc(*a) - alloc(*b)) / 1e4
    h = len(t) // 2
    boot = np.array([
        float((bp[p] * (alloc(a[0][p], None if a[1] is None else a[1][p])
                        - alloc(b[0][p], None if b[1] is None else b[1][p]))).sum() / 1e4 / len(p))
        for p in (np.concatenate([idxs[i] for i in RNG.integers(0, len(idxs), len(idxs))])
                  for _ in range(n))])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    o = np.argsort(-np.abs(d))
    print(f"  {label:<30} {d.sum():>+8.2f}$ | 전반 {d[:h].sum():>+7.1f} 후반 {d[h:].sum():>+7.1f}"
          f" | CI [{lo:>+6.2f},{hi:>+6.2f}] P(<=0)={np.mean(boot <= 0):.3f} {'✅' if lo > 0 else '🔴'}"
          f" | 1/5/10건 제거 {d[o[1:]].sum():>+6.0f}/{d[o[5:]].sum():>+6.0f}/{d[o[10:]].sum():>+6.0f}")


def main() -> int:
    t = GV.load_trips(hourly=False)                  # 배포 규약: 최신 5분봉
    bp = t["net_bp"].to_numpy()
    cur = t["notional"].to_numpy(float)
    flat = flat_base(t)
    q = t["evr_q"].to_numpy()
    rvq = realized_vol_pct(t, 12)                    # 1시간 실현변동성 백분위
    print(f"실원장 {len(t)}왕복 · {t['entry_at'].min():%Y-%m-%d}~{t['entry_at'].max():%Y-%m-%d}"
          f" · 독립일 {t['entry_at'].dt.floor('D').nunique()}일 · "
          f"명목↔E|r|백분위 스피어만 {t['evr_q'].corr(t['notional'], method='spearman'):+.3f}")

    print("\n=== 총노출을 현행과 같게 맞춘 성과 ===")
    print(f"  {'규칙':<22} {'실손익$':>9} {'누적MDD$':>9} {'수익/MDD':>8} {'최대건$':>9}")
    for lab, w in (("현행(사용자)", cur), ("플랫(신호 없음)", alloc(flat, None) * cur.sum() / flat.sum()),
                   ("플랫 × E|r|q", alloc(flat, q) * cur.sum() / flat.sum()),
                   ("플랫 × rv1h_q", alloc(flat, rvq) * cur.sum() / flat.sum()),
                   ("현행 × E|r|q", alloc(cur, q))):
        p, m = path_stats(bp, w)
        print(f"  {lab:<22} {p:>+9.2f} {m:>9.2f} {p/max(m,1e-9):>8.2f} {w.max():>9,.0f}")

    print("\n=== (1) 신호 없이: 명목 편차만 없앤다 ===")
    paired(t, (flat * cur.sum() / flat.sum(), None), (cur, None), "플랫 − 현행")
    print("\n=== (2) 같은 노출을 백분위에 비례해 나눈다 ===")
    paired(t, (flat, q), (flat, None), "플랫×E|r|q − 플랫")
    paired(t, (flat, rvq), (flat, None), "플랫×rv1h_q − 플랫")
    paired(t, (cur, q), (cur, None), "현행×E|r|q − 현행")
    print("\n=== (3) ⭐모델이 꼭 필요한가 — E|r| 이 단순 실현변동성을 이기나 ===")
    paired(t, (flat, q), (flat, rvq), "플랫: E|r|q − rv1h_q")
    paired(t, (cur, q), (cur, rvq), "현행: E|r|q − rv1h_q")
    print(f"\n  상관 E|r|q ↔ rv1h_q {np.corrcoef(q, rvq)[0,1]:+.3f} · "
          f"스피어만 실현bp↔E|r|q {pd.Series(bp).corr(pd.Series(q), method='spearman'):+.3f} "
          f"vs ↔rv1h_q {pd.Series(bp).corr(pd.Series(rvq), method='spearman'):+.3f}")
    return 0


def _selftest() -> None:
    b = np.array([100.0, 100.0]); s = np.array([0.9, 0.1])
    w = alloc(b, s)
    assert abs(w.sum() - b.sum()) < 1e-9                      # 노출 보존
    assert abs(w[0] / w[1] - 9.0) < 1e-9                      # 점수 비율대로
    assert np.allclose(alloc(b, s * 7), w)                    # 상수배는 같은 규칙
    assert np.allclose(alloc(b, None), b)
    p, m = path_stats(np.array([100.0, -100.0]), np.array([1e4, 1e4]))
    assert abs(p) < 1e-9 and abs(m - 100.0) < 1e-9            # +100 뒤 -100 → MDD 100
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest(); raise SystemExit(0)
    raise SystemExit(main())
