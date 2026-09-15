"""E|r| 게이트를 «진입 게이트»로 켜기 전 검증 (2026-09-16).

09-15 실원장 분석(§5.36-R, 72왕복 단조 +10.34/+14.32/+53.18bp)에는 **세 검정이 빠져 있었다** —
순차분할·날짜블록 CI·독립일수. 정확히 이 셋이 사이징 배수(③)를 죽였다. 여기에 더해
«한 건 의존성»([[account_t_gap_is_one_trade_not_sizing_20260915]])과 임계값 민감도를 본다.

재현: 패널 ETHUSDT + 동결 아티팩트 `evr` 회귀기(재계산이 evr_history 와 5e-7 이내 일치 확인).
백분위는 인과 확장창 rank(자기 자신 제외) — 시간봉 격자, 아티팩트 seed 와 동일.
"""
import json, os, sys, numpy as np, pandas as pd, joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/claude-1000")
# ⚠️ 배포 워커가 실제로 읽는 아티팩트를 쓴다 — `live_evr_gate_worker_20260915.py:49` 의
# DIR_ARTIFACT 기본값. 09-15 §5.36-R 이 검증한 direction_4h_top5 는 **은퇴본**이다.
ART_NAME = os.environ.get("DIR_ARTIFACT_NAME", "direction_1d_top10_20260915")
ART = ROOT / "data/models" / ART_NAME
EXPAND_H = 8064          # 확장창 하한 (시간봉 8064 = 336일)
CUT = 0.80               # 표본내 상위 20% — 09-15 분석이 자른 구간
RNG = np.random.default_rng(20260916)


def evr_series(hourly: bool) -> tuple[np.ndarray, np.ndarray]:
    """log E|r| 예측 시계열.

    `hourly=False`(기본)가 **배포 규약**이다 — 워커는 매 사이클 «최신 5분봉」으로 예측한다
    (`live_evr_gate_worker_20260915.py::tail_frame` → `predict(x[-1:])`). 순위는 기준분포에
    불변이므로 워커가 시간봉 seed 에 견주는 것과 여기 5분봉 히스토리에 견주는 것은 **같은 구간**을
    만든다.
    `hourly=True`는 «직전 정시봉만 본다» 는 대조군 — 진입까지 중앙 31분 묵은 예측이다.
    ⭐신선도가 전부다: 두 예측의 상관은 0.67 에 불과하고, 실현 bp 와의 스피어만이
    **0.327(신선) vs 0.127(31분 묵음)**. 09-15 §5.36-R 은 신선 쪽을 썼다.
    """
    tag = "evr1d" if "1d_top10" in ART_NAME else "evr_pred"
    tp, vp = CACHE / f"{tag}_ts.npy", CACHE / f"{tag}_val.npy"
    if tp.exists() and vp.exists():
        ts, val = np.load(tp), np.load(vp)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rx", ROOT / "scripts/research_direction_event_expansion_20260915.py")
        RX = importlib.util.module_from_spec(spec); spec.loader.exec_module(RX)
        art = joblib.load(ART / "models.joblib")
        raw = pd.read_parquet(ROOT / "data/binance_vision/panel/ETHUSDT.parquet")
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        p = RX._features(raw, "ETH", False, "2000-01-01")
        x = p[art["cols"]].to_numpy(np.float32); ok = np.isfinite(x).all(1)
        ts = pd.to_datetime(p["timestamp"]).dt.tz_localize("UTC").astype("int64").to_numpy()[ok]
        val = art["evr"].predict(x[ok])
    if not hourly:
        return ts, val
    m = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).minute == 0
    return ts[m], val[m]


def load_trips(hourly: bool = True) -> pd.DataFrame:
    ts, val = evr_series(hourly)
    warm = EXPAND_H if hourly else EXPAND_H * 12
    rows = [json.loads(l) for l in open(ROOT / "data/live/account_round_trips.jsonl")]
    t = pd.DataFrame([r for r in rows if r.get("closed")])
    t["entry_at"] = pd.to_datetime(t["entry_time"], unit="ms", utc=True)   # entry_at 컬럼은 68/72 결측
    t["notional"] = t["max_qty"] * t["entry_price"]
    t["net_bp"] = t["net_pnl"] / t["notional"] * 1e4
    # 진입 시각 **이전에 확정된** 마지막 시간봉만 본다. 백분위는 그 시점까지의 과거만으로.
    j = np.searchsorted(ts, t["entry_at"].astype("int64").to_numpy(), side="left") - 1
    t["evr_q"] = [float((val[:k] < val[k]).mean()) if k >= warm else np.nan for k in j]
    t["bar_at"] = pd.to_datetime(ts[np.array(j).clip(0)], utc=True)
    t["lag_min"] = (t["entry_at"] - t["bar_at"]).dt.total_seconds() / 60
    return t.dropna(subset=["evr_q"]).sort_values("entry_at").reset_index(drop=True)


def increment(bp: np.ndarray, qq: np.ndarray, cut: float = CUT) -> float:
    """게이트 통과분 평균 − 전체 평균 (bp/건). 구간은 **그 표본 안의 순위**로 자른다
    (09-15 §5.36-R 와 같은 방식 — 절대 임계 q>=cut 은 `--absolute` 로 따로 본다)."""
    if bp.size == 0:
        return float("nan")
    r = pd.Series(qq).rank(pct=True).to_numpy()
    g = bp[r > cut]
    return float("nan") if g.size == 0 else float(g.mean() - bp.mean())


def date_block_ci(t: pd.DataFrame, cut: float, n=20000) -> tuple:
    """날짜 단위 블록 부트스트랩 — 같은 날 왕복은 통째로 함께 뽑힌다."""
    groups = [(g["net_bp"].to_numpy(), g["evr_q"].to_numpy())
              for _, g in t.groupby(t["entry_at"].dt.floor("D"))]
    out = []
    for _ in range(n):
        pick = RNG.integers(0, len(groups), len(groups))
        bp = np.concatenate([groups[i][0] for i in pick])
        qq = np.concatenate([groups[i][1] for i in pick])
        v = increment(bp, qq, cut)
        if np.isfinite(v):
            out.append(v)
    a = np.array(out)
    return np.percentile(a, 2.5), np.percentile(a, 97.5), float((a <= 0).mean()), len(a) / n


def main() -> int:
    hourly = "--stale-hourly" in sys.argv        # 기본 = 배포 규약(최신 5분봉)
    t = load_trips(hourly)
    t["r"] = t["evr_q"].rank(pct=True)
    bp, qq = t["net_bp"].to_numpy(), t["evr_q"].to_numpy()
    g = t[t.r > CUT]
    print(f"[아티팩트 {ART_NAME} · 격자 "
          f"{'대조군: 직전 정시봉(31분 묵음)' if hourly else '배포 규약: 최신 5분봉'}]")
    print(f"왕복 {len(t)}건 · {t['entry_at'].min():%Y-%m-%d} ~ {t['entry_at'].max():%Y-%m-%d}"
          f" · 봉→진입 지연 중앙 {t['lag_min'].median():.0f}분")
    print(f"전체 평균 {bp.mean():+.2f}bp · 게이트({CUT:.0%}+) n={len(g)} 평균 "
          f"{g['net_bp'].mean():+.2f}bp · 증분 {increment(bp,qq,CUT):+.2f}bp/건")

    print("\n== 검정 1: 순차 분할 (전반/후반) ==")
    half = len(t) // 2
    for name, part in (("전반", t.iloc[:half]), ("후반", t.iloc[half:])):
        pg = part[part["evr_q"].rank(pct=True) > CUT]
        gm = pg["net_bp"].mean() if len(pg) else float("nan")
        print(f"  {name} n={len(part):2d} (게이트 {len(pg):2d}) "
              f"{part['entry_at'].min():%m-%d}~{part['entry_at'].max():%m-%d} · "
              f"전체 {part['net_bp'].mean():+7.2f} 게이트 {gm:+7.2f} 증분 "
              f"{increment(part['net_bp'].to_numpy(), part['evr_q'].to_numpy(), CUT):+7.2f}bp")

    print("\n== 검정 2: 날짜블록 부트스트랩 CI (증분) ==")
    lo, hi, pneg, cov = date_block_ci(t, CUT)
    print(f"  95% CI [{lo:+.2f}, {hi:+.2f}] · P(증분<=0) = {pneg:.3f} · "
          f"게이트 비는 표본 {1-cov:.1%} · {'0 배제' if lo > 0 else '🔴 0 포함'}")

    print("\n== 검정 3: 독립일 수 ==")
    days = t["entry_at"].dt.floor("D"); gd = g["entry_at"].dt.floor("D")
    print(f"  전체 {days.nunique()}일 (하루 {len(t)/days.nunique():.1f}건) · "
          f"게이트 통과 {len(g)}건이 **{gd.nunique()}일**에 몰려 있다 "
          f"(하루 {len(gd)/max(gd.nunique(),1):.1f}건)")
    print(f"  게이트 일자별: " + " ".join(
        f"{d:%m-%d}×{c}" for d, c in gd.dt.floor('D').value_counts().sort_index().items()))

    print("\n== 검정 4: 한 건 의존성 (게이트 통과분 최고수익 k건 제거) ==")
    for k in (0, 1, 2, 3):
        tt = t.drop(index=t[t.r > CUT].nlargest(k, "net_bp").index) if k else t
        print(f"  상위 {k}건 제거 → 증분 "
              f"{increment(tt['net_bp'].to_numpy(), tt['evr_q'].to_numpy(), CUT):+7.2f}bp "
              f"(게이트 n={int((tt.evr_q.rank(pct=True) > CUT).sum())})")

    print("\n== 검정 5: 임계값 민감도 ==")
    for c in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        gc = t[t.r > c]
        m = gc["net_bp"].mean() if len(gc) else float("nan")
        print(f"  표본내 상위 {1-c:>4.0%}: n={len(gc):2d} 평균 {m:+7.2f} "
              f"증분 {increment(bp,qq,c):+7.2f}bp")

    print("\n== 검정 6: 실손익(달러) — «켜면 무엇을 잃는가» ==")
    print(f"  전체 ${t['net_pnl'].sum():+.2f} / 게이트 통과분만 ${g['net_pnl'].sum():+.2f}")
    print(f"  게이트가 거절하는 {len(t)-len(g)}건 합계 "
          f"${t['net_pnl'].sum()-g['net_pnl'].sum():+.2f}")
    print(f"  평균명목 게이트 ${g['notional'].mean():,.0f} vs 전체 ${t['notional'].mean():,.0f} "
          f"· 명목↔백분위 상관 {t['evr_q'].corr(t['notional']):+.3f}")
    return 0


def _selftest() -> None:
    bp = np.array([1.0, 2.0, 3.0, 10.0]); qq = np.array([0.1, 0.5, 0.9, 0.95])
    assert abs(increment(bp, qq, 0.8) - (10.0 - 4.0)) < 1e-9   # 표본내 순위 상위 20% = 마지막 1건
    assert abs(increment(bp, qq, 0.4) - (5.0 - 4.0)) < 1e-9    # 상위 60% = 뒤 3건
    assert np.isnan(increment(np.array([]), np.array([])))
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest(); raise SystemExit(0)
    raise SystemExit(main())
