"""사이징용 **전방 변동성 예측 모델** — 피쳐 빌더 + 학습 + 추론 (2026-09-12).

현행 배포는 `수량 = 기준수량 × (기준ATR / 현재ATR)` 이다. 이 파일은 분모를 **모델 예측**으로
바꾼다: `수량 = 기준수량 × (기준예측 / 현재예측)`.

## 근거 (표본외 18,024건, 2025-09~2026-09)
| | 예측상관 | 손익 SD | 50배 청산 도달률 |
|---|---|---|---|
| 1/ATR (현행) | 0.630 | 115.9 | 7.72% |
| **1/모델** | **0.789** | **101.6 (−12.3%)** | **6.11% (−1.61%p)** |
견고성: 랜덤 시드 8/8 · 분기 5/5 · 겹침 제거(1/8 표본)에서도 −11.2% ·
일 군집 부트 95% CI SD [−15.6%, −9.1%] · 청산율 [−1.99%p, −1.15%p] 둘 다 0 배제.
검정 스크립트: research_sizing_model_vs_formula / _feature_expansion / _model_robustness_20260912.py

## ⭐이 파일이 하나인 이유
학습과 라이브가 **같은 `build_features`** 를 쓴다. 두 벌로 두면 조용히 어긋나고, 그 어긋남은
에러가 아니라 «좀 이상한 수량»으로만 나타난다(이 저장소의 반복 사고 유형).

## 🔴TabPFN 은 안 쓴다 — 정확도가 아니라 구조 때문
같은 10피쳐에서 정확도는 사실상 동급(상관 0.801 vs 0.789)인데 **1행 예측이 57.8초**다.
in-context learner 라 «학습된 모델»이 없고 매 예측마다 5,000행 문맥을 다시 통과시킨다
(적합 0.7초 / 1행 예측 57.8초). 워커 주기 300초의 19% 를 한 코어가 계속 먹는다.
GBM 은 같은 일을 ~0.001초에 한다. 재보고 안 쓰기로 한 것이지 안 재본 게 아니다.

## 2026-09-12 확장 — 10피쳐 → 22피쳐 (가격기하 12열 추가)
재료 스택 41열(증거신호·트리거·문맥·극점확률)과 **같은 타깃·분할·모델·지표**로 붙여 보니
41열이 더 나았다(산포 .4775 → .4544 · 분기 5/5 · 일블록 부트 100%). 그런데 **이득의 대부분이
가격기하 12열**이었고(−3.1% vs 41열 전부 −5.0%) 그 12열은 **klines 만으로 계산된다** —
나머지는 compute_signals·BTC·펀딩·극점 아티팩트를 끌어와야 해서 아래 «파일 하나» 설계가 깨진다.
그래서 12열만 넣었다: 표본외 107,862행에서 1/ATR288 산포 .6116 → **.4634**,
|z|>3 1.42% → **0.74%**, |z|>5 0.29% → **0.09%**. 분기 4/5 개선 · 부트 CI [−.0266,−.0043] · 99.8%.
검정: research_eth_sizing_features_stack_vs_deployed_20260912.py
⚠️**피쳐를 바꾸면 아티팩트를 반드시 다시 학습해야 한다** — `load_model` 이 피쳐 목록을
대조하므로 옛 아티팩트는 거절되고 워커가 조용히 1/ATR 로 떨어진다(죽지는 않는다).

학습: python scripts/live_eth_sizing_vol_model_20260912.py --train
점검: python scripts/live_eth_sizing_vol_model_20260912.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data" / "live" / "eth_sizing_vol_model.joblib"
KL_CSV = ROOT / "binance_data" / "klines" / "ETHUSDT" / "ETHUSDT-5m-api.csv"
HOLD = 48                       # 4시간 -- 타깃 지평(실계좌 왕복 중앙 간격에 가깝다)
TRAIN_END = "2025-08-31"
SEEDS = (990143, 220759, 380136, 923411, 331386, 602160, 700199, 982044)  # 랜덤 추출, 리포트에 고정
# 2026-09-12 확장: 가격기하 12열 추가(10 → 22). 재료 스택 41열 비교에서 **이득의 대부분이
# 이 12열**이었다(산포 −3.1% vs 41열 전부 −5.0%). 나머지는 compute_signals·BTC·펀딩·극점
# 아티팩트를 끌어와야 해서 «학습과 라이브가 같은 함수 하나» 설계가 깨진다 — 그래서 안 넣는다.
# 근거: research_eth_sizing_features_stack_vs_deployed_20260912.py
FEATURES = ["atr288", "rv12", "rv48", "rv288", "volexp",
            "hour_sin", "hour_cos", "dow", "qv_z", "nt_z"] + [
    f"{k}{w}" for w in (12, 48, 144) for k in ("ret", "pos_in_range", "dist_lo", "dist_hi")]
WARMUP = 400                    # 288 롤링(+기하 144) + 여유. 이보다 짧은 프레임은 거절한다


def build_features(ts: pd.Series, close: np.ndarray, quote_vol: np.ndarray,
                   trades: np.ndarray, high: np.ndarray, low: np.ndarray) -> pd.DataFrame:
    """**순수 함수** — 전부 해당 봉까지의 정보만 본다. 학습과 라이브가 이 함수 하나를 공유한다.

    로그를 씌우는 것들(atr·rv)은 양수·우편향이고 크기가 1/예측 이라 **비율 오차**가 중요하다.
    volexp(단기/장기 비)는 이미 비율이라 그대로 둔다.

    ⚠️`high`/`low` 는 2026-09-12 에 **필수 인자로** 추가했다(기본값을 두고 종가로 떨어지게 하면
    호출부가 조용히 다른 피쳐를 만든다 — 이 파일이 애초에 막으려던 사고 유형이다).
    종가만으로 만든 변형도 재봤는데 산포 −2.5% 로 고저판(−3.1%)보다 나빴다.
    """
    c = np.asarray(close, float)
    hi = np.asarray(high, float)
    lo = np.asarray(low, float)
    lr = np.diff(np.log(np.maximum(c, 1e-12)), prepend=0.0)
    tr = np.abs(np.diff(c, prepend=c[0]))
    f = {}
    f["atr288"] = np.log(np.maximum(
        pd.Series(tr).rolling(288, min_periods=72).mean().to_numpy() / np.maximum(c, 1e-9), 1e-12))
    for w in (12, 48, 288):
        f[f"rv{w}"] = np.log(np.maximum(
            pd.Series(lr).rolling(w, min_periods=max(8, w // 4)).std().to_numpy(), 1e-12))
    f["volexp"] = np.exp(f["rv12"]) / np.maximum(np.exp(f["rv288"]), 1e-12)
    ti = pd.DatetimeIndex(ts)
    hour = ti.hour.to_numpy() + ti.minute.to_numpy() / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow"] = ti.dayofweek.to_numpy().astype(float)
    for arr, nm in ((quote_vol, "qv"), (trades, "nt")):
        s = pd.Series(np.asarray(arr, float))
        f[f"{nm}_z"] = ((s - s.rolling(288).mean()) / s.rolling(288).std()).to_numpy()
    # 가격기하 — 「어디에 서 있나」. 변동성 수준(atr·rv)이 못 담는 축이다.
    cs = pd.Series(c)
    for w in (12, 48, 144):
        rmin = pd.Series(lo).rolling(w).min().to_numpy()
        rmax = pd.Series(hi).rolling(w).max().to_numpy()
        f[f"ret{w}"] = (cs / cs.shift(w) - 1).to_numpy()
        f[f"pos_in_range{w}"] = (c - rmin) / np.maximum(rmax - rmin, 1e-9)
        f[f"dist_lo{w}"] = (c - rmin) / np.maximum(c, 1e-9)
        f[f"dist_hi{w}"] = (rmax - c) / np.maximum(c, 1e-9)
    out = pd.DataFrame(f)[FEATURES]
    assert list(out.columns) == FEATURES, "피쳐 순서 계약 위반"
    return out


def predict_vol(models, X: pd.DataFrame) -> np.ndarray:
    """시드 앙상블은 **로그 공간 평균**(=기하평균). 예측 대상이 양수·우편향이라 자연스럽고,
    한 시드가 튀어도 산술평균만큼 끌려가지 않는다."""
    A = np.nan_to_num(X.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.exp(np.mean([m.predict(A) for m in models], axis=0))


def load_model():
    """없거나 깨졌으면 None -- 호출부가 현행 1/ATR 공식으로 떨어진다. 여기서 예외를 올리면
    워커가 죽고 화면이 통째로 빈다."""
    try:
        import joblib
        art = joblib.load(ARTIFACT)
        assert art["features"] == FEATURES, "아티팩트 피쳐가 코드와 다르다"
        assert art["ref_pred"] > 0
        return art
    except Exception as exc:  # noqa: BLE001
        print(f"sizing_vol_model load failed: {exc}", flush=True)
        return None


def train() -> int:
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor

    d = pd.read_csv(KL_CSV, usecols=["timestamp", "close", "high", "low", "quote_volume", "trades"],
                    parse_dates=["timestamp"]).dropna(subset=["timestamp"])
    d = d.sort_values("timestamp").reset_index(drop=True)
    X = build_features(d["timestamp"], d["close"].to_numpy(float),
                       d["quote_volume"].to_numpy(float), d["trades"].to_numpy(float),
                       d["high"].to_numpy(float), d["low"].to_numpy(float))
    lr = np.diff(np.log(np.maximum(d["close"].to_numpy(float), 1e-12)), prepend=0.0)
    y = pd.Series(lr).rolling(HOLD, min_periods=HOLD).std().shift(-HOLD).to_numpy()

    ok = np.isfinite(y) & (y > 0) & np.isfinite(X.to_numpy(float)).all(1)
    is_tr = (d["timestamp"] <= TRAIN_END).to_numpy() & ok
    A = np.nan_to_num(X.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"학습 {is_tr.sum():,}행 · 피쳐 {len(FEATURES)} · 시드 {len(SEEDS)}", flush=True)

    models = []
    for s in SEEDS:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6,
                                          random_state=int(s))
        m.fit(A[is_tr], np.log(y[is_tr]))
        models.append(m)

    # 기준값은 **학습구간 예측의 중앙값**이다 -- 현행 `atr_pct_ref`(학습구간 ATR 중앙값)와
    # 같은 역할이라 수량 눈금이 급변하지 않는다.
    ref = float(np.median(predict_vol(models, X[is_tr])))
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "features": FEATURES, "ref_pred": ref,
                 "seeds": list(SEEDS), "train_end": TRAIN_END, "hold_bars": HOLD,
                 "n_train": int(is_tr.sum())}, ARTIFACT)
    print(f"저장 {ARTIFACT} · 기준예측 {ref:.6g} · {ARTIFACT.stat().st_size / 1e6:.1f}MB", flush=True)
    return 0


def _self_check() -> None:
    """네트워크·아티팩트 없이 도는 계약 검사."""
    n = 900
    ts = pd.Series(pd.date_range("2026-01-01", periods=n, freq="5min"))
    rng = np.random.default_rng(0)
    c = 2000 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    qv = np.abs(rng.normal(1e6, 2e5, n)); nt = np.abs(rng.normal(5000, 800, n))
    hi = c * (1 + np.abs(rng.normal(0, 0.0006, n)))      # 고가 ≥ 종가 ≥ 저가
    lo = c * (1 - np.abs(rng.normal(0, 0.0006, n)))

    X = build_features(ts, c, qv, nt, hi, lo)
    assert list(X.columns) == FEATURES, X.columns
    assert len(X) == n
    assert np.isfinite(X.to_numpy(float)[WARMUP:]).all(), "워밍업 뒤에는 결측이 없어야 한다"

    # ⭐인과성: 앞부분만 잘라 만든 피쳐가 전체판의 같은 구간과 일치해야 한다.
    #   미래를 보는 항이 하나라도 있으면 여기서 어긋난다.
    m = 700
    Xp = build_features(ts[:m], c[:m], qv[:m], nt[:m], hi[:m], lo[:m])
    a, b = X.iloc[WARMUP:m].to_numpy(float), Xp.iloc[WARMUP:].to_numpy(float)
    assert np.allclose(a, b, rtol=1e-9, atol=1e-12), f"인과성 위반 최대차 {np.abs(a - b).max():.3g}"

    # 시간 피쳐가 실제로 시간을 담는가(상수면 조용히 죽은 피쳐가 된다)
    assert X["hour_sin"].nunique() > 100 and X["dow"].nunique() >= 1
    # 로그 스케일 확인 -- atr288 은 로그라 음수여야 한다(원시 비율이면 양수)
    assert X["atr288"].iloc[-1] < 0, "atr288 이 로그가 아니다"
    # volexp 는 비율이라 양수
    assert X["volexp"].iloc[-1] > 0
    # 2026-09-12 추가한 가격기하 12열
    assert len(FEATURES) == 22 and "pos_in_range144" in FEATURES, FEATURES
    for w in (12, 48, 144):
        v = X[f"pos_in_range{w}"].to_numpy()[WARMUP:]
        assert (v >= -1e-9).all() and (v <= 1 + 1e-9).all(), f"pos_in_range{w} 가 [0,1] 밖"
        assert (X[f"dist_lo{w}"].to_numpy()[WARMUP:] >= -1e-9).all(), f"dist_lo{w} 음수"
        assert (X[f"dist_hi{w}"].to_numpy()[WARMUP:] >= -1e-9).all(), f"dist_hi{w} 음수"
        assert X[f"ret{w}"].nunique() > 100, f"ret{w} 가 죽은 피쳐다"

    class _Stub:
        def __init__(self, v): self.v = v
        def predict(self, A): return np.full(len(A), self.v)
    p = predict_vol([_Stub(np.log(0.001)), _Stub(np.log(0.004))], X.iloc[-3:])
    assert np.allclose(p, 0.002), f"로그평균(기하평균)이 아니다: {p}"

    print("통과 9/9 — 피쳐 계약(22열) · 인과성 · 로그스케일 · 기하평균 · 가격기하 범위")


if __name__ == "__main__":
    if "--train" in sys.argv:
        raise SystemExit(train())
    _self_check()
