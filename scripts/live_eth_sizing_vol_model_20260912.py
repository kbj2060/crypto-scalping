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

학습·매달: python scripts/live_eth_sizing_vol_model_20260912.py --monthly (2026-10-15 서버 cron -- data/live/vol_panel 빠진 날 받기 +
      지난 판 표본외로 확대 확률 상수 + 전 기간 재학습 + 지난 판과 맞대는 품질 문). 🔴10-19 리뷰: 옛 --train(짧은 KL_CSV 로 배포 판을
      백업 없이 덮어썼다)은 지웠다 -- 긴 데이터 판(2021-12~, MSLE −19%)이 실수 한 번에 사라질 수 있었다.
검증: 같은 파일 --verify (배포된 아티팩트를 그 기계의 데이터로 표본외 재검)
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
# 2026-10-15 긴 데이터 재학습(2021-12~ 50만 행) -- 2025-09~2026-10 표본외 비교: 짧은 창(2024-10~) MSLE .1435 → 긴 창 .1161(−19%, 분기 5/5) · 스피어만 .744 → .781
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


def verify() -> int:
    """배포된 아티팩트를 **그 기계의 데이터로** 표본외 재검한다(2026-09-12).

    학습 CSV 길이가 기계마다 다르다(로컬 385k · 서버 175k). 연구 수치를 다른 기계에서
    그대로 주장하면 안 되므로, 배포 직후 이 자리에서 다시 낸다. 현행 이전 공식인 1/ATR288 과
    같은 행에서 비교한다 — 수량이 1/예측 이므로 지표는 **실현변동성/예측의 산포**다.
    """
    import joblib
    from scipy.stats import spearmanr
    art = joblib.load(ARTIFACT)
    d = pd.read_csv(KL_CSV, usecols=["timestamp", "close", "high", "low", "quote_volume", "trades"],
                    parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    X = build_features(d["timestamp"], d["close"].to_numpy(float),
                       d["quote_volume"].to_numpy(float), d["trades"].to_numpy(float),
                       d["high"].to_numpy(float), d["low"].to_numpy(float))
    lr = np.diff(np.log(np.maximum(d["close"].to_numpy(float), 1e-12)), prepend=0.0)
    y = pd.Series(lr).rolling(HOLD, min_periods=HOLD).std().shift(-HOLD).to_numpy()
    ts = d["timestamp"].to_numpy()
    te = (np.isfinite(y) & (y > 0) & (ts > np.datetime64(str(art["train_end"]) + "T23:59:59"))
          & np.isfinite(X.to_numpy(float)).all(1))
    if te.sum() < 1000:
        print(f"표본외 {te.sum()}행 — 너무 적어 판정하지 않는다")
        return 1
    pred = predict_vol(art["models"], X[te])
    atr = np.exp(X["atr288"].to_numpy())[te]
    yt = y[te]
    print(f"아티팩트 피쳐 {len(art['features'])} · 학습 {art['n_train']:,}행 · "
          f"표본외 {te.sum():,}행 ({str(ts[te][0])[:10]} ~ {str(ts[te][-1])[:10]})")
    print("               스피어만    산포   |z|>3   |z|>5")
    for nm, v in (("1/ATR288", atr), ("모델", pred)):
        z = yt / np.maximum(v, 1e-12)
        z = z / np.median(z)
        print("%-12s %8.4f %7.4f %6.2f%% %6.2f%%" % (
            nm, spearmanr(v, yt).statistic, np.std(z), (z > 3).mean() * 100, (z > 5).mean() * 100))
    mult = art["ref_pred"] / pred
    print("수량 배수(기준/예측): 5%% %.2f · 중앙 %.2f · 95%% %.2f" % (
        np.percentile(mult, 5), np.median(mult), np.percentile(mult, 95)))
    return 0


LONG = ROOT / "data" / "live" / "vol_panel" / "ETHUSDT_5m.parquet"   # 2026-10-15 긴 5분봉(2021-12~) -- 매달 vision 으로 이어 받는다
EXPAND_LABEL = 1.3                                                  # 확대 = 다음 4h 실현변동 ≥ 직전 4h × 1.3


def _vision_day(d: str) -> pd.DataFrame | None:
    """data.binance.vision 선물 5분봉 하루(정적 아카이브 -- REST 아님). 없으면 None."""
    import io, urllib.request, zipfile
    try:
        with urllib.request.urlopen(f"https://data.binance.vision/data/futures/um/daily/klines/ETHUSDT/5m/ETHUSDT-5m-{d}.zip", timeout=60) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read())); txt = z.read(z.namelist()[0]).decode()
    except Exception:
        return None
    k = pd.read_csv(io.StringIO(txt))
    if "open_time" not in k.columns:
        k = pd.read_csv(io.StringIO(txt), header=None, names=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                                             "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"])
    return pd.DataFrame({"timestamp": pd.to_datetime(k.open_time, unit="ms"), "close": k.close.astype(float), "high": k.high.astype(float),
                         "low": k.low.astype(float), "quote_volume": k.quote_volume.astype(float), "trades": k["count"].astype(float)})


def fit_expand_logit(pred: np.ndarray, rv48: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """P(y ≥ 1.3·rv48) = sigmoid(a + b·ln(pred/rv48)) -- 표본외 예측으로만 부를 것(학습 예측이면 과신)."""
    from sklearn.linear_model import LogisticRegression
    x = np.log(pred / rv48).reshape(-1, 1)
    lab = (y / rv48 >= EXPAND_LABEL).astype(int)
    m = LogisticRegression(C=1e6).fit(x, lab)
    return float(m.intercept_[0]), float(m.coef_[0][0])


def vs_prev(models, old_models, X: pd.DataFrame) -> dict:
    """품질 문: 새 판과 지난 판을 지난 판 표본외 행에서 맞댄다. 이웃 달 판끼리 로그 예측 상관 .98~.998·중앙 비 .995~1.002
    (10-19 실측) -- 크게 어긋나면 망가진 학습(데이터 구멍·단위 사고)이라 교체하지 않는다. «더 나은가»는 재학습 때 잴 수 없다."""
    ln, lo = np.log(predict_vol(models, X)), np.log(predict_vol(old_models, X))
    out = {"corr_log": round(float(np.corrcoef(ln, lo)[0, 1]), 4), "median_ratio": round(float(np.exp(np.median(ln - lo))), 4), "rows": len(X)}
    if out["corr_log"] < 0.9 or not 0.8 <= out["median_ratio"] <= 1.25:
        raise SystemExit(f"새 판이 지난 판과 너무 다르다 {out} -- 교체 안 함(지난 판 유지)")
    return out


def monthly() -> int:
    """매달 1일 cron: LONG 을 어제까지 이어 받고 → 지난 판으로 최근 30일 표본외 예측 → 확대 확률 상수 적합
    → 전 기간 재학습 → 품질 문(지난 판과 맞대기) → 아티팩트 교체(지난 판은 교체 시각 이름으로 남김)."""
    import joblib
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.ensemble import HistGradientBoostingRegressor
    d = pd.read_parquet(LONG)
    y0 = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta("1D")
    have = set(d.timestamp.dt.normalize())   # 10-19: 빠진 날은 어디든 받는다(옛: 마지막 날 뒤만 -- 서버 LONG 에 2023-12 한 달 구멍이 남아 있었다)
    days = [x.strftime("%Y-%m-%d") for x in pd.date_range(d.timestamp.min().normalize(), y0) if x not in have]
    with ThreadPoolExecutor(8) as ex:
        add = [x for x in ex.map(_vision_day, days) if x is not None]
    d = pd.concat([d] + add).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    tmp = LONG.with_suffix(".tmp")
    d.to_parquet(tmp); tmp.replace(LONG)   # 쓰다 죽어도 지난 LONG 이 남게
    gap = d.timestamp.diff().max()
    print(f"긴 5분봉 {len(d):,}행 ~{d.timestamp.max()} (+{len(add)}/{len(days)}일) · 최대 간격 {gap}", flush=True)
    if gap > pd.Timedelta("1h"):   # 롤링 창이 위치 기준이라 구멍 = 5분봉 하나짜리 거대 수익률로 학습된다
        raise SystemExit(f"5분봉 구멍 {gap} -- 받지 못한 날이 있다, 재학습 안 함(지난 판 유지)")
    X = build_features(d["timestamp"], d["close"].to_numpy(float), d["quote_volume"].to_numpy(float),
                       d["trades"].to_numpy(float), d["high"].to_numpy(float), d["low"].to_numpy(float))
    lr = np.diff(np.log(np.maximum(d["close"].to_numpy(float), 1e-12)), prepend=0.0)
    y = pd.Series(lr).rolling(HOLD, min_periods=HOLD).std().shift(-HOLD).to_numpy()
    ok = np.isfinite(y) & (y > 0) & np.isfinite(X.to_numpy(float)).all(1)
    ts = d["timestamp"].to_numpy()
    old = load_model()
    logit = None
    te = ok & (ts > np.datetime64(pd.Timestamp(old["train_end"]) + pd.Timedelta("1D"))) if old is not None else np.zeros(len(ts), bool)   # 지난 판 표본외
    if old is not None:
        if te.sum() >= 5000:
            logit = fit_expand_logit(predict_vol(old["models"], X[te]), np.exp(X["rv48"].to_numpy())[te], y[te])
            print(f"확대 확률 상수 {logit} · 표본외 {int(te.sum()):,}행", flush=True)
    A = np.nan_to_num(X.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    models = [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=int(s)).fit(A[ok], np.log(y[ok]))
              for s in SEEDS]
    art = {"models": models, "features": FEATURES, "ref_pred": float(np.median(predict_vol(models, X[ok]))), "seeds": list(SEEDS),
           "train_end": str(pd.Timestamp(ts[ok].max()).date()), "hold_bars": HOLD, "n_train": int(ok.sum()),
           "train_start": str(pd.Timestamp(ts[ok].min()).date()),
           "expand_logit": logit or (old or {}).get("expand_logit") or [-1.4950, 5.1671]}
    if te.sum() >= 2000:   # 지난 판 표본외 1주 이상 -- 첫 달(10-14 판 → 11-01)은 16일
        art["vs_prev"] = vs_prev(models, old["models"], X[te])
        print("지난 판 대비", art["vs_prev"], flush=True)
    tmp = ARTIFACT.with_suffix(".tmp")
    joblib.dump(art, tmp)   # 다 쓴 뒤 교체 -- 쓰다 죽어도 지난 판이 남게
    if ARTIFACT.exists():   # 10-19: 교체 시각 이름으로 전부 남긴다(옛 .prev 하나는 다음 교체에 덮였다)
        ARTIFACT.replace(ARTIFACT.with_suffix(f".{pd.Timestamp.utcnow():%Y%m%dT%H%M}.joblib"))
    tmp.replace(ARTIFACT)
    print("저장", {k: v for k, v in art.items() if k != "models"}, flush=True)
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

    # 2026-10-15 확대 확률 적합: 알려진 상수로 만든 라벨을 다시 맞혀야 한다
    rng2 = np.random.default_rng(1)
    rv = np.exp(rng2.normal(-6.5, 0.3, 40000)); xr = rng2.normal(0, 0.3, 40000)
    pr = rv * np.exp(xr); lab_p = 1 / (1 + np.exp(-(-1.5 + 5.0 * xr)))
    yy = rv * np.where(rng2.random(40000) < lab_p, 1.5, 1.0)          # 라벨 = y ≥ 1.3·rv48
    a_, b_ = fit_expand_logit(pr, rv, yy)
    assert abs(a_ + 1.5) < 0.1 and abs(b_ - 5.0) < 0.3, (a_, b_)

    # 2026-10-19 품질 문: 같은 모양 예측은 통과, 눈금이 2배 뛴 판은 거절
    class _Lin:
        def __init__(self, k, b=0.0): self.k, self.b = k, b
        def predict(self, A): return self.b + self.k * np.asarray(A, float)[:, 1]   # rv12 따라가는 가짜 판
    Xg = X.iloc[WARMUP:]
    assert vs_prev([_Lin(1.0)], [_Lin(1.0, 0.01)], Xg)["corr_log"] > 0.99
    try:
        vs_prev([_Lin(1.0, np.log(2.0))], [_Lin(1.0)], Xg); raise AssertionError("눈금 2배 판이 통과했다")
    except SystemExit:
        pass

    print("통과 11/11 — 피쳐 계약(22열) · 인과성 · 로그스케일 · 기하평균 · 가격기하 범위 · 확대 확률 적합 · 품질 문")


if __name__ == "__main__":
    if "--monthly" in sys.argv:
        raise SystemExit(monthly())
    if "--verify" in sys.argv:
        raise SystemExit(verify())
    _self_check()
