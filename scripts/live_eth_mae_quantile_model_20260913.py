"""**보유시간 조건부 MAE 분위 모델** — 크기를 «변동성 나눗셈»이 아니라 학습으로 정한다.

사용자: *"이 정도 변동성과 보유 시간 등의 상황이면 내 증거금 어느 정도를 써서 리스크
관리를 하겠다는 확실한 모델이 있어야해. 단순 atr 나눗셈이 아닌 학습 모델이면 더 좋아."*

## 왜 타깃이 MAE 인가
청산은 **종가가 아니라 보유 중 최대 역행폭(MAE)** 에서 일어난다. 기존 경로는 전방 변동성을
예측해 그 역수로 크기를 정했는데, 변동성과 MAE 사이에는 보유시간이 끼어 있다 --
2026-09-13 실측에서 보유상한만 4시간으로 걸어도 허용 레버리지가 3.7배 -> 12.6배가 됐다.
그래서 여기서는 **MAE 자체를, 보유시간을 입력으로 받아** 예측한다.

## 계약
`predict_mae(models, X)` 는 분위 q 별 MAE(%) 를 준다. 크기는 그 역수다:
    leverage = SAFETY / MAE_q(horizon, 시장상태)      · 명목 = 순자산 × leverage
q 는 «몇 %의 거래에서 청산을 허용하는가»다. q=0.99 면 건당 1% -- 55건이면 42% 라 너무 높다.
운영값은 `TARGET_Q` 를 보라(건당 0.1% 목표).

## 피쳐
`live_eth_sizing_vol_model_20260912.build_features` 를 **그대로** 쓴다(22열) + 보유시간
log(H) + 방향(롱/숏). 피쳐 빌더를 두 벌로 두면 조용히 어긋난다 -- 그 파일의 설계 이유와 같다.

## 검증 방식
표본외에서 **적중률(coverage)** 을 본다: 예측 분위를 실제가 넘는 비율이 1-q 에 맞는가.
맞지 않으면 그 모델은 «크기를 정하는 데» 쓸 수 없다. 상관계수는 보지 않는다 -- 여기서
중요한 건 순위가 아니라 **수준**이다.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_sizing_vol_model_20260912 as svm  # noqa: E402

ARTIFACT = ROOT / "data" / "live" / "eth_mae_quantile_model.joblib"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
HORIZONS_BARS = [12, 24, 48, 96, 288]        # 5분봉 -> 1h, 2h, 4h, 8h, 24h
QUANTILES = [0.5, 0.9, 0.99, 0.999]
# 🔴운영에는 **q=0.9 + 보정계수**를 쓴다. 외삽한 극단분위를 그대로 쓰면 안 된다 --
# 2026-09-13 표본외 검증에서 q=0.999 의 실제 초과율이 0.46%(목표 0.10%)로 **4.6배 과신**,
# q=0.99 도 1.70%(목표 1.00%)였다. 반면 q=0.9 는 9.92%(목표 10%)로 거의 완벽하다.
# 꼬리 표본이 적은 분위회귀의 전형적 실패라, 잘 맞는 분위 위에 «몇 배를 곱하면 목표
# 초과율이 되는가»를 학습구간에서 재서 쓴다(CAViaR 계열의 관행 -- 분위를 직접 추정하되
# 수준은 실측 적중률로 고정한다).
BASE_Q = 0.9
TARGET_EXCEED = 0.001                         # 건당 0.1% 청산 허용
# 🔴3분할이다. 보정계수를 학습구간에서 재면 그 자체가 표본내라 적중률이 낙관적으로 나온다.
#   학습(분위회귀 적합) -> 보정(계수 m) -> 검증(건드리지 않음)
TRAIN_END = "2025-06-30"
CALIB_END = "2025-12-31"
WARMUP = svm.WARMUP


def _load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES)
    ts = pd.to_datetime(df.timestamp, unit="ms", errors="coerce")
    if ts.isna().all():       # timestamp 가 epoch ms 가 아니라 문자열인 판본이 있다
        ts = pd.to_datetime(df.timestamp, errors="coerce")
    assert not ts.isna().all(), "timestamp 파싱 실패 -- 피쳐가 전부 NaN 이 되어 조용히 표본 0 이 된다"
    df["ts"] = ts
    return df


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """봉 × 보유시간 × 방향 을 쌓은 학습 패널. MAE 는 **그 방향의** 역행폭이다."""
    base = svm.build_features(df.ts, df.close.to_numpy(float), df.quote_volume.to_numpy(float),
                              df.trades.to_numpy(float), df.high.to_numpy(float),
                              df.low.to_numpy(float))
    c, hi, lo = df.close.to_numpy(float), df.high.to_numpy(float), df.low.to_numpy(float)
    rows = []
    for h in HORIZONS_BARS:
        fmax = pd.Series(hi).iloc[::-1].rolling(h, min_periods=h).max().iloc[::-1].shift(-1).values
        fmin = pd.Series(lo).iloc[::-1].rolling(h, min_periods=h).min().iloc[::-1].shift(-1).values
        for side, mae in ((1, (c - fmin) / c * 100), (-1, (fmax - c) / c * 100)):
            f = base.copy()
            f["log_h"] = np.log(h * 5.0)     # 분 단위 로그 -- 스케일이 √t 에 가깝다
            f["side"] = side
            f["_mae"] = np.maximum(mae, 0.0)
            f["_ts"] = df.ts.values
            rows.append(f)
    panel = pd.concat(rows, ignore_index=True)
    return panel.replace([np.inf, -np.inf], np.nan).dropna()


FEATURES = svm.FEATURES + ["log_h", "side"]


def predict_mae(models: dict, X: pd.DataFrame) -> dict[float, np.ndarray]:
    return {q: m.predict(X[FEATURES]) for q, m in models.items()}


def calibrate(models: dict, tr: pd.DataFrame, target: float = TARGET_EXCEED) -> float:
    """`MAE_q90 × m` 을 실제가 넘는 비율이 target 이 되는 m 을 학습구간에서 찾는다."""
    p = np.maximum(predict_mae(models, tr)[BASE_Q], 1e-9)
    ratio = tr._mae.to_numpy() / p
    return float(np.quantile(ratio, 1.0 - target))


def safe_mae(models: dict, X: pd.DataFrame, mult: float) -> np.ndarray:
    """운영용 «이만큼은 역행할 수 있다» 값(%)."""
    return np.maximum(predict_mae(models, X)[BASE_Q], 1e-6) * mult


def leverage_for(models: dict, X: pd.DataFrame, mult: float,
                 safety: float = 1.0, cap: float = 25.0) -> np.ndarray:
    """예측 MAE 의 역수. safety<1 이면 더 보수적. cap 은 거래소/상식 상한."""
    return np.minimum(cap, safety * 100.0 / safe_mae(models, X, mult))


def train() -> int:
    from sklearn.ensemble import HistGradientBoostingRegressor
    import joblib
    df = _load_klines()
    panel = build_panel(df)
    tr = panel[panel._ts <= TRAIN_END]
    ca = panel[(panel._ts > TRAIN_END) & (panel._ts <= CALIB_END)]
    te = panel[panel._ts > CALIB_END]
    print(f"패널 {len(panel):,} · 학습 {len(tr):,} · 보정 {len(ca):,} · 검증 {len(te):,}")
    print(f"  학습 {tr._ts.min().date()}~{tr._ts.max().date()} · 보정 {ca._ts.min().date()}"
          f"~{ca._ts.max().date()} · 검증 {te._ts.min().date()}~{te._ts.max().date()}")
    models = {}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                          learning_rate=0.06, max_depth=6,
                                          min_samples_leaf=200, random_state=20260913)
        m.fit(tr[FEATURES], tr._mae)
        models[q] = m
        print(f"  q={q} 적합 완료")
    mult = calibrate(models, ca)
    print(f"보정계수 m = {mult:.3f}  (**보정구간**에서 q90×m 초과율 = {100*TARGET_EXCEED:.2f}%)")
    joblib.dump({"models": models, "features": FEATURES, "mult": mult,
                 "base_q": BASE_Q, "target_exceed": TARGET_EXCEED,
                 "train_end": TRAIN_END, "calib_end": CALIB_END,
                 "horizons_bars": HORIZONS_BARS}, ARTIFACT)
    print(f"저장 {ARTIFACT}")
    return verify(models, te, mult)


def verify(models: dict, te: pd.DataFrame, mult: float) -> int:
    """**적중률**이 전부다. 예측 분위를 실제가 넘는 비율이 1-q 여야 크기에 쓸 수 있다."""
    print(f"\n표본외 적중률 (목표 = 1-q)")
    print(f"{'q':>7} {'목표초과율':>10} {'실제초과율':>10} {'예측중앙':>9} {'실제중앙':>9}")
    ok = True
    pm = predict_mae(models, te)
    for q in QUANTILES:
        exceed = float((te._mae.to_numpy() > pm[q]).mean())
        print(f"{q:>7} {100*(1-q):>9.2f}% {100*exceed:>9.2f}% "
              f"{np.median(pm[q]):>8.3f}% {te._mae.median():>8.3f}%")
        # 보수적이면(초과율이 목표보다 낮으면) 안전측. 목표의 2배를 넘으면 못 쓴다.
        if exceed > 2.0 * (1 - q) + 0.002:
            ok = False
            print(f"        🔴 q={q} 가 과신한다 -- 이 분위로 크기를 정하면 안 된다")
    # 🔴운영 규칙은 q90×m 이다. 여기 적중률이 목표에 맞아야 크기에 쓸 수 있다.
    sm = safe_mae(models, te, mult)
    ex_all = float((te._mae.to_numpy() > sm).mean())
    print(f"\n**운영 규칙** q{BASE_Q}×{mult:.3f} 표본외 초과율 {100*ex_all:.3f}% "
          f"(목표 {100*TARGET_EXCEED:.3f}%)")
    if ex_all > 3.0 * TARGET_EXCEED:
        ok = False
        print("        🔴 보정 후에도 과신한다 -- 크기에 쓰면 안 된다")
    print(f"\n보유시간별 운영값")
    print(f"{'보유':>6} {'안전MAE중앙':>11} {'레버리지':>8} {'초과율':>8}")
    for h in HORIZONS_BARS:
        s = te[np.isclose(te.log_h, np.log(h * 5.0))]
        if not len(s):
            continue
        v = safe_mae(models, s, mult)
        ex = float((s._mae.to_numpy() > v).mean())
        print(f"{h*5:>5}분 {np.median(v):>10.2f}% {100/np.median(v):>7.1f}배 {100*ex:>7.3f}%")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(train())
