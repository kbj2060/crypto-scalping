"""ETH 4h 변동성 GBM «조기 종료 무작위 10% 검증» 문제 검정 (2026-10-19, 사용자 «변동성 모델도 같은 문제 있는지»). 로컬 CPU 만.

배경: live_eth_sizing_vol_model_20260912 는 HistGradientBoostingRegressor(max_iter=300, lr .06, max_depth 6, random_state=시드) 8개 로그평균,
  early_stopping 미지정 = 'auto' → 행 > 1만이면 무작위 10% 검증(5분 이웃 = 근접 중복, 라벨 48봉 겹침). 서버 배포 판 실측(10-19):
  8개 전부 n_iter_ 300 = 한 번도 안 멈춤 · 시드 쌍 로그 예측 상관 .997 · 시드 1개/평균 비 1~99분위 .95~1.05(최대 1.22) -- 4h 방향(.88)보다 훨씬 작다.

사전 등록(계산 전 고정):
  데이터 = 로컬 KL_CSV(binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv, 2021-12~2026-09-15) · 피쳐·라벨 = 라이브 build_features·y(다음 48봉 실현 σ).
  학습 ≤ 2025-08-31 · 표본 밖 = 2025-09-01~ (모듈 문서의 비교 구간과 같다).
  A  = 라이브 그대로(early_stopping auto, SEEDS 8개 로그평균) · B2 = early_stopping=False, 모델 1개(300회, 결정적).
  지표 = 로그 제곱 오차 MSLE((ln pred − ln y)²) 의 일 평균 · 보조 = 스피어만 · 산포(y/pred 를 중앙값으로 나눈 표준편차, 모듈 문서 지표).
  판정 = B2 − A 일 MSLE 차의 5일 블록 95% CI:
         상한 < 0 → «B2 가 낫다» · 상한 < A 평균 MSLE 의 +1% → «나쁘지 않음»(결정성 이득으로 교체 후보) · 그 밖 → 현행 유지.
  결정성 확인 = B2 를 시드 0·1 로 적합해 표본 밖 예측이 같은가.

결과(2026-10-19, tmp/vol_early_stop_20261019/report.json) = «현행 유지»:
  같은 문제는 있다: A 8개 전부 n_iter_ 300(조기 종료 한 번도 안 걸림) · 시드 쌍 로그 상관 .996.
  그러나 해롭지 않다: 표본 밖(2025-09-01~2026-09-15) MSLE A(8개 평균) .1155 · A 시드 1개 .1158~.1162 · B2 .1168
  → B2 − A +1.13% [+0.3%, +2.5%](일 블록) = 상한이 +1% 넘음 → 판정 «현행 유지». 스피어만 .783 vs .781 · 산포 .466 vs .468.
  해석: 무작위 10% 빼기 × 8시드 평균 = 사실상 배깅 -- 조기 종료 기능은 헛돌지만 결과엔 작은 이득(≈1%)이다.
  🔴B2 도 결정적이 아니다(시드 0·1 예측 다름): 학습 행 37만 > 20만 → sklearn 이 구간 나누기(binning)에 무작위 20만 행 표본을 쓴다.
    4h 방향(학습 5.2만 행)은 이 조건 밖이라 결정적(10-19 확인) -- 그쪽 결정성 문은 1.2만 행으로 검사해 20만 넘는 조건은 못 본다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_sizing_vol_model_20260912 as svm  # noqa: E402
from research_eth_dir4h_reverify_20261019 import block_ci  # noqa: E402

KL = Path("/home/kbj20/crypto-scalping/binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
OUT = ROOT / "tmp/vol_early_stop_20261019"
CUT, TE0 = "2025-08-31 23:55", "2025-09-01"
GBM = dict(max_iter=300, learning_rate=0.06, max_depth=6)   # = svm.monthly() 학습 설정


def data():
    d = pd.read_csv(KL, usecols=["timestamp", "close", "high", "low", "quote_volume", "trades"], parse_dates=["timestamp"])
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    X = svm.build_features(d.timestamp, d.close.to_numpy(float), d.quote_volume.to_numpy(float), d.trades.to_numpy(float),
                           d.high.to_numpy(float), d.low.to_numpy(float))
    lr = np.diff(np.log(np.maximum(d.close.to_numpy(float), 1e-12)), prepend=0.0)
    y = pd.Series(lr).rolling(svm.HOLD, min_periods=svm.HOLD).std().shift(-svm.HOLD).to_numpy()
    ok = np.isfinite(y) & (y > 0) & np.isfinite(X.to_numpy(float)).all(1)
    return d.timestamp.to_numpy(), np.nan_to_num(X.to_numpy(float)), y, ok


def fit(A, ly, es, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=int(seed), early_stopping=es, **GBM).fit(A, ly)


def metrics(lp, y, ts):
    from scipy.stats import spearmanr
    e = (lp - np.log(y)) ** 2
    z = y / np.exp(lp); z = z / np.median(z)
    return {"msle": round(float(e.mean()), 5), "spearman": round(float(spearmanr(lp, y).statistic), 4), "spread": round(float(z.std()), 4)}, \
        pd.Series(e, index=pd.to_datetime(ts)).groupby(lambda t: t.normalize()).mean()


def main() -> dict:
    ts, A, y, ok = data()
    tr, te = ok & (ts <= np.datetime64(CUT)), ok & (ts >= np.datetime64(TE0))
    # 라벨 48봉이 표본 밖으로 넘어가지 않게: 학습 마지막 48봉 정화
    tr &= ts <= np.datetime64(pd.Timestamp(CUT) - pd.Timedelta(minutes=5 * svm.HOLD))
    ly = np.log(y[tr])
    mA = [fit(A[tr], ly, "auto", s) for s in svm.SEEDS]
    mB, mB1 = fit(A[tr], ly, False, 0), fit(A[tr], ly, False, 1)
    LA = np.array([m.predict(A[te]) for m in mA]); lpA, lpB = LA.mean(0), mB.predict(A[te])
    out = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "test": [str(ts[te][0])[:10], str(ts[te][-1])[:10]],
           "n_iter_A": [m.n_iter_ for m in mA], "n_iter_B2": mB.n_iter_,
           "B2_deterministic": bool(np.array_equal(lpB, mB1.predict(A[te]))),
           "A_seed_pair_corr_min": round(float(np.corrcoef(LA)[np.triu_indices(len(mA), 1)].min()), 4)}
    (out["A"], dA), (out["B2"], dB) = metrics(lpA, y[te], ts[te]), metrics(lpB, y[te], ts[te])
    out["A_single_seed"] = [metrics(L, y[te], ts[te])[0]["msle"] for L in LA]
    dd = (dB - dA).to_numpy(); lo, hi = block_ci(dd)
    out["B2_minus_A_daily_msle"] = {"mean": round(float(dd.mean()), 6), "ci": [round(lo, 6), round(hi, 6)], "pct_of_A": round(100 * float(dd.mean()) / out["A"]["msle"], 2)}
    out["verdict"] = "B2 낫다" if hi < 0 else ("나쁘지 않음" if hi < 0.01 * out["A"]["msle"] else "현행 유지")
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True); json.dump(out, open(OUT / "report.json", "w"), ensure_ascii=False, indent=1)
    return out


def selfcheck() -> None:
    ts = pd.date_range("2025-01-01", periods=600, freq="5min").to_numpy()
    y = np.exp(np.random.default_rng(0).normal(-6, .3, 600))
    m, d = metrics(np.log(y), y, ts)
    assert m["msle"] == 0 and abs(m["spread"]) < 1e-12 and len(d) == 3          # 완벽 예측 = 오차 0
    m2, _ = metrics(np.log(y) + 0.1, y, ts)
    assert abs(m2["msle"] - 0.01) < 1e-9 and m2["spread"] < 1e-12                # 일정 배율 편향 = MSLE 만, 산포 0
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
