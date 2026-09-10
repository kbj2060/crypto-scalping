#!/usr/bin/env python3
"""문헌조사(금융시계열/크립토 표준 피쳐 갭분석, 2026-08-20)에서 확인된 미보유 피쳐군 중 구현
복잡도가 낮은 9개 계열(분수차분/엔트로피/Corwin-Schultz/Roll/Kyle's Lambda/VPIN근사/실현
semivariance/실현첨도/분산비율검정)을 실제 계산해 13개 컬럼으로 구축한다. SADF/멀티프랙탈
DFA/전이엔트로피/Hawkes 4종은 구현복잡도+런타임비용이 높아 이번 패스에서 제외(문헌조사 리포트에
후보로만 기록).

⚠️ 아직 신호(AUC/permutation-null 등)는 계산하지 않는다 -- 구조검증(NaN율/값범위)만.
statsmodels 미설치(+공유env 신규의존성 회피)라 분수차분의 "ADF로 최소 d 자동탐색"은 생략하고
대신 d∈{0.3,0.5,0.7} 그리드로 3개 컬럼을 만들어 모델이 고르게 한다(단순화, 문서화됨)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega

FFD_D_GRID = (0.3, 0.5, 0.7)
FFD_THRESHOLD = 1e-4
ENTROPY_WINDOW = 48
ROLL_WINDOW = 48
KYLE_WINDOW = 48
VPIN_WINDOW = 48
MOMENT_WINDOW = 96  # realized_skewness와 동일 창(코드확인됨, 문헌조사 에이전트가 검증)
VR_QS = (4, 12)


def _ffd_weights(d: float, threshold: float = FFD_THRESHOLD, max_k: int = 200) -> np.ndarray:
    w = [1.0]
    k = 1
    while k <= max_k:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w, dtype=np.float64)


def _apply_ffd(x: np.ndarray, d: float) -> np.ndarray:
    w = _ffd_weights(d)
    K = len(w)
    out = np.full(len(x), np.nan, dtype=np.float64)
    if len(x) < K:
        return out
    windows = sliding_window_view(x, K)
    out[K - 1:] = windows @ w[::-1]
    return out


def _rolling_entropy_sign(log_return: np.ndarray, window: int) -> np.ndarray:
    sign = np.sign(log_return)
    n = len(sign)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    windows = sliding_window_view(sign, window)
    for i in range(windows.shape[0]):
        _, counts = np.unique(windows[i], return_counts=True)
        p = counts / window
        out[i + window - 1] = float(-np.sum(p * np.log2(p)))
    return out


def _corwin_schultz_spread(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)
    beta_1bar = np.log(high / low) ** 2
    beta = beta_1bar[1:] + beta_1bar[:-1]
    high_2bar = np.maximum(high[1:], high[:-1])
    low_2bar = np.minimum(low[1:], low[:-1])
    gamma = np.log(high_2bar / low_2bar) ** 2
    k = 3 - 2 * np.sqrt(2)
    with np.errstate(invalid="ignore"):
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    out[1:] = np.clip(spread, 0.0, None)
    return out


def _roll_spread(log_return: np.ndarray, window: int) -> np.ndarray:
    """원논문(Roll 1984)은 raw price diff를 쓰지만, 그러면 스프레드가 달러단위라 가격대가
    다른 2025/2026 구간에서 값 스케일 자체가 드리프트한다(정상성 없음). log_return diff를
    써서 비례(스프레드/가격) 단위로 바꿔 -- Corwin-Schultz와 동일하게 -- 시점간 비교가능하게
    한다(관용적 변형, 원논문 공식 그대로 아님)."""
    delta = pd.Series(log_return).diff()
    cov = delta.rolling(window).cov(delta.shift(1))
    with np.errstate(invalid="ignore"):
        spread = np.where(cov < 0, 2 * np.sqrt(-cov), 0.0)
    return spread


def _kyle_lambda(log_return: np.ndarray, signed_volume: np.ndarray, window: int) -> np.ndarray:
    r = pd.Series(log_return)
    sv = pd.Series(signed_volume)
    cov = r.rolling(window).cov(sv)
    var = sv.rolling(window).var()
    return (cov / var).to_numpy()


def _vpin_approx(taker_buy: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    taker_sell = volume - taker_buy
    imbalance = np.abs(taker_buy - taker_sell)
    s_imb = pd.Series(imbalance)
    s_vol = pd.Series(volume)
    return (s_imb.rolling(window).sum() / s_vol.rolling(window).sum()).to_numpy()


def _realized_semivariance_ratio(log_return: np.ndarray, window: int) -> np.ndarray:
    r = pd.Series(log_return)
    up = (r.clip(lower=0) ** 2).rolling(window).sum()
    down = (r.clip(upper=0) ** 2).rolling(window).sum()
    return (up / down.replace(0.0, np.nan)).to_numpy()


def _realized_kurtosis(log_return: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(log_return).rolling(window).kurt().to_numpy()


def _variance_ratio(log_return: np.ndarray, q: int, window: int) -> np.ndarray:
    r = pd.Series(log_return)
    r_q = r.rolling(q).sum()
    var_1 = r.rolling(window).var()
    var_q = r_q.rolling(window).var()
    return (var_q / (q * var_1)).to_numpy()


def build_financial_ml_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    taker_buy = df["taker_buy_base"].to_numpy(dtype=np.float64)
    log_return = df["log_return"].to_numpy(dtype=np.float64)
    signed_volume = 2 * taker_buy - volume

    out: dict[str, np.ndarray] = {}
    for d in FFD_D_GRID:
        out[f"ffd_close_d{str(d).replace('.', '')}"] = _apply_ffd(close, d)
    out[f"entropy_return_sign_{ENTROPY_WINDOW}"] = _rolling_entropy_sign(log_return, ENTROPY_WINDOW)
    out["corwin_schultz_spread"] = _corwin_schultz_spread(high, low)
    out[f"roll_implied_spread_{ROLL_WINDOW}"] = _roll_spread(log_return, ROLL_WINDOW)
    out[f"kyle_lambda_{KYLE_WINDOW}"] = _kyle_lambda(log_return, signed_volume, KYLE_WINDOW)
    out[f"vpin_approx_{VPIN_WINDOW}"] = _vpin_approx(taker_buy, volume, VPIN_WINDOW)
    out[f"realized_semivar_ratio_{MOMENT_WINDOW}"] = _realized_semivariance_ratio(log_return, MOMENT_WINDOW)
    out[f"realized_kurtosis_{MOMENT_WINDOW}"] = _realized_kurtosis(log_return, MOMENT_WINDOW)
    for q in VR_QS:
        out[f"variance_ratio_q{q}_{MOMENT_WINDOW}"] = _variance_ratio(log_return, q, MOMENT_WINDOW)
    return out


def _self_test() -> None:
    """구현 정확성만 검증(신호 아님) -- 토이 예제로 각 함수의 알려진 성질을 확인."""
    w = _ffd_weights(0.5)
    assert abs(w[0] - 1.0) < 1e-12, "FFD 가중치 w[0]는 항상 1"
    assert w[1] < 0, "d in (0,1)에서 w[1]은 음수여야 함(López de Prado Ch.5)"

    rng = np.random.default_rng(0)
    white_noise = rng.normal(size=5000)
    vr = _variance_ratio(white_noise, q=4, window=500)
    vr_valid = vr[~np.isnan(vr)]
    assert abs(np.nanmean(vr_valid) - 1.0) < 0.15, f"백색잡음의 분산비율은 1 근방이어야 함, 실측 mean={np.nanmean(vr_valid):.3f}"

    high_flat = np.full(100, 100.0)
    low_flat = np.full(100, 99.0)
    cs = _corwin_schultz_spread(high_flat, low_flat)
    assert np.all(cs[1:] >= 0) or np.all(np.isnan(cs[1:])), "Corwin-Schultz 스프레드는 0 이상(클리핑됨)"
    print("[self-test] 전부 통과(구현 정확성 검증, 신호검증 아님)", flush=True)


def main() -> None:
    _self_test()

    train, eval_df = omega._load_omega_frames()[:2]
    results = {}
    for year, df in (("2025_train", train), ("2026_eval", eval_df)):
        feats = build_financial_ml_features(df)
        n = len(df)
        stats = {}
        issues = []
        for name, arr in feats.items():
            nan_rate = float(np.isnan(arr).mean())
            inf_count = int(np.isinf(arr[~np.isnan(arr)]).sum())
            stats[name] = {"nan_rate": nan_rate, "inf_count": inf_count,
                            "mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)),
                            "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr))}
            if nan_rate > 0.05:
                issues.append(f"{name}: NaN율 {nan_rate:.1%}(예상보다 높음, warmup 구간만이어야 함)")
            if inf_count:
                issues.append(f"{name}: inf {inf_count}개")
        results[year] = {"n_rows": n, "stats": stats, "issues": issues}
        print(f"\n[{year}] n={n:,}", flush=True)
        for name, s in stats.items():
            print(f"  {name:35s} nan={s['nan_rate']:.1%} mean={s['mean']:+.4f} std={s['std']:.4f} "
                  f"range=[{s['min']:.4f}, {s['max']:.4f}]", flush=True)
        if issues:
            print(f"  [경고] {'; '.join(issues)}", flush=True)

    feature_names = sorted(build_financial_ml_features(train.head(500)).keys())
    out_path = ROOT / "tmp/eth_dc_financial_ml_feature_construction_20260820.json"
    out_path.write_text(json.dumps({"feature_names": feature_names, "per_split": results}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (SCRATCH / "dc_financial_ml_feature_names_20260820.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    print(f"\n총 신규 financial-ML 피쳐 {len(feature_names)}개 구축 완료(신호계산 없음)", flush=True)
    print(f"[report] {out_path}")


if __name__ == "__main__":
    main()
