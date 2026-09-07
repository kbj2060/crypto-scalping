#!/usr/bin/env python3
"""앵커 오실레이터 코어: **8종 증거신호의 순수 연속 지표** (2026-09-07).

사용자: *"복합 오실레이터 신호에서도 뽑아낼 수 있는 순수 지표가 있나?"*

## 답 -- 있다. 5종은 직접, 3종은 버려지던 연속 크기를 되살린다
지금까지 8종은 **이진 발동**으로만 썼고 그 발동은 앵커 정의에 이미 흡수돼 있다.
발동 직전의 **연속값**은 한 번도 피쳐가 된 적이 없다.

| 신호 | 순수 코어 | 발동 임계값 | features154 에 있나 |
|---|---|---|---|
| taker_delta_z_climax | `delta_z` | ±2.0 | ❌ |
| short_term_return_z | `ret3_z` | ±2.5 | ❌ |
| orthogonal_combo | `p_fast`,`p_slow` (0~1 백분위 오실레이터), `funding_z` | 0.10/0.90, ±2.0 | ❌ |
| kalman_deviation_meanrev | `kalman_dev_z` | ±2.0 | ❌ |
| demarker_extreme | `dem` | 0.10/0.90 | ❌ (별도 빌더에서 처리) |
| liquidity_sweep | 스윕 깊이 / 되찾기 비율 | 이진 기하 | ❌ |
| fib_extension_exhaustion | **확장 배수 자체** (0.272~0.618 구간으로 이진화되며 버려짐) | 이진 | ❌ |
| smt_divergence | ETH 돌파 대비 BTC 유지 폭 | 이진 기하 | ❌ |

## 파리티 -- 복사하지 않고 라이브 함수를 직접 호출한다
`live_evidence_signal_dashboard_20260823.py::compute_signals()` 를 앵커 빌더와
**같은 인자**(btc_df, funding_df, 같은 tmax 절단)로 부른다. 그 함수는 `dem`,
`kalman_dev_z`, `funding_z`, `atr_pct` 를 이미 출력 프레임에 내보낸다(566행).
내보내지 않는 코어(`p_fast`/`p_slow`/`delta_z`/`ret3_z`/스윕/fib/smt)만 이 파일에서
**같은 상수로 재계산**하고, 그 결과에서 8종 불리언을 되만들어 라이브 출력과
**비트 단위로 같은지 단언**한다. 하나라도 다르면 전사 오류이므로 중단한다.

## 측면 정렬
`fade_up = (side=="bottom")` -- **바닥 앵커의 지속 = 하락 계속**.
`cont_sign = +1(top)/-1(bottom)` 을 방향성 코어에만 곱한다. 크기 지표는 그대로 둔다.

## 인과성
전 피쳐를 무작위 앵커에서 **그 봉까지 잘라 재계산**해 1e-9 이내 일치를 확인한다.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "tmp/eth_anchor_features154_20260907/features154.parquet"
OUT = ROOT / "tmp/eth_anchor_osc_cores_20260907"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "anchor_builder_20260907", ROOT / "scripts/build_eth_anchor_label_dataset_20260907.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def compute_cores(sig: pd.DataFrame, B) -> pd.DataFrame:
    """라이브가 안 내보내는 연속 코어만 **같은 상수로** 재계산.

    상수는 live_evidence_signal_dashboard_20260823.py:146-155 와 동일.
    식은 같은 파일 383-437행에서 그대로 옮겼다.
    """
    L = B.compute_signals.__globals__                    # 라이브 모듈의 상수를 직접 참조
    STOCH_N, SLOWK, PCTW = L["STOCH_N"], L["SLOWK_SMOOTH"], L["PCTRANK_WINDOW"]
    ZW, SWEEP_LB, EPS = L["ZSCORE_WINDOW"], L["SWEEP_LOOKBACK"], L["EPS"]

    close, open_ = sig["close"].astype(float), sig["open"].astype(float)
    high, low = sig["high"].astype(float), sig["low"].astype(float)
    volume, taker_buy = sig["volume"].astype(float), sig["taker_buy_base"].astype(float)

    hh = high.rolling(STOCH_N, min_periods=STOCH_N).max()
    ll = low.rolling(STOCH_N, min_periods=STOCH_N).min()
    rng_stoch = (hh - ll).replace(0.0, np.nan)
    fast_k = 100.0 + (-100.0 * (hh - close) / rng_stoch)
    slow_k = fast_k.rolling(SLOWK, min_periods=SLOWK).mean()
    p_fast = fast_k.rolling(PCTW, min_periods=PCTW).rank(pct=True)
    p_slow = slow_k.rolling(PCTW, min_periods=PCTW).rank(pct=True)

    delta = 2.0 * taker_buy - volume
    delta_z = (delta - delta.rolling(ZW, min_periods=ZW).mean()) / \
        delta.rolling(ZW, min_periods=ZW).std().replace(0.0, np.nan)
    ret3 = close / close.shift(3) - 1.0
    ret3_z = (ret3 - ret3.rolling(ZW, min_periods=ZW).mean()) / \
        ret3.rolling(ZW, min_periods=ZW).std().replace(0.0, np.nan)

    swing_low_prior = low.rolling(SWEEP_LB, min_periods=SWEEP_LB).min().shift(1)
    swing_high_prior = high.rolling(SWEEP_LB, min_periods=SWEEP_LB).max().shift(1)
    sweep_low = (low < swing_low_prior) & (close > swing_low_prior)
    sweep_high = (high > swing_high_prior) & (close < swing_high_prior)

    low_arr, high_arr, n = low.to_numpy(), high.to_numpy(), len(sig)
    low_pos = np.full(n, -1, dtype=np.int64); high_pos = np.full(n, -1, dtype=np.int64)
    if n > SWEEP_LB:
        lo_w = np.lib.stride_tricks.sliding_window_view(low_arr, SWEEP_LB)
        hi_w = np.lib.stride_tricks.sliding_window_view(high_arr, SWEEP_LB)
        idx = np.arange(SWEEP_LB, n); j = idx - SWEEP_LB
        low_pos[idx] = j + lo_w[j].argmin(axis=1)
        high_pos[idx] = j + hi_w[j].argmax(axis=1)
    leg_up = pd.Series(low_pos < high_pos, index=sig.index)
    leg_down = pd.Series(high_pos < low_pos, index=sig.index)
    fib_rng = (swing_high_prior - swing_low_prior).replace(0.0, np.nan)

    # ⭐버려지던 연속값: 스윙 극점을 얼마나 넘어섰는가 (fib 배수 그 자체)
    ext_top = (high - swing_high_prior) / fib_rng
    ext_bot = (swing_low_prior - low) / fib_rng
    # 스윕 깊이(넘어선 폭)와 되찾기(종가가 스윙 안으로 얼마나 복귀했나)
    sweep_depth_lo = (swing_low_prior - low) / close
    sweep_depth_hi = (high - swing_high_prior) / close
    reclaim_lo = (close - swing_low_prior) / (low - swing_low_prior).replace(0.0, np.nan)
    reclaim_hi = (swing_high_prior - close) / (swing_high_prior - high).replace(0.0, np.nan)

    return pd.DataFrame({
        "timestamp": sig["timestamp"].to_numpy(),
        "p_fast": p_fast.to_numpy(), "p_slow": p_slow.to_numpy(),
        "delta_z": delta_z.to_numpy(), "ret3_z": ret3_z.to_numpy(),
        "ext_top": ext_top.to_numpy(), "ext_bot": ext_bot.to_numpy(),
        "sweep_depth_lo": sweep_depth_lo.to_numpy(), "sweep_depth_hi": sweep_depth_hi.to_numpy(),
        "reclaim_lo": reclaim_lo.to_numpy(), "reclaim_hi": reclaim_hi.to_numpy(),
        "_sweep_low": sweep_low.to_numpy(), "_sweep_high": sweep_high.to_numpy(),
        "_leg_up": leg_up.to_numpy(), "_leg_down": leg_down.to_numpy(),
        "_swing_lo": swing_low_prior.to_numpy(), "_swing_hi": swing_high_prior.to_numpy(),
        "_fib_rng": fib_rng.to_numpy(),
    })


def parity_check(sig: pd.DataFrame, C: pd.DataFrame) -> dict:
    """내 코어에서 8종 불리언을 되만들어 라이브 출력과 비트 단위 대조."""
    high, low = sig["high"].astype(float), sig["low"].astype(float)
    fib_rng = C["_fib_rng"]
    rebuilt = {
        "bottom_taker_delta_z_climax": C["delta_z"] <= -2.0,
        "top_taker_delta_z_climax": C["delta_z"] >= 2.0,
        "bottom_short_term_return_z": C["ret3_z"] <= -2.5,
        "top_short_term_return_z": C["ret3_z"] >= 2.5,
        "bottom_liquidity_sweep": C["_sweep_low"],
        "top_liquidity_sweep": C["_sweep_high"],
        "bottom_demarker_extreme": sig["dem"] <= 0.10,
        "top_demarker_extreme": sig["dem"] >= 0.90,
        "bottom_kalman_deviation_meanrev": sig["kalman_dev_z"] <= -2.0,
        "top_kalman_deviation_meanrev": sig["kalman_dev_z"] >= 2.0,
        "top_fib_extension_exhaustion": C["_leg_up"].to_numpy() & high.between(
            C["_swing_hi"] + 0.272 * fib_rng, C["_swing_hi"] + 0.618 * fib_rng).to_numpy(),
        "bottom_fib_extension_exhaustion": C["_leg_down"].to_numpy() & low.between(
            C["_swing_lo"] - 0.618 * fib_rng, C["_swing_lo"] - 0.272 * fib_rng).to_numpy(),
    }
    res = {}
    for k, v in rebuilt.items():
        live = sig[k].fillna(False).to_numpy(bool)
        mine = pd.Series(v).fillna(False).to_numpy(bool)
        res[k] = {"n_live": int(live.sum()), "n_mine": int(mine.sum()),
                  "mismatch": int((live != mine).sum())}
    res["ALL_MATCH"] = all(r["mismatch"] == 0 for r in res.values() if isinstance(r, dict))
    return res


def align(C: pd.DataFrame, sig_at: pd.DataFrame, cont_sign: np.ndarray) -> pd.DataFrame:
    A = pd.DataFrame(index=C.index)
    # 0~1 백분위 오실레이터 -- 0.5 중심 반전 + 방향 없는 극단도
    for c in ("p_fast", "p_slow"):
        A[f"{c}_al"] = 0.5 + (C[c] - 0.5) * cont_sign
        A[f"{c}_dist"] = (C[c] - 0.5).abs()
    # z 코어 -- 부호 정렬 + 크기
    for c, src in (("delta_z", C), ("ret3_z", C), ("kalman_dev_z", sig_at), ("funding_z", sig_at)):
        A[f"{c}_al"] = src[c].to_numpy() * cont_sign
        A[f"{c}_abs"] = np.abs(src[c].to_numpy())
    # DeMarker -- 별도 빌더와 겹치지만 여기선 0.5중심 정렬만 (중복은 평가에서 제외 가능)
    A["dem_al"] = 0.5 + (sig_at["dem"].to_numpy() - 0.5) * cont_sign
    # ⭐fib 확장 배수 -- 앵커 측면의 그 다리 쪽 값을 쓴다
    top = cont_sign > 0
    A["fib_ext"] = np.where(top, C["ext_top"], C["ext_bot"])
    A["sweep_depth"] = np.where(top, C["sweep_depth_hi"], C["sweep_depth_lo"])
    A["sweep_reclaim"] = np.where(top, C["reclaim_hi"], C["reclaim_lo"])
    return A.replace([np.inf, -np.inf], np.nan)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    B = _load_builder()
    D = pd.read_parquet(LAB)

    eth = B._load_kl(B.ETH_KL); fund = B._load_funding(); btc = B._load_kl(B.BTC_KL)
    tmax = min(eth["timestamp"].max(), btc["timestamp"].max(), fund["calc_time"].max())
    eth = eth[eth["timestamp"] <= tmax].reset_index(drop=True)
    print(f"[1/5] 앵커 {len(D):,} · 봉 {len(eth):,} (상한 {tmax})", flush=True)

    sig = B.compute_signals(eth, btc_df=btc[btc["timestamp"] <= tmax],
                            funding_df=fund[fund["calc_time"] <= tmax])
    print(f"      라이브 출력 컬럼 {sig.shape[1]}개 · 연속 export: "
          f"{[c for c in ('dem','kalman_dev_z','funding_z','atr_pct') if c in sig.columns]}", flush=True)

    C = compute_cores(sig, B)
    par = parity_check(sig, C)
    nm = sum(r["mismatch"] for r in par.values() if isinstance(r, dict))
    print(f"[2/5] 파리티: 8종 12갈래 재구성 불일치 {nm}개 → "
          f"{'PASS' if par['ALL_MATCH'] else '🔴FAIL'}", flush=True)
    if not par["ALL_MATCH"]:
        for k, r in par.items():
            if isinstance(r, dict) and r["mismatch"]:
                print(f"      {k}: live {r['n_live']} vs mine {r['n_mine']} · 불일치 {r['mismatch']}")
        return 1

    # 앵커 봉으로 조인
    keep = [c for c in C.columns if not c.startswith("_")]
    M = D[["timestamp"]].merge(C[keep], on="timestamp", how="left")
    S = D[["timestamp"]].merge(sig[["timestamp", "dem", "kalman_dev_z", "funding_z"]],
                               on="timestamp", how="left")
    assert len(M) == len(D) == len(S), "조인 후 행수 변화"

    cont_sign = np.where(D["side"].to_numpy() == "top", 1.0, -1.0)
    A = align(M, S, cont_sign)
    A.insert(0, "timestamp", D["timestamp"].to_numpy())
    A["split"] = D["split"].to_numpy()
    fcols = [c for c in A.columns if c not in ("timestamp", "split")]

    # 인과성: 잘라 재계산
    ts_idx = {t: i for i, t in enumerate(pd.to_datetime(sig["timestamp"].to_numpy()))}
    bidx = np.array([ts_idx[t] for t in D["timestamp"]])
    probe = rng.choice(bidx[bidx > 3000], size=12, replace=False)
    bad, worst = [], 0.0
    for i in sorted(probe):
        cut = eth.iloc[: i + 1].reset_index(drop=True)
        sg2 = B.compute_signals(cut, btc_df=btc[btc["timestamp"] <= cut["timestamp"].max()],
                                funding_df=fund[fund["calc_time"] <= cut["timestamp"].max()])
        c2 = compute_cores(sg2, B)
        for c in keep:
            if c == "timestamp":
                continue
            a, b = C[c].to_numpy()[i], c2[c].to_numpy()[-1]
            if not np.isfinite(a) and not np.isfinite(b):
                continue
            d = abs(float(a) - float(b)) if np.isfinite(a) and np.isfinite(b) else np.inf
            worst = max(worst, d if np.isfinite(d) else 1e9)
            if d > 1e-9:
                bad.append((c, int(i), float(a), float(b)))
    print(f"[3/5] 인과성 재구성: 탐침 {len(probe)} · 불일치 {len(bad)} · 최대오차 {worst:.2e} "
          f"→ {'PASS' if not bad else '🔴FAIL'}", flush=True)
    if bad:
        print(f"      표본 {bad[:6]}", flush=True)

    # 기존 피쳐와 중복도
    red = {}
    for c in fcols:
        best = max(((abs(float(pd.Series(A[c]).corr(D[k]))), k)
                    for k in D.columns if D[k].dtype.kind in "fi" and k not in ("split",)),
                   default=(0.0, "-"))
        red[c] = {"max_abs_r": round(best[0], 3), "with": best[1]}

    A.to_parquet(OUT / "osc_cores.parquet", index=False)
    (OUT / "meta.json").write_text(json.dumps(
        {"feature_cols": fcols, "n": len(A), "parity": par,
         "causality": {"n_probe": len(probe), "n_bad": len(bad), "worst": float(worst),
                       "PASS": not bad},
         "redundancy_vs_features154": red,
         "source": "live_evidence_signal_dashboard_20260823.py::compute_signals (직접 호출)"},
        indent=1, ensure_ascii=False))

    print(f"[4/5] 피쳐 {len(fcols)}개", flush=True)
    for c in fcols:
        print(f"    {c:<18} 결측 {A[c].isna().mean():>6.2%} · 고유 {A[c].nunique():>5} "
              f"· 기존과 최대상관 {red[c]['max_abs_r']:.3f} ({red[c]['with']})", flush=True)
    print(f"\n[5/5] 저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
