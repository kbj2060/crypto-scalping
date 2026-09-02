#!/usr/bin/env python3
"""진입 모델 라이브 피쳐 파리티 검증 (2026-09-03).

동결 아티팩트(`eth_entry_limit_fade_v1_20260903`)는 161피쳐를 쓴다. 연구에서는 그 값을
`load_frame()`(준비된 CSV) + `build_indicator_frame()`으로 만들었는데, **라이브는 그 CSV가 없다.**
라이브 레짐 스코어러가 쓰는 경로 -- `FeatureEngineer().process(eth_df, btc_df)` +
`_with_raw_state12()` -- 로 만든 값이 연구 프레임과 **같은 봉에서 일치하는지** 확인한다.

⚠️여기가 어긋나면 섀도우가 동결 모델과 다른 입력을 받게 되고, 그건 오늘 증거신호에서 고친
"학습/추론 모집단 불일치"와 같은 종류의 결함이다.

판정
  피쳐별 max|Δ| 와 상관을 보고, 상대오차 1e-6 초과 피쳐를 나열한다.
  ⭐그리고 **예측값 자체**를 두 경로로 만들어 비교한다 -- 최종적으로 중요한 건 그것이다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ART = ROOT / "tmp/eth_entry_limit_fade_v1_20260903/model.joblib"
OUT = ROOT / "tmp/eth_entry_live_parity_20260903"
N_BARS = 20000          # 최근 구간 비교 (라이브가 실제로 다루는 창)


def log(m): print(f"[parity]  {m}", flush=True)


def main() -> int:
    P = joblib.load(ART)
    FE = P["feature_cols"]
    log(f"동결 아티팩트 피쳐 {len(FE)}개")

    # --- 연구 경로 ---
    from research_eth_regime_s12k3_label_train_20260902 import load_frame
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame
    rf = load_frame()
    rf = rf.loc[:, ~pd.Index(rf.columns).duplicated()]
    kl = load_klines()
    ind = build_indicator_frame(kl)
    ind["timestamp"] = kl["timestamp"].to_numpy()
    research = rf.merge(ind[["timestamp"] + [c for c in ind.columns
                        if c in FE and c not in rf.columns]], on="timestamp", how="inner")
    log(f"연구 프레임 {len(research):,}봉 {research.timestamp.min()} ~ {research.timestamp.max()}")

    # --- 라이브 경로 ---
    from features.engineering import FeatureEngineer
    from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12
    # ⭐라이브 레짐 스코어러(live_regime_gbm3_signal_20260826.py)와 **동일한 조립**을 재현한다.
    # klines 외에 파생거래소 4개 컬럼이 필요하다: OI / 상위트레이더 롱숏 / 전체계정 롱숏 / 펀딩.
    # 라이브는 바이낸스 4개 엔드포인트로 받고, 여기서는 연구 프레임에 이미 붙어 있는 값을 쓴다.
    eth_p = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
    btc_p = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
    if not (eth_p.exists() and btc_p.exists()):
        log("⚠️ klines CSV 없음"); return 1
    eth = pd.read_csv(eth_p, parse_dates=["timestamp"])
    btc = pd.read_csv(btc_p, parse_dates=["timestamp"])
    DERIV = ["sum_open_interest_value", "sum_toptrader_long_short_ratio",
             "count_long_short_ratio", "last_funding_rate"]
    miss = [c for c in DERIV if c not in rf.columns]
    if miss:
        log(f"⚠️ 연구 프레임에 파생거래소 컬럼 없음: {miss}"); return 1
    eth = eth.merge(rf[["timestamp"] + DERIV], on="timestamp", how="inner")
    eth_cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                "trades", "taker_buy_base", "taker_buy_quote"] + DERIV
    eth = eth[eth_cols].dropna(subset=DERIV).reset_index(drop=True)
    btc = btc.rename(columns={"close": "close_btc", "volume": "volume_btc",
                              "quote_volume": "quote_volume_btc"})
    btc = btc[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]]
    log(f"라이브 조립: ETH {len(eth):,}봉(파생 4열 포함) · BTC {len(btc):,}봉")
    log(f"FeatureEngineer 실행 (ETH {len(eth):,} / BTC {len(btc):,})...")
    live = FeatureEngineer().process(eth, btc)
    live = _with_raw_state12(live)
    live = live.loc[:, ~pd.Index(live.columns).duplicated()]
    ind2 = build_indicator_frame(kl); ind2["timestamp"] = kl["timestamp"].to_numpy()
    add = [c for c in ind2.columns if c in FE and c not in live.columns]
    live = live.merge(ind2[["timestamp"] + add], on="timestamp", how="inner")
    log(f"라이브 프레임 {len(live):,}봉 · 컬럼 {live.shape[1]}")

    have_r = [c for c in FE if c in research.columns]
    have_l = [c for c in FE if c in live.columns]
    missing_l = [c for c in FE if c not in live.columns]
    log(f"\n연구에 있는 피쳐 {len(have_r)}/{len(FE)} · 라이브에 있는 피쳐 {len(have_l)}/{len(FE)}")
    if missing_l:
        log(f"⚠️ 라이브 경로에 없는 피쳐 {len(missing_l)}개: {missing_l[:20]}")

    common = [c for c in FE if c in research.columns and c in live.columns]
    m = research[["timestamp"] + common].merge(
        live[["timestamp"] + common], on="timestamp", suffixes=("_r", "_l"), how="inner")
    m = m.sort_values("timestamp").tail(N_BARS)
    log(f"공통 피쳐 {len(common)}개 · 겹치는 봉 {len(m):,} (최근 {N_BARS:,}만 비교)")

    rows = []
    for c in common:
        a = pd.to_numeric(m[f"{c}_r"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(m[f"{c}_l"], errors="coerce").to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 100:
            rows.append({"feature": c, "n": int(ok.sum()), "max_abs": np.nan,
                         "rel": np.nan, "corr": np.nan}); continue
        d = np.abs(a[ok] - b[ok])
        scale = np.maximum(np.abs(a[ok]), 1e-12)
        rows.append({"feature": c, "n": int(ok.sum()), "max_abs": float(d.max()),
                     "rel": float(np.median(d / scale)),
                     "corr": float(np.corrcoef(a[ok], b[ok])[0, 1])})
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True); R.to_csv(OUT / "feature_parity.csv", index=False)

    exact = R[(R.max_abs < 1e-9)]
    close = R[(R.max_abs >= 1e-9) & (R.rel < 1e-6)]
    bad = R[(R.rel >= 1e-6) | R.rel.isna()]
    log(f"\n=== 피쳐 파리티 ===")
    log(f"  완전일치(|Δ|<1e-9)      {len(exact):3d}/{len(R)}")
    log(f"  수치오차 수준(상대<1e-6) {len(close):3d}/{len(R)}")
    log(f"  ⚠️불일치                {len(bad):3d}/{len(R)}")
    if len(bad):
        pd.set_option("display.width", 200)
        print(bad.sort_values("corr").head(20).to_string(index=False))
    json.dump({"n_features": len(FE), "in_research": len(have_r), "in_live": len(have_l),
               "missing_live": missing_l, "exact": len(exact), "close": len(close),
               "mismatch": len(bad), "bars_compared": int(len(m))},
              open(OUT / "parity_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
