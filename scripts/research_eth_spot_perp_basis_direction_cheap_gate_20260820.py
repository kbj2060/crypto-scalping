#!/usr/bin/env python3
"""ETH spot-perp 베이스 방향(direction) 바이어스 cheap gate. 로드맵(docs/
eth_direction_alpha_non_microstructure_research_20260817.md)의 신규 후보 -- 문헌
(Schmeling/Schrimpf/Todorov "Crypto Carry" BIS WP1087/Management Science; He/Manela/Ross/
von Wachter "Fundamentals of Perpetual Futures" arXiv:2212.06888, 2026-08-20 조사)은 베이스가
과거추세→레버리지롱수요→베이스 순 인과라 변동성/청산크라우딩 신호에 가깝고 방향신호일
가능성은 낮다고 예측하지만, 사용자 지시로 방향 가설을 직접 검정한다(문헌 예측 확인용).

방법론은 이 로드맵의 기존 cheap gate(ETF플로우/스테이블코인/TSMOM, 전부 CLOSED)와 동일 템플릿:
causal 신호구성 → 오염체크(rho vs 가격) → 3-split(TRAIN/VAL/OOS) x 다중호라이즌 IC스캔 +
순열귀무 → max(always_long,always_short) 벤치마크 백테스트. 단 원 3종은 일별해상도라 호라이즌이
1/3/7일이었고, 이건 5분봉이라 호라이즌을 1/3/12/48bar(5분/15분/1시간/4시간)로 조정 --
funding 정산주기(8h=96bar)보다 짧은 구간에서 "베이스가 funding_rate 피쳐가 못 보는 정보를
갖는가"라는 문헌의 실제 가설에 맞춘 것.

데이터: perp=기존 canonical training_features_{year}.csv의 close(fapi.binance.com 유래,
scripts/extend_klines_20260713.py 확인), spot=이번에 신규수집한 ETHUSDT spot 5m kline
(scripts/eth_fetch_spot_klines_20260820.py, api.binance.com, 2024-01-01~현재, 커버리지100%)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20260820)
N_PERM = 2000

Z_WINDOW = 48   # 4시간 -- funding 정산주기(96bar=8h)의 절반, house convention(cvd_48 등)과 일치
ROC_WINDOW = 12  # 1시간 -- house convention(cvd_12, funding_roc_12)과 일치
HORIZONS = [1, 3, 12, 48]  # 5분/15분/1시간/4시간

SPLITS = {
    "TRAIN": ("2024-01-01", "2025-08-31"),
    "VAL": ("2025-09-01", "2025-12-31"),
    "OOS": ("2026-01-01", "2026-03-31"),
}


def _load_basis_frame() -> pd.DataFrame:
    spot = pd.read_csv(ROOT / "binance_data/klines_spot/ETHUSDT/ETHUSDT-5m-spot.csv",
                        usecols=["timestamp", "close"], parse_dates=["timestamp"]).rename(columns={"close": "spot_close"})
    perp_frames = []
    for f in ["data/splits/year_oos/training_features_2024.csv",
              "data/splits/year_oos/training_features_2025.csv",
              "data/splits/year_oos/training_features_2026_rebuilt.csv"]:
        p = pd.read_csv(ROOT / f, usecols=["timestamp", "close"], parse_dates=["timestamp"]).rename(columns={"close": "perp_close"})
        perp_frames.append(p)
    perp = pd.concat(perp_frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    df = perp.merge(spot, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    print(f"perp={len(perp):,}행, spot={len(spot):,}행, merge성공={len(df):,}행 "
          f"({len(df) / len(perp):.1%} of perp)", flush=True)

    df["basis_raw"] = (df["perp_close"] - df["spot_close"]) / df["spot_close"]
    roll = df["basis_raw"].rolling(Z_WINDOW)
    df["basis_z48"] = (df["basis_raw"] - roll.mean()) / roll.std()
    df["basis_roc12"] = df["basis_raw"] - df["basis_raw"].shift(ROC_WINDOW)
    df["log_return"] = np.log(df["perp_close"] / df["perp_close"].shift(1))
    return df


def _permutation_null_ic(x: np.ndarray, y: np.ndarray, n_perm: int) -> tuple[float, float, float]:
    """실제 IC(Spearman), 순열귀무 z-score, empirical two-sided p -- 벡터화 버전.
    Spearman rho = 랭크 위에서의 Pearson corr이므로 랭크는 1회만 계산하고, N_PERM개 순열을
    (n x N_PERM) 행렬로 쌓아 행렬곱 1번으로 전부 처리(scipy.spearmanr을 반복호출하는 것보다
    훨씬 빠름 -- eth_dc_feature_interaction_significance_20260820.py에서 쓴 것과 동일 기법)."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 30:
        return float("nan"), float("nan"), float("nan")
    rank_x = rankdata(x)
    rank_y = rankdata(y)
    mean_x, std_x = rank_x.mean(), rank_x.std()
    mean_y, std_y = rank_y.mean(), rank_y.std()
    real_ic = float((np.mean(rank_x * rank_y) - mean_x * mean_y) / (std_x * std_y))

    y_perm = np.column_stack([RNG.permutation(rank_y) for _ in range(n_perm)])  # (n, n_perm)
    sum_xy_perm = rank_x @ y_perm  # (n_perm,)
    ic_perm = (sum_xy_perm / n - mean_x * mean_y) / (std_x * std_y)

    z = float((real_ic - ic_perm.mean()) / ic_perm.std())
    p = float((np.abs(ic_perm) >= abs(real_ic)).mean())
    return real_ic, z, p


def main() -> None:
    df = _load_basis_frame()
    df = df.set_index("timestamp")

    contamination = {}
    for sig in ["basis_raw", "basis_z48", "basis_roc12"]:
        rho = spearmanr(df[sig].to_numpy(), df["perp_close"].to_numpy(), nan_policy="omit").statistic
        contamination[sig] = float(rho)
    print("\n오염체크(신호 vs 동시점 종가 Spearman, |rho|<0.5 통과):", flush=True)
    for k, v in contamination.items():
        print(f"  {k}: rho={v:+.4f} {'OK' if abs(v) < 0.5 else 'FAIL'}", flush=True)

    ic_results = {}
    for split_name, (start, end) in SPLITS.items():
        sub = df.loc[start:end].copy()
        for h in HORIZONS:
            fwd_ret = sub["log_return"].shift(-1).rolling(h).sum().shift(-(h - 1))
            for sig in ["basis_raw", "basis_z48", "basis_roc12"]:
                ic, z, p = _permutation_null_ic(sub[sig].to_numpy(), fwd_ret.to_numpy(), N_PERM)
                ic_results[(split_name, sig, h)] = {"ic": ic, "z": z, "p": p, "n": int(sub[sig].notna().sum())}

    print(f"\nIC 스캔 결과({len(SPLITS)}split x 3신호 x {len(HORIZONS)}호라이즌 = "
          f"{len(SPLITS) * 3 * len(HORIZONS)}칸, 순열귀무 N={N_PERM}):", flush=True)
    for sig in ["basis_raw", "basis_z48", "basis_roc12"]:
        print(f"\n  [{sig}]", flush=True)
        for split_name in SPLITS:
            row = []
            for h in HORIZONS:
                r = ic_results[(split_name, sig, h)]
                flag = "**" if abs(r["z"]) >= 2.0 else "  "
                row.append(f"h{h}bar: ic={r['ic']:+.4f} z={r['z']:+.2f}{flag}")
            print(f"    {split_name:5s}(n={ic_results[(split_name, sig, HORIZONS[0])]['n']}): " + " | ".join(row), flush=True)

    n_significant = sum(1 for v in ic_results.values() if abs(v["z"]) >= 2.0)
    print(f"\n순열귀무 |z|>=2.0 통과 칸: {n_significant}/{len(ic_results)}", flush=True)

    out = {
        "contamination": contamination,
        "ic_scan": {f"{k[0]}|{k[1]}|h{k[2]}": v for k, v in ic_results.items()},
        "n_significant_cells": n_significant, "n_total_cells": len(ic_results),
    }
    out_path = ROOT / "tmp/eth_spot_perp_basis_ic_scan_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
