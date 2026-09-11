#!/usr/bin/env python3
"""V자 급등락 재학습: **체결 가능한 앵커**(발동봉 종가) vs 현행 극점 앵커.

사용자 지시(2026-09-11): *"고가에서 샀다고 가정하는 게 문제다 … 성적이 부풀려진 것 같다."*
선행 측정([[eth_v_rebound_extreme_anchor_inflates_skill_20260911]])에서 **발동봉 레인지÷ATR
단일피쳐만으로 VAL AUC 0.637** 이 나왔다(배포 23피쳐 모델 0.6942). 여기서는 **같은 피쳐·같은
모집단**으로 두 앵커를 나란히 학습해 **모델이 그 단일피쳐를 얼마나 넘는지**를 앵커별로 잰다.

## 설계
피쳐  기존 tier0 프레임을 **재사용**한다(재생성하지 않는다) -- 배포 모델과 같은 23개.
      `data/labels/eth_5m_v_rebound_multitrigger_feeder_optimized_20260901/…features_tier0.csv`
      ⚠️이건 **9트리거 후보풀** 모집단이다(배포는 every-bar). 절대 AUC 를 배포 0.6942 와
      직접 비교하지 않는다 -- 두 앵curity 팔의 **차이**와 단일피쳐 대비 **순증분**만 읽는다.
라벨  git 에서 복원한 `realized_outcome`(research_eth_v_rebound_sweep_gate_recall_check_90d_
      20260831.py)을 **그대로** 옮긴다. 앵커만 바꾼 두 판을 만든다:
        extreme : 발동봉 저가(하락쪽)/고가(상승쪽)  ← 현행 배포
        close   : 발동봉 종가                      ← 가장 이른 체결 가능 가격
      🔴먼저 **파리티 게이트**: 동결 컨텍스트(18,000행)의 label 을 extreme 판으로 재현한다.
        일치율 < 0.99 면 즉시 중단한다 -- 라벨이 다르면 그 위 숫자는 전부 무의미하다.
모델  HistGradientBoosting. 이 저장소가 검증한 TabPFN 프록시다(2026-09-01 confirm:
      GBM 0.6953 vs TabPFN 0.6942). 시드 5개, 창 경계는 저장소 표준 split.
기준선 **발동봉 레인지÷ATR 단일피쳐**를 항상 같이 낸다(사용자 지시 2번).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
FEATS_CSV = ROOT / ("data/labels/eth_5m_v_rebound_multitrigger_feeder_optimized_20260901/"
                    "eth_5m_v_rebound_multitrigger_feeder_optimized_features_tier0.csv")
FROZEN = ROOT / ("data/labels/eth_5m_v_rebound_every_bar_20260901/"
                 "tabpfn_train_context_frozen_every_bar_20260901.csv")
KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT = Path(__file__).resolve().parents[1] / "tmp/v_rebound_close_anchor_20260912"

FEATURES = ["is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864", "range_width_pct",
            "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z", "p_fast", "p_slow", "ret3_z",
            "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio", "upper_wick_ratio",
            "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile", "rsi"]
FAST_BARS, FULL_BARS, ATR_MULT, T_SUSTAIN = 6, 12, 1.5, 0.20        # realized_outcome 상수 그대로
SEEDS = [20260912, 517758, 30473, 771233, 154777]
WINDOWS = [("TRAIN", None, "2025-08-31"), ("VAL", "2025-09-01", "2025-12-31"),
           ("OOS", "2026-01-01", "2026-03-31"), ("FWD", "2026-04-01", None)]


def log(m):
    print(m, flush=True)


def load_klines() -> pd.DataFrame:
    d = pd.read_csv(KL, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    d = d.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if d["timestamp"].dt.tz is None:
        d["timestamp"] = d["timestamp"].dt.tz_localize("UTC")
    tr = np.maximum(d.high - d.low,
                    np.maximum((d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()))
    tr.iloc[0] = d.high.iloc[0] - d.low.iloc[0]
    d["atr14"] = tr.rolling(14, min_periods=14).mean().shift(1)      # **직전 봉** ATR
    return d


def build_labels(kl: pd.DataFrame) -> pd.DataFrame:
    """realized_outcome 을 봉 전체에 벡터화. 앵커 두 판을 같이 낸다.

    원본: fast = max/min(close[t+1..t+6]) 과 anchor 의 거리 / pre_atr
          peak = max/min(high/low[t+1..t+12]) · end = close[t+12]
          giveback = (peak-end)/(peak-anchor)  (하락쪽; 상승쪽은 부호 반전)
          라벨 = fast_mult>=1.5 AND giveback<=0.20
    """
    h, l, c = kl.high.to_numpy(), kl.low.to_numpy(), kl.close.to_numpy()
    atr = kl.atr14.to_numpy()
    n = len(kl)

    def fwd(x, k, how):
        r = pd.Series(x[::-1]).rolling(k, min_periods=k)
        v = (r.max() if how == "max" else r.min()).to_numpy()[::-1]
        out = np.full(n, np.nan)
        out[:n - k] = v[1:n - k + 1]
        return out

    fmax_c, fmin_c = fwd(c, FAST_BARS, "max"), fwd(c, FAST_BARS, "min")
    fmax_h, fmin_l = fwd(h, FULL_BARS, "max"), fwd(l, FULL_BARS, "min")
    end = np.full(n, np.nan)
    end[:n - FULL_BARS] = c[FULL_BARS:]

    out = {"timestamp": kl.timestamp}
    for anchor_name, a_dn, a_up in (("extreme", l, h), ("close", c, c)):
        for side, is_dn in (("down", True), ("up", False)):
            anc = a_dn if is_dn else a_up
            fast = (fmax_c - anc) if is_dn else (anc - fmin_c)
            peak = fmax_h if is_dn else fmin_l
            denom = (peak - anc) if is_dn else (anc - peak)
            with np.errstate(invalid="ignore", divide="ignore"):
                give = np.where(np.abs(denom) < 1e-12, np.nan,
                                ((peak - end) / denom) if is_dn else ((end - peak) / denom))
            mult = fast / atr
            y = ((mult >= ATR_MULT) & np.isfinite(give) & (give <= T_SUSTAIN)).astype(float)
            y[~(np.isfinite(mult) & np.isfinite(peak) & np.isfinite(end) & np.isfinite(atr) & (atr > 0))] = np.nan
            out[f"y_{anchor_name}_{side}"] = y
    out["rng_atr"] = (h - l) / atr                                   # 기준선 단일피쳐
    return pd.DataFrame(out)


def parity_gate(lab: pd.DataFrame) -> float:
    """🔴동결 컨텍스트의 label 을 extreme 판으로 재현한다. 안 맞으면 그 위 전부 무의미."""
    f = pd.read_csv(FROZEN, usecols=["timestamp", "label", "is_downside"], parse_dates=["timestamp"])
    if f["timestamp"].dt.tz is None:
        f["timestamp"] = f["timestamp"].dt.tz_localize("UTC")
    m = f.merge(lab[["timestamp", "y_extreme_down", "y_extreme_up"]], on="timestamp", how="inner")
    mine = np.where(m.is_downside.to_numpy() == 1, m.y_extreme_down, m.y_extreme_up)
    ok = np.isfinite(mine)
    agree = float((mine[ok] == m.label.to_numpy()[ok]).mean())
    log(f"[파리티] 동결 {len(f):,}행 중 조인 {len(m):,} · 유효 {int(ok.sum()):,} · **일치율 {agree:.6f}**")
    return agree


def auc(score, y):
    y = np.asarray(y, int)
    n1, n0 = int(y.sum()), len(y) - int(y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    OUT.mkdir(parents=True, exist_ok=True)
    kl = load_klines()
    log(f"[데이터] 5분봉 {len(kl):,} · {kl.timestamp.min():%Y-%m-%d} ~ {kl.timestamp.max():%Y-%m-%d}")
    lab = build_labels(kl)

    agree = parity_gate(lab)
    if agree < 0.99:
        log(f"🔴 파리티 실패({agree:.4f}) -- 라벨 재현이 다르다. 중단한다.")
        return 1
    log("✅ 파리티 통과 -- 아래 숫자는 배포 라벨과 같은 정의 위에 있다.\n")

    F = pd.read_csv(FEATS_CSV, parse_dates=["timestamp"])
    if F["timestamp"].dt.tz is None:
        F["timestamp"] = F["timestamp"].dt.tz_localize("UTC")
    D = F.merge(lab, on="timestamp", how="inner")
    side_dn = D.is_downside.to_numpy() == 1
    for a in ("extreme", "close"):
        D[f"y_{a}"] = np.where(side_dn, D[f"y_{a}_down"], D[f"y_{a}_up"])
    log(f"[모집단] 후보풀 {len(D):,}행 · {D.timestamp.min():%Y-%m-%d} ~ {D.timestamp.max():%Y-%m-%d}"
        f" · 하락쪽 {side_dn.mean():.1%}")

    ts = D.timestamp
    win = {}
    for nm, a, b in WINDOWS:
        m = np.ones(len(D), bool)
        if a: m &= (ts >= pd.Timestamp(a, tz="UTC")).to_numpy()
        if b: m &= (ts <= pd.Timestamp(b + " 23:59:59", tz="UTC")).to_numpy()
        win[nm] = m
    X = D[FEATURES].to_numpy(float)
    rep = {"population": "9트리거 후보풀(배포 every-bar 아님)", "n": int(len(D)), "parity": agree,
           "seeds": SEEDS, "arms": {}}

    for a in ("extreme", "close"):
        y = D[f"y_{a}"].to_numpy()
        ok = np.isfinite(y)
        tr = win["TRAIN"] & ok
        log(f"\n══ 앵커 {a.upper()} ══  TRAIN {int(tr.sum()):,}행 · 라벨률 {y[tr].mean():.1%}")
        preds = []
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, early_stopping=True,
                                               validation_fraction=0.15, random_state=sd)
            m.fit(X[tr], y[tr].astype(int))
            preds.append(m.predict_proba(X)[:, 1])
        P = np.mean(preds, axis=0)
        arm = {"train_rate": float(y[tr].mean())}
        log(f"  {'창':>6} {'n':>8} {'기저':>7} {'모델':>8} {'단일피쳐':>9} {'순증분':>8}")
        for nm, _, _ in WINDOWS:
            m = win[nm] & ok
            if nm == "TRAIN" or m.sum() < 200:
                continue
            am, ab = auc(P[m], y[m]), auc(D.rng_atr.to_numpy()[m], y[m])
            arm[nm] = {"n": int(m.sum()), "base": float(y[m].mean()), "model_auc": am,
                       "single_feature_auc": ab, "delta": am - ab}
            log(f"  {nm:>6} {m.sum():>8,} {y[m].mean():>6.1%} {am:>8.4f} {ab:>9.4f} {am-ab:>+8.4f}")
        rep["arms"][a] = arm
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    log(f"\n산출물 {OUT/'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
