#!/usr/bin/env python3
"""돌파/되돌림 섀도우 **모델 아티팩트 동결** (2026-09-08).

규칙 (2026-09-08 전 축 검증 종료 후 확정):
    모집단  앵커 `first_fire` -- 8종 증거신호 중 **어느 하나의 첫 발동**(신호별 GAP dedup).
            평가창 실측 25.2건/일.
    발현    앵커 다음 봉 시가를 기준가로, **15분(3봉) 안에 ±0.75×ATR** 최초 터치.
            그 분이 트리거이고 터치한 쪽이 **발현 방향**(예측 대상 아님, 관측값).
            실측 22.1건/일 (앵커의 88%).
    라벨    트리거 분부터 **1시간(12봉)** 안에 진입가 **±0.8×ATR** 중 먼저 닿는 쪽.
            발현 방향 = 돌파(1) · 반대 = 되돌림(0) · 시간청산이면 12봉 뒤 종가 부호.
    피쳐    69개 (아래 FEATURES 순서 고정). 모두 **트리거 봉 bt 의 직전 봉 bt-1** 기준.
    모델    HistGradientBoostingClassifier 5시드 평균.
            ⚠️모델 축은 종결됐다 -- TabPFN/TabICL/LightGBM/고전GBM 전부 T2 노이즈 안이었고
            HGB(20초)가 TabPFN(594초)과 같은 값을 냈다. 회귀는 분류에 2.6~5.1pp 뒤진다.

## 사전등록 기대치 (시드 20개 워크포워드, 단일시드 평균 -- 5시드 평균은 이보다 낫거나 같다)
전건(커버 100%, 22.1건/일)   VAL .5748  OOS .6046  HOLDOUT .5716
  셔플 귀무                       .5457      .5697          .5157
  초과                          +2.91pp    +3.49pp        +5.59pp
상위 절반(11건/일 부근)        VAL .6363  OOS .6693  HOLDOUT .6335   초과 +9.1/+10.2/+11.9pp
⭐**정확도는 셔플 귀무와 함께 읽는다.** 창마다 클래스 균형이 달라 원시 정확도끼리 비교하면 안 된다.
⭐시드 20개 전부 세 창 동시에 귀무 위(20/20). B=100 셔플에서 p=0.010.

사용:
    python scripts/build_eth_breakout_shadow_artifact_20260908.py
    python scripts/build_eth_breakout_shadow_artifact_20260908.py --verify
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

DS = ROOT / "tmp/eth_breakout_atr_state_20260908_s1/dataset_v2.parquet"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
OUT = ROOT / "data/live/breakout_reversal_shadow_artifact"
ANCHOR, T_MULT, H, W_TRIG = "first_fire", 0.75, 12, 3
# ⭐2026-09-08 개정: 배리어를 **ATR 상대**로 바꾼다(P = K_ATR × atr_at_anchor).
#   절대 0.25% 는 ATR 구간마다 질문의 난이도가 딴판이었다 --
#     저ATR: 배리어/ATR 2.33× · 시간청산 25.3% · 돌파율 0.515
#     고ATR: 배리어/ATR 0.46× · 시간청산  0.0% · 돌파율 **0.334**(심한 불균형)
#   고ATR "정확도 69.1%"의 대부분이 클래스 불균형이었다(셔플 귀무 66.1%, 초과 +3.0pp).
#   0.8×ATR 로 바꾸면 돌파율이 전 구간 0.46~0.47 로 균등해지고 시간청산이 9.5%→3.2% 로 준다.
K_ATR = 0.8
SEEDS = [106645, 305610, 517758, 761154, 961476]   # 시드검정 20개에서 뽑은 5개(무작위 추출본)
CHUNK = 4000
RULE_ID = "breakout_reversal_ff075_atr08_h1h_20260908"

# 사전등록 기대치 -- 시드 20개 단일시드 평균과 셔플 귀무 (research_eth_breakout_seed_robustness_20260908)
PREREG = {
    # 2026-09-08 시드 20개 워크포워드 실측 (research_eth_breakout_atr08_validate_20260908)
    "cov100": {"acc": {"VAL": 0.5643, "OOS": 0.5934, "HOLDOUT_SPENT": 0.5823},
               "null": {"VAL": 0.5371, "OOS": 0.5490, "HOLDOUT_SPENT": 0.5312},
               "per_day": 22.1},
    "cov50":  {"acc": {"VAL": 0.6136, "OOS": 0.6570, "HOLDOUT_SPENT": 0.6323},
               "null": {"VAL": 0.5399, "OOS": 0.5508, "HOLDOUT_SPENT": 0.5298},
               "per_day": 11.6},
}


def first_touch(hi1, lo1, start, up, dn, nmin):
    n = len(start)
    tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = start[a:b, None] + np.arange(nmin)[None, :]
        hu = hi1[idx] >= up[a:b, None]; hd = lo1[idx] <= dn[a:b, None]
        au = hu.any(axis=1); ad = hd.any(axis=1)
        tu[a:b] = np.where(au, hu.argmax(axis=1), -1); td[a:b] = np.where(ad, hd.argmax(axis=1), -1)
    return tu, td


def feature_list(d: pd.DataFrame) -> list[str]:
    """⚠️학습·추론이 **같은 순서**를 써야 한다. 이 함수가 유일한 진실이다."""
    return ([c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] +
            [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] +
            ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"])


def build_xy():
    d = pd.read_parquet(DS)
    d = d[(d.anchor == ANCHOR) & (d.T_mult == T_MULT)].sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    bi = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float)
    entry = O5[np.minimum(bi + 1, len(O5) - 1)] * (1 + sgn * T)
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
    okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5)) & (bt >= 1)
    P = d["atr_at_anchor"].to_numpy(float) * K_ATR          # 사건별 배리어
    tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0), entry * (1 + P), entry * (1 - P), H * 5)
    big = 1 << 30
    uo, do_ = tu >= 0, td >= 0
    au, ad = np.where(uo, tu, big), np.where(do_, td, big)
    cont = np.where(sgn > 0, uo & (au < ad), do_ & (ad < au))
    rev = np.where(sgn > 0, do_ & (ad < au), uo & (au < ad))
    x5 = np.minimum(bt + H, len(C5) - 1)
    clo = (C5[x5] - entry) / entry * 1e4 * sgn
    y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))
    return d, y, okm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="저장된 아티팩트만 검사")
    a = ap.parse_args()
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    if a.verify:
        meta = json.loads((OUT / "meta.json").read_text())
        mods = joblib.load(OUT / "model.joblib")
        print(f"규칙 {meta['rule_id']} · 피쳐 {len(meta['features'])} · 시드 {meta['seeds']}")
        print(f"학습행 {meta['n_train']:,} ({meta['train_span'][0]} ~ {meta['train_span'][1]})")
        print(f"확신 등급 임계: {meta['confidence_tiers']}")
        print(f"모델 {len(mods)}개 로드 OK")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    d, y, okm = build_xy()
    feats = feature_list(d)
    X = np.nan_to_num(d[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    fit = np.flatnonzero(okm)
    print(f"앵커 {ANCHOR} T={T_MULT} · 트리거 {len(fit):,}건 · 돌파율 {y[fit].mean():.4f}", flush=True)
    print(f"피쳐 {len(feats)}개 · 기간 {d['timestamp'].iloc[0]} ~ {d['timestamp'].iloc[-1]}", flush=True)

    models = []
    for sd in SEEDS:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=sd)
        m.fit(X[fit], y[fit]); models.append(m)
        print(f"   시드 {sd} 학습 완료", flush=True)
    joblib.dump(models, OUT / "model.joblib")

    # 확신 등급: 학습셋 자체 예측의 |p-0.5| 분위 (표시용 등급일 뿐 게이트가 아니다)
    pin = np.mean([m.predict_proba(X[fit])[:, 1] for m in models], axis=0)
    conf = np.abs(pin - 0.5)
    tiers = {"high": float(np.quantile(conf, 0.75)), "mid": float(np.quantile(conf, 0.50)),
             "low": float(np.quantile(conf, 0.25))}
    meta = {
        "rule_id": RULE_ID, "created_utc": datetime.now(timezone.utc).isoformat(),
        "anchor": ANCHOR, "t_mult": T_MULT, "horizon_bars": H,
        "barrier_mode": "atr_relative", "barrier_k_atr": K_ATR,
        "barrier_pct": None,          # 절대 배리어는 더 이상 쓰지 않는다(개정 전 0.25%)
        "watch_bars": W_TRIG, "seeds": SEEDS, "features": feats,
        "n_train": int(len(fit)), "breakout_rate": float(y[fit].mean()),
        "train_span": [str(d["timestamp"].iloc[0]), str(d["timestamp"].iloc[-1])],
        "confidence_tiers": tiers, "prereg": PREREG,
        "note": ("정확도는 셔플 귀무와 함께 읽는다. 커버리지 상한을 두지 않는다 -- "
                 "전 트리거에 판정을 내고 확신 등급만 표시한다."),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"\n저장: {OUT}")
    print(f"   확신 등급 임계 |p-0.5|: 강 ≥{tiers['high']:.4f} · 중 ≥{tiers['mid']:.4f} · 약 ≥{tiers['low']:.4f}")
    print(json.dumps({"done": True, "n": int(len(fit)), "features": len(feats)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
