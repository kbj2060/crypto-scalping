#!/usr/bin/env python3
"""증거신호 칩 — **나이 인지 확률 모델** 학습·동결 (2026-09-06, 조치 B).

배경(부록 22): 칩 확률은 발동 봉에서 한 번 계산되고 호라이즌 내내 재사용된다(애프터글로우 캐시).
실측상 순위 AUC가 0.697/0.701(나이0) → 0.642/0.602(12봉)로 낡고, 표시 확률이 실제 조건부 도달률보다
9~12봉 뒤 **+0.12(약 1.5배)** 과대해진다. `age`·`bars_left`를 피쳐로 넣은 **단일 모델**이
6개 나이 × 두 창 전부에서 이겼다(+0.06~+0.16 AUC, 나이별 정합 모델 6개의 상한과 동급).

## 이 스크립트가 만드는 것
    질문   "발동 후 a봉이 지나도록 아직 목표 미달성일 때, 남은 (H−a)봉 안에 K×ATR에 닿는가"
           ⚠️현행 칩의 질문("발동 시점에 H봉 안에 닿는가")과 **다른 질문**이다 — 화면 의미가 바뀐다.
    모집단 8종 raw 첫발동(GAP=12) × 나이 0..H−1 중 **아직 미달성인 행**만
    피쳐   ⭐**라이브 `FEATURE_COLUMNS`(23) 그대로**(`build_indicator_frame`을 직접 import — 파리티 계약)
           + 신호 원핫 8 + `age` + `bars_left` = 33
    학습   HistGradientBoosting 5시드 평균 · TRAIN(<2025-09-01)만 · VAL/OOS는 표본외 1회
    산출   `data/models/eth_chip_age_aware_proba_20260906/{model.joblib, model_card.json}`

## 의도적으로 다른 점 (모델카드에 기록)
  · 신호별 추가 피쳐(demarker `dem`, kalman `kalman_dev_z`)와 orthogonal_combo의 20피쳐 부분집합을
    쓰지 않는다 — 8종 **공용 단일 모델**이고 신호 원핫이 그 자리를 대신한다.
  · 학습기가 TabPFN이 아니라 HGB다 — 매 봉 추론이 필요한데 공유 GPU 경합을 만들 수 없다.
    부록 22에서 HGB가 나이0 AUC 0.697/0.701로 배포 TabPFN(0.61~0.73)과 같은 수준임을 확인했다.
누수가드: VAL AUC ≥ 0.99면 중단. HOLDOUT(≥2026-04-01)은 로드 단계 차단.

Usage (서버에서): python scripts/train_eth_chip_age_aware_proba_20260906.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "data/models/eth_chip_age_aware_proba_20260906"
KL_ETH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
KL_BTC = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
SEEDS = [20260906, 771103, 480219, 913057, 264488]
GAP, WARMUP_START, HOLDOUT_START = 12, "2024-01-01", "2026-04-01"
SPLITS = {"TRAIN": ("2024-05-01", "2025-09-01"), "VAL": ("2025-09-01", "2026-01-01"), "OOS": ("2026-01-01", "2026-04-01")}
REPORT_AGES = (0, 2, 4, 6, 9, 12)
HP = {"max_iter": 400, "learning_rate": 0.06, "max_depth": 5, "l2_regularization": 1.0}


def log(m): print(f"[age-proba] {m}", flush=True)


def load_kl(p):
    d = pd.read_csv(p, usecols=["timestamp", "open", "high", "low", "close", "volume", "trades", "taker_buy_base"],
                    parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return d.loc[(d["timestamp"] >= pd.Timestamp(WARMUP_START)) & (d["timestamp"] < pd.Timestamp(HOLDOUT_START))].reset_index(drop=True)


def main() -> int:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from live_evidence_signal_dashboard_20260823 import compute_signals
    from live_evidence_signal_metalabel_20260829 import METALABEL_SIGNALS
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import FEATURE_COLUMNS, build_indicator_frame

    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    SIGNALS = list(METALABEL_SIGNALS)
    H = {s: METALABEL_SIGNALS[s]["horizon_bars"] for s in SIGNALS}
    K = {s: METALABEL_SIGNALS[s]["k"] for s in SIGNALS}
    log(f"신호 {len(SIGNALS)} · H {H} · K {K}")
    kl, btc = load_kl(KL_ETH), load_kl(KL_BTC)
    sig = compute_signals(kl.copy(), btc_df=btc)
    frame = build_indicator_frame(kl.copy())
    assert len(frame) == len(kl), f"프레임 길이 불일치 {len(frame)} vs {len(kl)}"
    base_cols = [c for c in FEATURE_COLUMNS if c != "is_bottom"]          # is_bottom은 행 단위라 따로 붙인다
    assert all(c in frame.columns for c in base_cols), sorted(set(base_cols) - set(frame.columns))
    XB = frame[base_cols].to_numpy(float)
    n = len(kl)
    h, l, c = (kl[x].to_numpy(float) for x in ("high", "low", "close"))
    prev = np.r_[np.nan, c[:-1]]
    trr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr_pct = (pd.Series(trr).rolling(14, min_periods=14).mean().to_numpy()) / c
    ts = kl["timestamp"].to_numpy()
    log(f"봉 {n:,} · 피쳐 {len(base_cols)}(+is_bottom) · 유효 {np.isfinite(XB).all(1).mean():.3f} ({time.time()-t0:.0f}s)")

    def first_fire(v, gap):
        keep = np.zeros(len(v), bool); last = -10 ** 9
        for j in np.flatnonzero(v):
            if j - last > gap:
                keep[j] = True
            last = j
        return keep

    # ── 행 생성: (발동, 나이) × 미달성
    rows_i, rows_a, rows_left, rows_ib, rows_si, rows_y, rows_ts = [], [], [], [], [], [], []
    n_fire = 0
    for si, s in enumerate(SIGNALS):
        H_, k_ = H[s], K[s]
        for side, ib in (("bottom", 1), ("top", 0)):
            col = f"{side}_{s}"
            if col not in sig.columns:
                continue
            ff = first_fire(sig[col].fillna(False).to_numpy(bool), GAP)
            for i in np.flatnonzero(ff):
                if not np.isfinite(atr_pct[i]) or i + H_ >= n:
                    continue
                n_fire += 1
                lvl = c[i] * (1 + k_ * atr_pct[i]) if side == "bottom" else c[i] * (1 - k_ * atr_pct[i])
                hit = (h[i + 1:i + H_ + 1] >= lvl) if side == "bottom" else (l[i + 1:i + H_ + 1] <= lvl)
                touch_off = int(np.argmax(hit)) + 1 if hit.any() else 10 ** 6
                for a in range(H_):
                    if touch_off <= a:                                     # 이미 확정 -- 칩이 꺼진 상태
                        break
                    rows_i.append(i + a); rows_a.append(a); rows_left.append(H_ - a)
                    rows_ib.append(ib); rows_si.append(si)
                    rows_y.append(1 if touch_off <= H_ else 0); rows_ts.append(ts[i])
    R = pd.DataFrame({"bar": rows_i, "age": rows_a, "left": rows_left, "is_bottom": rows_ib,
                      "si": rows_si, "y": rows_y, "fire_ts": rows_ts})
    R["split"] = "NONE"
    tsi = pd.DatetimeIndex(R["fire_ts"])
    for w, (a, b) in SPLITS.items():
        R.loc[(tsi >= pd.Timestamp(a)) & (tsi < pd.Timestamp(b)), "split"] = w
    R = R.loc[(R["split"] != "NONE") & (R["bar"] < n)].reset_index(drop=True)
    oh = np.zeros((len(R), len(SIGNALS))); oh[np.arange(len(R)), R["si"].to_numpy()] = 1.0
    X = np.hstack([XB[R["bar"].to_numpy()], R["is_bottom"].to_numpy().reshape(-1, 1).astype(float), oh,
                   R[["age", "left"]].to_numpy(float)])
    y = R["y"].to_numpy()
    ok = np.isfinite(X).all(1)
    log(f"첫발동 {n_fire:,} → 행 {len(R):,} (유효 {ok.mean():.3f}) · 창별 {R['split'].value_counts().to_dict()} · 양성률 {y.mean():.3f}")

    tr = (R["split"] == "TRAIN").to_numpy() & ok
    models, ps = [], []
    for sd in SEEDS:
        m = HistGradientBoostingClassifier(random_state=sd, **HP)
        m.fit(X[tr], y[tr]); models.append(m); ps.append(m.predict_proba(X)[:, 1])
    p = np.mean(ps, axis=0)
    auc = {w: round(float(roc_auc_score(y[(R["split"] == w).to_numpy() & ok], p[(R["split"] == w).to_numpy() & ok])), 4) for w in SPLITS}
    log(f"전체 AUC {auc}")
    if auc["VAL"] >= 0.99:
        log("⛔ 누수 가드 발동 (VAL AUC ≥ 0.99)"); return 1

    by_age = {}
    for a in REPORT_AGES:
        d = {}
        for w in SPLITS:
            m = ok & (R["split"] == w).to_numpy() & (R["age"].to_numpy() == a)
            if m.sum() < 200 or len(np.unique(y[m])) < 2:
                continue
            d[w] = {"n": int(m.sum()), "base_rate": round(float(y[m].mean()), 3),
                    "auc": round(float(roc_auc_score(y[m], p[m])), 4),
                    "mean_pred": round(float(p[m].mean()), 3),
                    "calib_gap": round(float(p[m].mean() - y[m].mean()), 3)}
        by_age[f"a{a}"] = d
        log(f"  나이 {a:>2}: " + " | ".join(f"{w} n={d[w]['n']:>6} AUC {d[w]['auc']:.4f} 기저 {d[w]['base_rate']:.3f} 예측평균 {d[w]['mean_pred']:.3f} (격차 {d[w]['calib_gap']:+.3f})" for w in SPLITS if w in d))

    joblib.dump({"models": models, "feature_order": base_cols + ["is_bottom"] + [f"sig_{s}" for s in SIGNALS] + ["age", "bars_left"],
                 "signals": SIGNALS, "horizon": H, "k": K}, OUT / "model.joblib")
    card = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "purpose": "칩 확률의 나이 인지 재계산 (조치 B, 부록 22)",
            "question": "발동 후 a봉 미달성일 때 남은 (H-a)봉 안 K*ATR 도달 확률",
            "differs_from_current_chip": "현행 칩은 '발동 시점에 H봉 안 도달'을 묻고 값을 호라이즌 내내 고정한다",
            "learner": "HistGradientBoostingClassifier", "hyperparams": HP, "seeds": SEEDS,
            "features": {"live_FEATURE_COLUMNS": FEATURE_COLUMNS, "extra": ["signal one-hot x8", "age", "bars_left"],
                         "n_total": int(X.shape[1]),
                         "deliberate_omissions": ["demarker dem", "kalman kalman_dev_z", "orthogonal_combo 20-feature subset"]},
            "data": {"klines_range": [str(kl['timestamp'].iloc[0]), str(kl['timestamp'].iloc[-1])],
                     "holdout_blocked_from": HOLDOUT_START, "splits": {k: list(v) for k, v in SPLITS.items()},
                     "n_first_fires": int(n_fire), "n_rows": int(len(R)), "positive_rate": round(float(y.mean()), 4)},
            "auc_overall": auc, "by_age": by_age,
            "reference_frozen_baseline_appendix22": {"a0": {"VAL": 0.6968, "OOS": 0.7011}, "a12": {"VAL": 0.6415, "OOS": 0.6020}},
            "leakage_guard": "VAL AUC >= 0.99 -> abort (not triggered)"}
    (OUT / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1))
    log(f"완료 {time.time()-t0:.0f}s → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
