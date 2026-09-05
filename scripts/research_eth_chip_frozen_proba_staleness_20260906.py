#!/usr/bin/env python3
"""증거신호 칩 — **발동 봉 확률을 호라이즌 내내 재사용하는 게 맞나** (2026-09-06).

§21은 칩의 `_active`(칩 점등·votes·net_score) 쪽만 검정했다. 같은 칩의 **나머지 절반**인
`model_side`/`model_proba`/`model_tp_price`는 메타라벨 스코어러의 애프터글로우 캐시로 **잠겨 있다**:

    if cached and 0 <= (latest_ts - cached["bar_ts"])/300 < horizon_bars:
        out[name] = {"fired": True, "side": cached["side"], "proba": cached["proba"], ...}

즉 확률은 **발동 봉에서 한 번 계산되고 호라이즌 끝까지 그대로**다(smt는 최대 72봉 = 6시간).
라이브 실측(2026-09-05 16:00Z)에서 smt가 39봉 전 확률 0.78을 그대로 달고 있었다.
⇒ 질문: **그 확률이 a봉 뒤에도 유효한가.** 매 봉 재계산이 더 낫다면 GPU 비용을 감수할 가치가 있다.

## 검정 설계 — 나이 a에서의 **조건부** 질문으로 정렬한다
칩이 a봉째 켜져 있다는 건 "아직 목표에 안 닿았다"는 뜻이다. 그 시점에 사용자가 알고 싶은 건
**남은 (H−a)봉 안에 닿는가**이다. 그래서:

    모집단(나이 a)  발동 후 a봉이 지나도록 목표 미달성이고 a < H 인 발동 (= 칩이 아직 켜진 상태)
    라벨            (i+a, i+H] 안에 목표(K×ATR, 고가/저가) 도달 여부
    L  잠금(현행)   발동 봉 피쳐로 낸 확률을 그대로 사용
    U2 재계산       **같은 모델**에 나이 a 봉의 피쳐를 넣어 다시 계산
    U3 재계산+정합  나이 a 행으로 **따로 학습한 모델**(상한선 — 실제 서빙엔 모델 6개가 필요)
    U4 ⭐**나이 인지 단일 모델**  전 나이를 한 데 모아 `age`·`bars_left`를 피쳐로 넣고 **모델 하나**로 학습
        (U3의 이득을 서빙 가능한 형태로 가져올 수 있는지 — 실제 배포 후보는 이것뿐이다)
    a ∈ {0, 2, 4, 6, 9, 12}봉

피쳐 Tier0 22 + is_downside + 신호 원핫 8 · HistGradientBoosting 5시드 · TRAIN 적합, VAL/OOS 표본외 1회.
(§13에서 HGB가 칩 라벨 AUC 0.72로 배포 TabPFN(0.61~0.73)과 같은 수준임을 확인했다.)
판정: VAL·OOS 두 창 모두 U2 AUC > L AUC 이면 "잠금이 낡았다". HOLDOUT 로드 단계 차단.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


XA = _load("xa_st", "scripts/research_crossasset_fire_continuation_replication_20260906" if False else "scripts/research_crossasset_fire_continuation_replication_20260905.py")
FL = _load("fl_st", "scripts/research_eth_chip_side_flip_after_target_20260906.py")
V2 = _load("v2_st", "scripts/research_homer_entry_v2_20260904.py")
OUT = ROOT / "data/research/eth_chip_frozen_proba_staleness_20260906"
AGES = (0, 2, 4, 6, 9, 12)
SEEDS = [20260906, 771103, 480219, 913057, 264488]
FEATS = ["sweep_penetration_atr", "atr_percentile_864", "range_width_pct", "hour_utc", "weekday",
         "p_fast", "p_slow", "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio", "upper_wick_ratio",
         "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile", "ret3_z", "rsi", "delta_z",
         "flow_aligned_delta_z", "atr_pct"]
GAP, WINDOWS = 12, ("TRAIN", "VAL", "OOS")


def log(m): print(f"[stale] {m}", flush=True)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    FL.verify_constants()
    kl = XA.load_kl("ETHUSDT"); btc = XA.load_kl("BTCUSDT")           # HOLDOUT 차단
    sig = XA.DASH.compute_signals(kl.copy(), btc_df=btc, funding_df=None)
    n = len(kl)
    h, l, c = (kl[x].to_numpy(float) for x in ("high", "low", "close"))
    prev = np.r_[np.nan, c[:-1]]
    tr_ = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr_pct = (pd.Series(tr_).rolling(14, min_periods=14).mean().to_numpy()) / c
    ts_all = kl["timestamp"].to_numpy()

    # 봉별 Tier0 피쳐 (F0 프레임에서 타임스탬프로 병합 -- 피쳐는 봉 단위, is_bottom만 행 단위)
    D = pd.read_parquet(V2.OUT / "frame.parquet", columns=["pos", "timestamp"] + FEATS).drop_duplicates("pos")
    X_bar = kl[["timestamp"]].merge(D.drop(columns=["pos"]), on="timestamp", how="left")
    XB = X_bar[FEATS].to_numpy(float)
    log(f"봉 {n:,} · 피쳐 병합 유효 {np.isfinite(XB).all(1).mean():.3f}")

    rows = []
    for si, s in enumerate(XA.SIGNALS):
        H_, k_ = FL.HORIZON[s], FL.K[s]
        for sd, sdv in (("bottom", 1), ("top", 0)):
            ff = XA.first_fire_mask(sig[f"{sd}_{s}"].fillna(False).to_numpy(bool), GAP)
            for i in np.flatnonzero(ff):
                if not np.isfinite(atr_pct[i]) or i + H_ >= n:
                    continue
                lvl = c[i] * (1 + k_ * atr_pct[i]) if sd == "bottom" else c[i] * (1 - k_ * atr_pct[i])
                seg = (h[i + 1:i + H_ + 1] >= lvl) if sd == "bottom" else (l[i + 1:i + H_ + 1] <= lvl)
                first_touch = int(np.argmax(seg)) + 1 if seg.any() else None      # 발동 후 몇 봉째 도달
                rows.append({"sig_i": si, "signal": s, "i": int(i), "is_bottom": sdv, "H": H_,
                             "touch_off": first_touch if first_touch is not None else 10 ** 6})
    F = pd.DataFrame(rows)
    tsi = pd.DatetimeIndex(ts_all[F["i"].to_numpy()]); F["split"] = "NONE"
    for w, (a, b) in XA.SPLITS.items():
        F.loc[(tsi >= pd.Timestamp(a)) & (tsi < pd.Timestamp(b)), "split"] = w
    F = F.loc[F["split"] != "NONE"].reset_index(drop=True)
    log(f"첫발동 {len(F):,} · 전체 목표 도달률 {(F['touch_off'] < F['H'] + 1).mean():.3f}")

    def design(idx_bars, is_bottom, sig_i):
        oh = np.zeros((len(idx_bars), len(XA.SIGNALS)))
        oh[np.arange(len(idx_bars)), sig_i] = 1.0
        return np.hstack([XB[idx_bars], is_bottom.reshape(-1, 1).astype(float), oh])

    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "ages": list(AGES), "seeds": SEEDS,
           "holdout_excluded": True, "n_fires": int(len(F)), "by_age": {}}
    # 나이 0 모델 (현행 서빙과 같은 학습 형태)
    tr_mask = (F["split"] == "TRAIN").to_numpy()
    i_ = F["i"].to_numpy(); ib = F["is_bottom"].to_numpy(); sgi = F["sig_i"].to_numpy()
    y0 = (F["touch_off"].to_numpy() <= F["H"].to_numpy()).astype(int)
    X0 = design(i_, ib, sgi)
    ok0 = np.isfinite(X0).all(1)

    def fit_predict(Xtr, ytr, Xall):
        ps = []
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4,
                                               l2_regularization=1.0, random_state=sd)
            m.fit(Xtr, ytr); ps.append(m.predict_proba(Xall)[:, 1])
        return np.mean(ps, axis=0)

    p_frozen = np.full(len(F), np.nan)
    p_frozen[ok0] = fit_predict(X0[tr_mask & ok0], y0[tr_mask & ok0], X0[ok0])
    log(f"나이0 모델 자기라벨 AUC: " + str({w: round(float(roc_auc_score(y0[(F['split'] == w).to_numpy() & ok0], p_frozen[(F['split'] == w).to_numpy() & ok0])), 4) for w in WINDOWS}))

    for a in AGES:
        H_ = F["H"].to_numpy(); to = F["touch_off"].to_numpy()
        alive = (a < H_) & (to > a)                                   # 나이 a에 아직 칩이 켜진 발동
        lab = ((to > a) & (to <= H_)).astype(int)                     # 남은 창에서 도달하는가
        ia = i_ + a
        valid = alive & (ia < n)
        Xa = np.full((len(F), X0.shape[1]), np.nan)
        Xa[valid] = design(ia[valid], ib[valid], sgi[valid])
        oka = valid & np.isfinite(Xa).all(1) & ok0
        p_res = np.full(len(F), np.nan)
        p_res[oka] = fit_predict(X0[tr_mask & ok0], y0[tr_mask & ok0], Xa[oka])          # U2: 같은 모델, 나이 a 피쳐
        m_tr = tr_mask & oka
        p_m = np.full(len(F), np.nan)
        if m_tr.sum() > 400:
            p_m[oka] = fit_predict(Xa[m_tr], lab[m_tr], Xa[oka])                          # U3: 나이 a 정합 모델
        rec = {"n_alive_total": int(oka.sum()), "windows": {}}
        for w in WINDOWS:
            m = oka & (F["split"] == w).to_numpy()
            if m.sum() < 200 or len(np.unique(lab[m])) < 2:
                continue
            d = {"n": int(m.sum()), "base_rate": round(float(lab[m].mean()), 3),
                 "L_locked_auc": round(float(roc_auc_score(lab[m], p_frozen[m])), 4),
                 "U2_rescored_auc": round(float(roc_auc_score(lab[m], p_res[m])), 4)}
            if np.isfinite(p_m[m]).all():
                d["U3_matched_auc"] = round(float(roc_auc_score(lab[m], p_m[m])), 4)
            d["U2_minus_L"] = round(d["U2_rescored_auc"] - d["L_locked_auc"], 4)
            rec["windows"][w] = d
        rep["by_age"][f"a{a}"] = rec
        log(f"  a={a:>2}봉 살아있음 {int(oka.sum()):>6} · " + " | ".join(
            f"{w} n={rec['windows'][w]['n']:>5} 기저 {rec['windows'][w]['base_rate']:.3f} L {rec['windows'][w]['L_locked_auc']:.4f} U2 {rec['windows'][w]['U2_rescored_auc']:.4f} (Δ{rec['windows'][w]['U2_minus_L']:+.4f})"
            + (f" U3 {rec['windows'][w]['U3_matched_auc']:.4f}" if "U3_matched_auc" in rec["windows"][w] else "")
            for w in WINDOWS if w in rec["windows"]))
    # ---- U4: 나이 인지 **단일** 모델 (전 나이 풀링, age·bars_left 피쳐 추가)
    log("U4 나이 인지 단일 모델 …")
    Xs, ys, ws, ags = [], [], [], []
    for a in AGES:
        H_ = F["H"].to_numpy(); to = F["touch_off"].to_numpy(); ia = i_ + a
        alive = (a < H_) & (to > a) & (ia < n)
        lab = ((to > a) & (to <= H_)).astype(int)
        Xa = np.full((len(F), X0.shape[1]), np.nan); Xa[alive] = design(ia[alive], ib[alive], sgi[alive])
        m = alive & np.isfinite(Xa).all(1)
        extra = np.column_stack([np.full(m.sum(), float(a)), (H_[m] - a).astype(float)])
        Xs.append(np.hstack([Xa[m], extra])); ys.append(lab[m]); ws.append(F["split"].to_numpy()[m]); ags.append(np.full(m.sum(), a))
    XU = np.vstack(Xs); yU = np.concatenate(ys); wU = np.concatenate(ws); aU = np.concatenate(ags)
    pU = fit_predict(XU[wU == "TRAIN"], yU[wU == "TRAIN"], XU)
    for a in AGES:
        for w in WINDOWS:
            m = (aU == a) & (wU == w)
            if m.sum() < 200 or len(np.unique(yU[m])) < 2:
                continue
            d = rep["by_age"][f"a{a}"]["windows"].get(w)
            if d is None:
                continue
            d["U4_age_aware_auc"] = round(float(roc_auc_score(yU[m], pU[m])), 4)
            d["U4_minus_L"] = round(d["U4_age_aware_auc"] - d["L_locked_auc"], 4)
    for a in AGES:
        ws_ = rep["by_age"][f"a{a}"]["windows"]
        log(f"  a={a:>2} " + " | ".join(f"{w} L {ws_[w]['L_locked_auc']:.4f} → U4 {ws_[w].get('U4_age_aware_auc', float('nan')):.4f} (Δ{ws_[w].get('U4_minus_L', float('nan')):+.4f}) [U3 상한 {ws_[w].get('U3_matched_auc', float('nan')):.4f}]" for w in WINDOWS if w in ws_))
    P = [a for a in rep["by_age"] if all(rep["by_age"][a]["windows"].get(w, {}).get("U2_minus_L", -9) > 0 for w in ("VAL", "OOS"))]
    rep["verdict_U4"] = {"ages_where_age_aware_wins": [a for a in rep["by_age"]
                         if all(rep["by_age"][a]["windows"].get(w, {}).get("U4_minus_L", -9) > 0 for w in ("VAL", "OOS"))]}
    rep["verdict"] = {"rule": "VAL·OOS 두 창 모두 U2 AUC > L AUC 이면 잠금이 낡았다", "ages_where_rescoring_wins": P}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'} · 재계산이 이기는 나이 {P}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
