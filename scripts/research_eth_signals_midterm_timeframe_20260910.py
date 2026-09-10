#!/usr/bin/env python3
"""증거신호·이벤트 트리거를 **중기 타임프레임으로 재계산**했을 때의 성능 비교 (2026-09-10).

사용자: *"모든 증거신호, 이벤트트리거를 중기로 바꿨을 때 성능 비교해줘."*

같은 코드(`compute_signals`)를 5m/15m/1h/4h 봉에서 각각 돌린다 -- 임계값·창이 **봉 수** 기반이라
타임프레임을 바꾸면 같은 모양의 신호가 더 긴 시계로 옮겨간다. 그 8종×양측을 세 축으로 비교한다.

축 1  **같은 봉 수 지평**(H=12/48/144봉): 모양은 그대로, 시계만 길어진다(5m H12=1시간 ↔ 4h H12=2일).
축 2  **같은 시계 지평**(12h/1d/3d/7d): 질문을 고정하고 신호 해상도만 바꾼다.
축 3  **이벤트 트리거**: 극점 탐지기(각 TF 에서 같은 라벨·같은 피쳐로 재학습) · 돌파/되돌림(±0.8ATR
      선착 배리어를 각 TF 에서 재정의)의 표본외 성능.

측도  발동당 초과 bp = side_sign · (close[i+H]−close[i])/close[i], **귀무는 순환이동**(발동 간격·군집·
      측면 보존, 가격 정렬만 파괴 -- 무작위 추출 귀무는 CI 가 가짜로 좁아진다). 비용 10bp(테이커 왕복).
      측면별로 내고 **양측 평균을 헤드라인**으로 쓴다(거울상이면 잔존 베타).
경계  피쳐/발동은 봉 i 종가까지, 수익은 i+1 부터. 형성 중인 봉 없음.
출력  tmp/eth_signal_midterm_20260910/report.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_exit_synth_dataset_20260910 as BD  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
import live_eth_extreme_detector_20260909 as X  # noqa: E402

OUT = ROOT / "tmp/eth_signal_midterm_20260910"
SIGNALS = X.B.SIGNALS
TFS = {"5m": 1, "15m": 3, "1h": 12, "4h": 48}          # 5분봉 몇 개로 만든 봉인가
BARS_H = [12, 48, 144]                                  # 축 1: 같은 봉 수
CLOCK_H_MIN = {"12h": 720, "1d": 1440, "3d": 4320, "7d": 10080}   # 축 2: 같은 시계(분)
COST_BP = 10.0
B_NULL = 400
SPLIT = "2025-09-01"                                    # 이벤트 트리거 재학습용 TRAIN/TEST 경계


def seed_of(*parts) -> int:
    """프로세스마다 달라지는 hash() 대신 결정론적 시드."""
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % 100000


def log(m):
    print(f"[mid {time.strftime('%H:%M:%S')}] {m}", flush=True)


def resample(kl: pd.DataFrame, mult: int) -> pd.DataFrame:
    if mult == 1:
        return kl.reset_index(drop=True)
    kl = kl.copy()
    for c in ("quote_volume", "trades"):
        kl[c] = pd.to_numeric(kl[c], errors="coerce")
    rule = f"{5 * mult}min"
    g = kl.set_index("timestamp").resample(rule, label="left", closed="left")
    o = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
               "taker_buy_base": "sum", "quote_volume": "sum", "trades": "sum"}).dropna(subset=["close"])
    return o.reset_index()


def excess_with_null(fires: np.ndarray, fwd: np.ndarray, sign: float, b: int = B_NULL, seed: int = 0):
    """발동 지점의 평균 방향수익(bp)과 순환이동 귀무 분포. fires/fwd 는 같은 길이, fwd 는 NaN 허용."""
    ok = np.isfinite(fwd)
    idx = np.flatnonzero(fires & ok)
    n = len(idx)
    if n < 20:
        return None
    val = sign * fwd
    obs = float(np.nanmean(val[idx]) * 1e4)
    acc = float(np.mean(val[idx] > 0))
    N = len(fwd)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, N, b)
    nulls = np.empty(b)
    for j, s in enumerate(shifts):
        sh = (idx + s) % N
        v = val[sh]
        nulls[j] = np.nanmean(v) * 1e4
    lo, hi = np.percentile(nulls, [2.5, 97.5])
    p = float((np.abs(nulls - np.nanmean(nulls)) >= abs(obs - np.nanmean(nulls))).mean())
    return {"n": int(n), "obs_bp": obs, "acc": acc, "null_mean_bp": float(np.mean(nulls)),
            "null_ci95": [float(lo), float(hi)], "excess_bp": obs - float(np.mean(nulls)),
            "p_cyclic": p, "beats_null": bool(obs > hi), "net_bp": obs - float(np.mean(nulls)) - COST_BP}


def fwd_returns(close: np.ndarray, h: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if h < len(close):
        out[:-h] = close[h:] / close[:-h] - 1.0
    return out


def evidence_axis(sigs: dict, closes: dict) -> dict:
    res = {"matched_bars": {}, "matched_clock": {}}
    for tf, sig in sigs.items():
        c = closes[tf]
        mult = TFS[tf]
        # 축 1: 같은 봉 수
        for H in BARS_H:
            fwd = fwd_returns(c, H)
            cells = {}
            for s in SIGNALS:
                for side, sign in (("bottom", 1.0), ("top", -1.0)):
                    f = sig[f"{side}_{s}"].fillna(False).to_numpy(bool)
                    r = excess_with_null(f, fwd, sign, seed=seed_of(tf, H, s, side))
                    if r:
                        cells[f"{s}|{side}"] = r
            res["matched_bars"].setdefault(f"H{H}bars", {})[tf] = cells
        # 축 2: 같은 시계
        for name, minutes in CLOCK_H_MIN.items():
            H = int(round(minutes / (5 * mult)))
            if H < 1 or H >= len(c) // 4:
                continue
            fwd = fwd_returns(c, H)
            cells = {}
            for s in SIGNALS:
                for side, sign in (("bottom", 1.0), ("top", -1.0)):
                    f = sig[f"{side}_{s}"].fillna(False).to_numpy(bool)
                    r = excess_with_null(f, fwd, sign, seed=seed_of(tf, name, s, side))
                    if r:
                        cells[f"{s}|{side}"] = r
            res["matched_clock"].setdefault(name, {})[tf] = {"bars": H, "cells": cells}
    return res


def summarize(cells: dict) -> dict:
    if not cells:
        return {"n_cells": 0}
    ex = np.array([v["excess_bp"] for v in cells.values()])
    acc = np.array([v["acc"] for v in cells.values()])
    nb = sum(v["beats_null"] for v in cells.values())
    net = sum(v["net_bp"] > 0 for v in cells.values())
    bot_ = [v["excess_bp"] for k, v in cells.items() if k.endswith("bottom")]
    top = [v["excess_bp"] for k, v in cells.items() if k.endswith("top")]
    return {"n_cells": len(cells), "median_excess_bp": float(np.median(ex)), "mean_excess_bp": float(np.mean(ex)),
            "max_excess_bp": float(np.max(ex)), "mean_acc": float(np.mean(acc)),
            "cells_beat_null": int(nb), "cells_net_positive": int(net),
            "bottom_mean_bp": float(np.mean(bot_)) if bot_ else None,
            "top_mean_bp": float(np.mean(top)) if top else None,
            "median_n_fires": float(np.median([v["n"] for v in cells.values()]))}


def extreme_detector_per_tf(sigs: dict, btcs: dict) -> dict:
    """극점 탐지기를 각 TF 에서 **같은 피쳐·같은 라벨**(i+1..i+12 극값 유지)로 재학습해 표본외 비교."""
    out = {}
    for tf, sig in sigs.items():
        try:
            A = X.build_rows(sig, btcs[tf], with_label=True)
        except Exception as e:  # noqa: BLE001
            out[tf] = {"error": f"{type(e).__name__}: {e}"}
            continue
        A = A[A["_y"] >= 0].reset_index(drop=True)
        ts = pd.to_datetime(A["_ts"])
        tr = (ts < SPLIT).to_numpy(); te = ~tr
        if tr.sum() < 400 or te.sum() < 100:
            out[tf] = {"error": "too_few_rows", "n": int(len(A))}
            continue
        Xm = A[X.FEATS].to_numpy(np.float32); y = A["_y"].to_numpy()
        P = np.zeros(te.sum())
        for sd in (11, 22, 33):
            m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                               min_samples_leaf=40, l2_regularization=1.0, random_state=sd).fit(Xm[tr], y[tr])
            P += m.predict_proba(Xm[te])[:, 1] / 3
        yte = y[te]
        row = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "base_rate_test": float(yte.mean()),
               "auc_test": float(roc_auc_score(yte, P))}
        for cov in (0.1, 0.25):
            thr = np.quantile(P, 1 - cov)
            sel = P >= thr
            row[f"precision_top{int(cov * 100)}"] = float(yte[sel].mean())
            row[f"n_top{int(cov * 100)}"] = int(sel.sum())
        # 상위 10% 콜의 방향 초과(H=12봉) -- 발동 = 그 TF 의 12봉
        c = sig["close"].to_numpy(float)
        fwd = fwd_returns(c, 12)
        thr = np.quantile(P, 0.9)
        idx_test = np.flatnonzero(te)
        picks = A["_i"].to_numpy()[idx_test[P >= thr]]
        longs = A["_long"].to_numpy()[idx_test[P >= thr]]
        fire_b = np.zeros(len(c), bool); fire_t = np.zeros(len(c), bool)
        fire_b[picks[longs]] = True; fire_t[picks[~longs]] = True
        for side, f, sign in (("bottom", fire_b, 1.0), ("top", fire_t, -1.0)):
            r = excess_with_null(f, fwd, sign, seed=seed_of(tf, "ext", side))
            if r:
                row[f"excess_{side}_H12bars"] = r
        out[tf] = row
        log(f"극점 {tf}: n={len(A)} AUC {row['auc_test']:.3f} 상위10% 정밀도 {row['precision_top10']:.3f} (기저 {row['base_rate_test']:.3f})")
    return out


def breakout_reversal_per_tf(sigs: dict, btcs: dict) -> dict:
    """돌파/되돌림: 앵커=첫 발동(직전 3봉 무발동), 배리어 ±0.8×ATR, 12봉 내 선착. 각 TF 재정의·재학습."""
    out = {}
    for tf, sig in sigs.items():
        c = sig["close"].to_numpy(float); hi = sig["high"].to_numpy(float); lo = sig["low"].to_numpy(float)
        atr = sig["atr_pct"].to_numpy(float)
        n = len(c)
        anyf = np.zeros(n, bool); side_b = np.zeros(n, bool)
        for s in SIGNALS:
            fb = sig[f"bottom_{s}"].fillna(False).to_numpy(bool); ft = sig[f"top_{s}"].fillna(False).to_numpy(bool)
            anyf |= fb | ft; side_b |= fb
        prev3 = pd.Series(anyf).rolling(3, min_periods=1).max().shift(1).fillna(0).to_numpy(bool)
        anchor = anyf & (~prev3)
        idx = np.flatnonzero(anchor)
        idx = idx[(idx >= 900) & (idx < n - 13)]
        if len(idx) < 200:
            out[tf] = {"error": "too_few_anchors", "n": int(len(idx))}
            continue
        # 라벨: 돌파(발현 방향으로 배리어 선착)=1, 되돌림=0, 미도달 제외
        up = side_b[idx]                       # 바닥 발동이면 위쪽이 "되돌림" 방향 -- 발현 방향은 그 반대
        dir_up = ~up                           # 발현(직전 움직임) 방향: 천장 발동이면 상승 발현
        y = np.full(len(idx), -1)
        for k, i in enumerate(idx):
            b = 0.8 * max(atr[i], 1e-6)
            up_px = c[i] * (1 + b); dn_px = c[i] * (1 - b)
            seg_hi = hi[i + 1:i + 13]; seg_lo = lo[i + 1:i + 13]
            t_up = np.argmax(seg_hi >= up_px) if (seg_hi >= up_px).any() else 99
            t_dn = np.argmax(seg_lo <= dn_px) if (seg_lo <= dn_px).any() else 99
            if t_up == 99 and t_dn == 99:
                continue
            hit_up = t_up < t_dn
            y[k] = int(hit_up == dir_up[k])    # 발현 방향으로 먼저 닿으면 돌파
        m = y >= 0
        A = X.build_rows(sig, btcs[tf])
        feat_by_i = {int(v): j for j, v in enumerate(A["_i"].to_numpy())}
        rows = [feat_by_i.get(int(i), -1) for i in idx[m]]
        keep = np.array([r >= 0 for r in rows])
        if keep.sum() < 200:
            out[tf] = {"error": "feature_join_too_small", "n": int(keep.sum())}
            continue
        Xm = A[X.FEATS].to_numpy(np.float32)[np.array([r for r in rows if r >= 0])]
        yy = y[m][keep]; tt = pd.to_datetime(sig["timestamp"].to_numpy()[idx[m][keep]])
        tr = np.asarray(tt < SPLIT); te = ~tr
        if tr.sum() < 200 or te.sum() < 60 or len(np.unique(yy[te])) < 2:
            out[tf] = {"error": "split_too_small", "n_train": int(tr.sum()), "n_test": int(te.sum())}
            continue
        P = np.zeros(te.sum())
        for sd in (11, 22, 33):
            mdl = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                                 min_samples_leaf=40, l2_regularization=1.0, random_state=sd).fit(Xm[tr], yy[tr])
            P += mdl.predict_proba(Xm[te])[:, 1] / 3
        pred = (P > 0.5).astype(int); yte = yy[te]
        rng = np.random.default_rng(7)
        nulls = [float((pred == rng.permutation(yte)).mean()) for _ in range(400)]
        acc = float((pred == yte).mean())
        conf = np.abs(P - 0.5); sel = conf >= np.quantile(conf, 0.5)
        out[tf] = {"n_anchors": int(len(idx)), "n_labeled": int(m.sum()), "n_train": int(tr.sum()), "n_test": int(te.sum()),
                   "breakout_rate": float(yte.mean()), "acc_test": acc,
                   "acc_null_p97.5": float(np.percentile(nulls, 97.5)),
                   "auc_test": float(roc_auc_score(yte, P)),
                   "acc_cov50": float((pred[sel] == yte[sel]).mean()),
                   "barrier_median_bp": float(np.median(0.8 * atr[idx[m][keep]]) * 1e4)}
        log(f"돌파/되돌림 {tf}: n_test={out[tf]['n_test']} 정확도 {acc:.3f} (귀무 97.5% {out[tf]['acc_null_p97.5']:.3f}) "
            f"배리어 중앙 {out[tf]['barrier_median_bp']:.0f}bp")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    eth = BD.load_klines("eth", "ETHUSDT"); btc = BD.load_klines("btc", "BTCUSDT")
    sigs, btcs, closes = {}, {}, {}
    for tf, mult in TFS.items():
        e, b = resample(eth, mult), resample(btc, mult)
        s = compute_signals(e, btc_df=b, funding_df=None)
        sigs[tf], btcs[tf], closes[tf] = s, b, s["close"].to_numpy(float)
        fb = sum(s[f"bottom_{x}"].fillna(False).sum() for x in SIGNALS)
        ft = sum(s[f"top_{x}"].fillna(False).sum() for x in SIGNALS)
        log(f"{tf}: {len(s):,}봉 · 발동 바닥 {fb:,} / 천장 {ft:,}")
    rep = {"cost_bp": COST_BP, "b_null": B_NULL, "split_for_models": SPLIT,
           "span": [str(eth.timestamp.iloc[0]), str(eth.timestamp.iloc[-1])],
           "bars": {tf: int(len(s)) for tf, s in sigs.items()}}
    ev = evidence_axis(sigs, closes)
    rep["evidence_raw"] = ev
    rep["evidence_summary"] = {
        "matched_bars": {h: {tf: summarize(cells) for tf, cells in d.items()} for h, d in ev["matched_bars"].items()},
        "matched_clock": {h: {tf: {**summarize(v["cells"]), "bars": v["bars"]} for tf, v in d.items()} for h, d in ev["matched_clock"].items()},
    }
    for h, d in rep["evidence_summary"]["matched_bars"].items():
        log(f"[같은 봉수 {h}] " + " · ".join(f"{tf} 중앙 {v['median_excess_bp']:+.1f}bp 귀무통과 {v['cells_beat_null']}/{v['n_cells']} 순양수 {v['cells_net_positive']}" for tf, v in d.items()))
    for h, d in rep["evidence_summary"]["matched_clock"].items():
        log(f"[같은 시계 {h}] " + " · ".join(f"{tf}({v['bars']}봉) 중앙 {v['median_excess_bp']:+.1f}bp 귀무통과 {v['cells_beat_null']}/{v['n_cells']} 순양수 {v['cells_net_positive']}" for tf, v in d.items()))
    rep["extreme_detector"] = extreme_detector_per_tf(sigs, btcs)
    rep["breakout_reversal"] = breakout_reversal_per_tf(sigs, btcs)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"→ {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
