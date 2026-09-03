#!/usr/bin/env python3
"""BTC 레짐 **격자 확장 + Phase3/3b(깨끗한 분할) + 시드 견고성** -- XRP와 같은 감사를 BTC에.

## 왜

2026-09-03 XRP 레짐 감사에서:
  · `SCALES=(6,12,24,48)` / `DEBOUNCES=(1,3,6)` 격자를 S->192 / K->12로 넓히니
    **S96_K9가 배포본을 네 축 전부 이겼다** -> 교체 배포.
  · 그 결정이 단일 시드였음이 드러나 8시드로 재확인했다(8/8 우위).

**BTC는 그 감사를 받지 않았다.** BTC도 같은 격자(`SCALES=(6,12,24,48)`, `DEBOUNCES=(1,3,6)`,
ETH 스크립트에서 import)를 쓰고 배포 라벨은 `S24_K3`, 분류기는 `SEED=7529` **단일 시드**다.

⚠️BTC `S24_K3`은 그 격자의 **내부값**이라 경계 위반은 아니다. 그러나 XRP에서 최적이 격자
**바깥**(S96)에 있었으므로, "경계가 아니니 괜찮다"는 확인이 아니다.

## 3단계 (XRP와 동일 절차)

  1) **Phase 2 격자 확장** -- S(6..192) x K(1..12), 파트너 ETH 고정(BTC의 교차자산 슬롯).
  2) **Phase 3 + 3b (깨끗한 분할)** -- 상위 후보와 배포 S24_K3, REF를 비교.
     ⚠️원본 Phase 3는 **2026-07-01~08-01을 1회 소진**했다. 그 창을 **읽지 않는다**:
        학습 2024-01-01~2025-08-31 / 평가 2025-09-01~2026-03-31
  3) **시드 견고성** -- 상위 후보 vs 배포본을 8시드(랜덤 추출)로.

⚠️격자 lift만으로 교체하지 않는다(포팅 프로토콜 §5-A). Phase 3/3b + 시드까지 본다.
⚠️HOLDOUT 미터치.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.ensemble import HistGradientBoostingClassifier                   # noqa: E402

_S = importlib.util.spec_from_file_location(
    "btcp3", ROOT / "scripts/research_btc_regime_s24k3_label_train_20260902.py")
_p = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_p)

from features.elite import RegimeEngine                                       # noqa: E402
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_regime_label_conditional_lift_20260902 import N_NULL, seg_lift    # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import (            # noqa: E402
    _debounce, efficiency_ratio, scaled_label,
)
from research_btc_regime_label_conditional_lift_20260902 import (             # noqa: E402
    BTC_KLINES, ETH_KLINES, build_btc_pivots, load_btc_funding_z,
)

OUT = ROOT / "data/research/btc_regime_grid_extension_and_seed_20260903.json"

SCALES_EXT = (6, 12, 24, 48, 96, 192)
DEBOUNCES_EXT = (1, 3, 6, 9, 12)
DEPLOYED = "S24_K3"

FIT_START = pd.Timestamp("2024-01-01T00:00:00")
FIT_END = pd.Timestamp("2025-08-31T23:55:00")
EVAL_START = pd.Timestamp("2025-09-01T00:00:00")
EVAL_END = pd.Timestamp("2026-03-31T23:55:00")
NEVER_READ_FROM = pd.Timestamp("2026-04-01T00:00:00")   # HOLDOUT + 소진된 Phase3 창

SEEDS = [7529, 811453, 30011, 947, 260317, 5387291, 68041, 1299709]
TOP_N_PHASE3 = 4          # Phase 2 상위 몇 개를 Phase 3으로 넘길지


def log(m): print(f"[btc-reg] {m}", flush=True)


def make_label(close, fit_mask, scale, k):
    c = close[fit_mask]
    r1 = float((efficiency_ratio(c, 24) >= 0.20).mean())
    r2 = float((efficiency_ratio(c, 48) >= 0.16).mean())
    t1 = float(efficiency_ratio(close, scale).quantile(1.0 - r1))
    t2 = float(efficiency_ratio(close, 2 * scale).quantile(1.0 - r2))
    y0 = scaled_label(close, scale, t1, t2)
    return y0 if k == 1 else _debounce(y0, k)


def both_positive(frame, pivot_pos, windows, chop):
    rows = []
    for wn, wm in windows.items():
        seg = chop & wm
        for sname, _ in SIGNAL_ORDER:
            for side in ("bottom", "top"):
                sig = frame[f"{side}_{sname}"].fillna(False).to_numpy()
                la, _n = seg_lift(sig, pivot_pos[side], wm)
                lc, _n2 = seg_lift(sig, pivot_pos[side], seg)
                if not (np.isfinite(la) and np.isfinite(lc)) or la <= 0:
                    continue
                rows.append({"w": wn, "s": sname, "side": side, "imp": lc / la - 1.0})
    d = pd.DataFrame(rows)
    if not len(d):
        return 0, 0, float("nan"), float("nan")
    pv = d.pivot_table(index=["s", "side"], columns="w", values="imp")
    if "VAL" not in pv or "OOS" not in pv:
        return 0, int(len(pv)), float("nan"), float("nan")
    return (int(((pv["VAL"] > 0) & (pv["OOS"] > 0)).sum()), int(len(pv)),
            float(pv["VAL"].mean()), float(pv["OOS"].mean()))


def main() -> int:
    t0 = time.time()
    rep = {"asset": "BTCUSDT", "deployed": DEPLOYED, "holdout_touched": False,
           "scales": list(SCALES_EXT), "debounces": list(DEBOUNCES_EXT),
           "prev_grid": {"scales": [6, 12, 24, 48], "debounces": [1, 3, 6]},
           "never_read_from": str(NEVER_READ_FROM), "seeds": SEEDS}

    # ---------- 1) Phase 2: 격자 확장 (증거신호 프레임 기준) ----------
    raw = pd.read_csv(BTC_KLINES, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    partner = pd.read_csv(ETH_KLINES, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    # ⚠️⚠️이 저장소에서 **반복된** 함정: pandas datetime64 해상도([ns] vs [us]) 불일치로
    # `compute_signals` 내부 `merge_asof`가 MergeError로 죽는다. `load_btc_funding_z()`가 [us]를
    # 돌려주는 반면 klines는 parse_dates로 [ns]다. **입력 3개를 먼저 [ns]로 통일**한다.
    # ⚠️펀딩 프레임의 조인 키는 `timestamp`가 아니라 **`calc_time`**이다 -- 시간형 컬럼을
    # 이름으로 특정하지 말고 dtype으로 훑어야 한다(XRP 로더엔 이 수정이 들어 있고 BTC엔 없다).
    funding = load_btc_funding_z()
    for _d in (raw, partner, funding):
        for _c in _d.columns:
            if str(_d[_c].dtype).startswith("datetime64"):
                _d[_c] = _d[_c].astype("datetime64[ns]")
    frame = compute_signals(raw, btc_df=partner, funding_df=funding)
    frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
    pivots = build_btc_pivots()
    ts_e, close_e = frame["timestamp"], frame["close"]
    windows = {"VAL": ((ts_e >= _p.EV_VAL_START) & (ts_e <= _p.EV_VAL_END)).to_numpy(),
               "OOS": ((ts_e >= _p.EV_OOS_START) & (ts_e <= _p.EV_OOS_END)).to_numpy()}
    pivot_pos = {s: frame.index[frame["timestamp"].isin(
        pivots.loc[pivots["pivot_type"] == s, "timestamp"])].to_numpy() for s in ("bottom", "top")}
    log(f"BTC 증거신호 프레임 {len(frame):,} | pivots {len(pivots):,} | 파트너 ETH")
    log(f"Phase2 격자 확장: S{SCALES_EXT} x K{DEBOUNCES_EXT} = {len(SCALES_EXT)*len(DEBOUNCES_EXT)}종")

    fit_e = (ts_e <= FIT_END).to_numpy()
    p2 = {}
    for s in SCALES_EXT:
        for k in DEBOUNCES_EXT:
            y = make_label(close_e, fit_e, s, k)
            bp, nc, mv, mo = both_positive(frame, pivot_pos, windows, y == 2)
            p2[f"S{s}_K{k}"] = {"both_positive": bp, "n_cells": nc, "mean_val": mv, "mean_oos": mo,
                                "label_flip_rate": float((np.diff(y) != 0).mean()),
                                "chop_share": float((y == 2).mean())}
    ranked = sorted(p2.items(), key=lambda kv: (-kv[1]["both_positive"], -kv[1]["mean_oos"]))
    log("")
    log(f"{'라벨':<12}{'양쪽창':>9}{'meanVAL':>10}{'meanOOS':>10}{'전환율':>9}")
    for n, v in ranked[:10]:
        mark = " ⭐배포" if n == DEPLOYED else ""
        log(f"{n:<12}{v['both_positive']:>4}/{v['n_cells']:<4}{v['mean_val']:>+10.4f}"
            f"{v['mean_oos']:>+10.4f}{v['label_flip_rate']:>9.4f}{mark}")
    dv = p2.get(DEPLOYED, {})
    log(f"  배포 {DEPLOYED}: 양쪽창 {dv.get('both_positive')}/{dv.get('n_cells')} "
        f"OOS {dv.get('mean_oos', float('nan')):+.4f}  (순위 "
        f"{[n for n, _ in ranked].index(DEPLOYED)+1 if DEPLOYED in p2 else '-'}/{len(ranked)})")
    rep["phase2"] = {"by_label": p2, "ranking": [n for n, _ in ranked]}

    cands = [n for n, _ in ranked[:TOP_N_PHASE3]]
    if DEPLOYED not in cands:
        cands.append(DEPLOYED)
    log(f"  ⇒ Phase 3 대상: {cands}")

    # ---------- 2) Phase 3 + 3b (소진 창 미사용) ----------
    _p._assert_btc_canon() if hasattr(_p, "_assert_btc_canon") else None
    payload = joblib.load(_p.GBM3_MODEL_PATH)
    feat_cols, medians = payload["feature_cols"], payload["feature_medians"]
    df = _p.load_btc_frame(feat_cols)
    ts = df["timestamp"]
    fit = ((ts >= FIT_START) & (ts <= FIT_END)).to_numpy()
    ev = ((ts >= EVAL_START) & (ts <= EVAL_END)).to_numpy()
    log("")
    log(f"Phase3 캐노니컬 {len(df):,}행 | 학습 {int(fit.sum()):,} / 평가 {int(ev.sum()):,}")
    log(f"⚠️{NEVER_READ_FROM.date()} 이후 {int((ts >= NEVER_READ_FROM).sum()):,}봉 미사용"
        " (HOLDOUT + 소진된 Phase3 창)")
    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))

    labels = {}
    for n in cands:
        s, k = int(n.split("_")[0][1:]), int(n.split("_K")[1])
        labels[n] = make_label(df["close"], fit, s, k)
    ref = df.copy()
    ref["mtf_trend_1h"] = df["close"].ewm(span=12, adjust=False).mean().pct_change().fillna(0.0)
    lab = RegimeEngine().compute(ref)
    yr = np.full(len(df), 2, dtype=int)
    yr[lab["regime_bull"].to_numpy() > 0] = 0
    yr[lab["regime_bear"].to_numpy() > 0] = 1
    labels["REF_RegimeEngine"] = yr

    log("")
    log(f"{'라벨':<18}{'bal_acc':>9}{'chop_R':>9}{'chop_P':>9}{'flip':>9}{'3b양쪽창':>10}{'3b OOS':>10}")
    p3 = {}
    for n, y in labels.items():
        m = HistGradientBoostingClassifier(random_state=_p.SEED, **_p.GBM3_HP).fit(
            x.loc[fit, feat_cols], y[fit])
        r = _p.evaluate(y[ev], m.predict(x.loc[ev, feat_cols]))
        pf = pd.DataFrame({"timestamp": ts.astype("datetime64[ns]"),
                           "pred": m.predict(x[feat_cols])})
        merged = frame[["timestamp"]].merge(pf, on="timestamp", how="left")
        bp, nc, mv, mo = both_positive(frame, pivot_pos, windows,
                                       merged["pred"].to_numpy() == 2)
        p3[n] = {"bal_acc": r["balanced_accuracy"], "chop_recall": r["chop_recall"],
                 "chop_precision": r["chop_precision"], "flip_rate": r["flip_rate"],
                 "gate_both_positive": bp, "gate_cells": nc,
                 "gate_val_mean": mv, "gate_oos_mean": mo}
        mark = " ⭐배포" if n == DEPLOYED else ""
        log(f"{n:<18}{r['balanced_accuracy']:>9.4f}{r['chop_recall']:>9.4f}"
            f"{r['chop_precision']:>9.4f}{r['flip_rate']:>9.4f}{bp:>6}/{nc:<3}{mo:>+10.4f}{mark}")
    rep["phase3"] = p3

    ranked3 = sorted((n for n in p3 if n != "REF_RegimeEngine"),
                     key=lambda n: (-p3[n]["gate_both_positive"], -p3[n]["gate_oos_mean"]))
    top = ranked3[0]
    log("")
    log(f"⇒ Phase 3b 1위: **{top}**  (배포 {DEPLOYED})")

    # ---------- 3) 시드 견고성 (상위 후보 vs 배포본) ----------
    rep["seed_robustness"] = None
    if top != DEPLOYED:
        log("")
        log(f"시드 견고성: {top} vs {DEPLOYED}  (랜덤 추출 {len(SEEDS)}시드)")
        wins = {"bal_acc": 0, "flip": 0, "gate": 0}
        rows = []
        for sd in SEEDS:
            per = {}
            for n in (DEPLOYED, top):
                y = labels[n]
                m = HistGradientBoostingClassifier(random_state=sd, **_p.GBM3_HP).fit(
                    x.loc[fit, feat_cols], y[fit])
                r = _p.evaluate(y[ev], m.predict(x.loc[ev, feat_cols]))
                pf = pd.DataFrame({"timestamp": ts.astype("datetime64[ns]"),
                                   "pred": m.predict(x[feat_cols])})
                merged = frame[["timestamp"]].merge(pf, on="timestamp", how="left")
                bp, nc, _mv, mo = both_positive(frame, pivot_pos, windows,
                                                merged["pred"].to_numpy() == 2)
                per[n] = {"bal_acc": r["balanced_accuracy"], "flip_rate": r["flip_rate"],
                          "gate_both_positive": bp, "gate_oos_mean": mo}
            wins["bal_acc"] += per[top]["bal_acc"] > per[DEPLOYED]["bal_acc"]
            wins["flip"] += per[top]["flip_rate"] < per[DEPLOYED]["flip_rate"]
            wins["gate"] += per[top]["gate_both_positive"] > per[DEPLOYED]["gate_both_positive"]
            rows.append({"seed": sd, **per})
            log(f"  seed={sd:<9} {DEPLOYED} bal {per[DEPLOYED]['bal_acc']:.4f} / "
                f"{top} bal {per[top]['bal_acc']:.4f}   "
                f"게이트 {per[DEPLOYED]['gate_both_positive']} vs {per[top]['gate_both_positive']}")
        n = len(rows)
        d = [r[top]["bal_acc"] - r[DEPLOYED]["bal_acc"] for r in rows]
        log("")
        for kk, lb in (("bal_acc", "bal_acc 우위"), ("flip", "플리커 우위"), ("gate", "게이트 우위")):
            log(f"  {lb:<14} {wins[kk]}/{n}  {'✅전부' if wins[kk] == n else '⚠️일부 뒤집힘'}")
        log(f"  bal_acc 차이: 평균 {np.mean(d):+.4f} 최소 {min(d):+.4f} 최대 {max(d):+.4f}")
        ok = wins["bal_acc"] == n and wins["gate"] == n
        log(f"  ⇒ {'⚠️**교체 검토 가능** -- 미사용 창 확인 필요' if ok else '✅현행 유지 -- 시드에 흔들림'}")
        rep["seed_robustness"] = {"candidate": top, "deployed": DEPLOYED, "per_seed": rows,
                                  "wins": wins, "n_seeds": n,
                                  "bal_acc_delta": {"mean": float(np.mean(d)),
                                                    "min": float(min(d)), "max": float(max(d))},
                                  "candidate_robust": bool(ok)}
    else:
        log(f"  ⇒ Phase 3b 1위가 배포본({DEPLOYED})이다 -- 교체 후보 없음, 시드 비교 불필요")

    rep["top_phase3b"] = top
    rep["change_recommended"] = bool(top != DEPLOYED
                                     and (rep["seed_robustness"] or {}).get("candidate_robust"))
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"⇒ 교체 권고: {'⚠️예' if rep['change_recommended'] else '✅아니오 (현행 유지)'}")
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
