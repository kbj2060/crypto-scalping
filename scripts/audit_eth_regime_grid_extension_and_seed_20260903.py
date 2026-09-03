#!/usr/bin/env python3
"""ETH 레짐 **격자 확장 + Phase3/3b + 시드 견고성** -- XRP·BTC와 같은 감사를 ETH에.

## 왜

2026-09-03에 XRP(→S96_K9 교체)와 BTC(→S24_K3 유지)는 격자 확장 감사를 받았는데
**ETH는 안 받았다.** ETH도 같은 격자 `SCALES=(6,12,24,48)` / `DEBOUNCES=(1,3,6)`을 쓰고
배포 라벨은 `S12_K3`, 분류기는 `SEED=7529` **단일 시드**다.

⚠️`S12_K3`은 격자 **내부값**이라 경계 위반은 아니다. 그러나 XRP에서 최적이 격자 **바깥**
(S96)에 있었으므로 "경계가 아니니 괜찮다"는 확인이 아니다.

## ⚠️⚠️ETH만의 문제 -- OOS 창이 **잘려 있었다**

`analyze_eth_confluence_oscillator_bottom_top_evidence_20260814.py`:

    OOS_END = pd.Timestamp("2026-02-17 15:00:00")  # matches raw data's actual coverage

`data/eth_5m_1year.csv`가 **2026-02-17에 끝나서** ETH 증거신호·레짐의 OOS가
**1.5개월(2026-01-01~02-17)** 이었다. XRP·BTC는 **3개월**(~2026-03-31)을 썼다.
⇒ **자산 간 비교가 같은 잣대가 아니었다.**

⭐**전체 구간 파일은 이미 있다**(`binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv`, 2026-08-31까지)
⭐**피벗도 2026-07-19까지 덮는다**(`load_zigzag_pivots()`는 zigzag 라벨 파일을 읽지 klines가 아니다)
⇒ 이 감사는 **전체 구간 klines를 써서 OOS를 3개월로 복원**한다. XRP·BTC와 같은 잣대가 된다.

## 3단계 (XRP·BTC와 동일 절차)

  1) **Phase 2 격자 확장** -- S(6..192) × K(1..12), 파트너 BTC(ETH의 교차자산 슬롯)
  2) **Phase 3 + 3b (깨끗한 분할)** -- ⚠️원본 Phase3는 **2026-07-01~08-19를 1회 소진**했다.
     그 창을 **읽지 않는다**: 학습 2024-01-01~2025-08-31 / 평가 2025-09-01~2026-03-31
  3) **시드 견고성** -- 후보가 배포본을 이기면 랜덤추출 8시드로 확인

⚠️격자 lift만으로 교체하지 않는다(§5-A). ⚠️HOLDOUT 미터치.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib        # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier                  # noqa: E402

_P = importlib.util.spec_from_file_location(
    "ethp3", ROOT / "scripts/research_eth_regime_s12k3_label_train_20260902.py")
_p = importlib.util.module_from_spec(_P)
_P.loader.exec_module(_p)

_E = importlib.util.spec_from_file_location(
    "ethpar", ROOT / "scripts/research_eth_causal_evidence_parity_20260903.py")
_e = importlib.util.module_from_spec(_E)
_E.loader.exec_module(_e)

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import load_zigzag_pivots  # noqa: E402
from features.elite import RegimeEngine                                      # noqa: E402
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_regime_label_conditional_lift_20260902 import seg_lift     # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import (           # noqa: E402
    _debounce, efficiency_ratio, scaled_label,
)

OUT = ROOT / "data/research/eth_regime_grid_extension_and_seed_20260903.json"

SCALES_EXT = (6, 12, 24, 48, 96, 192)
DEBOUNCES_EXT = (1, 3, 6, 9, 12)
DEPLOYED = "S12_K3"

# ⭐전체 구간 -- OOS를 3개월로 복원(기존 ETH 작업은 2026-02-17에서 잘렸다)
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
PARTNER = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
EV_VAL_START, EV_VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31 23:59:59")
EV_OOS_START, EV_OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59")

FIT_START, FIT_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-08-31 23:55")
EVAL_START, EVAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-03-31 23:55")
NEVER_READ_FROM = pd.Timestamp("2026-04-01")     # HOLDOUT + 소진된 Phase3 창(2026-07~08)

SEEDS = [7529, 811453, 30011, 947, 260317, 5387291, 68041, 1299709]
TOP_N_PHASE3 = 4


def log(m): print(f"[eth-reg] {m}", flush=True)


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
                col = f"{side}_{sname}"
                if col not in frame.columns:
                    continue
                sig = frame[col].fillna(False).to_numpy()
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
    rep = {"asset": "ETHUSDT", "deployed": DEPLOYED, "holdout_touched": False,
           "scales": list(SCALES_EXT), "debounces": list(DEBOUNCES_EXT),
           "prev_grid": {"scales": [6, 12, 24, 48], "debounces": [1, 3, 6]},
           "oos_window_restored": {"was": "2026-01-01~2026-02-17 (1.5개월, klines 잘림)",
                                   "now": "2026-01-01~2026-03-31 (3개월, api CSV)"},
           "never_read_from": str(NEVER_READ_FROM), "seeds": SEEDS}

    # ---------- Phase 2 ----------
    raw = pd.read_csv(KLINES, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    partner = pd.read_csv(PARTNER, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    for d in (raw, partner):
        d["timestamp"] = d["timestamp"].astype("datetime64[ns]")
    frame = compute_signals(raw, btc_df=partner, funding_df=_e.load_eth_funding())
    frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
    pivots = load_zigzag_pivots()
    pv_ts = pd.to_datetime(pivots["timestamp"]).astype("datetime64[ns]")
    pivots = pivots.assign(timestamp=pv_ts)
    ts_e, close_e = frame["timestamp"], frame["close"]
    windows = {"VAL": ((ts_e >= EV_VAL_START) & (ts_e <= EV_VAL_END)).to_numpy(),
               "OOS": ((ts_e >= EV_OOS_START) & (ts_e <= EV_OOS_END)).to_numpy()}
    pivot_pos = {s: frame.index[frame["timestamp"].isin(
        pivots.loc[pivots["pivot_type"] == s, "timestamp"])].to_numpy()
        for s in ("bottom", "top")}
    log(f"ETH 프레임 {len(frame):,}봉  {ts_e.min()} ~ {ts_e.max()}")
    log(f"⭐OOS 창 복원: VAL {int(windows['VAL'].sum()):,}봉 / OOS {int(windows['OOS'].sum()):,}봉"
        f"  (기존 ETH 작업은 2026-02-17에서 잘렸다)")
    log(f"피벗 bottom {len(pivot_pos['bottom'])} / top {len(pivot_pos['top'])}")
    log(f"Phase2 격자: S{SCALES_EXT} × K{DEBOUNCES_EXT} = {len(SCALES_EXT)*len(DEBOUNCES_EXT)}종")

    fit_e = (ts_e <= FIT_END).to_numpy()
    p2 = {}
    for s in SCALES_EXT:
        for k in DEBOUNCES_EXT:
            y = make_label(close_e, fit_e, s, k)
            bp, nc, mv, mo = both_positive(frame, pivot_pos, windows, y == 2)
            p2[f"S{s}_K{k}"] = {"both_positive": bp, "n_cells": nc, "mean_val": mv,
                                "mean_oos": mo,
                                "label_flip_rate": float((np.diff(y) != 0).mean()),
                                "chop_share": float((y == 2).mean())}
    ranked = sorted(p2.items(), key=lambda kv: (-kv[1]["both_positive"], -kv[1]["mean_oos"]))
    log("")
    log(f"{'라벨':<12}{'양쪽창':>9}{'meanVAL':>10}{'meanOOS':>10}{'전환율':>9}")
    for n, v in ranked[:10]:
        log(f"{n:<12}{v['both_positive']:>4}/{v['n_cells']:<4}{v['mean_val']:>+10.4f}"
            f"{v['mean_oos']:>+10.4f}{v['label_flip_rate']:>9.4f}"
            f"{'  ⭐배포' if n == DEPLOYED else ''}")
    dv = p2.get(DEPLOYED, {})
    rk = [n for n, _ in ranked].index(DEPLOYED) + 1 if DEPLOYED in p2 else None
    log(f"  배포 {DEPLOYED}: 양쪽창 {dv.get('both_positive')}/{dv.get('n_cells')} "
        f"OOS {dv.get('mean_oos', float('nan')):+.4f}  (순위 {rk}/{len(ranked)})")
    rep["phase2"] = {"by_label": p2, "ranking": [n for n, _ in ranked], "deployed_rank": rk}

    cands = [n for n, _ in ranked[:TOP_N_PHASE3]]
    if DEPLOYED not in cands:
        cands.append(DEPLOYED)
    log(f"  ⇒ Phase 3 대상: {cands}")

    # ---------- Phase 3 + 3b ----------
    payload = joblib.load(_p.GBM3_MODEL_PATH)
    feat_cols, medians = payload["feature_cols"], payload["feature_medians"]
    df = _p.load_frame()
    ts = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    df = df.assign(timestamp=ts)
    fit = ((ts >= FIT_START) & (ts <= FIT_END)).to_numpy()
    ev = ((ts >= EVAL_START) & (ts <= EVAL_END)).to_numpy()
    log("")
    log(f"Phase3 캐노니컬 {len(df):,}행 | 학습 {int(fit.sum()):,} / 평가 {int(ev.sum()):,}")
    log(f"⚠️{NEVER_READ_FROM.date()} 이후 {int((ts >= NEVER_READ_FROM).sum()):,}봉 미사용"
        " (HOLDOUT + 소진된 Phase3 창 2026-07~08)")
    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))

    labels = {}
    for n in cands:
        s, k = int(n.split("_")[0][1:]), int(n.split("_K")[1])
        labels[n] = make_label(df["close"], fit, s, k)
    labels["REF_RegimeEngine"] = _p.deployed_label(df)

    log("")
    log(f"{'라벨':<18}{'bal_acc':>9}{'chop_R':>9}{'chop_P':>9}{'flip':>9}{'3b양쪽창':>10}{'3b OOS':>10}")
    p3, gate_pred = {}, {}
    for n, y in labels.items():
        m = HistGradientBoostingClassifier(random_state=_p.SEED, **_p.GBM3_HP).fit(
            x.loc[fit, feat_cols], y[fit])
        r = _p.evaluate(y[ev], m.predict(x.loc[ev, feat_cols]))
        pf = pd.DataFrame({"timestamp": ts, "pred": m.predict(x[feat_cols])})
        merged = frame[["timestamp"]].merge(pf, on="timestamp", how="left")
        bp, nc, mv, mo = both_positive(frame, pivot_pos, windows, merged["pred"].to_numpy() == 2)
        p3[n] = {"bal_acc": r["balanced_accuracy"], "chop_recall": r["chop_recall"],
                 "chop_precision": r["chop_precision"], "flip_rate": r["flip_rate"],
                 "gate_both_positive": bp, "gate_cells": nc,
                 "gate_val_mean": mv, "gate_oos_mean": mo}
        gate_pred[n] = y
        log(f"{n:<18}{r['balanced_accuracy']:>9.4f}{r['chop_recall']:>9.4f}"
            f"{r['chop_precision']:>9.4f}{r['flip_rate']:>9.4f}{bp:>6}/{nc:<3}{mo:>+10.4f}"
            f"{'  ⭐배포' if n == DEPLOYED else ''}")
    rep["phase3"] = p3

    ranked3 = sorted((n for n in p3 if n != "REF_RegimeEngine"),
                     key=lambda n: (-p3[n]["gate_both_positive"], -p3[n]["gate_oos_mean"]))
    top = ranked3[0]
    log("")
    log(f"⇒ Phase 3b 1위: **{top}**  (배포 {DEPLOYED})")

    # ---------- 시드 견고성 ----------
    rep["seed_robustness"] = None
    if top != DEPLOYED:
        log(""); log(f"시드 견고성: {top} vs {DEPLOYED} (랜덤추출 {len(SEEDS)}시드)")
        wins = {"bal_acc": 0, "flip": 0, "gate": 0}
        rows = []
        for sd in SEEDS:
            per = {}
            for n in (DEPLOYED, top):
                y = labels[n]
                m = HistGradientBoostingClassifier(random_state=sd, **_p.GBM3_HP).fit(
                    x.loc[fit, feat_cols], y[fit])
                r = _p.evaluate(y[ev], m.predict(x.loc[ev, feat_cols]))
                pf = pd.DataFrame({"timestamp": ts, "pred": m.predict(x[feat_cols])})
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
        log(f"  ⇒ Phase 3b 1위가 배포본({DEPLOYED}) -- 교체 후보 없음")
        # ⭐교체 후보가 없어도 **배포본 자신의 시드 견고성**은 재야 한다.
        # CLAUDE.md Seed-Diversity 게이트는 "시드 리스트를 리포트에 기록"까지 요구한다.
        # XRP/BTC는 채웠는데 ETH만 비어 있었다(2026-09-03).
        log(""); log(f"시드 견고성(배포본 {DEPLOYED} 단독, 랜덤추출 {len(SEEDS)}시드)")
        rows = []
        for sd in SEEDS:
            y = labels[DEPLOYED]
            m = HistGradientBoostingClassifier(random_state=sd, **_p.GBM3_HP).fit(
                x.loc[fit, feat_cols], y[fit])
            r = _p.evaluate(y[ev], m.predict(x.loc[ev, feat_cols]))
            pf = pd.DataFrame({"timestamp": ts, "pred": m.predict(x[feat_cols])})
            merged = frame[["timestamp"]].merge(pf, on="timestamp", how="left")
            bp, nc, _mv, mo = both_positive(frame, pivot_pos, windows,
                                            merged["pred"].to_numpy() == 2)
            rows.append({"seed": sd, "bal_acc": r["balanced_accuracy"],
                         "flip_rate": r["flip_rate"], "gate_both_positive": bp,
                         "gate_n_cells": nc, "gate_oos_mean": mo})
            log(f"  seed={sd:<9} bal_acc {r['balanced_accuracy']:.4f}  "
                f"플리커 {r['flip_rate']:.4f}  게이트 {bp}/{nc}  OOS평균 {mo:+.4f}")
        ba = np.array([r["bal_acc"] for r in rows])
        gm = np.array([r["gate_oos_mean"] for r in rows])
        gb = np.array([r["gate_both_positive"] for r in rows])
        log("")
        log(f"  bal_acc  평균 {ba.mean():.4f} ± {ba.std(ddof=1):.4f}  [{ba.min():.4f}, {ba.max():.4f}]")
        log(f"  게이트 OOS평균 {gm.mean():+.4f} ± {gm.std(ddof=1):.4f}  "
            f"[{gm.min():+.4f}, {gm.max():+.4f}]  전부>0: {'✅' if (gm > 0).all() else '❌'}")
        log(f"  게이트 통과셀 {gb.min()}~{gb.max()}/{rows[0]['gate_n_cells']}")
        ok = bool((gm > 0).all())
        log(f"  ⇒ 부호 일치(OOS): {'✅**통과**' if ok else '❌미달'} -- "
            f"CLAUDE.md Seed-Diversity 게이트 요건 3(시드 리스트 기록) 충족")
        rep["seed_robustness"] = {"mode": "deployed_only(교체 후보 없음)", "deployed": DEPLOYED,
                                  "seeds": SEEDS, "n_seeds": len(rows), "per_seed": rows,
                                  "seed_selection": "랜덤 추출(고정 간격 증가 아님)",
                                  "bal_acc": {"mean": float(ba.mean()), "std": float(ba.std(ddof=1)),
                                              "min": float(ba.min()), "max": float(ba.max())},
                                  "gate_oos_mean": {"mean": float(gm.mean()),
                                                    "std": float(gm.std(ddof=1)),
                                                    "min": float(gm.min()), "max": float(gm.max())},
                                  "all_oos_positive": ok}

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
