#!/usr/bin/env python3
"""XRP 레짐 `S96_K9` 채택 결정의 **시드 견고성** -- 단일 시드로 배포한 것을 사후 검증.

## 왜 -- 내가 단일 시드로 배포했다

2026-09-03에 `S48_K6` -> `S96_K9` 교체를 배포했는데, Phase 3/3b 비교도 미사용 창 확인도
전부 **`SEED = 7529` 단일 시드**로 돌렸다. CLAUDE.md의 Seed-Diversity 게이트는 문구상
단일 시드 구조를 요건 대상에서 제외하지만, 그건 "통과했다"가 아니라 **측정이 없다**는 뜻이다.

⇒ 결정이 시드에 흔들리면 나는 **노이즈를 배포한 것**이다. 지금 확인한다.

## 설계

라벨 자체는 `close`에서 결정론적으로 나오므로 시드와 무관하다. **GBM 학습만** 시드에 의존한다.
그래서 라벨 2종(S48_K6 / S96_K9)을 고정하고 **분류기 시드만 N종**으로 바꿔 학습한다.

  분할: 학습 2024-01-01~2025-08-31 / 평가 2025-09-01~2026-03-31
        (원본 Phase3의 소진 창 2026-07~08과 승격확인에 쓴 2026-04~06을 **둘 다 읽지 않는다**)
  시드: 8종, **랜덤 추출**(고정 간격 증가 금지 -- CLAUDE.md Sigma3-1h 전례)

## 판정 (실행 전 고정)

  S96_K9가 **모든 시드에서** S48_K6보다 bal_acc가 높아야 한다(부호 일관성).
  하나라도 뒤집히면 배포 결정이 시드 노이즈 위에 있었다는 뜻이므로 사용자에게 보고한다.
  Phase 3b(예측-chop 게이팅) 우열도 같이 센다.
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

from sklearn.ensemble import HistGradientBoostingClassifier                  # noqa: E402

_S = importlib.util.spec_from_file_location(
    "xrpp3", ROOT / "scripts/research_xrp_regime_s48k6_label_train_20260903.py")
_p = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_p)

_C = importlib.util.spec_from_file_location(
    "xrpclean", ROOT / "scripts/research_xrp_regime_extended_label_phase3_clean_20260903.py")
_c = importlib.util.module_from_spec(_C)
_C.loader.exec_module(_c)

from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_regime_label_conditional_lift_20260902 import seg_lift            # noqa: E402

OUT = ROOT / "data/research/xrp_regime_s96k9_seed_robustness_20260903.json"

FIT_START = pd.Timestamp("2024-01-01T00:00:00")
FIT_END = pd.Timestamp("2025-08-31T23:55:00")
EVAL_START = pd.Timestamp("2025-09-01T00:00:00")
EVAL_END = pd.Timestamp("2026-03-31T23:55:00")
NEVER_READ_FROM = pd.Timestamp("2026-04-01T00:00:00")   # 승격확인창 + 소진 Phase3창 + HOLDOUT

# ⭐랜덤 추출 8종. 고정 간격 증가 금지(CLAUDE.md: Sigma3-1h가 +5 증분으로 OOS 부호가 뒤집혔다).
SEEDS = [7529, 811453, 30011, 947, 260317, 5387291, 68041, 1299709]
LABELS = {"S48_K6": (48, 6), "S96_K9": (96, 9)}
DEPLOYED, PREVIOUS = "S96_K9", "S48_K6"


def log(m): print(f"[seedrob] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    _p._assert_xrp_canon()
    payload = joblib.load(_p.GBM3_MODEL_PATH)
    feat_cols, medians = payload["feature_cols"], payload["feature_medians"]
    df = _p.load_btc_frame(feat_cols)
    ts = df["timestamp"]
    fit = ((ts >= FIT_START) & (ts <= FIT_END)).to_numpy()
    ev = ((ts >= EVAL_START) & (ts <= EVAL_END)).to_numpy()
    log(f"XRP canonical {len(df):,}행 | 학습 {int(fit.sum()):,} / 평가 {int(ev.sum()):,}")
    log(f"⚠️{NEVER_READ_FROM.date()} 이후 {int((ts >= NEVER_READ_FROM).sum()):,}봉은 읽지 않는다"
        " (승격확인창·소진 Phase3창·HOLDOUT)")
    log(f"시드 {len(SEEDS)}종 (랜덤 추출): {SEEDS}")

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))
    ys = {n: _c.make_label(df["close"], fit, s, k) for n, (s, k) in LABELS.items()}

    # Phase 3b용 증거신호 프레임 (라벨/시드와 무관하게 1회만)
    raw = pd.read_csv(_p.XRP_KLINES, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    partner = pd.read_csv(_p.PARTNER_KLINES, usecols=["timestamp", "high", "low"],
                          parse_dates=["timestamp"])
    frame = compute_signals(raw, btc_df=partner, funding_df=_p.load_xrp_funding_z())
    pivots = _p.build_xrp_pivots()
    ts_e = frame["timestamp"]
    windows = {"VAL": ((ts_e >= _p.EV_VAL_START) & (ts_e <= _p.EV_VAL_END)).to_numpy(),
               "OOS": ((ts_e >= _p.EV_OOS_START) & (ts_e <= _p.EV_OOS_END)).to_numpy()}
    pivot_pos = {s: frame.index[frame["timestamp"].isin(
        pivots.loc[pivots["pivot_type"] == s, "timestamp"])].to_numpy() for s in ("bottom", "top")}

    def gate_both_positive(pred_full):
        merged = frame[["timestamp"]].merge(
            pd.DataFrame({"timestamp": ts, "pred": pred_full}), on="timestamp", how="left")
        chop = (merged["pred"].to_numpy() == 2)
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
            return 0, 0, float("nan")
        pv = d.pivot_table(index=["s", "side"], columns="w", values="imp")
        if "VAL" not in pv or "OOS" not in pv:
            return 0, int(len(pv)), float("nan")
        return int(((pv["VAL"] > 0) & (pv["OOS"] > 0)).sum()), int(len(pv)), float(pv["OOS"].mean())

    log("")
    log(f"{'시드':>9} {'라벨':<9} {'bal_acc':>8} {'chop_R':>8} {'flip':>8} {'3b양쪽창':>9} {'3b OOS':>9}")
    rows = []
    for sd in SEEDS:
        per = {}
        for lname, y in ys.items():
            m = HistGradientBoostingClassifier(random_state=sd, **_p.GBM3_HP).fit(
                x.loc[fit, feat_cols], y[fit])
            r = _p.evaluate(y[ev], m.predict(x.loc[ev, feat_cols]))
            bp, nc, oos = gate_both_positive(m.predict(x[feat_cols]))
            per[lname] = {"bal_acc": r["balanced_accuracy"], "chop_recall": r["chop_recall"],
                          "flip_rate": r["flip_rate"], "gate_both_positive": bp,
                          "gate_cells": nc, "gate_oos_mean": oos}
            log(f"{sd:>9} {lname:<9} {r['balanced_accuracy']:>8.4f} {r['chop_recall']:>8.4f} "
                f"{r['flip_rate']:>8.4f} {bp:>5}/{nc:<3} {oos:>+9.4f}")
        rows.append({"seed": sd, **{k: v for k, v in per.items()}})
        log("")

    log("=" * 74)
    log("판정 (사전 고정: S96_K9가 모든 시드에서 bal_acc 우위여야 한다)")
    log("=" * 74)
    wins = {"bal_acc": 0, "flip": 0, "gate": 0}
    for r in rows:
        a, b = r[DEPLOYED], r[PREVIOUS]
        wins["bal_acc"] += a["bal_acc"] > b["bal_acc"]
        wins["flip"] += a["flip_rate"] < b["flip_rate"]
        wins["gate"] += a["gate_both_positive"] > b["gate_both_positive"]
    n = len(rows)
    for k, lab in (("bal_acc", "bal_acc 우위"), ("flip", "플리커 우위(낮음)"),
                   ("gate", "3b 양쪽창 우위")):
        log(f"  {lab:<18} {wins[k]}/{n} 시드  {'✅전부' if wins[k] == n else '⚠️일부 뒤집힘'}")
    ba = [(r[DEPLOYED]["bal_acc"], r[PREVIOUS]["bal_acc"]) for r in rows]
    d = [x - y for x, y in ba]
    log(f"  bal_acc 차이: 평균 {np.mean(d):+.4f}  최소 {min(d):+.4f}  최대 {max(d):+.4f}")
    ok = wins["bal_acc"] == n
    log("")
    log(f"⇒ {'✅**시드 견고 -- 배포 결정 유효**' if ok else '⚠️**시드에 흔들림 -- 배포 결정 재검토 필요**'}")

    rep = {"seeds": SEEDS, "seed_selection": "랜덤 추출(고정 간격 증가 아님)",
           "n_seeds": n, "fit": [str(FIT_START), str(FIT_END)],
           "eval": [str(EVAL_START), str(EVAL_END)], "never_read_from": str(NEVER_READ_FROM),
           "labels": {k: {"scale": v[0], "debounce_k": v[1]} for k, v in LABELS.items()},
           "deployed": DEPLOYED, "previous": PREVIOUS, "per_seed": rows,
           "wins": wins, "bal_acc_delta": {"mean": float(np.mean(d)), "min": float(min(d)),
                                           "max": float(max(d))},
           "seed_robust": bool(ok), "holdout_touched": False,
           "runtime_sec": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
