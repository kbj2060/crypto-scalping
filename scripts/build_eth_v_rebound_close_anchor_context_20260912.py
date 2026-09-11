#!/usr/bin/env python3
"""배포 V자 동결 컨텍스트를 **체결 가능한 앵커(발동봉 종가)** 판으로 만든다 + 임계 재보정.

사용자 지시(2026-09-12, 옵션 a): "종가 앵커로 배포본 교체(임계 재보정 포함)".
근거: `research_eth_v_rebound_close_anchor_retrain_20260912.py` -- 앵커 교체로 라벨률은
1.9배 정직해지는데(15.2%->8.2%) 모델 AUC 는 0.013~0.019 만 내려가고 단일피쳐 대비
순증분은 같거나 커졌다.

## 왜 피쳐를 다시 안 만드나
배포는 18,000행 동결 컨텍스트를 TabPFN in-context 학습에 쓴다. 그 행들은 every-bar TRAIN
182,969행에서 **균등 무작위 추출**된 표본이다(context_report.json). 앵커만 바꾸는 일이므로
**같은 행·같은 피쳐·같은 표집을 그대로 두고 label 열만 교체**하면 모집단이 보존된다.
피쳐 프레임을 다시 만들면 오히려 표집이 달라져 비교가 깨진다.

## 임계 재보정
기저가 15.15% -> 새 값으로 바뀌므로 확률의 의미가 달라진다(09-01 context_report 의 caveat).
**커버리지(발동 빈도)를 현행과 맞추는** 임계를 고른다 -- 화면이 켜지는 빈도를 유지한 채
라벨만 정직해지게 하는 선택이다. 정밀도는 그 결과로 보고한다.
확률은 OOF(5-fold) GBM 으로 낸다 -- 라이브 TabPFN in-context 의 프록시(09-01 실측 0.6953 vs
0.6942). ⚠️TRAIN 창 표본 위 추정이라 라이브 분포와 완전히 같지는 않다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/kbj20/crypto-scalping")
SRC = ROOT / ("data/labels/eth_5m_v_rebound_every_bar_20260901/"
              "tabpfn_train_context_frozen_every_bar_20260901.csv")
OUT_DIR = ROOT / "data/labels/eth_5m_v_rebound_close_anchor_20260912"
CUR_THRESHOLD = 0.60


def load_builder():
    p = HERE / "research_eth_v_rebound_close_anchor_retrain_20260912.py"
    spec = importlib.util.spec_from_file_location("R", p)
    m = importlib.util.module_from_spec(spec); sys.modules["R"] = m; spec.loader.exec_module(m)
    return m


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    R = load_builder()
    kl = R.load_klines()
    lab = R.build_labels(kl)

    agree = R.parity_gate(lab)                     # 🔴같은 게이트를 다시 통과시킨다
    if agree < 0.99:
        print(f"🔴 파리티 실패 {agree:.4f} -- 중단"); return 1

    F = pd.read_csv(SRC, parse_dates=["timestamp"])
    if F["timestamp"].dt.tz is None:
        F["timestamp"] = F["timestamp"].dt.tz_localize("UTC")
    M = F.merge(lab[["timestamp", "y_close_down", "y_close_up", "y_extreme_down", "y_extreme_up"]],
                on="timestamp", how="left")
    dn = M.is_downside.to_numpy() == 1
    y_close = np.where(dn, M.y_close_down, M.y_close_up)
    y_extreme = np.where(dn, M.y_extreme_down, M.y_extreme_up)
    assert np.isfinite(y_extreme).all(), "극점 라벨 결측 -- 조인 실패"
    assert float((y_extreme == M.label.to_numpy()).mean()) == 1.0, "동결 label 과 불일치"
    keep = np.isfinite(y_close)
    print(f"[컨텍스트] {len(M):,}행 · 종가라벨 유효 {int(keep.sum()):,} "
          f"· 라벨률 극점 {y_extreme.mean():.4f} -> 종가 {y_close[keep].mean():.4f}")

    out = F.loc[keep].copy()
    out["label"] = y_close[keep]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / "tabpfn_train_context_frozen_close_anchor_20260912.csv"
    out.to_csv(dst, index=False)

    # ── 임계 재보정 ─────────────────────────────────────────────────────────────
    X = out[R.FEATURES].to_numpy(float)
    res = {}
    for tag, y in (("extreme", y_extreme[keep]), ("close", y_close[keep])):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=20260912).split(X, y.astype(int)):
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, early_stopping=True,
                                               validation_fraction=0.15, random_state=20260912)
            m.fit(X[tr], y[tr].astype(int))
            p[te] = m.predict_proba(X[te])[:, 1]
        res[tag] = (p, y)
        print(f"  OOF AUC {tag:8s} {R.auc(p, y):.4f} · 기저 {y.mean():.4f}")

    p_ex, y_ex = res["extreme"]
    cov_now = float((p_ex >= CUR_THRESHOLD).mean())
    prec_now = float(y_ex[p_ex >= CUR_THRESHOLD].mean()) if (p_ex >= CUR_THRESHOLD).any() else float("nan")
    p_cl, y_cl = res["close"]
    new_thr = float(np.quantile(p_cl, 1 - cov_now))          # 커버리지 일치
    sel = p_cl >= new_thr
    prec_new = float(y_cl[sel].mean()) if sel.any() else float("nan")
    print(f"\n[재보정] 현행 임계 {CUR_THRESHOLD:.2f} · 커버리지 {cov_now:.2%} · 정밀도 {prec_now:.3f}(기저 {y_ex.mean():.3f})")
    print(f"         **새 임계 {new_thr:.3f}** · 커버리지 {sel.mean():.2%} · 정밀도 {prec_new:.3f}(기저 {y_cl.mean():.3f})")
    print(f"         리프트 현행 {prec_now/y_ex.mean():.2f}배 -> 신규 {prec_new/y_cl.mean():.2f}배")
    print("\n  참고: 임계별 커버리지·정밀도(종가 라벨)")
    for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
        s = p_cl >= t
        if s.sum() < 20: continue
        print(f"    {t:.2f}  커버 {s.mean():>6.2%}  정밀도 {y_cl[s].mean():.3f}  리프트 {y_cl[s].mean()/y_cl.mean():.2f}배")

    rep = {"source": str(SRC), "parity_vs_frozen": agree, "n": int(keep.sum()),
           "label_rate": {"extreme": float(y_extreme.mean()), "close": float(y_close[keep].mean())},
           "oof_auc": {k: R.auc(*res[k]) for k in res},
           "threshold": {"current": CUR_THRESHOLD, "recommended": new_thr,
                         "coverage_matched": cov_now,
                         "precision": {"extreme@0.60": prec_now, "close@new": prec_new}},
           "caveat": "OOF GBM 프록시 · TRAIN 창 표본 위 추정. 라이브 TabPFN in-context 분포와 동일하지 않다."}
    (OUT_DIR / "context_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"\n산출물 {dst}\n        {OUT_DIR/'context_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
