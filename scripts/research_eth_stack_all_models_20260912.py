#!/usr/bin/env python3
"""**모든 재료를 한 모델에 쌓으면** 베이스라인을 이기는가 (2026-09-12).

사용자: *"이때까지 진행한 모든 이더리움 모델의 테스트들을 모아서 조합해보면 어떨까."*

## 무엇을 쌓나 — 이미 한 프레임에 있다
`build_eth_signal_trigger_material_panel_20260912` 62열 = 증거신호 8 · 이벤트 트리거 7(+연속값)
· 문맥 29(ATR분위·ret/pos/dist 3창·BTC·시각) · 레짐 2. 여기에 **극점 탐지기 확률**
(누수 없이 재산출한 판, `build_eth_extreme_prob_cleancut_20260912`)을 붙인다.

## 🔴베이스라인이 이 검정의 전부다
이 저장소는 «모델이 뭔가 한다»가 아니라 **«모델이 공짜 기준선을 이기는가»** 에서 계속 졌다:
  · 크기: **ATR(96) 단독 rho +0.465 > 오메가 모델 +0.303**, 0/6개월 우세(2026-09-11)
  · 방향: 어디서 재도 0.50 (4,000셀 · 11,934규칙 · 문헌 처방 3종 전부)
그래서 «다 쌓은 모델» 을 **ATR 단독**(크기)과 **동전**(방향)에 직접 붙인다. 이기지 못하면
조합의 문제가 아니라 **재료에 그 정보가 없다**는 뜻이다.

## 규약
전진 검증: 매월 재적합, 그 달만 평가(확장 창). 라벨은 봉 i 종가 이후만 본다
(피쳐 i · 진입 i+1 시가 · 「사건 라벨 경계 계약」).
  크기 목표  |다음 H봉 수익|  → 스피어만 rho (순위 문제다)
  방향 목표  다음 H봉 수익 부호 → 정확도(그 달 기저 병기)
H = 12(1시간) · 48(4시간). 모델 = HistGradientBoosting(빠르고 이 저장소 표준).

자체점검 --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import research_eth_rule_direction_probability_20260912 as RD  # noqa: E402

HS = (12, 48)
WARM = 900
MIN_TRAIN = 20_000


def build() -> tuple[pd.DataFrame, np.ndarray]:
    p = pd.read_parquet(RD.PANEL)
    X = p.drop(columns=[c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in p])
    # 극점 확률(누수 없는 판) — 측면별로 붙인다. 발동 안 한 봉은 NaN(모델이 결측을 직접 다룬다).
    if RD.XPROB.exists():
        q = pd.read_parquet(RD.XPROB)
        pos = pd.Series(np.arange(len(p)), index=pd.DatetimeIndex(p["timestamp"]))
        for lg, nm in ((1, "xp_bottom"), (0, "xp_top")):
            sub = q[q["long"] == lg]
            i = pos.reindex(pd.DatetimeIndex(sub["timestamp"].to_numpy())).to_numpy()
            ok = np.isfinite(i)
            col = np.full(len(p), np.nan)
            col[i[ok].astype(int)] = sub["p"].to_numpy()[ok]
            X[nm] = col
    return X, p["timestamp"].to_numpy()


def targets(H: int) -> tuple[np.ndarray, np.ndarray]:
    p = pd.read_parquet(RD.PANEL)
    op = p["open"].to_numpy(float); cl = p["close"].to_numpy(float)
    n = len(p)
    ent = np.roll(op, -1); ent[-1] = np.nan
    fwd = np.full(n, np.nan)
    fwd[: n - H - 1] = cl[H + 1 : n] / ent[: n - H - 1] - 1.0
    return np.abs(fwd), np.sign(fwd)


def walk(X: pd.DataFrame, ts: np.ndarray, y: np.ndarray, kind: str, base: np.ndarray) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from scipy.stats import spearmanr
    months = pd.PeriodIndex(pd.to_datetime(ts), freq="M")
    uniq = [m for m in months.unique() if m >= pd.Period("2024-07", "M")]
    rows = []
    Xv = X.to_numpy(np.float32)
    for m in uniq:
        te = (months == m) & np.isfinite(y)
        tr = (months < m) & np.isfinite(y)
        tr[np.arange(len(tr)) < WARM] = False
        if tr.sum() < MIN_TRAIN or te.sum() < 200:
            continue
        if kind == "size":
            mdl = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_depth=6,
                                                random_state=0)
            mdl.fit(Xv[tr], y[tr])
            pr = mdl.predict(Xv[te])
            r_model = float(spearmanr(pr, y[te]).statistic)
            r_base = float(spearmanr(base[te], y[te]).statistic)
        else:
            yb = (y > 0).astype(int)
            mdl = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=6,
                                                 random_state=0)
            mdl.fit(Xv[tr], yb[tr])
            pr = mdl.predict(Xv[te])
            r_model = float((pr == yb[te]).mean())
            r_base = float(max(yb[te].mean(), 1 - yb[te].mean()))   # 그 달 다수 클래스
        rows.append({"month": str(m), "n": int(te.sum()), "model": r_model, "base": r_base,
                     "delta": r_model - r_base})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        p = pd.read_parquet(RD.PANEL)
        m, s = targets(12)
        cl = p["close"].to_numpy(float); op = p["open"].to_numpy(float)
        i = 5000
        assert abs(m[i] - abs(cl[i + 13] / op[i + 1] - 1)) < 1e-12, "라벨은 i+1 시가 → i+13 종가"
        assert np.isnan(m[len(p) - 5]), "끝에서는 라벨이 없다"
        X, ts = build()
        assert "timestamp" not in X.columns and "close" not in X.columns, "가격 원열이 피쳐에 남으면 안 된다"
        print(f"selftest OK — 라벨 경계(i+1 시가 기준) · 피쳐 {X.shape[1]}열(가격 원열 제외)")
        return 0

    X, ts = build()
    p = pd.read_parquet(RD.PANEL)
    atr = p["atr_pct"].to_numpy(float)
    print(f"피쳐 {X.shape[1]}열 × {len(X):,}행  (증거신호·트리거·문맥·레짐·극점확률)\n")
    for H in HS:
        mag, sgn = targets(H)
        for kind, y, base, blabel in (("size", mag, atr, "ATR 단독"),
                                      ("dir", sgn, atr, "그 달 다수 클래스")):
            d = walk(X, ts, y, kind, base)
            if not len(d):
                continue
            metric = "스피어만 rho" if kind == "size" else "정확도"
            win = int((d.delta > 0).sum())
            print(f"H={H:>2} · {'크기' if kind=='size' else '방향'} ({metric}) — 전진 {len(d)}개월")
            print(f"   모델 중앙 {d.model.median():.4f} · 기준선({blabel}) 중앙 {d.base.median():.4f} · "
                  f"**증분 중앙 {d.delta.median():+.4f}** · 모델 우세 **{win}/{len(d)}개월**")
            t = d.delta.mean() / (d.delta.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            print(f"   월별 증분 t = {t:+.2f}   최악/최고 {d.delta.min():+.4f} / {d.delta.max():+.4f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
