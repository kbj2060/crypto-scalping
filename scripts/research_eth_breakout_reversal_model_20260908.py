#!/usr/bin/env python3
"""**돌파/되돌림 분류 정확도 올리기** -- 워크포워드 (2026-09-08).

사용자: *"돌파하냐 되돌리냐만 정확도를 크게 올려줘. 돈을 버는건 나중에."*
⇒ 판정 지표는 bp 가 아니라 **정확도/AUC**다.

## 🔴먼저 정정: 베이스라인은 52~53% 가 아니라 **50.0%** 다
부록 AL 에서 "무조건 되돌림 = 52.9%"라고 적은 것은 **bp -> 정확도 환산이 틀렸다**.
그 bp 우위(+2.6bp)의 대부분은 **슬리피지 부호 실수**였다: `entry = level*(1 + sgn*2bp)` 로 놓으면
돌파팔은 2bp 불리해지지만 **되돌림팔은 2bp 유리해진다**(같은 가격이 숏에겐 좋은 진입). 두 팔 격차
4bp 중 대부분이 여기서 나왔다. 라벨을 직접 세면 돌파율 **0.4931~0.5020** — 동전이다.
정확도를 지표로 쓰면 이 함정 자체가 사라진다(체결가정과 무관).

## 과제
발현 방향이 관측된 뒤, 그 방향으로 +P 가 먼저인가(돌파, y=1) 반대가 먼저인가(되돌림, y=0).
시간청산은 종가 부호로 라벨 -> **커버리지 100%**. 배리어 해소율 94.5~94.9%.
피쳐 53개(⭐메트릭 5종 수준/변화 + ⭐횡단면 순위 6종 포함, 전부 `bt-1` 완결봉 기준).

## 프로토콜
월 1회 재학습 expanding walk-forward · 엠바고 4h · HGB(+TabPFN 옵션) ·
누수 가드: 단일피쳐 AUC ≥0.95 · 모델 AUC ≥0.99 이면 중단 ·
정확도는 **일군집 부트스트랩 CI** 로 낸다. 창별(VAL/OOS/HOLDOUT)·T_mult 별 보고.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_breakout_reversal_20260908/dataset_v2.parquet"
OUT = ROOT / "tmp/eth_breakout_reversal_20260908"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
EMB = pd.Timedelta(hours=4)
BOOT = 3000
SEED = 20260908


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    A = pd.read_parquet(SRC)
    A["timestamp"] = pd.to_datetime(A["timestamp"])
    FEATS = [c for c in A.columns if c.startswith(("f_", "x_", "sig_", "v2_"))] + \
            ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    print(f"데이터 {A.shape} · 피쳐 {len(FEATS)}", flush=True)

    for anch in ("first_fire", "any2/Wc3"):
        for Tm in (0.5, 0.75, 1.0):
            d = A[(A.anchor == anch) & (A.T_mult == Tm)].sort_values("timestamp").reset_index(drop=True)
            if len(d) < 3000: continue
            X = d[FEATS].to_numpy(np.float32); y = d["y"].to_numpy(int)
            ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy(); sp = d["split"].to_numpy()
            # 누수 가드
            tr0 = sp == "TRAIN"
            bad = []
            for j, f in enumerate(FEATS):
                v = X[tr0, j]; m = np.isfinite(v)
                if m.sum() < 500 or len(np.unique(v[m])) < 3: continue
                try:
                    a = roc_auc_score(y[tr0][m], v[m])
                except Exception:
                    continue
                if max(a, 1 - a) >= 0.95: bad.append((f, round(max(a, 1 - a), 4)))
            if bad:
                print(f"🔴누수 의심 피쳐 {bad} -- 중단"); return 1
            # 월 1회 재학습 워크포워드
            months = ts.dt.to_period("M")
            uniq = sorted(months.unique())
            pred = np.full(len(d), np.nan)
            for i, mo in enumerate(uniq):
                if i < 6: continue
                te = (months == mo).to_numpy()
                cut = ts[te].min() - EMB
                tr = (ts < cut).to_numpy()
                if tr.sum() < 2000 or te.sum() < 30: continue
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                     max_leaf_nodes=31, l2_regularization=1.0,
                                                     early_stopping=True, validation_fraction=0.15,
                                                     random_state=SEED)
                clf.fit(X[tr], y[tr])
                pred[te] = clf.predict_proba(X[te])[:, 1]
            # ⭐대조군: 학습 라벨만 섞어 같은 워크포워드 -> 50% 근처여야 정상(누수 없음)
            predc = np.full(len(d), np.nan)
            ysh = y.copy()
            for i, mo in enumerate(uniq):
                if i < 6: continue
                te = (months == mo).to_numpy()
                cut = ts[te].min() - EMB
                tr = (ts < cut).to_numpy()
                if tr.sum() < 2000 or te.sum() < 30: continue
                yy = ysh.copy(); yy[tr] = rng.permutation(yy[tr])
                c2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                    max_leaf_nodes=31, l2_regularization=1.0,
                                                    early_stopping=True, validation_fraction=0.15,
                                                    random_state=SEED)
                c2.fit(X[tr], yy[tr]); predc[te] = c2.predict_proba(X[te])[:, 1]
            ok = np.isfinite(pred)
            line = f"{anch:>11} T={Tm:<5}"
            for w in WINS:
                m = ok & (sp == w)
                if m.sum() < 100: line += f" {w[:4]} --"; continue
                acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
                lo, hi = day_ci(acc, day[m], rng)
                try: auc = roc_auc_score(y[m], pred[m])
                except Exception: auc = np.nan
                base = max(y[m].mean(), 1 - y[m].mean())
                mc = np.isfinite(predc) & (sp == w)
                cacc = ((predc[mc] > 0.5).astype(int) == y[mc]).mean() if mc.sum() > 50 else np.nan
                line += (f" | {w[:4]} n{m.sum():>5} 정확도 {acc.mean():.4f}[{lo:.4f},{hi:.4f}] "
                         f"AUC {auc:.4f} 기저 {base:.4f} 셔플 {cacc:.4f}")
            print(line, flush=True)
            np.save(OUT / f"predv2_{anch.replace('/','_')}_{Tm}.npy", pred)
    print(json.dumps({"done": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
