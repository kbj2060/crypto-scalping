#!/usr/bin/env python3
"""**적응형 스톱폭 규칙** — 고정 ATR 배수 vs 조건부 분위수 (2026-09-07, 서버 GPU).

사용자: *"스톱폭 규칙으로 만들어줘"*

## 근거
`research_eth_excursion_vs_best_vol_20260907.py` 결과:
  15분 이탈폭 예측 `atr_pct` 단독 0.578 → **변동성 계열 조합 0.747** → 전체150 0.750
  (전체−계열 증분 0/3 창 -- 변동성 계열 밖의 정보는 없다)
⇒ 새 매매 축은 아니지만 **스톱폭 추정에는 실질 개선**이다. 배포는 `atr_pct` 하나를 쓴다.

## 규칙 설계
스톱은 방향이 있다. 진입이 **지속 방향**이므로 역행(adverse) 이탈폭만 본다:
    롱(천장앵커): adv = 1 - min(low)/entry      숏(바닥앵커): adv = max(high)/entry - 1
    adv_atr = adv / atr_pct[t]                  ← ATR 단위
스톱폭 w(ATR 배수)를 놓으면 **손절 = adv_atr > w**.

목표 손절률 p 에 대해 필요한 것은 **조건부 분위수** q_{1-p}(adv_atr | x) 다.
CDF 를 K 격자에서 분류기로 추정해 역산한다:
    K ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0} 각각에 P(adv_atr > K | x) 를 적합
    → 그 곡선에서 P = p 가 되는 K 를 선형보간 = 그 앵커의 스톱폭
피쳐는 **변동성 계열만** 쓴다(전체150 이 증분 0 이므로 표면적을 줄인다).

## 비교 (판정 잣대 = AUC 가 아니다)
  현행  w = 상수 K0. TRAIN 에서 실현 손절률이 정확히 p 가 되도록 K0 를 잡는다(공정 비교).
  모델  w_i = 앵커별 조건부 분위수.
평가창에서 **(1) 실현 손절률이 p 에 맞는가(캘리브레이션) (2) 같은 손절률에서 평균 스톱폭이
더 좁은가** 를 본다. 좁은 스톱 = 같은 리스크로 더 큰 사이즈 = 더 나은 R.

⚠️이건 청산 파라미터 추정이지 **진입 신호가 아니다**. 방향은 여전히 0.53 이다.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_anchor_features154_20260907"
OUT = ROOT / "tmp/eth_adaptive_stop_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H_GRID = (12, 48)                      # 1시간 · 4시간(섀도우 실보유 상한)
# ⚠️H=48 은 역행/ATR 중앙이 2.97, 90분위가 8.25 다. 상한 4.0 격자로는 p=10% 분위수를 못 잡아
#   곡선이 클램프되고 캘리브레이션이 깨진다(목표 10% → 실현 37%). 상한을 12 로 넓힌다.
K_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 12.0)
P_TARGETS = (0.10, 0.20, 0.30)
SEED, N_EST, BOOT = 20260907, 4, 800
VOL_PAT = re.compile(r"vol|atr|range|width|parkinson|garch|rv_|realized|bb_|std|sigma", re.I)


def day_boot(fn, d, rng, B=BOOT):
    u = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        o.append(fn(i))
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    import importlib.util, torch
    from tabpfn import TabPFNClassifier
    s = importlib.util.spec_from_file_location("ab", ROOT / "scripts/build_eth_anchor_label_dataset_20260907.py")
    B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    D = pd.read_parquet(SRC / "features154.parquet")
    meta = json.loads((SRC / "meta.json").read_text())
    cols = [c for c in meta["feature_cols"] if c in D.columns]
    volc = [c for c in cols if VOL_PAT.search(c)]
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    atr = D["atr_pct"].to_numpy(float)
    side = D["side"].to_numpy()
    kl = B._load_kl(B.ETH_KL)
    pos = pd.Series(np.arange(len(kl)), index=pd.to_datetime(kl["timestamp"]))
    idx = pos.reindex(D["timestamp"]).to_numpy().astype(int)
    O = kl["open"].to_numpy(float); HI = kl["high"].to_numpy(float); LO = kl["low"].to_numpy(float)
    Xv = D[volc].to_numpy(np.float64)
    print(f"[입력] 앵커 {len(D):,} · 변동성 피쳐 {len(volc)} · {dev}", flush=True)

    out_rows = []
    for H in H_GRID:
        # 역행 이탈폭 (지속 방향 진입 기준): 천장앵커=롱→아래가 역행, 바닥앵커=숏→위가 역행
        adv = np.full(len(D), np.nan)
        for j, i in enumerate(idx):
            a, b = i + 1, i + 1 + H
            if b > len(kl) or not np.isfinite(atr[j]) or atr[j] <= 0: continue
            entry = O[a]
            adv[j] = ((1.0 - LO[a:b].min() / entry) if side[j] == "top"
                      else (HI[a:b].max() / entry - 1.0)) / atr[j]
        ok = np.isfinite(adv); tr = ok & (sp == "TRAIN")
        print(f"\n{'='*104}\nH={H}봉({H*5}분) · 유효 {ok.sum():,} · 역행/ATR 중앙 {np.nanmedian(adv):.3f} "
              f"· 90분위 {np.nanpercentile(adv[ok],90):.3f}", flush=True)

        # CDF: 각 K 에서 P(adv > K | x)
        PK = np.full((len(D), len(K_GRID)), np.nan)
        for c, K in enumerate(K_GRID):
            yk = (adv > K).astype(float)
            t = tr & np.isfinite(yk)
            if len(np.unique(yk[t])) < 2: continue
            clf = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                                   ignore_pretraining_limits=True, memory_saving_mode=True)
            clf.fit(np.nan_to_num(Xv[t]).astype(np.float32), yk[t].astype(int))
            m = ok & np.isin(sp, WINS + ("TRAIN",))
            PK[m, c] = clf.predict_proba(np.nan_to_num(Xv[m]).astype(np.float32))[:, 1]
        # 단조화 (P(adv>K) 는 K 에 대해 비증가여야 한다)
        PK = np.minimum.accumulate(np.where(np.isfinite(PK), PK, 1.0), axis=1)

        def q_of(p):
            """P(adv>K)=p 가 되는 K 를 선형보간. 곡선이 p 아래로 안 내려가면 최대 K."""
            w = np.full(len(D), np.nan)
            for j in range(len(D)):
                v = PK[j]
                if not np.isfinite(v).all(): continue
                if v[0] <= p: w[j] = K_GRID[0]; continue
                if v[-1] >= p: w[j] = K_GRID[-1]; continue
                c = int(np.argmax(v < p))
                v0, v1 = v[c - 1], v[c]; k0, k1 = K_GRID[c - 1], K_GRID[c]
                w[j] = k0 + (v0 - p) / max(v0 - v1, 1e-9) * (k1 - k0)
            return w

        for p in P_TARGETS:
            wm = q_of(p)
            # 현행: TRAIN 실현 손절률이 정확히 p 가 되는 상수 K0
            K0 = float(np.nanquantile(adv[tr], 1 - p))
            rec = {"H": H, "p_target": p, "K0_fixed": round(K0, 4)}
            line = f"  목표손절 {p:.0%} · 현행상수 K0={K0:.3f}"
            for w in WINS:
                te = ok & (sp == w) & np.isfinite(wm)
                if te.sum() < 40: continue
                so_f = float((adv[te] > K0).mean())          # 현행 실현 손절률
                so_m = float((adv[te] > wm[te]).mean())       # 모델 실현 손절률
                width_f, width_m = K0, float(np.mean(wm[te]))
                # 같은 손절률로 맞춘 뒤 폭 비교: 모델 폭을 상수배 스케일해 손절률을 p 로 재보정
                # ⚠️이 재보정은 **평가창 자체**로 스케일을 잡는다 -- 모델에 유리한 쪽 편향이다.
                #   그럼에도 모델이 지면 그 결론은 더 강하다.
                sc = np.nanquantile(adv[te] / np.maximum(wm[te], 1e-9), 1 - p)
                width_m_matched = float(np.mean(wm[te] * sc))
                lo, hi = day_boot(lambda i: float(np.mean(wm[te][i] * sc)) - width_f, day[te], rng)
                rec[f"{w}_so_fixed"], rec[f"{w}_so_model"] = so_f, so_m
                rec[f"{w}_calib_bad"] = bool(abs(so_m - p) > 0.10)   # 캘리브레이션 붕괴 표시
                rec[f"{w}_width_fixed"], rec[f"{w}_width_model"] = width_f, width_m_matched
                rec[f"{w}_gain_pct"] = (width_f - width_m_matched) / width_f * 100
                rec[f"{w}_lo"], rec[f"{w}_hi"] = lo, hi
                line += (f"\n     {w:<14} 손절률 현행 {so_f:.3f} / 모델 {so_m:.3f}"
                         f" · 폭 현행 {width_f:.3f} → 모델 {width_m_matched:.3f} ATR"
                         f"  ({rec[f'{w}_gain_pct']:+.1f}%)  [{lo:+.3f},{hi:+.3f}]")
            out_rows.append(rec)
            print(line, flush=True)

    A = pd.DataFrame(out_rows); A.to_csv(OUT / "stop_rule.csv", index=False)
    print("\n" + "=" * 104, flush=True)
    print("요약 — 같은 손절률에서 스톱이 좁아지는가", flush=True)
    print("  ⚠️gain = (현행폭 − 모델폭)/현행폭 · **양수 = 모델이 좁다 = 이득** · 음수 = 모델이 넓다 = 손해",
          flush=True)
    print("  ⚠️CI 는 (모델폭 − 현행폭) 이므로 **CI 상한 < 0 이어야 모델 우세**", flush=True)
    print("=" * 104, flush=True)
    for _, r in A.iterrows():
        gains = [r.get(f"{w}_gain_pct", np.nan) for w in WINS]
        sig = sum(1 for w in WINS if np.isfinite(r.get(f"{w}_hi", np.nan)) and r[f"{w}_hi"] < 0)
        print(f"  H={int(r.H):>2}봉 p={r.p_target:.0%}  폭 절감 " +
              " / ".join(f"{g:+.1f}%" for g in gains) +
              f"   모델우세(CI<0) {sig}/3 {'✅' if sig >= 2 else '❌'}", flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
