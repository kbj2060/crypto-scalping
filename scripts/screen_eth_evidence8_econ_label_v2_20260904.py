#!/usr/bin/env python3
"""증거신호 경제성 재라벨링 v2 -- **라벨 로직 4종 + 선별력 직접 측정** (ETH 8종).

## 왜 이걸 먼저 하나

BTC는 이 작업(경제성 라벨 이식)을 **이미 4번 시도해 전부 실패**했고, 원인은 라벨이 아니라
**입력 피쳐에 방향 정보가 없어서**였다(`btc_v_rebound_econ_label_closed_no_direction_skill_20260902`):
1시간봉으로 비용/ATR을 62%→15.1%까지 낮춰 비용장벽을 실제로 제거했는데도 OOS AUC가
0.4932로 5분봉(0.4912)과 똑같았다. ⇒ **라벨을 아무리 잘 정의해도 Tier0가 그 결과를 예측
못 하면 끝난다.** 그래서 TabPFN·관문·8종 확장 같은 비싼 걸 짓기 전에 **GBM 프록시 AUC로
몇 분 만에 죽일 수 있는지부터** 본다.

## 무엇을 재나

  모집단   liquidity_sweep **인과적 첫 발동**(cluster_dedup 앵커 금지 -- 5.16절 A버그).
           `fire[i] AND 직전 GAP봉에 발동 없음` -- 뒤만 보므로 인과적이다.
  방향     bottom fire -> 롱, top fire -> 숏
  진입     o[i+1] (시장가)
  라벨     **net_bp > 0** (자유도 0). 중간지대 제외 안 함 -- 이 저장소에서 "kept-only 착시"가
           3번 재발했고 V자반등 칩을 죽인 구조다.
  청산     ⭐두 규약을 나란히 만든다. **청산 규칙이 라벨의 질문 자체를 바꾸기 때문이다**:
             시간청산(HORIZON=30봉) -> "150분 뒤 가격이 위인가"      = 순수 방향 예측
             트레일링               -> "일찍 유리하게 움직였는가"     = 경로 형태 + 방향
           V자반등이 성공한 건 트레일링이었고, BTC가 실패한 건 방향력 부재였다.
           ⚠️트레일링 셀은 **진단 전용**이다(승격 주장 아님). 저ARM 노이즈수확 구간을 피해
           ARM>=1.0만 쓴다(`feedback_trailing_stop_low_arm_noise_harvest_artifact_20260901`).

  ⭐무작위 진입 기준선도 같이 잰다 -- BTC 메모가 "ETH는 무작위 기준선이 양수라 약한 선택력
  으로도 이익이 났다"고 지목했다. 발동 모집단 자체가 이미 양수면, 모델 성과의 상당 부분이
  선택력이 아니라 기준선일 수 있다.

## 판정

  VAL AUC ~= 0.50  ->  BTC와 같은 결론. **여기서 종료**(TabPFN도 관문도 불필요).
  VAL AUC 유의미   ->  TabPFN + 정식 관문(A1/A2) + 8종 확장으로 진행.

⚠️TRAIN/VAL만 쓴다. **OOS·HOLDOUT 미터치.**
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_pf = _load("pf_screen", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0 = _pf.TIER0
sim_exit = _pf.sim_exit

# live_evidence_signal_metalabel_20260829.py::METALABEL_SIGNALS의 확립된 horizon_bars.
# ⚠️신호마다 다르다 -- 이게 "자유도 0"의 근거다(임의로 고른 값이 아니라 이미 확립된 신호 고유값).
SIGNALS = {"liquidity_sweep": 30, "taker_delta_z_climax": 24, "short_term_return_z": 12,
           "orthogonal_combo": 24, "smt_divergence": 72, "fib_extension_exhaustion": 20,
           "demarker_extreme": 8, "kalman_deviation_meanrev": 12}
GAP = 12                # 인과적 첫 발동 판정용 고정값(뒤만 봄). 신호별 metalabel GAP과 무관한
                        # 우리 자신의 dedup 규약이므로 8종에 동일하게 적용한다.
COST_BP = 10.0          # 사전등록. V자반등 판정과 동일 기준(실측 메이커 8.11bp는 참고치).
TRAIL_CELLS = [(3.0, 1.5, 0.1), (4.0, 1.0, 0.1)]   # 진단 전용, ARM>=1.0

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")     # ⚠️여기서 끊는다 -- OOS 미터치
START = pd.Timestamp("2024-01-01", tz="UTC")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20260904
OUT = ROOT / "data/research/eth_evidence8_econ_label_v2_20260904/report_seed{SEED}.json"
TOPQ = 0.10
NULL_B = 200


def log(m): print(f"[screen] {m}", flush=True)


def causal_first_fire(fire: np.ndarray, gap: int) -> np.ndarray:
    """발동봉 중 **직전 gap봉에 발동이 없던** 것만 남긴다. 뒤만 보므로 인과적이다
    (cluster_dedup은 클러스터의 미래 최극단을 봐야 해서 A버그가 된다)."""
    keep = np.zeros(len(fire), bool)
    last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def screen_one(SIGNAL, HORIZON, sig, feat, eth, long) -> dict:
    bcol, tcol = f"bottom_{SIGNAL}", f"top_{SIGNAL}"
    if bcol not in sig.columns or tcol not in sig.columns:
        log(f"❌{SIGNAL}: 컬럼 없음"); return {}

    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    if kl["timestamp"].dt.tz is not None:
        kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(kl)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}

    S = sig[["timestamp", bcol, tcol]].copy()
    if S["timestamp"].dt.tz is not None:
        S["timestamp"] = S["timestamp"].dt.tz_localize(None)
    S["pos"] = [pos_of.get(np.datetime64(t), -1) for t in S["timestamp"].to_numpy()]
    S = S.loc[S["pos"] >= 0]
    fb = np.zeros(n, bool); ft = np.zeros(n, bool)
    fb[S["pos"].to_numpy()] = S[bcol].fillna(False).to_numpy(bool)
    ft[S["pos"].to_numpy()] = S[tcol].fillna(False).to_numpy(bool)
    kb, kt = causal_first_fire(fb, GAP), causal_first_fire(ft, GAP)

    lts = long["timestamp"].to_numpy()
    if getattr(long["timestamp"].dt, "tz", None) is not None:
        lts = long["timestamp"].dt.tz_localize(None).to_numpy()
    lpos = np.array([pos_of.get(np.datetime64(t), -1) for t in lts])
    is_down = long["is_downside"].to_numpy().astype(bool)
    keep = (lpos >= 0) & (lpos + 1 + HORIZON < n)
    keep &= np.where(is_down, kb[np.clip(lpos, 0, n - 1)], kt[np.clip(lpos, 0, n - 1)])
    D = long.loc[keep].reset_index(drop=True)
    ii = lpos[keep]
    if len(D) < 300:
        log(f"  ❌{SIGNAL}: 후보 {len(D)} (부족)"); return {}
    sg = np.where(D["is_downside"].to_numpy() == 1, 1.0, -1.0)

    entry = o[ii + 1]
    H = np.stack([h[i + 1:i + 1 + HORIZON] for i in ii])
    L = np.stack([l[i + 1:i + 1 + HORIZON] for i in ii])
    C = np.stack([c[i + 1:i + 1 + HORIZON] for i in ii])
    a_ = D["atr"].to_numpy(float)
    X = D[[c_ for c_ in TIER0 if c_ in D.columns]].to_numpy(float)
    split = D["split"].to_numpy()
    tr, va = split == "TRAIN", split == "VAL"
    if tr.sum() < 200 or va.sum() < 100:
        log(f"  ❌{SIGNAL}: TRAIN {tr.sum()} / VAL {va.sum()} (부족)"); return {}

    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    from scipy.stats import spearmanr
    rng = np.random.default_rng(SEED)
    out = {}

    for cell in TRAIL_CELLS:
        pn, _ = sim_exit(entry, a_, sg, H, L, C, *cell)
        net = pn * 1e4 - COST_BP
        base_all = float(net[va].mean())
        k = max(10, int(round(va.sum() * TOPQ)))

        # ⭐무작위 선택 귀무: 같은 크기로 무작위로 뽑았을 때 상위선별 평균의 분포
        nv = net[va]
        null = np.array([nv[rng.choice(len(nv), k, replace=False)].mean() for _ in range(NULL_B)])

        for lab in ("L1_sign", "L2_reg", "L3_tail", "L4_exclmid"):
            try:
                if lab == "L2_reg":
                    m = HistGradientBoostingRegressor(random_state=SEED, max_iter=300,
                                                      learning_rate=0.05)
                    m.fit(X[tr], net[tr]); pred = m.predict(X[va])
                else:
                    if lab == "L1_sign":
                        y = (net > 0).astype(int); fit = tr
                    elif lab == "L3_tail":
                        thr = np.quantile(net[tr], 0.25)
                        y = (net < thr).astype(int); fit = tr      # 1 = 대형손실
                    else:
                        lo, hi = np.quantile(net[tr], [0.25, 0.75])
                        y = np.where(net >= hi, 1, np.where(net <= lo, 0, -1))
                        fit = tr & (y >= 0)                        # 중간 제외는 **학습만**
                    if len(np.unique(y[fit])) < 2:
                        continue
                    m = HistGradientBoostingClassifier(random_state=SEED, max_iter=300,
                                                       learning_rate=0.05)
                    m.fit(X[fit], y[fit])
                    pred = m.predict_proba(X[va])[:, 1]
                    if lab == "L3_tail":
                        pred = -pred                                # 손실확률 낮은 순
                top = np.argsort(-pred)[:k]
                top_bp = float(nv[top].mean())
                ic = float(spearmanr(pred, nv).correlation)
                pval = float((null >= top_bp).mean())
                tag = f"{cell[0]}/{cell[1]}"
                pass
                out[f"{tag}|{lab}"] = {"all_mean_bp": base_all, "top_mean_bp": top_bp,
                                       "lift_bp": top_bp - base_all, "spearman_ic": ic,
                                       "random_null_p": pval, "k": k, "n_val": int(va.sum())}
            except Exception as e:                                  # noqa: BLE001
                log(f"  ⚠️{SIGNAL} {lab}: {type(e).__name__}")
    return out


def main() -> int:
    t0 = time.time()
    log("프레임 빌드(1회, 8종 공용)...")
    sig, feat, eth = _s1.build_sig()
    dummy = np.full(len(sig), "none", dtype=object)
    long = _s1.long_frame_for(sig, feat, dummy, dummy)
    log(f"  long {len(long):,}행 · {dict(long['split'].value_counts())}")
    log(f"⭐핵심 질문: 상위{int(TOPQ*100)}% 선별 평균 bp가 **양수**인 조합이 있는가\n")


    allrep = {}
    for name, hz in SIGNALS.items():
        r = screen_one(name, hz, sig, feat, eth, long)
        if r:
            allrep[name] = r

    flat = [(s_, k_, v) for s_, r in allrep.items() for k_, v in r.items()]
    pos = [x for x in flat if x[2]["top_mean_bp"] > 0]
    sigp = [x for x in pos if x[2]["random_null_p"] < 0.05]
    print()
    log(f"조합 {len(flat)}개 · ⭐상위10% 평균 bp 양수: **{len(pos)}**개 · "
        f"그중 무작위귀무 p<0.05: **{len(sigp)}**개")
    for s_, k_, v in sorted(pos, key=lambda x: -x[2]["top_mean_bp"])[:8]:
        log(f"    {s_:>22s} {k_:>18s}  상위{v['top_mean_bp']:+.2f}bp "
            f"(전체{v['all_mean_bp']:+.2f}) IC {v['spearman_ic']:+.3f} p={v['random_null_p']:.3f}")
    OUT2 = pathlib.Path(str(OUT).replace("{SEED}", str(SEED)))
    OUT2.parent.mkdir(parents=True, exist_ok=True)
    OUT2.write_text(json.dumps({"seed": SEED, "signals": allrep, "top_quantile": TOPQ, "cost_bp": COST_BP,
                               "n_combos": len(flat), "n_positive": len(pos),
                               "n_significant": len(sigp), "oos_touched": False,
                               "runtime_sec": round(time.time() - t0, 1)},
                              ensure_ascii=False, indent=2))
    log(f"산출: {OUT2} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
