#!/usr/bin/env python3
"""증거신호 **경제성 재라벨링** Phase 0 -- 학습가능성 값싼 선별 (ETH liquidity_sweep).

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
TRAIL_CELLS = [(3.0, 1.5, 0.1), (5.0, 1.5, 0.1), (4.0, 1.0, 0.1)]   # 진단 전용, ARM>=1.0

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")     # ⚠️여기서 끊는다 -- OOS 미터치
START = pd.Timestamp("2024-01-01", tz="UTC")
SEED = 20260904
OUT = ROOT / "data/research/eth_evidence8_econ_label_screen_20260904/report.json"


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
        log(f"❌{SIGNAL}: 컬럼 없음")
        return {}

    # ⭐검증된 빌더를 그대로 쓴다. `is_downside`/`sweep_penetration_atr`/`flow_aligned_delta_z`는
    # 봉 단위가 아니라 **(봉,측면) 단위**라 단순 merge로는 못 만든다. 이 함수가 그걸 처리하고,
    # 덤으로 [START, VAL_END)로 잘라 **OOS를 원천 차단**한다. status는 V자반등 라벨용이라
    # 더미를 넣고 무시한다(우리는 경제성 라벨을 따로 만든다).
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    if kl["timestamp"].dt.tz is not None:
        kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(kl)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}

    # 인과적 첫 발동 마스크 (봉 인덱스 기준)
    S = sig[["timestamp", bcol, tcol]].copy()
    if S["timestamp"].dt.tz is not None:
        S["timestamp"] = S["timestamp"].dt.tz_localize(None)
    S["pos"] = [pos_of.get(np.datetime64(t), -1) for t in S["timestamp"].to_numpy()]
    S = S.loc[S["pos"] >= 0]
    fb = np.zeros(n, bool); ft = np.zeros(n, bool)
    fb[S["pos"].to_numpy()] = S[bcol].fillna(False).to_numpy(bool)
    ft[S["pos"].to_numpy()] = S[tcol].fillna(False).to_numpy(bool)
    kb, kt = causal_first_fire(fb, GAP), causal_first_fire(ft, GAP)
    log(f"  raw 발동    bottom {int(fb.sum()):,} / top {int(ft.sum()):,}")
    log(f"  인과 첫발동 bottom {int(kb.sum()):,} / top {int(kt.sum()):,}  (GAP={GAP})")

    # long의 각 (봉,측면) 행이 우리 모집단인지 판정
    lts = long["timestamp"].to_numpy()
    if getattr(long["timestamp"].dt, "tz", None) is not None:
        lts = long["timestamp"].dt.tz_localize(None).to_numpy()
    lpos = np.array([pos_of.get(np.datetime64(t), -1) for t in lts])
    is_down = long["is_downside"].to_numpy().astype(bool)
    keep = (lpos >= 0) & (lpos + 1 + HORIZON < n)
    keep &= np.where(is_down, kb[np.clip(lpos, 0, n - 1)], kt[np.clip(lpos, 0, n - 1)])
    D = long.loc[keep].reset_index(drop=True)
    ii = lpos[keep]
    sg = np.where(D["is_downside"].to_numpy() == 1, 1.0, -1.0)
    nl, ns = int((sg > 0).sum()), int((sg < 0).sum())
    log(f"  후보 {len(D):,}건 (롱 {nl:,} / 숏 {ns:,})")
    if not len(D):
        log(f"  ❌{SIGNAL}: 후보 없음"); return {}

    entry = o[ii + 1]
    H = np.stack([h[i + 1:i + 1 + HORIZON] for i in ii])
    L = np.stack([l[i + 1:i + 1 + HORIZON] for i in ii])
    C = np.stack([c[i + 1:i + 1 + HORIZON] for i in ii])
    a_ = D["atr"].to_numpy(float)

    labels = {}
    labels["time_h30"] = (C[:, -1] / entry - 1.0) * sg * 1e4 - COST_BP
    for cell in TRAIL_CELLS:
        pn, _ = sim_exit(entry, a_, sg, H, L, C, *cell)
        labels[f"trail_{cell[0]}_{cell[1]}_{cell[2]}"] = pn * 1e4 - COST_BP

    split = D["split"].to_numpy()
    tr, va = split == "TRAIN", split == "VAL"
    inwin = np.ones(len(D), bool)
    log(f"  TRAIN {int(tr.sum()):,} / VAL {int(va.sum()):,}  ⚠️OOS 미터치")

    # ⭐무작위 진입 기준선
    rng = np.random.default_rng(SEED)
    ok = np.flatnonzero((np.arange(n) > 0) & (np.arange(n) + 1 + HORIZON < n))
    ri = rng.choice(ok, size=len(ii), replace=False)
    rs = sg.copy(); rng.shuffle(rs)
    rentry = o[ri + 1]
    rH = np.stack([h[i + 1:i + 1 + HORIZON] for i in ri])
    rL = np.stack([l[i + 1:i + 1 + HORIZON] for i in ri])
    rC = np.stack([c[i + 1:i + 1 + HORIZON] for i in ri])
    ratr = np.full(len(ri), float(np.nanmedian(a_)))
    rand_lab = {"time_h30": (rC[:, -1] / rentry - 1.0) * rs * 1e4 - COST_BP}
    for cell in TRAIL_CELLS:
        pn, _ = sim_exit(rentry, ratr, rs, rH, rL, rC, *cell)
        rand_lab[f"trail_{cell[0]}_{cell[1]}_{cell[2]}"] = pn * 1e4 - COST_BP

    X = D[[c_ for c_ in TIER0 if c_ in D.columns]].to_numpy(float)


    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    rep = {}
    for name, net in labels.items():
        y = (net > 0).astype(int)
        base_tr, base_va = float(y[tr].mean()), float(y[va].mean())
        mean_bp, rand_bp = float(net[inwin].mean()), float(rand_lab[name].mean())
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2 or tr.sum() < 100:
            continue
        m = HistGradientBoostingClassifier(random_state=SEED, max_iter=300,
                                           learning_rate=0.05, max_leaf_nodes=31)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        auc = float(roc_auc_score(y[va], p))
        side_auc = {}
        for lbl, mk in (("long", sg[va] > 0), ("short", sg[va] < 0)):
            yy, pp = y[va][mk], p[mk]
            side_auc[lbl] = float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else np.nan
        print(f"{SIGNAL[:20]:>21s}{name[:18]:>19s}{base_va:9.3f}{mean_bp:9.2f}{rand_bp:9.2f}"
              f"{auc:9.4f}{side_auc['long']:7.3f}{side_auc['short']:7.3f}")
        rep[name] = {"base_rate_train": base_tr, "base_rate_val": base_va,
                     "mean_bp_all": mean_bp, "random_entry_mean_bp": rand_bp,
                     "val_auc": auc, "val_auc_long": side_auc["long"],
                     "val_auc_short": side_auc["short"]}

    return {"n_candidates": int(len(D)), "n_train": int(tr.sum()), "n_val": int(va.sum()),
            "horizon": HORIZON, "labels": rep}


def main() -> int:
    t0 = time.time()
    log("프레임 빌드(1회, 8종 공용)...")
    sig, feat, eth = _s1.build_sig()
    dummy = np.full(len(sig), "none", dtype=object)
    long = _s1.long_frame_for(sig, feat, dummy, dummy)
    log(f"  long (봉,측면) {len(long):,}행 · {dict(long['split'].value_counts())}")

    print(f"\n{'신호':>21s}{'라벨':>19s}{'양성률VA':>9s}{'평균bp':>9s}{'무작위bp':>9s}"
          f"{'VAL AUC':>9s}{'롱AUC':>7s}{'숏AUC':>7s}")
    print("-" * 91)
    allrep = {}
    for name, hz in SIGNALS.items():
        r = screen_one(name, hz, sig, feat, eth, long)
        if r:
            allrep[name] = r
        print("-" * 91)

    flat = [(s_, l_, v["val_auc"], v["mean_bp_all"], v["random_entry_mean_bp"])
            for s_, r in allrep.items() for l_, v in r["labels"].items()]
    best = max((x[2] for x in flat), default=0.0)
    worse_than_random = sum(1 for x in flat if x[3] < x[4])
    print()
    log(f"⭐전체 최고 VAL AUC {best:.4f} (조합 {len(flat)}개)")
    log(f"⭐발동이 무작위 진입보다 **나쁜** 조합: {worse_than_random}/{len(flat)}")
    log(f"⭐평균bp가 양수인 조합: {sum(1 for x in flat if x[3] > 0)}/{len(flat)}")
    log("판정: " + ("❌BTC와 같은 결론(방향력 없음) -- 축 종결 권고"
                   if best < 0.55 else "✅유의미 조합 존재 -- 정식관문 검토"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"signals": allrep, "cost_bp": COST_BP, "gap": GAP,
                               "best_val_auc": best, "n_combos": len(flat),
                               "n_worse_than_random": worse_than_random,
                               "oos_touched": False,
                               "runtime_sec": round(time.time() - t0, 1)},
                              ensure_ascii=False, indent=2))
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
