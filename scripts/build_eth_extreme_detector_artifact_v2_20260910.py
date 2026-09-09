#!/usr/bin/env python3
"""극점 탐지기 **v2 아티팩트** -- 손실가중 학습으로 하드게이트를 대체 (2026-09-10, 사용자 지시).

## 무엇을 바꾸나
피쳐·모집단·라벨·분할은 v1 그대로. **학습 표본 가중치만** 바꾼다:
    강추세 역방향 자리의 **오답**(_y=0)에 가중치 (상한 4배)
그러면 "빗나가면 비싼 자리"를 모델이 스스로 피한다. 하드게이트(ret144 7일분위 ≥.8/≤.2 에서
콜 억제)는 **끈다** -- 아래 실측대로 게이트는 정밀도를 깎고 하루 4.39건을 버린다.

## 근거 (2026-09-10, 표본외 06-20~09-08 81일, HGB 5시드 통제 비교)
    같은 건/일     정밀도                역추세 비중(독립 정의 T1~T4 평균)
    1.5건    v1게이트 .650 -> v2 .707   v1 .164 -> v2 .155
    2.5건    v1게이트 .614 -> v2 .641   v1 .160 -> v2 .141
    4.0건    v1게이트 .590 -> v2 .598   v1 .140 -> v2 .155
⭐v2 가 내는 소수의 역추세 콜은 **정밀도가 더 높다**(.769~.800 vs 전체 .641) -- 게이트는
   바로 그 좋은 콜을 지우고 있었다.
⭐게이트의 "역추세 0%"는 **자기가 게이팅하는 통계 안에서만** 성립한다. 다른 추세 정의로 재면
   게이트도 .083~.220 이라 v2 와 대등하다. 순환 점검 4정의 전부에서 v2 는 무게없음 대비
   역추세 비중 -40~-76%, 정밀도 +.017.
⚠️λ 는 실효 파라미터가 아니다 -- clip 상한에서 포화한다(상한 3~10 이 전부 같은 대역).

⚠️등급 컷은 표본외 **전반부**, 정밀도는 **후반부**에서 잰다(v1 규약, 순환 방지).
   등급은 배타 구간(강 > 중 > 약)이라 라이브 `grade_of` 와 같은 정의다.
"""
from __future__ import annotations
import argparse, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "tmp/eth_signal_map_20260909"
LIVE = ROOT / "data/live/eth_extreme_detector_artifact"
SEEDS = [20260909, 771233, 305610, 517758, 961476]
TIER_Q = (("강", 0.05), ("중", 0.10), ("약", 0.25))
TQ_HI, TQ_LO, CAP_W = 0.8, 0.2, 4.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hgb", "tabpfn"], default="hgb")
    ap.add_argument("--weight", choices=["counter", "none"], default="counter")
    ap.add_argument("--cap", type=int, default=10000, help="TabPFN 컨텍스트 상한")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import joblib
    from sklearn.metrics import roc_auc_score
    OUT = Path(a.out) if a.out else EX / f"artifact_v2_{a.model}_{a.weight}"
    OUT.mkdir(parents=True, exist_ok=True)

    A = pd.read_parquet(EX / "extreme_frame.parquet")
    fm = json.load(open(EX / "extreme_frame_meta.json"))
    feats, VAL0, W = fm["feats"], pd.Timestamp(fm["val0"]), fm["w"]
    A = A[A["_y"] >= 0].sort_values("_ts").reset_index(drop=True)
    X = np.nan_to_num(A[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = A["_y"].to_numpy(int); ts = A["_ts"]
    lg = A["_long"].to_numpy(bool); tq = A["_tq"].to_numpy(float)
    counter = (lg & (tq <= TQ_LO)) | (~lg & (tq >= TQ_HI))
    tr = (ts < VAL0).to_numpy(); oos = ~tr

    w = np.ones(len(A))
    if a.weight == "counter":
        w[(y == 0) & counter] = CAP_W
        for c in (0, 1):                       # 클래스별 평균 1 -- 재가중과 클래스 재균형 분리
            m = tr & (y == c); w[m] /= w[m].mean()
    print(f"학습 {tr.sum():,} · 표본외 {oos.sum():,} · 피쳐 {len(feats)} · 기저 {y[oos].mean():.4f}")
    print(f"가중 {a.weight} · 역추세 오답 {int(((y==0)&counter&tr).sum()):,}건 "
          f"(학습의 {((y==0)&counter&tr).sum()/tr.sum()*100:.1f}%)\n", flush=True)

    models, P = [], np.zeros(len(A))
    for sd in SEEDS:
        itr = np.flatnonzero(tr); t0 = time.time()
        if a.model == "tabpfn":
            from tabpfn import TabPFNClassifier
            if len(itr) > a.cap:               # 가중 컨텍스트 표집 -- TabPFN 의 sample_weight 대응물
                p = w[itr] / w[itr].sum()
                itr = np.sort(np.random.default_rng(sd).choice(itr, a.cap, replace=False, p=p))
            m = TabPFNClassifier(device=a.device, random_state=sd, n_estimators=1)
            m.fit(X[itr], y[itr])
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                               min_samples_leaf=40, l2_regularization=1.0,
                                               random_state=sd)
            m.fit(X[itr], y[itr], sample_weight=w[itr])
        P += m.predict_proba(X)[:, 1] / len(SEEDS); models.append(m)
        print(f"   시드 {sd} · 학습 {len(itr):,}행 · {time.time()-t0:.0f}s", flush=True)
    A["p"] = P
    auc = float(roc_auc_score(y[oos], P[oos]))

    o = np.flatnonzero(oos); k = len(o) // 2
    cut_m = np.zeros(len(A), bool); cut_m[o[:k]] = True
    ev_m = np.zeros(len(A), bool); ev_m[o[k:]] = True
    cuts = {g: float(np.quantile(P[cut_m], 1 - q)) for g, q in TIER_Q}
    days = (ts[ev_m].max() - ts[ev_m].min()).total_seconds() / 86400.0
    prec, per_day, cnt_share, cnt_prec = {}, {}, {}, {}
    prev = None
    for g, _ in TIER_Q:
        sel = ev_m & (P >= cuts[g]) & ((P < cuts[prev]) if prev else True)
        prec[g] = round(float(y[sel].mean()), 4) if sel.sum() else None
        per_day[g] = round(sel.sum() / days, 2)
        cnt_share[g] = round(float(counter[sel].mean()), 4) if sel.sum() else None
        cs = sel & counter
        cnt_prec[g] = round(float(y[cs].mean()), 4) if cs.sum() >= 10 else None
        prev = g
    print(f"\n표본외 AUC {auc:.4f} · 컷창 {cut_m.sum():,} · 평가창 {ev_m.sum():,} ({days:.0f}일)")
    print(f"{'등급':<4}{'컷':>9}{'정밀도':>9}{'건/일':>8}{'역추세비중':>11}{'역추세정밀':>11}")
    for g, _ in TIER_Q:
        cp = f"{cnt_prec[g]:.3f}" if cnt_prec[g] is not None else "  -  "
        print(f"{g:<4}{cuts[g]:>9.4f}{prec[g]:>9.4f}{per_day[g]:>8.2f}{cnt_share[g]:>11.4f}{cp:>11}")

    joblib.dump(models, OUT / "model.joblib")
    prev_meta = json.loads((LIVE / "meta.json").read_text()) if (LIVE / "meta.json").exists() else {}
    meta = {**prev_meta,
            "rule_id": f"eth_extreme_detector_w{W}_costw_{a.model}_20260910",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": a.model, "context_cap": a.cap if a.model == "tabpfn" else None,
            "features": feats, "seeds": SEEDS, "label_window_bars": W,
            "n_train": int(tr.sum()), "auc_oos": round(auc, 4),
            "base_rate": round(float(y[oos].mean()), 4),
            "cuts": cuts, "precision": prec, "per_day": per_day,
            "counter_share": cnt_share, "counter_precision": cnt_prec,
            "cost_weight": {"kind": a.weight, "target": "counter_trend_negatives",
                            "cap": CAP_W, "tq_hi": TQ_HI, "tq_lo": TQ_LO},
            "gate": {"kind": "none"} if a.weight == "counter" else prev_meta.get("gate"),
            "cut_window": [str(ts[cut_m].min()), str(ts[cut_m].max())],
            "eval_window": [str(ts[ev_m].min()), str(ts[ev_m].max())],
            "note": ("사람이 보는 위치 탐지기다. 매매 트리거가 아니다. | 2026-09-10 v2: 강추세 "
                     "역방향 **오답**에 가중치(상한 4배)를 줘 학습으로 역추세 콜을 억제한다. "
                     "그래서 v1 의 하드게이트는 끈다 -- 게이트는 정밀도를 깎고 하루 4.39건을 "
                     "버렸으며, '역추세 0%'는 자기가 게이팅하는 통계 안에서만 성립했다. "
                     "등급 컷은 표본외 전반부, 정밀도는 후반부(순환 방지).")}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"\n저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
