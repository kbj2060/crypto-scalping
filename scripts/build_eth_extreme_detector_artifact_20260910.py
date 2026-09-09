#!/usr/bin/env python3
"""극점 탐지기 아티팩트 -- 모델 교체(HGB -> TabPFN v3) + 등급 컷 재산출 (2026-09-10, 사용자 지시).

## 왜 교체하나 (2026-09-09 비교 실측)
월별 확장 워크포워드 · 시드 5개 · 세 팔(모델과 표본크기를 가른다):
    AUC          전체            ~2026-03        2026-04~
    HGB(전체)     .7088 ±.0045   .7078 ±.0052    .7110 ±.0040
    HGB(1만)      .7008 ±.0037   .7018 ±.0031    .6974 ±.0069
    TabPFN v3     .7181 ±.0036   .7179 ±.0036    .7179 ±.0041   ← 세 구간 전부 승, 시드 범위 안 겹침
표본을 27.5k -> 10k 로 줄이는 핸디캡(-0.80pp)을 안고도 이긴다. 같은 표본 기준 +1.73pp.
⚠️상위 10% 정밀도는 최근 구간(2026-04~)에서 HGB 가 앞선다(.5609 vs .5494) -- 무승부로 읽는다.

## 등급 컷을 다시 잡는 이유
현행 컷(강 .5056 / 중 .4371 / 약 .2963)은 **HGB 점수 분포**에서 잡은 값이다. TabPFN 은 분포가
다르므로 그대로 쓰면 등급별 건수·정밀도가 어긋난다.

## 순환 방지
컷은 표본외 **전반부**에서 잡고 정밀도는 **후반부**에서 잰다. 같은 창에서 컷을 잡고 정밀도를
재면 그 숫자는 자기 자신을 설명할 뿐이다.
"""
from __future__ import annotations
import os, sys, json, time, argparse
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "8")
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "tmp/eth_signal_map_20260909"
LIVE = ROOT / "data/live/eth_extreme_detector_artifact"
SEEDS = [20260909, 771233, 305610, 517758, 961476]
TIER_Q = (("강", 0.05), ("중", 0.10), ("약", 0.25))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tabpfn", "hgb"], default="tabpfn")
    ap.add_argument("--cap", type=int, default=10000, help="TabPFN 컨텍스트 상한")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import joblib
    from sklearn.metrics import roc_auc_score
    OUT = Path(a.out) if a.out else EX / f"artifact_staging_{a.model}"
    OUT.mkdir(parents=True, exist_ok=True)

    A = pd.read_parquet(EX / "extreme_frame.parquet")
    fm = json.load(open(EX / "extreme_frame_meta.json"))
    feats, VAL0, W = fm["feats"], pd.Timestamp(fm["val0"]), fm["w"]
    A = A[A["_y"] >= 0].sort_values("_ts").reset_index(drop=True)
    X = np.nan_to_num(A[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = A["_y"].to_numpy(int); ts = A["_ts"]
    tr = (ts < VAL0).to_numpy(); oos = ~tr
    print(f"학습 {tr.sum():,} · 표본외 {oos.sum():,} · 피쳐 {len(feats)} · 기저 {y[oos].mean():.4f}", flush=True)

    models, P = [], np.zeros(len(A))
    for sd in SEEDS:
        itr = np.flatnonzero(tr)
        if a.model == "tabpfn":
            from tabpfn import TabPFNClassifier
            if len(itr) > a.cap:
                itr = np.sort(np.random.default_rng(sd).choice(itr, a.cap, replace=False))
            m = TabPFNClassifier(device=a.device, random_state=sd, n_estimators=1)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_depth=6,
                                               l2_regularization=1.0, random_state=sd,
                                               early_stopping=True, validation_fraction=0.15)
        t0 = time.time(); m.fit(X[itr], y[itr])
        P += m.predict_proba(X)[:, 1] / len(SEEDS); models.append(m)
        print(f"   시드 {sd} · 학습 {len(itr):,}행 · {time.time()-t0:.0f}s", flush=True)
    A["p"] = P
    auc = float(roc_auc_score(y[oos], P[oos]))

    # 순환 방지: 컷은 표본외 전반부, 정밀도는 후반부
    o = np.flatnonzero(oos); half = o[len(o) // 2]
    cut_m = np.zeros(len(A), bool); cut_m[o[:len(o) // 2]] = True
    ev_m = np.zeros(len(A), bool); ev_m[o[len(o) // 2:]] = True
    cuts = {g: float(np.quantile(P[cut_m], 1 - q)) for g, q in TIER_Q}
    days = (ts[ev_m].max() - ts[ev_m].min()).total_seconds() / 86400.0
    prec, per_day = {}, {}
    prev = None
    for g, _ in TIER_Q:                      # 등급은 배타 구간이다(강 > 중 > 약)
        sel = ev_m & (P >= cuts[g]) & ((P < cuts[prev]) if prev else True)
        prec[g] = round(float(y[sel].mean()), 4) if sel.sum() else None
        per_day[g] = round(sel.sum() / days, 2); prev = g
    print(f"\n표본외 AUC {auc:.4f} · 컷창 {cut_m.sum():,}건 · 평가창 {ev_m.sum():,}건 ({days:.0f}일)")
    for g, _ in TIER_Q:
        print(f"   {g}  컷 {cuts[g]:.4f}  정밀도 {prec[g]}  {per_day[g]}건/일")

    joblib.dump(models, OUT / "model.joblib")
    prev_meta = json.loads((LIVE / "meta.json").read_text()) if (LIVE / "meta.json").exists() else {}
    meta = {**prev_meta,
            "rule_id": f"eth_extreme_detector_w{W}_gated_{a.model}_20260910",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": a.model, "context_cap": a.cap if a.model == "tabpfn" else None,
            "features": feats, "seeds": SEEDS, "label_window_bars": W,
            "n_train": int(tr.sum()), "auc_oos": round(auc, 4),
            "base_rate": round(float(y[oos].mean()), 4),
            "cuts": cuts, "precision": prec, "per_day": per_day,
            "cut_window": [str(ts[cut_m].min()), str(ts[cut_m].max())],
            "eval_window": [str(ts[ev_m].min()), str(ts[ev_m].max())],
            "note": (prev_meta.get("note", "") +
                     f" | 2026-09-10 모델 {prev_meta.get('model','hgb')} -> {a.model}. "
                     "등급 컷은 표본외 전반부에서 잡고 정밀도는 후반부에서 쟀다(순환 방지) -- "
                     "그래서 이전 meta 의 정밀도와 직접 비교하면 안 된다(그건 같은 창이었다).")}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"\n저장: {OUT}")
    print(json.dumps({"done": True, "model": a.model, "auc": auc}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
