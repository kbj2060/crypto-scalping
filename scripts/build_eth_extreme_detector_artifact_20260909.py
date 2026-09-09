#!/usr/bin/env python3
"""ETH **극점 탐지기** 모델 아티팩트 동결 (2026-09-09).

용도: 사람이 보는 위치 탐지기. "이 봉이 ±60분 국소 극점일 확률"을 등급으로 준다.
      ⚠️매매 트리거가 아니다 — 이 등급으로 매매하면 비용 뒤 여유가 없다(아래 note).

라벨   i+1..i+12 에서 봉 i 의 저점/고점을 깨지 않는가 (피쳐는 봉 i 까지 — 사건 라벨 경계 계약)
모집단 증거신호 8종 중 하나라도 발동한 봉(측면별)
모델   HistGradientBoosting 5시드 평균
게이트 🔴강한 추세 구간(ret144 7일 롤링 분위 ≥0.80 또는 ≤0.20)에서는 콜을 억제한다.
      근거(표본외 161일): 강한상승 천장 콜은 정밀도 51.6% 인데 적중 +10.8 / 빗나감 -60.7bp 로
      완전 비대칭(순 -23.8bp). 추세 피쳐를 넣고 재학습해도 안 고쳐진다(역추세 비중 30.8→30.2%)
      -- 라벨이 손익을 벌하지 않으므로 **하드 게이트**여야 한다.

동결 원칙: 평가된 모델 = 배포되는 모델. 학습을 2026-03-31 에서 끊고 그 뒤를 표본외로 잰 값을
          그대로 등급 정밀도로 싣는다(재학습하면 그 수치가 이 아티팩트의 것이 아니게 된다).
"""
from __future__ import annotations
import argparse, glob, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import live_evidence_signal_dashboard_20260823 as EV
import build_eth_anchor_label_dataset_20260907 as B
import live_eth_extreme_detector_20260909 as LD          # 피쳐 정의는 라이브와 한 곳에서

ART = ROOT / "data/live/eth_extreme_detector_artifact"
TRAIN_END = pd.Timestamp("2026-04-01")
SEEDS = [20260909, 771233, 305610, 517758, 961476]
RULE_ID = "eth_extreme_detector_w12_gated_20260909"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--klcache", default=str(ROOT / "tmp/eth_signal_map_20260909/klcache"))
    a = ap.parse_args()

    def newest(sym):
        best, bn = None, 0
        for f in glob.glob(f"{a.klcache}/{sym}_5m_*.parquet"):
            d = pd.read_parquet(f)
            if len(d) > bn: best, bn = d, len(d)
        if best is None: raise SystemExit(f"klcache 에 {sym} 5분봉이 없다: {a.klcache}")
        return best.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    kl, btc = newest("ETHUSDT"), newest("BTCUSDT")
    try:
        fund = EV.fetch_funding_history(limit=1000)
        fund["calc_time"] = pd.to_datetime(fund["calc_time"]).dt.tz_localize(None)
    except Exception:
        fund = None
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=fund)
    A = LD.build_rows(sig, btc, with_label=True)
    print(f"모집단 {len(A):,} ({A._ts.min()} ~ {A._ts.max()})", flush=True)

    tr = A[(A._ts < TRAIN_END) & (A._y >= 0)]
    oo = A[(A._ts >= TRAIN_END) & (A._y >= 0)].copy()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    import joblib
    models, P = [], np.zeros(len(oo))
    for sd in SEEDS:
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_depth=6,
                                           l2_regularization=1.0, random_state=sd,
                                           early_stopping=True, validation_fraction=0.15)
        m.fit(tr[LD.FEATS], tr._y); models.append(m)
        P += m.predict_proba(oo[LD.FEATS])[:, 1] / len(SEEDS)
    oo["p"] = P
    cuts = {g: float(oo.p.quantile(1 - q)) for g, q in (("강", 0.05), ("중", 0.10), ("약", 0.25))}
    oo["grade"] = LD.grade_of(oo.p.to_numpy(), cuts)
    oo["gated"] = LD.gated_of(oo._tq.to_numpy(), oo._long.to_numpy())
    days = (oo._ts.max() - oo._ts.min()).total_seconds() / 86400
    auc = float(roc_auc_score(oo._y, oo.p))
    prec, cov = {}, {}
    ung = oo[~oo.gated]
    for g in ("강", "중", "약"):
        q = ung[ung.grade == g]
        prec[g] = round(float(q._y.mean()), 4); cov[g] = round(len(q) / days, 2)
    meta = {
        "rule_id": RULE_ID, "created_utc": datetime.now(timezone.utc).isoformat(),
        "features": LD.FEATS, "seeds": SEEDS, "label_window_bars": LD.W,
        "train_span": [str(tr._ts.min()), str(tr._ts.max())], "n_train": int(len(tr)),
        "oos_span": [str(oo._ts.min()), str(oo._ts.max())], "n_oos": int(len(oo)),
        "auc_oos": round(auc, 4), "base_rate": round(float(oo._y.mean()), 4),
        "cuts": cuts, "gate": {"kind": "trend_quantile", "hi": LD.GATE_HI, "lo": LD.GATE_LO,
                               "window_bars": LD.TREND_W, "rank_bars": LD.RANK_W},
        "precision": prec, "per_day": cov,
        "gated_suppressed_per_day": round(len(oo[oo.gated & (oo.grade != "-")]) / days, 2),
        "note": ("사람이 보는 위치 탐지기다. 매매 트리거가 아니다 -- 등급대로 매매하면 "
                 "표본외 순 +2.27bp(강+중, 2.93건/일)로 비용 여유가 없다. "
                 "게이트 없이는 -3.36bp 였다."),
    }
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, ART / "model.joblib")
    (ART / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"AUC(OOS) {auc:.4f} · 기저 {meta['base_rate']*100:.1f}% · {days:.0f}일")
    print(f"등급 정밀도(게이트 후)  " + " · ".join(
        f"{g} {prec[g]*100:.1f}%({cov[g]:.2f}건/일)" for g in ("강", "중", "약")))
    print(f"→ {ART}")
    print(json.dumps({"done": True, "rule_id": RULE_ID}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
