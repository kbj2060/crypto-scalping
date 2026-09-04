#!/usr/bin/env python3
"""추세 신호 v1 -- 통과·경계 신호의 **메타라벨(TabPFN) 평가**: 발동 봉 Tier0 23 피쳐로 '지속 방향이 반대를 이기는가'를 예측 (2026-09-04).

입력: data/research/eth_trend_signals_v1_screen_20260904/triggers_<sig>.parquet (변형 탐색에서 내보낸 첫발동 (timestamp, is_downside=지속 방향))
      tmp/homer_entry_v2_20260904/frame.parquet (같은 (봉,측면) 행의 Tier0 23 + net_bp/net_bp_flip)
라벨: y_dir = net_bp(지속) > net_bp_flip(반대)  (두 측면 경제라벨의 승부) ; 보조 y_pos = net_bp > 0
학습: TRAIN 행 컨텍스트(TabPFN, 5시드) -> VAL/OOS AUC·Brier, 상위 30% 선별 시 지속 순손익(선별 경제성).
판정(사전): VAL·OOS AUC 둘 다 ≥ 0.55 이고 상위30% 지속 순손익 > 전체 평균 -> 칩 후보(다음: 층 게이트).
"""
import json, sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
TRIG = ROOT / "data/research/eth_trend_signals_v1_screen_20260904"
FRAME = ROOT / "tmp/homer_entry_v2_20260904/frame.parquet"
SEEDS = [20260829, 141592, 271828, 577215, 20260904]
SIGS = ["regime_pullback_resume", "oi_confirmed_breakout", "spot_led_move", "btc_leadlag", "liquidity_vacuum"]


def main(learner="tabpfn"):
    from sklearn.metrics import roc_auc_score, brier_score_loss
    t0 = time.time()
    card = json.loads((ROOT / "tmp/homer_entry_v2_20260904/model_card.json").read_text()); F0 = card["arms"]["F0"]
    cols = list(dict.fromkeys(["timestamp", "is_downside", "split", "net_bp", "net_bp_flip"] + F0))   # F0에 is_downside 포함 -> 중복 제거
    D = pd.read_parquet(FRAME, columns=cols); D["timestamp"] = pd.to_datetime(D["timestamp"])
    rep = {"learner": learner, "seeds": SEEDS, "signals": {}}
    for s in SIGS:
        p = TRIG / f"triggers_{s}.parquet"
        if not p.exists():
            continue
        tg = pd.read_parquet(p); tg["timestamp"] = pd.to_datetime(tg["timestamp"])
        X = tg.merge(D, on=["timestamp", "is_downside"], how="inner").dropna(subset=F0)
        X["y_dir"] = (X["net_bp"] > X["net_bp_flip"]).astype(int)
        tr = X[X.split == "TRAIN"]; S = {w: X[X.split == w] for w in ("VAL", "OOS")}
        R = {"n": {"TRAIN": int(len(tr)), **{w: int(len(S[w])) for w in S}}, "base_dir": {w: round(float(S[w].y_dir.mean()), 3) for w in S}}
        per = {w: [] for w in S}
        for sd in SEEDS:
            if learner == "tabpfn":
                from tabpfn import TabPFNClassifier
                clf = TabPFNClassifier(device="cuda", random_state=int(sd), ignore_pretraining_limits=True).fit(tr[F0], tr["y_dir"])
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, random_state=int(sd)).fit(tr[F0], tr["y_dir"])
            for w in S:
                per[w].append(clf.predict_proba(S[w][F0])[:, 1])
        for w in S:
            pm = np.mean(per[w], axis=0); y = S[w].y_dir.to_numpy(); net = S[w].net_bp.to_numpy()
            k = max(int(len(y) * 0.3), 10); top = np.argsort(-pm)[:k]
            R[w] = {"auc": round(float(roc_auc_score(y, pm)), 4) if len(np.unique(y)) > 1 else None, "brier": round(float(brier_score_loss(y, pm)), 4),
                    "auc_per_seed": [round(float(roc_auc_score(y, q)), 4) for q in per[w]], "cont_bp_all": round(float(net.mean()), 2),
                    "cont_bp_top30": round(float(net[top].mean()), 2), "diff_bp_top30": round(float((net[top] - S[w].net_bp_flip.to_numpy()[top]).mean()), 2)}
        ok = all(R[w]["auc"] is not None and R[w]["auc"] >= 0.55 and R[w]["cont_bp_top30"] > R[w]["cont_bp_all"] for w in S)
        R["verdict"] = "CHIP_CANDIDATE" if ok else "NOT_YET"
        rep["signals"][s] = R
        print(f"{s:>26s} n {R['n']} | " + " | ".join(f"{w} AUC {R[w]['auc']} br {R[w]['brier']} top30 {R[w]['cont_bp_top30']} vs all {R[w]['cont_bp_all']}" for w in S) + f" => {R['verdict']}  ({time.time()-t0:.0f}s)", flush=True)
    (TRIG / f"report_metalabel_{learner}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str)); print("METALABEL_DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tabpfn")
