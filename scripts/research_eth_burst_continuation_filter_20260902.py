#!/usr/bin/env python3
"""버스트 지속 예측 필터 -- 인과적 발동 규칙의 재구성 (2026-09-02).

경위
----
앵커 미래참조로 기존 발동집합이 무효화된 뒤, 수익이 클러스터의 어느 자리에 있는지 분해했더니
**64.6%가 버스트의 첫 봉(①단일봉 46.3% + ②다봉-첫봉 18.3%)에 있었다.** 첫 봉은 인과적으로
식별 가능하다(직전 GAP봉에 트리거가 없으면 첫 봉이다). 변형 B(첫발동)가 이 둘을 모두 잡는데도
손실이었던 이유는, ③/④ 클러스터의 첫 봉(= 앵커가 아닌 봉)까지 같이 잡기 때문이다.

그래서 문제가 다시 정의된다 -- 미래참조 없이:
  **"지금 시작된 이 버스트가 GAP봉 안에 이어질 것인가?"**
  이어지지 않으면(=고립) 이 봉이 곧 앵커다. 이어지면 이 봉은 잡으면 안 되는 자리다.

설계
----
  모집단 : 버스트 첫 트리거 (직전 GAP봉에 동측 트리거 없음) -- 완전히 인과적
  라벨   : 이 버스트가 이어졌는가 (GAP봉 안 동측 트리거 재발생). 학습 대상일 뿐,
           추론 시점에는 쓰지 않는다.
  피처   : Tier0 23종 (전부 그 봉까지의 정보)
  분할   : TRAIN < 2025-09-01 / VAL 2025-09~12 / OOS 2026-01~03 (HOLDOUT 미접촉)
  판정   : ORACLE(완전예지) 상한을 먼저 확인하고, 학습 필터가 그 상한에 얼마나 접근하는지 본다.
           임계값은 VAL에서만 고르고 OOS에 그대로 적용. 1슬롯 순차 + 방향뒤집기 대조.

ORACLE이 3창 전부 양수+방향뒤집기 통과가 아니면 이 축도 닫힌다 -- 학습기가 오라클을 못 넘는다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, OOS_START, VAL_START)
from research_eth_evidence_signal_ensemble_pnl_20260902 import (  # noqa: E402
    per_fire_outcomes, sequential_portfolio, summarize)
from research_eth_causal_climax_predictor_20260902 import clusters_of  # noqa: E402

OUT_DIR = ROOT / "tmp/eth_burst_continuation_filter_20260902"
START = pd.Timestamp("2024-01-01")
SEED = 7529
SPEC = {"short_term_return_z": {"gap": 3, "sl": 3.0, "arm": 1.0, "trail": 0.1, "horizon": 12},
        "demarker_extreme": {"gap": 12, "sl": 2.0, "arm": 1.5, "trail": 0.1, "horizon": 8}}
TOP2 = ["short_term_return_z", "demarker_extreme"]
HP = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80,
          l2_regularization=2.0, early_stopping=True, validation_fraction=0.15,
          n_iter_no_change=20)
CONTRACT = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
            "lookahead_in_fire_construction": False, "holdout_touched": False}


def log(m): print(f"[burst] {m}", flush=True)


def main() -> int:
    from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (
        FEATURE_COLUMNS, build_indicator_frame)

    src = load_klines(); ind = build_indicator_frame(src)
    ret3_z = ind["ret3_z"].to_numpy(); dem = compute_demarker(src["high"], src["low"]).to_numpy()
    atr_src = ind["atr_pct"].to_numpy(); src_ts = pd.DatetimeIndex(src["timestamp"])
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = kl["timestamp"]; o, h, l, c = (kl[k].to_numpy() for k in ("open", "high", "low", "close"))
    pos_of = {t: i for i, t in enumerate(pd.DatetimeIndex(ts))}
    raw = {"short_term_return_z": {"bottom": ret3_z <= -2.5, "top": ret3_z >= 2.5, "ex": ret3_z},
           "demarker_extreme": {"bottom": dem <= 0.10, "top": dem >= 0.90, "ex": dem}}
    base = [x for x in FEATURE_COLUMNS if x != "is_bottom"]

    data = {}
    for name, spec in SPEC.items():
        gap = spec["gap"]; r = raw[name]; recs = []
        for side in ("bottom", "top"):
            mneg = side == "bottom"
            idx = np.flatnonzero(np.nan_to_num(r[side].astype(float), nan=0.0).astype(bool))
            idx = idx[(idx < len(src) - spec["horizon"] - gap - 1) &
                      (src_ts[idx].to_numpy() >= np.datetime64(START))]
            ex = r["ex"]
            for cl in clusters_of(idx, gap):
                first = cl[0]
                vals = [ex[j] for j in cl]
                k = int(np.argmin(vals)) if mneg else int(np.argmax(vals))
                recs.append({"pos_src": first, "side": side,
                             "continued": int(len(cl) > 1),        # 학습 라벨 (미래)
                             "first_is_anchor": int(k == 0)})      # 참고용
        d = pd.DataFrame(recs)
        for cn in base:
            d[cn] = ind[cn].to_numpy()[d["pos_src"].to_numpy()]
        d["is_bottom"] = (d["side"] == "bottom").astype(int)
        d["timestamp"] = src_ts[d["pos_src"].to_numpy()]
        d["atr"] = atr_src[d["pos_src"].to_numpy()]
        d = d[d["timestamp"].isin(pos_of) & np.isfinite(d["atr"]) & (d["atr"] > 0)].copy()
        d["pos"] = [pos_of[t] for t in d["timestamp"]]
        d = d.sort_values("pos").reset_index(drop=True)
        data[name] = d
        log(f"{name}: 버스트 첫봉 {len(d):,}건 | 이어짐 {d.continued.mean():.1%} | "
            f"첫봉이 앵커 {d.first_is_anchor.mean():.1%}")

    feats = base + ["is_bottom"]
    for name, d in data.items():
        tr = d["timestamp"] < VAL_START
        va = (d["timestamp"] >= VAL_START) & (d["timestamp"] < OOS_START)
        oo = (d["timestamp"] >= OOS_START) & (d["timestamp"] < HOLDOUT_START)
        X = d[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X[tr].median())
        y = d["continued"].to_numpy()
        m = HistGradientBoostingClassifier(random_state=SEED, **HP).fit(X[tr], y[tr])
        d["p_cont"] = m.predict_proba(X)[:, 1]
        log(f"{name}: 지속예측 AUC TRAIN {roc_auc_score(y[tr], d.p_cont[tr]):.4f} "
            f"VAL {roc_auc_score(y[va], d.p_cont[va]):.4f} OOS {roc_auc_score(y[oo], d.p_cont[oo]):.4f}")

    W = (("VAL", VAL_START, OOS_START), ("OOS", OOS_START, HOLDOUT_START))
    prio = {n: i for i, n in enumerate(TOP2)}
    QS = [0.2, 0.3, 0.5, 0.7]
    variants = ["B_무필터", "ORACLE_고립만"] + [f"F_저지속{int(q*100)}%" for q in QS]
    rows = []
    for tag in variants:
        tabs = {}
        for name, d in data.items():
            spec = SPEC[name]
            if tag == "B_무필터":
                sel = d
            elif tag == "ORACLE_고립만":
                sel = d[d.continued == 0]
            else:
                q = int(tag.split("저지속")[1].rstrip("%")) / 100
                thr = np.quantile(d.loc[(d.timestamp >= VAL_START) & (d.timestamp < OOS_START), "p_cont"], q)
                sel = d[d.p_cont <= thr]
            f = sel.sort_values("pos").reset_index(drop=True)
            for sgn, lb in ((1.0, "real"), (-1.0, "flip")):
                t = per_fire_outcomes(ts, o, h, l, c, f["pos"].to_numpy(np.int64),
                                      np.where(f["side"] == "bottom", 1.0, -1.0) * sgn,
                                      f["atr"].to_numpy(float), spec["horizon"],
                                      spec["sl"], spec["arm"], spec["trail"])
                t["signal"] = name
                t["decision_pos"] = [pos_of[x] for x in t["decision_ts"]]
                tabs[(name, lb)] = t
        for lb in ("real", "flip"):
            allc = pd.concat([tabs[(n, lb)] for n in TOP2], ignore_index=True)
            allc["prio"] = allc["signal"].map(prio)
            for wn, lo, hi in W:
                w = allc[(allc.decision_ts >= lo) & (allc.decision_ts < hi)]
                for k in (1, 2):
                    s = summarize(sequential_portfolio(w[w["signal"].isin(TOP2[:k])], prio), f"top{k}")
                    s.update({"window": wn, "kind": lb, "variant": tag}); rows.append(s)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True); df.to_csv(OUT_DIR / "burst_filter.csv", index=False)

    for arm in ("top1", "top2"):
        log(f"\n=== {arm} (real) ===")
        r = df[(df.kind == "real") & (df.arm == arm)]
        p = r.pivot_table(index="variant", columns="window", values=["n", "mean_bp", "pf"])
        fl = df[df.arm == arm].pivot_table(index="variant", columns=["window", "kind"], values="total_bp")
        p[("flip양창", "")] = np.where(
            (fl[("VAL", "real")] > np.maximum(fl[("VAL", "flip")], 0)) &
            (fl[("OOS", "real")] > np.maximum(fl[("OOS", "flip")], 0)), "O", "X")
        print(p.reindex(index=variants).reindex(columns=["VAL", "OOS", ""], level=1).round(2).to_string())
    (OUT_DIR / "contract.json").write_text(json.dumps(CONTRACT, indent=2))
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
