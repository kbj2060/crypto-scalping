#!/usr/bin/env python3
"""인과적 클라이맥스 예측기 -- 앵커 미래참조의 정면 돌파 시도 (2026-09-02).

배경
----
2026-09-02 승격 감사에서 `cluster_dedup` 앵커 선택이 미래참조임이 확인됐다. 인과 대안 중
C(후행최극단: 그 봉의 극단성이 직전 GAP봉 트리거 전부보다 크면 발동)는 **앵커를 반드시 포함하는
상위집합**인데도 손실이다(top2 VAL/OOS/HOLDOUT −5.12/−2.12/−4.03).

이유는 1슬롯 구조다. C의 여분 발동(= 아직 갱신될 봉)이 **슬롯을 먼저 점유**해서, 진짜 앵커가
왔을 때는 이미 손실 포지션에 들어가 있다. 거래 수는 거의 같은데(504 vs 486) 구성이 완전히 바뀐다.

그러므로 진짜 질문은 하나로 좁혀진다:
**"지금 이 봉이 이 버스트의 최종 극단인가"를 그 봉의 정보만으로 예측할 수 있는가?**

설계
----
  모집단 : 변형 C의 발동 (클러스터 내 후행 최극단 봉) -- 앵커를 100% 포함
  라벨   : 이 봉이 클러스터의 최종 앵커인가 (= 남은 클러스터에서 더 극단인 봉이 안 나옴)
  피처   : Tier0 23종 + 인과적 클러스터 상태 4종 (버스트 시작 후 경과봉/누적 트리거수/
           극단성 크기/직전봉 대비 극단성 변화)
  분할   : TRAIN < 2025-09-01, VAL 2025-09~12, OOS 2026-01~03  (HOLDOUT 미접촉)
  모델   : HistGradientBoosting (CPU, 시드 고정)
  판정   : AUC가 아니라 **경제성** -- 임계값을 VAL에서 고르고 OOS에 그대로 적용,
           1슬롯 순차 + 방향뒤집기 대조. A(미래참조)와 C(무필터)를 양옆에 둔다.
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

OUT_DIR = ROOT / "tmp/eth_causal_climax_predictor_20260902"
START = pd.Timestamp("2024-01-01")
SEED = 7529
SPEC = {"short_term_return_z": {"gap": 3, "sl": 3.0, "arm": 1.0, "trail": 0.1, "horizon": 12},
        "demarker_extreme": {"gap": 12, "sl": 2.0, "arm": 1.5, "trail": 0.1, "horizon": 8}}
TOP2 = ["short_term_return_z", "demarker_extreme"]
HP = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
          l2_regularization=1.0, early_stopping=False)


def log(m): print(f"[climax] {m}", flush=True)


def clusters_of(idx: np.ndarray, gap: int):
    """[(members...), ...] -- GAP 안 연속 트리거 묶음."""
    out, cur = [], [int(idx[0])]
    for i in idx[1:]:
        if int(i) - cur[-1] > gap:
            out.append(cur); cur = [int(i)]
        else:
            cur.append(int(i))
    out.append(cur)
    return out


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
    hold_end = ts.max()
    raw = {"short_term_return_z": {"bottom": ret3_z <= -2.5, "top": ret3_z >= 2.5, "ex": ret3_z},
           "demarker_extreme": {"bottom": dem <= 0.10, "top": dem >= 0.90, "ex": dem}}
    base_cols = [x for x in FEATURE_COLUMNS if x != "is_bottom"]
    EXTRA = ["bars_since_burst", "trig_count_so_far", "ex_mag", "ex_delta"]

    all_rows = {}
    for name, spec in SPEC.items():
        gap = spec["gap"]; r = raw[name]; rows = []
        for side in ("bottom", "top"):
            mneg = side == "bottom"
            idx = np.flatnonzero(np.nan_to_num(r[side].astype(float), nan=0.0).astype(bool))
            idx = idx[(idx < len(src) - spec["horizon"] - gap - 1) &
                      (src_ts[idx].to_numpy() >= np.datetime64(START))]
            ex = r["ex"]
            for cl in clusters_of(idx, gap):
                vals = [ex[j] for j in cl]
                anchor = cl[int(np.argmin(vals))] if mneg else cl[int(np.argmax(vals))]
                run = None
                for k, j in enumerate(cl):
                    v = ex[j]
                    is_new_extreme = run is None or (v < run if mneg else v > run)
                    if not is_new_extreme:
                        continue                      # 변형 C 모집단 = 후행 최극단 봉만
                    run = v
                    rows.append({
                        "pos_src": j, "side": side, "is_anchor": int(j == anchor),
                        "bars_since_burst": j - cl[0], "trig_count_so_far": k + 1,
                        "ex_mag": abs(float(v)) if name == "short_term_return_z" else abs(float(v) - 0.5),
                        "ex_delta": float(v - ex[cl[k - 1]]) if k > 0 else 0.0,
                    })
        d = pd.DataFrame(rows)
        for cname in base_cols:
            d[cname] = ind[cname].to_numpy()[d["pos_src"].to_numpy()]
        d["is_bottom"] = (d["side"] == "bottom").astype(int)
        d["timestamp"] = src_ts[d["pos_src"].to_numpy()]
        d["atr_pct_src"] = atr_src[d["pos_src"].to_numpy()]
        d = d[d["timestamp"].isin(pos_of)].reset_index(drop=True)
        all_rows[name] = d
        log(f"{name}: C 모집단 {len(d):,}건, 그중 앵커 {d.is_anchor.mean():.1%}")

    # ---- 학습 + 경제성 ----
    feats = base_cols + ["is_bottom"] + EXTRA
    scored = {}
    for name, d in all_rows.items():
        tr = d["timestamp"] < VAL_START
        va = (d["timestamp"] >= VAL_START) & (d["timestamp"] < OOS_START)
        oo = (d["timestamp"] >= OOS_START) & (d["timestamp"] < HOLDOUT_START)
        X = d[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X[tr].median())
        y = d["is_anchor"].to_numpy()
        m = HistGradientBoostingClassifier(random_state=SEED, **HP).fit(X[tr], y[tr])
        p = m.predict_proba(X)[:, 1]
        d["p_anchor"] = p
        log(f"{name}: 앵커예측 AUC TRAIN {roc_auc_score(y[tr], p[tr]):.4f} "
            f"VAL {roc_auc_score(y[va], p[va]):.4f} OOS {roc_auc_score(y[oo], p[oo]):.4f} "
            f"(기준선 앵커비율 TRAIN {y[tr].mean():.3f})")
        scored[name] = d

    # ---- arm 구성: A(미래참조) / C(무필터) / F_q(예측필터) ----
    W = (("VAL", VAL_START, OOS_START), ("OOS", OOS_START, HOLDOUT_START))
    QS = [0.0, 0.3, 0.5, 0.7, 0.8]
    prio = {n: i for i, n in enumerate(TOP2)}
    rows = []
    for tag in ["A_미래참조", "C_무필터"] + [f"F_상위{int((1-q)*100)}%" for q in QS if q > 0]:
        tabs = {}
        for name, d in scored.items():
            spec = SPEC[name]
            if tag == "A_미래참조":
                sel = d[d.is_anchor == 1]
            elif tag == "C_무필터":
                sel = d
            else:
                q = 1 - int(tag.split("상위")[1].rstrip("%")) / 100
                thr = np.quantile(d.loc[(d.timestamp >= VAL_START) & (d.timestamp < OOS_START),
                                        "p_anchor"], q)          # 임계는 VAL에서만
                sel = d[d.p_anchor >= thr]
            f = sel[["timestamp", "side", "atr_pct_src"]].copy()
            f["pos"] = [pos_of[t] for t in f["timestamp"]]
            f = f[np.isfinite(f.atr_pct_src) & (f.atr_pct_src > 0)].sort_values("pos").reset_index(drop=True)
            for sgn, lb in ((1.0, "real"), (-1.0, "flip")):
                t = per_fire_outcomes(ts, o, h, l, c, f["pos"].to_numpy(np.int64),
                                      np.where(f["side"] == "bottom", 1.0, -1.0) * sgn,
                                      f["atr_pct_src"].to_numpy(float), spec["horizon"],
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
    OUT_DIR.mkdir(parents=True, exist_ok=True); df.to_csv(OUT_DIR / "climax_filter.csv", index=False)

    for arm in ("top1", "top2"):
        log(f"\n=== {arm} (real) ===")
        r = df[(df.kind == "real") & (df.arm == arm)]
        p = r.pivot_table(index="variant", columns="window", values=["n", "mean_bp", "pf"])
        fl = df[df.arm == arm].pivot_table(index="variant", columns=["window", "kind"], values="total_bp")
        p[("flip양창", "")] = np.where(
            (fl[("VAL", "real")] > np.maximum(fl[("VAL", "flip")], 0)) &
            (fl[("OOS", "real")] > np.maximum(fl[("OOS", "flip")], 0)), "O", "X")
        print(p.reindex(columns=["VAL", "OOS", ""], level=1).round(2).to_string())
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
