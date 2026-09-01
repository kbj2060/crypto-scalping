#!/usr/bin/env python3
"""매 봉 스코어링 재설계 -- 사건당 1봉 샘플링 학습셋 구성 (GBM 프록시, CPU only).

9-6이 발견한 문제: 전 봉 라벨은 사건당 평균 3.2~5.8봉이 중복돼, 그대로 학습하면 **긴 사건이
짧은 사건보다 구조적으로 큰 가중치**를 받고(10봉 사건=10행, 1봉=1행), 상관 표본 때문에 유효
표본수도 과대추정된다. 사용자 결정(2026-09-01): **사건당 1봉만 샘플링**으로 간다.

## 설계 결정

- **대표봉 = 사건의 첫 봉**. 라이브에서는 어느 봉이 진짜 극값인지 미리 알 수 없으므로 "가장 이른
  시점에 감지"가 유일하게 정직한 선택이고, max fast_mult 같은 기준처럼 강한 결과 쪽으로 편향되지
  않는다.
- **음성 클래스(chop)도 같은 규칙으로 대칭 처리**. 양성만 dedup하면 기저율이 붕괴한다.
- 사건 정의는 9-6과 동일: 같은 클래스의 라벨봉이 GAP봉 이내로 인접하면 한 사건(side별 독립,
  중간에 ambiguous/invalid 봉이 끼어도 GAP 이내면 같은 사건).

## 비교하는 학습셋 구성 3종 (전부 동일 피쳐 Tier0 23개, 동일 라벨식 V0)

  S0_all_bars       : 모든 라벨봉 (9-3의 모델 B와 동일, 기준선)
  S1_event_sampled  : 사건당 첫 봉 1개만 (양성/음성 대칭)          <- 사용자 채택안
  S2_uniform_thin   : 클래스 무관, 직전 채택봉으로부터 GAP봉 이상 떨어진 봉만 (대조군)

## 평가는 두 모집단 모두에서

  - **VAL 전체봉**: 라이브 추론 모집단 그대로 -- 정직한 배포 성능
  - **VAL 사건샘플**: 상관 제거된 추정치 -- AUC의 통계적 신뢰도용

학습셋을 줄이면 성능이 떨어질 수 있으므로(표본수 효과) 5시드로 노이즈 바닥도 함께 잰다.

⚠️ 진단 전용: 라벨식 V0 그대로, 라이브 코드 변경 없음, OOS/HOLDOUT 미터치
(TRAIN < 2025-09-01, VAL 2025-09-01~2025-12-31).

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_v_rebound_every_bar_event_sampled_training_20260901.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FEAS_SCRIPT = ROOT / "scripts/research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py"
_fspec = importlib.util.spec_from_file_location("everybar_feas_evtsamp_20260901", FEAS_SCRIPT)
_feas = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(_feas)

AUDIT_SCRIPT = ROOT / "scripts/research_eth_v_rebound_every_bar_label_event_audit_20260901.py"
_aspec = importlib.util.spec_from_file_location("event_audit_20260901", AUDIT_SCRIPT)
_audit = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(_audit)

OUT_JSON = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/every_bar_event_sampled_training_report.json"

FEATURE_COLUMNS = _feas.FEATURE_COLUMNS
TRAIN_END = _feas.TRAIN_END
SEEDS = [20260901, 141592, 271828, 577215, 20260829]
GAP = 12  # matches 9-6's middle setting and this repo's existing dedup convention


def log(msg: str) -> None:
    print(f"[event_sampled] {msg}", flush=True)


def add_sampling_masks(long: pd.DataFrame, gap: int = GAP) -> pd.DataFrame:
    """Adds the sampling masks used by S1/S1b/S2.

    ⚠️ Asymmetry that a first implementation got wrong: a V_REBOUND *event* is a discrete, bounded
    thing (3-6 bars, 9-6's audit), but `chop` is a continuous BACKGROUND STATE occupying ~85% of
    labeled bars -- applying the same "consecutive same-class bars within gap = one event" rule to
    chop merges it into a handful of giant blobs (TRAIN: 156k chop bars -> 154 'events'), which
    blows the base rate from 14.9% to 96.8% and destroys the model. So:

      pos_event_first : positives only -- FIRST bar of each V_REBOUND event (the user's chosen rule)
      is_uniform_thin : class-agnostic -- a labeled bar is kept only if >= gap bars after the last
                        kept one. For chop this is the right notion of an independent observation
                        (a 100-bar chop stretch genuinely contains ~8 independent chop samples at
                        gap=12, it is not "one event").

    S1  = pos_event_first  OR (chop AND is_uniform_thin)          -- 1/event positives, thinned negatives
    S1b = pos_event_first  OR (chop AND random subsample sized to preserve the all-bars base rate)
    S2  = is_uniform_thin  (both classes)                          -- control
    """
    long = long.sort_values(["side", "timestamp"]).reset_index(drop=True)
    long["pos_event_first"] = False
    long["is_uniform_thin"] = False
    long["event_id"] = -1

    ev_counter = 0
    for side in ("bottom", "top"):
        side_pos = np.flatnonzero((long["side"] == side).to_numpy())
        sub = long.iloc[side_pos]
        bar_no = np.arange(len(sub))  # positional index within this side, time-ordered

        # --- positives: cluster into events, keep the FIRST bar of each ---
        pos_local = np.flatnonzero(sub["label"].to_numpy() == 1.0)
        if len(pos_local):
            for ev in _audit.cluster_events(bar_no[pos_local], gap):
                long.loc[side_pos[pos_local[np.searchsorted(bar_no[pos_local], ev[0])]], "pos_event_first"] = True
                long.loc[side_pos[pos_local[np.searchsorted(bar_no[pos_local], ev)]], "event_id"] = ev_counter
                ev_counter += 1

        # --- class-agnostic uniform thinning over labeled bars ---
        lab_local = np.flatnonzero(sub["label"].notna().to_numpy())
        last = -10**9
        keep = []
        for li in lab_local:
            if bar_no[li] - last >= gap:
                keep.append(side_pos[li])
                last = bar_no[li]
        long.loc[keep, "is_uniform_thin"] = True

    # S1c control: 1 bar per positive event too, but a RANDOM bar instead of the first -- separates
    # "1-per-event is the problem" from "first-bar specifically is the problem".
    long["pos_event_random"] = False
    rng_ev = np.random.default_rng(SEEDS[0])
    for eid, grp in long.loc[long["event_id"] >= 0].groupby("event_id").groups.items():
        long.loc[rng_ev.choice(np.asarray(grp)), "pos_event_random"] = True

    is_pos = long["label"].to_numpy() == 1.0
    long["s1_mask"] = long["pos_event_first"].to_numpy() | (~is_pos & long["is_uniform_thin"].to_numpy())
    long["s1c_mask"] = long["pos_event_random"].to_numpy() | (~is_pos & long["is_uniform_thin"].to_numpy())

    # S1b: same positives, but negatives randomly subsampled to restore the all-bars base rate
    all_bars_base = float(is_pos.mean())
    n_pos_kept = int(long["pos_event_first"].sum())
    n_neg_target = int(round(n_pos_kept * (1 - all_bars_base) / all_bars_base))
    neg_idx = np.flatnonzero(~is_pos)
    rng = np.random.default_rng(SEEDS[0])
    neg_keep = rng.choice(neg_idx, size=min(n_neg_target, len(neg_idx)), replace=False)
    s1b = np.zeros(len(long), dtype=bool)
    s1b[neg_keep] = True
    long["s1b_mask"] = long["pos_event_first"].to_numpy() | s1b
    return long


def auc_or_none(y, p):
    if len(np.unique(y)) < 2:
        return None
    return round(float(roc_auc_score(y, p)), 4)


def main() -> int:
    t0 = time.time()
    log("building all-bar long frame (features + V0 labels)...")
    long = _feas.build_long_frame()
    long = long.loc[long["label"].notna()].dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    log(f"labeled+clean bar-sides: {len(long)}")

    log(f"computing event/thinning masks (GAP={GAP})...")
    long = add_sampling_masks(long, GAP)

    tr = long.loc[long["split"] == "TRAIN"]
    va = long.loc[long["split"] == "VAL"]

    train_sets = {
        "S0_all_bars": tr,
        "S1_pos1perevent_negthin": tr.loc[tr["s1_mask"]],
        "S1b_pos1perevent_negbaserate": tr.loc[tr["s1b_mask"]],
        "S1c_pos1perevent_RANDOMbar_negthin": tr.loc[tr["s1c_mask"]],
        "S2_uniform_thin_both": tr.loc[tr["is_uniform_thin"]],
    }
    # Eval populations: the honest inference population (all bars) + a decorrelated one that
    # PRESERVES the base rate (uniform thinning, not the degenerate positives-only dedup).
    eval_sets = {
        "VAL_all_bars": va,
        "VAL_uniform_thin": va.loc[va["is_uniform_thin"]],
    }

    log("=== 학습셋/평가셋 구성 ===")
    sizes = {}
    for k, d in train_sets.items():
        sizes[k] = {"n": int(len(d)), "base_rate": round(float(d["label"].mean()), 4),
                    "n_pos": int(d["label"].sum())}
        log(f"  TRAIN {k:20s} n={len(d):7d}  base={d['label'].mean():.4f}  pos={int(d['label'].sum()):6d}")
    for k, d in eval_sets.items():
        sizes[k] = {"n": int(len(d)), "base_rate": round(float(d["label"].mean()), 4),
                    "n_pos": int(d["label"].sum())}
        log(f"  EVAL  {k:20s} n={len(d):7d}  base={d['label'].mean():.4f}  pos={int(d['label'].sum()):6d}")

    results = {}
    for tname, tset in train_sets.items():
        log(f"=== training {tname} (5 seeds) ===")
        per_eval = {k: [] for k in eval_sets}
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(random_state=sd, max_iter=300, early_stopping=True,
                                               validation_fraction=0.15)
            m.fit(tset[FEATURE_COLUMNS], tset["label"].to_numpy())
            for ename, eset in eval_sets.items():
                p = m.predict_proba(eset[FEATURE_COLUMNS])[:, 1]
                per_eval[ename].append(roc_auc_score(eset["label"].to_numpy(), p))
        entry = {}
        for ename, aucs in per_eval.items():
            a = np.array(aucs)
            entry[ename] = {"auc_mean": round(float(a.mean()), 4), "auc_std": round(float(a.std()), 4),
                            "auc_min": round(float(a.min()), 4), "auc_max": round(float(a.max()), 4)}
            log(f"  -> {ename:20s} AUC {a.mean():.4f} ± {a.std():.4f}  [{a.min():.4f}, {a.max():.4f}]")
        results[tname] = entry

    base = results["S0_all_bars"]
    for tname, e in results.items():
        for ename in eval_sets:
            e[ename]["delta_vs_S0"] = round(e[ename]["auc_mean"] - base[ename]["auc_mean"], 4)

    log("=== S0 대비 델타 (같은 평가셋 기준) ===")
    for tname, e in results.items():
        if tname == "S0_all_bars":
            continue
        for ename in eval_sets:
            log(f"  {tname:20s} on {ename:20s} delta={e[ename]['delta_vs_S0']:+.4f} "
                f"(std {e[ename]['auc_std']:.4f})")

    report = {
        "signal": "v_rebound_every_bar_event_sampled_training", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {
            "screening_only": True, "model": "HistGradientBoostingClassifier proxy (not TabPFN)",
            "tabpfn_training_done": False, "economic_cost_gate_done": False,
            "live_code_changed": False, "holdout_touched": False, "oos_touched": False,
            "label_formula": "V0 unchanged (current live label_side())",
            "splits": {"TRAIN": f"< {TRAIN_END}", "VAL": f"{TRAIN_END} .. 2026-01-01"},
            "purpose": ("Implement the user's chosen fix for 9-6's event-size weighting problem: "
                        "one bar per event (first bar), applied symmetrically to both classes."),
        },
        "event_gap_bars": GAP,
        "representative_bar_rule": "first bar of each event (live-realistic earliest detection, no outcome-strength bias)",
        "set_sizes": sizes,
        "seeds": SEEDS,
        "results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    log(f"total runtime: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
