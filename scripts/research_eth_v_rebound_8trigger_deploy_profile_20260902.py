#!/usr/bin/env python3
"""8트리거 일치 모델의 **배포 프로파일** -- 임계값별 빈도/정밀도/소진율.

## 왜

사용자 요청(2026-09-02): 대시보드 칩을 매 봉 모델에서 **8트리거 일치 모델**로 교체.
근거는 정직한 AUC가 최고(VAL 0.7551 / OOS 0.7654)라는 점, 목적은 "모양을 맞춘다면 보고 매매".

배포하려면 두 가지를 먼저 알아야 한다:

  1. **임계값** -- 8트리거풀은 기저율이 다르다(0.2280 vs 매 봉 0.1515). 매 봉의 0.60을
     그대로 쓰면 의미가 달라진다. 임계값별 **빈도(건/일)·정밀도·무신호일 비율**을 낸다.
  2. ⭐**소진율** -- 매 봉 모델은 호출 시점에 목표가 이미 121~128% 소진돼 있었다
     (사용자가 본 "5분봉 하나에 익절"의 정체). 8트리거는 증거신호 발동 시점에 채점하므로
     **더 이를 수도, 같을 수도** 있다. 이게 "보고 매매"의 성패를 가른다.

부수: 8트리거 각각의 발동 기여도(어느 신호가 호출을 만드는가)와 커버리지.

⚠️읽기 전용. VAL/OOS만 사용(HOLDOUT 미터치).
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


_s1 = _load("s1_dp", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_feas, _vs = _s1._feas, _s1._vs
FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
ALL9 = _feas.ALL9
EIGHT = [t for t in ALL9 if t != "local_extreme"]
DEPLOYED = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
ATR_MULT = 1.50
CONTEXT_N, SEED = 18000, 20260829
THRESHOLDS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_8trigger_deploy_profile_20260902/report.json"


def log(m): print(f"[profile] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    import torch
    log(f"cuda: {torch.cuda.is_available()}")
    log(f"8트리거: {', '.join(EIGHT)}")

    _s1.VAL_END = OOS_END
    log("building frame ...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **DEPLOYED)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **DEPLOYED)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"

    # 8트리거 게이트 + 기여도
    parts = []
    for side in ("bottom", "top"):
        cols = {t: sig[f"{side}_{t}"].fillna(False).to_numpy() for t in EIGHT}
        g8 = np.any(list(cols.values()), axis=0)
        d = {"timestamp": sig["timestamp"].to_numpy(), "side": side, "gate8": g8}
        for t in EIGHT:
            d[f"fire_{t}"] = cols[t]
        parts.append(pd.DataFrame(d))
    long = long.merge(pd.concat(parts, ignore_index=True), on=["timestamp", "side"], how="left")
    long["gate8"] = long["gate8"].fillna(False)

    # 소진율 = (open[i+1] − 앵커) / (1.5×ATR)
    ts_pos = {t: i for i, t in enumerate(sig["timestamp"].dt.tz_localize(None).to_numpy())}
    long["pos"] = [ts_pos.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    long = long.loc[long["pos"] >= 0].reset_index(drop=True)
    low, high, op, atr = (sig[c].to_numpy() for c in ("low", "high", "open", "atr"))
    pre_atr = _vs.shifted_at(atr, -1)
    i = long["pos"].to_numpy().astype(int)
    dn = long["is_downside"].to_numpy() == 1
    anc = np.where(dn, low[i], high[i])
    ent = op[np.minimum(i + 1, len(op) - 1)]
    long["consumed"] = np.where(dn, ent - anc, anc - ent) / (ATR_MULT * pre_atr[i])

    pool = long.loc[long["gate8"]].copy()
    log("")
    log("=== 커버리지 ===")
    cov = {}
    for spn in ("VAL", "OOS"):
        a = int((long["split"] == spn).sum()); b = int(((long["split"] == spn) & long["gate8"]).sum())
        cov[spn] = {"all_rows": a, "gated_rows": b, "pct": round(b / a * 100, 2)}
        log(f"  {spn}: 전체 {a:,}행 중 8트리거 발동 {b:,}행 (**{b/a*100:.2f}%**)")

    log("")
    log("=== 트리거별 기여 (게이트풀 내 발동 비율) ===")
    contrib = {}
    for t in EIGHT:
        r = float(pool[f"fire_{t}"].mean())
        contrib[t] = round(r * 100, 2)
        log(f"  {t:28s} {r*100:5.2f}%")

    lab = pool.loc[pool["label"].notna()]
    tr = lab.loc[lab["split"] == "TRAIN"]
    log("")
    log(f"=== 학습 (TRAIN 라벨행 {len(tr):,}, 라벨률 {tr['label'].mean():.4f}) ===")
    rng = np.random.default_rng(SEED)
    ctx = tr.iloc[np.sort(rng.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))]
    clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
    clf.fit(ctx[FEATURE_COLUMNS], ctx["label"].to_numpy())

    prof = {}
    for spn in ("VAL", "OOS"):
        s = pool.loc[pool["split"] == spn].copy()
        s["p"] = np.concatenate([clf.predict_proba(s[FEATURE_COLUMNS].iloc[k:k+20000])[:, 1]
                                 for k in range(0, len(s), 20000)])
        sl_ = s.loc[s["label"].notna()]
        auc = float(roc_auc_score(sl_["label"], sl_["p"])) if sl_["label"].nunique() == 2 else None
        days = (s["timestamp"].max() - s["timestamp"].min()).total_seconds() / 86400
        log("")
        log(f"=== {spn}  풀 {len(s):,}행 / {days:.0f}일  AUC {auc:.4f} ===")
        log(f"  {'임계값':>6s} {'호출':>6s} {'건/일':>6s} {'정밀도':>7s} {'무신호일':>7s} "
            f"{'소진중앙':>8s} {'소진100%↑':>9s}")
        rows = []
        for thr in THRESHOLDS:
            sel = s.loc[s["p"] >= thr]
            if len(sel) < 10:
                continue
            lb = sel.loc[sel["label"].notna()]
            prec = float(lb["label"].mean()) if len(lb) else np.nan
            dd = sel["timestamp"].dt.date.nunique()
            alld = s["timestamp"].dt.date.nunique()
            cons = sel["consumed"].to_numpy()
            cons = cons[np.isfinite(cons)]
            r = {"threshold": thr, "calls": int(len(sel)),
                 "per_day": round(len(sel) / max(days, 1), 2),
                 "precision": round(prec, 4) if prec == prec else None,
                 "labeled_pct": round(len(lb) / len(sel) * 100, 1),
                 "dry_day_pct": round((1 - dd / alld) * 100, 1),
                 "consumed_median": round(float(np.median(cons)) * 100, 1) if len(cons) else None,
                 "consumed_ge100_pct": round(float((cons >= 1).mean()) * 100, 1) if len(cons) else None}
            rows.append(r)
            log(f"  {thr:6.2f} {r['calls']:6,d} {r['per_day']:6.2f} {r['precision']:7.4f} "
                f"{r['dry_day_pct']:6.1f}% {r['consumed_median']:7.1f}% {r['consumed_ge100_pct']:8.1f}%")
        prof[spn] = {"auc": round(auc, 4) if auc else None, "rows": int(len(s)),
                     "days": round(days, 1), "thresholds": rows}

    log("")
    log("=== ⭐매 봉 모델과의 소진율 비교 (핵심) ===")
    log("  매 봉 모델 호출(thr 0.60): VAL 121% / OOS 128%  (100%↑ 68%)")
    for spn in ("VAL", "OOS"):
        r6 = next((x for x in prof[spn]["thresholds"] if abs(x["threshold"] - 0.60) < 1e-9), None)
        if r6:
            log(f"  8트리거 호출(thr 0.60): {spn} {r6['consumed_median']}%  "
                f"(100%↑ {r6['consumed_ge100_pct']}%)")

    report = {"signal": "v_rebound_8trigger_deploy_profile", "asset": "ETHUSDT",
              "scope": {"triggers": EIGHT, "excluded": "local_extreme",
                        "label": "giveback (wick 앵커, 1.5xATR, t_sustain 0.20, 60분)",
                        "context_n": CONTEXT_N, "seed": SEED,
                        "holdout_touched": False, "live_code_changed": False},
              "coverage": cov, "trigger_contribution_pct": contrib,
              "train_rows": int(len(tr)), "train_label_rate": round(float(tr["label"].mean()), 4),
              "profile": prof, "runtime_sec": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
