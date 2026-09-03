#!/usr/bin/env python3
"""BTC 증거신호 7종 **8시드 견고성** -- XRP와 같은 절차를 BTC에 적용.

## 왜

2026-09-03 XRP 시드 감사에서 CLAUDE.md Seed-Diversity 게이트 요건 미달이 드러나 8시드로
채웠다. **BTC는 그 감사를 받지 않았다** -- 같은 날 XRP만 돌렸다.

실태 확인 결과 BTC도 동일하다: 7종 전부 `SEEDS = [20260829, 141592, 271828, 577215]`
**4시드**로 N>=5 미달이고, 프로모션 리포트에 시드 리스트 기록도 없다.

⚠️이건 이 저장소에서 반복된 실패 유형이다 -- 2026-09-02 앵커 미래참조 감사도 ETH만 받고
BTC는 누락돼, 다음 날 "3종 생존"이 전부 무효로 뒤집혔다.
⇒ **한 자산에서 만든 감사는 같은 빌더를 쓰는 모든 자산에 즉시 적용한다.**

## 설계 (XRP `audit_xrp_evidence_signals_seed_robustness_20260903.py`와 동일)

  · 발동/라벨 빌드는 결정론적 -> 신호당 1회만 만들고 **TabPFN만 시드별 재적합**
  · 시드 **8종, 랜덤 추출**(고정 간격 증가 금지 -- Sigma3-1h 전례). XRP와 같은 리스트를 쓴다.
  · **VAL + OOS만**. ⚠️HOLDOUT은 이미 1회 소진(demarker 0.7286 등) -- 시드별 재평가는 재노출.
  · 각 모듈의 **자기 상수**(배포 셀)를 그대로 쓴다. 셀 탐색이 아니라 시드 견고성만 본다.

⚠️tz 규약이 모듈마다 다르다(orthogonal/liquidity_sweep/fib는 tz-aware) -- 분할 상수에 맞춘다.

## 판정 (실행 전 고정)

  각 신호가 **모든 시드에서 OOS AUC > 0.5**(부호 일치).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_C = importlib.util.spec_from_file_location(
    "btcctx", ROOT / "scripts/build_btc_evidence_signal_frozen_contexts_20260902.py")
_c = importlib.util.module_from_spec(_C)
_C.loader.exec_module(_c)

CAND_CSV = _c.CAND_CSV
EXPECTED_ROWS = 277_191                    # BTC 후보 CSV 행수 -- 자산 오염 가드
OUT = ROOT / "data/research/btc_evidence_signals_seed_robustness_20260903.json"

# XRP 감사와 **같은** 8시드 (랜덤 추출, 고정 간격 증가 아님)
SEEDS = [20260903, 811453, 30011, 947, 260317, 5387291, 68041, 1299709]

# 기록된 4시드 HOLDOUT AUC (참고용 -- 재측정하지 않는다)
RECORDED_HOLDOUT = {"demarker_extreme": 0.7286, "kalman_deviation_meanrev": 0.6709,
                    "short_term_return_z": 0.6443, "taker_delta_climax": 0.6276,
                    "orthogonal_combo": 0.5933, "fib_extension_exhaustion": 0.5657,
                    "liquidity_sweep": 0.5214}


def log(m): print(f"[btc-seed] {m}", flush=True)


def build_one(name, rel, builder, prep, kind):
    """동결 컨텍스트 빌더의 호출 규약을 그대로 복제. ⚠️`prep[0]` 로더는 호출하지 않는다
    (그 모듈 자신의 경로를 읽는다 -- XRP에서 BTC 오염이 났던 지점. 여기선 BTC가 맞지만
    같은 규율을 유지한다)."""
    mod = _c.load_mod(rel)
    # ⚠️tz 규약이 모듈마다 다르다 -- 모듈 자신의 `VAL_START`에서 읽어서 맞춘다.
    # (demarker/kalman/str_z/taker는 naive, orthogonal/liquidity_sweep/fib는 tz-aware.
    #  한쪽으로 고정하면 반드시 다른 쪽이 "offset-naive and offset-aware" TypeError로 터진다.)
    want_aware = getattr(mod.VAL_START, "tzinfo", None) is not None
    f = pd.read_csv(CAND_CSV)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
    if not want_aware:
        f["timestamp"] = f["timestamp"].dt.tz_localize(None)
    for pname in prep[1:]:
        fnp = getattr(mod, pname, None)
        if fnp is not None:
            f = fnp(f)
    if abs(len(f) - EXPECTED_ROWS) > 200:
        raise RuntimeError(f"{name}: 행수 {len(f):,} != BTC 기대치 {EXPECTED_ROWS:,}")
    fn = getattr(mod, builder)
    if kind == "demarker":
        g = _c.GRID_CHOSEN["demarker"]
        out = fn(f, g["horizon"], g["k"], mod.CLUSTER_GAP)
    elif kind == "kalman":
        g = _c.GRID_CHOSEN["kalman"]
        f["kalman_dev_z"] = mod.compute_kalman_dev_z(f["close"].to_numpy())
        bt = (f["kalman_dev_z"] <= -2.0).fillna(False).to_numpy()
        tt = (f["kalman_dev_z"] >= 2.0).fillna(False).to_numpy()
        out = fn(f, bt, tt, g["horizon"], g["k"], mod.CLUSTER_GAP)
    else:
        out = fn(f)
    fires = out[0] if isinstance(out, tuple) else out
    ts = pd.to_datetime(fires["timestamp"])
    if want_aware and ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    elif not want_aware and ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    return mod, fires.assign(timestamp=ts)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier

    log(f"BTC 후보 CSV {CAND_CSV.name}")
    log(f"시드 {len(SEEDS)}종 (랜덤 추출, XRP 감사와 동일): {SEEDS}")
    log("⚠️VAL+OOS만 평가 -- HOLDOUT은 1회 소진됨, 시드별 재평가는 재노출이다")

    rep = {"asset": "BTCUSDT", "seeds": SEEDS, "n_seeds": len(SEEDS),
           "seed_selection": "랜덤 추출(고정 간격 증가 아님)",
           "holdout_touched": False, "recorded_holdout_auc_4seed": RECORDED_HOLDOUT,
           "note": "원본은 7종 전부 4시드([20260829,141592,271828,577215]) -- N>=5 미달이었다",
           "signals": {}}

    for name, rel, builder, prep, kind in _c.SIGNALS:
        try:
            mod, fires = build_one(name, rel, builder, prep, kind)
        except Exception as e:                                       # noqa: BLE001
            log(f"{name:<26} ⚠️{type(e).__name__}: {str(e)[:70]}")
            rep["signals"][name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        feats = [c for c in mod.FEATURE_COLUMNS if c in fires.columns]
        tr = fires.loc[fires["timestamp"] < mod.VAL_START].reset_index(drop=True)
        splits = {"VAL": fires.loc[(fires["timestamp"] >= mod.VAL_START)
                                   & (fires["timestamp"] < mod.OOS_START)].reset_index(drop=True),
                  "OOS": fires.loc[(fires["timestamp"] >= mod.OOS_START)
                                   & (fires["timestamp"] < mod.HOLDOUT_START)].reset_index(drop=True)}
        log("")
        log(f"=== {name} ===")
        log(f"  TRAIN {len(tr):,} (hit {tr['hit'].mean():.4f}) | "
            f"VAL {len(splits['VAL'])} / OOS {len(splits['OOS'])} | 피쳐 {len(feats)}")
        if len(tr) < 50 or tr["hit"].nunique() < 2:
            log("  ⚠️TRAIN 부족 -- 건너뜀")
            rep["signals"][name] = {"error": f"insufficient train ({len(tr)})"}
            continue
        per = {"VAL": [], "OOS": []}
        rows = []
        for sd in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
            clf.fit(tr[feats], tr["hit"].to_numpy().astype(int))
            row = {"seed": sd}
            for s_, ev in splits.items():
                if len(ev) < 30 or ev["hit"].nunique() < 2:
                    row[s_] = None; continue
                p = np.concatenate([clf.predict_proba(ev[feats].iloc[k:k+20000])[:, 1]
                                    for k in range(0, len(ev), 20000)])
                a = float(roc_auc_score(ev["hit"].astype(int), p))
                row[s_] = round(a, 4); per[s_].append(a)
            rows.append(row)
            log(f"  seed={sd:<9} VAL {row['VAL']}  OOS {row['OOS']}")
        res = {"n_train": int(len(tr)), "train_hit_rate": float(tr["hit"].mean()),
               "per_seed": rows}
        for s_ in ("VAL", "OOS"):
            if not per[s_]:
                res[s_] = {"error": "no eval"}; continue
            v = np.array(per[s_])
            res[s_] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                       "min": float(v.min()), "max": float(v.max()),
                       "all_above_half": bool((v > 0.5).all())}
            log(f"  {s_} 평균 {v.mean():.4f} ± {v.std(ddof=1):.4f}  "
                f"[{v.min():.4f}, {v.max():.4f}]  전부>0.5: {'✅' if (v > 0.5).all() else '❌'}")
        rep["signals"][name] = res

    log("")
    log("=" * 74)
    log("판정 (사전 고정: 모든 시드에서 OOS AUC > 0.5)")
    log("=" * 74)
    allok, decided = True, 0
    for n, v in rep["signals"].items():
        if "error" in v or "error" in v.get("OOS", {}):
            log(f"  {n:<26} ⚠️판정 불가"); continue
        decided += 1
        ok = v["OOS"]["all_above_half"]; allok &= ok
        log(f"  {n:<26} OOS {v['OOS']['mean']:.4f} ± {v['OOS']['std']:.4f}  "
            f"[{v['OOS']['min']:.4f}, {v['OOS']['max']:.4f}]  {'✅' if ok else '❌'}")
    log("")
    log(f"⇒ {'✅**부호 일관성 통과** (N=8, 랜덤 추출)' if allok and decided else '⚠️**일부 시드에서 0.5 이하**'}")
    rep["all_oos_above_half"] = bool(allok and decided)
    rep["n_decided"] = decided
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
