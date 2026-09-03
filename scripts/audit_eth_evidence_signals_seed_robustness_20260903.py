#!/usr/bin/env python3
"""ETH 증거신호 **8시드 견고성** -- XRP·BTC와 같은 절차를 ETH에 (3자산 파리티 완성).

## 왜

2026-09-03에 XRP 5종·BTC 7종은 CLAUDE.md Seed-Diversity 게이트 요건(N>=5 랜덤추출,
OOS 부호 일치, **시드 리스트 리포트 기록**)을 8시드로 채웠는데 **ETH는 안 받았다.**

⇒ 오늘 세 번째로 나온 "한 자산만 감사하고 다른 자산은 누락" 패턴이다. ETH도 채운다.

## 설계

ETH는 BTC/XRP 같은 통합 frozen-contexts SIGNALS 스펙이 없고 신호별 빌더 규약이 제각각이다.
⇒ **이미 만들어진 라벨(fires CSV)을 쓴다.** 라벨은 결정론적이라 시드와 무관하고,
시드 검증이 묻는 건 **모델 적합의 시드 민감도**뿐이다.

  · 시드 **8종, 랜덤 추출** -- XRP·BTC 감사와 **같은 리스트**
  · **VAL + OOS만**. ⚠️HOLDOUT은 1회 소진 -- 시드별 재평가는 재노출이다.
  · 피쳐는 각 CSV의 숫자형 컬럼에서 라벨/메타 컬럼을 제외해 구성한다.

⚠️**ETH OOS 절단은 레짐 경로에만 해당한다**: `OOS_END = 2026-02-17`(`data/eth_5m_1year.csv`
커버리지)은 **레짐 conditional-lift 경로**의 상수다. **증거신호 fires CSV는 온전하다**
(str_z 실측: 2026-08-27까지, OOS 439건이 2026-03-31까지 덮는다).
⇒ 그래도 **각 분할의 실제 종료일을 출력**해 신호별로 확인한다.

## 판정 (실행 전 고정)

  각 신호가 **모든 시드에서 OOS AUC > 0.5**(부호 일치).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score   # noqa: E402

from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,   # 배포 빌더가 실제로 쓰는 Tier0 피쳐 목록 -- 재정의 금지
)

OUT = ROOT / "data/research/eth_evidence_signals_seed_robustness_20260903.json"
L = ROOT / "data/labels"

# XRP·BTC 감사와 **같은** 8시드 (랜덤 추출, 고정 간격 증가 아님)
SEEDS = [20260903, 811453, 30011, 947, 260317, 5387291, 68041, 1299709]
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

# 배포 중인 ETH 신호의 fires CSV (호메로스 README 배포 열 기준)
FIRES = {
    # ⚠️demarker/kalman은 **한 디렉토리**에 같이 있다(kalman_demarker 합동 작업), 파일명에 셀이 박혀 있다
    "demarker_extreme": L / "eth_5m_kalman_demarker_metalabel_20260831/eth_5m_demarker_extreme_metalabel_features_H8_GAP12_K0.7.csv",
    "kalman_deviation_meanrev": L / "eth_5m_kalman_demarker_metalabel_20260831/eth_5m_kalman_deviation_meanrev_metalabel_features_H12_GAP12_K2.5.csv",
    "short_term_return_z": L / "eth_5m_short_term_return_z_metalabel_20260829/eth_5m_short_term_return_z_metalabel_features.csv",
    "taker_delta_climax": L / "eth_5m_taker_delta_climax_metalabel_20260829/eth_5m_taker_delta_climax_metalabel_features.csv",
    "orthogonal_combo": L / "eth_5m_orthogonal_combo_metalabel_20260830/eth_5m_orthogonal_combo_metalabel_features.csv",
    "smt_divergence": L / "eth_5m_smt_divergence_metalabel_20260831/eth_5m_smt_divergence_metalabel_features.csv",
    "liquidity_sweep": L / "eth_5m_liquidity_sweep_topdown_metalabel_20260830/eth_5m_liquidity_sweep_topdown_metalabel_features_H30_GAP12_K4.0.csv",
    "fib_extension_exhaustion": L / "eth_5m_fib_extension_exhaustion_metalabel_20260831/eth_5m_fib_extension_exhaustion_metalabel_FINAL_features.csv",
}
# ⛔**블랙리스트 방식은 여기서 한 번 실패했다**(2026-09-03 1차 실행):
# smt/fib CSV에 `move_atr_mult`(진입 후 실현 초과폭 ATR배수)와 `mae_atr_mult`가 들어 있는데
# 라벨이 정의상 `move_atr_mult >= K`라 **AUC가 8시드 전부 정확히 1.0000**으로 나왔다.
# 완벽분리는 통과가 아니라 누수 신호다.
# ⇒ 화이트리스트로 바꾼다: 배포 빌더가 실제로 쓰는 `FEATURE_COLUMNS`(Tier0)에
#    신호별 추가 피쳐만 더한다. 라벨/결과 컬럼은 애초에 후보에 들어올 수 없다.
EXTRA_FEATS = {"dem", "kalman_dev_z"}          # 신호별 고유 피쳐(배포 빌더 기준)
LEAK_HINTS = {"move_atr_mult", "mae_atr_mult", "entry", "pred_dir_ret",
              "fast_mult", "giveback", "barrier_end_i", "tb_reason"}
NOT_FEATURE = {"is_bottom"}                    # 배포 빌더가 명시적으로 건너뛰는 컬럼
LEAK_AUC = 0.99                                # 이 이상이면 통과가 아니라 누수로 판정


def log(m): print(f"[eth-seed] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier

    log(f"시드 {len(SEEDS)}종 (랜덤 추출, XRP·BTC 감사와 동일): {SEEDS}")
    log("⚠️VAL+OOS만 -- HOLDOUT은 1회 소진, 시드별 재평가는 재노출이다")
    rep = {"asset": "ETHUSDT", "seeds": SEEDS, "n_seeds": len(SEEDS),
           "seed_selection": "랜덤 추출(고정 간격 증가 아님)", "holdout_touched": False,
           "note": "라벨은 결정론적이라 기존 fires CSV를 그대로 쓴다(모델 적합의 시드 민감도만 측정)",
           "signals": {}}

    for name, path in FIRES.items():
        if not path.exists():
            log(f"{name:<26} ⚠️CSV 없음: {path.name}")
            rep["signals"][name] = {"error": f"missing {path.name}"}
            continue
        df = pd.read_csv(path)
        # ⭐demarker/kalman 계보는 라벨 컬럼이 `hit_plain`이다(그 계보 자체 규약).
        # 배포 스크립트(research_eth_kalman_demarker_tabpfn_confirm_20260831.py)와 동일하게
        # hit_plain을 라벨로 쓰고 exclude_v2로 걸러내지 않는다.
        if "hit" not in df.columns and "hit_plain" in df.columns:
            df = df.rename(columns={"hit_plain": "hit"})
        if "hit" not in df.columns or "timestamp" not in df.columns:
            log(f"{name:<26} ⚠️hit/timestamp 컬럼 없음")
            rep["signals"][name] = {"error": "no hit/timestamp"}
            continue
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df = df.assign(timestamp=ts)
        feats = [c for c in FEATURE_COLUMNS if c in df.columns and c not in NOT_FEATURE]
        feats += [c for c in EXTRA_FEATS if c in df.columns]
        feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
        leaks = sorted(LEAK_HINTS & set(df.columns))
        if leaks:
            log(f"  ⓘ 결과성 컬럼 {leaks} 은 피쳐에서 제외됨(화이트리스트 방식)")
        df = df.dropna(subset=feats + ["hit"]).reset_index(drop=True)
        tr = df.loc[df["timestamp"] < VAL_START].reset_index(drop=True)
        sp = {"VAL": df.loc[(df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)]
                       .reset_index(drop=True),
              "OOS": df.loc[(df["timestamp"] >= OOS_START) & (df["timestamp"] < HOLDOUT_START)]
                       .reset_index(drop=True)}
        log("")
        log(f"=== {name} ===")
        log(f"  CSV {len(df):,}행 {ts.min().date()}~{ts.max().date()} | 피쳐 {len(feats)}")
        log(f"  TRAIN {len(tr):,}(hit {tr['hit'].mean():.4f}) | "
            f"VAL {len(sp['VAL'])}(~{sp['VAL']['timestamp'].max().date() if len(sp['VAL']) else '-'}) | "
            f"OOS {len(sp['OOS'])}(~{sp['OOS']['timestamp'].max().date() if len(sp['OOS']) else '-'})")
        if len(tr) < 50 or tr["hit"].nunique() < 2:
            log("  ⚠️TRAIN 부족 -- 건너뜀")
            rep["signals"][name] = {"error": f"insufficient train ({len(tr)})"}
            continue
        per, rows = {"VAL": [], "OOS": []}, []
        for sd in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
            clf.fit(tr[feats], tr["hit"].to_numpy().astype(int))
            row = {"seed": sd}
            for s_, ev in sp.items():
                if len(ev) < 30 or ev["hit"].nunique() < 2:
                    row[s_] = None; continue
                p = np.concatenate([clf.predict_proba(ev[feats].iloc[k:k+20000])[:, 1]
                                    for k in range(0, len(ev), 20000)])
                a = float(roc_auc_score(ev["hit"].astype(int), p))
                row[s_] = round(a, 4); per[s_].append(a)
            rows.append(row)
            log(f"  seed={sd:<9} VAL {row['VAL']}  OOS {row['OOS']}")
        res = {"n_train": int(len(tr)), "train_hit_rate": float(tr["hit"].mean()),
               "n_features": len(feats), "per_seed": rows,
               "csv_range": [str(ts.min().date()), str(ts.max().date())],
               "oos_end_actual": str(sp["OOS"]["timestamp"].max().date()) if len(sp["OOS"]) else None}
        for s_ in ("VAL", "OOS"):
            if not per[s_]:
                res[s_] = {"error": "no eval"}; continue
            v = np.array(per[s_])
            leaked = bool((v >= LEAK_AUC).any())
            res[s_] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                       "min": float(v.min()), "max": float(v.max()),
                       "all_above_half": bool((v > 0.5).all()),
                       "leak_suspected": leaked}
            if leaked:
                log(f"  ⛔{s_} AUC>={LEAK_AUC} -- 통과가 아니라 **누수**다. 피쳐 목록 재점검 필요")
            log(f"  {s_} 평균 {v.mean():.4f} ± {v.std(ddof=1):.4f}  "
                f"[{v.min():.4f}, {v.max():.4f}]  전부>0.5: {'✅' if (v > 0.5).all() else '❌'}")
        rep["signals"][name] = res

    log(""); log("=" * 76)
    log("판정 (사전 고정: 모든 시드에서 OOS AUC > 0.5)")
    log("=" * 76)
    allok, dec = True, 0
    for n, v in rep["signals"].items():
        if "error" in v or "error" in (v.get("OOS") or {}):
            log(f"  {n:<26} ⚠️판정 불가"); continue
        dec += 1
        ok = v["OOS"]["all_above_half"] and not v["OOS"].get("leak_suspected")
        allok &= ok
        log(f"  {n:<26} OOS {v['OOS']['mean']:.4f} ± {v['OOS']['std']:.4f}  "
            f"[{v['OOS']['min']:.4f}, {v['OOS']['max']:.4f}]  OOS끝 {v['oos_end_actual']}  "
            f"{'✅' if ok else ('⛔누수의심' if v['OOS'].get('leak_suspected') else '❌')}")
    log("")
    log(f"⇒ {'✅**부호 일관성 통과**' if allok and dec else '⚠️일부 미달'} (판정가능 {dec}종)")
    rep["all_oos_above_half"] = bool(allok and dec)
    rep["n_decided"] = dec
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
