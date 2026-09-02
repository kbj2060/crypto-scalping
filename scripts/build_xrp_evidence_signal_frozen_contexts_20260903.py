#!/usr/bin/env python3
"""BTC 증거신호 7종의 **라이브 서빙용 동결 컨텍스트** 일괄 생성.

## 왜 필요한가

2026-09-01에 BTC 증거신호 7종의 그리드스크린 + TabPFN 메타라벨 검증이 끝났지만
**서빙 아티팩트가 없다** -- 연구 스크립트가 매번 라벨을 새로 만들고 학습한 뒤 리포트만 남겼다.
ETH는 신호별 동결 컨텍스트로 라이브가 도는데(`live_evidence_signal_metalabel_20260829.py`),
BTC엔 그게 없어 라이브 스코어러를 못 만든다.

## 설계 -- 라벨 정의를 재구현하지 않는다

각 연구 스크립트의 `build_fires_and_features()`(orthogonal_combo는 `build_fires`,
demarker는 `build_final_fires`)를 **그대로 import**해서 쓴다. 재구현하면 BTC 전용으로 튜닝된
HIT정의/H/K/GAP이 조용히 어긋난다.

## BTC 전용 파라미터 (ETH와 다름 -- 2026-09-01 그리드스크린이 자산별로 재선정)

    신호                       HIT정의        H     K      GAP
    demarker_extreme          touch MFE      8     0.70    6
    kalman_deviation_meanrev  touch MFE     10     2.5     6
    liquidity_sweep           터치+되돌림    20     2.0     6
    short_term_return_z       MAE 상한       6     2.0    12
    taker_delta_climax        종가기준       6     2.0     3
    fib_extension_exhaustion  종가기준      10     2.75    6
    orthogonal_combo          touch MFE      8     2.0     6

## 산출

신호별 `{signal}_frozen_context.csv` (TRAIN 구간 fires + hit 라벨 + FEATURE_COLUMNS) +
`contexts_report.json`. 라이브 스코어러가 이걸 읽어 TabPFN을 적합한다.

⚠️TRAIN(<2025-09-01)만 사용. VAL/OOS/HOLDOUT 미터치. 라이브 코드 변경 없음.

## ⚠️XRP 포팅 (2026-09-03)

BTC 동결 컨텍스트 빌더의 자산 상수 + **확정 셀**만 바꾼 포팅.

XRP 확정 셀(BTC와 전부 다름):

    demarker_extreme          H=2  K=1.5   (격자 경계 확장 3회 끝에 확정, HOLDOUT AUC 0.6759)
    kalman_deviation_meanrev  H=5  K=2.0   (HOLDOUT 0.6223)
    short_term_return_z       H=12 K=1.5   touch_mae_capped (0.6132)
    taker_delta_climax        H=9  K=1.5   touch_giveback_sustained, **해상 18봉** (0.6091)
    orthogonal_combo          H=8  K=2.0   touch_mfe (0.5599)

⚠️`liquidity_sweep`(0.4886)과 `fib_extension_exhaustion`(0.4738)은 HOLDOUT AUC가 **무작위 미만**
이라 서빙에서 제외했다. BTC에서도 `liquidity_sweep`이 유일하게 무작위(0.5214)였다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CAND_CSV = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/xrp_5m_evidence_signal_candidates_tier0.csv"
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
EXPECTED_ROWS = 272_490      # XRP 후보 CSV 행수 -- 자산 오염 가드용
# ⚠️tz 규약이 모듈마다 다르다. demarker/taker는 naive `START`와 비교하고(원본 load_tier0가
# naive를 준다), orthogonal_combo는 tz-aware를 기대한다. 한쪽으로 고정하면 반드시 한쪽이 터진다.
TZ_AWARE = {"orthogonal_combo": True}
OUTDIR = ROOT / "data/labels/xrp_5m_evidence_signal_live_contexts_20260903"

# ⚠️스크립트마다 (a)프레임 전처리 함수와 (b)빌더 시그니처가 다르다. 원본 main()의 호출부를
# 직접 읽어 맞췄다 -- 재구현하면 BTC 전용 HIT정의/H/K/GAP이 조용히 어긋난다.
#   prep: 프레임 준비 함수 이름들 (순서대로 적용)
#   args: 빌더에 넘길 추가 인자 (원본 main()이 넘기는 것과 동일)
# demarker/kalman은 main()이 격자로 H/K를 고르는데, 확정값은 리포트의 label_definition에
# 기록돼 있다 -- demarker H=8/K=0.70, kalman H=10/K=3.5(스크립트 상수 K=2.5가 아님).
# ⚠️생존 5종만. `liquidity_sweep`(HOLDOUT AUC 0.4886)과 `fib_extension_exhaustion`(0.4738)은
# **무작위 미만**이라 서빙에서 제외했다 -- 근거: docs/experiments/xrp_evidence_signal_and_regime_20260903.md
SIGNALS = [
    ("demarker_extreme", "research_xrp_demarker_extreme_metalabel_tabpfn_20260903.py",
     "build_final_fires", ["load_tier0", "add_missing_features"], "demarker"),
    ("kalman_deviation_meanrev", "research_xrp_kalman_deviation_meanrev_metalabel_tabpfn_20260903.py",
     "build_fires_and_features", ["load_tier0", "add_missing_features"], "kalman"),
    ("short_term_return_z", "research_btc_short_term_return_z_metalabel_tabpfn_20260901.py",
     "build_fires_and_features", ["load_frame", "add_derived_features"], "plain"),
    ("taker_delta_climax", "research_btc_taker_delta_climax_metalabel_tabpfn_20260901.py",
     "build_fires_and_features", ["load_tier0", "add_missing_features"], "plain"),
    ("orthogonal_combo", "research_btc_orthogonal_combo_metalabel_tabpfn_20260901.py",
     "build_fires", ["load_tier0", "add_missing_features"], "plain"),
]
# 격자 선정형 신호의 확정 파라미터 (리포트 label_definition에서 추출)
# ⭐XRP 자신의 격자 결과(BTC와 전부 다르다)
GRID_CHOSEN = {"demarker": {"horizon": 2, "k": 1.5},
               "kalman": {"horizon": 5, "k": 2.0}}


def log(m): print(f"[xrp-ctx] {m}", flush=True)


def load_mod(rel: str):
    spec = importlib.util.spec_from_file_location(f"m_{Path(rel).stem}", ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    t0 = time.time()
    log(f"후보 CSV 로드: {CAND_CSV.name}")
    frame = pd.read_csv(CAND_CSV)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    log(f"  {len(frame):,}행 ({frame.timestamp.min()} ~ {frame.timestamp.max()})")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    report = {"asset": "XRPUSDT", "source": str(CAND_CSV.relative_to(ROOT)),
              "train_end": str(TRAIN_END), "signals": {},
              "note": "라벨 정의는 각 연구 스크립트의 빌더를 그대로 import -- 재구현 없음"}

    for name, script, fn_name, prep, kind in SIGNALS:
        log("")
        log(f"########## {name} ##########")
        try:
            mod = load_mod(script)
            # 각 스크립트 자체 전처리를 그대로 쓴다(nyse_open_flag/er_24/atr_pct 등을 여기서 만든다)
            # ⚠️⚠️2026-09-03: 여기서 **BTC 데이터를 읽는 사고**가 실제로 났다.
            # prep[0](`load_tier0`/`load_frame`)은 **그 모듈 자신의 TIER0_PATH**를 읽는데,
            # str_z/taker/orthogonal은 BTC 모듈을 재사용하므로 BTC CSV를 읽어버렸다
            # (행수 277,191 = BTC, hit률이 BTC와 소수점까지 동일해서 발각).
            # ⇒ **로더는 절대 호출하지 않는다.** XRP 후보 CSV를 직접 읽고 변환 prep만 태운다.
            # ⚠️tz 규약이 모듈마다 다르다: demarker/taker는 naive `START`와 비교하고,
            # orthogonal_combo는 tz-aware를 기대한다. 한쪽으로 고정하면 반드시 한쪽이 터진다.
            # naive를 먼저 시도하고 tz 관련 TypeError면 aware로 재시도한다.
            def _load(tz_aware: bool):
                g = pd.read_csv(CAND_CSV)
                g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
                if not tz_aware:
                    g["timestamp"] = g["timestamp"].dt.tz_localize(None)
                for pname in prep[1:]:
                    fnp = getattr(mod, pname, None)
                    if fnp is not None:
                        g = fnp(g)
                return g

            f = _load(tz_aware=TZ_AWARE.get(name, False))
            if f is None:
                f = frame.copy()
            # ⚠️일부 스크립트(demarker/taker)는 naive `START`와 비교한다 -- 원본 load_tier0가
            # naive를 주므로 여기서 tz를 붙이면 "can't compare offset-naive and offset-aware"로
            # 터진다. 전처리 결과를 그대로 두고, split 시점에만 tz를 맞춘다.
            if "timestamp" in f.columns:
                f["timestamp"] = pd.to_datetime(f["timestamp"])
            log(f"  전처리 완료: {len(f):,}행, 컬럼 {len(f.columns)}")
            # ⭐자산 오염 가드 -- 다른 자산 CSV를 읽으면 여기서 죽는다
            if abs(len(f) - EXPECTED_ROWS) > 200:
                raise RuntimeError(
                    f"{name}: 행수 {len(f):,} != XRP 기대치 {EXPECTED_ROWS:,} "
                    f"-- 다른 자산 데이터를 읽었을 가능성")

            fn = getattr(mod, fn_name)
            if kind == "demarker":
                g = GRID_CHOSEN["demarker"]
                out = fn(f, g["horizon"], g["k"], mod.CLUSTER_GAP)
                log(f"  격자확정 H={g['horizon']} K={g['k']} GAP={mod.CLUSTER_GAP}")
            elif kind == "kalman":
                g = GRID_CHOSEN["kalman"]
                # 원본 main() 664~668행과 동일: kalman_dev_z를 계산해 ±2.0 임계로 트리거를 만든다
                f["kalman_dev_z"] = mod.compute_kalman_dev_z(f["close"].to_numpy())
                bt = (f["kalman_dev_z"] <= -2.0).fillna(False).to_numpy()
                tt = (f["kalman_dev_z"] >= 2.0).fillna(False).to_numpy()
                out = fn(f, bt, tt, g["horizon"], g["k"], mod.CLUSTER_GAP)
                log(f"  격자확정 H={g['horizon']} K={g['k']} GAP={mod.CLUSTER_GAP}")
            else:
                out = fn(f)
            fires = out[0] if isinstance(out, tuple) else out
            if "hit" not in fires.columns:
                log(f"  ⚠️'hit' 컬럼 없음 (컬럼: {list(fires.columns)[:8]}) -- 건너뜀")
                report["signals"][name] = {"error": "no hit column"}
                continue
            feats = [c for c in getattr(mod, "FEATURE_COLUMNS", []) if c in fires.columns]
            if not feats:
                log("  ⚠️FEATURE_COLUMNS 없음 -- 건너뜀")
                report["signals"][name] = {"error": "no FEATURE_COLUMNS"}
                continue
            ts = pd.to_datetime(fires["timestamp"])
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC")
            fires["timestamp"] = ts
            tr = fires.loc[fires["timestamp"] < TRAIN_END].reset_index(drop=True)
            if len(tr) < 200 or tr["hit"].nunique() < 2:
                log(f"  ⚠️TRAIN {len(tr)}행 / 클래스 {tr['hit'].nunique()} -- 부족, 건너뜀")
                report["signals"][name] = {"error": f"insufficient train ({len(tr)})"}
                continue
            keep = ["timestamp", "side", "hit"] + feats
            ctx = tr[keep].copy()
            csv = OUTDIR / f"{name}_frozen_context.csv"
            ctx.to_csv(csv, index=False)
            params = {k: getattr(mod, k) for k in
                      ("HORIZON", "H", "K", "ATR_HIT_MULT", "K_MULT", "K_LOSS_MULT",
                       "GAP", "CLUSTER_GAP", "CLUSTER_GAP_MERGE", "GIVEBACK_CEILING",
                       "ZSCORE_WINDOW", "FULL_WINDOW")
                      if hasattr(mod, k)}
            info = {"rows": int(len(ctx)), "hit_rate": round(float(ctx["hit"].mean()), 4),
                    "features": feats, "n_features": len(feats),
                    "bottom": int((ctx["side"] == "bottom").sum()),
                    "top": int((ctx["side"] == "top").sum()),
                    "asset_params": params, "artifact": str(csv.relative_to(ROOT)),
                    "range": [str(ctx.timestamp.min()), str(ctx.timestamp.max())]}
            report["signals"][name] = info
            log(f"  ✅ TRAIN {len(ctx):,}행 (bottom {info['bottom']:,}/top {info['top']:,}) "
                f"hit률 {info['hit_rate']:.4f}  피쳐 {len(feats)}개")
            log(f"     BTC 파라미터: {params}")
        except Exception as e:                                    # noqa: BLE001
            log(f"  ❌실패: {type(e).__name__}: {e}")
            log("     " + traceback.format_exc().splitlines()[-3])
            report["signals"][name] = {"error": f"{type(e).__name__}: {e}"}

    ok = [k for k, v in report["signals"].items() if "error" not in v]
    log("")
    log(f"=== 완료: {len(ok)}/{len(SIGNALS)} 신호 ===")
    for k in ok:
        v = report["signals"][k]
        log(f"  ✅ {k:28s} {v['rows']:>6,}행  hit {v['hit_rate']:.4f}")
    for k, v in report["signals"].items():
        if "error" in v:
            log(f"  ❌ {k:28s} {v['error']}")
    report["ok_signals"] = ok
    report["runtime_sec"] = round(time.time() - t0, 1)
    (OUTDIR / "contexts_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUTDIR/'contexts_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
