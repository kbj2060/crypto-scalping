"""Phase 0 게이트 — 레짐 GBM 사이드카의 offline↔live 계산경로 파리티 감사.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계약: docs/model_contracts/omega461_regimegbm_rebuild_contract.md (Phase 0 킬 기준)

무엇을 재는가
------------
`build_omega461_regimegbm_cutoff_sidecar_20260909.py` 는 **캐노니컬 CSV 의 미리 계산된 136피쳐**를
그대로 읽어 레짐 확률을 낸다. 반면 라이브 스코어러(`live_regime_gbm3_signal_20260826.py`)는
**원시 kline/메트릭에서 `FeatureEngineer().process()` 로 피쳐를 직접 만든 뒤** 같은 모델에 넣는다.
두 경로가 갈리면 오프라인에서 학습·평가한 값과 라이브가 실제로 보는 값이 어긋난다 — 이 저장소가
반복해서 당한 부류의 결함이다(2026-08-26 `_with_raw_state12()` 누락으로 8개 컬럼이 조용히 median
으로 대체됐던 사고가 같은 계열).

그래서 이 감사는 **같은 타임스탬프에서** 두 경로를 붙여 비교한다:
  경로 A (오프라인) = 캐노니컬 CSV 의 136피쳐 그대로
  경로 B (라이브 재현) = 캐노니컬 CSV 의 **원시 컬럼만** 떼어내 `FeatureEngineer().process()` →
                        `_with_raw_state12()` → medians 대체 (라이브 스코어러와 동일 순서/동일 인자)
원시 컬럼 목록은 `live_regime_gbm3_signal_20260826.py:107-113` 의 eth_raw_cols/btc_raw_cols 를
그대로 가져온다.

⚠️ 이 감사가 덮지 못하는 것: 바이낸스에서 실제로 fetch 한 원시값이 캐노니컬 CSV 의 원시값과
같은지는 여기서 확인하지 않는다(네트워크·아카이브 시점 차이는 별개 축). 여기서 잡는 것은
**"같은 원시값에서 출발했을 때 피쳐 파이프라인이 갈리는가"** 하나다.

킬 기준: 136피쳐 중 하나라도 상대오차가 TOL 을 넘거나, 6개 출력 컬럼의 최대 절대차가 TOL 을
넘으면 FAIL — 계약상 Phase 1 로 넘어가지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from features.engineering import FeatureEngineer  # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import TRAIN_CSVS  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

OUT_DIR = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"
MODEL = OUT_DIR / "regime_cut2509_model.joblib"

# live_regime_gbm3_signal_20260826.py:107-113 그대로
ETH_RAW = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
           "trades", "taker_buy_base", "taker_buy_quote",
           "sum_open_interest_value", "sum_toptrader_long_short_ratio",
           "count_long_short_ratio", "last_funding_rate"]
BTC_RAW = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]

TOL = 1e-6
WARMUP_BARS = int(sys.argv[2]) if len(sys.argv) > 2 else 4320   # 기본 15일(라이브 DAYS_BACK)
COMPARE_BARS = 2000     # 워머업 뒤 비교 구간

# 비교 창은 반드시 **이 라인의 평가 구간 안**에서 잡는다. 파일 꼬리(최근 확장분)를 쓰면
# 2026-09-09 첫 실행처럼 캐노니컬 CSV 자체가 오염된 구간을 재게 되어, 파이프라인 결함이
# 아닌 데이터 결함으로 게이트가 FAIL 한다. 기본값은 이 라인의 OOS 끝이다.
WINDOW_END = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2026-02-28T23:55:00")


def main() -> int:
    pay = joblib.load(MODEL)
    cols, medians = pay["feature_cols"], pd.Series(pay["feature_medians"])
    prefix = pay["prefix"]

    df = pd.concat([pd.read_csv(p_, low_memory=False, parse_dates=["timestamp"]) for p_ in TRAIN_CSVS],
                   ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    need = WARMUP_BARS + COMPARE_BARS
    df = df[df["timestamp"] <= WINDOW_END].tail(need).reset_index(drop=True)
    tail = df.tail(COMPARE_BARS).index
    print(f"[창끝] {WINDOW_END}  워머업 {WARMUP_BARS}봉")
    print(f"[프레임] 워머업 {WARMUP_BARS} + 비교 {COMPARE_BARS} = {len(df)}봉  "
          f"비교구간 {df.timestamp.iloc[tail[0]]} ~ {df.timestamp.iloc[tail[-1]]}", flush=True)

    # --- 경로 A: 캐노니컬 CSV 의 계산된 피쳐 그대로 ---
    a_feats = _with_raw_state12(df.copy())

    # --- 경로 B: 원시 컬럼에서 라이브와 같은 순서로 재구성 ---
    missing_raw = [c for c in ETH_RAW + BTC_RAW if c not in df.columns]
    if missing_raw:
        raise RuntimeError(f"원시 컬럼 누락 {missing_raw} -- 라이브 경로를 재현할 수 없음")
    b_feats = FeatureEngineer().process(df[ETH_RAW].copy(), df[BTC_RAW].copy())
    b_feats = _with_raw_state12(b_feats)

    a_feats = a_feats.set_index("timestamp")
    b_feats = b_feats.set_index("timestamp")
    common = a_feats.index.intersection(b_feats.index)
    cmp_idx = common[-COMPARE_BARS:]
    print(f"[정렬] 공통 타임스탬프 {len(common)}봉, 비교 대상 {len(cmp_idx)}봉", flush=True)

    def mat(fr):
        x = fr.reindex(index=cmp_idx, columns=cols).apply(pd.to_numeric, errors="coerce")
        return x.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)

    A, B = mat(a_feats), mat(b_feats)
    denom = np.maximum(np.abs(A.to_numpy()), 1.0)
    rel = np.abs(A.to_numpy() - B.to_numpy()) / denom
    per_col = pd.Series(rel.max(axis=0), index=cols).sort_values(ascending=False)
    n_bad = int((per_col > TOL).sum())
    print(f"\n[피쳐 파리티] 136개 중 상대오차>{TOL:g} 인 컬럼: {n_bad}개", flush=True)
    print("  상위 8개 최대 상대오차:", flush=True)
    for c, v in per_col.head(8).items():
        print(f"    {c:34s} {v:.3e}", flush=True)

    # --- 6개 출력 컬럼 비교 ---
    def six(fr):
        p = pay["model"].predict_proba(fr)
        p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
        s = np.sort(p, axis=1)
        out = {f"{prefix}{n}_prob": p[:, i] for i, n in enumerate(pay["classes"])}
        out[f"{prefix}confidence"] = p.max(axis=1)
        out[f"{prefix}margin"] = s[:, -1] - s[:, -2]
        out[f"{prefix}entropy"] = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1) / np.log(3.0)
        return pd.DataFrame(out, index=cmp_idx)

    SA, SB = six(A), six(B)
    six_diff = (SA - SB).abs().max().sort_values(ascending=False)
    print(f"\n[출력 파리티] 6컬럼 최대 절대차:", flush=True)
    for c, v in six_diff.items():
        print(f"    {c:38s} {v:.3e}", flush=True)
    argmax_agree = float((SA[[f"{prefix}{n}_prob" for n in pay['classes']]].to_numpy().argmax(1)
                          == SB[[f"{prefix}{n}_prob" for n in pay['classes']]].to_numpy().argmax(1)).mean())
    print(f"    argmax 라벨 일치율 {argmax_agree*100:.3f}%", flush=True)

    passed = bool(n_bad == 0 and float(six_diff.max()) <= TOL)
    print(f"\n{'='*60}\nPhase 0 파리티 게이트: {'PASS' if passed else 'FAIL'}", flush=True)

    report = {"tolerance": TOL, "compare_bars": int(len(cmp_idx)),
              "compare_range": [str(cmp_idx[0]), str(cmp_idx[-1])],
              "feature_cols_over_tol": n_bad,
              "worst_features": {k: float(v) for k, v in per_col.head(15).items()},
              "six_column_max_abs_diff": {k: float(v) for k, v in six_diff.items()},
              "argmax_agreement": argmax_agree, "passed": passed,
              "not_covered": "바이낸스 실제 fetch 원시값 vs 캐노니컬 CSV 원시값 일치 여부(별개 축)"}
    (OUT_DIR / "phase0_parity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"산출물: {OUT_DIR}/phase0_parity_report.json", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
