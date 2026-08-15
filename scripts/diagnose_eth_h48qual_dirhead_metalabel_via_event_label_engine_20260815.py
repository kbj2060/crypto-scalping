"""h48qual TabM parent(direction_head)의 실제 out-of-sample 예측을 core/event_label_engine.py의
메타라벨링 경로(side 인자)에 넣어보는 진단 스크립트.

부모 아티팩트: tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h384_20260811_v2_e40_r30000_s179660/
  - validation_predictions_q050.csv (2025-10-01~2025-12-31) + oos_predictions_q050.csv (2026-01-01~2026-02-28)
  - train_predictions_q050.csv는 의도적으로 쓰지 않음 — in-sample 예측을 side로 쓰면 1차 모델의
    과적합 확신이 메타라벨에 새는 leakage이기 때문(core/event_label_engine.py의 generate_labels
    docstring 참고). validation/oos는 이 부모 모델이 학습 중 보지 못한 진짜 holdout이라 안전하다.

side = direction_head의 raw argmax(dir_action: 0=CASH/1=LONG/2=SHORT, quality_head 게이팅 이전).
quality_threshold로 이미 게이팅된 final_action이 아니라 dir_action을 쓴 이유: 이게 "1차 모델이
고른 방향"이라는 메타라벨링의 표준 정의(primary side)에 더 가깝고, final_action은 표본이
9:1로 SHORT에 쏠려(계약서에 이미 문서화된 현상) 표본수가 너무 적어짐.

주의: 여기서 나오는 "적중률"은 이 스크립트가 새로 고른 배리어(pt=sl=2.0×EWMA-vol, 48bar) 기준이며,
h48qual 실제 배포 배리어(h48_conservative: tp=1.2/sl=0.8×ATR96, 48bar)와 다르다. 즉 이 결과는
"이 엔진의 메타라벨링 배관이 실제 모델 출력으로 정상 동작하는가"의 검증이지, h48qual에 새로운
방향성 edge가 있다는 주장이 아니다 — direction_head/h48qual의 방향 edge 없음은 이미
docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md와 관련 experiment
문서 10건 이상에서 N>=5 시드로 반복 확인된 상태다(저장소 CLAUDE.md의 Fresh-Forward 규칙에 따라
이 진단 결과 하나로 그 결론을 뒤집지 않는다).
"""
import importlib.util
import os

import numpy as np
import pandas as pd

# core/__init__.py는 core.binance_client(python-binance 필요)를 즉시 import한다. 이 dev
# 셸에는 python-binance가 없어(서버 전용 의존성으로 보임) `from core.x import y` 형태의 일반
# 패키지 import가 막힌다 — core/__init__.py를 건드리는 대신 이 파일 하나만 패키지를 거치지
# 않고 직접 로드해 우회한다(다른 core/* 소비 스크립트도 이 dev 셸에서는 동일하게 막힘).
_spec = importlib.util.spec_from_file_location(
    "event_label_engine", os.path.join(os.path.dirname(__file__), "..", "core", "event_label_engine.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LabelEngineConfig, TripleBarrierConfig, generate_labels = _mod.LabelEngineConfig, _mod.TripleBarrierConfig, _mod.generate_labels

ARTIFACT_DIR = (
    "tmp/causal_regen_20260516/"
    "omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h384_20260811_v2_e40_r30000_s179660"
)


def _read_dir_action(path: str) -> pd.DataFrame:
    # validation_predictions는 컬럼명에 "_oof_"가 끼어있고(k-fold out-of-fold 예측),
    # oos_predictions는 순수 forward 추론이라 "_oof_"가 없다 — 두 네이밍을 모두 흡수.
    header = pd.read_csv(path, nrows=0).columns
    col = next(c for c in header if c.endswith("dir_action"))
    df = pd.read_csv(path, usecols=["timestamp", col])
    return df.rename(columns={col: "dir_action"})


def load_parent_side() -> pd.Series:
    val = _read_dir_action(f"{ARTIFACT_DIR}/validation_predictions_q050.csv")
    oos = _read_dir_action(f"{ARTIFACT_DIR}/oos_predictions_q050.csv")
    preds = pd.concat([val, oos], ignore_index=True)
    preds["timestamp"] = pd.to_datetime(preds["timestamp"])

    side_map = {0: np.nan, 1: 1.0, 2: -1.0}  # 0=CASH(제외) 1=LONG 2=SHORT
    preds["side"] = preds["dir_action"].map(side_map)
    return preds.set_index("timestamp")["side"]


def main():
    df = pd.read_csv("data/eth_5m_1year.csv")[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    parent_side = load_parent_side()
    print(f"부모 예측 holdout 구간: {parent_side.index.min()} ~ {parent_side.index.max()}  (bar 수 {len(parent_side):,})")
    print(f"side 분포(0=CASH 제외 전): LONG={int((parent_side == 1).sum()):,}  SHORT={int((parent_side == -1).sum()):,}  CASH={int(parent_side.isna().sum()):,}")

    ts_overlap = parent_side.index.isin(df["timestamp"]).sum()
    print(f"\ntimestamp range 정합(순수 겹침): {ts_overlap:,} / {len(parent_side):,}"
          f"  (부족분은 대부분 data/eth_5m_1year.csv가 2026-02-17까지만 있어 OOS 뒷부분 11일이 빠졌기 때문)")

    side_aligned = df.set_index("timestamp").index.map(parent_side.to_dict())
    side_aligned = pd.Series(side_aligned, index=df.index, dtype=float)
    n_matched = side_aligned.notna().sum()
    print(f"정합 후 side가 LONG/SHORT(비-CASH)인 bar: {n_matched:,}  ← 이게 실제로 meta-label 대상이 되는 표본 수")

    cfg = LabelEngineConfig(
        event_method="all_bars",  # 부모가 매 bar 스코어링했으므로 CUSUM 재샘플링 없이 그 bar 전부 사용
        barrier=TripleBarrierConfig(pt_mult=2.0, sl_mult=2.0, max_hold=48),
        compute_trend_scan=False,  # 이 진단은 side를 이미 부모가 정하므로 불필요
    )
    meta = generate_labels(df, cfg, side=side_aligned)

    print(f"\n메타라벨 이벤트 수: {len(meta):,}")
    print(f"side 분포: LONG={int((meta['side'] == 1).sum()):,}  SHORT={int((meta['side'] == -1).sum()):,}")
    print(f"전체 적중률(label==1): {(meta['label'] == 1).mean():.4f}")
    for s, name in [(1, "LONG"), (-1, "SHORT")]:
        sub = meta[meta["side"] == s]
        if len(sub) > 0:
            print(f"  {name} 적중률: {(sub['label'] == 1).mean():.4f}  (n={len(sub):,}, 평균 uniqueness={sub['weight_uniqueness'].mean():.3f})")

    # validation과 oos 구간을 나눠서도 확인 (fresh-forward 정신에 맞춰 별도 확인, 하나로 뭉개지 않음)
    for label, lo, hi in [("validation(2025-10~12)", "2025-10-01", "2025-12-31"),
                           ("oos(2026-01~02)", "2026-01-01", "2026-02-28")]:
        mask = (meta["event_time"] >= lo) & (meta["event_time"] <= hi)
        sub = meta[mask]
        if len(sub) > 0:
            print(f"{label}: n={len(sub):,}  적중률={(sub['label'] == 1).mean():.4f}")


if __name__ == "__main__":
    main()
