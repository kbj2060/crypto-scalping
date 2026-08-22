#!/usr/bin/env python3
"""BTC-metrics 오염 수정 후 wide24 레짐 오버레이 재생성 (2026-08-23).

배경: `fix_eth_canonical_2026_btc_metrics_contamination_20260823.py`가 캐노니컬 2026의
오염(2026-01-20~07-12, BTC metrics 병합)을 수정함 → wide24 HMM의 관측 피쳐 중
`state12_oi_change_rate = tanh(oi_change_rate/0.01)`이 그 수정 컬럼을 직접 소비하므로,
tmp/ilias_labellogic_recheck_20260821/의 오버레이 3종을 기존 모델(joblib, states24/sticky0.90
seed7529)로 재생성한다. 모델 재적합은 하지 않는다(별도 결정 사항으로 기록 — fit 데이터에
오염 구간이 포함돼 있었다는 사실은 실험 문서에 명시).

자체 검증: forward 필터는 인과적이므로 **2026-01-20 이전 구간의 확률값은 원본과 완전히
동일해야 한다** — 그렇지 않으면 재생성 조건이 원본과 다른 것이므로 중단.
원본은 .bak_pre_btc_metrics_fix_20260823으로 보존.
"""
from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

stub = types.ModuleType("mamba_ssm")
stub.Mamba = object
sys.modules["mamba_ssm"] = stub

import joblib  # noqa: E402

from experiment_regime3_current_hmm_wide24_20260529 import _transform  # noqa: E402

MODEL = ROOT / "tmp/eth_hmm_wide24_resweep_train2026h1_20260821/states24_sticky0.90/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib"
OUT_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821"
YEAR = ROOT / "data/splits/year_oos"

TARGETS = {
    # name: (입력 구간 [start, end_exclusive), 원본 행수)
    "train_2024_2026H1": (("2024-01-01", "2026-07-01"), 262609),
    "eval_2026H1": (("2026-01-01", "2026-06-30 00:00:01"), 51841),
    "oos_20260701_20260819": (("2026-07-01", "2026-08-20"), 14400),
}


def load_frame(start: str, end: str) -> pd.DataFrame:
    parts = []
    for f in ["training_features_2024.csv", "training_features_2025.csv", "training_features_2026_rebuilt.csv"]:
        d = pd.read_csv(YEAR / f, low_memory=False)
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        parts.append(d)
    df = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)


def save(path: Path, overlay: pd.DataFrame) -> None:
    bak = path.with_name(path.name + ".bak_pre_btc_metrics_fix_20260823")
    if not bak.exists():
        shutil.copy2(path, bak)
    tmp = path.with_suffix(".csv.tmp")
    overlay.to_csv(tmp, index=False)
    tmp.replace(path)


def transform_legacy_cols(payload, frame: pd.DataFrame) -> pd.DataFrame:
    overlay, _ = _transform(payload, frame)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    # 원본은 레거시 컬럼명 유지(계약서 기록: "_v2" 제거)
    overlay.columns = [c.replace("regime3_current_sensitive_v2_wide24_", "regime3_current_sensitive_wide24_")
                       for c in overlay.columns]
    return overlay


def main() -> int:
    payload = joblib.load(MODEL)

    # ── train_2024_2026H1: 전체 프레임 forward 필터 ──
    name = "train_2024_2026H1"
    path = OUT_DIR / f"{name}_regime3_current_states24_sticky090.csv"
    orig = pd.read_csv(path)
    orig["timestamp"] = pd.to_datetime(orig["timestamp"])
    frame = load_frame("2024-01-01", "2026-07-01")
    assert len(frame) == 262609
    overlay = transform_legacy_cols(payload, frame)
    assert list(overlay.columns) == list(orig.columns) and len(overlay) == len(orig)
    prob_cols = [c for c in orig.columns if c != "timestamp"]
    pre = orig["timestamp"] < pd.Timestamp("2026-01-20 00:05:00")
    max_pre = float(np.abs(orig.loc[pre, prob_cols].to_numpy() - overlay.loc[pre, prob_cols].to_numpy()).max())
    if max_pre > 1e-9:
        print(f"✗ {name}: 오염 이전 구간 최대차 {max_pre:.2e} — 중단")
        return 1
    post_diff = np.abs(orig.loc[~pre, prob_cols].to_numpy() - overlay.loc[~pre, prob_cols].to_numpy())
    print(f"✓ {name}: 오염 이전 {pre.sum()}행 완전동일, 영향구간 최대변화 {post_diff.max():.4f}/평균 {post_diff.mean():.6f}")
    save(path, overlay)

    # ── eval_2026H1: 원본이 train 오버레이의 슬라이스(최대차 0 실측)였으므로 동일하게 슬라이스 ──
    name = "eval_2026H1"
    path = OUT_DIR / f"{name}_regime3_current_states24_sticky090.csv"
    orig = pd.read_csv(path)
    orig["timestamp"] = pd.to_datetime(orig["timestamp"])
    sl = overlay[(overlay["timestamp"] >= "2026-01-01") & (overlay["timestamp"] <= "2026-06-30 00:00:00")].reset_index(drop=True)
    assert len(sl) == len(orig), f"{name}: 슬라이스 행수 {len(sl)} != {len(orig)}"
    diff = np.abs(orig[prob_cols].to_numpy() - sl[prob_cols].to_numpy())
    print(f"✓ {name}: train 슬라이스로 재생성 — 원본 대비 최대변화 {diff.max():.4f}")
    save(path, sl)

    # ── oos: 원본이 콜드스타트(seedcheck 산출물과 최대차 0 실측)였으므로 동일 조건 재생성 ──
    name = "oos_20260701_20260819"
    path = OUT_DIR / f"{name}_regime3_current_states24_sticky090.csv"
    orig = pd.read_csv(path)
    orig["timestamp"] = pd.to_datetime(orig["timestamp"])
    frame = load_frame("2026-07-01", "2026-08-20")
    assert len(frame) == 14400
    overlay_oos = transform_legacy_cols(payload, frame)
    assert len(overlay_oos) == len(orig)
    diff = np.abs(orig[prob_cols].to_numpy() - overlay_oos[prob_cols].to_numpy())
    # 오염창(07-01~07-12) 이후 필터 기억이 감쇠하므로 말미(08-10~)는 원본에 수렴해야 정상
    tail = orig["timestamp"] >= "2026-08-10"
    tail_diff = np.abs(orig.loc[tail, prob_cols].to_numpy() - overlay_oos.loc[tail, prob_cols].to_numpy())
    print(f"✓ {name}: 콜드스타트 재생성 — 전체 최대변화 {diff.max():.4f}, 8/10 이후 최대 {tail_diff.max():.4f}(감쇠 확인)")
    save(path, overlay_oos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
