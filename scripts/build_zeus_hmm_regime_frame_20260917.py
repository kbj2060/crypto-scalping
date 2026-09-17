#!/usr/bin/env python3
"""Zeus — **HMM 레짐 6열** 프레임 생성 (2026-09-17). balnobb 과 «같은 자»로 붙이기 위한 것.

사용자 지시: balnobb 으로 진행하되 **이전 HMM 분류기가 더 좋으면 알려달라**.

라이브 산출식(`omega4_6_2_source_parent_live.py:225-236`)을 그대로 따른다:
    xz = scaler.transform(raw24) → model.filter_proba(xz) → _class_proba(state, state_class_matrix)
    → 정규화 → {prefix}{bull,bear,chop}_prob · confidence=max · margin=top1-top2
    → entropy = -(p·log p).sum() / log(3)        ← log(3) 정규화
⭐**HMM 24 피쳐가 Zeus 프레임에 전부 있다**(확인 완료) -- 피쳐 재계산 없이 그대로 먹인다.

⚠️**공정성 경고**: balnobb 의 적합 창(배포본 2024-01~2026-06 · 재적합본 2022-01~2023-12)이
**테스트 4폴드를 전부 덮는다.** HMM 아티팩트도 마찬가지일 수 있다. 그래서 이 비교는
「이 결함을 양쪽에 똑같이 두고 **어느 키가 나은가**」이지 인과적 주장이 아니다.
레짐 라벨은 같은 봉의 기하(ADX·기울기·BB)를 기술하는 것이라 수익 예측기만큼 치명적이진
않지만, 라이브보다 깨끗한 확률을 쓰고 있는 건 사실이다.

출력: features_with_regime_2022_2026_HMM.parquet (열 이름은 balnobb 판과 동일 -- 학습
코드가 그대로 돌아간다. **두 프레임을 섞지 말 것.**)
"""
from __future__ import annotations
import os, sys, numpy as np, pandas as pd, joblib
from pathlib import Path

ROOT = Path(os.environ.get("ZEUS_ROOT") or Path.home() / "crypto-scalping")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "trading_bot_modules"))
from omega4_6_2_source_parent_live import CURRENT_PREFIX  # noqa: E402
from odyssey_regime3_live import _class_proba  # noqa: E402

OUT = ROOT / "tmp/omega461_longwindow_20260917"
HMM = (ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
       / "regime3_current_sensitive_hmm_wide24_2024.joblib")


def main() -> int:
    art = joblib.load(HMM)
    FC, CLS = list(art["feature_cols"]), list(art["classes"])
    print(f"HMM {art['model_id']} · 라벨 {art['label_mode']} · 상태 {art['state_count']} "
          f"· sticky {art['sticky']} · 피쳐 {len(FC)}", flush=True)

    F = pd.read_parquet(OUT / "features_136_2022_2026_realfunding.parquet")
    F["timestamp"] = pd.to_datetime(F["timestamp"])
    miss = [c for c in FC if c not in F.columns]
    assert not miss, f"HMM 피쳐 결손 {len(miss)}개: {miss[:5]}"
    raw = F[FC].to_numpy(np.float64)
    assert np.isfinite(raw).all(), "HMM 입력에 비유한값"

    xz = art["scaler"].transform(raw)
    state = art["model"].filter_proba(xz)
    p = _class_proba(state, np.asarray(art["state_class_matrix"], dtype=np.float64))
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    assert np.allclose(p.sum(1), 1.0), "확률 정규화 실패"

    sp = np.sort(p, axis=1)
    for i, c in enumerate(CLS):
        F[f"{CURRENT_PREFIX}{c}_prob"] = p[:, i]
    F[f"{CURRENT_PREFIX}confidence"] = p.max(axis=1)
    F[f"{CURRENT_PREFIX}margin"] = sp[:, -1] - sp[:, -2]
    F[f"{CURRENT_PREFIX}entropy"] = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1) / np.log(3.0)

    arg = p.argmax(1)
    print(f"HMM 라우팅 비율 {np.bincount(arg, minlength=3) / len(F)} · "
          f"평균 confidence {p.max(1).mean():.4f}", flush=True)

    # balnobb 판과의 배정 일치율 -- 「키가 얼마나 다른가」를 숫자로 남긴다
    B = pd.read_parquet(OUT / "features_with_regime_2022_2026_realfunding.parquet",
                        columns=["timestamp"] + [f"{CURRENT_PREFIX}{c}_prob" for c in CLS])
    B["timestamp"] = pd.to_datetime(B["timestamp"])
    j = F[["timestamp"]].merge(B, on="timestamp", how="inner")
    barg = j[[f"{CURRENT_PREFIX}{c}_prob" for c in CLS]].to_numpy().argmax(1)
    harg = arg[F.timestamp.isin(j.timestamp).to_numpy()]
    agree = float((barg == harg) .mean())
    print(f"⭐balnobb vs HMM 전문가 배정 일치율 {agree:.4f} ({len(j):,}봉) · "
          f"balnobb 비율 {np.bincount(barg, minlength=3)/len(barg)}", flush=True)

    out = OUT / "features_with_regime_2022_2026_HMM.parquet"
    F.to_parquet(out, index=False)
    print(f"저장: {out}  ({len(F):,}행)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
