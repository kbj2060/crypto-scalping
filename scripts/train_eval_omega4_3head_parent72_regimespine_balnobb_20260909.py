"""Phase 2 — 부모 TabM 재학습 래퍼: 레짐 척추를 `balnobb` 로 교체.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계약: docs/model_contracts/omega461_regimegbm_rebuild_contract.md
결정: 2026-09-09 사용자 "balnobb bb덮어쓰기 제거 k=0 버전으로 진행하자."

무엇을 바꾸나 — base_cols 102개 중 레짐 6개만
--------------------------------------------
`train_eval_omega4_3head_parent72_pinned102_20260727.py` 의 두 몽키패치를 그대로 승계하고
(7개 결측 컬럼 복구 + base_cols 고정), 거기에 **세 번째 치환**을 얹는다:

    regime3_current_sensitive_wide24_{bull_prob,bear_prob,chop_prob,confidence,entropy,margin}
      ↓ 이름까지 교체 (값만 덮어쓰지 않는다)
    regime3_balnobb_cut2509_{bull_prob,bear_prob,chop_prob,confidence,entropy,margin}

**왜 이름까지 바꾸는가 — 조용한 train/inference 불일치 방지.**
라이브 어댑터(`omega4_6_1_live.py::_Regime3CurrentLiveFeatures`)는 `wide24_*` 이름으로 값을
**자체 계산해서 넣는다**. 같은 이름에 새 값만 덮어써서 학습하면 학습은 balnobb 값으로,
라이브는 wide24 HMM 값으로 돌아가 조용히 어긋난다 — 이 저장소가 반복해서 당한 결함 유형이다.
새 접두사를 쓰면 라이브 어댑터가 그 컬럼을 만들지 못해 **fail-fast** 하므로, 어댑터를 함께
고치기 전까지 실수로 배포될 수 없다. 그게 의도된 안전장치다.

base_cols 개수(102)와 **순서**는 유지한다 — 이름만 위치 그대로 갈아끼운다.

프레임 병합
----------
train/eval 프레임 양쪽에 `data/ensemble/supervised/omega461_balnobb_cut2509_20260909/` 의
사이드카를 timestamp 로 조인한다. 결측이 하나라도 있으면 fail-fast(조용한 median 대체 금지).

debounce
--------
K=0 (사용자 결정). 사이드카 확률을 그대로 쓴다. 나중에 K 를 걸고 싶으면 사이드카 생성 단계나
예측 후처리에서 `_debounce` 를 적용하면 되고, 이 래퍼는 바꾸지 않아도 된다.

준수: 아키텍처/라벨/HP/에폭/split 은 20260620 원본 트레이너 그대로. 라이브 파일 미변경.
      Seed-Diversity Gate 를 위해 호출부에서 N>=5 진짜 무작위 시드를 돌린다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_eval_omega4_3head_parent72_pinned102_20260727 as pinned  # noqa: E402

omega = pinned.omega
parent_script = pinned.parent_script

OLD_PREFIX = "regime3_current_sensitive_wide24_"
NEW_PREFIX = "regime3_balnobb_cut2509_"
SUFFIXES = ("bull_prob", "bear_prob", "chop_prob", "confidence", "entropy", "margin")
SIDECAR_DIR = ROOT / "data/ensemble/supervised/omega461_balnobb_cut2509_20260909"
SIDECAR_TAGS = ("2024", "2025", "2026_rebuilt")

_orig_load_frames = omega._load_omega_frames


def _load_sidecar() -> pd.DataFrame:
    parts = []
    for tag in SIDECAR_TAGS:
        p = SIDECAR_DIR / f"training_features_{tag}_{NEW_PREFIX}sidecar.csv"
        if not p.exists():
            raise RuntimeError(f"regimespine: 사이드카 없음 {p}")
        parts.append(pd.read_csv(p, parse_dates=["timestamp"],
                                 usecols=["timestamp"] + [f"{NEW_PREFIX}{s}" for s in SUFFIXES]))
    sc = (pd.concat(parts, ignore_index=True).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    return sc


def _attach(frame: pd.DataFrame, sc: pd.DataFrame, what: str) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"])
    joined = sc.set_index("timestamp").reindex(ts)
    for s in SUFFIXES:
        col = f"{NEW_PREFIX}{s}"
        vals = pd.to_numeric(joined[col], errors="coerce").to_numpy()
        n_missing = int(pd.isna(vals).sum())
        if n_missing:
            raise RuntimeError(f"regimespine: {what} 프레임의 {col} 에 결측 {n_missing}개 "
                               f"-- 조용한 대체를 막기 위해 중단한다")
        out[col] = vals
    return out


def _patched_load_frames():
    train_all, eval_df, overlay_report = _orig_load_frames()
    train_all = pinned._repair_train_columns(train_all)
    sc = _load_sidecar()
    print(f"[regimespine] 사이드카 {len(sc):,}행 "
          f"({sc.timestamp.min()} ~ {sc.timestamp.max()})", flush=True)
    train_all = _attach(train_all, sc, "train")
    eval_df = _attach(eval_df, sc, "eval")
    print(f"[regimespine] 레짐 6컬럼 부착 완료 (접두사 {NEW_PREFIX})", flush=True)
    return train_all, eval_df, overlay_report


def _install(component: str) -> list[str]:
    import torch

    bundle = torch.load(pinned.LIVE_BUNDLES[component], map_location="cpu", weights_only=False)
    live_cols = list(bundle["base_cols"])
    swap = {f"{OLD_PREFIX}{s}": f"{NEW_PREFIX}{s}" for s in SUFFIXES}
    missing = [c for c in swap if c not in live_cols]
    if missing:
        raise RuntimeError(f"regimespine: 라이브 base_cols 에 wide24 컬럼 누락 {missing}")
    new_cols = [swap.get(c, c) for c in live_cols]
    assert len(new_cols) == len(live_cols) == 102, f"base_cols 개수 변화: {len(new_cols)}"
    print(f"[regimespine] base_cols {len(new_cols)}개 — 레짐 6개를 {OLD_PREFIX}* → {NEW_PREFIX}* 로 "
          f"이름까지 치환(순서 유지)", flush=True)

    def _patched_numeric_feature_cols(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
        mt = [c for c in new_cols if c not in train_df.columns]
        me = [c for c in new_cols if c not in eval_df.columns]
        if mt or me:
            raise RuntimeError(f"regimespine: base_cols 부재 (train {mt}, eval {me})")
        return list(new_cols)

    omega._load_omega_frames = _patched_load_frames
    omega._numeric_feature_cols = _patched_numeric_feature_cols
    return new_cols


def main() -> int:
    argv = list(sys.argv[1:])
    if "--pin-component" not in argv:
        raise SystemExit("--pin-component {h48qual,zig075} is required")
    i = argv.index("--pin-component")
    component = argv[i + 1]
    if component not in pinned.LIVE_BUNDLES:
        raise SystemExit(f"--pin-component must be one of {sorted(pinned.LIVE_BUNDLES)}")
    del argv[i: i + 2]
    sys.argv = [sys.argv[0], *argv]

    _install(component)
    return parent_script.main()


if __name__ == "__main__":
    raise SystemExit(main())
