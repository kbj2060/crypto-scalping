#!/usr/bin/env python3
"""사용자 지시: 문헌조사(VIF)에서 발견한 사각지대를 "1개 정확 케이스만" 고치는 데 그치지 않고
제대로(iterative) 적용 -- 아직 신호(AUC/permutation-null 등)는 계산하지 않는다, 순수 피쳐셋
구축 단계.

133개(리던던시감사 결과) -> chop_prob 제거(133개 확률단체 완전선형종속 확인됨, 2026-08-20
VIF점검) -> 132개에서 시작. VIF는 매 반복마다 "표준화 상관행렬의 역행렬 대각원소"로 재계산
(133회 개별 OLS보다 훨씬 빠름, statsmodels 불필요, 수학적으로 동일). 매 스텝 최고VIF 피쳐 1개만
제거 후 재계산 -- 한 피쳐를 지우면 나머지 VIF가 다같이 바뀌므로(다중공선성은 상대적) 반드시
반복적으로 해야 하고, 상위N개를 한번에 지우면 안 됨(O'Brien 2007 등 VIF문헌의 표준 권고)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")
VIF_THRESHOLD = 10.0  # 관용 임계값(O'Brien 2007: 유도값 아님, convention) -- 문헌조사 결과 그대로 사용

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega


def _vif_all(corr_df: pd.DataFrame) -> pd.Series:
    corr = corr_df.to_numpy()
    try:
        inv_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr)
    return pd.Series(np.diag(inv_corr), index=corr_df.columns)


def main() -> None:
    pruned_133 = json.loads((SCRATCH / "dc_pruned_features_20260820.json").read_text())
    features = [c for c in pruned_133 if c != "regime3_current_sensitive_wide24_chop_prob"]
    assert len(features) == 132
    print(f"시작: {len(features)}개(133개 리던던시감사 결과 - 확률단체 1개 제거)", flush=True)

    train, eval_df = omega._load_omega_frames()[:2]
    pooled = pd.concat(
        [train[features].apply(pd.to_numeric, errors="coerce"),
         eval_df[features].apply(pd.to_numeric, errors="coerce")],
        ignore_index=True,
    ).dropna()
    print(f"VIF 계산 대상: {len(pooled):,}행(완전케이스)", flush=True)

    current = list(features)
    trace = []
    while len(current) > 2:
        vif = _vif_all(pooled[current].corr(method="pearson"))
        worst_feat = vif.idxmax()
        worst_vif = float(vif[worst_feat])
        if worst_vif < VIF_THRESHOLD:
            break
        trace.append({"step": len(trace) + 1, "removed": worst_feat, "vif_at_removal": worst_vif, "n_remaining_after": len(current) - 1})
        print(f"  step{len(trace):3d}: 제거={worst_feat:45s} VIF={worst_vif:12.2f} -> 잔여 {len(current) - 1}개", flush=True)
        current.remove(worst_feat)

    final_vif = _vif_all(pooled[current].corr(method="pearson"))
    print(f"\n수렴: {len(current)}개 (VIF<{VIF_THRESHOLD} 전부 만족), 총 {len(trace)}개 제거", flush=True)
    print(f"최종 최고 VIF = {final_vif.max():.2f} ({final_vif.idxmax()})", flush=True)

    out = {
        "start_count": len(features), "vif_threshold": VIF_THRESHOLD,
        "n_removed": len(trace), "elimination_trace": trace,
        "final_feature_count": len(current), "final_features": sorted(current),
        "final_max_vif": float(final_vif.max()), "final_vif_all": {k: float(v) for k, v in final_vif.sort_values(ascending=False).items()},
    }
    out_path = ROOT / "tmp/eth_dc_feature_vif_iterative_elimination_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (SCRATCH / "dc_vif_clean_features_20260820.json").write_text(json.dumps(sorted(current), indent=2), encoding="utf-8")
    print(f"\n[report] {out_path}")
    print(f"[최종 피쳐리스트] {SCRATCH / 'dc_vif_clean_features_20260820.json'}")


if __name__ == "__main__":
    main()
