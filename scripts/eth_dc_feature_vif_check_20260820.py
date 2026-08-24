#!/usr/bin/env python3
"""문헌조사 결과 반영 1/2: 정리(리던던시 감사) 방법의 구조적 사각지대(3개 이상 피쳐에 분산된
다중공선성 -- pairwise correlation은 못 봄) 점검. VIF(Variance Inflation Factor)는
correlation matrix 역행렬의 대각원소로 계산 가능(표준화 데이터 가정 -- statsmodels 없이도
정확히 동일한 값, 133회 개별 OLS보다 훨씬 빠름): VIF_i = (corr_matrix^-1)_ii.
VIF>=10(관용적 임계, O'Brien 2007이 지적하듯 유도값 아님 -- 참고용)인 피쳐를 보고한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")
PRUNED = json.loads((SCRATCH / "dc_pruned_features_20260820.json").read_text())

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega


def main() -> None:
    train, eval_df = omega._load_omega_frames()[:2]
    pooled = pd.concat(
        [train[PRUNED].apply(pd.to_numeric, errors="coerce"),
         eval_df[PRUNED].apply(pd.to_numeric, errors="coerce")],
        ignore_index=True,
    ).dropna()
    print(f"VIF 계산 대상: {len(pooled):,}행(완전케이스) x {len(PRUNED)}피쳐", flush=True)

    corr = pooled.corr(method="pearson").to_numpy()
    cond_number = float(np.linalg.cond(corr))
    print(f"상관행렬 조건수(condition number) = {cond_number:.1f} (클수록 다중공선성 심함)", flush=True)

    try:
        inv_corr = np.linalg.inv(corr)
        method = "inv"
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr)
        method = "pinv(특이행렬이라 의사역행렬로 대체)"
    vif = np.diag(inv_corr)

    vif_series = pd.Series(vif, index=PRUNED).sort_values(ascending=False)
    print(f"\n[{method}] VIF 상위 20개:", flush=True)
    for feat, v in vif_series.head(20).items():
        print(f"    {feat:45s} VIF={v:.2f}", flush=True)

    high_vif = vif_series[vif_series >= 10.0]
    print(f"\nVIF>=10인 피쳐: {len(high_vif)}개", flush=True)
    for feat, v in high_vif.items():
        print(f"    {feat:45s} VIF={v:.2f}", flush=True)

    out = {
        "n_rows": len(pooled), "n_features": len(PRUNED), "condition_number": cond_number,
        "vif_all": {k: float(v) for k, v in vif_series.items()},
        "high_vif_ge10": {k: float(v) for k, v in high_vif.items()},
    }
    out_path = ROOT / "tmp/eth_dc_feature_vif_check_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
