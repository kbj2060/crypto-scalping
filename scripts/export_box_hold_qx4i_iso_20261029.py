#!/usr/bin/env python3
"""4h 박스 유지 QX4I 등조 지도 내보내기 -- 연구 research_box_hold_vs_qx4i_20261029.fit_iso() 그대로(같은 학습 행·같은 등조) → 서빙 JSON.

    BOXVOL_IN=<연구 입력 폴더> python scripts/export_box_hold_qx4i_iso_20261029.py dashboard/box_hold_qx4i_iso_20261029.json
서빙(dashboard/box_hold.py qx4i_hold_p)은 np.interp(p_raw, x, y) -- 등조 예측(선형 보간 · 양 끝 고정)과 같다는 걸 여기서 학습 행 전부로 확인한다.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_box_hold_vs_qx4i_20261029 as R   # noqa: E402

_, T, iso = R.fit_iso()
x, y = iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()
assert np.allclose(np.interp(T.p_raw, x, y), iso.predict(T.p_raw), atol=1e-12, rtol=0), "등조 예측 ≠ 선형 보간"
grid = np.linspace(0, 1, 21)
out = {"x": x, "y": y,
       "c_qx4i": R.C_QX4I,                                                          # 12라운드 QX4I 평균 RV 눈금(동결)
       "c_qx4": json.loads((R.IN / "report_r12.json").read_text())["models"]["QX4"]["c"],   # IV 없을 때 QX4 눈금
       "fit": {"rows": len(T), "days": int(T.index.normalize().nunique()), "range": [str(T.index.min()), str(T.index.max())]},
       "parity": [[float(g), float(iso.predict([g])[0])] for g in grid],
       "source": "scripts/research_box_hold_vs_qx4i_20261029.py fit_iso (사전등록 bd9f8325 · 결과 e5a45d3c 시험 ΔBrier −.00732 단측90 통과)"}
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("ok", len(x), "점 · 학습", out["fit"])
