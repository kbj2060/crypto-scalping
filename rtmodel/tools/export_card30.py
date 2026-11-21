"""30분 카드 방향·닿음 모델(서버 tmp/card30_direction_20260926/model.joblib · model_reach.joblib) → rtmodel JSON (2026-11-21 · 읽기만).
    python rtmodel/tools/export_card30.py <모델 디렉터리> <출력 json>
"""
import json
import sys
from pathlib import Path

import joblib
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_hgb import SKLEARN_OK, b1, model, sha  # noqa: E402


def export(d: Path, out: Path) -> None:
    if sklearn.__version__ != SKLEARN_OK:
        raise SystemExit(f"scikit-learn {sklearn.__version__} != {SKLEARN_OK}")
    o = {"sklearn": sklearn.__version__}
    for key, f in (("dir", "model.joblib"), ("reach", "model_reach.joblib")):
        v = joblib.load(d / f)
        o[key] = {"source": f, "sha256": sha(d / f), "cols": v["cols"], "k": b1(v["k"]), "base": b1(v["base"]), "models": [model(x) for x in v["models"]]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(o))


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))
