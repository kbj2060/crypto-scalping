"""«띠 안·이탈» 모델(joblib) → rtmodel JSON (2026-11-21). 서버 data/live/regime_band_h48_model.joblib · 개발기 사본 md5 f62774c6 (읽기만).
    python rtmodel/tools/export_regime.py <joblib> <출력 json>       # 서버 그림자: … data/live/regime_band_h48_model.joblib data/live/rt_regime_band/regime_band.json
"""
import json
import sys
from pathlib import Path

import joblib
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_hgb import SKLEARN_OK, model, sha  # noqa: E402


def export(src: Path, out: Path) -> None:
    if sklearn.__version__ != SKLEARN_OK:
        raise SystemExit(f"scikit-learn {sklearn.__version__} != {SKLEARN_OK}")
    b = joblib.load(src)
    assert list(b["model"].feature_names_in_) == list(b["features"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sklearn": sklearn.__version__, "source": src.name, "sha256": sha(src), "features": list(b["features"]), "model": model(b["model"])}))


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))
