"""4h 변동성 HGB(배포 22열 · 8시드 · scripts/live_eth_sizing_vol_model_20260912) joblib → rtmodel JSON (2026-11-21 · 읽기만).
화면 «4h 변동성»은 QX4 가 먼저고 이것은 QX4 가 실패할 때의 대체 길이다(server.py api_vol_forecast).
    python rtmodel/tools/export_svm.py <joblib> <출력 json>     # 서버: data/live/eth_sizing_vol_model.joblib (매달 재학습 cron 뒤 다시) · xrp_sizing_vol_model.joblib
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
    a = joblib.load(src)
    assert a["ref_pred"] > 0 and len(a["features"]) == 22, "load_model 과 같은 검사"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sklearn": sklearn.__version__, "source": src.name, "sha256": sha(src), "features": a["features"],
                               "train_end": a.get("train_end"), "expand_logit": a.get("expand_logit"), "models": [model(m) for m in a["models"]]}))


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))
