"""4시간 방향 모델(joblib) → rtmodel JSON (2026-11-21). 개발기에 있는 SOL·XRP 판만(메인 체크아웃 data/live/ -- 읽기만). ETH 판은 서버에만 있다.
    python rtmodel/tools/export_dir4h.py [joblib 디렉터리] [출력 디렉터리]
"""
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_hgb import b1, model, sha  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*_dir4h_model.joblib")):
        a = joblib.load(p)
        # payload_from_features 가 읽는 것만: 피쳐 순서·문턱·bp 환산·학습 달·학습 창(표본 밖 기록 oos 는 문턱을 만들 때만 -- 문턱은 lo·hi 로 이미 정해져 있다)
        d = {"source": p.name, "sha256": sha(p), "coin": a.get("coin", "ETH"), "arm": a.get("arm", "S"), "features": list(a["features"]), "lo": b1(a["lo"]), "hi": b1(a["hi"]),
             "bp_scale": b1(a.get("bp_scale", 1.0)), "month": a["month"], "train_window": list(a["train_window"]), "models": [model(m) for m in a["models"]]}
        for m in a["models"]:
            assert list(m.feature_names_in_) if hasattr(m, "feature_names_in_") else True
        f = out / f"dir4h_{d['coin'].lower()}.json"; f.write_text(json.dumps(d)); print(f.name, f"{f.stat().st_size:,} 바이트")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]) if a else Path("/home/kbj20/crypto-scalping/data/live"), Path(a[1]) if len(a) > 1 else ROOT / "rtmodel" / "tests" / "fixtures")
