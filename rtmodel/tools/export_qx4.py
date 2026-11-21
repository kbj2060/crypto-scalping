"""4h 변동성 QX4 아티팩트(joblib) → rtmodel JSON (2026-11-21). 개발기 사본 tmp/vol4h_qx4/qx4_<끝>.joblib(서버 data/live/vol4h_qx4.joblib 과 같은지 미확인 · 읽기만).
    python rtmodel/tools/export_qx4.py [joblib] [출력 json]
HGB 셋(시드 3) = export_hgb.model · MLP = numpy (W, b) float32 층 + 표준화 mu·sd(f64) + 출력 ym·ys. 실수는 비트(f64 · f32 는 f32 비트).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_hgb import b1, bits, model, sha  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def f32bits(a):
    a = np.ascontiguousarray(np.asarray(a, np.float32))
    return {"shape": list(a.shape), "b": a.view(np.uint32).ravel().tolist()}


def hgbs(m):
    models, cols = m
    return {"type": "hgb", "models": [model(h) for h in models], "cols": list(cols)}


def mlp(m):
    for L in m["layers"]:
        for W, b in L:
            assert W.dtype == np.float32 and b.dtype == np.float32
    return {"type": "mlp", "layers": [[{"W": f32bits(W), "b": f32bits(b)} for W, b in L] for L in m["layers"]], "mu": bits(m["mu"]), "sd": bits(m["sd"]),
            "ym": b1(m["ym"]), "ys": b1(m["ys"]), "cols": list(m["cols"])}


def main(src: Path, out: Path) -> None:
    a = joblib.load(src)
    d = {"source": src.name, "sha256": sha(src), "train_end": a.get("train_end"), "mkt": a["mkt"], "members": list(a["members"]),
         "coins": {c: {"M3": hgbs(v["M3"]), "M6": mlp(v["M6"]), "c": b1(v["c"]), "expand_logit": [b1(x) for x in v["expand_logit"]]} for c, v in a["coins"].items()},
         "PX3": hgbs(a["PX3"]), "PX6": mlp(a["PX6"])}
    out.write_text(json.dumps(d)); print(out.name, f"{out.stat().st_size:,} 바이트")


if __name__ == "__main__":
    s = sys.argv[1:]
    main(Path(s[0]) if s else Path("/home/kbj20/crypto-scalping/tmp/vol4h_qx4/qx4_2026-10-23.joblib"), Path(s[1]) if len(s) > 1 else ROOT / "rtmodel/tests/fixtures/qx4.json")
