"""저장된 scikit-learn 모델(joblib) → rtmodel 이 읽는 JSON (2026-11-20 · R5 첫 조각).

실수는 전부 float64 비트(u64 정수)로 쓴다 -- 십진 왕복·파서 반올림 걱정이 없다. NaN·무한대도 그대로 간다.
추론에 필요한 것만 쓴다(학습 통계·gain·count 는 뺀다). 아는 구조가 아니면 멈춘다(조용히 다른 값을 내느니 안 낸다):
  scikit-learn 버전 · HGB 노드 dtype · 범주형 · 다중 클래스 · 손실 · 전처리기 · 파이프라인 단계 종류와 설정.

    python rtmodel/tools/export_hgb.py [출력 디렉터리]     # 기본 rtmodel/tests/fixtures
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"
BOX = {"eth": "box_hold_hgb_20261016", "btc": "box_hold_hgb_btc_20261031", "sol": "box_hold_hgb_sol_20261031",
       "soxl": "box_hold_hgb_soxl_20261117", "xrp": "box_hold_hgb_xrp_20261020"}
SR = "sr_lean_model_20261118"
ISO = "box_hold_qx4i_iso_20261029.json"
# 내부 구조를 소스에서 확인한 버전(_predictor.pyx · _tree.pyx · _base.py · _data.py · _logistic.py). 다른 버전이면 소스를 다시 읽고 여기를 고친다.
SKLEARN_OK = "1.8.0"
HGB_NODE = ('value', 'count', 'feature_idx', 'num_threshold', 'missing_go_to_left', 'left', 'right', 'gain', 'depth',
            'is_leaf', 'bin_threshold', 'is_categorical', 'bitset_idx')


def bits(a) -> list[int]:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64).tolist()


def b1(x) -> int:
    return bits([x])[0]


def hgb(m) -> dict:
    assert type(m).__name__ == "HistGradientBoostingClassifier", type(m)
    assert m.is_categorical_ is None and m._preprocessor is None, "범주형·전처리기 있는 HGB 는 아직 안 옮김"
    assert type(m._loss).__name__ == "HalfBinomialLoss" and type(m._loss.link).__name__ == "LogitLink", m._loss
    assert m.n_trees_per_iteration_ == 1 and list(m.classes_) == [0, 1], m.classes_
    assert m._baseline_prediction.shape == (1, 1)
    trees = []
    for it in m._predictors:
        assert len(it) == 1
        n = it[0].nodes
        assert n.dtype.names == HGB_NODE, n.dtype.names
        assert not n["is_categorical"].any()
        assert n["value"].dtype == np.float64 and n["num_threshold"].dtype == np.float64
        trees.append({"feat": n["feature_idx"].tolist(), "thr": bits(n["num_threshold"]), "mgl": n["missing_go_to_left"].astype(bool).tolist(),
                      "left": n["left"].tolist(), "right": n["right"].tolist(), "leaf": n["is_leaf"].astype(bool).tolist(), "value": bits(n["value"])})
    return {"type": "hgb", "n_features": int(m.n_features_in_), "baseline": b1(m._baseline_prediction[0, 0]), "trees": trees}


def step(name: str, s) -> dict:
    t = type(s).__name__
    if t == "SimpleImputer":
        assert s.strategy == "median" and np.isnan(s.missing_values) and not s.add_indicator, s
        assert not np.isnan(s.statistics_).any(), "통계가 NaN 인 열은 sklearn 이 열을 버린다 -- 아직 안 옮김"
        return {"type": "imputer", "fill": bits(s.statistics_)}
    if t == "StandardScaler":
        assert s.with_mean and s.with_std
        return {"type": "scaler", "mean": bits(s.mean_), "scale": bits(s.scale_)}
    if t == "ExtraTreesClassifier":
        assert s.n_outputs_ == 1 and list(s.classes_) == [0, 1], s.classes_
        trees = []
        for e in s.estimators_:
            tr = e.tree_
            assert tr.n_outputs == 1 and tr.value.shape[1:] == (1, 2)
            trees.append({"feat": tr.feature.tolist(), "thr": bits(tr.threshold), "left": tr.children_left.tolist(), "right": tr.children_right.tolist(),
                          "mgl": tr.missing_go_to_left.astype(bool).tolist(), "p1": bits(tr.value[:, 0, 1])})
        return {"type": "et", "trees": trees}
    if t == "LogisticRegression":
        assert s.coef_.shape[0] == 1 and list(s.classes_) == [0, 1], s.coef_.shape
        return {"type": "lr", "coef": bits(s.coef_[0]), "intercept": b1(s.intercept_[0])}
    if t == "HistGradientBoostingClassifier":
        return hgb(s)
    raise SystemExit(f"모르는 단계 {name}:{t}")


def model(m) -> list[dict]:
    return [step(n, s) for n, s in m.steps] if hasattr(m, "steps") else [step("", m)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(out: Path) -> None:
    if sklearn.__version__ != SKLEARN_OK:
        raise SystemExit(f"scikit-learn {sklearn.__version__} != {SKLEARN_OK} -- 내부 구조가 바뀌었을 수 있다. 소스를 다시 읽고 SKLEARN_OK 를 고쳐라")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"sklearn": sklearn.__version__}
    for asset, f in BOX.items():
        p = DASH / f"{f}.joblib"
        b = joblib.load(p)
        scales = {}
        for name, cf in b["scales"].items():
            if "model" not in cf:
                continue                                  # SOXL 30m·4h = 모델 없음(deployable False)
            scales[name] = {"deployable": bool(cf.get("deployable", True)), "cols": list(cf["cols"]), "train_base": b1(cf["train_base"]),
                            "q20": b1(cf["q20"]), "q80": b1(cf["q80"]), "model": model(cf["model"])}   # q20·q80 = 화면 «깨질 쪽·머물 쪽» 문턱
            assert list(cf["model"].feature_names_in_) == list(cf["cols"])
        prof = np.asarray(b["prof"], float)   # 요일×시각 168칸(box_hold.payload 가 np.asarray(bb["prof"], float) 로 쓴다)
        assert prof.shape == (168,), prof.shape
        (out / f"box_{asset}.json").write_text(json.dumps({**meta, "source": p.name, "sha256": sha(p), "prof": bits(prof), "scales": scales}))
    p = DASH / f"{SR}.joblib"
    a = joblib.load(p)
    for m in a["models"]:
        assert list(m.feature_names_in_) == list(a["cols"])
    (out / "sr_lean.json").write_text(json.dumps({**meta, "source": p.name, "sha256": sha(p), "cols": list(a["cols"]), "thr": bits(a["thr"]),
                                                  "rate": a["rate"], "models": [model(m) for m in a["models"]]}))
    p = DASH / ISO
    d = json.loads(p.read_text())
    (out / "qx4i_iso.json").write_text(json.dumps({"source": p.name, "sha256": sha(p), "x": bits(d["x"]), "y": bits(d["y"]),
                                                   "c_qx4i": b1(d["c_qx4i"]), "c_qx4": b1(d["c_qx4"])}))
    for f in sorted(out.glob("*.json")):
        print(f"{f.name} {f.stat().st_size:,} 바이트")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "rtmodel" / "tests" / "fixtures")
