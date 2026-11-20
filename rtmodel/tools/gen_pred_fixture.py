"""rtmodel 일치 시험의 정답지: 고정 시드 입력 행 → scikit-learn predict_proba 와 감싼 식의 확률 (2026-11-20).

정답은 저장 모델(joblib)과 scikit-learn 으로만 만든다. dashboard/box_hold 는 import 해도 네트워크를 안 부른다(소스 확인:
joblib.load·json 읽기뿐) -- p_stay·QX4I_ISO·logit 을 거기서 그대로 가져온다. scripts/live_sr_lean_20261118 은 REST·웹소켓
모듈이라 import 하지 않고 그 식(np.mean([m.predict_proba(x)[:, 1] ...]) · thr)만 옮겨 적었다. card30 모델은 개발기에 없어
(tmp/card30_direction_20260926/ 서버 전용) 식만 대역 모델(박스 30m HGB 셋)로 시험한다.

    python rtmodel/tools/gen_pred_fixture.py [출력 디렉터리]   # 기본 rtmodel/tests/fixtures
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_hold as BH  # noqa: E402
from export_hgb import BOX, DASH, SR, bits  # noqa: E402

N_RAND, N_EDGE, N_SPECIAL = 500, 700, 300
SPECIAL = [np.nan, np.inf, -np.inf, 0.0, -0.0, 1e300, -1e300, 1e39, -1e39, 3.4028235e38, 1e-320, -5.0, 1e6]
F32MAX = float(np.finfo(np.float32).max)


def trees(m) -> list[tuple]:
    """모델 안 트리마다 (열, 임계값, 왼쪽, 오른쪽, 잎). 파이프라인이면 원 입력 공간에 있는 단계(HGB·ET -- 앞에 대치기뿐)만."""
    out = []
    for s in ([x for _, x in m.steps] if hasattr(m, "steps") else [m]):
        if type(s).__name__ == "HistGradientBoostingClassifier":
            out += [(n["feature_idx"], n["num_threshold"], n["left"], n["right"], n["is_leaf"].astype(bool)) for n in (it[0].nodes for it in s._predictors)]
        elif type(s).__name__ == "ExtraTreesClassifier":
            out += [(t.feature, t.threshold, t.children_left, t.children_right, t.children_left == -1) for t in (e.tree_ for e in s.estimators_)]
    return out


def rows_for(models, nf: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = [t for m in models for t in trees(m)]
    lo, hi = np.full(nf, -1.0), np.full(nf, 1.0)
    for f in range(nf):
        v = np.concatenate([thr[~leaf & (feat == f)] for feat, thr, _, _, leaf in T])
        if len(v):
            lo[f], hi[f] = v.min(), v.max()
    span = np.maximum(hi - lo, 1e-9)
    rand = lambda n: lo - 0.1 * span + rng.random((n, nf)) * 1.2 * span  # noqa: E731 -- 정상 범위(임계값 범위 ±10%)
    R = [rand(N_RAND)]
    # 임계값 바로 위·아래: 무작위 트리에서 이 행이 실제로 지나는 분기 하나를 골라 그 열을 그 자리 · 다음 f64 위/아래 ·
    #   f32 로 내리면 달라지는 이웃으로(아무 노드나 고르면 대부분 그 행이 안 지나가 시험이 못 잡는다 -- 양성 대조에서 확인)
    E = rand(N_EDGE)
    for i in range(N_EDGE):
        feat, thr, left, right, leaf = T[rng.integers(len(T))]
        j, path = 0, []
        while not leaf[j]:
            path.append(j)
            j = left[j] if E[i, feat[j]] <= thr[j] else right[j]
        j = path[rng.integers(len(path))]
        f, t = int(feat[j]), float(thr[j])
        t32 = np.float32(t)
        cand = [t, np.nextafter(t, np.inf), np.nextafter(t, -np.inf), float(t32),
                float(np.nextafter(t32, np.float32(np.inf))), float(np.nextafter(t32, np.float32(-np.inf)))]
        E[i, f] = cand[i % len(cand)]
    R.append(E)
    S = rand(N_SPECIAL)                               # NaN·±무한대·0·-0·아주 큼·f32 넘침·준정규·음수
    for i in range(N_SPECIAL):
        for f in rng.choice(nf, size=rng.integers(1, 6), replace=False):
            S[i, f] = SPECIAL[rng.integers(len(SPECIAL))]
    R.append(S)
    return np.vstack(R)


def p1(m, X: np.ndarray, cols) -> list:
    """행마다 양성 확률 또는 None(scikit-learn 이 예외를 내는 행). 위험한 행(무한대·f32 넘침)만 한 줄씩 다시 본다."""
    risky = ~np.isfinite(X).all(1) | (np.abs(np.nan_to_num(X)) > F32MAX).any(1)
    out: list = [None] * len(X)
    safe = np.flatnonzero(~risky)
    for i, v in zip(safe, m.predict_proba(pd.DataFrame(X[safe], columns=cols))[:, 1]):
        out[i] = float(v)
    for i in np.flatnonzero(risky):
        try:
            out[i] = float(m.predict_proba(pd.DataFrame(X[i:i + 1], columns=cols))[0, 1])
        except ValueError:
            pass
    return out


def nb(v) -> int | None:
    return None if v is None else bits([v])[0]


def main(out: Path) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # f32 넘침 캐스트 경고
    warnings.filterwarnings("ignore", message="X does not have valid feature names")   # card30 은 이름 없는 float32 배열을 넘긴다(서버 그대로)
    fx: dict = {"hgb_cases": []}
    seed = 20261120
    for asset, f in BOX.items():
        b = joblib.load(DASH / f"{f}.joblib")
        for name, cf in b["scales"].items():
            if "model" not in cf:
                continue
            seed += 1
            X = rows_for([cf["model"]], len(cf["cols"]), seed)
            # box_hold.payload 와 같은 꼴: 열 이름 있는 DataFrame 한 장 → predict_proba[:, 1]
            p = cf["model"].predict_proba(pd.DataFrame(X, columns=cf["cols"]))[:, 1]
            fx["hgb_cases"].append({"file": f"box_{asset}.json", "scale": name, "rows": [bits(r) for r in X], "p": bits(p)})
            if name == "1h":   # 1h Platt90 식(payload 그대로) -- 실제 보정 파일은 서버 전용이라 a·b 는 대역 값
                a_, b_ = 1.0873, -0.0412
                fx["hgb_cases"][-1]["platt"] = {"a": nb(a_), "b": nb(b_), "p": bits([float(1 / (1 + np.exp(-(a_ * BH.logit(q) + b_)))) for q in p])}
    # Platt90 자르기 경계(모델 출력은 1e-6 바깥으로 잘 안 간다)
    fx["platt_edge"] = [[nb(a_), nb(b_), nb(q), nb(float(1 / (1 + np.exp(-(a_ * BH.logit(q) + b_)))))]
                        for a_, b_ in ((1.0873, -0.0412), (0.6, 0.3)) for q in (0.0, 1e-12, 5e-7, 1e-6, 1.0000001e-6, 0.5, 1 - 1e-6, 1 - 1e-12, 1.0)]
    # sr_lean: live_sr_lean_20261118._score 의 식
    a = joblib.load(DASH / f"{SR}.joblib")
    for m in a["models"]:                              # live_sr_lean.load_art 그대로(n_jobs=1). 8 이면 트리 합 순서가 실행마다 달라 파이썬끼리도 끝자리가 다르다
        for _, st in getattr(m, "steps", [(None, m)]):
            if hasattr(st, "n_jobs"):
                st.n_jobs = 1
    X = rows_for(a["models"], len(a["cols"]), 7)
    each = [p1(m, X, a["cols"]) for m in a["models"]]
    ens = [None if any(e[i] is None for e in each) else float(np.mean([e[i] for e in each])) for i in range(len(X))]
    lo, hi = a["thr"]
    fx["sr_lean"] = {"rows": [bits(r) for r in X], "p_each": [[nb(v) for v in e] for e in each], "p": [nb(v) for v in ens],
                     "grp": [None if v is None else ("top" if v >= hi else "bot" if v <= lo else "mid") for v in ens]}
    # card30 식(dashboard/server.py _card30_pred): float32 행 · predict_proba 평균 · k·logit(raw) + (1-k)·logit(base) -- 대역 모델
    stand = [("box_eth.json", "box_hold_hgb_20261016"), ("box_btc.json", "box_hold_hgb_btc_20261031"), ("box_sol.json", "box_hold_hgb_sol_20261031")]
    ms = [joblib.load(DASH / f"{f}.joblib")["scales"]["30m"]["model"] for _, f in stand]
    X = rows_for(ms, 27, 30)
    X = X[np.isfinite(X).all(1)]                       # card30 피쳐는 유한값만(f 가 None 이면 안 부른다) -- NaN 행은 HGB 케이스가 본다
    k, base = 0.731, 0.2605
    lg = lambda q: np.log(q / (1 - q))  # noqa: E731
    pc = []
    for r in X:
        row = np.array([r], dtype=np.float32)
        raw = float(np.mean([m.predict_proba(row)[0, 1] for m in ms]))
        pc.append(float(1 / (1 + np.exp(-(k * lg(raw) + (1 - k) * lg(base))))))
    fx["card30"] = {"files": [s for s, _ in stand], "scale": "30m", "k": nb(k), "base": nb(base), "rows": [bits(r) for r in X], "p": bits(pc)}
    # QX4I 4h 등조(box_hold.with_qx4i 의 식 -- 반올림 전 값)
    rng = np.random.default_rng(11)
    q = []
    for i in range(3000):
        lrv = float(np.log(rng.uniform(0.0005, 0.05) ** 2)); liv = float(np.log(rng.uniform(0.0005, 0.05) ** 2)) if i % 3 else None
        hf = float(rng.uniform(0.0002, 0.03)) if i % 50 else float(rng.uniform(1.0, 5.0))   # 가끔 s < 0.02 쪽
        lq = (liv + BH.QX4I_ISO["c_qx4i"]) if liv is not None else (lrv + BH.QX4I_ISO["c_qx4"])
        p = float(np.interp(BH.p_stay(np.sqrt(np.exp(lq)) / hf), BH.QX4I_ISO["x"], BH.QX4I_ISO["y"]))
        q.append([nb(lrv), nb(liv), nb(hf), nb(p)])
    fx["qx4i"] = q
    out.mkdir(parents=True, exist_ok=True)
    (out / "pred_fixture.json").write_text(json.dumps(fx))
    print(f"pred_fixture.json {(out / 'pred_fixture.json').stat().st_size:,} 바이트 · HGB 케이스 {len(fx['hgb_cases'])} · sr_lean {len(fx['sr_lean']['rows'])}행"
          f"(오류 행 {sum(v is None for v in fx['sr_lean']['p'])}) · card30 {len(pc)}행 · qx4i {len(q)}")


if __name__ == "__main__":
    with threadpool_limits(2):
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "rtmodel" / "tests" / "fixtures")
