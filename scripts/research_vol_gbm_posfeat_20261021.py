"""4h 변동성 GBM 에 1일 포지션 고통 피쳐(M1 롱·숏 분리 · Msym1 대칭) 2열 추가 — 배포판과 표본외·전진 오차 비교 (2026-10-21,
사용자 «M1·Msym1 을 4h 변동성 GBM 피쳐로 넣어 재학습하고, 배포판과 표본외·전진 오차를 비교»). 앞 단계: research_position_side_vol_forward_20261021.py.

🔒사전등록(결과 전 커밋):
- 모델 셋, 전부 배포 방식(live_eth_sizing_vol_model_20260912: HGB max_iter300 lr.06 depth6 · 시드 8 · 로그 rv 4h 타깃 · 로그 공간 평균):
  A 배포판 = 22열 · 자료 전부(2021-12~) · B 같은 행 = 22열 · C 와 같은 행(포지션 피쳐가 있는 2022-02~) · C = 22열 + M1 + Msym1.
  XRP 는 배포 그대로 ETH+XRP 합쳐 학습(C 의 ETH 행도 ETH 장부 피쳐).
- M1 = 1일 창(288봉) 물린 롱 고통 + 물린 숏 고통 (롱·숏 분리 장부, s 식 10-10) / OI · Msym1 = 같은 것을 대칭 장부(롱=숏)로. 매 5분봉 계산, OI 1봉 지연.
- 분할: 표본외 = ≤2024-12-31 19:55 학습 → 2025-01-01~2026-09-14 평가 · 전진 = ≤2026-09-14 19:55 학습 → 2026-09-15~2026-10-20 평가.
- 지표: 로그 오차제곱 평균(MSLE, 배포 품질 문과 같은 축) · 스피어만. 차 = C − A, C − B, 일 블록 부트스트랩 2000.
- 통과(표본외, 주 판정): 두 코인 모두 C−A 와 C−B MSLE 차 95% CI 상한 < 0. 전진은 부호만 보조(🔴이 구간은 앞 단계에서 이미 본 자료 — 완전한 새 자료 아님, 36일).
- 실용 기준: 통과해도 MSLE 개선이 1% 미만이면 «배포 가치 낮음»으로 적는다(라이브에 롱·숏 분리 장부를 새로 들여야 하므로).
사용: --feats → --fit [--set 7d] → --test [--set 7d]

🔒2차 사전등록(2026-10-21, 1차 불통과 뒤 사용자 «7일 창 M7·Msym7(+|VWAP 7일 거리|)도 GBM 위에서» — 결과 전 커밋):
- 근거: 앞 단계 H3a 7일 총 고통 → |4h| 가 BTC·XRP 사전등록 통과 · ETH Holm .12. GBM 가격 위치 피쳐는 최대 144봉(12h)이라 7일 창은 못 본다.
- C7 = 22열 + M7 + Msym7 + |vw7| · D7 = 22열 + |vw7| (VWAP 거리만 — 포지션 몫 분리용) · A·B 는 1차와 같다. 출력 tmp/vol_gbm_posfeat_7d_20261021.
- 통과(주): 표본외 두 코인 모두 C7−A·C7−B MSLE CI 상한 < 0. 보조: C7−D7(포지션 몫) · 전진 부호.
- 🔴표본외 2025~ 는 H3a 를 찾은 TEST 구간이라 완전한 새 자료가 아니다 · 전진 36일도 1차에서 본 구간 → 통과해도 «후보», 배포 전 새 전진 기간 필요.
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import research_position_side_price_impact_20261021 as R   # noqa: E402
import live_eth_sizing_vol_model_20260912 as V               # noqa: E402

SRC = "tmp/position_side_vol_forward_20261021"               # 앞 단계가 받은 자료(vol 패널 · 09-15~ 이어 붙인 프로파일 패널)
OUT = "tmp/vol_gbm_posfeat_20261021"
SYMS = ("ETHUSDT", "XRPUSDT")
SPLITS = {"oos": (pd.Timestamp("2024-12-31 19:55"), "2025-01-01", "2026-09-15"),
          "fwd": (pd.Timestamp("2026-09-14 19:55"), "2026-09-15", "2026-10-21")}
POS = ["M1", "Msym1", "M7", "Msym7", "avw7"]                  # 저장 열(1차는 앞 둘만 씀)
SETS = {"1d": {"C": ["M1", "Msym1"]}, "7d": {"C": ["M7", "Msym7", "avw7"], "D": ["avw7"]}}
SET = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--set" else "1d"
if SET == "7d":
    OUT_RES = "tmp/vol_gbm_posfeat_7d_20261021"
else:
    OUT_RES = OUT


def feats():
    """매 5분봉 M1·Msym1 → {SYM}_pos.parquet (timestamp = 봉 시작, 값은 봉 마감 시점)."""
    os.makedirs(OUT, exist_ok=True)
    R.BV = f"{os.getcwd()}/{SRC}/panel"
    for sym in SYMS:
        p = R.load(sym); c = p.close.to_numpy(float)
        snap = np.arange(R.M + 1, len(p))
        F = R.build_books(p.oi.to_numpy(float), p.volume.to_numpy(float), p.high.to_numpy(float), p.low.to_numpy(float), c,
                          R.side_share(c), snap, R.M)
        tp = (p.high + p.low + p.close) / 3
        avw7 = (p.close / ((tp * p.volume).rolling(2016).sum() / p.volume.rolling(2016).sum()) - 1).abs().to_numpy()
        out = pd.DataFrame({"timestamp": p.timestamp.to_numpy()[snap],
                            "M1": F[:, 0, 0, 3] + F[:, 1, 0, 4], "Msym1": F[:, 2, 0, 3] + F[:, 2, 0, 4],
                            "M7": F[:, 0, 1, 3] + F[:, 1, 1, 4], "Msym7": F[:, 2, 1, 3] + F[:, 2, 1, 4], "avw7": avw7[snap]})
        out.to_parquet(f"{OUT}/{sym}_pos.parquet")
        print(sym, len(out), out.timestamp.min(), out.timestamp.max(), out[POS].describe().loc[["50%", "max"]].round(5).to_dict(), flush=True)


def frame(sym):
    d = pd.read_parquet(f"{SRC}/{sym}_vol.parquet")
    X, y, ok, ts = V._frame(d)
    pos = pd.read_parquet(f"{OUT}/{sym}_pos.parquet").set_index("timestamp").reindex(pd.DatetimeIndex(ts))
    X = X.copy(); X[POS] = pos[POS].to_numpy()
    return X, y, ok, pd.DatetimeIndex(ts)


def _fit(A, yt):
    from sklearn.ensemble import HistGradientBoostingRegressor
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=int(s)).fit(A, np.log(yt)) for s in V.SEEDS]


def _pred(models, A):
    return np.exp(np.mean([m.predict(np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)) for m in models], axis=0))


def fit():
    fr = {s: frame(s) for s in SYMS}
    os.makedirs(OUT_RES, exist_ok=True)
    allc = sorted({c for v in SETS[SET].values() for c in v})
    cols = {"A": V.FEATURES, "B": V.FEATURES, **{k: V.FEATURES + v for k, v in SETS[SET].items()}}
    for split, (cut, e0, e1) in SPLITS.items():
        for sym in SYMS:
            parts = [sym] if sym == "ETHUSDT" else ["XRPUSDT", "ETHUSDT"]
            X, y, ok, ts = fr[sym]; ev = ok & (ts >= e0) & (ts < e1) & X[allc].notna().all(1).to_numpy()
            res = pd.DataFrame({"ts": ts[ev], "y": y[ev]})
            for name, cl in cols.items():
                rows = {s: fr[s][2] & (fr[s][3] <= cut) & (fr[s][0][allc].notna().all(1).to_numpy() if name != "A" else True) for s in parts}
                A = np.vstack([fr[s][0][cl].to_numpy(float)[rows[s]] for s in parts]); yt = np.concatenate([fr[s][1][rows[s]] for s in parts])
                res[name] = _pred(_fit(A, yt), X[cl].to_numpy(float)[ev])
                print(split, sym, name, "학습", len(yt), "행", flush=True)
            res.to_parquet(f"{OUT_RES}/{sym}_{split}.parquet")


def boot_diff(e1, e0, day, nb=2000, seed=20261021):
    ud = np.unique(day); g = {u: np.flatnonzero(day == u) for u in ud}; rng = np.random.default_rng(seed)
    d = e1 - e0
    bs = [d[np.concatenate([g[u] for u in rng.choice(ud, len(ud))])].mean() for _ in range(nb)]
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def test():
    rep = {}
    for split in SPLITS:
        for sym in SYMS:
            r = pd.read_parquet(f"{OUT_RES}/{sym}_{split}.parquet"); day = r.ts.dt.floor("D").to_numpy(); ly = np.log(r.y.to_numpy())
            ms = [k for k in "ABCD" if k in r]
            se = {k: (ly - np.log(r[k].to_numpy())) ** 2 for k in ms}
            row = {k: {"MSLE": round(float(se[k].mean()), 5), "spearman": round(float(r[k].corr(r.y, method="spearman")), 4)} for k in ms}
            for a, b in (("C", "A"), ("C", "B"), ("B", "A")) + ((("D", "A"), ("C", "D")) if "D" in ms else ()):
                ci = boot_diff(se[a], se[b], day)
                row[f"{a}−{b}"] = {"차": round(float((se[a] - se[b]).mean()), 6), "%": round(100 * float(se[a].mean() / se[b].mean() - 1), 3),
                                   "CI": [round(x, 6) for x in ci]}
            row["n"] = len(r); row["일수"] = int(len(np.unique(day)))
            rep[f"{split} {sym}"] = row
    rep["통과(표본외 두 코인 C−A·C−B CI<0)"] = all(rep[f"oos {s}"][k]["CI"][1] < 0 for s in SYMS for k in ("C−A", "C−B"))
    rep["전진 부호(C<A 두 코인)"] = all(rep[f"fwd {s}"]["C−A"]["차"] < 0 for s in SYMS)
    rep["실용(표본외 C−A 개선 ≥1%)"] = all(rep[f"oos {s}"]["C−A"]["%"] <= -1 for s in SYMS)
    json.dump(rep, open(f"{OUT_RES}/report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"--feats": feats, "--fit": fit, "--test": test}[sys.argv[1]]()
