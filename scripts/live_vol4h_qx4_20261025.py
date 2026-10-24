#!/usr/bin/env python3
"""4h 변동성 QX4 서빙판 — 굴림 원점 3창 통과 레시피(docs/prereg/vol4h_forecast_horserace_20261025.md 5·6라운드)를 아티팩트로.

QX4 = 평균(M3 코인별 HGB · M6 코인별 MLP · PX3 4코인 합쳐 HGB(+MKT) · PX6 4코인 합쳐 MLP(+MKT)) 로그 RV + 코인별 눈금 상수 c.
피쳐는 연구 스크립트 features()/add_mkt 그대로 import(학습 = 라이브 같은 함수). 🔴qv 는 «거래량 × 종가»(학습 패널에 견적 거래량이 없어 그렇게 학습함).

    python3 scripts/live_vol4h_qx4_20261025.py --parity     # 4라운드 경계로 다시 학습 → 저장된 연구 t_QX4·c 와 일치 + klines 경로 = 연구 F + 인과성
    python3 scripts/live_vol4h_qx4_20261025.py --train      # 최근: 학습 2년 + 눈금 5개월(끝 = SYS.END) → tmp/vol4h_qx4/qx4_<끝>.joblib

배포 안 함(대시보드·서버 연결은 사용자 승인 후). 실주문 없음.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_vol4h_forecast_horserace_20261025 as R   # noqa: E402

OUT = ROOT / "tmp" / "vol4h_qx4"
COINS, MEMBERS = R.COINS, ("M3", "M6", "PX3", "PX6")


# ---------- 학습 (연구 fit_predict 와 같은 초모수·시드, 모델 객체를 남긴다) ----------
def hgb_fit(F, y, cols):
    from sklearn.ensemble import HistGradientBoostingRegressor
    A = np.nan_to_num(F[cols].to_numpy(float))
    return [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=int(s)).fit(A, y) for s in R.SEEDS], cols


def mlp_fit(F, y, cols, seeds=R.SEEDS[:2]):
    """R.mlp 와 같은 절차(15분 간격 표본 · 표준화 · 64-64 · Adam 15 에폭) — 예측 대신 망과 정규화 상수를 돌려준다."""
    import torch
    torch.set_num_threads(8)
    keep = np.arange(0, len(F), 3); A = np.nan_to_num(F[cols].to_numpy(float)[keep])
    mu, sd = A.mean(0), A.std(0) + 1e-9; A, yy = (A - mu) / sd, y[keep]; ym, ys = yy.mean(), yy.std(); nets = []
    for s in seeds:
        torch.manual_seed(int(s)); rng = np.random.default_rng(int(s))
        net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        At, yt = torch.tensor(A, dtype=torch.float32), torch.tensor((yy - ym) / ys, dtype=torch.float32)
        for _ in range(15):
            for bi in np.array_split(rng.permutation(len(A)), max(1, len(A) // 1024)):
                opt.zero_grad(); loss = ((net(At[bi]).squeeze(1) - yt[bi]) ** 2).mean(); loss.backward(); opt.step()
        nets.append(net)
    return {"nets": nets, "mu": mu, "sd": sd, "ym": float(ym), "ys": float(ys), "cols": cols}


def _pred(m, F):
    if isinstance(m, dict):
        import torch
        B = (np.nan_to_num(F[m["cols"]].to_numpy(float)) - m["mu"]) / m["sd"]
        with torch.no_grad():
            return np.mean([n(torch.tensor(B, dtype=torch.float32)).squeeze(1).numpy() * m["ys"] + m["ym"] for n in m["nets"]], 0)
    models, cols = m
    return np.mean([h.predict(np.nan_to_num(F[cols].to_numpy(float))) for h in models], 0)


def predict_log(art, coin, F):
    """QX4 로그 RV(눈금 c 전). F = add_mkt 를 거친 그 코인 피쳐 프레임."""
    a = art["coins"][coin]
    return np.mean([_pred(a["M3"], F), _pred(a["M6"], F), _pred(art["PX3"], F), _pred(art["PX6"], F)], 0)


def masks(oks, fr, tr0, tr1, ca0, ca1):
    """연구 4라운드와 같은 행 규칙: 학습 = 모든 봉(라벨 끝 ≤ tr1) · 눈금 = 정시(:55) 결정 봉."""
    M = {}
    for s in COINS:
        ts, ok = oks[s]; ok = ok & np.isfinite(fr[s][0][R.MKT].to_numpy(float)).all(1); le = ts + pd.Timedelta(minutes=5 * R.H)
        M[s] = {"tr": ok & (ts >= tr0) & (le <= tr1), "ca": ok & (ts.minute == 55) & (ts >= ca0) & (le <= ca1)}
    return M


def fit(fr, M):
    from sklearn.linear_model import LogisticRegression
    art = {"coins": {}, "members": MEMBERS, "mkt": R.MKT}
    for s in COINS:
        F, y = fr[s]; tr = M[s]["tr"]
        art["coins"][s] = {"M3": hgb_fit(F[tr], y[tr], R.SETS["M3"]), "M6": mlp_fit(F[tr], y[tr], R.SETS["M4"])}
        print(s, "코인별 학습", int(tr.sum()), flush=True)
    A = pd.concat([fr[s][0][M[s]["tr"]] for s in COINS]); ya = np.concatenate([fr[s][1][M[s]["tr"]] for s in COINS])
    art["PX3"] = hgb_fit(A, ya, R.SETS["M3X"]); art["PX6"] = mlp_fit(A, ya, R.SETS["M4X"]); print("합쳐 학습", len(ya), flush=True)
    for s in COINS:
        F, y = fr[s]; ca = M[s]["ca"]; p = predict_log(art, s, F[ca]); a = art["coins"][s]
        a["c"] = float(np.log(np.mean(np.exp(y[ca] - p))))
        x = (p + a["c"] - F.lrv48.values[ca]) / 2                      # 확대 확률 입력 = ln(σ̂4h / 직전 4h σ)
        lr = LogisticRegression().fit(x[:, None], (y[ca] > F.lrv48.values[ca]).astype(int))
        a["expand_logit"] = (float(lr.intercept_[0]), float(lr.coef_[0, 0]))
    return art


def load_frames():
    fr, oks = {}, {}
    for s in COINS:
        x, F, y, ok, miss = R.frame(s); fr[s] = (F, y); oks[s] = (x.index, ok)
    R.add_mkt(fr)
    return fr, oks


# ---------- 라이브: 바이낸스 5분 klines 4코인 → QX4 ----------
def frames_from_klines(rows_by_coin: dict, now_ms: float) -> dict:
    """완결봉만 · 4코인 공통 시각 · h/l/c · v = 거래량(열 5) · n = 체결 수(열 8). 반환 = 코인 → add_mkt 거친 피쳐 프레임."""
    xs = {}
    for s in COINS:
        rows = [r for r in rows_by_coin[s] if int(r[6]) < now_ms]
        xs[s] = pd.DataFrame({"h": [float(r[2]) for r in rows], "l": [float(r[3]) for r in rows], "c": [float(r[4]) for r in rows],
                              "v": [float(r[5]) for r in rows], "n": [float(r[8]) for r in rows]}, index=pd.to_datetime([int(r[0]) for r in rows], unit="ms"))
    idx = xs[COINS[0]].index
    for s in COINS[1:]:
        idx = idx.intersection(xs[s].index)
    fr = {s: (R.features(xs[s].loc[idx]), None) for s in COINS}; R.add_mkt(fr)
    return {s: fr[s][0] for s in COINS}


def qx4_from_klines(rows_by_coin: dict, art: dict, now_ms: float, coin: str = "ETH") -> dict:
    """대시보드 vol4h_from_klines 와 같은 키. lvl = 최근 30일 예측 중앙 대비 · x_now = 직전 4h 실현 대비."""
    F = frames_from_klines(rows_by_coin, now_ms)[coin]
    good = F[np.isfinite(F[R.SETS["M4"] + R.MKT].to_numpy(float)).all(1)]
    if len(good) < 288:
        return {"ok": False, "reason": f"봉 부족({len(good)})"}
    a = art["coins"][coin]; tail = good.tail(30 * 288); sig = np.exp((predict_log(art, coin, tail) + a["c"]) / 2)
    s4, ref, now4 = float(sig[-1]), float(np.median(sig)), float(np.exp(good.lrv48.iloc[-1] / 2))
    px = float([r for r in rows_by_coin[coin] if int(r[6]) < now_ms][-1][4]); a0, a1 = a["expand_logit"]
    return {"ok": True, "asof_ms": int(good.index[-1].value // 10**6) + 300_000, "px": px, "sig4h_pct": round(100 * s4, 3),
            "sig4h_usd": round(px * s4, 2 if px >= 10 else 4), "lvl": round(s4 / ref, 3), "x_now": round(s4 / now4, 3),
            "p_expand": round(float(1 / (1 + np.exp(-(a0 + a1 * np.log(s4 / now4))))), 3), "ref_days": round(len(tail) / 288, 1),
            "model": "QX4", "model_train_end": art.get("train_end")}


# ---------- 점검·학습 ----------
def parity():
    """P1 4라운드 경계로 다시 학습한 QX4 = 저장된 연구 t_QX4(시험 행)·c · P2 klines 경로 피쳐 = 연구 F · P3 인과성(뒤 봉 망가뜨려도 같은 값)."""
    fr, oks = load_frames(); art = fit(fr, masks(oks, fr, R.TR4_START, R.TR4_END, R.VA4_START, R.VA4_END)); rep = {}
    for s in COINS:
        P = pd.read_parquet(R.OUT / f"pred_test_r4_{s}.parquet"); d = float(np.max(np.abs(predict_log(art, s, fr[s][0].loc[P.index]) - P.t_QX4.values)))
        c_ref = json.loads((R.OUT / f"report_r4_{s}.json").read_text())["models"]["QX4"]["c"]
        rep[s] = {"max_abs_diff": d, "c": art["coins"][s]["c"], "c_research": c_ref}; print(s, rep[s], flush=True)
    # P2·P3: 연구 5분봉(최근 53일)을 klines 행 모양으로 바꿔 라이브 경로에 넣는다
    xs = {s: R.load(s)[0].loc["2026-09-01":] for s in COINS}
    rows = {s: [[int(t.value // 10**6), 0, h, l, c, v, int(t.value // 10**6) + 299_999, 0, n] for t, h, l, c, v, n in
                zip(x.index, x.h, x.l, x.c, x.v, x.n)] for s, x in xs.items()}
    now = rows["ETH"][-1][6] + 1; Fk = frames_from_klines(rows, now)["ETH"]
    cols = R.SETS["M4"] + R.MKT; t = Fk.index[np.isfinite(Fk[cols].to_numpy(float)).all(1)]
    rep["P2_feature_maxdiff"] = float(np.max(np.abs(Fk.loc[t, cols].to_numpy(float) - fr["ETH"][0].loc[t, cols].to_numpy(float)))) if len(t) else None
    print("P2 klines 경로 피쳐 최대 차", rep["P2_feature_maxdiff"], "행", len(t))
    cut = rows["ETH"][-200][6] + 1; rows2 = {s: [list(r) for r in rows[s]] for s in COINS}
    for s in COINS:
        for r in rows2[s]:
            if r[6] >= cut:
                r[2] *= 1.3; r[4] *= 0.7
    rep["P3_causal"] = qx4_from_klines(rows, art, cut) == qx4_from_klines(rows2, art, cut); rep["live_sample"] = qx4_from_klines(rows, art, now)
    print("P3", rep["P3_causal"], rep["live_sample"])
    rep["pass"] = all(rep[s]["max_abs_diff"] < 1e-4 and abs(rep[s]["c"] - rep[s]["c_research"]) < 1e-4 for s in COINS) and rep["P3_causal"] and rep["P2_feature_maxdiff"] is not None and rep["P2_feature_maxdiff"] < 1e-9
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "parity.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str)); print("parity pass", rep["pass"])


def train():
    import joblib
    end = pd.Timestamp(R.SYS.END) + pd.Timedelta("23:55:00"); ca0 = end - pd.DateOffset(months=5); tr0 = ca0 - pd.DateOffset(years=2)
    fr, oks = load_frames(); art = fit(fr, masks(oks, fr, tr0, ca0, ca0, end))
    art.update(train_start=str(tr0), train_end=str(ca0.date()), calib_end=str(end.date()), recipe="docs/prereg/vol4h_forecast_horserace_20261025.md 5·6라운드")
    OUT.mkdir(parents=True, exist_ok=True); p = OUT / f"qx4_{end.date()}.joblib"; joblib.dump(art, p)
    print(p, {s: (round(art["coins"][s]["c"], 4), art["coins"][s]["expand_logit"]) for s in COINS})


if __name__ == "__main__":
    parity() if "--parity" in sys.argv else train() if "--train" in sys.argv else print(__doc__)
