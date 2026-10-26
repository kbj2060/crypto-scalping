#!/usr/bin/env python3
"""4h 변동성 QX4 서빙판 — 굴림 원점 3창 통과 레시피(docs/prereg/vol4h_forecast_horserace_20261025.md 5·6라운드)를 아티팩트로.

QX4 = 평균(M3 코인별 HGB · M6 코인별 MLP · PX3 4코인 합쳐 HGB(+MKT) · PX6 4코인 합쳐 MLP(+MKT)) 로그 RV.
피쳐는 연구 스크립트 features()/add_mkt 그대로 import(학습 = 라이브 같은 함수). 🔴qv 는 «거래량 × 종가»(학습 패널에 견적 거래량이 없어 그렇게 학습함).
출력 눈금 = 배포 HGB 와 같은 «중앙값형»(로그 공간 평균 σ, 눈금 상수 c 미적용 · 10-25 사용자 시안 B). 평균 RV 기준 σ 는 sig4h_mean_pct 로 따로.
확대 확률 = 배포판과 같은 정의(다음 4h σ ≥ 직전 4h σ × 1.3 · C=1e6).

    python3 scripts/live_vol4h_qx4_20261025.py --parity     # P1 연구 t_QX4·c 일치 · P2 klines 피쳐 = 연구 F · P3 절단 인과성 · P4 아티팩트 왕복 · P5 코인 지연 · P6 형성 봉 · P7 30일 기준 · P8 코인 빠짐
    python3 scripts/live_vol4h_qx4_20261025.py --train      # 최근: 학습 2년 + 눈금 5개월(끝 = SYS.END) → tmp/vol4h_qx4/qx4_<끝>.joblib

배포 안 함(대시보드·서버 연결은 사용자 승인 후). 실주문 없음. 서버에서 torch 불필요(MLP 는 numpy 가중치로 저장).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_vol4h_forecast_horserace_20261025 as R   # noqa: E402

OUT = ROOT / "tmp" / "vol4h_qx4"
COINS, MEMBERS = R.COINS, ("M3", "M6", "PX3", "PX6")
REF_BARS, WARM = 30 * 288, 8640              # 30일 기준 중앙 · lrv8640 예열
NEED_BARS = REF_BARS + WARM + 288            # 라이브 klines 코인마다 최소 봉 수(≈61일, 1000봉 × 19쪽)
STALE_MS, MAX_FILLED = 600_000, 36           # 마지막 공통 봉이 10분 넘게 늦으면 / 60일 안 빈 봉 36개(3시간) 넘으면 거절
EXPAND_K = 1.3                               # 배포판 p_expand 정의(live_eth_sizing_vol_model_20260912)


# ---------- 학습 (연구 fit_predict 와 같은 초모수·시드, 모델 객체를 남긴다) ----------
def hgb_fit(F, y, cols):
    from sklearn.ensemble import HistGradientBoostingRegressor
    A = np.nan_to_num(F[cols].to_numpy(float))
    return [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=int(s)).fit(A, y) for s in R.SEEDS], cols


def mlp_fit(F, y, cols, seeds=R.SEEDS[:2]):
    """R.mlp 와 같은 절차(15분 간격 표본 · 표준화 · 64-64 · Adam 15 에폭). 망은 numpy (W, b) 층 목록으로 내보낸다(서빙에 torch 불필요)."""
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
        nets.append([(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy()) for m in net if isinstance(m, torch.nn.Linear)])
    return {"layers": nets, "mu": mu, "sd": sd, "ym": float(ym), "ys": float(ys), "cols": cols}


def _mlp_np(layers, B):
    h = B.astype(np.float32)
    for k, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if k < len(layers) - 1:
            h = np.maximum(h, 0)
    return h[:, 0]


def _pred(m, F):
    if isinstance(m, dict):
        B = (np.nan_to_num(F[m["cols"]].to_numpy(float)) - m["mu"]) / m["sd"]
        return np.mean([_mlp_np(L, B) * m["ys"] + m["ym"] for L in m["layers"]], 0)
    models, cols = m
    return np.mean([h.predict(np.nan_to_num(F[cols].to_numpy(float))) for h in models], 0)


def predict_log(art, coin, F):
    """QX4 로그 RV(로그 공간 평균 = 중앙값형). F = add_mkt 를 거친 그 코인 피쳐 프레임."""
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
    import sklearn
    art = {"coins": {}, "members": MEMBERS, "mkt": list(R.MKT), "cols": {k: list(R.SETS[k]) for k in ("M3", "M4", "M3X", "M4X")},
           "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__}}
    for s in COINS:
        F, y = fr[s]; tr = M[s]["tr"]
        art["coins"][s] = {"M3": hgb_fit(F[tr], y[tr], R.SETS["M3"]), "M6": mlp_fit(F[tr], y[tr], R.SETS["M4"])}
        print(s, "코인별 학습", int(tr.sum()), flush=True)
    A = pd.concat([fr[s][0][M[s]["tr"]] for s in COINS]); ya = np.concatenate([fr[s][1][M[s]["tr"]] for s in COINS])
    art["PX3"] = hgb_fit(A, ya, R.SETS["M3X"]); art["PX6"] = mlp_fit(A, ya, R.SETS["M4X"]); print("합쳐 학습", len(ya), flush=True)
    for s in COINS:
        F, y = fr[s]; ca = M[s]["ca"]; p = predict_log(art, s, F[ca]); a = art["coins"][s]; prev = F.lrv48.values[ca]
        a["c"] = float(np.log(np.mean(np.exp(y[ca] - p))))                       # 평균 RV 눈금(sig4h_mean_pct 용)
        x = (p - prev) / 2                                                       # ln(중앙값형 σ̂4h / 직전 4h σ) — 서빙 입력과 같음
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(x[:, None], (y[ca] > prev + 2 * np.log(EXPAND_K)).astype(int))
        a["expand_logit"] = (float(lr.intercept_[0]), float(lr.coef_[0, 0]))
    return art


def load_frames():
    fr, oks = {}, {}
    for s in COINS:
        x, F, y, ok, miss = R.frame(s); fr[s] = (F, y); oks[s] = (x.index, ok)
    R.add_mkt(fr)
    return fr, oks


def load_qx4(path):
    """없거나 깨졌거나 피쳐 계약이 코드와 다르면 None — 호출부는 배포 HGB(vol4h_from_klines)로 떨어진다."""
    try:
        import joblib, sklearn
        art = joblib.load(path)
        assert art["mkt"] == list(R.MKT), "MKT 열 불일치"
        assert all(art["cols"][k] == list(R.SETS[k]) for k in art["cols"]), "피쳐 열 불일치"
        assert set(art["coins"]) == set(COINS)
        if art.get("versions", {}).get("sklearn") != sklearn.__version__:
            print(f"qx4: sklearn {art.get('versions', {}).get('sklearn')} 로 학습 · 지금 {sklearn.__version__}", flush=True)
        return art
    except Exception as exc:  # noqa: BLE001
        print(f"qx4 load failed: {exc}", flush=True)
        return None


# ---------- 라이브: 바이낸스 5분 klines 4코인 → QX4 ----------
def merge_klines(have: list, got: list, keep: int = NEED_BARS + 2000):
    """증분 캐시: 최근 1000봉(got)을 기존 행(have) 뒤에 잇는다. 겹침은 got 쪽으로 바꾼다(형성 중이던 봉 갱신).
    이어지지 않으면(서버 정지 등으로 got 첫 봉 > have 마지막 봉 + 5분) None → 호출부가 처음부터 다시 받는다."""
    if not have or not got:
        return None
    if int(got[0][0]) > int(have[-1][0]) + 300_000:
        return None
    first = int(got[0][0])
    return ([r for r in have if int(r[0]) < first] + list(got))[-keep:]


SYMS = {"ETH": "ETHUSDT", "XRP": "XRPUSDT", "BTC": "BTCUSDT", "SOL": "SOLUSDT"}


async def refresh_rows(fetch, rows: dict) -> bool:
    """fetch(params) → klines 행 목록(실패면 None). 코인마다 최근 1000봉을 이어 붙이고, 처음이거나 끊겼으면 NEED_BARS 까지 뒤로 받는다.
    rows 를 제자리에서 갱신 · 하나라도 실패하면 False(호출부는 배포 HGB 로)."""
    for coin, sym in SYMS.items():
        got = await fetch({"symbol": sym, "interval": "5m", "limit": 1000})
        if not got:
            return False
        merged = merge_klines(rows.get(coin) or [], got)
        if merged is None or len(merged) < NEED_BARS:
            out, end = list(got), int(got[0][0]) - 1
            while len(out) < NEED_BARS + 300:
                part = await fetch({"symbol": sym, "interval": "5m", "limit": 1000, "endTime": end})
                if not part:
                    return False
                out, end = part + out, int(part[0][0]) - 1
            merged = out
        rows[coin] = merged
    return True

def frames_from_klines(rows_by_coin: dict, now_ms: float):
    """완결봉만 · 4코인 공통 5분 격자(가장 늦게 시작 ~ 가장 일찍 끝난 코인) · 빈 봉은 학습(bars5)과 같은 규칙(종가 이어 붙임 · 고저 = 종가 · 거래 0).
    반환 = (코인 → add_mkt 거친 피쳐 프레임, 코인 → 5분 프레임(h/l/c/v/n), 코인 → 최근 60일 빈 봉 수)."""
    xs = {}
    for s in COINS:
        rows = [r for r in rows_by_coin[s] if int(r[6]) < now_ms]
        x = pd.DataFrame({"h": [float(r[2]) for r in rows], "l": [float(r[3]) for r in rows], "c": [float(r[4]) for r in rows],
                          "v": [float(r[5]) for r in rows], "n": [float(r[8]) for r in rows]}, index=pd.to_datetime([int(r[0]) for r in rows], unit="ms"))
        xs[s] = x[~x.index.duplicated(keep="last")].sort_index()
    grid = pd.date_range(max(x.index[0] for x in xs.values()), min(x.index[-1] for x in xs.values()), freq="5min"); filled = {}
    for s in COINS:
        x = xs[s].reindex(grid); miss = x.c.isna()
        x["c"] = x.c.ffill(); x["h"] = x.h.fillna(x.c); x["l"] = x.l.fillna(x.c); x[["v", "n"]] = x[["v", "n"]].fillna(0.0)
        xs[s] = x; filled[s] = int(miss.tail(60 * 288).sum())
    fr = {s: (R.features(xs[s]), None) for s in COINS}; R.add_mkt(fr)
    return {s: fr[s][0] for s in COINS}, xs, filled


def qx4_from_klines(rows_by_coin: dict, art: dict, now_ms: float, coin: str = "ETH") -> dict:
    """대시보드 vol4h_from_klines 와 같은 키·같은 눈금(중앙값형). lvl = 최근 30일 예측 중앙 대비 · x_now = 직전 4h 실현 대비."""
    try:
        if any(len(rows_by_coin.get(s) or []) == 0 for s in COINS):
            return {"ok": False, "reason": "코인 klines 빠짐 " + ",".join(s for s in COINS if not rows_by_coin.get(s))}
        Fs, xs, filled = frames_from_klines(rows_by_coin, now_ms); F = Fs[coin].iloc[WARM:]   # 예열 구간 버림: 첫 봉 수익 = 0(features 의 prepend) 이 들어간 창 제거
        good = F[np.isfinite(F[R.SETS["M4"] + R.MKT].to_numpy(float)).all(1)]
        if len(good) < REF_BARS:
            return {"ok": False, "reason": f"봉 부족({len(good)} < {REF_BARS}, 코인마다 {NEED_BARS}봉 필요)"}
        if max(filled.values()) > MAX_FILLED:
            return {"ok": False, "reason": f"빈 봉 많음 {filled}"}
        asof = int(good.index[-1].value // 10**6) + 300_000
        if now_ms - asof > STALE_MS:
            return {"ok": False, "reason": f"지연 {int((now_ms - asof) / 60000)}분"}
        a = art["coins"][coin]; tail = good.tail(REF_BARS); p = predict_log(art, coin, tail); sig = np.exp(p / 2)
        s4, ref, now4 = float(sig[-1]), float(np.median(sig)), float(np.exp(good.lrv48.iloc[-1] / 2))
        px = float(xs[coin].c.loc[good.index[-1]]); a0, a1 = a["expand_logit"]; sm = s4 * float(np.exp(a["c"] / 2))
        return {"ok": True, "asof_ms": asof, "px": px, "sig4h_pct": round(100 * s4, 3), "sig4h_usd": round(px * s4, 2 if px >= 10 else 4),
                "lvl": round(s4 / ref, 3), "x_now": round(s4 / now4, 3), "p_expand": round(float(1 / (1 + np.exp(-(a0 + a1 * np.log(s4 / now4))))), 3),
                "ref_days": round(len(tail) / 288, 1), "sig4h_mean_pct": round(100 * sm, 3), "filled_bars_60d": filled[coin],
                "model": "QX4", "model_train_end": art.get("train_end")}
    except Exception as exc:  # noqa: BLE001 — 화면이 죽지 않게: 호출부는 ok False 면 배포 HGB 로 떨어진다
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:160]}


# ---------- 점검·학습 ----------
def _rows(x):
    return [[int(t.value // 10**6), 0, h, l, c, v, int(t.value // 10**6) + 299_999, 0, n] for t, h, l, c, v, n in zip(x.index, x.h, x.l, x.c, x.v, x.n)]


def parity():
    import joblib
    fr, oks = load_frames(); art = fit(fr, masks(oks, fr, R.TR4_START, R.TR4_END, R.VA4_START, R.VA4_END)); rep = {}
    for s in COINS:   # P1 연구 시험 예측·눈금 c 재현(numpy MLP 포함)
        P = pd.read_parquet(R.OUT / f"pred_test_r4_{s}.parquet"); d = float(np.max(np.abs(predict_log(art, s, fr[s][0].loc[P.index]) - P.t_QX4.values)))
        c_ref = json.loads((R.OUT / f"report_r4_{s}.json").read_text())["models"]["QX4"]["c"]
        rep[s] = {"max_abs_diff": d, "c": art["coins"][s]["c"], "c_research": c_ref}; print(s, rep[s], flush=True)
    xs = {s: R.load(s)[0].loc["2026-08-01":] for s in COINS}; rows = {s: _rows(x) for s, x in xs.items()}; now = rows["ETH"][-1][6] + 1
    full = frames_from_klines(rows, now)[0]; Fk = full["ETH"].iloc[WARM:]; cols = R.SETS["M4"] + R.MKT; t = Fk.index[np.isfinite(Fk[cols].to_numpy(float)).all(1)]
    rep["P2_feature_maxdiff"] = float(np.max(np.abs(Fk.loc[t, cols].to_numpy(float) - fr["ETH"][0].loc[t, cols].to_numpy(float))))   # P2
    worst = 0.0
    for k in (1, 7, 100, 500):   # P3 절단 인과성: 뒤 k봉을 빼도 공통 행 피쳐 동일(ETH·SOL)
        Ft = frames_from_klines({s: r[:-k] for s, r in rows.items()}, now)[0]
        for s in ("ETH", "SOL"):
            ix = Ft[s].index.intersection(t); worst = max(worst, float(np.nanmax(np.abs(Ft[s].loc[ix, cols].to_numpy(float) - full[s].loc[ix, cols].to_numpy(float)))))
    rep["P3_truncation_maxdiff"] = worst
    live = qx4_from_klines(rows, art, now); OUT.mkdir(parents=True, exist_ok=True); joblib.dump(art, OUT / "_parity_art.joblib")
    art2 = load_qx4(OUT / "_parity_art.joblib"); rep["P4_roundtrip_equal"] = art2 is not None and qx4_from_klines(rows, art2, now) == live
    lag = dict(rows, SOL=rows["SOL"][:-1]); lv = qx4_from_klines(lag, art, now)   # P5 SOL 한 봉 늦음 → 한 봉 이전 · px 도 그 봉
    rep["P5_lag"] = bool(lv["ok"] and lv["asof_ms"] == live["asof_ms"] - 300_000 and lv["px"] == float(xs["ETH"].c.iloc[-2]))
    form = {s: r + [[r[-1][0] + 300_000, 0, 9e9, 0, 9e9, 9e9, now + 10**9, 0, 1]] for s, r in rows.items()}   # P6 형성 중 봉은 무시
    rep["P6_forming_ignored"] = qx4_from_klines(form, art, now) == live
    rep["P7_ref_days"] = live.get("ref_days"); rep["live_sample"] = live
    rep["P8_missing_coin"] = qx4_from_klines(dict(rows, SOL=[]), art, now)["ok"] is False
    rep["pass"] = (all(rep[s]["max_abs_diff"] < 1e-4 and abs(rep[s]["c"] - rep[s]["c_research"]) < 1e-4 for s in COINS) and rep["P2_feature_maxdiff"] < 1e-6
                   and rep["P3_truncation_maxdiff"] == 0.0 and rep["P4_roundtrip_equal"] and rep["P5_lag"] and rep["P6_forming_ignored"]
                   and (rep["P7_ref_days"] or 0) >= 30 and rep["P8_missing_coin"] and live["ok"])
    (OUT / "parity.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str)); print(json.dumps({k: v for k, v in rep.items() if k not in COINS}, ensure_ascii=False, default=str))


def train():
    import joblib, subprocess
    end = pd.Timestamp(R.SYS.END) + pd.Timedelta("23:55:00"); ca0 = end - pd.DateOffset(months=5); tr0 = ca0 - pd.DateOffset(years=2)
    fr, oks = load_frames(); art = fit(fr, masks(oks, fr, tr0, ca0, ca0, end))
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    art.update(train_start=str(tr0), train_end=str(ca0.date()), calib_end=str(end.date()), code_sha=sha, seeds=list(map(int, R.SEEDS)),
               recipe="docs/prereg/vol4h_forecast_horserace_20261025.md 5·6라운드")
    OUT.mkdir(parents=True, exist_ok=True); p = OUT / f"qx4_{end.date()}.joblib"; joblib.dump(art, p)
    assert load_qx4(p) is not None
    print(p, {s: (round(art["coins"][s]["c"], 4), art["coins"][s]["expand_logit"]) for s in COINS})


if __name__ == "__main__":
    parity() if "--parity" in sys.argv else train() if "--train" in sys.argv else print(__doc__)
