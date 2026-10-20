#!/usr/bin/env python3
"""**XRP 박스 유지 확률** — 대시보드 ETH 모델(dashboard/box_hold_hgb_20261016.joblib)과 같은 정의·같은 학습법을 XRPUSDT 에 (2026-10-20).

라벨  scripts/research_eth_range_hold_multiscale_20261016.build() 그대로(import): 결정 = 5분봉 마감 분(30m·1h 5분마다, 4h 1시간마다),
      박스 = 종가 ± 0.5×지난 W분 고저폭, 유지 = 다음 W분(i+1..i+W) 1분봉 고가 < 상단 & 저가 > 하단. 특징은 i 분 마감까지(경계 계약).
특징  scripts/research_eth_box_hold_improve_20261016.features() 의 서빙 27칸(F·V·R·Q·S·M)만 복사 -- 서빙에서 뺀 7일 칸(V_lsig7d·Q_volz)과
      X 묶음(BTC·OI·eth_premium_5m)은 계산 안 함(서빙 모델이 안 씀 → XRP 프리미엄 원천도 필요 없음). S 계절표(PROF)는 XRP 2022~24 로 다시.
모델  ETH export 와 같다: 30m·1h = HGB 27칸, 4h = HGB F 6칸, 2022~24 학습. 비교: 기하 G(log d_σ 로지스틱) · F 로지스틱 6칸 ·
      ETH 배포 모델을 XRP 특징에(그대로 box_hold.py 에 XRP 1분봉을 넣는 경우 = S 칸은 ETH PROF) · 진단 4h HGB 27칸.
채점  2025 · 2026 · 2025~ 따로. AUC · 10분위 보정(최대 차 pp) · ΔAUC 일 블록 부트스트랩 95% CI(B=200).
판정(사전)  W 마다 2025·2026 둘 다 AUC ≥ .72 · 보정 최대 차 ≤ 6pp · ΔAUC(HGB − G) CI 하한 > 0. 실패한 W 는 artifact 에 deployable=False.
  nice python scripts/research_xrp_box_hold_20261020.py            # 라벨·특징·평가·export·패리티·라이브 점검
  python scripts/research_xrp_box_hold_20261020.py --selftest
"""
from __future__ import annotations

import glob
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/kbj20/crypto-scalping")
BV = MAIN / "data/binance_vision"
SYM = "XRPUSDT"
OUT = MAIN / "tmp/xrp_parity_20261020/box_hold"
ART = OUT / "box_hold_hgb_xrp_20261020.joblib"
ETH_ART = ROOT / "dashboard/box_hold_hgb_20261016.joblib"
# 거시 일정 캐시(FRED 날짜 + 관례 시각) -- ETH 연구 산출 그대로 복사해 쓴다(같은 발표 = 자산 무관)
EV_SRC = MAIN / ".claude/worktrees/futures-discount-basis-c5126d/tmp/box_improve_20261016/events.json"
SCALES = {"30m": 30, "1h": 60, "4h": 240}
log = lambda m: print(m, flush=True)  # noqa: E731


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


ms = _load("research_eth_range_hold_multiscale_20261016")
imp = _load("research_eth_box_hold_improve_20261016")
ms.SCALES = {k: v for k, v in ms.SCALES.items() if k in SCALES}      # 24h 는 서빙에 없음
imp.OUT = OUT                                                         # events() 캐시 위치
F_BASE, roll = imp.F_BASE, imp.roll
LEAN = F_BASE + ["V_lsig5m", "V_lsig15m", "V_lsig1h", "V_lsig4h", "V_lsig24", "V_park1h", "V_park24", "V_jump1h",
                 "R_edge", "R_trend", "R_compress", "R_squeeze", "Q_vol5_vs_W", "Q_imbW", "Q_imb1h", "S_season_fut", "S_season_now",
                 "M_major_in", "M_major_pos", "M_major_recent", "M_claims_in"]
RECIPE = {"30m": LEAN, "1h": LEAN, "4h": F_BASE}
hgb = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,  # noqa: E731
                                             l2_regularization=1.0, random_state=0)


def load_1m() -> pd.DataFrame:
    fs = sorted(glob.glob(f"{BV}/klines1m/{SYM}-1m-*.parquet")) + sorted(glob.glob(f"{BV}/klines1m_daily/{SYM}-1m-*.parquet"))
    K = pd.concat([pd.read_parquet(f)[["t", "h", "l", "c", "v", "tb"]] for f in fs]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    assert (np.diff(K.t) == 60000).all(), "1분봉 빈칸"
    return K


def season_prof(K: pd.DataFrame) -> np.ndarray:
    """요일×시각 평균 1분 수익² / 전체 평균 -- 학습 구간(2022~24)만 (imp.features 와 같은 식)."""
    t = K.t.to_numpy(); r2 = np.r_[0, np.diff(np.log(K.c.to_numpy(float)))] ** 2
    ts1 = pd.to_datetime(t, unit="ms"); how = (ts1.dayofweek * 24 + ts1.hour).to_numpy()
    tr = t < pd.Timestamp("2025-01-01").value // 10**6
    prof = pd.Series(r2[tr]).groupby(how[tr]).mean().reindex(range(168)).to_numpy()
    return prof / np.nanmean(r2[tr])


def features(D: pd.DataFrame, K: pd.DataFrame, prof: np.ndarray) -> pd.DataFrame:
    """imp.features 의 서빙 칸만(7일·X 묶음 뺌) -- 식은 한 줄씩 그대로."""
    t = K.t.to_numpy(); h, l, c, v, tb = (K[k].to_numpy(float) for k in ("h", "l", "c", "v", "tb"))
    r = np.r_[0, np.diff(np.log(c))]
    r2 = r ** 2
    s24 = np.sqrt(roll(r2, 1440))
    pk = np.log(h / l) ** 2 / (4 * np.log(2))
    pos = pd.Series(np.arange(len(t)), index=t)
    i = pos.reindex(D.ts.astype("int64").to_numpy() // 10**6).to_numpy()
    assert np.isfinite(i).all(), "결정 분이 1분봉에 없음"
    i = i.astype(int)
    tclose = t[i] + 60000
    X = pd.DataFrame(index=D.index)
    for w, nm in ((5, "5m"), (15, "15m"), (60, "1h"), (240, "4h")):
        X[f"V_lsig{nm}"] = np.log(np.sqrt(roll(r2, w, mp=w // 2))[i] + 1e-9) - np.log(s24[i])
    X["V_lsig24"] = np.log(s24[i])
    X["V_park1h"] = np.log(np.sqrt(roll(pk, 60))[i] + 1e-9) - np.log(s24[i])
    X["V_park24"] = np.log(np.sqrt(roll(pk, 1440))[i] + 1e-9) - np.log(s24[i])
    X["V_jump1h"] = roll(np.abs(r), 60, "max")[i] / s24[i]
    for nm, W in SCALES.items():
        m = (D.scale == nm).to_numpy()
        ii = i[m]
        whi, wlo = roll(h, W, "max"), roll(l, W, "min")
        rng = whi - wlo
        prng = np.r_[np.full(W, np.nan), rng[:-W]]
        p = (c - wlo) / rng
        X.loc[m, "R_edge"] = np.minimum(p, 1 - p)[ii]
        X.loc[m, "R_trend"] = (np.abs(c - np.r_[np.full(W, np.nan), c[:-W]]) / rng)[ii]
        X.loc[m, "R_compress"] = np.log(rng / prng)[ii]
        X.loc[m, "R_squeeze"] = np.log(rng / roll(rng, 1440, "median", mp=720))[ii]
        vW = roll(v, W)
        X.loc[m, "Q_vol5_vs_W"] = np.log(roll(v, 5) / vW)[ii]
        X.loc[m, "Q_imbW"] = np.abs(roll(2 * tb - v, W, "sum") / roll(v, W, "sum"))[ii]
    X["Q_imb1h"] = np.abs(roll(2 * tb - v, 60, "sum") / roll(v, 60, "sum"))[i]
    # S: 서빙(box_hold.py)과 비트 단위로 같은 식 -- 분-of-주(10080칸)마다 같은 순서의 np.mean 표를 만들어 찾는다.
    #   10-20 패리티 실패에서 찾음: imp.features 의 pandas rolling 평균은 누적합 잡음(1e-16)이 있고, S 칸은 값 종류가 적어 HGB 경계가
    #   데이터 값 그 자체에 선다 → 서빙 np.mean 값이 경계 반대편에 떨어져 확률이 최대 ~0.7pp 어긋났다. ETH 배포판(0929bb79)도 같은 잡음.
    ts1 = pd.to_datetime(t[i], unit="ms")
    mw = (ts1.dayofweek * 1440 + ts1.hour * 60 + ts1.minute).to_numpy()
    for nm, W in SCALES.items():
        m = (D.scale == nm).to_numpy()
        tab = np.array([np.log(np.mean([prof[((u + k) % 10080) // 60] for k in range(1, W + 1)])) for u in range(10080)])
        X.loc[m, "S_season_fut"] = tab[mw[m]]
    X["S_season_now"] = np.array([np.log(np.mean([prof[((u - k) % 10080) // 60] for k in range(60)])) for u in range(10080)])[mw]
    ev = imp.events()
    W_row = D.scale.map(SCALES).to_numpy()
    nm_ = imp.next_ev(tclose, ev["major"])
    X["M_major_in"] = (nm_ <= W_row).astype(float)
    X["M_major_pos"] = np.where(nm_ <= W_row, nm_ / W_row, 1.0)
    X["M_major_recent"] = (imp.prev_ev(tclose, ev["major"]) <= 60).astype(float)
    X["M_claims_in"] = (imp.next_ev(tclose, ev["claims"]) <= W_row).astype(float)
    return X


def leak_check(D: pd.DataFrame, K: pd.DataFrame, prof: np.ndarray) -> None:
    """결정 분 뒤 1분봉을 망가뜨려도 특징(S·M = 미리 아는 달력 제외)이 같아야 한다."""
    for nm in SCALES:
        sub = D[D.scale == nm].iloc[[len(D[D.scale == nm]) // 2]]
        j = int(pd.Series(np.arange(len(K)), index=K.t)[int(sub.ts.iloc[0].value // 10**6)]) + 1
        K2 = K.copy(); K2.loc[j:, ["h", "l", "c"]] *= 1.5; K2.loc[j:, ["v", "tb"]] *= 7
        a, b = features(sub, K, prof), features(sub, K2, prof)
        cols = [c for c in a.columns if not c.startswith(("S_", "M_"))]
        assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True), (nm, (a[cols] - b[cols]).T)
    log("누수 점검 OK — 결정 분 뒤 1분봉을 바꿔도 V·R·Q 특징 불변(3척도)")


def reliability(y, p):
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((g.p - g.y).abs().max()), g.round(4).values.tolist()


def evaluate(A: pd.DataFrame, A_ethprof: pd.DataFrame, eth: dict) -> dict:
    rep = {}
    for nm in SCALES:
        g = A[A.scale == nm].dropna(subset=LEAN)
        ge = A_ethprof.loc[g.index]
        tr, te = g[g.ts < "2025-01-01"], g[g.ts >= "2025-01-01"]
        P, thr = {}, {}
        specs = {"G 기하": ("lr0", ["log_ds"]), "F 로지스틱": ("lr", F_BASE), "XRP HGB(배포 구성)": ("hgb", RECIPE[nm])}
        if nm == "4h":
            specs["진단 4h HGB 27칸"] = ("hgb", LEAN)
        for k, (kind, cols) in specs.items():
            mdl = {"lr0": lambda: LogisticRegression(C=1.0, max_iter=1000), "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                   "hgb": hgb}[kind]().fit(tr[cols], tr.hold)
            ptr = mdl.predict_proba(tr[cols])[:, 1]
            thr[k] = (float(np.quantile(ptr, .2)), float(np.quantile(ptr, .8)))
            P[k] = mdl.predict_proba(te[cols])[:, 1]
        ec = eth["scales"][nm]
        P["ETH 모델 그대로"] = ec["model"].predict_proba(ge.loc[te.index, ec["cols"]])[:, 1]
        thr["ETH 모델 그대로"] = (ec["q20"], ec["q80"])
        main = "XRP HGB(배포 구성)"
        rep[nm] = {"train_n": int(len(tr)), "train_base": float(tr.hold.mean()), "cols": RECIPE[nm], "periods": {}}
        log(f"\n[{nm}] 학습 {len(tr):,} (유지 {tr.hold.mean():.3f}) · 채점 {len(te):,} {te.ts.min():%Y-%m-%d}~{te.ts.max():%Y-%m-%d}")
        for per in ("2025", "2026", "2025~"):
            mk = np.ones(len(te), bool) if per == "2025~" else (te.ts.dt.year == int(per)).to_numpy()
            y = te.hold.to_numpy()[mk]; day = te.ts.dt.floor("D").to_numpy()[mk]
            base = np.full(mk.sum(), tr.hold.mean())
            R = {"n": int(mk.sum()), "days": int(len(np.unique(day))), "hold_rate": float(y.mean()), "models": {}}
            log(f"  {per} n {mk.sum():,} 유지율 {y.mean():.3f}")
            for k, pp in P.items():
                p = pp[mk]; auc = roc_auc_score(y, p); ce, tab = reliability(y, p); lo, hi = thr[k]
                d = {"auc": auc, "cal_maxgap_pp": ce * 100, "reliability_p_y_n": tab, "logloss": log_loss(y, np.clip(p, 1e-4, 1 - 1e-4)),
                     "bss": float(1 - np.mean((p - y) ** 2) / np.mean((base - y) ** 2)), "mean_p": float(p.mean()),
                     "lo_q20_hold": float(y[p < lo].mean()), "lo_share": float(np.mean(p < lo)),
                     "hi_q80_hold": float(y[p > hi].mean()), "hi_share": float(np.mean(p > hi))}
                if k == main:
                    for ref in ("G 기하", "F 로지스틱", "ETH 모델 그대로"):
                        d[f"dauc_vs[{ref}]"] = auc - roc_auc_score(y, P[ref][mk])
                        d[f"dauc_ci_vs[{ref}]"] = imp.block_ci(y, p, P[ref][mk], day)
                R["models"][k] = d
                extra = "".join(f" Δ{r[:3]} {d[f'dauc_vs[{r}]']:+.4f}{d[f'dauc_ci_vs[{r}]']}" for r in ("G 기하", "F 로지스틱", "ETH 모델 그대로")) if k == main else ""
                log(f"    {k:18s} AUC {auc:.4f} 보정 {ce * 100:4.1f}pp 평균예측 {p.mean():.3f} BSS {d['bss']:+.3f} "
                    f"깨질 {d['lo_q20_hold'] * 100:4.1f}%({d['lo_share'] * 100:.0f}%) 머물 {d['hi_q80_hold'] * 100:4.1f}%({d['hi_share'] * 100:.0f}%){extra}")
            rep[nm]["periods"][per] = R
        m = {per: rep[nm]["periods"][per]["models"][main] for per in ("2025", "2026")}
        gate = {"auc>=.72": all(x["auc"] >= .72 for x in m.values()), "cal<=6pp": all(x["cal_maxgap_pp"] <= 6 for x in m.values()),
                "beats_G_ci>0": all(x["dauc_ci_vs[G 기하]"][0] > 0 for x in m.values())}
        gate["pass"] = all(gate.values())
        rep[nm]["gate"] = gate
        log(f"  판정 {gate}")
    return rep


def export(A: pd.DataFrame, prof: np.ndarray, gate: dict) -> None:
    """box_hold.py 가 읽는 구조 그대로(prof · train · scales[W]{model,cols,train_base,n,q20,q80}) + symbol · deployable."""
    out = {"prof": prof.tolist(), "train": ["2022-01-01", "2024-12-31"], "symbol": SYM, "model": "hgb-xrp-20261020", "scales": {}}
    for nm in SCALES:
        cols = RECIPE[nm]
        tr = A[(A.scale == nm) & (A.ts < "2025-01-01")].dropna(subset=LEAN)
        m = hgb().fit(tr[cols], tr.hold)
        p = m.predict_proba(tr[cols])[:, 1]
        out["scales"][nm] = {"model": m, "cols": cols, "train_base": float(tr.hold.mean()), "n": int(len(tr)),
                             "q20": float(np.quantile(p, .2)), "q80": float(np.quantile(p, .8)), "deployable": bool(gate[nm]["pass"])}
        log(f"[export {nm}] 칸 {len(cols)} · 학습 {len(tr):,} · 기저 {tr.hold.mean():.3f} · q20 {np.quantile(p, .2):.3f} q80 {np.quantile(p, .8):.3f} · deployable {gate[nm]['pass']}")
    joblib.dump(out, ART, compress=3)
    log(f"→ {ART} ({ART.stat().st_size / 1e6:.2f} MB)")


def _bh():
    """dashboard/box_hold.py 를 고치지 않고 XRP 묶음으로 바꿔 끼운 모듈."""
    sys.path.insert(0, str(ROOT))
    import dashboard.box_hold as bh
    art = joblib.load(ART)
    bh.B, bh.PROF = art, np.asarray(art["prof"], float)
    return bh


def _evl():
    ev = imp.events()
    return [{"time_utc": pd.Timestamp(int(t), unit="ms", tz="UTC").isoformat(), "category": "fomc", "title_ko": "FOMC 금리결정 발표"} for t in ev["major"]] + \
           [{"time_utc": pd.Timestamp(int(t), unit="ms", tz="UTC").isoformat(), "category": "econ", "detail": "release_id=180"} for t in ev["claims"]]


def parity(A: pd.DataFrame, K: pd.DataFrame) -> float:
    """연구 행 40개(발표 낀 10 포함) = box_hold.features/payload 가 같은 1분봉 1499개로 낸 특징·확률."""
    bh, evl = _bh(), _evl()
    S = A[A.ts >= "2025-01-01"].dropna(subset=LEAN)
    pick = pd.concat([S[S.M_major_in == 1].sample(10, random_state=1), S.sample(30, random_state=2)])
    pos = pd.Series(np.arange(len(K)), index=K.t); worst = 0.0
    for _, row in pick.iterrows():
        i = int(pos[int(row.ts.value // 10**6)])
        k = K.iloc[i - bh.NEED + 1:i + 1]
        k1m = pd.DataFrame({"timestamp": pd.to_datetime(k.t, unit="ms", utc=True), "high": k.h, "low": k.l, "close": k.c,
                            "volume": k.v, "taker_buy_base": k.tb}).reset_index(drop=True)
        cols = bh.B["scales"][row.scale]["cols"]
        a, b = np.array([bh.features(k1m, evl)[row.scale][c] for c in cols]), row[cols].to_numpy(float)
        worst = max(worst, float(np.max(np.abs(a - b))))
        assert np.allclose(a, b, rtol=1e-6, atol=1e-8), (row.scale, row.ts, [(c, x, y) for c, x, y in zip(cols, a, b) if not np.isclose(x, y)])
        pp = bh.payload(k1m, evl)["scales"][row.scale]["p"]
        assert abs(pp - bh.B["scales"][row.scale]["model"].predict_proba(row[cols].to_frame().T.astype(float))[0, 1]) < 1e-4
    log(f"패리티 OK — 연구 행 {len(pick)}개 특징 최대 차 {worst:.2e} · box_hold.payload 확률 같음")
    return worst


def live_check() -> dict:
    """공개 REST(서명 없음) 최신 1500개 1분봉 → server.liq_klines_df 와 같은 모양 → box_hold.payload."""
    import time
    import requests
    # 공개 GET 1회뿐(이 PC 는 라이브 서버와 공인 IP 공유 -- 10-20 22:08 다른 작업 429 로 서버 손절 감시가 잠깐 막힘). 418/429 면 바로 멈춘다.
    res = requests.get("https://fapi.binance.com/fapi/v1/klines", params={"symbol": SYM, "interval": "1m", "limit": 1500}, timeout=20)
    res.raise_for_status()
    raw = res.json()
    df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
                                    "taker_buy_base", "taker_buy_quote", "ignore"])
    for c in ("high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    df["close_time"] = df["close_time"].astype("int64")
    df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    if int(df.iloc[-1]["close_time"]) >= int(time.time() * 1000):
        df = df.iloc[:-1].reset_index(drop=True)
    out = _bh().payload(df, _evl())
    log(f"라이브 {len(df)}봉 · 마지막 {df.timestamp.iloc[-1]} → {json.dumps(out, ensure_ascii=False)}")
    assert out["ok"] and all(s["ok"] and 0 < s["p"] < 1 for s in out["scales"].values()), out
    return out


def selftest():
    imp.selftest(); ms.selftest()
    assert len(LEAN) == 27 and len(set(LEAN)) == 27
    log("selftest OK")


if __name__ == "__main__":
    selftest()
    if "--selftest" in sys.argv:
        sys.exit()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "events.json").exists():
        shutil.copy(EV_SRC, OUT / "events.json")
    K = load_1m()
    log(f"{SYM} 1분봉 {len(K):,} · {pd.Timestamp(K.t.iloc[0], unit='ms')} ~ {pd.Timestamp(K.t.iloc[-1], unit='ms')}")
    D = ms.build(K[["t", "h", "l", "c"]])
    prof = season_prof(K)
    leak_check(D, K, prof)
    eth = joblib.load(ETH_ART)
    X = features(D, K, prof)
    Xe = features(D, K, np.asarray(eth["prof"], float))           # ETH 모델 그대로 = S 칸도 ETH 계절표
    A = pd.concat([D[["scale", "ts", "hold"] + F_BASE], X], axis=1).replace([np.inf, -np.inf], np.nan)
    Ae = pd.concat([D[["scale", "ts", "hold"] + F_BASE], Xe], axis=1).replace([np.inf, -np.inf], np.nan)
    rep = evaluate(A, Ae, eth)
    export(A, prof, {nm: rep[nm]["gate"] for nm in SCALES})
    A.to_parquet(OUT / "features.parquet")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))   # 패리티·라이브 전에 한 번 남김
    rep["_meta"] = {"symbol": SYM, "klines": [str(pd.Timestamp(K.t.iloc[0], unit="ms")), str(pd.Timestamp(K.t.iloc[-1], unit="ms"))],
                    "train": "2022-01-01~2024-12-31", "artifact": str(ART), "parity_max_abs_diff": parity(A, K),
                    "dropped": "X 묶음(BTC σ·OI·eth_premium_5m·상위 포지션) -- 서빙 모델이 안 쓰는 칸이라 계산 안 함",
                    "live_check": live_check()}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"→ {OUT / 'report.json'}")
