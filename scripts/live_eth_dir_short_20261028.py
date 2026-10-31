#!/usr/bin/env python3
"""ETH 15분·1시간 방향 예측 — 서빙 (2026-10-28, docs/rl_v27_handoff_20261028.md §6 · 사용자 «15분·1시간 방향 예측 서빙 모델을 만들어줘»).

모델 = 연구 걸어가기 단계 F 최종(claude/frosty-aryabhata-2c90b2 scripts/diag_rl_dir_walkforward_20261028.py · ARMS ens_hl180·ens_monthly · F65 · EXTRA=book).
  입력 75열 = 65피쳐(live_eth_dir4h.build_features) + 호가 깊이 10열(diag_rl_dir_liqfeat feats()["book"] 식) — 결정 시각 t 에 봉 t−5분 값.
  표적 = 다음 h봉 단순수익 합 / σ24(봉 마감 t 까지 288봉 수익 표준편차) · 4표적(rg·±2 절단·Huber α1·부호)을 따로 학습해 학습 행 예측 표준편차로 나눠 합.
  LightGBM 진단 설정 BASE(조기 종료 없음) × 시드 5 평균 = 지평당 20모형. 학습 행 = 2024-01-01 ~ 표적이 그 달 첫 봉 전에 확정된 행(i ≤ m0−h−2).
  15분(h=3) = 매달 1일 확장 창 + 반감기 180일 가중 · 1시간(h=12) = 매달 1일 확장 창(가중 없음).
  연구 IC 시험 .061/.059 · 전진 .084/.070(적중 53%) — 약한 신호. 출력 = 무단위 점수(크기 비교용, bp 아님).
원천: 학습 = vision 패널 PANEL(2022-01-01~ — 연구 panel_fwd 와 같은 시작이라 굴림 std·ewm 이 비트 단위로 같다)
            + bookDepth BOOK(liqlab 동결본 씨앗 + 일별 아카이브 이어 받기, liqlab_data_20261028 과 같은 규칙)
      실시간 = live_eth_dir4h.rest_frames(4h 와 같은 REST 경로) + 호가 깊이 워커 sqlite(warm=0 봉만 — 아니면 예측을 비운다).

    python scripts/live_eth_dir_short_20261028.py --train [--month 2026-11-01]          # 이어 받기 + 학습 → ARTIFACT
    python scripts/live_eth_dir_short_20261028.py --parity PANEL_DIR LIQ_PARQUET DIAG_DIR  # (서버) 연구 전진 예측과 같은 달 모형 대조
    python scripts/live_eth_dir_short_20261028.py --train65 [--month 2026-11-01]        # BTC·SOL·XRP 용 65열판(학습 코인 ETH·BTC·XRP) → ART65
    python scripts/live_eth_dir_short_20261028.py --selftest
    python scripts/live_eth_dir_short_20261028.py [--coin btc|sol|xrp]                  # (서버) 지금 예측 한 줄

2026-10-31 BTC·SOL·XRP(docs/experiments/coin_dir_short_btc_sol_20261031.md · xrp_dir_short_20261031.md, 시험 26-06~10 엄격 통과):
  호가 깊이 없는 65열판 · 지평마다 연구가 고른 학습 코인의 모형을 대상 코인 피쳐에 적용(TABLE) · 대상 바꾸기 = d4.frames_for.
"""
from __future__ import annotations

import io
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_dir4h_20261015 as d4  # noqa: E402

ARTIFACT = ROOT / "data/live/eth_dirshort_model.joblib"
PANEL = ROOT / "data/live/dirshort_panel"             # 처음엔 연구 panel_fwd(2022-01-01~) 복사본을 씨앗으로 둔다 — 없으면 2022~ 전부를 받으려 들므로 멈춘다
BOOK = ROOT / "data/live/dirshort_book.parquet"
LIQ = ROOT / "data/liqlab/ETHUSDT_5m.parquet"         # BOOK 첫 씨앗 = 연구 학습 원천(2024-01-01~2026-10-23)
BOOK_DB = ROOT / "data/live/bookdepth/ethusdt.sqlite"
# 표본 밖 예측 = 연구 같은 구성 5시드 평균(시험 2026-06-01~09-14 + 전진 09-15~10-23, 41,760봉, ts = 결정 시각) — 문턱·분위의 잣대.
#   그 창에서 위·아래 20% 밖 적중 15분 56.1%(시험)·56.1%(전진) vs 가운데 51% · 1시간 54.6%·55.1% vs 51~52% → 4h 칸과 같은 20/80 규칙.
OOS_SEED = ROOT / "data/live/eth_dirshort_oos_seed.parquet"
PANEL_START, START, DAY = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01"), 86_400_000_000_000
BCOLS = ["d-1", "d-2", "d-5", "d1", "d2", "d5", "d_age"]
BOOKF = ["dimb1", "dimb2", "dimb5", "dimb1_d1h", "dimb2_d1h", "dimb5_d1h", "depth1_z", "bid_near", "ask_near", "d_age"]   # = live_bookdepth_feat FEAT
HZ = {3: 180, 12: None}                               # 지평(봉) → 반감기 일수(None = 가중 없음)
NAMES = {3: "15m", 12: "1h"}
SEEDS = [613866575, 836385207, 1170305695, 2083046697, 504567861]
ENS = ("rg", "clip2", "huber", "sign")
BASE = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.5, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0)
NJ = 4                                                # 연구와 같은 스레드 수 — 히스토그램 합 순서까지 같게
# 코인 → {지평: 학습 코인}(연구 검증 IC 로 고른 팔). ETH 는 75열판 ARTIFACT 그대로.
TABLE = {"BTC": {3: "BTC", 12: "ETH"}, "SOL": {3: "ETH", 12: "ETH"}, "XRP": {3: "XRP", 12: "ETH"}}
ART65 = {s: ROOT / f"data/live/{s.lower()}_dirshort65_model.joblib" for s in ("ETH", "BTC", "XRP")}
NJ65 = {"ETH": 1, "BTC": 1, "XRP": 4}                # 그 팔을 낸 연구 실행의 스레드 수(코인 연구 1 · XRP 연구 4) — 예측이 비트 단위로 같게
OOS_SEEDS = {"ETH": OOS_SEED} | {c: ROOT / f"data/live/{c.lower()}_dirshort_oos_seed.parquet" for c in TABLE}   # 연구 시험 창 고른 팔 5시드 평균


# ── 피쳐·표적 (연구와 같은 식) ──────────────────────────────────────────────
def book_feats(E: pd.DataFrame) -> pd.DataFrame:
    """liqlab 꼴 d±k 명목·d_age(행 τ = 봉 τ 끝 1초 전 스냅숏) → 10열. 연구 feats()["book"] 그대로(행 기준 shift·rolling)."""
    b = {k: E[f"d-{k}"] for k in (1, 2, 5)}; a = {k: E[f"d{k}"] for k in (1, 2, 5)}
    imb = {k: (b[k] - a[k]) / (b[k] + a[k]) for k in (1, 2, 5)}; ld1 = np.log(b[1] + a[1])
    return pd.DataFrame(dict(
        **{f"dimb{k}": imb[k] for k in (1, 2, 5)}, **{f"dimb{k}_d1h": imb[k] - imb[k].shift(12) for k in (1, 2, 5)},
        depth1_z=ld1 - ld1.rolling(288, min_periods=144).mean(), bid_near=b[1] / b[5], ask_near=a[1] / a[5], d_age=E.d_age)).astype(np.float32)


def dataset(frames: dict, book: pd.DataFrame | None, dc: pd.Series | None = None):
    """→ t(결정 시각 = 봉 마감, 2024-01-01~ 끊김 없는 5분) · X(N×75 float32, book=None 이면 65열) · ret(봉 t 수익) · sig24. 연구 프레임(v17 build)과 같은 정의."""
    e = frames["ETH"].drop_duplicates("timestamp").set_index("timestamp").close.astype(float).sort_index()
    g = pd.date_range(e.index[0], e.index[-1], freq="5min")
    P = pd.Series(e.reindex(g).ffill().to_numpy(), index=g + pd.Timedelta("5min"))
    t = P.index[P.index >= START]
    ret = P.reindex(t + pd.Timedelta("5min")).to_numpy() / P.reindex(t).to_numpy() - 1
    sig = P.pct_change().rolling(288).std().reindex(t).to_numpy()
    if dc is None:   # 연구 diag_rl_dir_feat65 와 같은 일봉(그날 마지막 5분봉 종가)
        dc = frames["ETH"].set_index("timestamp").close.ffill().groupby(lambda x: x.normalize()).last()
    F = d4.build_features(frames, dc).set_index("ts").reindex(t - pd.Timedelta("5min"))[d4.FEATS]
    if book is None:   # 65열판 = 연구 research_xrp_dir_short_20261031.dataset65
        return t, F.to_numpy(np.float32), ret, sig
    B = book_feats(book).reindex(t - pd.Timedelta("5min"))
    return t, np.hstack([F.to_numpy(np.float32), B.to_numpy(np.float32)]), ret, sig


def fwd(ret, sig, h):
    """y_t = (ret_t + … + ret_{t+h−1}) / σ24_t — 연구 diag_rl_feature_ceiling.fwd(코인 하나)."""
    y = np.full(len(ret), np.nan); cs = np.r_[0.0, np.cumsum(ret)]; k = np.arange(len(ret) - h + 1)
    y[k] = (cs[k + h] - cs[k]) / np.where(sig[k] > 0, sig[k], np.nan)
    return y


def target(y, k):
    from scipy.stats import norm, rankdata
    if k == "rg":
        return norm.ppf((rankdata(y) - 0.5) / len(y))
    if k == "sign":
        return np.sign(y)
    return np.clip(y, -2, 2) if k == "clip2" else np.clip(y, -10, 10)


def train_rows(t_ns, ok, m0, h):
    """그 달 첫 봉 m0 전에 표적이 확정된 행: y_i 는 봉 i+h 종가에 확정 → i ≤ m0−h−2(한 봉 여유) · 2024-01-01 부터."""
    i = np.flatnonzero(ok[: max(m0 - h - 1, 0)]); return i[t_ns[i] >= START.value]


# ── 학습 ──────────────────────────────────────────────────────────────────
def book_day(d: str) -> pd.DataFrame | None:
    """bookDepth 일별 아카이브 → 그날 288봉 d±k·d_age(liqlab_data_20261028.build 규칙: 봉 끝 1초 전까지 마지막 스냅숏, 나이 +1초). 없으면 None.
    ponytail: 그날 스냅숏만 본다(liqlab 은 전날 것도 봄) — 그날 첫 스냅숏이 00:04:59 뒤인 날만 첫 봉이 NaN 으로 갈린다."""
    from build_binance_vision_panel_20260915 import BASE as VB, fetch
    s = fetch(f"{VB}/bookDepth/ETHUSDT/ETHUSDT-bookDepth-{d}.zip")
    if s is None:
        return None
    b = pd.read_csv(io.StringIO(s)); b["timestamp"] = pd.to_datetime(b["timestamp"])
    b = b[b["percentage"].abs().between(.5, 5.5)]
    w = b.pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="first").sort_index()
    w.columns = [f"d{int(c)}" for c in w.columns]
    bars = pd.date_range(d, periods=288, freq="5min")
    j = pd.merge_asof(pd.DataFrame({"cut": bars + pd.Timedelta("5min") - pd.Timedelta("1s")}),
                      w.reset_index().rename(columns={"timestamp": "snap"}), left_on="cut", right_on="snap", direction="backward")
    out = pd.DataFrame({c: j[c].to_numpy() for c in BCOLS[:-1]}, index=bars)
    out["d_age"] = (j.cut - j.snap).dt.total_seconds().to_numpy() + 1.0
    return out


def book_update(end_day: str) -> pd.DataFrame:
    B = pd.read_parquet(BOOK) if BOOK.exists() else pd.read_parquet(LIQ, columns=BCOLS)
    for d in pd.date_range(B.index.max().normalize() + pd.Timedelta("1D"), end_day, freq="D"):
        x = book_day(str(d.date()))
        if x is None:   # 아직 안 올라온 날 — 뒤 날을 건너뛰어 구멍을 만들지 않는다(학습 문이 멈춘다)
            break
        B = pd.concat([B, x]); print(f"  bookDepth +{d.date()}", flush=True)
    tmp = BOOK.with_suffix(".tmp"); B.to_parquet(tmp); tmp.replace(BOOK)
    return B


def train(frames: dict, book: pd.DataFrame | None, month: pd.Timestamp, dc: pd.Series | None = None, hz=tuple(HZ), nj: int = NJ) -> dict:
    import lightgbm as lgb
    cut = month - pd.Timedelta("5min")
    for sym, f in frames.items():   # 빠진 날을 앞채움 종가로 조용히 학습하지 않게(4h train 과 같은 문)
        g = pd.concat([pd.Series([START]), f.timestamp[(f.timestamp >= START) & (f.timestamp <= cut)], pd.Series([cut])]).diff().max()
        if g > pd.Timedelta("1h"):
            raise ValueError(f"{sym} 학습 창 구멍 {g} -- vision_update 를 다시 돌린다")
    if book is not None and book.index.max() < cut - pd.Timedelta("1h"):
        raise ValueError(f"bookDepth 가 {book.index.max()} 까지뿐 -- {cut} 까지 있어야 한다(아카이브 하루 늦게 공개)")
    t, X, ret, sig = dataset(frames, book, dc); t_ns = t.asi8
    m0 = int(np.searchsorted(t_ns, month.value))
    if m0 >= len(t) or t_ns[m0] != month.value:
        raise ValueError(f"패널이 {month} 결정 시각(봉 {cut})까지 없다")
    valid = np.isfinite(ret) & (sig > 0); models, info = {}, {}
    for h in hz:
        hl = HZ[h]; y = fwd(ret, sig, h); tr = train_rows(t_ns, valid & np.isfinite(y), m0, h)
        w = None if hl is None else 0.5 ** ((t_ns[m0] - t_ns[tr]) / DAY / hl)
        models[h] = []
        for seed in SEEDS:
            ms = []
            for k in ENS:
                m = lgb.LGBMRegressor(**BASE, **({"objective": "huber", "alpha": 1.0} if k == "huber" else {}), n_jobs=nj, verbose=-1,
                                      random_state=seed).fit(X[tr], target(y[tr], k), sample_weight=w)
                ms.append((m, float(m.predict(X[tr[::50]]).std() + 1e-12)))   # 학습 행 예측 척도(검증 통계 안 씀)
            models[h].append(ms)
        info[NAMES[h]] = dict(n_train=int(len(tr)), first=str(t[tr[0]]), last=str(t[tr[-1]]), half_life_days=hl)
        print(f"  h{h} 학습 {len(tr):,}행 ~{t[tr[-1]]}", flush=True)
    return {"models": models, "month": str(month.date()), "train": info, "seeds": SEEDS, "base": BASE, "feats": d4.FEATS + (BOOKF if book is not None else []),
            "book_end": str(book.index.max()) if book is not None else None, "panel_end": str(frames["ETH"].timestamp.max())}


def train65(frames: dict, month: pd.Timestamp, src: str) -> dict:
    """BTC·SOL·XRP 용 65열판: 학습 코인 src 를 ETH 자리에 두고 TABLE 이 그 코인에 맡긴 지평만 학습."""
    hz = tuple(h for h in HZ if any(t[h] == src for t in TABLE.values()))
    return train(d4.frames_for(frames, src), None, month, hz=hz, nj=NJ65[src]) | {"src": src}


def art_paths(coin: str) -> list[Path]:
    return [ARTIFACT] if coin == "ETH" else [ART65[s] for s in dict.fromkeys(TABLE[coin].values())]


def load_art(coin: str) -> dict:
    """코인 → 서빙 모형. BTC·SOL·XRP = 지평마다 TABLE 의 학습 코인 65열판 모형을 모은 것."""
    import joblib
    if coin == "ETH":
        return joblib.load(ARTIFACT)
    A = {s: joblib.load(ART65[s]) for s in dict.fromkeys(TABLE[coin].values())}
    return {"coin": coin, "models": {h: A[s]["models"][h] for h, s in TABLE[coin].items()},
            "month": min(a["month"] for a in A.values()), "src": {NAMES[h]: s for h, s in TABLE[coin].items()}}


def predict(art: dict, X: np.ndarray) -> dict:
    """→ {h: 시드 5 평균(각 시드 = Σ 표적 예측/척도)}."""
    return {h: np.mean([sum(m.predict(X) / s for m, s in ms) for ms in seeds], 0) for h, seeds in art["models"].items()}


# ── 실시간 ────────────────────────────────────────────────────────────────
def book_row(con, bar_ms: int) -> tuple[np.ndarray | None, str]:
    """워커 봉(t = 봉 시작 ms) 10열 → (값, 상태). warm·결측·빈 열 = 학습 분포 밖이라 None(예측을 비운다)."""
    r = con.execute(f"SELECT warm, {', '.join(BOOKF)} FROM bars WHERE t = ?", (bar_ms,)).fetchone()
    if r is None:
        return None, "missing"
    if r[0]:
        return None, "warm"
    if any(v is None for v in r[1:]):
        return None, "nan"
    return np.array(r[1:], np.float32), "ok"


_OOS: dict = {}


def oos_sorted(coin: str = "ETH") -> dict:
    """{"15m": 정렬된 표본 밖 예측, …}. ponytail: 씨앗 고정(4h 처럼 매달 지난 판 예측을 이어 붙이지 않음) — 달마다 예측 폭이 바뀌면 분위가 어긋난다, 그때 이어 붙이기."""
    if coin not in _OOS:
        S = pd.read_parquet(OOS_SEEDS[coin]); _OOS[coin] = {nm: np.sort(S[f"p{nm}"].to_numpy(float)) for nm in NAMES.values()}
    return _OOS[coin]


def live_payload(art: dict | None = None, n_hist: int = 13) -> dict:
    """지금 봉 예측 + 표본 밖 분위(pct 0~100)·문턱(20/80 분위 밖 = signal ±1, 안 = 0) + 최근 n_hist 봉 이력(호가 깊이 온전한 봉만)."""
    import joblib
    art = art or joblib.load(ARTIFACT)
    coin = art.get("coin", "ETH")
    now_ms = int(time.time() * 1000)
    F = d4.build_features(d4.frames_for(d4.rest_frames(now_ms, coin=coin), coin), d4.daily_closes(now_ms, coin)).tail(n_hist)
    return payload_from_features(art, F)


def payload_from_features(art: dict, F: pd.DataFrame) -> dict:
    """피쳐 프레임(최근 n_hist 봉) → 화면 응답. 서빙·파리티·렌더 시험이 같은 함수."""
    coin = art.get("coin", "ETH")
    bars = [int(pd.Timestamp(x).value // 10**6) for x in F.ts]
    if coin == "ETH":
        con = sqlite3.connect(f"file:{BOOK_DB}?mode=ro", uri=True)
        try:
            rows = [book_row(con, b) for b in bars]
        finally:
            con.close()
    else:   # 65열판 — 호가 깊이 안 씀
        rows = [(np.zeros(0, np.float32), "n/a")] * len(bars)
    ok = [i for i, (bk, _) in enumerate(rows) if bk is not None]
    P = predict(art, np.hstack([F.iloc[ok][d4.FEATS].to_numpy(np.float32), np.vstack([rows[i][0] for i in ok])])) if ok else {}
    Q = oos_sorted(coin); pct = lambda nm, x: round(100 * float(np.searchsorted(Q[nm], x)) / len(Q[nm]), 1)
    thr = {nm: [round(float(np.quantile(q, .2)), 4), round(float(np.quantile(q, .8)), 4)] for nm, q in Q.items()}
    out = {"ok": True, "asset": coin.lower(), "src": art.get("src"), "decided_at_ms": bars[-1] + 300_000, "model_month": art["month"], "book": rows[-1][1],
           "nan_feats": [c for c in d4.FEATS if not np.isfinite(F.iloc[-1][c])], "thr": thr, "pred": None,
           "hist": [[bars[i] + 300_000] + [pct(NAMES[h], float(P[h][j])) for h in HZ] for j, i in enumerate(ok)]}   # [결정 ms, 15분 분위, 1시간 분위]
    if ok and ok[-1] == len(bars) - 1:   # 지금 봉의 호가 깊이가 온전할 때만(warm·결측 = 비움)
        out["pred"] = {}
        for h, nm in NAMES.items():
            x = float(P[h][-1]); lo, hi = thr[nm]
            out["pred"][nm] = {"score": round(x, 4), "pct": pct(nm, x), "signal": 1 if x >= hi else -1 if x <= lo else 0}
    return out


# ── 점검 ──────────────────────────────────────────────────────────────────
def parity(panel_dir: str, liq: str, diag: str) -> None:
    """(서버) 연구 전진 접힘의 10월 모형(학습 경계 2026-10-01)을 이 코드로 다시 학습 → 10-01~10-23 예측을 연구 f65f_s*.preds.npz 와 대조.
    + bookDepth 이어 받기 경로(book_day)가 liqlab 행과 같은지 10-22·10-23 으로 대조."""
    frames = {s: pd.read_parquet(Path(panel_dir) / f"{s}USDT.parquet") for s in ["ETH"] + d4.COINS}
    book, month = pd.read_parquet(liq, columns=BCOLS), pd.Timestamp("2026-10-01")
    t0 = time.time(); art = train(frames, book, month); t, X, _, _ = dataset(frames, book)
    rep, mine, ref = {"train_sec": round(time.time() - t0)}, {3: [], 12: []}, {3: [], 12: []}
    for i, seed in enumerate(SEEDS):
        Z = np.load(Path(diag) / f"f65f_s{seed}.preds.npz"); va = Z["va_W"]   # 연구 프레임 행 번호 — 같은 격자(2024-01-01~ 끊김 없음)
        assert t[va[0]] == pd.Timestamp("2026-09-15") and t[va[-1]] == pd.Timestamp("2026-10-23 23:55"), "연구 프레임과 격자 불일치"
        sel = va[t[va] >= month]; s = t[va] >= month
        for h, arm in ((3, "ens_hl180"), (12, "ens_monthly")):
            a = sum(m.predict(X[sel]) / sc for m, sc in art["models"][h][i]); b = Z[f"FW_h{h}_{arm}"][s]
            mine[h].append(a); ref[h].append(b); rep[f"s{seed}/{NAMES[h]}"] = float(np.abs(a - b).max())
    for h in HZ:
        a, b = np.mean(mine[h], 0), np.mean(ref[h], 0)
        rep[f"mean5/{NAMES[h]}"] = dict(rows=len(a), maxabs=float(np.abs(a - b).max()), corr=float(np.corrcoef(a, b)[0, 1]), pred_std=float(b.std()))
    L = book.loc["2026-10-22":"2026-10-23 23:55"]; V = pd.concat([book_day("2026-10-22"), book_day("2026-10-23")]).reindex(L.index)
    rep["book_day_vs_liqlab"] = {c: float(np.nanmax(np.abs(V[c].to_numpy() - L[c].to_numpy()))) for c in BCOLS} | {"nan_diff": int((V.isna() != L.isna()).to_numpy().sum())}
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    # 10-28 첫 대조: 8/10 칸 ≤4e-15 · 15분 시드 1170305695 7.7e-5·2083046697 3e-8(예측 표준편차 4.8). 연구 스크립트를 그대로 다시 돌려도 저장본과
    #   같은 7.677e-5 차(09-15 구간 3.1e-4) — 저장 예측이 지금 재현 안 되는 실행이고, 이 코드 = 지금 연구 코드. X 75열·가중치는 셀 단위로 같음 → 문턱 1e-4.
    worst = max(v for k, v in rep.items() if k.startswith("s"))
    assert worst < 1e-4 and rep["book_day_vs_liqlab"]["nan_diff"] == 0 and max(v for k, v in rep["book_day_vs_liqlab"].items() if k != "nan_diff") < 1e-6, "패리티 실패"
    print(f"parity ok — 시드별 최대 차 {worst:.2e} · bookDepth 이어 받기 = liqlab")


def selftest() -> None:
    t_ns = START.value + np.arange(20) * DAY; ok = np.ones(20, bool)
    assert train_rows(t_ns, ok, 10, 3).tolist() == [0, 1, 2, 3, 4, 5]                          # i ≤ m0−h−2
    assert train_rows(t_ns - 5 * DAY, ok, 10, 0).tolist() == [5, 6, 7, 8]                       # 2024-01-01 전 행 제외
    y = fwd(np.array([1., 2, 3, 4]), np.array([1., 2, 0, 1]), 2)
    assert y[0] == 3 and y[1] == 2.5 and np.isnan(y[2]) and np.isnan(y[3])                       # σ=0 · 끝 넘음 = NaN
    assert target(np.array([5., -3]), "clip2").tolist() == [2, -2] and target(np.array([12.]), "huber").tolist() == [10]
    con = sqlite3.connect(":memory:"); con.execute(f"CREATE TABLE bars(t INTEGER PRIMARY KEY, warm REAL, {', '.join(c + ' REAL' for c in BOOKF)})")
    con.execute("INSERT INTO bars VALUES (1, 0, " + ", ".join(["0.1"] * 10) + ")")
    con.execute("INSERT INTO bars VALUES (2, 1, " + ", ".join(["NULL"] * 9) + ", 29)")
    con.execute("INSERT INTO bars VALUES (3, 0, NULL, " + ", ".join(["0.1"] * 9) + ")")
    assert book_row(con, 1)[1] == "ok" and [book_row(con, k) for k in (2, 3, 4)] == [(None, "warm"), (None, "nan"), (None, "missing")]
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    coin = a[a.index("--coin") + 1].upper() if "--coin" in a else "ETH"
    if "--selftest" in a:
        selftest()
    elif "--train" in a:
        import joblib
        if not PANEL.exists():
            raise SystemExit(f"{PANEL} 없음 -- 연구 panel_fwd(2022-01-01~) 를 복사해 씨앗으로 둔다(없으면 2022~ 전부를 받는다)")
        month = pd.Timestamp(a[a.index("--month") + 1]) if "--month" in a else pd.Timestamp.utcnow().tz_localize(None).normalize().replace(day=1)
        end = str((pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta("1D")).date())
        frames = d4.vision_update(end, days=(pd.Timestamp(end) - PANEL_START).days, panel=PANEL)
        art = train(frames, book_update(end), month)
        tmp = ARTIFACT.with_suffix(".tmp"); joblib.dump(art, tmp)   # 다 쓴 뒤 교체 · 지난 판은 교체 시각 이름으로 남긴다(4h 와 같은 규칙)
        if ARTIFACT.exists():
            ARTIFACT.replace(ARTIFACT.with_suffix(f".{pd.Timestamp.utcnow():%Y%m%dT%H%M}.joblib"))
        tmp.replace(ARTIFACT)
        print("학습", {k: v for k, v in art.items() if k != "models"})
    elif "--train65" in a:
        import joblib
        month = pd.Timestamp(a[a.index("--month") + 1]) if "--month" in a else pd.Timestamp.utcnow().tz_localize(None).normalize().replace(day=1)
        end = str((pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta("1D")).date())
        frames = d4.vision_update(end, days=(pd.Timestamp(end) - PANEL_START).days, panel=PANEL)   # ETH --train 이 먼저 돌면 이어 받기 없음
        for src, path in ART65.items():
            art = train65(frames, month, src)
            tmp = path.with_suffix(".tmp"); joblib.dump(art, tmp)
            if path.exists():
                path.replace(path.with_suffix(f".{pd.Timestamp.utcnow():%Y%m%dT%H%M}.joblib"))
            tmp.replace(path)
            print("학습65", src, {k: v for k, v in art.items() if k != "models"})
    elif "--parity" in a:
        parity(*a[a.index("--parity") + 1:][:3])
    else:
        print(json.dumps(live_payload(load_art(coin)), ensure_ascii=False))
