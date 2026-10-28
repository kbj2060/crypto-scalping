"""liqlab 판정 실행기 — 사전등록 docs/prereg/liqlab_liquidity_arms_20261028.md 의 7팔. 2026-10-28.

  python liqlab_run_20261028.py --controls   # 실데이터 위 양성·음성 대조(판정 전 장치 검증)
  python liqlab_run_20261028.py --run        # 학습(24·25) 임계 → 검증(26-01~05) 변형 선택 → 시험(26-06~10-23) 1회
  python liqlab_run_20261028.py --incr       # §13-2 예측 입력 → 배포 QX4·A1 박스 1h 위 증분(검증 적합 · 시험 1회)
  python liqlab_run_20261028.py --forward [--dry]   # §13-1 T1 전진 판정(2027-04-24~ · LIQLAB_D=data/liqlab_fwd) · --dry = 시험창 재현
창: TRAIN 2024-01-01~2025-12-31 · VAL 2026-01-01~2026-05-31 · TEST 2026-06-01~2026-10-23.
"""
from __future__ import annotations

import itertools
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import liqlab_harness_20261028 as L  # noqa: E402

D = pathlib.Path(os.environ.get("LIQLAB_D", "data/liqlab")); OUT = pathlib.Path("tmp/liqlab_20261028"); OUT.mkdir(parents=True, exist_ok=True)
COINS = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "XRPUSDT"]
WIN = {"train": ("2024-01-01", "2025-12-31 23:59"), "val": ("2026-01-01", "2026-05-31 23:59"), "test": ("2026-06-01", "2026-10-23 23:59"),
       "fwd": ("2026-10-24", "2027-04-23 23:59")}   # §13-1 전진 창
EXPECTED_BP = {"T1": 5.0, "T2": 2.0, "T3": 3.0}   # 사전등록 기대 효과(MDE 판정 불능 기준)
T4_STRESS_BP = 2.0                                # 일봉 리밸런스 메이커 + 역선택 스트레스(다리당)


# ───────────────────────────── 피쳐(행 τ = 봉 τ 끝에 앎) ─────────────────────────────
def panel(sym):
    return feats(pd.read_parquet(D / f"{sym}_5m.parquet"))


def feats(p):
    lc = np.log(p.c)
    v2 = p.rv ** 2
    f = pd.DataFrame(index=p.index)
    f["r12"] = lc - lc.shift(12)
    f["sig1h"] = np.sqrt(v2.rolling(288).sum() / 24)
    f["z"] = f.r12 / f.sig1h
    f["rv24"] = np.sqrt(v2.rolling(288).sum())
    f["amihud24"] = p.ab.rolling(288).sum() / p.qv.rolling(288).sum() * 1e9
    f["imb_book"] = ((p["d-1"] - p["d1"]) / (p["d-1"] + p["d1"])).where(p.d_age <= 120)
    f["hour"] = p.index.hour
    for k, w in (("1h", 12), ("24h", 288)):                      # 미시구조 측정치(Easley 2024 계열)
        f[f"lrv_{k}"] = np.log(v2.rolling(w).sum() + 1e-12)
        f[f"lamihud_{k}"] = np.log(p.ab.rolling(w).sum() / p.qv.rolling(w).sum() + 1e-18)
        flow = (2 * p.tbqv - p.qv).rolling(w).sum().abs()
        f[f"lkyle_{k}"] = np.log((lc - lc.shift(w)).abs() / (flow + 1) + 1e-15)
        f[f"roll_{k}"] = np.sqrt(np.maximum(0, -p.rr.rolling(w).sum() / p.k.rolling(w).sum())) * 1e4
        f[f"vpin_{k}"] = p.im_s.abs().rolling(w).mean()
        s, s2, k_ = p.im_s.rolling(w).sum(), p.im_s2.rolling(w).sum(), p.k.rolling(w).sum()
        f[f"limbvar_{k}"] = np.log((s2 - s * s / k_) / k_ + 1e-9)
    f["lrv_7d"] = np.log(v2.rolling(2016).sum() + 1e-12)
    f["fr"] = p.fr
    return p, f


def win(idx, name):
    a, b = WIN[name]
    return (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))


# ───────────────────────────── 매매 팔(신호만) ─────────────────────────────
def arm_T1(p, f, tr, state, q, H):
    """Nagel 2012 상태 조건부 단기 반전: |1h 수익 z| ≥ 학습 분위 q & 상태 상위 3분위 → 반대 방향."""
    fire = f.z.abs() >= np.nanquantile(f.z[tr].abs(), q)
    if state != "none":
        col = {"rv": "rv24", "amihud": "amihud24"}[state]
        fire &= f[col] >= np.nanquantile(f[col][tr], 2 / 3)
    return (-np.sign(f.z) * fire).fillna(0).astype(int).to_numpy(), H


def arm_T2(p, f, tr, dirn, q, H):
    """Albers 2025 «Reversals» 메이커: ±1% 호가 불균형 극단에서 불균형 반대(against)/같은(follow) 쪽 지정가."""
    hi, lo = np.nanquantile(f.imb_book[tr], q), np.nanquantile(f.imb_book[tr], 1 - q)
    s = np.where(f.imb_book >= hi, 1, np.where(f.imb_book <= lo, -1, 0))
    return (s if dirn == "follow" else -s).astype(int), H


def arm_T3(p, f, tr, cond):
    """Shen·Urquhart·Wang 2022 일중 TSMOM: UTC 00:00~00:30 수익 부호 → 23:30~24:00 보유. 결정 = 23:25 봉 끝."""
    lc = np.log(p.c); day = p.index.normalize()
    at030 = lc.where((p.index.hour == 0) & (p.index.minute == 25)).groupby(day).transform("max")   # 00:25 봉 종가 = 00:30
    prev = lc.shift(1).where((p.index.hour == 0) & (p.index.minute == 0)).groupby(day).transform("max")  # 전날 23:55 종가
    t = (p.index.hour == 23) & (p.index.minute == 25)
    keep = pd.Series(t, index=p.index)
    if cond != "none":
        cum = (p.qv if cond == "highvol" else p.rv ** 2).groupby(day).cumsum()
        keep &= cum >= np.nanmedian(cum[t & tr])
    return (np.sign(at030 - prev) * keep).fillna(0).astype(int).to_numpy(), 6


# ───────────── 내부 기검증 팔(I1~I8) — 원 스크립트 버그 고쳐 이식(사전등록 §11) ─────────────
def _rq(s, q, w=288):
    """직전 w 봉(현재 제외) 굴림 분위 — 인과 문턱(원본들의 «창 전체 rank» 미래참조 수정)."""
    return s.shift(1).rolling(w, min_periods=w * 2 // 3).quantile(q)


def arm_I1(p, f, tr, H):
    """깊은 벽 근사(±1% — 원본 ±50bp 1초 상태는 이식 불가): 불균형 직전 1일 80/20분위 밖 → 벽 쪽(따라감)."""
    x = f.imb_book
    return np.where(x >= _rq(x, .8), 1, np.where(x <= _rq(x, .2), -1, 0)).astype(int), H


def arm_I2(p, f, tr, dirn, H):
    """5분 테이커 불균형(거래대금 가중) — 인과 중앙값 뺀 뒤 |·| 직전 1일 90분위 이상 → 따라감/거스름. 원본 종가 진입 → o[τ+1]."""
    tfi = 2 * p.tbqv / p.qv.replace(0, np.nan) - 1
    d = tfi - tfi.shift(1).rolling(288, min_periods=192).median()
    s = (np.sign(d) * (d.abs() >= _rq(d.abs(), .9))).fillna(0)
    return (s if dirn == "follow" else -s).astype(int).to_numpy(), H


def arm_I3(p, f, tr, H):
    """CVD 다이버전스(레포트 S2b): 직전 48봉 신고가인데 CVD 는 신고점 아님 → 숏(거울 롱)."""
    cvd = (2 * p.tbqv - p.qv).cumsum()
    sh = (p.h > p.h.shift(1).rolling(48).max()) & (cvd < cvd.shift(1).rolling(48).max())
    lg = (p.l < p.l.shift(1).rolling(48).min()) & (cvd > cvd.shift(1).rolling(48).min())
    return (lg.astype(int) - sh.astype(int)).to_numpy(), H


def arm_I4(p, f, tr, kind, H):
    """Campbell·Grossman·Wang: 1h 급변 + 1h 거래대금이 30일 중앙 대비 높으면 거스름 / 낮으면 따라감(문턱 학습창 고정)."""
    qv1 = p.qv.rolling(12).sum(); ratio = qv1 / qv1.shift(1).rolling(8640, min_periods=2000).median()
    zq = np.nanquantile(f.z[tr].abs(), .8)
    if kind == "fade_high":
        fire = (f.z.abs() >= zq) & (ratio >= np.nanquantile(ratio[tr], .8)); sgn = -1
    else:
        fire = (f.z.abs() >= zq) & (ratio <= np.nanquantile(ratio[tr], .2)); sgn = 1
    return (sgn * np.sign(f.z) * fire).fillna(0).astype(int).to_numpy(), H


def arm_I5(p, f, tr, H):
    """극단 펀딩 거스름: 정산 봉(UTC 0·8·16시 00분 봉 — 그 정산 포함)에서 직전 180일 정산값 중위 순위(동점 ½, 자신 제외) ≥.9 숏 / ≤.1 롱."""
    ev = np.flatnonzero((p.index.minute == 0) & np.isin(p.index.hour, [0, 8, 16]))
    v = p.fr.to_numpy()[ev]; s = np.zeros(len(p), int)
    for k in range(len(ev)):
        w = v[max(0, k - 540):k]
        if len(w) >= 180 and np.isfinite(v[k]):
            rk = ((w < v[k]).sum() + .5 * (w == v[k]).sum()) / len(w)
            s[ev[k]] = -1 if rk >= .9 else (1 if rk <= .1 else 0)
    return s, H


def arm_I6(p, f, tr, cond):
    """15분 되돌림(2608.21888): UTC 15분 블록 끝 봉에서 |15분 수익| ≥ 학습 95분위 → 반대, 다음 15분 보유. 원본 전 표본 문턱 → 학습창 고정."""
    lc = np.log(p.c); end = pd.Series(p.index.minute % 15 == 10, index=p.index)
    r15 = lc - lc.shift(3)
    fire = end & (r15.abs() >= np.nanquantile(r15[end & tr].abs(), .95))
    if cond == "taker":
        im15 = 2 * p.tbqv.rolling(3).sum() / p.qv.rolling(3).sum() - 1
        fire &= im15.abs() >= np.nanquantile(im15[end & tr].abs(), .8)
    return (-np.sign(r15) * fire).fillna(0).astype(int).to_numpy(), 3


def arm_I7(p, f, tr, dirn):
    """소진형 대량 델타(09-30 사전등록 E1~E4 그대로·직전 288봉 인과) → 4h 거스름(fade) / 지속(follow)."""
    lc = np.log(p.c); dl = 2 * p.tbqv - p.qv; ad = dl.abs(); rng = np.log(p.h / p.l); r1h = (lc - lc.shift(12)).shift(1)
    e = (ad >= 20 * ad.shift(1).rolling(288, min_periods=192).median()) & (rng >= _rq(rng, .9))
    up = e & (dl > 0) & (p.h >= p.h.shift(1).rolling(288).max()) & (r1h > 0)
    dn = e & (dl < 0) & (p.l <= p.l.shift(1).rolling(288).min()) & (r1h < 0)
    s = up.astype(int) - dn.astype(int)
    return (s if dirn == "follow" else -s).to_numpy(), 48


def arm_I8(p, f, tr, H):
    """ICT 유동성 스윕: 저가가 직전 48봉 최저를 깨고 종가는 그 위 → 롱(거울 숏)."""
    lo, hi = p.l.shift(1).rolling(48).min(), p.h.shift(1).rolling(48).max()
    lg = (p.l < lo) & (p.c > lo); sh = (p.h > hi) & (p.c < hi)
    return (lg.astype(int) - sh.astype(int)).to_numpy(), H


I_GRID = {"I1": [dict(H=H) for H in (1, 3)],
          "I2": [dict(dirn=d, H=H) for d, H in itertools.product(["follow", "fade"], [1, 3, 12])],
          "I3": [dict(H=H) for H in (12, 24)],
          "I4": [dict(kind=k, H=H) for k, H in itertools.product(["fade_high", "follow_low"], [12, 48])],
          "I5": [dict(H=H) for H in (12, 96, 288)],
          "I6": [dict(cond=c) for c in ("none", "taker")],
          "I7": [dict(dirn=d) for d in ("fade", "follow")],
          "I8": [dict(H=H) for H in (12, 48)]}
I_EXPECTED_BP = {"I1": 2.0, "I2": 2.0, "I3": 3.0, "I4": 3.0, "I5": 10.0, "I6": 3.0, "I7": 15.0, "I8": 3.0}


T_GRID = {"T1": [dict(state=s, q=q, H=H) for s, q, H in itertools.product(["rv", "amihud", "none"], [.90, .95], [12, 48])],
          "T2": [dict(dirn=d, q=q, H=H) for d, q, H in itertools.product(["against", "follow"], [.95, .99], [1, 6])],
          "T3": [dict(cond=c) for c in ["none", "highvol", "highrv"]]}
ARM_FN = {"T1": arm_T1, "T2": arm_T2, "T3": arm_T3, "I1": arm_I1, "I2": arm_I2, "I3": arm_I3, "I4": arm_I4,
          "I5": arm_I5, "I6": arm_I6, "I7": arm_I7, "I8": arm_I8}
SELECTABLE = {"T1": lambda v: v["state"] != "none", "T2": lambda v: True, "T3": lambda v: True}   # T1 «none» 은 대조군


def score_arm(arm, data, variant, windows=("val", "test"), B_null=300, null_modes=("maker",)):
    out = {}
    for sym, (p, f) in data.items():
        tr = win(p.index, "train")
        sig, H = ARM_FN[arm](p, f, tr, **variant)
        rvq = np.nanquantile(f.rv24[tr], [1 / 3, 2 / 3])
        for w in windows:
            m = win(p.index, w); px = {k: p[k].to_numpy()[m] for k in "ohlc"}
            day = ((p.index[m] - p.index[m][0]).days).to_numpy()   # 달력일 군집(코인 공통 창 시작) -- 10-28 검수: 코인·일 군집은 같은 날 코인 간 상관을 무시해 CI 가 좁았다
            st = L.strata_of(f.hour.to_numpy()[m], f.rv24.to_numpy()[m], rvq)
            out.setdefault(w, {})[sym] = L.evaluate_window(sig[m], px, day, st, H, B_null=B_null, null_modes=null_modes)
    return out


def pooled(res_w, mode="maker"):
    net = np.concatenate([np.asarray(v[mode]["net"], float) for v in res_w.values()])
    day = np.concatenate([np.asarray(v[mode]["day"], int) for v in res_w.values()])
    if len(net) < 10:
        return {"n": int(len(net))}
    d = L.cluster_boot_mean(net, day)
    dd = pd.Series(net).groupby(day).sum()
    return {"n": int(len(net)), "mean": float(net.mean()), "ci95": L.ci95(d), "p": L.p_two_sided(d),
            "daily_sr": float(dd.mean() / dd.std()) if dd.std() > 0 else 0.0, "daily": dd.tolist()}


# ───────────────────────────── T4 횡단면(20코인 일봉) ─────────────────────────────
def t4_returns(cond, cost_bp):
    """Bianchi·Babiak·Dickerson 2022: 거래량 조건부 1일 반전 — 패자 롱·승자 숏(동일가중 1/3씩), d 종가 → d+1 종가."""
    syms = [x.stem[3:] for x in sorted(D.glob("D1_*.parquet"))]
    c = pd.DataFrame({s: pd.read_parquet(D / f"D1_{s}.parquet").c for s in syms})
    qv = pd.DataFrame({s: pd.read_parquet(D / f"D1_{s}.parquet").qv for s in syms})
    r = np.log(c).diff(); abn = qv / qv.shift(1).rolling(30).mean()
    nxt = r.shift(-1)
    W = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    for d in r.index:
        ok = r.loc[d].notna() & abn.loc[d].notna() & nxt.loc[d].notna()
        if cond != "none":
            med = abn.loc[d][ok].median()
            ok &= (abn.loc[d] <= med) if cond == "lowvol" else (abn.loc[d] > med)
        x = r.loc[d][ok]
        if len(x) < 6:
            continue
        k = len(x) // 3
        W.loc[d, x.nsmallest(k).index] = 1 / k; W.loc[d, x.nlargest(k).index] = -1 / k
    gross = (W * nxt).sum(1) * 1e4
    turn = (W - W.shift(1).fillna(0)).abs().sum(1)
    return gross - turn * cost_bp, W, nxt


def t4_null(W, nxt, B=500, seed=17):
    """같은 날 같은 가중 묶음을 코인 사이에서 무작위 재배치(건수·노출 보존)."""
    rng = np.random.default_rng(seed); Wv, Nv = W.to_numpy(), np.nan_to_num(nxt.to_numpy())
    out = np.empty(B)
    for b in range(B):
        Wp = np.array([rng.permutation(row) for row in Wv])
        out[b] = (Wp * Nv).sum(1).mean() * 1e4
    return out


# ───────────────────────────── 예측 팔 ─────────────────────────────
BASE_X = ["lrv_1h", "lrv_24h", "lrv_7d"]
P_ADD = {"P1": ["lamihud_1h", "lamihud_24h", "lkyle_1h", "lkyle_24h", "roll_1h", "roll_24h", "vpin_1h", "vpin_24h", "x_lrv_1h", "x_lrv_24h"],
         "P2": ["limbvar_1h", "limbvar_24h", "imbvar_ratio"],
         "P3": ["fr_last", "fr_7d", "fr_z30"]}


def pred_frame(sym, data):
    p, f = data[sym]
    x = f.copy()
    ref = "ETHUSDT" if sym == "BTCUSDT" else "BTCUSDT"          # 교차시장(Easley: BTC↔ETH)
    x["x_lrv_1h"], x["x_lrv_24h"] = data[ref][1].lrv_1h.reindex(x.index), data[ref][1].lrv_24h.reindex(x.index)
    x["imbvar_ratio"] = x.limbvar_1h - x.limbvar_24h
    x["fr_last"] = x.fr
    x["fr_7d"] = x.fr.rolling(2016, min_periods=288).mean()
    x["fr_z30"] = (x.fr - x.fr.rolling(8640, min_periods=2000).mean()) / x.fr.rolling(8640, min_periods=2000).std()
    v2 = p.rv ** 2; lc = np.log(p.c)
    fwd = lambda s, w: s[::-1].rolling(w).sum()[::-1].shift(-1)                       # noqa: E731  τ+1..τ+w
    x["y_P1_1h"] = np.log(fwd(v2, 12) + 1e-12)
    x["y_P1_4h"] = np.log(fwd(v2, 48) + 1e-12)
    x["y_P2"] = ((lc.shift(-12) - lc).abs() > 2.5 * f.sig1h).astype(float).where(lc.shift(-12).notna())
    lowmin = np.log(p.l[::-1].rolling(288).min()[::-1].shift(-1)) - lc                # 다음 24h 최저가
    x["y_P3"] = (lowmin <= -2.5 * f.rv24).astype(float).where(lowmin.notna())
    x["is_grid"] = x.index.minute == 55                                               # 1시간 격자
    x["is_day"] = (x.index.hour == 23) & (x.index.minute == 55)                       # 일 격자
    return x


def qlike(y_log_true, y_log_pred, s2):
    vt, vp = np.exp(y_log_true), np.exp(y_log_pred + s2 / 2)
    return vt / vp - np.log(vt / vp) - 1


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score_pred(arm, data, target):
    out = {}
    for sym in data:
        x = pred_frame(sym, data)
        g = x[x.is_day] if arm == "P3" else x[x.is_grid]
        cols0 = BASE_X + ([] if arm == "P1" else ["r12"])
        cols1 = cols0 + P_ADD[arm]
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=cols1 + [target])
        tr, res = win(g.index, "train"), {}
        for w in ("val", "test"):
            m = win(g.index, w); day = g.index[m].normalize()
            losses = {}
            for nm, cols in (("base", cols0), ("add", cols1)):
                mu, sd = g.loc[tr, cols].mean(), g.loc[tr, cols].std() + 1e-12
                Xtr, Xw = (g.loc[tr, cols] - mu) / sd, (g.loc[m, cols] - mu) / sd
                if arm == "P1":
                    mdl = LinearRegression().fit(Xtr, g.loc[tr, target]); s2 = np.var(g.loc[tr, target] - mdl.predict(Xtr))
                    losses[nm] = qlike(g.loc[m, target].to_numpy(), mdl.predict(Xw), s2)
                else:
                    mdl = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, g.loc[tr, target])
                    losses[nm] = logloss(g.loc[m, target].to_numpy(), mdl.predict_proba(Xw)[:, 1])
            dl = pd.Series(losses["add"] - losses["base"], index=day).groupby(level=0).mean()
            bt = L.cluster_boot_mean(dl.to_numpy(), np.arange(len(dl)) // 5)            # 5일 블록
            res[w] = {"n": int(m.sum()), "days": len(dl), "event_rate": None if arm == "P1" else float(g.loc[m, target].mean()),
                      "d_loss": float(dl.mean()), "ci90_one_sided_upper": float(np.nanquantile(bt, .90)),
                      "base_loss": float(np.mean(losses["base"])), "rel_pct": float(dl.mean() / np.mean(losses["base"]) * 100)}
        out[sym] = res
    return out


def pred_gate(res):
    """되돌리기 쉬운 예측 개선 = 단측 90%: 시험 ≥3/4 코인 개선 · 검증 ≥3/4 개선 · 단측 90% 상한<0 코인 ≥2."""
    t = [v["test"] for v in res.values()]; vv = [v["val"] for v in res.values()]
    g = {"test_upper90<0": sum(x["ci90_one_sided_upper"] < 0 for x in t),
         "test_mean<0": sum(x["d_loss"] < 0 for x in t), "val_mean<0": sum(x["d_loss"] < 0 for x in vv)}
    ok = g["test_mean<0"] >= 3 and g["val_mean<0"] >= 3 and g["test_upper90<0"] >= 2
    return {"verdict": "통과(조건부)" if ok else "불통과", **g}


# ───────────────────────────── 실데이터 대조(장치 검증) ─────────────────────────────
def controls(data):
    p, f = data["ETHUSDT"]; tr = win(p.index, "train"); rep = {}
    m = win(p.index, "val"); px = {k: p[k].to_numpy()[m] for k in "ohlc"}
    day = ((p.index[m] - p.index[m][0]).days).to_numpy(); rvq = np.nanquantile(f.rv24[tr], [1 / 3, 2 / 3])
    st = L.strata_of(f.hour.to_numpy()[m], f.rv24.to_numpy()[m], rvq)
    lc = np.log(p.c.to_numpy()[m]); fwd = np.r_[lc[6:] - lc[:-6], np.zeros(6)]   # τ 종가 → τ+6 종가(진입 τ+1)
    rng = np.random.default_rng(0); fire = rng.random(m.sum()) < .005
    for acc in (.5, .6, .7):
        sg = np.zeros(m.sum(), int)
        sg[fire] = np.where(rng.random(fire.sum()) < acc, np.sign(fwd[fire]), -np.sign(fwd[fire])).astype(int)
        r = L.evaluate_window(sg, px, day, st, 6, B_null=200)
        rep[f"planted_{acc}"] = {k: {kk: r[k].get(kk) for kk in ("n", "mean_gross", "mean_net", "ci95", "null_random_pct", "null_circular_pct")} for k in r}
    fp = 0
    for k in range(30):
        g_ = np.random.default_rng(1000 + k); sg = np.where(g_.random(m.sum()) < .005, g_.choice([-1, 1], m.sum()), 0)
        tr_ = L.simulate(sg, *(px[c] for c in "ohlc"), H=6, mode="taker")
        fp += L.ci95(L.cluster_boot_mean(tr_["gross"], day[tr_["i"]], B=800))[0] > 0
    rep["negative_false_pos_of_30"] = fp
    r1 = np.log(p.c).diff().shift(-1)                       # 트립와이어: 다음 봉 수익과 |상관| > .2 면 미래참조 의심
    rep["tripwire_corr_next_bar"] = {c: float(abs(f[c].corr(r1))) for c in f.columns if c != "hour"}
    rep["tripwire_flag"] = [c for c, v in rep["tripwire_corr_next_bar"].items() if v > .2]
    return rep


def diag(data):
    """사후 진단(판정 불변): ① P2 증분이 시간대·거래량 그림자인가 ② T1 상태 조건의 시험창 리프트."""
    rep = {}
    for sym in data:
        x = pred_frame(sym, data); g = x[x.is_grid].copy()
        p = data[sym][0]
        g["lqv_1h"] = np.log(p.qv.rolling(12).sum()).reindex(g.index); g["lqv_24h"] = np.log(p.qv.rolling(288).sum()).reindex(g.index)
        for h in range(24):
            g[f"h{h}"] = (g.index.hour == h).astype(float)
        ctl = BASE_X + ["r12", "lqv_1h", "lqv_24h"] + [f"h{h}" for h in range(1, 24)]
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=ctl + P_ADD["P2"] + ["y_P2"])
        tr = win(g.index, "train"); out = {}
        for nm, cols in (("ctl", ctl), ("ctl+imb", ctl + P_ADD["P2"])):
            mu, sd = g.loc[tr, cols].mean(), g.loc[tr, cols].std() + 1e-12
            mdl = LogisticRegression(C=1.0, max_iter=3000).fit((g.loc[tr, cols] - mu) / sd, g.loc[tr, "y_P2"])
            out[nm] = mdl; out[nm + "_cols"] = (cols, mu, sd)
        res = {"coef_imb(std)": dict(zip(P_ADD["P2"], np.round(out["ctl+imb"].coef_[0][-3:], 4)))}
        for w in ("val", "test"):
            m = win(g.index, w); day = g.index[m].normalize(); ls = {}
            for nm in ("ctl", "ctl+imb"):
                cols, mu, sd = out[nm + "_cols"]
                ls[nm] = logloss(g.loc[m, "y_P2"].to_numpy(), out[nm].predict_proba((g.loc[m, cols] - mu) / sd)[:, 1])
            dl = pd.Series(ls["ctl+imb"] - ls["ctl"], index=day).groupby(level=0).mean()
            bt = L.cluster_boot_mean(dl.to_numpy(), np.arange(len(dl)) // 5)
            res[w] = {"d_loss": float(dl.mean()), "upper90": float(np.nanquantile(bt, .9)), "rel_pct": float(dl.mean() / ls["ctl"].mean() * 100)}
        rep.setdefault("P2_hour_volume_control", {})[sym] = res
    for v in (dict(state="rv", q=.95, H=12), dict(state="none", q=.95, H=12), dict(state="amihud", q=.95, H=12)):
        r = score_arm("T1", data, v, windows=("test",), B_null=50)
        pt = pooled(r["test"]); rep.setdefault("T1_test_by_state", {})[v["state"]] = {k: pt.get(k) for k in ("n", "mean", "ci95", "p")}
    return rep


def relaxed(data):
    """10-28 사용자 «수수료 0 · 조금이라도 개선·예측되면 통과 후보» — 사전등록 §10 완화 기준, 전 변형."""
    rows = relaxed_rows(data, T_GRID)
    t4 = []
    for cond in ("lowvol", "highvol", "none"):
        net, W, nxt = t4_returns(cond, 0.0)
        v, t = net[win(net.index, "val")], net[win(net.index, "test")]
        npct = float((t4_null(W.loc[t.index], nxt.loc[t.index], B=300) < t.mean()).mean() * 100)
        t4.append({"cond": cond, "val_mean_bp_day": float(v.mean()), "test_mean_bp_day": float(t.mean()), "null_pct": npct,
                   "candidate": bool(v.mean() > 0 and t.mean() > 0 and npct >= 50)})
    pred = {}
    for arm, target in (("P1", "y_P1_1h"), ("P1_4h", "y_P1_4h"), ("P2", "y_P2"), ("P3", "y_P3")):
        res = score_pred(arm[:2], data, target)
        tm = sum(v["test"]["d_loss"] < 0 for v in res.values()); vm = sum(v["val"]["d_loss"] < 0 for v in res.values())
        pred[arm] = {"test_improve": tm, "val_improve": vm, "candidate": tm >= 3 and vm >= 3,
                     "rel_pct_test": {s: round(v["test"]["rel_pct"], 3) for s, v in res.items()}}
    return {"trading": rows, "T4": t4, "pred": pred}


def relaxed_rows(data, grids):
    """비용 0 · 전 변형 · 메이커/테이커 — 완화 «통과 후보» 칸(사전등록 §10)."""
    L.MAKER_BP = L.MAKER_FALLBACK_BP = L.TAKER_BP = 0.0          # 관통 체결 규칙은 유지
    rows = []
    for arm, grid in grids.items():
        for v in grid:
            r = score_arm(arm, data, v, windows=("val", "test"), B_null=100, null_modes=("maker", "taker"))
            for mode in ("maker", "taker"):
                pv, pt = pooled(r["val"], mode), pooled(r["test"], mode)
                ok = [x[mode] for x in r["test"].values() if x[mode]["n"] >= 10]
                npct = float(np.mean([x["null_random_pct"] for x in ok])) if ok else 0.0
                cpct = float(np.mean([x["null_circular_pct"] for x in ok])) if ok else 0.0
                pos = sum(x["mean_net"] > 0 for x in ok)
                cand = (pv.get("mean", -1) > 0 and pt.get("mean", -1) > 0 and pos >= max(1, -(-len(ok) // 2)) and npct >= 50)
                rows.append({"arm": arm, "variant": v, "mode": mode, "val_mean": pv.get("mean"), "val_n": pv.get("n"),
                             "test_mean": pt.get("mean"), "test_n": pt.get("n"), "test_ci95": pt.get("ci95"),
                             "test_p_one_sided": (pt["p"] / 2 if pt.get("mean", -1) > 0 else 1 - pt["p"] / 2) if "p" in pt else None,
                             "coins_pos": f"{pos}/{len(ok)}", "per_coin": {s: x[mode].get("mean_net") for s, x in r["test"].items()},
                             "null_random_pct": npct, "null_circular_pct": cpct, "candidate": bool(cand)})
            print(arm, v, [(x["mode"], round(x["val_mean"] or 0, 2), round(x["test_mean"] or 0, 2), x["coins_pos"], round(x["null_random_pct"]), x["candidate"]) for x in rows[-2:]], flush=True)
    return rows


def internal(data):
    """내부 기검증 팔 I1~I8 — ①엄격(실제 비용·검증 선택·시험 1회·G1~G7·Holm 8팔·DSR 시행 23) ②완화(비용 0·전 변형)."""
    rep, trial_srs, keep, pvals = {}, [], {}, {}
    # 미래 불변: ETH 를 시각 cut 까지 자른 데이터로 다시 계산한 신호[cut] == 전체 신호[cut] (전 팔·전 변형, 시험창 안 cut 5개)
    p_full, f_full = data["ETHUSDT"]; tr_full = win(p_full.index, "train")
    full = {(a, json.dumps(v)): ARM_FN[a](p_full, f_full, tr_full, **v)[0] for a, g in I_GRID.items() for v in g}
    rng = np.random.default_rng(28); test_idx = np.flatnonzero(win(p_full.index, "test"))
    bad = []
    for cut in np.sort(rng.choice(test_idx[:-300], 5, replace=False)):
        pc, fc = feats(p_full.iloc[:cut + 1]); trc = win(pc.index, "train")
        for (a, vs), sf in full.items():
            sc = ARM_FN[a](pc, fc, trc, **json.loads(vs))[0]
            win_ = slice(max(0, cut - 300), cut + 1)            # 마지막 300봉 전부 일치해야
            if not np.array_equal(np.asarray(sc)[win_], np.asarray(sf)[win_]):
                bad.append((a, vs, int(cut)))
    rep["future_invariance"] = {"cuts": 5, "bars_checked_per_cut": 301, "violations": bad}
    assert not bad, f"미래참조: {bad[:5]}"
    for arm, grid in I_GRID.items():
        rows = []
        for v in grid:
            pv = pooled(score_arm(arm, data, v, windows=("val",), B_null=50)["val"])
            trial_srs.append(pv.get("daily_sr", 0.0))
            rows.append({"variant": v, "val_pooled": {k: pv.get(k) for k in ("n", "mean", "ci95", "daily_sr")}})
        cand = [x for x in rows if (x["val_pooled"]["n"] or 0) >= 30]
        best = max(cand, key=lambda x: x["val_pooled"]["mean"]) if cand else None
        rep[arm] = {"val_grid": rows, "chosen": best}
        if best is None:
            rep[arm]["verdict"] = {"verdict": "판정 불능", "why": "검증창 거래 < 30"}; continue
        r = score_arm(arm, data, best["variant"], windows=("val", "test"), B_null=300)
        pt, ptk = pooled(r["test"]), pooled(r["test"], "taker")
        rep[arm].update(test_pooled_maker={k: pt.get(k) for k in ("n", "mean", "ci95", "p")},
                        test_pooled_taker={k: ptk.get(k) for k in ("n", "mean", "ci95", "p")},
                        test_per_coin={s: {m_: {k: v_[m_].get(k) for k in ("n", "mean_net", "mean_gross", "null_random_pct", "null_circular_pct", "mde80")}
                                           for m_ in ("maker", "taker")} for s, v_ in r["test"].items()})
        if pt.get("n", 0) >= 10:
            pvals[arm] = pt["p"]; keep[arm] = (r, pt)
        else:
            rep[arm]["verdict"] = {"verdict": "판정 불능", "why": "시험창 거래 < 10"}
    hm = L.holm(pvals)
    for arm, (r, pt) in keep.items():
        dv = L.dsr(np.asarray(pt["daily"]), trial_srs)
        rep[arm]["verdict"] = L.gates(r["val"], r["test"], I_EXPECTED_BP[arm], pt["ci95"], hm[arm]["reject"], dv) | {"dsr": dv}
    rep["holm"] = hm; rep["n_trials"] = len(trial_srs)
    rep["relaxed"] = relaxed_rows(data, I_GRID)                  # 비용 0 으로 바꾸므로 엄격 뒤에
    return rep


# ───────────────────────────── 다음 단계(사전등록 §13) ─────────────────────────────
FWD_VERDICT = pd.Timestamp("2027-04-24")
T1_FWD = [dict(state=s, q=q, H=12) for s in ("rv", "amihud") for q in (.95, .90)]   # [0] = 주 판정(엄격 실행의 검증 선택)


def forward(data, dry):
    """§13-1 T1 전진 판정 — 실제 비용(메이커 0.7·대체 4bp) · 문턱 = 학습 분위. dry = 같은 코드로 시험창(엄격 결과 재현)."""
    w = "test" if dry else "fwd"
    f_out = OUT / ("forward_T1_dry.json" if dry else "forward_T1.json")
    if not dry:
        assert pd.Timestamp.now("UTC").tz_localize(None) >= FWD_VERDICT, "판정일(2027-04-24) 전"
        assert not f_out.exists(), "판정은 한 번만"
        assert all(p.index[-1] >= pd.Timestamp(WIN["fwd"][1]) - pd.Timedelta("5min") for p, _ in data.values()), "자료가 창 끝까지 없음"
    rep = {"window": WIN[w], "cost_bp": {"maker": L.MAKER_BP, "maker_fallback_taker": L.MAKER_FALLBACK_BP}, "variants": []}
    for v in T1_FWD:
        r = score_arm("T1", data, v, windows=(w,), B_null=300)[w]
        pt, ok = pooled(r), [x["maker"] for x in r.values() if x["maker"]["n"] >= 10]
        net = np.concatenate([np.asarray(x["maker"]["net"], float) for x in r.values()])
        day = np.concatenate([np.asarray(x["maker"]["day"], int) for x in r.values()])
        rep["variants"].append({"variant": v, "pooled": {k: pt.get(k) for k in ("n", "mean", "ci95", "p")},
                                "mde80": L.mde80(net, day) if len(net) >= 10 else None,
                                "coins_pos": [sum(x["mean_net"] > 0 for x in ok), len(ok)],
                                "null_random_pct": float(np.mean([x["null_random_pct"] for x in ok])) if ok else None,
                                "per_coin": {s: x["maker"].get("mean_net") for s, x in r.items()}})
    a = rep["variants"][0]; pt = a["pooled"]
    r0 = score_arm("T1", data, T1_FWD[0], windows=(w,), B_null=10)[w]                       # 집중도(§14, 보고만)
    net = np.concatenate([np.asarray(x["maker"]["net"], float) for x in r0.values()])
    day = np.concatenate([np.asarray(x["maker"]["day"], int) for x in r0.values()])
    if len(net) >= 10:
        tot = pd.Series(net).groupby(day).sum().sort_values(ascending=False)
        srt = np.sort(net)[::-1]
        rep["concentration"] = {"n_days": int(len(tot)), "top1_share": float(srt[0] / net.sum()), "top5_share": float(srt[:5].sum() / net.sum()),
                                "mean_drop_top10_trades": float(srt[10:].mean()), "mean_drop_top3_days": float(net[~np.isin(day, tot.index[:3])].mean())}
    rep["F1_candidate_kept"] = bool((pt.get("mean") or -1) > 0 and a["coins_pos"][0] >= -(-a["coins_pos"][1] // 2) and (a["null_random_pct"] or 0) >= 50)
    rep["F2_strict"] = ("통과" if pt.get("ci95") and pt["ci95"][0] > 0 else
                        "판정 불능" if (a["mde80"] or np.inf) > 2 * EXPECTED_BP["T1"] else "불통과")
    json.dump(rep, open(f_out, "w"), ensure_ascii=False, indent=1, default=float)
    return rep


def _ql(y, yh):
    e = y - yh
    return np.exp(e) - e - 1


def _c(y, yh):
    """눈금 상수(12라운드 규칙): ln mean exp(y − ŷ)."""
    return float(np.log(np.mean(np.exp(y - yh))))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _upper90(d, ts):
    """차이 → 일 평균 → 5일 블록 부트 → (평균, 단측 90% 상한)."""
    dl = pd.Series(np.asarray(d, float), index=pd.DatetimeIndex(ts).normalize()).groupby(level=0).mean()
    return float(dl.mean()), float(np.nanquantile(L.cluster_boot_mean(dl.to_numpy(), np.arange(len(dl)) // 5), .9))


def _z(F, cols, fit):
    """적합 행(검증) 평균·표준편차로 표준화 · 결측·무한 0(= 평균) 대입 · 결측 비율."""
    X = F[cols].replace([np.inf, -np.inf], np.nan)
    mu, sd = X[fit].mean(), X[fit].std() + 1e-12
    return ((X - mu) / sd).fillna(0.0).to_numpy(), float(X.isna().any(axis=1).mean())


def rolling_platt90(ts, p, y, W=60, N=90, first="2025-10-01"):
    """배포 A1 재현(fit_box_hold_platt90 와 같은 규칙): 매월 1일 T 에 직전 N일·T 전에 해결된 행(ts+W+1분 ≤ T)으로 Platt → 그 달에 적용."""
    out, z, ts = np.full(len(p), np.nan), _logit(p), pd.DatetimeIndex(ts)
    ab = {}
    for T in pd.date_range(first, ts.max() + pd.offsets.MonthBegin(1), freq="MS"):
        fit = (ts >= T - pd.Timedelta(days=N)) & (ts + pd.Timedelta(minutes=W + 1) <= T)
        m = LogisticRegression(C=1e6, max_iter=1000).fit(z[fit, None], y[fit])
        app = (ts >= T) & (ts < T + pd.offsets.MonthBegin(1))
        if app.any():
            out[app] = m.predict_proba(z[app, None])[:, 1]
        ab[str(T.date())] = (float(m.coef_[0, 0]), float(m.intercept_[0]))
    return out, ab


def incr(data):
    """§13-2 예측 입력(P1·P2) → 배포 A2 QX4(4h 변동성, 4코인) · A1 박스 1h(ETH) 위 증분. 계수는 검증 2026-01~05 만 · 시험 1회."""
    f_out = OUT / "incr.json"
    assert not f_out.exists(), "판정은 한 번만"
    rep = {"qx4": {}, "box": {}}
    for sym in COINS:                                        # ── QX4: ŷ' = ŷ + a + b·z (OLS 잔차) · 기준 = QX4 + c
        s, x = sym[:-4], pred_frame(sym, data)
        Pv, Pt = (pd.read_parquet(OUT / "qx4" / f"pred_{w}_r4_{s}.parquet") for w in ("val", "test"))
        idx = Pv.index.append(Pt.index)
        assert (idx.minute == 55).all(), "QX4 행 = 정시 결정의 5분봉 시작(liqlab 같은 τ 행이 봉 끝 = 결정 시각)"
        F, fit = x.reindex(idx), np.r_[np.ones(len(Pv), bool), np.zeros(len(Pt), bool)]
        h, yy = np.r_[Pv.v_QX4.to_numpy(), Pt.t_QX4.to_numpy()], np.r_[Pv.y.to_numpy(), Pt.y.to_numpy()]
        for arm in ("P1", "P2"):
            Z, nanf = _z(F, P_ADD[arm], fit)
            A = np.c_[np.ones(len(Z)), Z]
            b = np.linalg.lstsq(A[fit], (yy - h)[fit], rcond=None)[0]
            h1 = h + A @ b
            d = _ql(yy[~fit], h1[~fit] + _c(yy[fit], h1[fit])) - _ql(yy[~fit], h[~fit] + _c(yy[fit], h[fit]))
            m, u = _upper90(d, Pt.index)
            rep["qx4"].setdefault(arm, {})[s] = {"d_qlike": m, "upper90": u, "rel_pct": m / float(_ql(yy[~fit], h[~fit] + _c(yy[fit], h[fit])).mean()) * 100,
                                                 "coef": dict(zip(P_ADD[arm], np.round(b[1:], 5))), "nan_frac": nanf, "n_val": int(fit.sum()), "n_test": int((~fit).sum()),
                                                 "monthly": {str(k): round(float(v), 5) for k, v in pd.Series(d, index=Pt.index).groupby(Pt.index.to_period("M")).mean().items()}}
    for arm, r in rep["qx4"].items():
        imp = sum(v["d_qlike"] < 0 for v in r.values())
        r["gate"] = {"coins_improve": imp, "strict": bool(r["ETH"]["upper90"] < 0 and imp >= 3), "candidate": bool(imp >= 3)}
    # ── 박스 1h: 기준 = 로지스틱[1, logit p_A1] · 추가 = [1, logit p_A1, z] — 둘 다 검증에서 다시 맞춤(재보정 이득을 증분에서 뺌)
    sys.path.insert(0, os.getcwd())
    from dashboard.box_hold import B as BH, PLATT90
    cf_ = BH["scales"]["1h"]
    g = pd.read_parquet(OUT / "box_rows.parquet", columns=["scale", "ts", "y"] + list(cf_["cols"]))
    g = g[(g.scale == "1h") & (g.ts >= "2025-06-01")].dropna(subset=list(cf_["cols"])).reset_index(drop=True)
    ts, y = pd.DatetimeIndex(g.ts), g.y.to_numpy(float)
    assert (ts.minute % 5 == 4).all(), "박스 1h 격자 = 분 i 시작(결정 i+1분) · liqlab 봉 τ = ts−4분이 결정에 끝남"
    pA, ab = rolling_platt90(ts, cf_["model"].predict_proba(g[list(cf_["cols"])])[:, 1], y)
    served = json.loads(PLATT90.read_text()) if PLATT90.exists() else None
    rep["box"]["platt_2026-10-01"] = {"replay": ab.get("2026-10-01"), "served": served}
    va = (ts >= "2026-01-01") & (ts + pd.Timedelta(minutes=61) <= pd.Timestamp("2026-06-01"))
    te = ts >= "2026-06-01"
    assert np.isfinite(pA[va | te]).all()
    F, lz = pred_frame("ETHUSDT", data).reindex(ts - pd.Timedelta("4min")), _logit(pA)[:, None]
    base = LogisticRegression(C=1e6, max_iter=2000).fit(lz[va], y[va])
    pb = base.predict_proba(lz[te])[:, 1]
    for arm in ("P1", "P2"):
        Z, nanf = _z(F, P_ADD[arm], va)
        X = np.c_[lz, Z]
        add = LogisticRegression(C=1e6, max_iter=4000).fit(X[va], y[va])
        pa = add.predict_proba(X[te])[:, 1]
        mB, uB = _upper90((pa - y[te]) ** 2 - (pb - y[te]) ** 2, ts[te])
        mL, uL = _upper90(logloss(y[te], pa) - logloss(y[te], pb), ts[te])
        rep["box"][arm] = {"d_brier": mB, "upper90_brier": uB, "d_logloss": mL, "upper90_logloss": uL,
                           "brier_base": float(np.mean((pb - y[te]) ** 2)), "brier_A1_asis": float(np.mean((pA[te] - y[te]) ** 2)),
                           "coef": dict(zip(P_ADD[arm], np.round(add.coef_[0][1:], 5))), "nan_frac": nanf, "n_val": int(va.sum()), "n_test": int(te.sum()),
                           "gate": {"strict": bool(uB < 0 and mL < 0), "candidate": bool(mB < 0 and mL < 0)}}
    json.dump(rep, open(f_out, "w"), ensure_ascii=False, indent=1, default=float)
    return rep


def main():
    data = {s: panel(s) for s in COINS}
    if "--forward" in sys.argv:
        print(json.dumps(forward(data, "--dry" in sys.argv), ensure_ascii=False, default=float)); return
    if "--incr" in sys.argv:
        print(json.dumps(incr(data), ensure_ascii=False, default=float)); return
    if "--internal" in sys.argv:
        rep = internal(data)
        json.dump(rep, open(OUT / "internal.json", "w"), ensure_ascii=False, indent=1, default=float)
        for k in I_GRID:
            print(k, json.dumps(rep[k].get("verdict"), ensure_ascii=False, default=float))
        print("relaxed cands", [(x["arm"], x["variant"], x["mode"]) for x in rep["relaxed"] if x["candidate"]]); return
    if "--relaxed" in sys.argv:
        rep = relaxed(data)
        json.dump(rep, open(OUT / "relaxed.json", "w"), ensure_ascii=False, indent=1, default=float)
        print(json.dumps({"T4": rep["T4"], "pred": rep["pred"], "cands": [(x["arm"], x["variant"], x["mode"]) for x in rep["trading"] if x["candidate"]]},
                         ensure_ascii=False, default=float)); return
    if "--diag" in sys.argv:
        rep = diag(data)
        json.dump(rep, open(OUT / "diag.json", "w"), ensure_ascii=False, indent=1, default=float)
        print(json.dumps(rep, ensure_ascii=False, default=float)); return
    if "--controls" in sys.argv:
        rep = controls(data)
        json.dump(rep, open(OUT / "controls.json", "w"), ensure_ascii=False, indent=1, default=float)
        print(json.dumps(rep, ensure_ascii=False, default=float)[:4000]); return
    report = {"meta": {"windows": WIN, "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
                       "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
                       "cost_bp": {"maker": L.MAKER_BP, "maker_fallback_taker": L.MAKER_FALLBACK_BP, "taker": L.TAKER_BP, "t4_stress": T4_STRESS_BP}}}
    trial_srs, chosen = [], {}
    for arm, grid in T_GRID.items():                        # 1) 검증창에서 전 변형 채점 → 선택
        rows = []
        for v in grid:
            pv = pooled(score_arm(arm, data, v, windows=("val",), B_null=50)["val"])
            trial_srs.append(pv.get("daily_sr", 0.0))
            rows.append({"variant": v, "val_pooled": {k: pv.get(k) for k in ("n", "mean", "ci95", "daily_sr")}})
        cand = [x for x in rows if SELECTABLE[arm](x["variant"]) and (x["val_pooled"]["n"] or 0) >= 30]
        chosen[arm] = max(cand, key=lambda x: x["val_pooled"]["mean"]) if cand else None
        report[arm] = {"val_grid": rows, "chosen": chosen[arm]}
    t4rows = []
    for cond in ("lowvol", "highvol", "none"):
        net, _, _ = t4_returns(cond, T4_STRESS_BP)
        v = net[win(net.index, "val")]; sr = float(v.mean() / v.std()) if v.std() > 0 else 0.0
        trial_srs.append(sr); t4rows.append({"cond": cond, "val_mean_bp_day": float(v.mean()), "val_sr": sr})
    best4 = max(t4rows, key=lambda x: x["val_mean_bp_day"]); report["T4"] = {"val_grid": t4rows, "chosen": best4}
    pvals, keep = {}, {}
    for arm in ("T1", "T2", "T3"):                          # 2) 시험창 1회
        if chosen[arm] is None:
            report[arm]["verdict"] = {"verdict": "판정 불능", "why": "검증창 거래 < 30"}; continue
        r = score_arm(arm, data, chosen[arm]["variant"], windows=("val", "test"), B_null=300)
        pt, ptk = pooled(r["test"]), pooled(r["test"], "taker")
        report[arm].update(
            test_per_coin={s: {m_: {k: v[m_].get(k) for k in ("n", "mean_net", "mean_gross", "ci95", "hit", "null_random_pct", "null_circular_pct",
                                                            "excess_vs_long", "excess_vs_short", "mde80", "exits")} for m_ in ("maker", "taker")}
                           for s, v in r["test"].items()},
            test_pooled_maker={k: pt.get(k) for k in ("n", "mean", "ci95", "p", "daily_sr")},
            test_pooled_taker={k: ptk.get(k) for k in ("n", "mean", "ci95", "p")})
        pvals[arm] = pt.get("p", 1.0); keep[arm] = (r, pt)
    W4 = None
    for cost, lab in ((T4_STRESS_BP, "maker_stress"), (L.TAKER_BP, "taker")):
        net, W, nxt = t4_returns(best4["cond"], cost)
        t = net[win(net.index, "test")]; d = L.cluster_boot_mean(t.to_numpy(), np.arange(len(t)) // 5)
        report["T4"][f"test_{lab}"] = {"days": len(t), "mean_bp_day": float(t.mean()), "ci95": L.ci95(d), "p": L.p_two_sided(d),
                                       "sr_daily": float(t.mean() / t.std())}
        if lab == "maker_stress":
            W4, t4_daily, gross_t = (W.loc[t.index], nxt.loc[t.index]), t.to_numpy(), float(((W * nxt).sum(1) * 1e4)[t.index].mean())
            pvals["T4"] = L.p_two_sided(d)
    report["T4"]["test_null_pct"] = float((t4_null(*W4) < gross_t).mean() * 100)
    hm = L.holm(pvals)
    for arm, (r, pt) in keep.items():
        dv = L.dsr(np.asarray(pt["daily"]), trial_srs)
        report[arm]["dsr"] = dv
        report[arm]["verdict"] = L.gates(r["val"], r["test"], EXPECTED_BP[arm], pt["ci95"], hm[arm]["reject"], dv)
    tt = report["T4"]["test_maker_stress"]; dv = L.dsr(t4_daily, trial_srs)
    g4 = {"G1": tt["ci95"][0] > 0, "G2": report["T4"]["test_null_pct"] >= 95,
          "G3": best4["val_mean_bp_day"] > 0 and tt["mean_bp_day"] > 0, "G5": hm["T4"]["reject"] and tt["mean_bp_day"] > 0, "G6": dv > .95}
    report["T4"].update(dsr=dv, verdict={"verdict": "통과" if all(g4.values()) else "불통과", **g4})
    report["holm"] = hm; report["n_trials"] = len(trial_srs)
    for arm, target in (("P1", "y_P1_1h"), ("P1_4h", "y_P1_4h"), ("P2", "y_P2"), ("P3", "y_P3")):
        res = score_pred(arm[:2], data, target); report[arm] = {"per_coin": res, "verdict": pred_gate(res)}
    json.dump(report, open(OUT / "report.json", "w"), ensure_ascii=False, indent=1, default=float)
    for k in ("T1", "T2", "T3", "T4", "P1", "P1_4h", "P2", "P3"):
        print(k, json.dumps(report[k].get("verdict"), ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
