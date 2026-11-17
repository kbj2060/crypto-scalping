#!/usr/bin/env python3
"""ETH 지지·저항 터치 순간의 «매매 강도»로 돌파 vs 되돌림 판별 (2026-11-18).

  selftest             합성 초 자료: 터치 초 찾기 · 피쳐가 결정 시각 뒤를 안 봄 · 라벨 직접 대조
  feats                사건(11-10 연구와 같은 ETH 첫 터치 4,648건)마다 초 단위 강도 피쳐·라벨 → tmp/sr_intensity_20261118/events_w{W}.parquet
  explore              학습 구간(~2025-12)만으로 결정 지연 W·라벨 지평별 교차검증 AUC(설계 고르기 · 판정 아님) — 화면 지지·저항 터치는 여기서 접음(AUC .48~.52)
  minutes              체결 1초 묶음 → 1분봉(고가·저가·종가·거래량·테이커 매수) 캐시
  hilo                 4시간 고점/저점 첫 돌파 사건의 피쳐·라벨 → tmp/sr_intensity_20261118/hilo_events.parquet
  runhilo              사전등록 docs/prereg/sr_break_intensity_hilo_20261118.md 사양으로 학습 → 검증·시험 1회 채점

사건 = 5분봉이 가장 가까운 지지/저항 L 에 처음 닿은 봉(직전 12봉에 같은 쪽 터치 없음) — `research_sr_break_vs_revert_20261110` 의 사건 그대로.
이번에 바꾼 것: 결정 시각 = 터치한 **초** t* 에서 W 초 뒤(t_d = t* + W). 피쳐 = 초 ≤ t_d 의 체결만. 기준가 = t_d 까지 마지막 체결가. 라벨 = t_d + 1 초부터.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/sr_intensity_20261118"
AGG = OUT / "agg1s"
EV0 = Path("/home/kbj20/crypto-scalping/.claude/worktrees/direction-prediction-model-test-085284/tmp/sr_break_vs_revert_20261110/events.parquet")
VAL0, TEST0, PG = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-01"), pd.Timedelta(hours=5)
HZ = (300, 900, 3600)                                                 # 라벨 지평(초)
KD = (0.25, 0.5)                                                      # 배리어 = KD × σ_1h
SEED, NB = 20261118, 2000


class Tape:
    """일 파일(체결 있던 초만)을 필요할 때 읽어 [a, b] 초 구간을 빈틈 없는 배열로 돌려준다."""
    def __init__(self, root: Path = AGG):
        self.root, self.cache = root, {}

    def day(self, d: str):
        if d not in self.cache:
            p = self.root / f"ETHUSDT-{d}.parquet"
            self.cache[d] = pd.read_parquet(p) if p.exists() else None
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[d]

    def window(self, a: int, b: int):
        """초 a..b(포함) → dict(배열 길이 b−a+1): pc(직전 값으로 채운 마지막 체결가)·ph·pl(체결 없는 초는 pc)·bq·sq·nt·q2 · ok(자료 있음)."""
        days = pd.date_range(pd.Timestamp(a, unit="s").normalize(), pd.Timestamp(b, unit="s").normalize(), freq="D").strftime("%Y-%m-%d")
        parts = [self.day(d) for d in days]
        if any(p is None for p in parts):
            return None
        g = pd.concat(parts) if len(parts) > 1 else parts[0]
        s = g.s.to_numpy(); i0, i1 = np.searchsorted(s, a), np.searchsorted(s, b, "right")
        if i0 == 0 and s[0] > a:                                      # 창 시작 전에 체결이 없으면 앞 날이 필요
            return None
        n = b - a + 1; W = {k: np.zeros(n) for k in ("bq", "sq", "nt", "q2")}
        idx = s[i0:i1] - a
        for k in W:
            W[k][idx] = g[k].to_numpy()[i0:i1]
        pc = np.full(n, np.nan); pc[idx] = g.pc.to_numpy()[i0:i1]
        if np.isnan(pc[0]):
            pc[0] = g.pc.to_numpy()[i0 - 1] if i0 > 0 else np.nan
        pc = pd.Series(pc).ffill().to_numpy()
        ph, pl = pc.copy(), pc.copy(); ph[idx], pl[idx] = g.ph.to_numpy()[i0:i1], g.pl.to_numpy()[i0:i1]
        W.update(pc=pc, ph=ph, pl=pl, traded=np.isin(np.arange(n), idx))
        return W


def events0() -> pd.DataFrame:
    """11-10 연구의 ETH 사건 + 레벨 가격 복원. ts = 터치 봉 마감(UTC). L = c·exp(−side·g_tb_cl·σ1h), σ1h = m_D_bp ÷ 5000."""
    d = pd.read_parquet(EV0); d = d[d.coin == "ETH"].reset_index(drop=True)
    d["s1h"] = d.m_D_bp / 5000.0; d["te"] = (d.ts.astype("int64") // 10**9).astype(np.int64)
    return d


def first_pass(ph, pl, P, D):
    """기준가 P 에서 ±D(로그) 중 먼저 닿는 쪽: +1 위 · −1 아래 · 0 같은 초 · nan 미도달."""
    up, dn = ph >= P * np.exp(D), pl <= P * np.exp(-D)
    iu, idn = (up.argmax() if up.any() else len(ph)), (dn.argmax() if dn.any() else len(ph))
    return 1.0 if iu < idn else -1.0 if idn < iu else (np.nan if iu == len(ph) else 0.0)


def one_event(T: Tape, te: int, side: int, s1h: float, g_tb_cl: float, W: int, tape_close: float | None = None):
    """사건 하나의 피쳐·라벨. te = 터치 봉 마감 초. 반환 dict 또는 None(자료 없음·터치 초 못 찾음)."""
    a, b = te - 300 - 3600, te + max(HZ) + W + 5
    X = T.window(a, b)
    if X is None:
        return None
    pc, ph, pl = X["pc"], X["ph"], X["pl"]; o = 3600                    # o = 터치 봉 시작의 배열 위치
    c_bar = pc[o + 299] if tape_close is None else tape_close           # 봉 마감 종가(= 초 te−1 까지 마지막 체결가)
    if not np.isfinite(c_bar):
        return None
    L = c_bar * np.exp(-side * g_tb_cl * s1h)
    seg_h, seg_l = ph[o:o + 300], pl[o:o + 300]
    hit = (seg_h >= L * (1 - 1e-9)) if side > 0 else (seg_l <= L * (1 + 1e-9))
    if not hit.any():
        return None
    k = o + int(hit.argmax()); kd = k + W                               # 터치 초 · 결정 초(배열 위치)
    vol = X["bq"] + X["sq"]; sgn_v = side * (X["bq"] - X["sq"]); nt = X["nt"]; lp = np.log(pc)
    eps = 1e-9
    def w(arr, i0, i1):                                                 # [i0, i1) 합
        return float(arr[max(i0, 0):max(i1, 0)].sum())
    base_v = w(vol, k - 3600, k) / 3600 + eps; base_n = w(nt, k - 3600, k) / 3600 + eps      # 직전 1시간 초당 평균
    s1s = s1h / 60.0                                                    # σ_1h → 1초 눈금(√3600)
    f = {}
    for n in (30, 60, 300):                                             # ── 터치 전 접근
        f[f"pre_imb{n}"] = w(sgn_v, k - n, k) / (w(vol, k - n, k) + eps)
        f[f"pre_vol{n}"] = np.log((w(vol, k - n, k) / n + eps) / base_v)
        f[f"pre_n{n}"] = np.log((w(nt, k - n, k) / n + eps) / base_n)
        f[f"pre_spd{n}"] = side * (lp[k] - lp[k - n]) / (s1s * np.sqrt(n))
    f["pre_accel"] = np.log((w(vol, k - 30, k) + eps) / (w(vol, k - 60, k - 30) + eps))
    f["pre_big60"] = w(X["q2"], k - 60, k) / (w(vol, k - 60, k) ** 2 + eps)
    f["pre_size60"] = np.log((w(vol, k - 60, k) / (w(nt, k - 60, k) + eps) + eps) / (base_v / base_n))
    f["pre_peak60"] = float(vol[k - 60:k].max() / (w(vol, k - 60, k) + eps))
    if W > 0:                                                           # ── 터치 뒤 관찰 창 [k, kd]
        f["obs_imb"] = w(sgn_v, k, kd + 1) / (w(vol, k, kd + 1) + eps)
        f["obs_vol"] = np.log((w(vol, k, kd + 1) / (W + 1) + eps) / base_v)
        f["obs_n"] = np.log((w(nt, k, kd + 1) / (W + 1) + eps) / base_n)
        beyond = side * (lp[k:kd + 1] - np.log(L))
        ext = side * (np.log(ph[k:kd + 1] if side > 0 else pl[k:kd + 1]) - np.log(L))
        f["obs_pen"] = float(ext.max() / s1h); f["obs_time_beyond"] = float((beyond > 0).mean()); f["obs_pos"] = float(beyond[-1] / s1h)
        f["obs_cross"] = float((np.diff(np.sign(beyond)) != 0).sum())
        f["obs_absorb"] = np.log((w(np.where(side > 0, X["bq"], X["sq"]), k, kd + 1) / (W + 1) + eps) / base_v) - np.log1p(max(ext.max(), 0) * 1e4)
        f["obs_ret"] = side * (lp[kd] - lp[k]) / (s1s * np.sqrt(W))
        f["obs_big"] = w(X["q2"], k, kd + 1) / (w(vol, k, kd + 1) ** 2 + eps)
    f["t_in_bar"] = (k - o) / 300.0
    P = pc[kd]; out = dict(f, kpos=int(a + k), tdec=int(a + kd), P=float(P), L=float(L))
    for hz in HZ:                                                       # ── 라벨: 결정 다음 초부터
        fh, fl_ = ph[kd + 1:kd + 1 + hz], pl[kd + 1:kd + 1 + hz]
        for kdm in KD:
            fp = first_pass(fh, fl_, P, kdm * s1h)
            out[f"y_{hz}_{kdm}"] = np.nan if (np.isnan(fp) or fp == 0) else float(fp == side)
        out[f"ret_{hz}"] = side * float(np.log(pc[kd + hz] / P)) * 1e4
    return out


def build(W: int, limit: int | None = None) -> pd.DataFrame:
    ev = events0(); T = Tape(); rows = []
    for j, r in enumerate(ev.itertuples()):
        if limit and j >= limit:
            break
        o = one_event(T, int(r.te), int(r.f_side), float(r.s1h), float(r.g_tb_cl), W)
        rows.append(dict(i=j, **(o or {})))
    X = pd.DataFrame(rows).set_index("i")
    keep = [c for c in ev.columns if c[:2] in ("f_", "g_")] + ["y", "y1", "m_D_bp", "ts", "s1h"]
    return ev[keep].join(X, how="left")


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(y) & np.isfinite(s)
    return float(roc_auc_score(y[ok], s[ok])) if ok.sum() > 50 and len(set(y[ok])) == 2 else np.nan


def cv_auc(d: pd.DataFrame, cols: list, yname: str, folds: int = 5, arm: str = "HGB") -> tuple:
    """시간순 토막 교차 예측(토막마다 AUC → 평균). 합친 AUC 는 토막 기저율 차로 치우치므로 토막 평균을 쓴다."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = d[d[yname].notna()].sort_values("ts").reset_index(drop=True); edges = np.linspace(0, len(d), folds + 1).astype(int); out = []
    for j in range(folds):
        te = np.arange(edges[j], edges[j + 1]); t0, t1 = d.ts[te[0]] - PG, d.ts[te[-1]] + PG
        tr = np.flatnonzero((d.ts < t0) | (d.ts > t1))
        m = (HistGradientBoostingClassifier(max_depth=3, learning_rate=.03, max_iter=300, l2_regularization=1.0, min_samples_leaf=60, early_stopping=False, random_state=0) if arm == "HGB"
             else make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=.1, max_iter=2000)))
        m.fit(d.loc[tr, cols], d.loc[tr, yname]); out.append(auc(d.loc[te, yname].to_numpy(), m.predict_proba(d.loc[te, cols])[:, 1]))
    return float(np.mean(out)), [round(x, 3) for x in out], len(d), float(d[yname].mean())


def explore() -> None:
    """학습 구간만(ts < 2026-01-01 − 5h). 설계 고르기용 — 검증·시험 행은 읽지 않는다."""
    res = {}
    for W in (5, 30, 60):
        p = OUT / f"events_w{W}.parquet"
        d = pd.read_parquet(p); d = d[(d.ts < VAL0 - PG) & d.P.notna()].reset_index(drop=True)
        new = [c for c in d.columns if c.startswith(("pre_", "obs_", "t_in_bar"))]; old = [c for c in d.columns if c[:2] == "f_"]
        print(f"\n== W={W}초 · 학습 사건 {len(d)} · 새 피쳐 {len(new)} · 옛 피쳐(봉 s−1 까지) {len(old)}")
        for hz in HZ:
            for kdm in KD:
                yn = f"y_{hz}_{kdm}"
                for nm, cols, arm in (("강도만·HGB", new, "HGB"), ("강도만·LOG", new, "LOG"), ("강도+옛·HGB", new + old, "HGB"), ("옛만·HGB", old, "HGB")):
                    a, fl, n, br = cv_auc(d, cols, yn, arm=arm); res[f"W{W}|{yn}|{nm}"] = a
                    print(f"   지평 {hz:>4}s · 배리어 {kdm}σ · {nm}: 토막 평균 AUC {a:.3f} {fl} · n {n} · 뚫음 비율 {br:.3f}")
        u = pd.Series({c: auc(d[f"y_900_0.25"].to_numpy(), d[c].to_numpy()) for c in new}).sort_values()
        print("   단변량 AUC(y_900_0.25) 낮은 5:", u.head(5).round(3).to_dict(), "· 높은 5:", u.tail(5).round(3).to_dict())
    (OUT / "explore.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


# ── 4시간 고점/저점 돌파(«격하게 뚫었나 · 두드리다 넘었나») ─────────────────────────────────────────
LOOK, GAP, WS, TS, DM = 240, 60, (1, 5), (60, 300), 0.25
SIG = ["ls", "h_sin", "h_cos", "is24"]                                  # 강도 없는 기준선(평소 변동성·시각)
INTEN = ["pre_vol60", "pre_vol10", "pre_n10", "pre_accel", "pre_imb60", "pre_imb10", "pre_spd60", "pre_spd300", "obs_vol{W}", "obs_imb{W}", "obs_pos{W}"]
CTX = ["lage", "near", "rng4", "mom15", "mom60", "mom1d", "pull", "tb60", "tb240", "side"]


def minute_bars(root: Path = AGG, cache: Path | None = None) -> pd.DataFrame:
    """체결 1초 묶음 → 1분봉. 색인 = 분(epoch 분, 빈틈 없는 격자 · 체결 없는 분은 NaN)."""
    cache = cache if cache is not None else OUT / "min_from_tape.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    parts = []
    for f in sorted(root.glob("ETHUSDT-*.parquet")):
        g = pd.read_parquet(f, columns=["s", "ph", "pl", "pc", "bq", "sq"]); g["m"] = g.s // 60
        parts.append(g.groupby("m").agg(h=("ph", "max"), l=("pl", "min"), c=("pc", "last"), bq=("bq", "sum"), sq=("sq", "sum")))
    m = pd.concat(parts); m = m[~m.index.duplicated()].sort_index(); m["v"], m["tb"] = m.bq + m.sq, m.bq
    m = m[["h", "l", "c", "v", "tb"]].reindex(range(int(m.index[0]), int(m.index[-1]) + 1)); m.to_parquet(cache); return m


def hilo_events(m: pd.DataFrame) -> pd.DataFrame:
    """직전 240분 고점(저점)을 처음 넘는 분(직전 60분에 같은 쪽 돌파 없음). 레벨·σ 는 그 분 이전 자료만.
    «같은 분에 반대쪽도 넘으면 제외»는 두지 않는다 — 그 분 끝(결정 뒤 최대 55초)까지의 고가·저가를 보게 된다(selftest 가 잡음, 채점 전 수정)."""
    lc = np.log(m.c); s1h = (lc.diff().rolling(1440, min_periods=1200).std() * np.sqrt(60)).shift(1)
    def sets(look):
        hh, ll = m.h.rolling(look, min_periods=int(look * .98)).max().shift(1), m.l.rolling(look, min_periods=int(look * .98)).min().shift(1)
        bu, bd = m.h > hh, m.l < ll
        first = lambda b: b & ~(b.astype(float).rolling(GAP).max().shift(1).fillna(0).astype(bool))
        return first(bu) & s1h.notna(), first(bd) & s1h.notna(), hh, ll
    u, dn, hh, ll = sets(LOOK); _, _, hh24, ll24 = sets(1440)
    # is24 = 넘은 레벨이 24시간 고점(저점)이기도 한가 — 분 i 이전에 정해진다. 🔴첫 채점(v1)은 «그 분에 24시간 고저도 넘었나»(분 끝까지의 고가)라 결정 뒤 최대 55초를 봤다(채점 뒤 발견·수정).
    ev = pd.concat([pd.DataFrame({"mi": m.index[u], "side": 1, "L": hh[u].to_numpy(), "s1h": s1h[u].to_numpy(), "is24": np.isclose(hh24[u].to_numpy(), hh[u].to_numpy(), rtol=1e-12).astype(float)}),
                    pd.DataFrame({"mi": m.index[dn], "side": -1, "L": ll[dn].to_numpy(), "s1h": s1h[dn].to_numpy(), "is24": np.isclose(ll24[dn].to_numpy(), ll[dn].to_numpy(), rtol=1e-12).astype(float)})])
    return ev.sort_values("mi").reset_index(drop=True)


def hilo_one(T: Tape, M_: dict, mi: int, side: int, L: float, s1h: float):
    """사건 하나: 돌파 초 k 를 찾아 접근(초 < k)·관찰(k..k+W)·맥락(분 < mi) 피쳐와 결정 뒤 라벨."""
    a, b = mi * 60 - 3700, mi * 60 + 60 + 3700; X = T.window(a, b)
    if X is None:
        return None
    o = 3700; seg = (X["ph"][o:o + 60] > L) if side > 0 else (X["pl"][o:o + 60] < L)
    if not seg.any():
        return None
    k = o + int(seg.argmax()); vol = X["bq"] + X["sq"]; sv = side * (X["bq"] - X["sq"]); lp = np.log(X["pc"]); eps = 1e-9
    base_v = vol[k - 3600:k].sum() / 3600 + eps; base_n = X["nt"][k - 3600:k].sum() / 3600 + eps; s1s = s1h / 60
    f = dict(ksec=int(a + k), pre_vol60=np.log((vol[k - 60:k].sum() / 60 + eps) / base_v), pre_vol10=np.log((vol[k - 10:k].sum() / 10 + eps) / base_v),
             pre_n10=np.log((X["nt"][k - 10:k].sum() / 10 + eps) / base_n), pre_imb60=sv[k - 60:k].sum() / (vol[k - 60:k].sum() + eps), pre_imb10=sv[k - 10:k].sum() / (vol[k - 10:k].sum() + eps),
             pre_spd60=side * (lp[k] - lp[k - 60]) / (s1s * np.sqrt(60)), pre_spd300=side * (lp[k] - lp[k - 300]) / (s1s * np.sqrt(300)),
             pre_accel=np.log((vol[k - 10:k].sum() + eps) / (vol[k - 60:k - 10].sum() / 5 + eps)))
    for W in WS:
        kd = k + W; P = X["pc"][kd]
        f[f"obs_imb{W}"] = sv[k:kd + 1].sum() / (vol[k:kd + 1].sum() + eps); f[f"obs_vol{W}"] = np.log((vol[k:kd + 1].sum() / (W + 1) + eps) / base_v); f[f"obs_pos{W}"] = side * np.log(P / L) / s1h
        for Tz in TS:
            fp = first_pass(X["ph"][kd + 1:kd + 1 + Tz], X["pl"][kd + 1:kd + 1 + Tz], P, DM * s1h)
            f[f"y{W}_{Tz}"] = 0 if (np.isnan(fp) or fp == 0) else int(fp * side); f[f"ret{W}_{Tz}"] = side * np.log(X["pc"][kd + Tz] / P) * 1e4
    H, Lo, C, V, TB, s1m, base = M_["H"], M_["L"], M_["C"], M_["V"], M_["TB"], M_["s1m"], M_["base"]
    i = mi - base; w = slice(i - LOOK, i); ext = H[w] if side > 0 else -Lo[w]; j = int(np.nanargmax(ext))
    f.update(lage=np.log(LOOK - j), near=float((np.abs(np.log((H[w] if side > 0 else Lo[w]) / L)) < 0.1 * s1h).sum()), rng4=np.log(np.nanmax(H[w]) / np.nanmin(Lo[w])) / s1h,
             mom15=side * np.log(C[i - 1] / C[i - 16]) / (s1m[i] * np.sqrt(15)), mom60=side * np.log(C[i - 1] / C[i - 61]) / (s1m[i] * np.sqrt(60)),
             mom1d=side * np.log(C[i - 1] / C[i - 1441]) / (s1m[i] * np.sqrt(1440)), pull=side * np.log(L / C[i - 1]) / s1h,
             tb60=side * (2 * np.nansum(TB[i - 60:i]) / max(np.nansum(V[i - 60:i]), 1e-9) - 1), tb240=side * (2 * np.nansum(TB[i - 240:i]) / max(np.nansum(V[i - 240:i]), 1e-9) - 1))
    return f


def marr(m: pd.DataFrame) -> dict:
    return dict(H=m.h.to_numpy(), L=m.l.to_numpy(), C=m.c.to_numpy(), V=m.v.to_numpy(), TB=m.tb.to_numpy(), base=int(m.index[0]),
                s1m=np.log(m.c).diff().rolling(1440, min_periods=1200).std().shift(1).to_numpy())


def build_hilo(m: pd.DataFrame | None = None, tape: Tape | None = None) -> pd.DataFrame:
    m = minute_bars() if m is None else m; ev = hilo_events(m); T = tape or Tape(); M_ = marr(m); rows = []
    for r in ev.itertuples():
        f = hilo_one(T, M_, int(r.mi), int(r.side), float(r.L), float(r.s1h))
        if f is not None:
            rows.append(dict(mi=r.mi, side=r.side, L=r.L, s1h=r.s1h, is24=r.is24, **f))
    d = pd.DataFrame(rows); d["ts"] = pd.to_datetime(d.mi * 60, unit="s"); hod = (d.mi % 1440) / 1440
    d["h_sin"], d["h_cos"], d["ls"] = np.sin(2 * np.pi * hod), np.cos(2 * np.pi * hod), np.log(d.s1h)
    return d


def _lr():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=.3, max_iter=3000))


def _cls(y):
    return np.where(y == 1, 0, np.where(y == -1, 1, 2))                 # 0 C(큰 이어짐) · 1 R(큰 되돌림) · 2 N


def _p3(m, X):
    """predict_proba 를 늘 [C, R, N] 3열로(학습에 없는 분류는 0)."""
    p = np.zeros((len(X), 3)); p[:, m.classes_.astype(int)] = m.predict_proba(X); return np.clip(p, 1e-9, 1)


def hilo_eval(te: pd.DataFrame, P: np.ndarray, Pb: np.ndarray, thr: tuple, rng, ycol: str) -> dict:
    """한 구간 채점. P·Pb = 모델·기준선의 [C, R, N] 확률. thr = 학습 교차 예측의 방향 점수 20·80 분위."""
    from sklearn.metrics import roc_auc_score
    cls = _cls(te[ycol].to_numpy()); n = len(te); day = te.ts.dt.floor("D").to_numpy(); res = cls < 2
    ll = lambda p: -np.log(p[np.arange(n), cls]); dl = ll(P) - ll(Pb)
    r, rb = np.log(P[:, 0] / P[:, 1]), np.log(Pb[:, 0] / Pb[:, 1]); top, bot = r >= thr[1], r <= thr[0]
    def auc_(score, m, c=None):                                        # c = 그 점수와 같은 순서의 분류(다시 뽑은 표본이면 반드시 넘긴다)
        c = cls if c is None else c; y = (c[m] == 0); return float(roc_auc_score(y, score[m])) if 0 < y.sum() < len(y) else np.nan
    def share(m):                                                      # 결판 중 C 비율
        k = (res & m).sum(); return float((cls[m] == 0).sum() / k) if k else np.nan
    days = np.unique(day); idx = {d_: np.flatnonzero(day == d_) for d_ in days}; b1, b2, b3, b4 = [], [], [], []
    for _ in range(NB):
        pk = np.concatenate([idx[d_] for d_ in rng.choice(days, len(days))]); rs = res[pk]
        b1.append(dl[pk].mean()); a_f = auc_(r[pk], rs, cls[pk]) if rs.sum() > 20 else np.nan; b2.append(a_f); b4.append(a_f - (auc_(rb[pk], rs, cls[pk]) if rs.sum() > 20 else np.nan))
        t_, o_ = top[pk] & rs, bot[pk] & rs
        b3.append(((cls[pk][t_] == 0).mean() - (cls[pk][o_] == 0).mean()) if t_.sum() > 5 and o_.sum() > 5 else np.nan)
    mon = te.ts.dt.to_period("M").astype(str).to_numpy(); pla = []
    for _ in range(1000):                                               # 위약: 결판 난 사건의 C/R 을 월 안에서 섞는다
        c2 = cls.copy()
        for mm in np.unique(mon[res]):
            j = np.flatnonzero(res & (mon == mm)); c2[j] = rng.permutation(cls[j])
        pla.append(roc_auc_score(c2[res] == 0, r[res]))
    cnt = lambda m: [int((cls[m] == q).sum()) for q in (0, 1, 2)]
    a_dir, a_base = auc_(r, res), auc_(rb, res)
    out = {"사건": n, "날": int(len(days)), "C·R·N": cnt(np.ones(n, bool)), "기저_C÷(C+R)": share(np.ones(n, bool)),
           "로그손실_모델": float(ll(P).mean()), "로그손실_기준선": float(ll(Pb).mean()), "로그손실_차": float(dl.mean()), "로그손실_차_상한90": float(np.nanpercentile(b1, 90)),
           "AUC_C": float(roc_auc_score(cls == 0, P[:, 0])), "AUC_R": float(roc_auc_score(cls == 1, P[:, 1])),
           "방향_AUC": a_dir, "방향_AUC_하한90": float(np.nanpercentile(b2, 10)), "위약_95분위": float(np.percentile(pla, 95)), "방향_AUC_기준선": a_base,
           "모델몫": a_dir - a_base, "모델몫_하한90": float(np.nanpercentile(b4, 10)),
           "집단(상위20%·가운데·하위20%) C·R·N": [cnt(top), cnt(~top & ~bot), cnt(bot)], "상위_C비율": share(top), "하위_C비율": share(bot), "집단차": share(top) - share(bot), "집단차_하한90": float(np.nanpercentile(b3, 10))}
    out["1_크기"] = bool(out["로그손실_차_상한90"] < 0); out["2_방향"] = bool(out["방향_AUC_하한90"] > 0.5 and a_dir > out["위약_95분위"])
    out["3_집단차"] = bool(out["집단차_하한90"] > 0); out["4_모델몫"] = bool(out["모델몫_하한90"] > 0); out["통과"] = bool(out["1_크기"] and out["2_방향"] and out["3_집단차"] and out["4_모델몫"])
    out["후보_점추정"] = bool(out["로그손실_차"] < 0 and a_dir > 0.5 and out["집단차"] > 0)
    return out


def run_hilo(W: int = 5, Tz: int = 60) -> None:
    from sklearn.metrics import roc_auc_score
    d = pd.read_parquet(OUT / "hilo_events.parquet").sort_values("ts").reset_index(drop=True); ycol = f"y{W}_{Tz}"; rng = np.random.default_rng(SEED)
    inten = [c.format(W=W) for c in INTEN]; full, base = SIG + inten + CTX, SIG
    tr, va, te = d[d.ts < VAL0 - PG].reset_index(drop=True), d[(d.ts >= VAL0) & (d.ts < TEST0 - PG)].reset_index(drop=True), d[d.ts >= TEST0].reset_index(drop=True)
    ctr = _cls(tr[ycol].to_numpy()); m, mb = _lr().fit(tr[full], ctr), _lr().fit(tr[base], ctr)
    edges = np.linspace(0, len(tr), 6).astype(int); oof = np.zeros((len(tr), 3))                      # 문턱용 학습 교차 예측
    for j in range(5):
        k = np.arange(edges[j], edges[j + 1]); t_ = np.flatnonzero((tr.ts < tr.ts[k[0]] - PG) | (tr.ts > tr.ts[k[-1]] + PG)); oof[k] = _p3(_lr().fit(tr.loc[t_, full], ctr[t_]), tr.loc[k, full])
    r_oof = np.log(oof[:, 0] / oof[:, 1]); thr = (float(np.quantile(r_oof, .2)), float(np.quantile(r_oof, .8)))
    out = {"사양": dict(W=W, T=Tz, D=DM, 피쳐=len(full), 문턱_r=thr), "학습": {"사건": len(tr), "C·R·N": [int((ctr == q).sum()) for q in (0, 1, 2)], "교차_방향_AUC": float(roc_auc_score(ctr[ctr < 2] == 0, r_oof[ctr < 2]))}}
    for nm, s_ in (("검증", va), ("시험", te)):
        e = hilo_eval(s_, _p3(m, s_[full]), _p3(mb, s_[base]), thr, rng, ycol); cls = _cls(s_[ycol].to_numpy()); res = cls < 2
        for an, cols in (("강도만", SIG + inten), ("맥락만", SIG + CTX)):                                # 떼어 보기(판정 아님)
            pa = _p3(_lr().fit(tr[cols], ctr), s_[cols]); e[f"방향_AUC_{an}"] = float(roc_auc_score(cls[res] == 0, np.log(pa[:, 0] / pa[:, 1])[res]))
        r = np.log(_p3(m, s_[full])[:, 0] / _p3(m, s_[full])[:, 1])
        e["위·아래 돌파 방향_AUC"] = {k_: float(roc_auc_score(cls[res & (s_.side == v).to_numpy()] == 0, r[res & (s_.side == v).to_numpy()])) for k_, v in (("위", 1), ("아래", -1))}
        e["월별(결판 수·방향 AUC)"] = {mm: [int(g.sum()), round(float(roc_auc_score(cls[g] == 0, r[g])), 3) if 0 < (cls[g] == 0).sum() < g.sum() else None] for mm in sorted(s_.ts.dt.to_period("M").astype(str).unique()) for g in [res & (s_.ts.dt.to_period("M").astype(str) == mm).to_numpy()]}
        uni = {c: float(roc_auc_score(cls[res] == 0, s_[c].to_numpy()[res])) for c in full if np.isfinite(s_[c].to_numpy()[res]).all()}
        e["단변량_방향_AUC(치우침 상위 6)"] = {c: round(v, 3) for c, v in sorted(uni.items(), key=lambda kv: -abs(kv[1] - .5))[:6]}
        e["누수_의심"] = bool(e["방향_AUC"] > 0.75 or max(abs(v - .5) for v in uni.values()) > 0.20); out[nm] = e
    out["판정"] = {"통과(시험 1~4)": out["시험"]["통과"], "통과_후보(검증·시험 점 추정)": bool(out["검증"]["후보_점추정"] and out["시험"]["후보_점추정"])}
    coef = m.named_steps["logisticregression"].coef_; out["계수(C−R, 표준화 단위)"] = {c: round(float(coef[0, i] - coef[1, i]), 3) for i, c in sorted(enumerate(full), key=lambda t: -abs(coef[0, t[0]] - coef[1, t[0]]))}
    (OUT / f"hilo_result_w{W}_t{Tz}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)); print(json.dumps(out, ensure_ascii=False, indent=1))


def selftest() -> None:
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); day0 = pd.Timestamp("2026-03-10"); s0 = int(day0.timestamp())
        for dd in range(3):                                            # 3일치 초 자료(일부 초는 체결 없음)
            n = 86400; s = s0 + dd * 86400 + np.arange(n); keep = rng.random(n) < 0.8; keep[0] = True
            p = 2000 * np.exp(np.cumsum(rng.normal(0, 2e-5, n))) if dd != 1 else None
            if dd == 1:
                p = np.full(n, 2000.0); p[40000:40010] = np.linspace(2000, 2003, 10); p[40010:] = 2003.0; p[40100:] = 2010.0     # 터치(2003) → 계속 상승
            bq, sq = rng.uniform(0, 2, n), rng.uniform(0, 2, n)
            pd.DataFrame({"s": s, "po": p, "ph": p, "pl": p, "pc": p, "bq": bq, "sq": sq, "nb": 1, "ns": 1, "nt": 2, "mq": 1.0, "q2": bq ** 2 + sq ** 2})[keep] \
                .to_parquet(root / f"ETHUSDT-{(day0 + pd.Timedelta(days=dd)).strftime('%Y-%m-%d')}.parquet")
        T = Tape(root); te = s0 + 86400 + 40200                        # 터치 봉 = [te−300, te) → 배열상 39900..40199, 터치 초 = 40009 근처
        s1h = 0.004; cbar = 2010.0; g = np.log(cbar / 2003.0) / s1h      # L = 2003 이 되게
        o = one_event(T, te, 1, s1h, g, 30)
        assert o is not None and abs(o["L"] - 2003.0) < 1e-6 and (s0 + 86400 + 40005) <= o["kpos"] <= (s0 + 86400 + 40012), o and o["kpos"]
        assert o["tdec"] == o["kpos"] + 30 and abs(o["P"] - 2003.0) < 1e-9
        assert o["y_300_0.25"] == 1.0 and o["ret_300"] > 20            # 결정 뒤 2010 으로 → 위 배리어(0.1%) 먼저
        assert o["pre_spd30"] > 0 and o["obs_pen"] >= 0 and 0 <= o["t_in_bar"] <= 1
        # 미래 불변: 결정 초 뒤 자료를 바꿔도 피쳐는 같다(라벨만 바뀐다)
        g2 = pd.read_parquet(root / f"ETHUSDT-{(day0 + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}.parquet"); m = g2.s > o["tdec"]
        g2.loc[m, ["po", "ph", "pl", "pc"]] = 1990.0; g2.loc[m, ["bq", "sq"]] = 99.0; g2.to_parquet(root / f"ETHUSDT-{(day0 + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}.parquet")
        o2 = one_event(Tape(root), te, 1, s1h, g, 30, tape_close=cbar)
        fk = [c for c in o if c.startswith(("pre_", "obs_", "t_in_bar"))]
        assert all(abs(o[c] - o2[c]) < 1e-12 for c in fk) and o2["y_300_0.25"] == 0.0, "피쳐가 결정 시각 뒤를 본다"
    # ── 고저 돌파 사건: 돌파 분 뒤를 바꿔도 사건·레벨·맥락 불변 · 결정 초 뒤 체결을 바꿔도 피쳐 불변 · 라벨 직접 대조
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); s0 = int(pd.Timestamp("2026-03-10").timestamp()); n = 3 * 86400
        p = 2000 * np.exp(np.cumsum(rng.normal(0, 3e-5, n))); p[2 * 86400 + 30000:] *= 1.0                         # 3일 무작위 보행
        bq, sq = rng.uniform(0, 2, n), rng.uniform(0, 2, n); allg = pd.DataFrame({"s": s0 + np.arange(n), "po": p, "ph": p * 1.00002, "pl": p * 0.99998, "pc": p, "bq": bq, "sq": sq, "nb": 1, "ns": 1, "nt": 2, "mq": 1.0, "q2": bq ** 2 + sq ** 2})
        def dump(g, r_):
            for dd in range(3):
                g[(g.s >= s0 + dd * 86400) & (g.s < s0 + (dd + 1) * 86400)].to_parquet(r_ / f"ETHUSDT-{(pd.Timestamp('2026-03-10') + pd.Timedelta(days=dd)).strftime('%Y-%m-%d')}.parquet")
        dump(allg, root); m = minute_bars(root, cache=root / "m.parquet"); ev = hilo_events(m)
        late = ev[(ev.mi * 60 >= s0 + 86400 + 7200) & (ev.mi * 60 < s0 + 2 * 86400 + 60000)]; assert len(late) >= 3, len(ev)
        e = late.iloc[len(late) // 2]; f = hilo_one(Tape(root), marr(m), int(e.mi), int(e.side), float(e.L), float(e.s1h)); assert f is not None
        i = int(e.mi) - int(m.index[0]); assert abs(e.L - (m.h.to_numpy()[i - LOOK:i].max() if e.side > 0 else m.l.to_numpy()[i - LOOK:i].min())) < 1e-9          # 레벨 = 직전 240분 고저
        g2 = allg.copy(); fut = g2.s > f["ksec"] + 5; g2.loc[fut, ["po", "ph", "pl", "pc"]] = g2.loc[fut, ["po", "ph", "pl", "pc"]] * 1.01; g2.loc[fut, ["bq", "sq"]] = 77.0
        root2 = Path(td) / "b"; root2.mkdir(); dump(g2, root2); m2 = minute_bars(root2, cache=root2 / "m.parquet"); ev2 = hilo_events(m2)
        e2 = ev2[(ev2.mi == e.mi) & (ev2.side == e.side)]; assert len(e2) == 1 and float(e2.is24.iloc[0]) == float(e.is24) and abs(float(e2.L.iloc[0]) - e.L) < 1e-9 and abs(float(e2.s1h.iloc[0]) - e.s1h) < 1e-15, "사건·레벨이 미래에 따라 바뀐다"
        f2 = hilo_one(Tape(root2), marr(m2), int(e.mi), int(e.side), float(e.L), float(e.s1h)); fk = [c for c in f if c.startswith(("pre_", "obs_")) and not c.endswith(("_1",))] + ["lage", "near", "rng4", "mom15", "mom60", "mom1d", "pull", "tb60", "tb240"]
        fk = [c for c in fk if c not in ("obs_imb1", "obs_vol1", "obs_pos1")] + ["obs_imb5", "obs_vol5", "obs_pos5"]
        assert f2["ksec"] == f["ksec"] and all(abs(f[c] - f2[c]) < 1e-9 for c in set(fk)), "피쳐가 결정 초 뒤를 본다"
        assert f2["y5_60"] == (1 if e.side > 0 else -1)                                                             # 결정 뒤 +1% → 위 배리어 먼저(위 돌파면 C)
    te = pd.DataFrame({"ts": pd.to_datetime("2026-06-01") + pd.to_timedelta(np.arange(600) * 3, unit="h"), "y": rng.choice([1, -1, 0], 600, p=[.25, .15, .6])})
    sig = np.where(te.y == 1, 1.0, np.where(te.y == -1, -1.0, 0.0)) + rng.normal(0, 1.0, 600)                       # 방향을 아는 점수 → 통과해야 한다
    mk = lambda z: np.clip(np.column_stack([.2 * np.exp(z), .2 * np.exp(-z), np.full(600, .6)]) / (.2 * np.exp(z) + .2 * np.exp(-z) + .6)[:, None], 1e-9, 1)
    good = hilo_eval(te, mk(sig), mk(np.zeros(600)), (-0.5, 0.5), np.random.default_rng(0), "y"); junk = hilo_eval(te, mk(rng.normal(0, 1, 600)), mk(np.zeros(600)), (-0.5, 0.5), np.random.default_rng(0), "y")
    assert good["2_방향"] and good["3_집단차"] and good["방향_AUC"] > 0.7 and not junk["2_방향"] and not junk["통과"], (good["방향_AUC"], junk["방향_AUC"])
    assert first_pass(np.array([1, 1.02, 1]), np.array([1, 1, 0.9]), 1.0, 0.01) == 1 and first_pass(np.array([1.0, 1.0]), np.array([0.98, 1]), 1.0, 0.01) == -1
    assert first_pass(np.array([1.02]), np.array([0.98]), 1.0, 0.01) == 0 and np.isnan(first_pass(np.array([1.0]), np.array([1.0]), 1.0, 0.01))
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["selftest", "feats", "explore", "minutes", "hilo", "runhilo"]); ap.add_argument("--W", type=int, nargs="*", default=[5, 30, 60])
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "feats":
        OUT.mkdir(parents=True, exist_ok=True)
        for W in a.W:
            d = build(W); d.to_parquet(OUT / f"events_w{W}.parquet"); print(W, len(d), "사건 · 피쳐 있는 사건", int(d.P.notna().sum()))
    elif a.cmd == "explore":
        explore()
    elif a.cmd == "minutes":
        m = minute_bars(); print(len(m), "분 ·", pd.to_datetime(m.index[[0, -1]] * 60, unit="s").astype(str).tolist(), "· 빈 분", int(m.c.isna().sum()))
    elif a.cmd == "hilo":
        d = build_hilo(); d.to_parquet(OUT / "hilo_events.parquet"); print(len(d), "사건 ·", d.ts.min(), "→", d.ts.max())
    elif a.cmd == "runhilo":
        run_hilo()
