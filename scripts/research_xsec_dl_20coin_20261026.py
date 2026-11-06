#!/usr/bin/env python3
"""20코인 «상대 강도» 4h 예측 — DL(GRU+피쳐 MLP+코인 간 어텐션) vs 기준선 B1~B4 (2026-10-26).

사전 등록: docs/prereg/xsec_dl_20coin_20261026.md (사양은 거기가 원본, 여기서 바꾸지 않는다).
단계:  --build   1시간봉·피쳐·라벨·시퀀스 → OUT/data.npz
       --checks  누수 트립와이어 + 장치 검증(심은 열 회수)  → OUT/checks.json
       --fit     기준선·DL 전부 학습, 검증으로만 선택, 시험 예측 저장(시험 성과는 계산 안 함)
       --judge   1회 판정(OUT/verdict.json 있으면 거부)
       --selftest 평가 함수 자체 점검
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PANEL = Path("/home/kbj20/crypto-scalping/data/binance_vision/panel")
OUT = Path(os.environ.get("XSEC_OUT", "/home/kbj20/crypto-scalping/tmp/xsec_dl_20261026"))   # 서버 GPU 실행 때만 바꿈
COINS = ["ADA", "APT", "ARB", "ATOM", "AVAX", "BNB", "BTC", "DOGE", "DOT", "ETH",
         "FIL", "INJ", "LINK", "LTC", "NEAR", "OP", "SOL", "SUI", "TRX", "XRP"]
SEEDS = [1961169912, 501218739, 418362838, 436606973, 2067129892]
CANDS = {"K1": (32, 1e-5), "K2": (64, 1e-5), "K3": (32, 1e-3)}
WINDOWS = {  # (학습 시작, 학습 끝, 검증 시작, 검증 끝, 시험 시작, 시험 끝) — 끝은 그날 포함
    "R1": ("2022-01-01", "2023-12-31", "2024-01-01", "2024-05-31", "2024-06-01", "2024-10-23"),
    "R2": ("2023-01-01", "2024-12-31", "2025-01-01", "2025-05-31", "2025-06-01", "2025-10-23"),
    "R3": ("2024-01-01", "2025-12-31", "2026-01-01", "2026-05-31", "2026-06-01", "2026-09-14"),
}
N_SEQ = 48
FEATS = ["r1", "r4", "r24", "r72", "r168", "r720", "lrv24", "lrv168", "lrv720", "rvratio",
         "vol24", "vol4", "trd24", "oi24", "oi168", "oiz", "tk4", "tk24", "topz", "glsz",
         "beta", "corr", "ldv30"]
COST = 4e-4
ANN = math.sqrt(6 * 365)


# ───────────────────────── build ─────────────────────────
def hourly(sym: str, d: pd.DataFrame | None = None) -> pd.DataFrame:
    """d = 패널과 같은 열의 5분봉(행 = 봉 시작). 없으면 패널 파일."""
    d = pd.read_parquet(PANEL / f"{sym}USDT.parquet").set_index("timestamp") if d is None else d
    g = d.resample("1h", label="left", closed="left")  # 행 H = 5분봉 H…H+55 → H+1h 에 확정
    return pd.DataFrame({
        "c": g["close"].last(), "h": g["high"].max(), "l": g["low"].min(),
        "v": g["volume"].sum(min_count=1), "n": g["trades"].sum(min_count=1),
        "tb": g["taker_buy_base"].sum(min_count=1), "oi": g["sum_open_interest"].last(),
        "top": g["sum_toptrader_long_short_ratio"].last(), "gls": g["count_long_short_ratio"].last()})


def z720(x: pd.DataFrame) -> pd.DataFrame:
    return (x - x.rolling(720, min_periods=360).mean()) / x.rolling(720, min_periods=360).std()


def build_hourly_frames(src=None, idx=None):
    H = {s: hourly(s, None if src is None else src[s]) for s in COINS}
    idx = pd.date_range("2022-01-01", "2026-09-14 23:00", freq="1h") if idx is None else idx
    W = {f: pd.DataFrame({s: H[s][f] for s in COINS}).reindex(idx) for f in H["BTC"].columns}
    lc = np.log(W["c"])
    lr = lc.diff()
    sig = lr.rolling(720, min_periods=720).std()
    F = {}
    for k in (1, 4, 24, 72, 168, 720):
        F[f"r{k}"] = (lc - lc.shift(k)) / (sig * math.sqrt(k))
    rv24, rv168 = lr.rolling(24).std(), lr.rolling(168).std()
    F["lrv24"], F["lrv168"], F["lrv720"] = np.log(rv24), np.log(rv168), np.log(sig)
    F["rvratio"] = np.log(rv24 / sig)
    vday = W["v"].rolling(720, min_periods=720).sum() / 30
    F["vol24"] = np.log(W["v"].rolling(24).sum() / vday)
    F["vol4"] = np.log(W["v"].rolling(4).sum() * 6 / vday)
    F["trd24"] = np.log(W["n"].rolling(24).sum() / (W["n"].rolling(720, min_periods=720).sum() / 30))
    loi = np.log(W["oi"].ffill(limit=3))
    F["oi24"], F["oi168"], F["oiz"] = loi - loi.shift(24), loi - loi.shift(168), z720(loi)
    F["tk4"] = W["tb"].rolling(4).sum() / W["v"].rolling(4).sum() - 0.5
    F["tk24"] = W["tb"].rolling(24).sum() / W["v"].rolling(24).sum() - 0.5
    F["topz"] = z720(np.log(W["top"].ffill(limit=3)))
    F["glsz"] = z720(np.log(W["gls"].ffill(limit=3)))
    b = lr["BTC"]
    bvar = b.rolling(168, min_periods=120).var()
    F["beta"] = pd.DataFrame({s: lr[s].rolling(168, min_periods=120).cov(b) / bvar for s in COINS})
    F["corr"] = pd.DataFrame({s: lr[s].rolling(168, min_periods=120).corr(b) for s in COINS})
    F["ldv30"] = np.log((W["c"] * W["v"]).rolling(720, min_periods=720).sum())
    # 시퀀스 채널(1시간봉 단위)
    xsm = lr.mean(axis=1)
    CH = [lr / sig, lr.sub(xsm, axis=0) / sig,
          np.log(W["v"] / W["v"].rolling(720, min_periods=720).mean()),
          W["tb"] / W["v"] - 0.5, loi.diff() * 100, (W["h"] - W["l"]) / W["c"] / sig]
    return idx, lc, sig, F, CH


def xs_rank(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """시점별 횡단면 순위 → [−1,1]; 무효·NaN 은 NaN."""
    out = np.full(a.shape, np.nan, dtype=np.float32)
    for t in range(a.shape[0]):
        m = valid[t] & np.isfinite(a[t])
        n = m.sum()
        if n >= 2:
            r = rankdata(a[t, m])
            out[t, m] = (r - 1) / (n - 1) * 2 - 1
    return out


def build(shift: int = 0, src=None, idx=None, first="2022-01-02") -> dict:
    """shift=1 이면 피쳐를 1시간봉 하나 미래로 민 판(트립와이어 전용). src·idx·first = 전진 판정용 원천·격자·첫 결정."""
    idx, lc, sig, F, CH = build_hourly_frames(src, idx)
    pos = np.where((idx.hour % 4 == 0) & (idx >= pd.Timestamp(first)))[0]
    pos = pos[pos + 3 < len(idx)]
    fi = pos - 1 + shift                                     # 피쳐 행 = t−1h (봉 t−5분까지)
    LC = lc.to_numpy()
    lab = LC[pos + 3] - LC[pos - 1]                          # close(t+4h) − close(t)
    valid = np.isfinite(lab) & np.isfinite(sig.to_numpy()[pos - 1]) & np.isfinite(LC[pos - 1])
    y = lab - np.nanmean(np.where(valid, lab, np.nan), axis=1, keepdims=True)
    y = np.where(valid, y, np.nan)
    raw = np.stack([F[f].to_numpy()[fi] for f in FEATS], -1).astype(np.float32)   # [T,C,23]
    raw[~valid[..., None] | ~np.isfinite(raw)] = np.nan
    rk = np.stack([xs_rank(raw[..., j], valid) for j in range(len(FEATS))], -1)
    ch = np.stack([c.to_numpy() for c in CH], -1).astype(np.float32)               # [Hrs,C,6]
    win = fi[:, None] + np.arange(-N_SEQ + 1, 1)[None, :]                         # [T,48]
    seq = ch[win]                                                                  # [T,48,C,6]
    seq = np.transpose(seq, (0, 2, 1, 3))                                          # [T,C,48,6]
    seq[~np.isfinite(seq)] = np.nan
    keep = valid.sum(1) >= 5                                                       # 횡단면 5코인 미만 시점 제외
    pos, y, lab, valid, raw, rk, seq = pos[keep], y[keep], lab[keep], valid[keep], raw[keep], rk[keep], seq[keep]
    return dict(ts=idx[pos].values.astype("datetime64[ns]").astype(np.int64), y=y.astype(np.float32),
                lab=np.where(valid, lab, np.nan).astype(np.float32), valid=valid,
                X=np.concatenate([raw, rk], -1), seq=seq.astype(np.float32))


# ───────────────────────── 평가 ─────────────────────────
def ic_series(pred: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for t in range(len(y)):
        m = valid[t] & np.isfinite(pred[t])
        if m.sum() >= 5 and np.ptp(pred[t, m]) > 0:
            a, b = rankdata(pred[t, m]), rankdata(y[t, m])
            out[t] = np.corrcoef(a, b)[0, 1]
    return out


def block_ci(x: np.ndarray, block: int = 30, nb: int = 2000, seed: int = 0):
    x = x[np.isfinite(x)]
    n = len(x)
    rng = np.random.default_rng(seed)
    k = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(nb, k))
    sample = x[(starts[..., None] + np.arange(block)).reshape(nb, -1)[:, :n]]
    m = sample.mean(1)
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def longshort(pred: np.ndarray, lab: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """시점별 순수익(비용 후). 상위 20% 롱·하위 20% 숏, 다리 합 1, 비용 4bp×Σ|Δw|."""
    T, C = pred.shape
    w_prev = np.zeros(C)
    net = np.full(T, np.nan)
    R = np.expm1(np.nan_to_num(lab))
    for t in range(T):
        m = valid[t] & np.isfinite(pred[t])
        w = np.zeros(C)
        n = m.sum()
        if n >= 5:
            k = max(1, int(round(0.2 * n)))
            ids = np.where(m)[0][np.argsort(pred[t, m], kind="stable")]
            w[ids[-k:]] += 1 / k
            w[ids[:k]] -= 1 / k
        net[t] = (w * R[t]).sum() - COST * np.abs(w - w_prev).sum()
        w_prev = w
    return net


def ls_stats(net: np.ndarray) -> dict:
    net = net[np.isfinite(net)]
    cum = np.cumsum(net)
    return dict(sharpe=float(net.mean() / net.std() * ANN), total_bp=float(cum[-1] * 1e4),
                mdd_bp=float((cum - np.maximum.accumulate(np.r_[0, cum])[1:]).min() * 1e4), n=int(len(net)))


def selftest():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(400, 20))
    v = np.ones_like(y, bool)
    assert abs(np.nanmean(ic_series(y, y, v)) - 1) < 1e-9
    assert abs(np.nanmean(ic_series(-y, y, v)) + 1) < 1e-9
    p = y + rng.normal(size=y.shape) * 3
    m, lo, hi = block_ci(ic_series(p, y, v))
    assert lo < m < hi and 0.15 < m < 0.45, (m, lo, hi)
    # 완벽한 점수의 롱숏 > 0, 같은 점수 유지 시 회전 0 → 비용은 첫 진입(Σ|w|=2)만
    const = np.tile(np.arange(20.0), (5, 1))
    net = longshort(const, np.zeros((5, 20)), np.ones((5, 20), bool))
    assert np.allclose(net, [-2 * COST, 0, 0, 0, 0]), net
    assert ls_stats(longshort(y, y, v))["sharpe"] > 5
    print("selftest OK")


# ───────────────────────── 분할 ─────────────────────────
def split_idx(ts: np.ndarray, start: str, end: str) -> np.ndarray:
    a = pd.Timestamp(start).value
    b = (pd.Timestamp(end) + pd.Timedelta("1D")).value
    return np.where((ts >= a) & (ts + 4 * 3600 * 10**9 <= b))[0]


def norm_stats(X: np.ndarray, valid: np.ndarray, tr: np.ndarray):
    sub = X[tr][valid[tr]]
    mu, sd = np.nanmean(sub, 0), np.nanstd(sub, 0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1
    return np.nan_to_num(mu), sd


def apply_norm(X, mu, sd):
    return np.clip(np.nan_to_num((X - mu) / sd), -5, 5).astype(np.float32)


def target(y):
    s = np.nanstd(y, axis=1, keepdims=True)
    return np.clip(np.nan_to_num(y / np.where(s > 0, s, 1)), -3, 3).astype(np.float32)


# ───────────────────────── DL ─────────────────────────
def make_net(h, F):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(6, h, batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(F, h), nn.ReLU(), nn.Dropout(0.1))
            self.mix = nn.Linear(2 * h, h)
            self.att = nn.TransformerEncoderLayer(h, 4, 2 * h, 0.1, batch_first=True)
            self.out = nn.Linear(h, 1)

        def forward(self, s, x, m):
            B, C = x.shape[:2]
            _, hn = self.gru(s.reshape(B * C, N_SEQ, 6))
            z = self.mix(torch.cat([hn[-1].reshape(B, C, h), self.mlp(x)], -1))
            return self.out(self.att(z, src_key_padding_mask=~m)).squeeze(-1)
    return Net()


def dl_job(args):
    """args = (창, 후보, 시드, 심기[, 가중치 저장 폴더]). 폴더를 주면 dl/ 대신 거기에 예측+가중치+정규화 상수를 쓴다."""
    win, cand, seed, plant, *wdir = args
    import torch
    import torch.nn as nn
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    D = np.load(OUT / "data.npz")
    ts, y, valid = D["ts"], D["y"], D["valid"]
    X = D["X"]
    if plant:
        X = np.concatenate([X, D["plant"][..., None]], -1)
    S = D["seq"]
    w = WINDOWS[win]
    tr, va, te = split_idx(ts, w[0], w[1]), split_idx(ts, w[2], w[3]), split_idx(ts, w[4], w[5])
    mu, sd = norm_stats(X, valid, tr)
    Xn = apply_norm(X, mu, sd)
    smu, ssd = norm_stats(S[:, :, -1, :], valid, tr)
    Sn = apply_norm(S, smu, ssd)
    Y = target(y)
    h, wd = CANDS[cand]
    F = Xn.shape[-1]

    net = make_net(h, F)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))

    def predict(ii):
        net.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(ii), 256):
                b = ii[k:k + 256]
                out.append(net(T(Sn[b]), T(Xn[b]), T(valid[b])).numpy())
        return np.concatenate(out)

    best, best_ic, best_ep, bad = None, -9, -1, 0
    rng = np.random.default_rng(seed)
    for ep in range(30):
        net.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 32)):
            m = T(valid[b])
            p = net(T(Sn[b]), T(Xn[b]), m)
            loss = ((p - T(Y[b])) ** 2)[m].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        vic = float(np.nanmean(ic_series(predict(va), y[va], valid[va])))
        if vic > best_ic:
            best_ic, best_ep, bad = vic, ep, 0
            best = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(best)
    tag = f"{'plant_' if plant else ''}{win}_{cand}_{seed}"
    if wdir:
        torch.save({"state": best, "h": h, "F": F, "mu": mu, "sd": sd, "smu": smu, "ssd": ssd}, Path(wdir[0]) / f"{tag}.pt")
        np.savez(Path(wdir[0]) / f"{tag}.npz", va=predict(va), te=predict(te), best_ep=best_ep, val_ic=best_ic)
    else:
        np.savez(OUT / "dl" / f"{tag}.npz", va=predict(va), te=predict(te), best_ep=best_ep, val_ic=best_ic)
    return tag, best_ep, best_ic


# ───────────────────────── 기준선 ─────────────────────────
def fit_baselines(D, X=None, wins=WINDOWS, save=True):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    ts, y, valid = D["ts"], D["y"], D["valid"]
    X = D["X"] if X is None else X
    Y = target(y)
    j1, j24 = FEATS.index("r168"), FEATS.index("r24")
    res = {}
    for win, w in wins.items():
        tr, va, te = split_idx(ts, w[0], w[1]), split_idx(ts, w[2], w[3]), split_idx(ts, w[4], w[5])
        mu, sd = norm_stats(X, valid, tr)
        Xn = apply_norm(X, mu, sd)
        vic = lambda p: float(np.nanmean(ic_series(p, y[va], valid[va])))
        out = {"B1": {"te": X[te][..., j1]}, "B2": {"te": -X[te][..., j24]}}
        mtr = valid[tr]
        Xtr, Ytr = Xn[tr][mtr], Y[tr][mtr]
        shape = lambda ii, f: f(Xn[ii].reshape(-1, Xn.shape[-1])).reshape(len(ii), 20)
        cands = []
        for a in (1, 10, 100, 1000):
            r = Ridge(alpha=a).fit(Xtr, Ytr)
            cands.append((vic(shape(va, r.predict)), a, r))
        s, a, r = max(cands, key=lambda c: c[0])
        out["B3"] = {"te": shape(te, r.predict), "val_ic": s, "param": a, "va": shape(va, r.predict)}
        cands = []
        for leaf, it in ((15, 200), (15, 500), (31, 200), (31, 500)):
            g = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=leaf, max_iter=it,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              early_stopping=False, random_state=0).fit(Xtr, Ytr)
            cands.append((vic(shape(va, g.predict)), (leaf, it), g))
        s, prm, g = max(cands, key=lambda c: c[0])
        out["B4"] = {"te": shape(te, g.predict), "val_ic": s, "param": prm, "va": shape(va, g.predict)}
        res[win] = out
        print(f"  [{win}] B3 α={out['B3']['param']} val IC {out['B3']['val_ic']:+.4f} · "
              f"B4 {out['B4']['param']} val IC {out['B4']['val_ic']:+.4f}", flush=True)
        if save:
            np.savez(OUT / f"base_{win}.npz", **{f"{k}_te": v["te"] for k, v in out.items()},
                     meta=json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("val_ic", "param")}
                                      for k, v in out.items()}))
    return res


# ───────────────────────── 단계 ─────────────────────────
def cmd_build():
    OUT.mkdir(parents=True, exist_ok=True)
    D = build(0)
    rng = np.random.default_rng(20261026)
    Y = target(D["y"])
    plant = np.where(D["valid"], 0.1 * Y / max(Y[D["valid"]].std(), 1e-9) + math.sqrt(0.99) * rng.normal(size=Y.shape), np.nan)
    np.savez(OUT / "data.npz", **D, plant=plant.astype(np.float32))
    v = D["valid"]
    print(f"T={len(D['ts'])} 시점 · 코인·시점 {int(v.sum()):,} · "
          f"{pd.Timestamp(D['ts'][0])} ~ {pd.Timestamp(D['ts'][-1])} · 시점당 코인 {v.sum(1).min()}~{v.sum(1).max()}")


def cmd_checks():
    D = dict(np.load(OUT / "data.npz"))
    ts, y, valid = D["ts"], D["y"], D["valid"]
    ii = split_idx(ts, "2022-01-01", "2024-05-31")
    rep = {"single_ic_current": {}}
    for j, f in enumerate(FEATS):
        rep["single_ic_current"][f] = float(np.nanmean(ic_series(D["X"][ii][..., j], y[ii], valid[ii])))
    mx = max(rep["single_ic_current"].items(), key=lambda kv: abs(kv[1]))
    rep["max_abs_single_ic"] = mx
    print("단일 피쳐 IC(2022-01~2024-05):", {k: round(v, 4) for k, v in rep["single_ic_current"].items()})
    if abs(mx[1]) > 0.3:
        print(f"🔴 단일 피쳐 |IC|>0.3: {mx} → 멈춤")
        json.dump(rep, open(OUT / "checks.json", "w"), indent=1)
        sys.exit(2)
    # (a) 1시간봉 한 칸 미래로 민 판
    Ds = build(1)
    j = FEATS.index("r1")
    rep["r1_ic_shift1h"] = float(np.nanmean(ic_series(Ds["X"][ii][..., j], y[ii], valid[ii])))
    # 5분봉 한 칸 민 1h 수익
    c5 = {s: pd.read_parquet(PANEL / f"{s}USDT.parquet", columns=["timestamp", "close"]).set_index("timestamp")["close"]
          for s in COINS}
    C5 = np.log(pd.DataFrame(c5))
    t = pd.to_datetime(ts[ii])
    def r1_5m(off):
        a = C5.reindex(t - pd.Timedelta("5min") + pd.Timedelta(minutes=5 * off)).to_numpy()
        b = C5.reindex(t - pd.Timedelta("65min") + pd.Timedelta(minutes=5 * off)).to_numpy()
        return a - b
    rep["r1_ic_5m_current"] = float(np.nanmean(ic_series(r1_5m(0), y[ii], valid[ii])))
    rep["r1_ic_5m_shift1bar"] = float(np.nanmean(ic_series(r1_5m(1), y[ii], valid[ii])))
    w1 = {"R1": WINDOWS["R1"]}
    rep["ridge_val_ic_current"] = fit_baselines(D, wins=w1, save=False)["R1"]["B3"]["val_ic"]
    rep["ridge_val_ic_shift1h"] = fit_baselines(Ds, wins=w1, save=False)["R1"]["B3"]["val_ic"]
    print({k: v for k, v in rep.items() if k != "single_ic_current"})
    # 장치 검증: 심은 열 단독 IC vs DL(K1, 첫 시드) 시험 IC
    (OUT / "dl").mkdir(exist_ok=True)
    rep["plant"] = {}
    with ProcessPoolExecutor(3) as ex:
        list(ex.map(dl_job, [(w, "K1", SEEDS[0], True) for w in WINDOWS]))
    for w, s in WINDOWS.items():
        te = split_idx(ts, s[4], s[5])
        p = np.load(OUT / "dl" / f"plant_{w}_K1_{SEEDS[0]}.npz")["te"]
        rep["plant"][w] = {"plant_col_ic": float(np.nanmean(ic_series(D["plant"][te], y[te], valid[te]))),
                           "dl_ic": float(np.nanmean(ic_series(p, y[te], valid[te])))}
    print("장치 검증:", rep["plant"])
    json.dump(rep, open(OUT / "checks.json", "w"), indent=1)


def cmd_fit():
    D = dict(np.load(OUT / "data.npz"))
    print("기준선 학습…", flush=True)
    fit_baselines(D)
    (OUT / "dl").mkdir(exist_ok=True)
    jobs = [(w, c, s, False) for w in WINDOWS for c in CANDS for s in SEEDS
            if not (OUT / "dl" / f"{w}_{c}_{s}.npz").exists()]
    print(f"DL {len(jobs)}개 학습…", flush=True)
    with ProcessPoolExecutor(5) as ex:
        for tag, ep, vic in ex.map(dl_job, jobs):
            print(f"  {tag} best_ep {ep} val IC {vic:+.4f}", flush=True)
    # 검증으로만 후보 선택(5시드 앙상블 검증 IC)
    ts, y, valid = D["ts"], D["y"], D["valid"]
    sel = {}
    for w, s in WINDOWS.items():
        va = split_idx(ts, s[2], s[3])
        scores = {c: float(np.nanmean(ic_series(np.mean([np.load(OUT / "dl" / f"{w}_{c}_{sd}.npz")["va"] for sd in SEEDS], 0),
                                                y[va], valid[va]))) for c in CANDS}
        sel[w] = {"chosen": max(scores, key=scores.get), "val_ic_ensemble": scores}
    print("선택:", sel)
    json.dump(sel, open(OUT / "dl_selection.json", "w"), indent=1)


def cmd_judge():
    vp = OUT / "verdict.json"
    if vp.exists():
        sys.exit(f"판정은 1회만 — {vp} 이미 있음")
    D = dict(np.load(OUT / "data.npz"))
    ts, y, lab, valid = D["ts"], D["y"], D["lab"], D["valid"]
    sel = json.load(open(OUT / "dl_selection.json"))
    ics, nets, rep = {}, {}, {"windows": {}}
    seed_ic = {}
    for w, s in WINDOWS.items():
        te = split_idx(ts, s[4], s[5])
        b = np.load(OUT / f"base_{w}.npz")
        c = sel[w]["chosen"]
        per = [np.load(OUT / "dl" / f"{w}_{c}_{sd}.npz")["te"] for sd in SEEDS]
        preds = {k: b[f"{k}_te"] for k in ("B1", "B2", "B3", "B4")}
        preds["DL"] = np.mean(per, 0)
        seed_ic[w] = [float(np.nanmean(ic_series(p, y[te], valid[te]))) for p in per]
        rep["windows"][w] = {}
        for k, p in preds.items():
            ic = ic_series(p, y[te], valid[te])
            net = longshort(p, lab[te], valid[te])
            ics.setdefault(k, []).append(ic)
            nets.setdefault(k, []).append(net)
            rep["windows"][w][k] = {"ic": float(np.nanmean(ic)), **ls_stats(net)}
    allic = {k: np.concatenate(v) for k, v in ics.items()}
    allnet = {k: np.concatenate(v) for k, v in nets.items()}
    rep["combined"] = {k: {"ic_ci": block_ci(allic[k]), **ls_stats(allnet[k])} for k in allic}
    base = ["B1", "B2", "B3", "B4"]
    best_ic_b = max(base, key=lambda k: np.nanmean(allic[k]))
    best_sh_b = max(base, key=lambda k: rep["combined"][k]["sharpe"])
    diff = allic["DL"] - allic[best_ic_b]
    rep["dl_minus_best"] = {"best_ic_baseline": best_ic_b, "diff_ci": block_ci(diff),
                            "best_sharpe_baseline": best_sh_b}
    rep["seed_test_ic"] = seed_ic
    X1 = rep["combined"]["DL"]["ic_ci"][1] > 0
    X2 = rep["dl_minus_best"]["diff_ci"][0] > 0 and rep["dl_minus_best"]["diff_ci"][1] > 0
    X3 = rep["combined"]["DL"]["sharpe"] > 0 and rep["combined"]["DL"]["sharpe"] > rep["combined"][best_sh_b]["sharpe"]
    X4 = all(rep["windows"][w]["DL"]["ic"] > 0 for w in WINDOWS)
    rep["gates"] = {"X1": X1, "X2": X2, "X3": X3, "X4": X4, "PASS": X1 and X2 and X3 and X4}
    rep["dl_selection"] = sel
    json.dump(rep, open(vp, "w"), indent=1, default=float)
    for w in WINDOWS:
        print(w, {k: f"IC {v['ic']:+.4f} SR {v['sharpe']:+.2f} {v['total_bp']:+.0f}bp" for k, v in rep["windows"][w].items()})
    for k, v in rep["combined"].items():
        m, lo, hi = v["ic_ci"]
        print(f"합침 {k}: IC {m:+.4f} [{lo:+.4f},{hi:+.4f}] SR {v['sharpe']:+.2f} 총 {v['total_bp']:+.0f}bp MDD {v['mdd_bp']:.0f}bp")
    print("DL−최고:", rep["dl_minus_best"], "\n시드 IC:", seed_ic, "\n관문:", rep["gates"])


# ───────────────────────── 2단계: 포트폴리오 구성(회전 줄이기) ─────────────────────────
GRID2 = [(h, band) for h in (0, 3, 6, 12) for band in (False, True)]


def smooth_rank(pred, valid, h):
    r = xs_rank(np.asarray(pred, float), valid)
    if h:
        r = pd.DataFrame(r).ewm(halflife=h, ignore_na=True).mean().to_numpy()
    return np.where(valid, r, np.nan)


def weights2(score, valid, band):
    """상·하위 20% 동일가중. band 면 보유 코인은 30% 안이면 유지, 빈자리는 20% 안에서 채움."""
    T, C = score.shape
    W = np.zeros((T, C))
    pl, ps = [], []
    for t in range(T):
        m = valid[t] & np.isfinite(score[t])
        n = m.sum()
        if n < 5:
            pl, ps = [], []
            continue
        k, kb = max(1, int(round(0.2 * n))), max(1, int(round(0.2 * n)), int(round(0.3 * n)))
        top = list(np.where(m)[0][np.argsort(-score[t, m], kind="stable")])

        def pick(order, prev):
            keep = [c for c in order[:kb] if c in prev][:k] if band else []
            keep += [c for c in order[:k] if c not in keep][:k - len(keep)]
            return keep
        pl, ps = pick(top, pl), pick(top[::-1], ps)
        W[t, pl] += 1 / k
        W[t, ps] -= 1 / k
    return W


def pnl2(W, lab):
    R = np.expm1(np.nan_to_num(lab))
    gross = (W * R).sum(1)
    turn = np.abs(np.diff(np.vstack([np.zeros(W.shape[1]), W]), axis=0)).sum(1)
    return gross, turn


def sharpe(x):
    return float(x.mean() / x.std() * ANN)


def sharpe_ci(x, block=30, nb=2000, seed=0):
    n = len(x)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(nb, math.ceil(n / block)))
    smp = x[(starts[..., None] + np.arange(block)).reshape(nb, -1)[:, :n]]
    sh = smp.mean(1) / smp.std(1) * ANN
    return float(np.quantile(sh, .025)), float(np.quantile(sh, .975))


def cmd_prep2():
    """리지·HGB 검증 예측 재생성(결정적) — 시험 예측이 1단계 저장본과 같은지 먼저 확인."""
    D = dict(np.load(OUT / "data.npz"))
    res = fit_baselines(D, save=False)
    for w in WINDOWS:
        b = np.load(OUT / f"base_{w}.npz")
        for k in ("B3", "B4"):
            d = np.nanmax(np.abs(res[w][k]["te"] - b[f"{k}_te"]))
            assert d < 1e-6, (w, k, d)
        np.savez(OUT / f"base2_va_{w}.npz", B3=res[w]["B3"]["va"], B4=res[w]["B4"]["va"])
    print("prep2 OK — 시험 예측 재현 일치")


def cmd_judge2():
    vp = OUT / "verdict2.json"
    if vp.exists():
        sys.exit(f"판정은 1회만 — {vp} 이미 있음")
    D = dict(np.load(OUT / "data.npz"))
    ts, lab, valid = D["ts"], D["lab"], D["valid"]
    sel1 = json.load(open(OUT / "dl_selection.json"))
    rep = {"selected": {}, "windows": {}, "combined": {}}
    series = {}
    for w, s in WINDOWS.items():
        va, te = split_idx(ts, s[2], s[3]), split_idx(ts, s[4], s[5])
        c = sel1[w]["chosen"]
        dl = [np.load(OUT / "dl" / f"{w}_{c}_{sd}.npz") for sd in SEEDS]
        b, b2 = np.load(OUT / f"base_{w}.npz"), np.load(OUT / f"base2_va_{w}.npz")
        P = {"DL": (np.mean([x["va"] for x in dl], 0), np.mean([x["te"] for x in dl], 0)),
             "B3": (b2["B3"], b["B3_te"]), "B4": (b2["B4"], b["B4_te"])}
        rep["windows"][w] = {}
        for k, (pva, pte) in P.items():
            val = {}
            for h, band in GRID2:
                g, tu = pnl2(weights2(smooth_rank(pva, valid[va], h), valid[va], band), lab[va])
                val[(h, band)] = sharpe(g - 4e-4 * tu)
            h, band = max(val, key=val.get)
            rep["selected"].setdefault(k, {})[w] = {"halflife": h, "band": band, "val_sharpe_4bp": val[(h, band)]}
            g, tu = pnl2(weights2(smooth_rank(pte, valid[te], h), valid[te], band), lab[te])
            series.setdefault(k, []).append((g, tu))
            rep["windows"][w][k] = {"net4_sharpe": sharpe(g - 4e-4 * tu), "net2_sharpe": sharpe(g - 2e-4 * tu),
                                    "gross_sharpe": sharpe(g), "gross_bp_4h": float(g.mean() * 1e4),
                                    "turnover_4h": float(tu.mean()), **{f"net4_{kk}": vv for kk, vv in ls_stats(g - 4e-4 * tu).items()}}
    for k, lst in series.items():
        g = np.concatenate([x[0] for x in lst])
        tu = np.concatenate([x[1] for x in lst])
        n4 = g - 4e-4 * tu
        rep["combined"][k] = {"net4_sharpe": sharpe(n4), "net4_sharpe_ci": sharpe_ci(n4),
                              "net2_sharpe": sharpe(g - 2e-4 * tu), "gross_sharpe": sharpe(g),
                              "gross_bp_4h": float(g.mean() * 1e4), "turnover_4h": float(tu.mean()),
                              "breakeven_cost_bp": float(g.mean() / tu.mean() * 1e4),
                              **{f"net4_{kk}": vv for kk, vv in ls_stats(n4).items()},
                              **{f"net2_{kk}": vv for kk, vv in ls_stats(g - 2e-4 * tu).items()}}
    C = rep["combined"]
    Y1 = C["DL"]["net4_sharpe"] > 0 and C["DL"]["net4_sharpe_ci"][0] > 0
    Y2 = C["DL"]["net4_sharpe"] > max(C["B3"]["net4_sharpe"], C["B4"]["net4_sharpe"])
    Y3 = sum(rep["windows"][w]["DL"]["net4_sharpe"] > 0 for w in WINDOWS) >= 2
    rep["gates"] = {"Y1": Y1, "Y2": Y2, "Y3": Y3, "PASS": Y1 and Y2 and Y3}
    json.dump(rep, open(vp, "w"), indent=1, default=float)
    print("선택:", rep["selected"])
    for w in WINDOWS:
        print(w, {k: f"4bp SR {v['net4_sharpe']:+.2f} 2bp {v['net2_sharpe']:+.2f} 전 {v['gross_bp_4h']:+.2f}bp 회전 {v['turnover_4h']:.2f}"
                  for k, v in rep["windows"][w].items()})
    for k, v in C.items():
        print(f"합침 {k}: 4bp SR {v['net4_sharpe']:+.2f} {v['net4_sharpe_ci']} 총 {v['net4_total_bp']:+.0f}bp MDD {v['net4_mdd_bp']:.0f} · "
              f"2bp SR {v['net2_sharpe']:+.2f} 총 {v['net2_total_bp']:+.0f} · 전 SR {v['gross_sharpe']:+.2f} {v['gross_bp_4h']:+.2f}bp/4h · 회전 {v['turnover_4h']:.2f} · 분기 {v['breakeven_cost_bp']:.2f}bp")
    print("관문:", rep["gates"])


def selftest2():
    rng = np.random.default_rng(2)
    sc = rng.normal(size=(300, 20))
    lab = rng.normal(size=(300, 20)) * 0.01
    v = np.ones_like(sc, bool)
    # 전체·h0 = 1단계 longshort 와 같은 손익
    g, tu = pnl2(weights2(smooth_rank(sc, v, 0), v, False), lab)
    assert np.allclose(g - COST * tu, longshort(sc, lab, v))
    # 띠는 회전을 줄이고, 매 시점 롱 4·숏 4 유지
    W = weights2(smooth_rank(sc, v, 0), v, True)
    assert pnl2(W, lab)[1].mean() < tu.mean()
    assert ((W > 0).sum(1) == 4).all() and ((W < 0).sum(1) == 4).all()
    # 보유가 30% 안이면 유지: 순위 5위로 내려간 롱은 남는다
    s2 = np.tile(np.arange(20.0), (2, 1)); s2[1, 19], s2[1, 15] = 15, 19.5
    W2 = weights2(s2, np.ones((2, 20), bool), True)
    assert W2[1, 19] > 0
    print("selftest2 OK")


def cmd_freeze_r3():
    """전진용 동결: 1단계 동결 예측(dl/R3_K1_*.npz)은 가중치를 담지 않는다 → 같은 시드·설정으로 다시 학습해 가중치를 남기고,
    다시 만든 검증·시험 예측이 동결 예측과 1e-6 안에서 같을 때만 통과. 리지 B3 R3 계수도 남기고 base_R3 B3_te 재현 확인."""
    from sklearn.linear_model import Ridge
    wd = OUT / "frozen_r3"
    wd.mkdir(exist_ok=True)
    with ProcessPoolExecutor(5) as ex:
        list(ex.map(dl_job, [("R3", "K1", sd, False, str(wd)) for sd in SEEDS]))
    rep = {}
    for sd in SEEDS:
        a, b = np.load(OUT / "dl" / f"R3_K1_{sd}.npz"), np.load(wd / f"R3_K1_{sd}.npz")
        rep[sd] = max(float(np.nanmax(np.abs(a[k] - b[k]))) for k in ("va", "te"))
    print("DL 재학습 vs 동결 예측 최대 차:", rep)
    assert max(rep.values()) < 1e-6, "재학습이 동결 예측을 재현 못 함 — 가중치 동결 불가"
    D = dict(np.load(OUT / "data.npz"))
    ts, y, valid, X = D["ts"], D["y"], D["valid"], D["X"]
    w = WINDOWS["R3"]
    tr, te = split_idx(ts, w[0], w[1]), split_idx(ts, w[4], w[5])
    mu, sd_ = norm_stats(X, valid, tr)
    Xn = apply_norm(X, mu, sd_)
    alpha = json.loads(str(np.load(OUT / "base_R3.npz")["meta"]))["B3"]["param"]
    r = Ridge(alpha=alpha).fit(Xn[tr][valid[tr]], target(y)[tr][valid[tr]])
    pte = r.predict(Xn[te].reshape(-1, Xn.shape[-1])).reshape(len(te), 20)
    d = float(np.nanmax(np.abs(pte - np.load(OUT / "base_R3.npz")["B3_te"])))
    print(f"리지 α={alpha} 재적합 vs base_R3 최대 차 {d:.2e}")
    assert d < 1e-6
    np.savez(wd / "ridge_R3.npz", coef=r.coef_, intercept=r.intercept_, alpha=alpha, mu=mu, sd=sd_)
    import hashlib
    lines = [f"{p.name} {hashlib.md5(p.read_bytes()).hexdigest()}" for p in sorted(wd.glob("*.pt")) + [wd / "ridge_R3.npz"]]
    (wd / "frozen_r3_md5.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("build", "checks", "fit", "judge", "selftest", "selftest2", "prep2", "judge2", "freeze_r3"):
        ap.add_argument(f"--{a}", action="store_true")
    A = ap.parse_args()
    if A.selftest:
        selftest()
    if A.build:
        cmd_build()
    if A.checks:
        cmd_checks()
    if A.fit:
        cmd_fit()
    if A.judge:
        cmd_judge()
    if A.selftest2:
        selftest2()
    if A.prep2:
        cmd_prep2()
    if A.judge2:
        cmd_judge2()
    if A.freeze_r3:
        cmd_freeze_r3()
