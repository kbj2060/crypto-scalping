#!/usr/bin/env python3
"""체결 정보 추출 A(체결 구성)·B(다른 거래소·현물)·D(흐름 기억·다음 흐름) — 사전등록 docs/prereg/tape_info_abd_20261115.md.

기준선 = 호가 연구의 ⓪ 가격·체결 23피쳐(frame_<COIN>.parquet 의 p_*). 그 위에 새 피쳐군 하나씩 더해 같은 LightGBM 으로
31칸(호가 연구 28칸 + 다음 흐름 3칸)을 맞히고, 행별 손실 차를 날짜 블록 부트스트랩으로 채점(research_ob_info_map_gbm 재사용).
결정 시각 τ = UTC 초 t 의 끝. 바이낸스 선물 테이프 피쳐는 ≤ t, 다른 거래소·현물은 ≤ t−1(기본, 수신 지연 보수) · 대상은 t+1 초부터.

사용: --feat COIN · --fit SPLIT ARM · --score SPLIT A B · --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OBIM = Path(os.environ.get("TAPEINFO_OBIM", "/home/kbj20/crypto-scalping/.claude/worktrees/orderbook-liquidity-delta-chart-6d3436/tmp/ob_info_map_20261115"))
os.environ.setdefault("OBIM_OUT", str(OBIM))
sys.path.insert(0, str(ROOT / "scripts"))
from research_ob_info_map_build_20261115 import COIN, DAY, MIRROR, _past_sum, _rel24, _fwd_ok, make_targets  # noqa: E402
import research_ob_info_map_gbm_20261115 as G  # noqa: E402

OUT = ROOT / "tmp/tape_info_20261115"
COINS = ("ETH", "BTC", "SOL", "XRP")
LAKES = (MIRROR / "lake", Path("/home/kbj20/crypto-scalping/tmp/obim_lake/lake"))   # 두 번째 = 호가 연구가 따로 받은 11-14 바이낸스 테이프
VENUE = {"bn": "binance/tape", "spot": "binance/spot_tape", "okx": "okx/tape", "byb": "bybit/tape"}
H_FLOW = (10, 60, 300)
FLOW = [(f"flow{h}", f"y_flow{h}", "reg") for h in H_FLOW]
G.TARGETS += FLOW                                                  # 채점 함수가 31칸을 돈다
# 구간 끝(미포함). final 시험 = 새 날짜(11-15~11-28), 11-14 18:00~24:00 은 어디에도 안 씀. dev = 호가 연구 분할(읽기 전용 보조).
SPLITS = {"final": [("train", "2026-11-04"), ("val", "2026-11-14T18:00:00"), ("gap", "2026-11-15"), ("test", "2026-11-29")],
          "dev": [("train", "2026-10-25"), ("val", "2026-11-04"), ("test", "2026-11-14T18:00:00")]}
# 크기 구간(고래·리테일) 경계가 바뀐 코인 — 바뀌기 전 행의 고래·리테일 피쳐는 결측(수집기 SIZE_BANDS_USD 주석: SOL·XRP 09-27 · BTC 10-31)
BAND_FROM = {"SOL": "2026-09-28", "XRP": "2026-09-28", "BTC": "2026-11-01"}
ARMS = {"P": ("p_ret", "p_lrv", "p_tod", "p_coin"), "0": ("p_",), "A": ("p_", "t_"), "B": ("p_", "x_"), "B0": ("p_", "x0_"),
        "D": ("p_", "s_"), "Z": ("p_", "z_"), "ALL": ("p_", "t_", "x_", "s_"), "1": ("p_", "b_"), "1A": ("p_", "b_", "t_")}
SEED = 20261115


def ts(s):
    return int(np.datetime64(s, "s").astype(np.int64))


# ── 초 배열 ──
def sec_arrays(coin, venue, t0, n):
    """lake 테이프(초 × 가격칸) → 초 배열. 명목 = 수량 × 칸 가운데 가격. 그 날 파일이 없으면 day_ok=0."""
    B = COIN[coin][2]
    A = {c: np.zeros(n) for c in ("tb", "ts", "wb", "ws", "rb", "rs", "nb", "ns", "ob", "os", "kb", "ks", "q", "spn")}
    A["mb"], A["ms"] = np.zeros(n), np.zeros(n)
    A["bf"], A["day_ok"] = np.zeros(n, bool), np.zeros(n, bool)
    hi, lo = np.full(n, -np.inf), np.full(n, np.inf)
    parts = []
    for d in range(t0 // DAY * DAY, t0 + n, DAY):
        D = np.datetime64(d, "s").astype(str)[:10]
        fs = next((f for f in (sorted((L / f"{VENUE[venue]}/coin={coin}/date={D}").glob("*.parquet")) for L in LAKES) if f), [])
        if not fs:
            continue
        A["day_ok"][max(d - t0, 0):max(min(d + DAY - t0, n), 0)] = True
        parts += [pd.read_parquet(f) for f in fs]
    if not parts:
        return A
    T = pd.concat(parts)
    k = T.ts_sec.values.astype(np.int64) - t0
    m = (k >= 0) & (k < n)
    T, k = T[m], k[m]
    ctr = T.price_bin.values * B
    bc = lambda w: np.bincount(k, weights=np.nan_to_num(np.asarray(w, float)), minlength=n)
    A["tb"], A["ts"] = bc(T.buy_qty * ctr), bc(T.sell_qty * ctr)
    A["wb"], A["ws"] = bc(T.whale_buy_qty * ctr), bc(T.whale_sell_qty * ctr)
    A["rb"], A["rs"] = bc(T.retail_buy_qty * ctr), bc(T.retail_sell_qty * ctr)
    A["nb"], A["ns"], A["ob"], A["os"] = bc(T.buy_n), bc(T.sell_n), bc(T.order_buy_n), bc(T.order_sell_n)
    A["q"] = bc(T.buy_qty + T.sell_qty)
    A["kb"], A["ks"] = bc(T.buy_qty > 0), bc(T.sell_qty > 0)
    np.maximum.at(A["mb"], k, np.nan_to_num(T.buy_max.values * ctr))
    np.maximum.at(A["ms"], k, np.nan_to_num(T.sell_max.values * ctr))
    A["bf"][k[T.buy_max.isna().values | T.sell_max.isna().values]] = True      # REST 복구 분: 최대 = NULL · 주문 = aggTrade(뜻이 다름)
    np.maximum.at(hi, k, ctr); np.minimum.at(lo, k, ctr)
    v = A["q"] > 0
    A["spn"][v] = (hi[v] - lo[v]) / ((A["tb"][v] + A["ts"][v]) / A["q"][v]) * 1e4   # 그 초 체결 가격 폭(bp)
    return A


def _ratio(a, b, fill=np.nan):
    return np.where(b > 0, a / np.where(b > 0, b, 1), fill)


def _imb(a, b):
    return _ratio(a - b, a + b, 0.0)


def _roll_max(x, k):
    return pd.Series(x).rolling(k, min_periods=1).max().values


def _lag(x, L):
    return x if L == 0 else np.r_[np.full(L, np.nan), x[:-L]]


def feats_A(S):
    """체결 구성 — 바이낸스 선물 ≤ t. REST 복구 초(bf)는 구성 피쳐 분자·분모에서 뺀다."""
    cm = (~S["bf"]).astype(float)
    F = {}
    for k in (10, 60, 300):
        P = lambda x: _past_sum(np.asarray(x) * cm, k)
        tot, w, r = P(S["tb"] + S["ts"]), (P(S["wb"]), P(S["ws"])), (P(S["rb"]), P(S["rs"]))
        o, nf = (P(S["ob"]), P(S["os"])), (P(S["nb"]), P(S["ns"]))
        sb, ss = P(S["tb"]), P(S["ts"])
        F[f"t_whsh{k}"], F[f"t_rtsh{k}"] = _ratio(w[0] + w[1], tot), _ratio(r[0] + r[1], tot)
        F[f"t_whimb{k}"], F[f"t_rtimb{k}"] = _imb(*w), _imb(*r)
        F[f"t_osz{k}"] = _rel24(np.log(_ratio(tot, o[0] + o[1], np.nan) + 1))
        F[f"t_oszimb{k}"] = np.log(_ratio(sb, o[0], np.nan) + 1) - np.log(_ratio(ss, o[1], np.nan) + 1)
        F[f"t_sweep{k}"] = np.log(_ratio(nf[0] + nf[1], o[0] + o[1]))
        F[f"t_swimb{k}"] = np.log(_ratio(nf[0], o[0])) - np.log(_ratio(nf[1], o[1]))
        mb, ms = _roll_max(S["mb"] * cm, k), _roll_max(S["ms"] * cm, k)
        F[f"t_max{k}"] = _rel24(np.log1p(np.maximum(mb, ms)))
        F[f"t_maximb{k}"] = np.log1p(mb) - np.log1p(ms)
        hv = _past_sum((S["q"] > 0).astype(float), k)
        F[f"t_span{k}"] = _ratio(_past_sum(S["spn"], k), hv)
        F[f"t_kbin{k}"] = _ratio(_past_sum(S["kb"] + S["ks"], k), hv)
    return F


def feats_B(V, bn, mid, L, pre):
    """다른 거래소·현물 테이프 — ≤ t−L. 그 날 파일 없으면 결측."""
    F = {}
    for v, S in V.items():
        bad = ~_lag(S["day_ok"].astype(float), L).astype(bool) if L else ~S["day_ok"]
        g = lambda x: np.where(bad, np.nan, _lag(x, L))
        for k in (1, 10, 60):
            F[f"{pre}{v}_imb{k}"] = g(_imb(_past_sum(S["tb"], k), _past_sum(S["ts"], k)))
        F[f"{pre}{v}_act60"] = g(_rel24(np.log((_past_sum(S["tb"] + S["ts"], 60) + 1) / (_past_sum(bn["tb"] + bn["ts"], 60) + 1))))
        vw = _ratio(S["tb"] + S["ts"], S["q"])
        dev = (vw / mid - 1) * 1e4                                   # 그 초 그 거래소 체결 평균가 − 바이낸스 선물 중간가(bp)
        ok = np.isfinite(dev).astype(float); dz = np.nan_to_num(dev)
        base = _ratio(_past_sum(dz, 3600), _past_sum(ok, 3600))       # 지난 1시간 평균(수준 = 현물·선물 차이 등 지속분을 뺀다)
        for k in (1, 10):
            F[f"{pre}{v}_dev{k}"] = g(_ratio(_past_sum(dz, k), _past_sum(ok, k)) - base)
    return F


def feats_D(S):
    """흐름 기억 — 바이낸스 선물 ≤ t. 초 부호 sg = sign(매수 − 매도 명목)."""
    F = {}
    for k in (900, 3600):
        F[f"s_timb{k}"] = _imb(_past_sum(S["tb"], k), _past_sum(S["ts"], k))
    for k in (10, 60, 300, 900):
        F[f"s_oimb{k}"] = _imb(_past_sum(S["ob"], k), _past_sum(S["os"], k))
    sg = np.sign(S["tb"] - S["ts"])
    nz = (sg != 0).astype(float)
    for k in (60, 300, 900):
        F[f"s_fracb{k}"] = _ratio(_past_sum((sg > 0).astype(float), k), _past_sum(nz, k)) - 0.5
    idx = np.flatnonzero(sg)                                          # 0 아닌 초만 이은 부호 열
    z = sg[idx]
    pp = np.zeros(len(sg)); pc = np.zeros(len(sg))
    pp[idx[1:]] = z[1:] * z[:-1]; pc[idx[1:]] = 1                     # 짝(앞, 뒤)은 뒤 초에 기록
    for k in (300, 1800):
        F[f"s_acf{k}"] = _ratio(_past_sum(pp, k), _past_sum(pc, k))
    run = np.ones(len(z))
    for j in range(1, len(z)):                                        # ponytail: 파이썬 루프(하루 수만 개) — 느리면 numba
        if z[j] == z[j - 1]:
            run[j] = min(run[j - 1] + 1, 600)
    r = np.zeros(len(sg)); r[idx] = run * z
    last = np.maximum.accumulate(np.where(sg != 0, np.arange(len(sg)), -1))
    F["s_run"] = np.where(last >= 0, r[np.maximum(last, 0)], 0.0)
    for hl in (30, 300, 3000):
        F[f"s_ewm{hl}"] = pd.Series(sg).ewm(halflife=hl, adjust=False).mean().values
    return F


def flow_targets(S, tape_ok):
    """다음 흐름: (t, t+h] 테이커 명목 불균형. 창 안 tape_ok=0 이 있거나 체결 0 이면 결측."""
    Y = {}
    bad = (~tape_ok.astype(bool)).astype(float)
    n = len(tape_ok)
    for h in H_FLOW:
        fb, fs = _past_sum(S["tb"], h), _past_sum(S["ts"], h)
        fb, fs = np.r_[fb[h:], np.full(h, np.nan)], np.r_[fs[h:], np.full(h, np.nan)]
        y = _ratio(fb - fs, fb + fs)
        Y[f"y_flow{h}"] = np.where(_fwd_ok(bad, h) & (np.arange(n) + h < n), y, np.nan)
    return Y


def build_feat(coin):
    R = np.load(OBIM / coin / "_reduced.npz")
    t, mid, tok = R["t"], np.where(R["ok"].astype(bool), R["mid"], np.nan), R["tape_ok"]
    t0, n = int(t[0]), len(t)
    fr = pd.read_parquet(OBIM / f"frame_{coin}.parquet", columns=["t"])
    i = fr.t.values.astype(np.int64) - t0
    early = t[i] < ts(BAND_FROM.get(coin, "2000-01-01"))
    cut = lambda F: {c: np.where(early & (c[2:4] in ("wh", "rt")), np.nan, np.asarray(v)[i]).astype(np.float64 if c.startswith("y_") else np.float32)
                     for c, v in F.items()}                         # 행 τ 만 남겨 메모리 절약 · 경계 바뀌기 전 고래·리테일 결측
    bn = sec_arrays(coin, "bn", t0, n)
    F = {**cut(feats_A(bn)), **cut(feats_D(bn)), **cut(flow_targets(bn, tok))}
    V = {v: sec_arrays(coin, v, t0, n) for v in ("okx", "byb", "spot")}
    F |= cut(feats_B(V, bn, mid, 1, "x_")) | cut(feats_B(V, bn, mid, 0, "x0_"))
    del V, bn
    df = pd.DataFrame({"t": fr.t.values, "coin": coin, **F})
    sh = df[[c for c in df if c[:2] in ("t_", "x_", "s_")]].copy()     # 위약: 같은 피쳐 하루 전 값
    sh.columns = ["z_" + c for c in sh.columns]
    sh["t"] = df.t.values + DAY
    df = df.merge(sh, on="t", how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"feat_{coin}.parquet")
    cov = {c: float(np.isfinite(df[c]).mean()) for c in df if c[:2] in ("t_", "x_", "s_", "y_")}
    json.dump(cov, open(OUT / f"coverage_{coin}.json", "w"), indent=0)
    print(coin, len(df), "rows", flush=True)


# ── 적합·채점 ──
def resplit(df, split):
    """구간 다시 매기기 + 각 구간 끝에서 τ + h 가 넘는 대상 퍼지."""
    t = df.t.values
    lab = np.full(len(df), "fwd", object); start = -np.inf
    for name, end in SPLITS[split]:
        lab[(t >= start) & (t < ts(end))] = name; start = ts(end)
    df["split"] = lab
    for name, end in SPLITS[split]:
        e = ts(end)
        for c in [c for c in df if c.startswith("y_")]:
            h = int("".join(ch for ch in c.split("_", 1)[1].split("_")[0] if ch.isdigit()))
            df.loc[(df.split == name) & (df.t + h >= e), c] = np.nan
    return df


def load(split, arms):
    need = set(p for a in arms for p in ARMS[a])
    out = []
    for c in COINS:
        fr = pd.read_parquet(OBIM / f"frame_{c}.parquet")
        fr = fr[[k for k in fr if k in ("t", "coin") or k.startswith("y_") or k.startswith(tuple(need))]]
        ft = pd.read_parquet(OUT / f"feat_{c}.parquet")
        ft = ft[[k for k in ft if k in ("t",) or k.startswith("y_") or k.startswith(tuple(need))]]
        out.append(fr.merge(ft, on="t", how="left"))
    df = pd.concat(out, ignore_index=True); del out
    df = resplit(df, split)
    df, th = make_targets(df)                                          # 분위 문턱 = 이 분할의 train
    df = df.drop(columns=[c for c in df if c.startswith(("y_ret", "y_rv", "y_jmp"))])
    df["coin"] = df["coin"].astype("category"); df["split"] = df["split"].astype("category")
    return df


def cols(df, arm):
    return [c for c in df if c.startswith(ARMS[arm]) and c != "p_coin"] + ["p_coin"]


def fit(split, arm, df=None):
    import lightgbm as lgb
    df = df if df is not None else load(split, [arm])
    X = cols(df, arm)
    od = OUT / split; (od / "models").mkdir(parents=True, exist_ok=True)
    out, meta = [], {}
    for name, col, kind in G.TARGETS:
        mf = od / "models" / f"arm{arm}_{name}.txt"
        if mf.exists():                                                # 이어 하기: 저장 모델(이미 조기 종료된 것)로 예측만
            m = lgb.Booster(model_file=str(mf)); m.best_iteration = 0; tr = va = ()
        else:
            m = None
            tr, va = df[(df.split == "train") & df[col].notna()], df[(df.split == "val") & df[col].notna()]
        params = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=1.0, num_threads=int(os.environ.get("TAPEINFO_THREADS", 6)), seed=SEED, verbose=-1,
                      **({"objective": "multiclass", "num_class": 3} if kind == "multi" else {"objective": "binary"} if kind == "bin" else {"objective": "regression"}))
        if m is None:
            m = lgb.train(params, lgb.Dataset(tr[X], tr[col]), 2000, valid_sets=[lgb.Dataset(va[X], va[col])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            m.save_model(str(mf), num_iteration=m.best_iteration)
        ev = df[(df.split == "test") & df[col].notna()]                 # 시험 행만 저장(메모리 — val+test 31칸 concat 에서 11G 초과로 죽음)
        p = m.predict(ev[X], num_iteration=m.best_iteration)
        o = ev[["t", "coin", "split"]].copy(); o["target"] = name
        if kind == "multi":
            o[["p0", "p1", "p2"]] = p.astype(np.float32)
        else:
            o["p"] = p.astype(np.float32)
        imp = dict(zip(X, m.feature_importance("gain")))
        meta[name] = dict(best_iter=int(m.best_iteration), n_train=len(tr), n_val=len(va),
                          top_gain=sorted(imp, key=imp.get, reverse=True)[:8])
        o["coin"] = o.coin.astype(str); o["split"] = o.split.astype(str)
        out.append(o); del tr, va; print(split, arm, name, meta[name]["best_iter"], flush=True)
    if out:
        P = pd.concat(out); del out; P["target"] = P.target.astype("category")
        P.to_parquet(od / f"pred_{arm}.parquet")
    json.dump(meta, open(od / f"meta_{arm}.json", "w"), indent=1)


def predict(split, arm):
    """동결 모델로 시험 행만 예측(시험 날짜가 나중에 생기는 final 분할용) — 적합·조기 종료 없음."""
    import lightgbm as lgb
    df = load(split, [arm]); X = cols(df, arm); od = OUT / split; out = []
    for name, col, kind in G.TARGETS:
        m = lgb.Booster(model_file=str(od / "models" / f"arm{arm}_{name}.txt"))
        ev = df[(df.split == "test") & df[col].notna()]
        p = m.predict(ev[X]); o = ev[["t", "coin", "split"]].copy(); o["target"] = name
        if kind == "multi":
            o[["p0", "p1", "p2"]] = p.astype(np.float32)
        else:
            o["p"] = p.astype(np.float32)
        out.append(o)
    P = pd.concat(out); P["coin"] = P.coin.astype(str); P["split"] = P.split.astype(str)
    P.to_parquet(od / f"pred_{arm}_test.parquet")


def score(split, a, b):
    df = load(split, ["P"])
    df = df[df.split == "test"][["t", "coin"] + [c for _, c, _ in G.TARGETS]].copy(); df["coin"] = df.coin.astype(str)
    rd = lambda x: pd.read_parquet(OUT / split / f"pred_{x}.parquet", filters=[("split", "==", "test")])
    LA, LB = G.preds_to_loss(rd(a), df), G.preds_to_loss(rd(b), df)
    R = G.score_pair(LA, LB, f"{a}vs{b}")
    R.to_csv(OUT / split / f"score_{a}vs{b}.csv", index=False)
    print(R[["target", "n", "days", "base_loss", "d", "rel", "up90", "p", "coins_better", "strict", "candidate"]].to_string())


def selftest():
    rng = np.random.default_rng(0)
    n = 5000
    S = {c: np.zeros(n) for c in ("tb", "ts", "wb", "ws", "rb", "rs", "nb", "ns", "ob", "os", "kb", "ks", "q", "spn", "mb", "ms")}
    S["tb"][rng.integers(0, n, 900)] = rng.uniform(1, 100, 900); S["ts"][rng.integers(0, n, 900)] = rng.uniform(1, 100, 900)
    for c in ("nb", "ns", "ob", "os", "q", "kb", "ks"):
        S[c] = (S["tb" if c[1] == "b" else "ts"] > 0) * 2.0 if c != "q" else (S["tb"] + S["ts"]) / 50
    S["wb"], S["rb"], S["mb"], S["ms"] = S["tb"] * .4, S["tb"] * .1, S["tb"], S["ts"]
    S["bf"], S["day_ok"] = np.zeros(n, bool), np.ones(n, bool)
    mid = np.full(n, 50.0); tok = np.ones(n, np.uint8)
    i = 3000
    S["tb"][i], S["ts"][i] = 5.0, 5.0                                 # 초 i 불균형 0(아래 지연 점검용)
    base = {**feats_A(S), **feats_D(S), **feats_B({"okx": S}, S, mid, 1, "x_"), **flow_targets(S, tok)}
    i = 3000
    # 인과: i 뒤 초를 바꿔도 피쳐(≤ i)는 그대로, 대상은 바뀐다
    S2 = {k: v.copy() for k, v in S.items()}
    for c in ("tb", "ts", "wb", "mb", "ob", "nb", "q"):
        S2[c][i + 1:] = S2[c][i + 1:] * 3 + 7
    alt = {**feats_A(S2), **feats_D(S2), **feats_B({"okx": S2}, S2, mid, 1, "x_"), **flow_targets(S2, tok)}
    for c in base:
        if not c.startswith("y_"):
            assert np.allclose(base[c][:i + 1], alt[c][:i + 1], equal_nan=True), c
    assert not np.allclose(base["y_flow60"][i], alt["y_flow60"][i], equal_nan=True)
    # 다른 거래소 지연 1: 초 i 의 값을 바꾸면 i 피쳐는 그대로, i+1 피쳐는 바뀐다
    S3 = {k: v.copy() for k, v in S.items()}; S3["tb"][i] += 1000; S3["q"][i] += 20
    b3 = feats_B({"okx": S3}, S, mid, 1, "x_")
    assert b3["x_okx_imb1"][i] == base["x_okx_imb1"][i] and b3["x_okx_imb1"][i + 1] != base["x_okx_imb1"][i + 1]
    # 대상 정의: (i, i+10] 합
    fb, fs = S["tb"][i + 1:i + 11].sum(), S["ts"][i + 1:i + 11].sum()
    assert abs(base["y_flow10"][i] - (fb - fs) / (fb + fs)) < 1e-12
    # 테이프 끊김이면 결측
    tk = tok.copy(); tk[i + 5] = 0
    assert np.isnan(flow_targets(S, tk)["y_flow10"][i]) and np.isfinite(flow_targets(S, tk)["y_flow10"][i - 20])
    # 부호 연속 길이
    T = {k: np.zeros(8) for k in S}; T["tb"] = np.array([1, 1, 0, 1, 0, 0, 0, 0.]); T["ts"] = np.array([0, 0, 0, 0, 1, 1, 0, 0.])
    T["bf"] = np.zeros(8, bool)
    assert feats_D(T)["s_run"].tolist() == [1, 2, 2, 3, -1, -2, -2, -2]
    # 구성 몫: 고래 몫 = .4
    assert abs(np.nanmedian(base["t_whsh300"][400:] / _ratio(_past_sum(S["tb"], 300), _past_sum(S["tb"] + S["ts"], 300))[400:]) - .4) < 1e-9
    # 퍼지: 구간 끝 넘는 대상 결측
    d = pd.DataFrame({"t": np.array([ts("2026-11-03T23:59:00"), ts("2026-11-03T23:50:00")]), "y_flow300": [1.0, 1.0]})
    d = resplit(d, "final"); assert np.isnan(d.y_flow300[0]) and d.y_flow300[1] == 1.0 and d.split[0] == "train"
    print("tape_info selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat")
    ap.add_argument("--fit", nargs=2)
    ap.add_argument("--score", nargs=3)
    ap.add_argument("--predict", nargs=2)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.feat:
        build_feat(a.feat)
    elif a.fit:
        fit(*a.fit)
    elif a.score:
        score(*a.score)
    elif a.predict:
        predict(*a.predict)
