#!/usr/bin/env python3
"""호가 정보 지도 — 팔 ⓪(가격·체결) ① (⓪ + 손 호가) LightGBM + 공용 채점(사전등록 docs/prereg/ob_info_map_dl_20261115.md).

입력: tmp/ob_info_map_20261115/frame_<COIN>.parquet (research_ob_info_map_build_20261115 --frame).
대상 28칸마다 4코인 합동 모델 하나(코인 = p_coin 피쳐). 학습 = train · 조기 종료 = val(시간순) · 예측 = val·test 저장.
채점(score_pair): 행별 손실 차를 UTC 일 블록 부트스트랩 2,000회 → 단측 90% 상한·p(개선 아님) · Holm(α .10, 계열 28칸).
사용: --fit 0|1|6 (6 = ① + 팔 P 표현, 표현 파일 필요) · --score A B (예: 1 0) · --selftest
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
sys.path.insert(0, str(ROOT / "scripts"))
from research_ob_info_map_build_20261115 import H_DIR, H_RV, H_FILL, DS, H_DEP, OUT, make_targets  # noqa: E402

COINS = tuple(os.environ.get("OBIM_COINS", "ETH,BTC,SOL,XRP").split(","))
TARGETS = ([(f"dir{h}", f"c_dir{h}", "multi") for h in H_DIR] + [(f"lrv{h}", f"c_lrv{h}", "reg") for h in H_RV]
           + [(f"jmp{h}", f"c_jmp{h}", "bin") for h in H_RV]
           + [(f"fill{sd}{h}_{int(d * 1e4)}", f"y_fill{sd}{h}_{int(d * 1e4)}", "bin") for h in H_FILL for d in DS for sd in "bs"]
           + [(f"dep{h}", f"y_dep{h}", "reg") for h in H_DEP])
assert len(TARGETS) == 28
SEED = 20261115


def load():
    df = pd.concat([pd.read_parquet(OUT / f"frame_{c}.parquet") for c in COINS], ignore_index=True)
    df, th = make_targets(df)
    df = df.drop(columns=[c for c in df if c.startswith(("y_ret", "y_rv", "y_jmp"))])   # 원 수익·RV·경로 최대는 c_* 로 바뀜(메모리)
    for c in ("coin", "split"):
        df[c] = df[c].astype("category")
    return df, th


def row_loss(kind, y, p):
    """행별 손실: multi = 3분류 로그손실(p (n,3)) · bin = 이진 로그손실 · reg = 제곱오차."""
    if kind == "multi":
        return -np.log(np.clip(p[np.arange(len(y)), y.astype(int)], 1e-7, 1))
    if kind == "bin":
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return (y - p) ** 2


def feats(df, arm):
    p = [c for c in df if c.startswith("p_")]
    if arm == 0:
        return p
    b = [c for c in df if c.startswith("b_")]
    return p + b + ([c for c in df if c.startswith("z_")] if arm == 6 else [])


def fit(arm, df=None):
    import lightgbm as lgb
    df = df if df is not None else load()[0]
    if arm == 6:
        Z = pd.read_parquet(OUT / "rep_P.parquet")          # t, coin, z_0..z_63 (팔 P 고정 표현)
        df = df.merge(Z, on=["t", "coin"], how="left")
    cols = feats(df, arm)
    out, meta = [], {}
    for name, col, kind in TARGETS:
        mf = OUT / "models" / f"arm{arm}_{name}.txt"
        if mf.exists():                                                 # 이어 하기: 이미 적합한 칸은 저장 모델로 예측만
            m = lgb.Booster(model_file=str(mf)); m.best_iteration = 0
            ev = df[df.split.isin(["val", "test"]) & df[col].notna()]
            p = m.predict(ev[cols]); o = ev[["t", "coin", "split"]].copy(); o["target"] = pd.Categorical([name] * len(o))
            if kind == "multi":
                o[["p0", "p1", "p2"]] = p.astype(np.float32)
            else:
                o["p"] = p.astype(np.float32)
            out.append(o); print(arm, name, "resumed", flush=True); continue
        tr, va = df[(df.split == "train") & df[col].notna()], df[(df.split == "val") & df[col].notna()]
        params = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=1.0, num_threads=6, seed=SEED, verbose=-1,
                      **({"objective": "multiclass", "num_class": 3} if kind == "multi" else {"objective": "binary"} if kind == "bin" else {"objective": "regression"}))
        m = lgb.train(params, lgb.Dataset(tr[cols], tr[col]), 2000, valid_sets=[lgb.Dataset(va[cols], va[col])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        (OUT / "models").mkdir(exist_ok=True); m.save_model(str(OUT / "models" / f"arm{arm}_{name}.txt"), num_iteration=m.best_iteration)
        ev = df[df.split.isin(["val", "test"]) & df[col].notna()]
        p = m.predict(ev[cols], num_iteration=m.best_iteration)
        o = ev[["t", "coin", "split"]].copy(); o["target"] = pd.Categorical([name] * len(o))
        if kind == "multi":
            o[["p0", "p1", "p2"]] = p.astype(np.float32)
        else:
            o["p"] = p.astype(np.float32)
        meta[name] = dict(best_iter=int(m.best_iteration), n_train=len(tr), n_val=len(va))
        out.append(o); del tr, va, m
        print(arm, name, meta[name], flush=True)
    P = pd.concat(out); P["target"] = P["target"].astype(str); P.to_parquet(OUT / f"pred_arm{arm}.parquet")
    json.dump(meta, open(OUT / f"meta_arm{arm}.json", "a"), indent=1)


def predict(arm, df=None):
    """저장 모델로 val·test 예측만 다시(11-14 테이프가 늦게 들어와 시험 하루 피쳐가 바뀐 경우) — 적합·조기 종료는 그대로."""
    import lightgbm as lgb
    df = df if df is not None else load()[0]
    if arm == 6:
        df = df.merge(pd.read_parquet(OUT / "rep_P.parquet"), on=["t", "coin"], how="left")
    cols = feats(df, arm); out = []
    for name, col, kind in TARGETS:
        m = lgb.Booster(model_file=str(OUT / "models" / f"arm{arm}_{name}.txt"))
        ev = df[df.split.isin(["val", "test"]) & df[col].notna()]
        p = m.predict(ev[cols]); o = ev[["t", "coin", "split"]].copy(); o["target"] = name
        if kind == "multi":
            o[["p0", "p1", "p2"]] = p
        else:
            o["p"] = p
        out.append(o)
    pd.concat(out).to_parquet(OUT / f"pred_arm{arm}.parquet")


def preds_to_loss(pred, df, split="test"):
    """예측 파일 → (target, t, coin) 행별 손실."""
    kinds = {n: (c, k) for n, c, k in TARGETS}
    res = []
    for name, g in pred[pred.split == split].groupby("target"):
        col, kind = kinds[name]
        g = g.merge(df[["t", "coin", col]], on=["t", "coin"])
        p = g[["p0", "p1", "p2"]].values if kind == "multi" else g["p"].values
        res.append(pd.DataFrame({"target": name, "t": g.t, "coin": g.coin, "loss": row_loss(kind, g[col].values, p)}))
    return pd.concat(res)


def boot(d, day, n=2000, seed=SEED, q=0.90):
    """행별 차 d(A − B, 음수 = A 가 나음) · 날짜 블록 부트스트랩 → (평균, q 분위(기본 단측 90% 상한), p = P(평균 ≥ 0))."""
    days = np.unique(day); rng = np.random.default_rng(seed)
    s = pd.DataFrame({"d": d, "day": day}).groupby("day").d.agg(["sum", "count"]).reindex(days)
    S, C = s["sum"].values, s["count"].values
    idx = rng.integers(0, len(days), (n, len(days)))
    m = S[idx].sum(1) / C[idx].sum(1)
    return float(S.sum() / C.sum()), float(np.quantile(m, q)), float((m >= 0).mean())


def holm(ps, alpha=0.10):
    o = np.argsort(ps); k = len(ps); ok = np.zeros(k, bool)
    for r, i in enumerate(o):
        if ps[i] <= alpha / (k - r):
            ok[i] = True
        else:
            break
    return ok


def score_pair(LA, LB, label):
    """LA·LB = preds_to_loss 결과(같은 split). 같은 행(교집합)에서만."""
    rows = []
    for name, _, _ in TARGETS:
        a, b = LA[LA.target == name], LB[LB.target == name]
        m = a.merge(b, on=["t", "coin"], suffixes=("_a", "_b"))
        if not len(m):
            continue
        d = (m.loss_a - m.loss_b).values
        day = (m.t.values // 86400)
        mean, up, p = boot(d, day)
        coin_sign = m.assign(d=d).groupby("coin").d.mean()
        rows.append(dict(cmp=label, target=name, n=len(m), days=len(np.unique(day)), base_loss=float(m.loss_b.mean()),
                         d=mean, rel=mean / float(m.loss_b.mean()), up90=up, p=p, coins_better=int((coin_sign < 0).sum()),
                         **{f"d_{c}": float(coin_sign.get(c, np.nan)) for c in COINS}))
    R = pd.DataFrame(rows)
    R["strict"] = holm(R.p.values) & (R.up90 < 0)
    R["candidate"] = (R.up90 < 0) & (R.coins_better >= 3)
    return R


def selftest():
    rng = np.random.default_rng(1)
    day = np.repeat(np.arange(11), 1000)
    m, up, p = boot(rng.normal(-0.1, 1, 11000), day)
    assert m < 0 and up < 0 and p < 0.1
    m, up, p = boot(rng.normal(0, 1, 11000), day)
    assert up > 0 and p > 0.1
    assert holm(np.array([0.001, 0.2, 0.04]), 0.10).tolist() == [True, False, True]
    y = np.array([0, 2]); P = np.array([[.5, .25, .25], [.1, .1, .8]])
    assert np.allclose(row_loss("multi", y, P), -np.log([.5, .8]))
    print("gbm selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int)
    ap.add_argument("--score", nargs=2, type=str)
    ap.add_argument("--predict", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.fit is not None:
        fit(a.fit)
    elif a.predict is not None:
        predict(a.predict)
    elif a.score:
        df, _ = load()
        df = df[df.split == "test"][["t", "coin"] + [c for _, c, _ in TARGETS]].copy()          # 채점은 시험 행·대상만(메모리)
        df["coin"] = df["coin"].astype(str)
        rd = lambda x: pd.read_parquet(OUT / f"pred_{x if not x.isdigit() else 'arm' + x}.parquet", filters=[("split", "==", "test")]).assign(coin=lambda d: d.coin.astype(str))
        LA, LB = (preds_to_loss(rd(x), df) for x in a.score)
        R = score_pair(LA, LB, f"{a.score[0]}vs{a.score[1]}")
        R.to_csv(OUT / f"score_{a.score[0]}vs{a.score[1]}.csv", index=False)
        print(R[["target", "n", "base_loss", "d", "rel", "up90", "p", "coins_better", "strict", "candidate"]].to_string())
