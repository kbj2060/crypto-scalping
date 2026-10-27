"""liqlab 판정 실행기 — 사전등록 docs/prereg/liqlab_liquidity_arms_20261028.md 의 7팔. 2026-10-28.

  python liqlab_run_20261028.py --controls   # 실데이터 위 양성·음성 대조(판정 전 장치 검증)
  python liqlab_run_20261028.py --run        # 학습(24·25) 임계 → 검증(26-01~05) 변형 선택 → 시험(26-06~10-23) 1회
창: TRAIN 2024-01-01~2025-12-31 · VAL 2026-01-01~2026-05-31 · TEST 2026-06-01~2026-10-23.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import liqlab_harness_20261028 as L  # noqa: E402

D = pathlib.Path("data/liqlab"); OUT = pathlib.Path("tmp/liqlab_20261028"); OUT.mkdir(parents=True, exist_ok=True)
COINS = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "XRPUSDT"]
WIN = {"train": ("2024-01-01", "2025-12-31 23:59"), "val": ("2026-01-01", "2026-05-31 23:59"), "test": ("2026-06-01", "2026-10-23 23:59")}
EXPECTED_BP = {"T1": 5.0, "T2": 2.0, "T3": 3.0}   # 사전등록 기대 효과(MDE 판정 불능 기준)
T4_STRESS_BP = 2.0                                # 일봉 리밸런스 메이커 + 역선택 스트레스(다리당)


# ───────────────────────────── 피쳐(행 τ = 봉 τ 끝에 앎) ─────────────────────────────
def panel(sym):
    p = pd.read_parquet(D / f"{sym}_5m.parquet")
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


T_GRID = {"T1": [dict(state=s, q=q, H=H) for s, q, H in itertools.product(["rv", "amihud", "none"], [.90, .95], [12, 48])],
          "T2": [dict(dirn=d, q=q, H=H) for d, q, H in itertools.product(["against", "follow"], [.95, .99], [1, 6])],
          "T3": [dict(cond=c) for c in ["none", "highvol", "highrv"]]}
ARM_FN = {"T1": arm_T1, "T2": arm_T2, "T3": arm_T3}
SELECTABLE = {"T1": lambda v: v["state"] != "none", "T2": lambda v: True, "T3": lambda v: True}   # T1 «none» 은 대조군


def score_arm(arm, data, variant, windows=("val", "test"), B_null=300):
    out = {}
    for sym, (p, f) in data.items():
        tr = win(p.index, "train")
        sig, H = ARM_FN[arm](p, f, tr, **variant)
        rvq = np.nanquantile(f.rv24[tr], [1 / 3, 2 / 3])
        for w in windows:
            m = win(p.index, w); px = {k: p[k].to_numpy()[m] for k in "ohlc"}
            day = ((p.index[m] - p.index[m][0]).days).to_numpy() + 10_000 * COINS.index(sym)
            st = L.strata_of(f.hour.to_numpy()[m], f.rv24.to_numpy()[m], rvq)
            out.setdefault(w, {})[sym] = L.evaluate_window(sig[m], px, day, st, H, B_null=B_null)
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


def main():
    data = {s: panel(s) for s in COINS}
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
          "G3": best4["val_mean_bp_day"] > 0 and tt["mean_bp_day"] > 0, "G5": hm["T4"]["reject"], "G6": dv > .95}
    report["T4"].update(dsr=dv, verdict={"verdict": "통과" if all(g4.values()) else "불통과", **g4})
    report["holm"] = hm; report["n_trials"] = len(trial_srs)
    for arm, target in (("P1", "y_P1_1h"), ("P1_4h", "y_P1_4h"), ("P2", "y_P2"), ("P3", "y_P3")):
        res = score_pred(arm[:2], data, target); report[arm] = {"per_coin": res, "verdict": pred_gate(res)}
    json.dump(report, open(OUT / "report.json", "w"), ensure_ascii=False, indent=1, default=float)
    for k in ("T1", "T2", "T3", "T4", "P1", "P1_4h", "P2", "P3"):
        print(k, json.dumps(report[k].get("verdict"), ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
