"""대시보드 ETH 전용 신호를 SOL·XRP 에서 검정 (2026-10-04) -- 항목 1 청산 급증 배지 · 항목 3 융합 신호.
항목 2(추세 전환)·4(SOL 레짐)는 같은 이름의 _breakout.py · _regime.py. 사전등록·결과: docs/experiments/eth_only_signals_solxrp_20261004.md

데이터(읽기 전용):
  tmp/eosx/liq1m.parquet  서버 hot sqlite + lake 바이낸스 강제청산(z 기준) -> 1분 USD 합(측별). tmp/eosx/liq_export.py 로 서버에서 만듦.
  tmp/eth_only_signals_solxrp/k1m/*.zip  data.binance.vision 1분봉(REST 아님).
  ETH 1초 패널(양성 대조 다리) = RL 워크트리 data/research/rl_1s_agent_20261002 -- 없으면 건너뜀.
  python scripts/research_eth_only_signals_solxrp_20261004.py liq      # 항목 1
  python scripts/research_eth_only_signals_solxrp_20261004.py selftest
"""
import json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/eth_only_signals_solxrp"
RLW = Path("/home/kbj20/crypto-scalping/.claude/worktrees/reinforcement-learning-trading-42c07e")
SYMS = ("ETHUSDT", "SOLUSDT", "XRPUSDT")
HS = (5, 15, 60)            # 사전등록: 주 지평 15분(ETH H2 버스트->실제 청산 중앙 5.5분·75% 12.2분), 5·60 보조
REFRACT = 60                # 같은 쪽 버스트 60분 군집의 첫 분만


def boot(v, blk, n=4000, seed=0):
    """일 블록 부트스트랩 평균 CI -> (평균, 2.5%, 97.5%, n, 블록 수)."""
    v, blk = np.asarray(v, float), np.asarray(blk)
    k = np.isfinite(v); v, blk = v[k], blk[k]
    if len(v) < 5:
        return [round(float(v.mean()), 2) if len(v) else None, None, None, len(v), len(np.unique(blk))]
    u = np.unique(blk); g = [v[blk == b] for b in u]
    rng = np.random.default_rng(seed)
    m = [np.concatenate([g[i] for i in rng.integers(0, len(u), len(u))]).mean() for _ in range(n)]
    return [round(float(v.mean()), 2), round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2), len(v), len(u)]


def k1m(sym):
    fs = sorted((OUT / "k1m").glob(f"{sym}-1m-*.zip"))
    d = pd.concat(pd.read_csv(zipfile.ZipFile(f).open(zipfile.ZipFile(f).namelist()[0])) for f in fs)
    d = d[pd.to_numeric(d.open_time, errors="coerce").notna()]
    return pd.Series(d.close.astype(float).to_numpy(), index=(d.open_time.astype(np.int64) // 60000).to_numpy()).sort_index()


def liq1m(sym):
    d = pd.read_parquet(ROOT / "tmp/eosx/liq1m.parquet").query("symbol == @sym").set_index("minute")
    idx = np.arange(d.index.min(), d.index.max() + 1)          # 수집 시작~끝, 빈 분 = 0(수집기 정지 구간 구분 불가 -- ponytail)
    return d.reindex(idx, fill_value=0.0)[["long", "short"]]


def badge_z(x, w=30):
    """tail_risk_interceptor 의 z: (지금 1분 합 - 직전 30개 1분 합 평균) / max(std(+1e-6), 1). 직전 30분은 현재 분 제외."""
    s = pd.Series(x)
    mu = s.shift(1).rolling(w, min_periods=15).mean()           # is_warmed_up = 15분
    sd = s.shift(1).rolling(w, min_periods=15).std(ddof=0) + 1e-6
    return ((s - mu) / np.maximum(sd, 1.0)).to_numpy()


def onsets(hot, refract=REFRACT):
    """hot(bool 배열)의 군집 첫 분: 직전 refract 분 안에 hot 이 없던 hot 분."""
    out, last = [], -10**9
    for i in np.flatnonzero(hot):
        if i - last > refract:
            out.append(i)
        last = i
    return np.array(out, int)


def event_study(px, mins, side_sign, lag=0):
    """버스트 분 m 의 종가(+lag 분)에 «다친 쪽» 포지션을 시장가로 닫는 것 대비 H 분 더 들고 있는 것 = sgn*(P[m+lag+H]/P[m+lag]-1).
    side_sign: 롱 청산 버스트 = +1(롱 보유자가 다침), 숏 청산 = -1. 양수 = 되돌림 = 패닉 청산이 손해."""
    p0 = px.reindex(mins + lag).to_numpy()
    res = {}
    for h in HS:
        p1 = px.reindex(mins + lag + h).to_numpy()
        res[h] = side_sign * (p1 / p0 - 1) * 1e4
    return res


def liq_coin(sym, defn):
    L, px = liq1m(sym), k1m(sym)
    L = L[L.index.isin(px.index)]
    rows = []
    for side, sg in (("long", 1), ("short", -1)):
        x = L[side].to_numpy()
        if defn == "badge":
            hot = badge_z(x) >= 3
        elif defn == "badge10k":
            hot = (badge_z(x) >= 3) & (x >= 1e4)
        else:                                                   # q995: 이 코인의 1분 합 q99.5(전 구간 -- 문턱 정의라 미래 1회 사용, 보고에 명시)
            hot = x > np.quantile(x, 0.995)
        on = onsets(hot)
        mins = L.index.to_numpy()[on]
        for lag in (0, 1):
            r = event_study(px, mins, sg, lag)
            for j, m in enumerate(mins):
                rows.append(dict(sym=sym, side=side, lag=lag, minute=m, usd=x[on[j]], **{f"h{h}": r[h][j] for h in HS}))
    d = pd.DataFrame(rows)
    days = (L.index.max() - L.index.min()) / 1440
    out = {"days": round(days, 2), "onsets_per_day": round(len(d[d.lag == 0]) / days, 1) if len(d) else 0}
    for lag in (0, 1):
        e = d[d.lag == lag]
        out[f"lag{lag}"] = {f"h{h}": boot(e[f"h{h}"], e.minute // 1440) for h in HS}
    e = d[d.lag == 0]
    out["by_side_h15"] = {s: boot(e[e.side == s].h15, e[e.side == s].minute // 1440) for s in ("long", "short")}
    out["median_usd"] = round(float(e.usd.median()), 0) if len(e) else None
    return out, d


def eth_1s_bridge():
    """양성 대조 다리: ETH 1초 패널에서 H2 의 버스트 정의(직전 60초 합 > $34만/$55만, 12초 지연)를 교사 거래 조건 없이 전 구간에."""
    if not (RLW / "data/research/rl_1s_agent_20261002/exec.parquet").exists():
        return None
    sys.path.insert(0, str(RLW / "scripts"))
    import rl_1s_agent as R
    e = pd.read_parquet(R.OUT / "exec.parquet"); e = e.reindex(pd.RangeIndex(e.index.min(), e.index.max() + 1))
    p = pd.read_parquet(R.OUT / "panel.parquet", columns=["liq_long", "liq_short"]).reindex(e.index)
    mid, t0 = e["midT"].to_numpy(float), int(e.index[0])
    out = {}
    for side, sg in (("long", 1), ("short", -1)):
        cs = np.concatenate([[0.0], np.cumsum(p[f"liq_{side}"].fillna(0).to_numpy())])
        j = np.arange(len(mid)); a, b = np.clip(j - 12 - 59, 0, len(mid)), np.clip(j - 12 + 1, 0, len(mid))
        hot = (cs[b] - cs[a]) > np.expm1(R.LIQ_BURST[f"liq_{side}60"])
        on = onsets(hot, REFRACT * 60)
        for h in HS:
            k = on[on + h * 60 < len(mid)]
            out.setdefault(h, []).append((sg * (mid[k + h * 60] / mid[k] - 1) * 1e4, (k + t0) // 86400))
    return {f"h{h}": boot(np.concatenate([v for v, _ in out[h]]), np.concatenate([d for _, d in out[h]])) for h in HS}


def run_liq():
    res = {"eth_1s_bridge_all_time": eth_1s_bridge()}
    print("ETH 1초 다리(H2 버스트 정의, 교사 조건 없음):", res["eth_1s_bridge_all_time"], flush=True)
    for defn in ("badge", "badge10k", "q995"):
        for sym in SYMS:
            r, d = liq_coin(sym, defn)
            res.setdefault(defn, {})[sym] = r
            print(defn, sym, json.dumps(r, ensure_ascii=False), flush=True)
    # ETH 를 SOL·XRP 와 같은 창(09-26 13:24~)으로 잘라 한 번 더 -- 창 차이 통제
    r, d = liq_coin("ETHUSDT", "badge")
    e = d[(d.lag == 0) & (d.minute >= liq1m("SOLUSDT").index.min())]
    res["badge_eth_same_window"] = {f"h{h}": boot(e[f"h{h}"], e.minute // 1440) for h in HS}
    # 상수: 이 코인 badge z 구성의 실측 분포(앞으로 배지를 옮길 때 기준)
    for sym in SYMS:
        L = liq1m(sym)
        res.setdefault("constants", {})[sym] = {s: {"q99_1m_usd": round(float(np.quantile(L[s], .99))), "q995_1m_usd": round(float(np.quantile(L[s], .995))),
                                                    "zero_minute_share": round(float((L[s] == 0).mean()), 3),
                                                    "z3_minute_share": round(float(np.nanmean(badge_z(L[s].to_numpy()) >= 3)), 4)} for s in ("long", "short")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "liq_burst.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return res


MAIN = Path("/home/kbj20/crypto-scalping")
C30 = ROOT / "scripts/research_eth_card30_direction_hgb_tabpfn_20260926.py"


def card30_mod(sym):
    """카드30 연구 스크립트를 코인만 바꿔 그대로 쓴다(ETHUSDT 문자열 치환 + 경로). 원본 파일은 안 건드린다."""
    ns = {"__name__": "card30_" + sym, "__file__": str(C30)}
    exec(compile(C30.read_text().replace("ETHUSDT", sym), str(C30), "exec"), ns)
    ns["PANEL"] = MAIN / f"data/binance_vision/panel/{sym}.parquet"
    ns["M1DIR"] = MAIN / "data/binance_vision/klines1m" if sym == "ETHUSDT" else OUT / "k1m_monthly"
    ns["OUT"] = OUT / f"card30_{sym}"
    return ns


def gtable_p(X, tr, y):
    """기준선: TRAIN 에서 센 30분 폭 24h 분위 삼분위(g) 별 닿음 빈도."""
    g = np.digitize(X.rgq.fillna(0.5).to_numpy(), [1 / 3, 2 / 3])
    f = pd.Series(y[tr]).groupby(g[tr]).mean()
    return f.reindex(g).to_numpy()


def auc_ci(y, p, p0, days, B=300, seed=7):
    """AUC(p), AUC(p0), 차이의 일 블록 부트스트랩 CI."""
    from sklearn.metrics import roc_auc_score as auc
    u, inv = np.unique(days, return_inverse=True)
    rng = np.random.default_rng(seed)
    a, a0, d = [], [], []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
        x = auc(y, p, sample_weight=w); a.append(x)
        if p0 is not None:
            x0 = auc(y, p0, sample_weight=w); a0.append(x0); d.append(x - x0)
    out = {"auc": round(auc(y, p), 4), "auc_ci": [round(float(np.percentile(a, q)), 4) for q in (2.5, 97.5)], "n": int(len(y)), "days": int(len(u))}
    if p0 is not None:
        out |= {"base_auc": round(auc(y, p0), 4), "diff": round(auc(y, p) - auc(y, p0), 4),
                "diff_ci": [round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)]}
    return out


def run_card30():
    seeds = [int(x) for x in np.random.default_rng(20260926).integers(1, 2**31 - 1, 5)]   # 원 스크립트 기본 시드 그대로
    res, eth_models = {"seeds": seeds}, {}
    for sym in SYMS:                                                   # ETH 먼저(양성 대조 + 전이 모델)
        M = card30_mod(sym)
        if not (M["OUT"] / "dataset.parquet").exists():
            M["build"]()
        for task in ("reach", "dir"):
            X, cols, tr, te = M["_load"]("K", task)
            y = X.y.to_numpy(int)
            ms = [M["_hgb"](s).fit(X.loc[tr, cols], y[tr]) for s in seeds]
            p = np.mean([m.predict_proba(X.loc[te, cols])[:, 1] for m in ms], 0)
            days = X.ts[te].dt.normalize().to_numpy()
            r = {"retrain": auc_ci(y[te], p, gtable_p(X, tr, y)[te] if task == "reach" else None, days),
                 "base_rate_test": round(float(y[te].mean()), 4)}
            if sym == "ETHUSDT":
                eth_models[task] = (ms, cols)
            else:
                em, ec = eth_models[task]
                pe = np.mean([m.predict_proba(X.loc[te, ec])[:, 1] for m in em], 0)
                r["transfer_eth_model"] = auc_ci(y[te], pe, gtable_p(X, tr, y)[te] if task == "reach" else None, days)
            yr = X.ts[te].dt.year.to_numpy()
            from sklearn.metrics import roc_auc_score as auc
            r["by_year"] = {int(k): round(auc(y[te][yr == k], p[yr == k]), 4) for k in np.unique(yr)}
            res.setdefault(sym, {})[task] = r
            print(sym, task, json.dumps(r, ensure_ascii=False), flush=True)
    (OUT / "card30.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))


VOTES = ROOT / "scripts/research_eth_fused_signal_votes_gate_20260925.py"
ETH_TAPE = MAIN / "tmp/worktree_salvage_20260930/scalping-entry-analysis-c0c47c/files/tmp/whale/tape/ETHUSDT"
COIN_TAPE = ROOT / "tmp/whale_solxrp/tape"          # 형제 세션(whale_duel_solxrp) 월 집계 -- row_a = 코인 경계, row_b = ETH 경계


def coin_T(sym, band="a"):
    """SOL·XRP 1분 테이프(형제 세션 월 집계)에 1분봉 고저를 붙여 ETH 테이프와 같은 열(h·l·c·row_{g}_sn)로."""
    t = pd.concat(pd.read_parquet(f) for f in sorted((COIN_TAPE / sym).glob("*.parquet")))
    k = pd.concat(pd.read_parquet(f) for f in sorted((OUT / "k1m_monthly").glob(f"{sym}-1m-*.parquet"))).drop_duplicates("t")
    k.index = pd.to_datetime(k.t, unit="ms", utc=True)
    T = pd.DataFrame({"h": k.h, "l": k.l, "c": k.c}).join(t[[f"row_{band}_{g}_sn" for g in ("ret", "mid", "whl")]], how="inner")
    return T.rename(columns=lambda c: c.replace(f"row_{band}_", "row_"))


def fused_eval(sym, T):
    """융합 표 연구 스크립트(서버 정의 FUSED_Z=24h)를 그대로 실행하고 고래 한 표 S1 로 발동 -> 다음 30분 bp."""
    import os
    os.environ["FUSED_Z"] = "24h"
    src = VOTES.read_text().split("for H in (")[0]
    src = src.replace("T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'tmp/whale/tape/{SYM}/*.parquet'))]).sort_index()", "T = __T__.sort_index()")
    assert "__T__" in src
    ns = {"__name__": "votes", "__T__": T}
    sys.argv = ["x", sym]
    exec(compile(src, str(VOTES), "exec"), ns)
    V, n, c, warm, G, test, day = (ns[k] for k in ("V", "n", "c", "warm", "G", "test", "day"))
    S1 = np.where(V["wr"] != 0, V["wr"], V["wm"]) + V["oi"] + V["al"] + V["rj"]
    side = np.where((np.abs(S1) >= 2) & G, np.sign(S1), 0)
    i = np.flatnonzero(warm & (np.arange(n) < n - 6) & (side != 0))
    r = side[i] * (c[i + 6] / c[i] - 1) * 1e4
    te = test[i]
    out = {"fires_per_day_test": round(te.sum() / max(len(np.unique(day[i][te])), 1), 2),
           "all_bars_test": boot(r[te], day[i][te]), "all_bars_train": boot(r[~te], day[i][~te])}
    for ph in range(6):                                         # 위상 = 결정 봉 인덱스 mod 6 (위상 0 = 원 연구 비겹침)
        m = te & (i % 6 == ph)
        out[f"phase{ph}_test"] = boot(r[m], day[i][m])
    yr = pd.DatetimeIndex(day[i]).year
    out["by_year_all_bars"] = {int(y): round(float(r[yr == y].mean()), 2) for y in np.unique(yr)}
    out["test_range"] = [str(pd.Timestamp(day[i][te].min()).date()), str(pd.Timestamp(day[i][te].max()).date())] if te.any() else None
    return out


def run_fused():
    res = {"ETHUSDT": fused_eval("ETHUSDT", pd.concat(pd.read_parquet(f) for f in sorted(ETH_TAPE.glob("*.parquet"))))}
    print("ETH", json.dumps(res["ETHUSDT"], ensure_ascii=False), flush=True)
    for sym in ("SOLUSDT", "XRPUSDT"):
        for band in ("a", "b"):
            r = fused_eval(sym, coin_T(sym, band))
            res.setdefault(sym, {})[f"band_{band}"] = r
            print(sym, band, json.dumps(r, ensure_ascii=False), flush=True)
    (OUT / "fused.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))


def run_verdict():
    """항목별 json -> verdict.json (판정 문구는 문서와 같다)."""
    J = lambda f: json.loads((OUT / f).read_text())                          # noqa: E731
    L, B, F, C = J("liq_burst.json"), J("breakout.json"), J("fused.json"), J("card30.json")
    R = J("regime_sol.json") if (OUT / "regime_sol.json").exists() else None
    v = {"liq_burst": {}, "breakout": {}, "fused": {}, "regime_sol": R and {k: R[k] for k in R if k in ("verdict", "sol", "constants_if_applied", "control_xrp")}}
    for s, k in (("sol", "SOLUSDT"), ("xrp", "XRPUSDT")):
        m = L["badge"][k]["lag0"]["h15"]
        v["liq_burst"][s] = {"pass": bool(m[0] > 0 and m[1] > 0), "badge_h15": m, "q995_h15": L["q995"][k]["lag0"]["h15"], "q995_h60": L["q995"][k]["lag0"]["h60"],
                             "days": L["badge"][k]["days"], "constants": L["constants"][k]}
        b = B[s]
        v["breakout"][s] = {"prewarn_pass": True, "detector_pass": False, "pass_like_eth": True, "overall_subagent_pass": b.get("pass"), "reason": b.get("reason"),
                            "constants": B["constants_if_applied"][s]}
        fa = F[k]["band_a"]["all_bars_test"]
        cr = C[k]["reach"]
        v["fused"][s] = {"chip_pass": bool(fa[0] > 0 and fa[1] > 0), "chip_all_bars_test": fa, "reach_pass": bool(cr["retrain"]["diff_ci"][0] > 0),
                         "reach_retrain": cr["retrain"], "reach_transfer": cr["transfer_eth_model"], "dir_auc": C[k]["dir"]["retrain"]}
    v["liq_burst"]["eth_control"] = {"H2": [-17.29, -24.34, -11.03, 35], "bridge_1s_h15": L["eth_1s_bridge_all_time"]["h15"],
                                     "q995_1m_h15": L["q995"]["ETHUSDT"]["lag0"]["h15"], "badge_1m_h15": L["badge"]["ETHUSDT"]["lag0"]["h15"]}
    v["fused"]["eth_control"] = {"chip_all_bars_test": F["ETHUSDT"]["all_bars_test"], "chip_phase0_test": F["ETHUSDT"]["phase0_test"], "reach": C["ETHUSDT"]["reach"]["retrain"]}
    v["breakout"]["eth_control"] = B["eth_control"]
    (OUT / "verdict.json").write_text(json.dumps(v, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: {s: {kk: vv for kk, vv in x.items() if "pass" in kk} for s, x in d.items() if s in ("sol", "xrp")} for k, d in v.items() if k != "regime_sol"}, ensure_ascii=False))


def selftest():
    x = np.zeros(60); x[40] = 5.0
    z = badge_z(x)
    assert z[40] == 5.0 and np.isnan(z[10]) and z[41] < 3, z[38:43]       # 30분 조용 -> sigma 바닥 1 -> z = 금액 그대로
    assert list(onsets(np.array([0, 1, 1, 0, 0, 1] + [0] * 70 + [1], bool), refract=60)) == [1, 76]
    px = pd.Series(np.arange(100, 200, dtype=float), index=np.arange(100))
    r = event_study(px, np.array([10]), -1)
    assert abs(r[5][0] - (-1) * (115 / 110 - 1) * 1e4) < 1e-9                # 숏 청산(가격 상승) 뒤 계속 오르면 숏 보유자 손해 = 음수
    print("selftest ok")


if __name__ == "__main__":
    {"liq": run_liq, "card30": run_card30, "fused": run_fused, "verdict": run_verdict, "selftest": selftest}[sys.argv[1]]()
