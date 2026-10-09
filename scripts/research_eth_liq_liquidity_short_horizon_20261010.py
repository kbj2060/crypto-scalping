"""청산·유동성 지표로 «바로 앞 5~15분» 방향 — 왜 안 되나를 분해 (2026-10-10).
사용자 «청산과 유동성에 대한 지표들로 방향을 어느 정도 바로 앞 5분이나 15분 정도는 예측가능할 것 같은데 왜 안되지? 제대로 다시 한 번 연구해봐».

질문을 넷으로 나눈다(결과 전 고정).
  Q1 정보가 있나 — 지평 1초~15분마다 각 지표 IC(스피어만). 지평이 길어질수록 어떻게 줄어드나(감쇠 곡선).
  Q2 언제 쓰이나 — 같은 지표의 «과거 같은 길이 수익»과의 상관(동행) vs 미래 수익(선행). 동행 ≫ 선행이면 «이미 가격에 반영».
  Q3 얼마나 빨리 사라지나 — 결합 모델 IC 를 진입 지연 0·1·5·30초로.
  Q4 돈이 되려면 — 상·하위 10% 매매의 평균 bp ≈ IC × σ_h × 1.755(정규 근사). 손익분기 IC = 비용 / (σ_h × 1.755).
     비용: 테이커 왕복 8bp · 메이커 왕복 ≈1.4bp(역선택 미끄러짐). 실측 상·하위 10% 평균 bp 도 같이 낸다.

A 초 실험실 `--sec` (ETH, 09-15~10-04, 10-07 order_battle 의 depthdiff 재생 1초 패널 + lake 청산(09-20~)·OI 1초(09-19~)):
  호가 imb10·imb50(±10/±50bp 깊이 불균형) · OFI 1/10/60초(지정가 순추가 매수쪽 − 매도쪽) · 체결 1/10/60/300초((B−S)/(B+S))
  · 청산 60/300초(숏청산$ − 롱청산$, 부호 log) · OI 변화 60/300초(+가격 방향 곱) · 과거 수익 10/60/300초(대조).
  피쳐 = 초 t 끝에 앎, 목표 = mid[t+d+h]/mid[t+d]. DEV 09-15~09-26T10 · HOLDOUT 09-26T10~10-04(10-07 과 같은 경계). 1시간 블록 CI.
B 분 단위 `--min` (ETH, 05-03~10-07, lake micro_1m·tail_risk_1m — 🔴행 ts 는 분 시작 표기지만 값은 분 끝 스냅샷 → 앎 시각 = ts+1분):
  obi · 테이커 매수비 · 고래/리테일 순흐름 · eai · OI 변화 · 흡수·큐 점수 · 청산 1/5/15분(07-19~, 숏−롱 비율) · 과거 수익 1/5/15/60분.
  목표 5·15분, 진입 = 스냅샷 그 순간(d=0, 낙관) / 다음 분(d=1). TRAIN 05-03~08-31 · TEST 09-01~10-07. 일 블록 CI. 리지 + HGB.
  트립와이어: 피쳐가 «앎 이전 1분 수익»보다 «앎 이후 1분 수익»과 더 강하게 상관하면 시각 어긋남 의심(10-01 게이트 L3).
판정: 결합 모델 TEST 상·하위 10% 평균 bp CI 하한 > 8(테이커) 또는 > 1.4(메이커)면 «매매 가능», 아니면 «정보 있음/없음»만 보고.
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · future_rows_used_for_entry=false.
"""
import glob, json, os, sys
import numpy as np, pandas as pd

LAKE = os.path.expanduser("~/backups/crypto-scalping-server/data/lake/binance")
PANEL = "/home/kbj20/crypto-scalping/.claude/worktrees/realtime-ledger-analysis-bdd373/tmp/order_battle_20261007/panel_1s.parquet"
OUT = "tmp/liq_liquidity_short_horizon_20261010"
SEC_SPLIT = (pd.Timestamp("2026-09-15", tz="UTC"), pd.Timestamp("2026-09-26T10", tz="UTC"), pd.Timestamp("2026-10-04", tz="UTC"))
MIN_SPLIT = (pd.Timestamp("2026-05-03", tz="UTC"), pd.Timestamp("2026-09-01", tz="UTC"), pd.Timestamp("2026-10-08", tz="UTC"))
TAKER, MAKER, ZTOP = 8.0, 1.4, 1.755                                         # ZTOP = E[z | 상위 10%] 정규 근사


def lake(stream, cols=None):
    fs = sorted(glob.glob(f"{LAKE}/{stream}/coin=ETH/date=*/part.parquet"))
    return pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)


def rsum(x, k):
    """뒤로 k 칸(현재 포함) 합. 앞쪽은 NaN."""
    c = np.r_[0.0, np.nancumsum(x)]
    out = np.full(len(x), np.nan); out[k - 1:] = c[k:] - c[:-k]
    return out


def ic(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    return float(pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank()))


def block_ci(blk, fn, n=400, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blk); g = {u: np.flatnonzero(blk == u) for u in ub}
    v = [fn(np.concatenate([g[u] for u in rng.choice(ub, len(ub))])) for _ in range(n)]
    return [round(float(x), 4) for x in np.nanquantile(v, [.025, .975])]


def ridge_fit(X, y, lam=1.0):
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-12
    Z = np.clip(np.nan_to_num((X - mu) / sd), -5, 5)
    b = np.linalg.solve(Z.T @ Z + lam * len(Z) * 1e-3 * np.eye(Z.shape[1]), Z.T @ y)
    return lambda X2: np.clip(np.nan_to_num((X2 - mu) / sd), -5, 5) @ b


def decile_bp(p, y):
    """상위 10% 롱 · 하위 10% 숏 평균 수익(bp, 비용 전)."""
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    lo, hi = np.quantile(p, [.1, .9])
    s = np.where(p >= hi, 1, np.where(p <= lo, -1, 0))
    return float((s[s != 0] * y[s != 0]).mean())


def fwd(lm, d, h):
    n = len(lm); y = np.full(n, np.nan)
    if d + h < n:
        y[:n - d - h] = (lm[d + h:] - lm[d:n - h]) * 1e4
    return y


# ───────────────────────── A. 초 실험실 ─────────────────────────
SEC_H = (1, 5, 10, 30, 60, 300, 900)
SEC_D = (0, 1, 5, 30)
SEC_FEATS = ["imb10", "imb50", "ofi1", "ofi10", "ofi60", "flow1", "flow10", "flow60", "flow300", "liq60", "liq300",
             "doi60", "doi300", "doixr60", "doixr300", "ret10", "ret60", "ret300"]


def sec_panel():
    p = pd.read_parquet(PANEL).reset_index()                                        # sec 는 인덱스
    full = np.arange(int(p.sec.min()), int(p.sec.max()) + 1)
    P = p.set_index("sec").reindex(full)
    mid = ((P.bb + P.ba) / 2).where(P.valid.fillna(0) > 0)
    F = pd.DataFrame({"sec": full})
    lm = np.log(mid.to_numpy())
    F["imb10"] = ((P.bid10 - P.ask10) / (P.bid10 + P.ask10)).to_numpy()
    F["imb50"] = ((P.bid50 - P.ask50) / (P.bid50 + P.ask50)).to_numpy()
    net = ((P.add_b - P.rem_b) - (P.add_a - P.rem_a)).to_numpy(); tot = (P.add_b + P.rem_b + P.add_a + P.rem_a).to_numpy()
    for k in (1, 10, 60):
        F[f"ofi{k}"] = rsum(net, k) / np.maximum(rsum(tot, k), 1e-9)
    bs, bt = (P.B - P.S).to_numpy(), (P.B + P.S).to_numpy()
    for k in (1, 10, 60, 300):
        F[f"flow{k}"] = rsum(bs, k) / np.maximum(rsum(bt, k), 1e-9)
    L = lake("liquidations", ["ts_ms", "side", "usd"])
    L["sec"] = L.ts_ms // 1000
    lq = L.assign(v=np.where(L.side == "short", L.usd, -L.usd)).groupby("sec").v.sum().reindex(full).fillna(0).to_numpy()   # 숏청산 = 강제 매수(+)
    liq_on = full >= L.sec.min() + 300
    for k in (60, 300):
        s = rsum(lq, k)
        F[f"liq{k}"] = np.where(liq_on, np.sign(s) * np.log1p(np.abs(s) / 1e4), np.nan)
    O = lake("oi_1s", ["ts_ms", "open_interest"])
    O["sec"] = O.ts_ms // 1000
    loi = np.log(O.groupby("sec").open_interest.last().reindex(full).ffill(limit=5).to_numpy())
    for k in (10, 60, 300):
        F[f"ret{k}"] = (lm - np.r_[np.full(k, np.nan), lm[:-k]]) * 1e4
    for k in (60, 300):
        d = (loi - np.r_[np.full(k, np.nan), loi[:-k]]) * 1e4
        F[f"doi{k}"] = d
        F[f"doixr{k}"] = d * np.sign(F[f"ret{k}"].to_numpy())                     # OI 늘며 오름(+)·OI 늘며 내림(−)
    F["lm"] = lm
    ts = pd.to_datetime(F.sec, unit="s", utc=True)
    F["split"] = np.select([(ts >= SEC_SPLIT[0]) & (ts < SEC_SPLIT[1]), (ts >= SEC_SPLIT[1]) & (ts < SEC_SPLIT[2])], ["DEV", "HO"], "-")
    F["blk"] = F.sec // 3600
    return F


def run_sec():
    os.makedirs(OUT, exist_ok=True)
    F = sec_panel(); lm = F.lm.to_numpy()
    dev, ho = (F.split == "DEV").to_numpy(), (F.split == "HO").to_numpy()
    blk = F.blk.to_numpy()
    sub = np.zeros(len(F), bool); sub[::5] = True                                # 단변량 IC 는 5초 간격 표본(속도)
    out = {"sigma": {}, "ic": {}, "coincident": {}, "model": {}, "delay": {}}
    for h in SEC_H:
        y = fwd(lm, 0, h); past = (lm - np.r_[np.full(h, np.nan), lm[:-h]]) * 1e4
        out["sigma"][h] = round(float(np.nanstd(y[ho])), 3)
        out["ic"][h] = {f: round(ic(F[f].to_numpy()[ho & sub], y[ho & sub]), 4) for f in SEC_FEATS}
        out["coincident"][h] = {f: round(ic(F[f].to_numpy()[ho & sub], past[ho & sub]), 4) for f in SEC_FEATS if not f.startswith("ret")}
        print("sec ic", h, json.dumps(out["ic"][h]), flush=True)
    X = F[SEC_FEATS].to_numpy()
    for h in SEC_H:
        for d in SEC_D:
            y = fwd(lm, d, h)
            tr = dev & np.isfinite(y)
            pred = ridge_fit(X[tr][::3], y[tr][::3])(X)
            te = ho & np.isfinite(y)
            pt, yt, bt = pred[te], y[te], blk[te]
            r = {"ic": round(ic(pt, yt), 4), "ic_ci": block_ci(bt, lambda ix: ic(pt[ix], yt[ix]), n=200),
                 "top10_bp": round(decile_bp(pt, yt), 3), "top10_ci": block_ci(bt, lambda ix: decile_bp(pt[ix], yt[ix]), n=200),
                 "breakeven_ic_taker": round(TAKER / (out["sigma"][h] * ZTOP), 3), "breakeven_ic_maker": round(MAKER / (out["sigma"][h] * ZTOP), 3)}
            (out["model"] if d == 0 else out["delay"]).setdefault(h, {})[d] = r
            print("sec model", h, d, json.dumps(r), flush=True)
    out["flags"] = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "future_rows_used_for_entry": False}
    out["n_sec"] = {"DEV": int(dev.sum()), "HO": int(ho.sum())}
    json.dump(out, open(f"{OUT}/sec.json", "w"), indent=1)


# ───────────────────────── B. 분 단위 ─────────────────────────
MIN_H = (1, 5, 15)
MIN_FEATS_MICRO = ["obi", "taker_buy_ratio", "nif_whale", "nif_retail", "eai", "oi_delta_pct", "shadow_absorption_score",
                   "shadow_queue_bias", "whale_position_score", "spoofing_score"]
MIN_FEATS = MIN_FEATS_MICRO + ["tbr5", "tbr15", "liq1", "liq5", "liq15", "ret1", "ret5", "ret15", "ret60"]


def min_panel():
    M = lake("micro_1m")
    M = M[~M.data_stale & M.valid_taker_flow].drop_duplicates("ts").copy()
    M["known"] = M.ts + pd.Timedelta("1min")                                      # 🔴 분 끝 스냅샷
    T = lake("tail_risk_1m", ["ts", "long_usd_1m", "short_usd_1m"]).drop_duplicates("ts")
    T["known"] = T.ts + pd.Timedelta("1min")
    grid = pd.DataFrame({"known": pd.date_range(M.known.min(), M.known.max(), freq="1min")})
    G = grid.merge(M[["known", "mark_price"] + MIN_FEATS_MICRO], on="known", how="left").merge(
        T[["known", "long_usd_1m", "short_usd_1m"]], on="known", how="left")
    lp = np.log(G.mark_price.to_numpy())
    for k in (1, 5, 15, 60):
        G[f"ret{k}"] = (lp - np.r_[np.full(k, np.nan), lp[:-k]]) * 1e4
    tb = G.taker_buy_ratio.to_numpy() - 0.5
    for k in (5, 15):
        G[f"tbr{k}"] = rsum(tb, k) / k
    liq_on = (G.known >= pd.Timestamp("2026-07-19", tz="UTC")).to_numpy()
    net = (G.short_usd_1m - G.long_usd_1m).fillna(0).to_numpy(); tot = (G.short_usd_1m + G.long_usd_1m).fillna(0).to_numpy()
    for k in (1, 5, 15):
        G[f"liq{k}"] = np.where(liq_on, rsum(net, k) / np.maximum(rsum(tot, k), 1.0), np.nan)    # 비율이라 09-25 l→z 규모 전환에 무관
    G["lp"] = lp
    G["split"] = np.select([(G.known >= MIN_SPLIT[0]) & (G.known < MIN_SPLIT[1]), (G.known >= MIN_SPLIT[1]) & (G.known < MIN_SPLIT[2])], ["TRAIN", "TEST"], "-")
    G["day"] = G.known.dt.floor("D").astype("int64") // 86_400_000_000_000
    return G


def run_min():
    from sklearn.ensemble import HistGradientBoostingRegressor
    os.makedirs(OUT, exist_ok=True)
    G = min_panel(); lp = G.lp.to_numpy()
    tr, te = (G.split == "TRAIN").to_numpy(), (G.split == "TEST").to_numpy()
    day = G.day.to_numpy()
    out = {"n": {"TRAIN": int(tr.sum()), "TEST": int(te.sum())}, "tripwire": {}, "sigma": {}, "ic": {}, "coincident": {}, "model": {}}
    nxt, prv = fwd(lp, 0, 1), (lp - np.r_[np.nan, lp[:-1]]) * 1e4
    for f in MIN_FEATS_MICRO + ["liq1"]:                                          # 앎 이후 1분 vs 이전 1분
        x = G[f].to_numpy()
        out["tripwire"][f] = {"ic_next1m": round(ic(x[te], nxt[te]), 4), "ic_prev1m": round(ic(x[te], prv[te]), 4)}
    print("tripwire", json.dumps(out["tripwire"]), flush=True)
    X = G[MIN_FEATS].to_numpy()
    for h in MIN_H:
        past = (lp - np.r_[np.full(h, np.nan), lp[:-h]]) * 1e4
        out["coincident"][h] = {f: round(ic(G[f].to_numpy()[te], past[te]), 4) for f in MIN_FEATS if not f.startswith("ret")}
        for d in (0, 1):
            y = fwd(lp, d, h)
            out["sigma"][f"{h}_{d}"] = round(float(np.nanstd(y[te])), 2)
            if d == 0:
                out["ic"][h] = {f: round(ic(G[f].to_numpy()[te], y[te]), 4) for f in MIN_FEATS}
                print("min ic", h, json.dumps(out["ic"][h]), flush=True)
            ok = np.isfinite(y)
            res = {}
            for nm in ("ridge", "hgb"):
                if nm == "ridge":
                    f_ = ridge_fit(X[tr & ok], y[tr & ok])
                else:
                    f_ = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=300, l2_regularization=1.0,
                                                       random_state=0).fit(X[tr & ok], y[tr & ok]).predict
                pt, yt, dt = f_(X[te & ok]), y[te & ok], day[te & ok]
                res[nm] = {"ic": round(ic(pt, yt), 4), "ic_ci": block_ci(dt, lambda ix: ic(pt[ix], yt[ix]), n=200),
                           "top10_bp": round(decile_bp(pt, yt), 2), "top10_ci": block_ci(dt, lambda ix: decile_bp(pt[ix], yt[ix]), n=200)}
            sg = out["sigma"][f"{h}_{d}"]
            res["breakeven_ic_taker"], res["breakeven_ic_maker"] = round(TAKER / (sg * ZTOP), 3), round(MAKER / (sg * ZTOP), 3)
            out["model"][f"{h}_{d}"] = res
            print("min model", h, d, json.dumps(res), flush=True)
    out["flags"] = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "future_rows_used_for_entry": False,
                    "micro_1m_known_at": "ts+1min"}
    json.dump(out, open(f"{OUT}/min.json", "w"), indent=1)


def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/malgun.ttf")          # WSL: 리눅스 쪽엔 한글 글꼴 없음
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    S, Mn = json.load(open(f"{OUT}/sec.json")), json.load(open(f"{OUT}/min.json"))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    hs = list(S["model"]); x = np.array([int(h) for h in hs], float)
    ax[0].plot(x, [max(S["model"][h]["0"]["ic"], 1e-4) for h in hs], "o-", label="호가·체결·청산·OI 결합 (초 패널 HOLDOUT)")
    ax[0].plot(x, [S["model"][h]["0"]["breakeven_ic_taker"] for h in hs], "--", label="손익분기 IC · 테이커 8bp")
    ax[0].plot(x, [S["model"][h]["0"]["breakeven_ic_maker"] for h in hs], ":", label="손익분기 IC · 메이커 1.4bp")
    ax[0].plot([h * 60 for h in MIN_H], [max(Mn["model"][f"{h}_0"]["hgb"]["ic"], 1e-4) for h in MIN_H], "s", color="C3", label="분 단위 HGB (TEST 09-01~10-07)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_yticks([.003, .01, .03, .1, .3, 1, 3], ["0.003", "0.01", "0.03", "0.1", "0.3", "1", "3"])
    ax[0].set_xticks([1, 10, 60, 300, 900], ["1초", "10초", "1분", "5분", "15분"]);ax[0].set_xlabel("예측 지평 (초)"); ax[0].set_ylabel("IC")
    ax[0].set_title("얻은 IC vs 돈이 되려면 필요한 IC"); ax[0].legend(fontsize=8)
    for h in ("60", "300", "900"):
        ds = [0] + [int(d) for d in S["delay"][h]]
        ax[1].plot(ds, [S["model"][h]["0"]["ic"]] + [S["delay"][h][str(d)]["ic"] for d in ds[1:]], "o-", label=f"지평 {h}초")
    ax[1].axhline(0, color="k", lw=.5); ax[1].set_xlabel("진입 지연 (초)"); ax[1].set_ylabel("IC"); ax[1].set_title("진입이 늦으면 IC 가 얼마나 남나"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/why.png", dpi=130)
    print(f"{OUT}/why.png")


def selftest():
    assert np.allclose(rsum(np.array([1., 2, 3, 4]), 2)[1:], [3, 5, 7]) and np.isnan(rsum(np.array([1., 2]), 2)[0])
    lm = np.log(np.array([100., 101, 102, 103]))
    assert np.allclose(fwd(lm, 1, 1)[:2], [np.log(102 / 101) * 1e4, np.log(103 / 102) * 1e4]) and np.isnan(fwd(lm, 1, 1)[2])
    p = np.arange(100.); y = np.where(p >= 90, 5.0, np.where(p < 10, -3.0, 0.0))
    assert abs(decile_bp(p, y) - 4.0) < 1e-12                                   # 롱 +5 · 숏 +3 평균
    rng = np.random.default_rng(0); X = rng.normal(size=(5000, 3)); yy = X @ [1., 0, -2] + rng.normal(size=5000) * .1
    assert ic(ridge_fit(X, yy)(X), yy) > .99
    print("selftest OK")


if __name__ == "__main__":
    {"--sec": run_sec, "--min": run_min, "--plot": plot, "--selftest": selftest}[sys.argv[1]]()
