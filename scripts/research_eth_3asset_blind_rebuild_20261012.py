"""«3자산 결합» 방향 전략 블라인드 재구성 (2026-10-12) — 원 코드(trade_eval·walk_forward·portfolio3 등) 안 보고 사양만으로.
사양: 부품 ① 4h HGB(월 재학습·위상 8분산·지정가→시장가 실행) ② 일봉 추세 5표 ③ UTC 00:00 개장 되돌림 → 자산별 위험 균등 결합 → 3자산 위험 균등.
원 스크립트에서 쓰는 것은 데이터 로더 load_panel 뿐.

사용:
  python scripts/research_eth_3asset_blind_rebuild_20261012.py --selftest
  python scripts/research_eth_3asset_blind_rebuild_20261012.py --preds          # HGB 예측(본·피쳐 1봉 지연·일 블록 섞기) 캐시, 오래 걸림
  python scripts/research_eth_3asset_blind_rebuild_20261012.py --report         # 부품·결합·트립와이어 표 + daily_*.parquet
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · future_rows_used_for_entry=false.
"""
import glob, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_eth_best_direction_model_20261010 import load_panel  # 데이터 로더만

PANEL_DIR = "/home/kbj20/crypto-scalping/.claude/worktrees/liquidation-price-feature-14eceb/tmp/best_dir_20261010/"
K1M = "/home/kbj20/crypto-scalping/data/binance_vision/klines1m/"
OUT = Path("tmp/blind_rebuild_20261012")
SYMS = ["ETH", "BTC", "SOL"]
H = 48
FEE_T = 4.0
D0, D1 = pd.Timestamp("2023-03-02"), pd.Timestamp("2026-09-12")
HGB = dict(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=1000, l2_regularization=1.0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def label(c, h=H):
    c = np.asarray(c, float)
    return np.r_[np.log(c[h:] / c[:-h]) * 1e4, np.full(h, np.nan)]


def load(sym):
    F = pd.read_parquet(PANEL_DIR + ("panel.parquet" if sym == "ETH" else f"panel_{sym}.parquet"))
    r = load_panel(sym)
    assert (F.ts.values == r.timestamp.values).all()
    cols = [x for x in F.columns if len(x) > 2 and x[1] == "_" and x[0] in "PTOBSMDXCU"]
    return dict(ts=r.timestamp.reset_index(drop=True), c=r.close.to_numpy(float), h=r.high.to_numpy(float),
                l=r.low.to_numpy(float), X=F[cols].to_numpy(np.float32), cols=cols, y_panel=F["y_4h"].to_numpy(float))


# ---------- 부품 ① ----------
def monthly_preds(ts, X, y, seeds):
    from sklearn.ensemble import HistGradientBoostingRegressor
    pred = np.full(len(ts), np.nan)
    t = ts.to_numpy()
    for M in pd.date_range("2023-01-01", "2026-09-01", freq="MS"):
        end = M - pd.Timedelta(minutes=240)
        tr = np.flatnonzero((t >= np.datetime64(end - pd.Timedelta(days=365))) & (t < np.datetime64(end)) & ~np.isnan(y))[::2]
        if len(tr) < 20000:
            log("skip", M.date(), len(tr)); continue
        te = np.flatnonzero((t >= np.datetime64(M)) & (t < np.datetime64(M + pd.offsets.MonthBegin(1))))
        p = [HistGradientBoostingRegressor(random_state=s, **HGB).fit(X[tr], y[tr]).predict(X[te]) for s in seeds]
        pred[te] = np.mean(p, axis=0)
        log(M.date(), len(tr), len(te))
    return pred


def thresholds(pred, warm=8640):
    """과거 예측만(한 칸 밀어) 확장 분위 q0.2/q0.8. NaN 예측은 건너뛴 순서열 기준."""
    ok = ~np.isnan(pred)
    s = pd.Series(pred[ok])
    ex = s.expanding(min_periods=warm)
    lo, hi = np.full(len(pred), np.nan), np.full(len(pred), np.nan)
    lo[ok], hi[ok] = ex.quantile(0.2).shift(1).to_numpy(), ex.quantile(0.8).shift(1).to_numpy()
    return lo, hi


def signal(pred, lo, hi):
    return np.where(pred >= hi, 1, np.where(pred <= lo, -1, 0))  # NaN 비교 = False → 0


def min1_arrays(t0, n_min):
    """ETH 1분봉 → 분 인덱스(t0 기준) 배열 h,l,c. 없는 분 = NaN."""
    h, l, c = (np.full(n_min, np.nan) for _ in range(3))
    for f in sorted(glob.glob(K1M + "ETHUSDT-1m-*.parquet")):
        k = pd.read_parquet(f, columns=["t", "h", "l", "c"])
        i = ((k.t.to_numpy() - t0) // 60000).astype(np.int64)
        m = (i >= 0) & (i < n_min)
        h[i[m]], l[i[m]], c[i[m]] = k.h.to_numpy()[m], k.l.to_numpy()[m], k.c.to_numpy()[m]
    return h, l, c


def fill(side, lim, hi_w, lo_w, close_end, fee_m):
    """side=+1 매수: 창 안 저가 < 지정가면 지정가 체결(fee_m), 아니면 창 끝 종가 시장가(4bp). side=-1 매도는 고가 > 지정가."""
    touched = np.where(side > 0, lo_w < lim, hi_w > lim)
    return np.where(touched, lim, close_end), np.where(touched, fee_m, FEE_T)


def exec_px(k, side, A, fee_m):
    """결정 봉 k(마감 T0) 에서 side 방향 주문의 체결가·수수료. A['m1'] 있으면 1분봉 W=5분, 아니면 다음 5분봉."""
    if A.get("m1") is not None:
        h1, l1, c1 = A["m1"]
        j = k * 5 + 5  # T0 의 분 인덱스(패널 첫 봉 시작 = 0)
        lim = c1[j - 1]
        hw = np.nanmax(np.stack([h1[j + d] for d in range(5)]), axis=0)
        lw = np.nanmin(np.stack([l1[j + d] for d in range(5)]), axis=0)
        return fill(side, lim, hw, lw, c1[j + 4], fee_m)
    return fill(side, A["c"][k], A["h"][k + 1], A["l"][k + 1], A["c"][k + 1], fee_m)


def part1(A, pred, fee_m=0.0):
    lo, hi = thresholds(pred)
    sig = signal(pred, lo, hi)
    n = len(A["c"])
    k = np.arange(n - H - 2)
    T0 = A["ts"].to_numpy()[k] + np.timedelta64(5, "m")
    phase = (A["ts"].dt.hour.to_numpy()[k] * 12 + A["ts"].dt.minute.to_numpy()[k] // 5) % 48
    keep = (phase % 6 == 0) & (sig[k] != 0) & (T0 >= np.datetime64("2023-02-01"))
    k, T0, d = k[keep], T0[keep], sig[k[keep]]
    pin, fin = exec_px(k, d, A, fee_m)
    pout, fout = exec_px(k + H, -d, A, fee_m)
    pnl = (d * np.log(pout / pin) * 1e4 - fin - fout) / 8
    bad = np.isnan(pnl)
    tr = pd.DataFrame({"T0": T0, "dir": d, "pnl": pnl, "maker_in": fin < FEE_T, "maker_out": fout < FEE_T})[~bad]
    daily = tr.groupby(tr.T0.dt.normalize()).pnl.sum()
    return daily, tr, int(bad.sum())


# ---------- 부품 ② ----------
def part2(A):
    C = pd.Series(A["c"], index=A["ts"]).groupby(A["ts"].dt.normalize().to_numpy()).last()
    votes = sum(np.sign(C - C.shift(L)) for L in (7, 14, 28, 56, 90)) / 5
    pos = np.sign(votes).where(votes.abs() >= 0.5, 0.0).fillna(0.0)  # 날 d 종가 기준, d+1 보유
    held = pos.shift(1).fillna(0.0)
    r = np.log(C / C.shift(1)) * 1e4
    return (held * r - held.diff().abs().fillna(held.abs()) * FEE_T).fillna(0.0)


# ---------- 부품 ③ ----------
def part3(A, cost=FEE_T):
    ts, c = A["ts"], A["c"]
    r = np.r_[np.nan, np.diff(np.log(c))] * 1e4
    sd288 = pd.Series(r).rolling(288, min_periods=288).std().to_numpy()
    k = np.flatnonzero((ts.dt.hour.to_numpy() == 23) & (ts.dt.minute.to_numpy() == 55))
    k = k[k + 54 < len(c)]  # 00:25 = k+6, 04:25 = k+54
    f30 = np.log(c[k + 6] / c[k]) * 1e4
    sig = sd288[k] * np.sqrt(6)
    d = np.where(np.abs(f30) > 0.5 * sig, -np.sign(f30), 0)
    pnl = np.where(d != 0, d * np.log(c[k + 54] / c[k + 6]) * 1e4 - cost, 0.0)
    return pd.Series(pnl, index=(ts.iloc[k + 6].dt.normalize()).to_numpy()).fillna(0.0)


# ---------- 결합·통계 ----------
def risk_parity(P, cap=3.0):
    sd = P.rolling(90, min_periods=60).std().shift(1)
    W = sd.mean(axis=1).to_numpy()[:, None] / sd
    W = W.clip(upper=cap)
    ok = W.notna().all(axis=1)
    return ((P * W).sum(axis=1) / P.shape[1])[ok]


def frame(parts):
    idx = pd.date_range(D0, D1, freq="D")
    return pd.DataFrame({k: v.reindex(idx).fillna(0.0) for k, v in parts.items()}, index=idx)


def stats(x, n_boot=2000, seed=0):
    x = x.dropna()
    sh = lambda v: v.mean() / v.std() * np.sqrt(365) if v.std() > 0 else np.nan
    rng = np.random.default_rng(seed)
    a = x.to_numpy()
    b = rng.integers(0, len(a), (n_boot, len(a)))
    bm = a[b].mean(1); bs = bm / a[b].std(1, ddof=1) * np.sqrt(365)
    cum = x.cumsum()
    mon = x.groupby(x.index.to_period("M")).sum()
    out = dict(n=len(x), sharpe=sh(x), sh_lo=np.percentile(bs, 2.5), sh_hi=np.percentile(bs, 97.5), bp_day=a.mean(),
               bp_lo=np.percentile(bm, 2.5), bp_hi=np.percentile(bm, 97.5), mdd=(cum - cum.cummax()).min(),
               pos_months=f"{int((mon > 0).sum())}/{len(mon)}")
    for y, g in x.groupby(x.index.year):
        out[f"sh{y}"] = sh(g)
    return out


# ---------- 실행 ----------
def pred_path(sym, tag):
    return OUT / f"pred_{sym}_{tag}.npy"


def run_preds():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, "base", (0, 1, 2)) for s in SYMS] + [(s, "lag1", (0, 1, 2)) for s in SYMS] + [(s, "shuf", (0,)) for s in SYMS]
    for sym, tag, seeds in jobs:
        if pred_path(sym, tag).exists():
            continue
        A = load(sym)
        y, X = label(A["c"]), A["X"]
        if tag == "lag1":
            X = np.r_[np.full((1, X.shape[1]), np.nan, np.float32), X[:-1]]
        if tag == "shuf":  # 일 블록 단위로 y 섞기(패널 = 완전 5분 격자, 288행/일)
            nd = len(y) // 288
            perm = np.random.default_rng(7).permutation(nd)
            y = np.r_[y[:nd * 288].reshape(nd, 288)[perm].ravel(), y[nd * 288:]]
        log("start", sym, tag)
        np.save(pred_path(sym, tag), monthly_preds(A["ts"], X, y, seeds))


def tripwire_ic(A, n=60000):
    y = label(A["c"])
    r = np.r_[np.nan, np.diff(np.log(A["c"]))]
    rn = np.r_[r[1:], np.nan]
    idx = np.random.default_rng(0).choice(np.flatnonzero(~np.isnan(y) & ~np.isnan(rn) & (A["ts"] >= "2023-01-01").to_numpy()), n, replace=False)
    rk = lambda v: pd.Series(v).rank().to_numpy()
    rows = []
    for j, col in enumerate(A["cols"]):
        x = A["X"][idx, j].astype(float)
        m = ~np.isnan(x)
        if m.sum() < 1000 or np.nanstd(x) == 0:
            continue
        cc = lambda v: np.corrcoef(rk(x[m]), rk(v[idx][m]))[0, 1]
        rows.append(dict(feat=col, ic_y=cc(y), corr_same=cc(r), corr_next=cc(rn)))
    T = pd.DataFrame(rows)
    T["next_minus_same"] = T.corr_next.abs() - T.corr_same.abs()
    return T


def report():
    R, daily, tw = {}, {}, {}
    t0 = int(pd.Timestamp("2022-01-01").value // 1_000_000)
    for sym in SYMS:
        A = load(sym)
        if sym == "ETH":
            A["m1"] = min1_arrays(t0, len(A["c"]) * 5 + 600)
            A["m1_last"] = str(pd.to_datetime(t0 + 60000 * np.flatnonzero(~np.isnan(A["m1"][2]))[-1], unit="ms"))
        y = label(A["c"])
        m = ~np.isnan(y) & ~np.isnan(A["y_panel"])
        tw[f"{sym}_label_match"] = dict(max_abs_diff=float(np.max(np.abs(y[m] - A["y_panel"][m]))),
                                        frac_lt_1e6=float(np.mean(np.abs(y[m] - A["y_panel"][m]) < 1e-6)))
        pred = np.load(pred_path(sym, "base"))
        p1, tr, nbad = part1(A, pred)
        parts = {"hgb": p1, "trend": part2(A), "open": part3(A)}
        P = frame(parts)
        P["combo"] = risk_parity(P[["hgb", "trend", "open"]])
        daily[sym] = P
        R[sym] = {k: stats(P[k]) for k in P}
        R[sym]["_trades"] = dict(n=len(tr), dropped_nan=nbad, maker_in=float(tr.maker_in.mean()), maker_out=float(tr.maker_out.mean()),
                                 ic_pred_y=float(pd.Series(pred).corr(pd.Series(y), method="spearman")),
                                 m1_last=A.get("m1_last", "5m 근사"))
        # 트립와이어 2·4·5 + ③ 비용 8bp 민감도
        for tag in ("lag1", "shuf"):
            if pred_path(sym, tag).exists():
                Q = P[["hgb", "trend", "open"]].copy()
                Q["hgb"] = part1(A, np.load(pred_path(sym, tag)))[0].reindex(Q.index).fillna(0.0)
                Q["combo"] = risk_parity(Q[["hgb", "trend", "open"]])
                tw[f"{sym}_{tag}"] = {"hgb": stats(Q.hgb), "combo": stats(Q.combo)}
                daily[f"{sym}_{tag}"] = Q
        Q = P[["hgb", "trend", "open"]].copy()
        Q["hgb"] = part1(A, pred, fee_m=1.0)[0].reindex(Q.index).fillna(0.0)
        Q["combo"] = risk_parity(Q[["hgb", "trend", "open"]])
        tw[f"{sym}_maker1bp"] = {"hgb": stats(Q.hgb), "combo": stats(Q.combo)}
        daily[f"{sym}_maker1bp"] = Q
        Q = P[["hgb", "trend"]].copy(); Q["open"] = part3(A, cost=8.0).reindex(Q.index).fillna(0.0)
        Q["combo"] = risk_parity(Q[["hgb", "trend", "open"]])
        tw[f"{sym}_open8bp"] = {"open": stats(Q.open), "combo": stats(Q.combo)}
        T = tripwire_ic(A)
        T.sort_values("next_minus_same", ascending=False).to_csv(OUT / f"tripwire_ic_{sym}.csv", index=False)
        tw[f"{sym}_ic_top10_absIC"] = T.reindex(T.ic_y.abs().sort_values(ascending=False).index).head(10).round(4).to_dict("records")
        tw[f"{sym}_ic_top10_next_minus_same"] = T.sort_values("next_minus_same", ascending=False).head(10).round(4).to_dict("records")
        log("done", sym)

    def three(key, src):
        C = pd.DataFrame({s: src[s if key == "base" else f"{s}_{key}"]["combo"] for s in SYMS})
        C = C.dropna(how="all").fillna(0.0)
        return C, risk_parity(C)
    C, c3 = three("base", daily)
    R["3asset"] = {"combo": stats(c3)}
    for key in ("lag1", "shuf", "maker1bp"):
        if all(f"{s}_{key}" in daily for s in SYMS):
            tw[f"3asset_{key}"] = stats(three(key, daily)[1])
    for s in SYMS:
        daily[s].to_parquet(OUT / f"daily_{s}.parquet")
    pd.DataFrame({**{s: C[s] for s in SYMS}, "combo": c3}).to_parquet(OUT / "daily_3asset.parquet")
    json.dump({"results": R, "tripwires": tw}, open(OUT / "report.json", "w"), indent=1, default=float, ensure_ascii=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for k, v in R.items():
        print(f"\n== {k}"); print(pd.DataFrame({a: b for a, b in v.items() if a != "_trades"}).T.round(3))
        if "_trades" in v: print(v["_trades"])
    print("\n== tripwires")
    for k, v in tw.items():
        print(k, json.dumps(v, default=float, ensure_ascii=False)[:2000])


def selftest():
    # 라벨
    c = np.exp(np.arange(100) / 1e4)
    y = label(c, 48)
    assert np.allclose(y[:52], 48.0) and np.isnan(y[52:]).all()
    # 임계: 과거만
    p = np.r_[np.arange(10.0), [np.nan], [100.0]]
    lo, hi = thresholds(p, warm=5)
    assert np.isnan(hi[4]) and hi[5] == np.quantile(np.arange(5.0), 0.8) and hi[11] == np.quantile(np.arange(10.0), 0.8)
    assert signal(np.array([1.0, 5, 9, np.nan]), np.full(4, 2.0), np.full(4, 8.0)).tolist() == [-1, 0, 1, 0]
    # 체결: 매수는 저가 < 지정가면 지정가·fee_m, 닿기만(=) 하면 시장가
    px, fee = fill(np.array([1, 1, -1, -1]), np.full(4, 100.0), np.array([101, 101, 100, 101.0]),
                   np.array([99.9, 100, 99, 99.0]), np.full(4, 100.5), 0.0)
    assert px.tolist() == [100, 100.5, 100.5, 100] and fee.tolist() == [0, 4, 4, 0]
    # 위상: 23:55 시작 봉 → T0 00:00 다음 날 귀속, 위상 0..42 step 6
    ts = pd.Series(pd.date_range("2023-01-31", periods=2000, freq="5min"))
    n = len(ts)
    A = dict(ts=ts, c=np.full(n, 100.0), h=np.full(n, 100.1), l=np.full(n, 99.9))
    pred = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    lo, hi = np.full(n, -0.5), np.full(n, 0.5)
    g = globals(); old = g["thresholds"]; g["thresholds"] = lambda p: (lo, hi)
    try:
        daily, tr, nbad = part1(A, pred)
    finally:
        g["thresholds"] = old
    k = ((tr.T0 - pd.Timedelta(minutes=5) - pd.Timestamp("2023-01-31")) // pd.Timedelta(minutes=5)).to_numpy()
    assert set(k % 48) <= set(range(0, 48, 6)) and (tr.T0 >= "2023-02-01").all() and nbad == 0
    # 평탄 가격, 저가<지정가 → 매수·매도 모두 지정가 체결, 손익 0
    assert np.allclose(tr.pnl, 0) and tr.maker_in.all() and tr.maker_out.all()
    # 개장 되돌림: 00:00~00:30 급등 → 숏
    ts = pd.Series(pd.date_range("2023-01-01", periods=288 * 3, freq="5min"))
    c = 100 * np.exp(np.random.default_rng(0).normal(0, 1e-4, len(ts)).cumsum())
    k = 288 * 2 - 1  # 23:55 of day 2
    c[k + 1:] *= 1.01  # 00:00~ 급등 → f30 ≈ +100bp
    c[k + 54:] *= 0.99
    s3 = part3(dict(ts=ts, c=c))
    d = pd.Timestamp("2023-01-03")
    assert abs(s3[d] - (-np.log(c[k + 54] / c[k + 6]) * 1e4 - 4)) < 1e-9 and s3[d] > 90
    # 일봉 추세: 꾸준한 상승 → 91일째부터 롱, 비용은 진입 날 한 번
    ts = pd.Series(pd.date_range("2023-01-01", periods=288 * 120, freq="5min"))
    c = np.exp(np.arange(len(ts)) / 288 * 1e-3)
    s2 = part2(dict(ts=ts, c=c))
    assert (s2.iloc[:91] == 0).all() and abs(s2.iloc[91] - (10 - 4)) < 1e-6 and abs(s2.iloc[92] - 10) < 1e-6
    # 위험 균등: 같은 분포 두 열 → 가중 ≈1
    rng = np.random.default_rng(1)
    P = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(0, 3, 300)}, index=pd.date_range("2023-01-01", periods=300))
    cb = risk_parity(P)
    assert len(cb) == 300 - 60 and abs(cb.std() - np.sqrt(8) / 2) < 0.3  # 가중 ≈ 2·0.67 → 두 기여 sd 2
    print("selftest OK")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if "--selftest" in sys.argv: selftest()
    if "--preds" in sys.argv: run_preds()
    if "--report" in sys.argv: report()
