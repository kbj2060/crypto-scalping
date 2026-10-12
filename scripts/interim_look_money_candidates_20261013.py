"""돈 벌 후보 3개 «지금까지 데이터» 중간 열람 (2026-10-13). 사용자 «돈을 벌 가능성이 남은 것들은 지금까지 데이터로 테스트 한 번씩 다 해줘».

🔴사전등록 판정 전 열람: 3자산 결합(동결 10-10, 전진 창 09-15~) · 추세4주(동결 10-07, 창 10-08~) · σ 사이징(동결 10-07, 창 10-08~) 의
전진 창을 판정일 전에 한 번 봤다. 판정 규칙·날짜는 바꾸지 않고, 판정 때 «10-13 중간 열람 1회»를 같이 적는다. 각 판정 스크립트의 looks.json 은 쓰지 않는다.
계산은 전부 기존 모듈 그대로: 3자산 = research_eth_3asset_blind_rebuild_20261012(사양만으로 독립 재구성, 원 값 재현) ·
패널 = research_eth_best_direction_model_20261010.build · 추세 = prereg_eth_trend4_fwd_20261007(trend_block) · σ 사이징 = prereg_eth_pingpong_vol_sizing_fwd_20261007.

    python3 scripts/interim_look_money_candidates_20261013.py --fetch     # 09-14~10-11 20코인 5분·metrics · ETH 프리미엄·현물 5분 · ETH 1분 (data.binance.vision)
    (체결·호가: build_aggtrades_orderflow_features_20260915.py SYM 2026-08-23 2026-10-11 tmp/interim_20261013/new/aggfeat · build_bookdepth_history_20260915.py 2026-08-23 2026-10-11 tmp/interim_20261013/new)
    python3 scripts/interim_look_money_candidates_20261013.py --panels    # 확장 BV → 3자산 패널 재생성 + 원본 패널과 겹침(~09-12) 대조
    python3 scripts/interim_look_money_candidates_20261013.py --combo     # 3자산: 캐시 예측 + 2026-09·10월만 재학습(09-01~캐시 끝 대조)
    python3 scripts/interim_look_money_candidates_20261013.py --trend
    python3 scripts/interim_look_money_candidates_20261013.py --pingpong
공유 데이터(data/binance_vision)는 건드리지 않는다 — 확장분은 tmp/interim_20261013/ 에만. 🔴fapi REST 안 씀(정적 아카이브만).
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · saved_parent_exit_timestamps_used=false · future_rows_used_for_entry=false
"""
import concurrent.futures as cf
import glob, json, os, sys, urllib.request, zipfile
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
BV0 = Path("/home/kbj20/crypto-scalping/data/binance_vision")
OUTI = ROOT / "tmp/interim_20261013"
NEW, BVX = OUTI / "new", OUTI / "bv"
ORIG_PANELS = Path("/home/kbj20/crypto-scalping/.claude/worktrees/liquidation-price-feature-14eceb/tmp/best_dir_20261010")
BLIND_CACHE = Path("/home/kbj20/crypto-scalping/.claude/worktrees/agent-a1a6a08b0c163256e/tmp/blind_rebuild_20261012")
VISION = "https://data.binance.vision/data"
D_NEW0, END_DAY = "2026-09-14", "2026-10-11"           # 09-14 = 겹침 대조용 하루
FRESH_3A = pd.Timestamp("2026-09-15")


def coins20():
    import research_eth_best_direction_model_20261010 as BD
    return ["ETH"] + BD.COINS


# ── 받기 ────────────────────────────────────────────────────────────────
def _get(path, dest):
    if dest.exists():
        return 0
    try:
        raw = urllib.request.urlopen(f"{VISION}/{path}", timeout=60).read()
    except Exception:  # noqa: BLE001 -- 아직 안 올라온 날
        return -1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return 1


def fetch():
    days = lambda a: [d.strftime("%Y-%m-%d") for d in pd.date_range(a, END_DAY)]   # noqa: E731
    jobs = []
    for c in coins20():
        s = f"{c}USDT"
        for d in days(D_NEW0):
            jobs.append((f"futures/um/daily/klines/{s}/5m/{s}-5m-{d}.zip", NEW / "k5" / f"{s}-5m-{d}.zip"))
            jobs.append((f"futures/um/daily/metrics/{s}/{s}-metrics-{d}.zip", NEW / "metrics" / f"{s}-metrics-{d}.zip"))
    for d in days("2026-10-01"):
        jobs.append((f"futures/um/daily/premiumIndexKlines/ETHUSDT/5m/ETHUSDT-5m-{d}.zip", NEW / "premium" / f"ETHUSDT-5m-{d}.zip"))
        jobs.append((f"spot/daily/klines/ETHUSDT/5m/ETHUSDT-5m-{d}.zip", NEW / "spot" / f"ETHUSDT-5m-{d}.zip"))
    for d in days("2026-09-15"):
        jobs.append((f"futures/um/daily/klines/ETHUSDT/1m/ETHUSDT-1m-{d}.zip", NEW / "k1m" / f"ETHUSDT-1m-{d}.zip"))
    with cf.ThreadPoolExecutor(8) as ex:
        r = list(ex.map(lambda j: _get(*j), jobs))
    print("받음", r.count(1), "있음", r.count(0), "없음", r.count(-1), "/", len(r))
    assert r.count(-1) == 0, "빠진 날이 있다 — END_DAY 를 줄일 것"


def _csv(z):
    with zipfile.ZipFile(z) as zf:
        d = pd.read_csv(zf.open(zf.namelist()[0]))
    if str(d.columns[0])[0].isdigit():                                       # 머리줄 없는 형식(현물)
        with zipfile.ZipFile(z) as zf:
            d = pd.read_csv(zf.open(zf.namelist()[0]), header=None)
        d.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
                     "taker_buy_volume", "taker_buy_quote_volume", "ignore"][:d.shape[1]]
    return d


def _ms(x):
    x = np.asarray(x, dtype="int64")
    return np.where(x > 10**14, x // 1000, x)                                  # 현물 2025~ 는 µs


def new_panel_rows(sym):
    k = pd.concat([_csv(z) for z in sorted((NEW / "k5").glob(f"{sym}USDT-5m-*.zip"))])
    k = pd.DataFrame({"timestamp": pd.to_datetime(_ms(k.open_time), unit="ms"), "high": k.high.astype(float), "low": k.low.astype(float),
                      "close": k.close.astype(float), "volume": k.volume.astype(float), "trades": k["count"].astype("int64"),
                      "taker_buy_base": k.taker_buy_volume.astype(float)})
    m = pd.concat([_csv(z) for z in sorted((NEW / "metrics").glob(f"{sym}USDT-metrics-*.zip"))])
    m["timestamp"] = pd.to_datetime(m.create_time)
    cols = ["sum_open_interest", "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
    return k.merge(m[["timestamp"] + cols].drop_duplicates("timestamp"), on="timestamp", how="left")


def ext_bv():
    """확장 BV: panel 20개 = 원본 + 새 행 · 프리미엄·현물 = 원본 + 10월 · aggfeat·bookDepth = 원본 링크 + 새 파일."""
    (BVX / "panel").mkdir(parents=True, exist_ok=True)
    (BVX / "aggfeat").mkdir(exist_ok=True)
    rep = {}
    for c in coins20():
        o = pd.read_parquet(BV0 / f"panel/{c}USDT.parquet")
        n = new_panel_rows(c)
        ov = o.merge(n, on="timestamp", suffixes=("", "_n"))                   # 09-14 겹침: 아카이브 = 원본?
        rep[c] = {col: float(np.nanmax(np.abs(ov[col] - ov[col + "_n"]))) for col in ("close", "volume", "sum_open_interest")}
        rep[c]["n_overlap"] = len(ov)
        pd.concat([o, n[n.timestamp > o.timestamp.max()][o.columns]], ignore_index=True).to_parquet(BVX / f"panel/{c}USDT.parquet")
    bad = {k: v for k, v in rep.items() if v["n_overlap"] < 288 or max(v[x] for x in ("close", "volume", "sum_open_interest")) > 1e-6}
    print("09-14 겹침(원본 vs 아카이브):", bad or "20코인 전부 288행 일치")
    for nm, sub in (("eth_premium_5m.parquet", "premium"), ("eth_spot_5m.parquet", "spot")):
        o = pd.read_parquet(BV0 / nm)
        k = pd.concat([_csv(z) for z in sorted((NEW / sub).glob("ETHUSDT-5m-*.zip"))])
        n = pd.DataFrame({"open_time": _ms(k.open_time).astype(float), "close": k.close.astype(float), "volume": k.volume.astype(float),
                          "tb": k.taker_buy_volume.astype(float)})
        pd.concat([o, n[n.open_time > o.open_time.max()]], ignore_index=True).to_parquet(BVX / nm)
    for src, dst, pat in ((BV0 / "aggfeat", BVX / "aggfeat", "*.parquet"), (BV0, BVX, "*_bookDepth_*.parquet"),
                          (NEW / "aggfeat", BVX / "aggfeat", "*.parquet"), (NEW, BVX, "*_bookDepth_*.parquet")):
        for f in glob.glob(str(src / pat)):
            d = dst / os.path.basename(f)
            if not d.exists():
                d.symlink_to(f)


def panels():
    import research_eth_best_direction_model_20261010 as BD
    ext_bv()
    BD.BV, BD.OUT = BVX, str(OUTI / "panels")
    os.makedirs(BD.OUT, exist_ok=True)
    out = {}
    for sym in ("ETH", "BTC", "SOL"):
        BD.build(sym); BD.add_open_feats(sym)
        a = pd.read_parquet(BD.panel_path(sym))
        o = pd.read_parquet(ORIG_PANELS / ("panel.parquet" if sym == "ETH" else f"panel_{sym}.parquet"))
        cols = [c for c in o.columns if c != "ts" and not c.startswith("y_")]
        m = o.ts <= pd.Timestamp("2026-09-12 23:55")
        A, O = a.set_index("ts").loc[o.ts[m], cols].to_numpy(float), o.loc[m, cols].to_numpy(float)
        bad = ~((A == O) | (np.isnan(A) & np.isnan(O)) | np.isclose(A, O, rtol=1e-9, atol=1e-12))
        out[sym] = dict(rows=int(m.sum()), cols=len(cols), mismatch_cells=int(bad.sum()),
                        worst=sorted(((cols[j], int(bad[:, j].sum())) for j in np.flatnonzero(bad.any(0))), key=lambda x: -x[1])[:8],
                        end=str(a.ts.max()), coverage_fresh={g: round(float(a.loc[a.ts >= FRESH_3A, [c for c in cols if c[0] == g]].notna().mean().mean()), 3)
                                                             for g in "MDBSO" if any(c[0] == g for c in cols)})
        print(sym, json.dumps(out[sym], ensure_ascii=False), flush=True)
    json.dump(out, open(OUTI / "panel_parity.json", "w"), ensure_ascii=False, indent=1)


# ── 3자산 ───────────────────────────────────────────────────────────────
def eth_1m():
    import research_eth_liq_wall_break_conditions_20261013 as R
    R.DL, R.END = str(NEW / "k1m"), (pd.Timestamp(END_DAY) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return R.load_1m()


def month_preds(ts, X, y, M, seeds=(0, 1, 2)):
    """BL.monthly_preds 의 한 달치(같은 창·간격·하이퍼·시드)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    import research_eth_3asset_blind_rebuild_20261012 as BL
    t = ts.to_numpy(); M = pd.Timestamp(M); end = M - pd.Timedelta(minutes=240)
    tr = np.flatnonzero((t >= np.datetime64(end - pd.Timedelta(days=365))) & (t < np.datetime64(end)) & ~np.isnan(y))[::2]
    te = np.flatnonzero((t >= np.datetime64(M)) & (t < np.datetime64(M + pd.offsets.MonthBegin(1))))
    return te, np.mean([HistGradientBoostingRegressor(random_state=s, **BL.HGB).fit(X[tr], y[tr]).predict(X[te]) for s in seeds], axis=0)


def window_stats(x, a, b):
    import research_eth_3asset_blind_rebuild_20261012 as BL
    s = x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]
    out = {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in BL.stats(s).items()}
    out["sum_bp"] = round(float(s.sum()), 1)
    return out


WINDOWS = (("pre_~09-12", (None, "2026-09-12")), ("fresh_09-15~10-11", (FRESH_3A, END_DAY)), ("all", (None, END_DAY)))


def combo():
    import research_eth_best_direction_model_20261010 as BD
    import research_eth_3asset_blind_rebuild_20261012 as BL
    BD.BV = BVX
    BL.PANEL_DIR, BL.D1 = str(OUTI / "panels") + "/", pd.Timestamp(END_DAY)
    win = [(nm, (a or BL.D0, b)) for nm, (a, b) in WINDOWS]
    rep, daily = {"pred_check": {}}, {}
    k1 = eth_1m()
    for sym in BL.SYMS:
        A = BL.load(sym)
        y = BL.label(A["c"])
        old = np.load(BLIND_CACHE / f"pred_{sym}_base.npy")
        pred = np.full(len(y), np.nan); pred[:len(old)] = old
        for M in ("2026-09-01", "2026-10-01"):
            te, p = month_preds(A["ts"], A["X"], y, M)
            if M == "2026-09-01":                                              # 재학습 루프 = 원본? 원본처럼 08-23~ 체결·호가를 비우면 캐시와 같아야 한다
                Xo = A["X"].copy()
                Xo[np.ix_((A["ts"] >= "2026-08-23").to_numpy(), np.array([c[0] in "MD" for c in A["cols"]]))] = np.nan
                _, po = month_preds(A["ts"], Xo, y, M)
                ov = te < len(old)
                rep["pred_check"][sym] = dict(n=int(ov.sum()), max_abs_orig_condition=float(np.nanmax(np.abs(po[ov] - old[te[ov]]))),
                                              max_abs_filled=float(np.nanmax(np.abs(p[ov] - old[te[ov]]))))
            pred[te] = p
        if sym == "ETH":                                                       # 지정가 체결 = 1분봉(원 사양), 패널 첫 봉 기준 분 인덱스
            t0 = int(pd.Timestamp("2022-01-01").value // 10**6); n = len(A["c"]) * 5 + 600
            h, l, c = (np.full(n, np.nan) for _ in range(3))
            i = ((k1.t.to_numpy() - t0) // 60000).astype(np.int64); m = (i >= 0) & (i < n)
            h[i[m]], l[i[m]], c[i[m]] = k1.h.to_numpy()[m], k1.l.to_numpy()[m], k1.c.to_numpy()[m]
            A["m1"] = (h, l, c)
        p1, tr, nbad = BL.part1(A, pred)
        P = BL.frame({"hgb": p1, "trend": BL.part2(A), "open": BL.part3(A)})
        P["combo"] = BL.risk_parity(P[["hgb", "trend", "open"]])
        daily[sym] = P
        rep[sym] = {nm: {k: window_stats(P[k], a, b) for k in P} for nm, (a, b) in win}
        f = tr[tr.T0 >= FRESH_3A]
        rep[sym]["fresh_trades"] = dict(n=len(f), maker_in=float(f.maker_in.mean()) if len(f) else None, dropped_nan=nbad)
        print(sym, json.dumps({k: {x: rep[sym][nm][k][x] for x in ("sharpe", "bp_day", "sum_bp")} for nm, _ in win for k in P}, ensure_ascii=False), flush=True)
    C = pd.DataFrame({s: daily[s]["combo"] for s in BL.SYMS}).dropna(how="all").fillna(0.0)
    c3 = BL.risk_parity(C)
    rep["3asset"] = {nm: window_stats(c3, a, b) for nm, (a, b) in win}
    rep["3asset_fresh_daily_bp"] = {str(d.date()): round(float(v), 1) for d, v in c3[c3.index >= FRESH_3A].items()}
    pd.DataFrame({**{s: C[s] for s in BL.SYMS}, "combo": c3}).to_parquet(OUTI / "daily_3asset.parquet")
    json.dump(rep, open(OUTI / "report_combo.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: rep[k] for k in ("pred_check", "3asset")}, ensure_ascii=False, indent=1, default=float))


# ── 추세 ────────────────────────────────────────────────────────────────
def trend():
    import prereg_eth_trend4_fwd_20261007 as T4
    rep = {}
    for sym in ("ETHUSDT", "BTCUSDT", "SOLUSDT"):
        T4.SYM, T4.WARM = sym, ("2022-05" if sym == "SOLUSDT" else "2022-01")     # SOL 선물 일봉 아카이브는 2022-02·04 에 빈 날이 있어 거기서 잘린다
        T4.closes(pd.Timestamp(END_DAY) + pd.Timedelta(days=1)) if sym == "ETHUSDT" else None
        for z in (T4.R.RAW / f"futures/um/monthly/klines/{sym}/1d").glob("*.zip"):   # 옛 월별 파일 일부는 머리줄이 없다(SOL 2022-03) → 붙인 사본으로
            with zipfile.ZipFile(z) as zf:
                nm, txt = zf.namelist()[0], zf.read(zf.namelist()[0]).decode()
            if txt[:1].isdigit():
                hdr = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
                with zipfile.ZipFile(z, "w") as zf:
                    zf.writestr(nm, hdr + txt)
        c = T4.closes(pd.Timestamp(END_DAY) + pd.Timedelta(days=1))
        fund, cov = T4.funding(c.index)                                        # 10월 펀딩 월별 아카이브 미공개 → 0
        rep[sym] = {"close_end": str(c.index[-1].date()), "funding_until": str(cov.date())}
        for nm, main in (("2025~", "2025-01-01"), ("2026~", "2026-01-01"), ("발표 09-27~", "2026-09-27"), ("동결 10-08~", "2026-10-08")):
            T4.R.MAIN = pd.Timestamp(main)
            t = T4.R.trend_block(c, fund)
            rep[sym][nm] = {a: t[a]["main"]["net"] for a in ("trend4_vs", "always_long")}
        rep[sym]["2022~"] = {a: t[a]["all"]["net"] for a in ("trend4_vs", "always_long")}
        print(sym, json.dumps(rep[sym], ensure_ascii=False, default=float)[:3000], flush=True)
    T4.R.MAIN = pd.Timestamp("2025-01-01")
    json.dump(rep, open(OUTI / "report_trend.json", "w"), ensure_ascii=False, indent=1, default=float)


# ── σ 사이징 ─────────────────────────────────────────────────────────────
def pingpong():
    import prereg_eth_pingpong_vol_sizing_fwd_20261007 as PP
    k = eth_1m()[["t", "h", "l", "c", "v", "tb"]]
    T = PP.trades(k, PP.candidates(k))
    rep = {}
    for nm, lo in (("2022~", "2022-01-01"), ("2025~", "2025-01-01"), ("2026~", "2026-01-01"), ("동결 10-08~", "2026-10-08")):
        sub = T[T.t >= pd.Timestamp(lo).value // 10**6].reset_index(drop=True)
        rep[nm] = PP.verdict(sub, 0) if sub.day.nunique() >= 3 else {"n": len(sub), "days": int(sub.day.nunique())}
        rep[nm]["sum_bp_equal"] = float(sub.pnl.sum())
        rep[nm]["sum_bp_weighted"] = float((sub.pnl * sub.mult / sub.mult.mean()).sum()) if len(sub) else 0.0
        print(nm, json.dumps(rep[nm], ensure_ascii=False, default=float), flush=True)
    json.dump(rep, open(OUTI / "report_pingpong.json", "w"), ensure_ascii=False, indent=1, default=float)


def selftest():
    assert _ms([1_727_740_800_000_000, 1_727_740_800_000]).tolist() == [1_727_740_800_000, 1_727_740_800_000]
    print("selftest OK")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    {"--fetch": fetch, "--panels": panels, "--combo": combo, "--trend": trend, "--pingpong": pingpong, "--selftest": selftest}[sys.argv[1]]()
