#!/usr/bin/env python3
"""SOXL 4h 방향(배포 B2 HGB)·박스 유지 30분·1h(HGB) — 팔 E/S/J (사전등록 docs/prereg/soxl_dir4h_box_20261117.md).

    OMP_NUM_THREADS=3 nice python scripts/research_soxl_dir4h_box_20261117.py data      # vision 정적 파일만(REST 아님)
    OMP_NUM_THREADS=3 nice python scripts/research_soxl_dir4h_box_20261117.py selftest
    OMP_NUM_THREADS=3 nice python scripts/research_soxl_dir4h_box_20261117.py control   # 양성 대조 → control.json
    OMP_NUM_THREADS=3 nice python scripts/research_soxl_dir4h_box_20261117.py run       # 팔 적합·예측(control ok 일 때만)
    OMP_NUM_THREADS=3 nice python scripts/research_soxl_dir4h_box_20261117.py score     # 검증으로 고르고 시험 1회 → report.json(있으면 거절)
레시피는 import 그대로: 4h = live_eth_dir4h_20261015.build_features/HGB · research_btc_sol_xrp_dir4h_20261031(windows·hit_excess·boot) ·
박스 = research_xrp_box_hold_20261020(ms.build·features·hgb·reliability·leak_check) · 양성 대조 = research_btc_sol_box_hold_20261031.control.
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · saved_parent_exit_timestamps_used=false · future_rows_used_for_entry=false.
"""
from __future__ import annotations

import glob
import io
import json
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import live_eth_dir4h_20261015 as d4                       # noqa: E402
import research_btc_sol_xrp_dir4h_20261031 as R            # noqa: E402 -- windows·hgb·hit_excess·boot·spear
import research_btc_sol_box_hold_20261031 as BB            # noqa: E402 -- control (XRP 양성 대조) · X · P
from research_eth_dir4h_reverify_20261019 import side      # noqa: E402
from research_soxl_session_vol_20261112 import SESS, clock  # noqa: E402

X, P, ms = BB.X, BB.P, BB.X.ms
OUT = ROOT / "tmp/soxl_dir4h_box_20261117"
PANEL_SRC = Path("/home/kbj20/backups/crypto-scalping-server/data/live/dirshort_panel")   # 서버 11-05 사본(20코인 2022-01~11-05)
K1M_SRC = Path("/home/kbj20/crypto-scalping/.claude/worktrees/support-resistance-breakout-signals-ae2e43/tmp/soxl_vol_20261112/klines")
VBASE = "https://data.binance.vision/data/futures/um/daily"
T = {k: (np.datetime64(a), np.datetime64(b)) for k, (a, b) in
     {"train": ("2026-05-22", "2026-09-01"), "val": ("2026-09-01", "2026-10-01"), "test": ("2026-10-01", "2026-11-17")}.items()}
MONTHS = pd.date_range("2026-05-01", "2026-11-01", freq="MS")
PURGE = R.PURGE                                            # 4h(48봉 × 5분)
COINS4 = ["ETH", "BTC", "SOL", "XRP"]
NAN_DROP = .5                                              # S·J: SOXL 학습 행 결측 > 50% 피쳐는 SOXL 행 전부 NaN
MINP = 2016                                                # 문턱 예열 7일(배포 8640 = 30일 -- S·J 표본 밖 기록이 09-01 시작이라 줄임)
BOX = {"30m": 30, "1h": 60}
NB = 2000
log = lambda m: print(m, flush=True)  # noqa: E731
FLAGS = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
         "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}


# ── 데이터 ─────────────────────────────────────────────────────────────────
def k1m_vision(d: str) -> pd.DataFrame | None:
    from build_binance_vision_panel_20260915 import fetch
    s = fetch(f"{VBASE}/klines/SOXLUSDT/1m/SOXLUSDT-1m-{d}.zip")
    if s is None:
        return None
    x = pd.read_csv(io.StringIO(s))
    return pd.DataFrame({"t": x.open_time.astype("int64"), "h": x.high, "l": x.low, "c": x.close, "v": x.volume, "tb": x.taker_buy_volume})


def data():
    from build_binance_vision_panel_20260915 import day
    pdir = OUT / "panel"; pdir.mkdir(parents=True, exist_ok=True)
    ext = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-11-06", "2026-11-16")]
    with ThreadPoolExecutor(8) as ex:
        for sym in ["ETH"] + d4.COINS:
            old = pd.read_parquet(PANEL_SRC / f"{sym}USDT.parquet")
            old = old[old.timestamp >= "2023-06-01"]
            new = [p for p in ex.map(lambda d: day(f"{sym}USDT", d), ext) if p is not None]
            df = pd.concat([old] + new).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
            df.to_parquet(pdir / f"{sym}USDT.parquet")
            gap = int((np.diff(df.timestamp.to_numpy()) != np.timedelta64(5, "m")).sum())
            log(f"  {sym} {len(df):,}행 ~{df.timestamp.max()} (+{len(new)}일) 빈칸 {gap}")
        days = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-05-15", "2026-11-16")]
        s = pd.concat([p for p in ex.map(lambda d: day("SOXLUSDT", d), days) if p is not None]).drop_duplicates("timestamp").sort_values("timestamp")
        s.reset_index(drop=True).to_parquet(pdir / "SOXLUSDT.parquet")
        log(f"  SOXL 5분 {len(s):,}행 {s.timestamp.min()}~{s.timestamp.max()} · metrics 결측 {s[d4.MCOLS].isna().mean().round(4).to_dict()}")
        old = pd.concat([pd.read_parquet(f) for f in sorted(K1M_SRC.glob("SOXLUSDT-1m-*.parquet"))])
        vd = {d: k for d, k in zip(["2026-11-10"] + ext[5:], ex.map(k1m_vision, ["2026-11-10"] + ext[5:]))}
    a = old[(old.t >= pd.Timestamp("2026-11-10").value // 10**6) & (old.t < pd.Timestamp("2026-11-11").value // 10**6)].reset_index(drop=True)
    b = vd["2026-11-10"]
    diff = float(np.abs(a[["h", "l", "c", "v", "tb"]].to_numpy() - b[["h", "l", "c", "v", "tb"]].to_numpy()).max())
    assert len(a) == len(b) == 1440 and (a.t.to_numpy() == b.t.to_numpy()).all() and diff < 1e-9, ("11-10 대조", len(a), len(b), diff)
    K = pd.concat([old] + [vd[d] for d in ext[5:]]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    assert (np.diff(K.t) == 60000).all(), "SOXL 1분봉 빈칸"
    K.to_parquet(OUT / "soxl_1m.parquet")
    log(f"  SOXL 1분 {len(K):,} {pd.Timestamp(K.t.iloc[0], unit='ms')}~{pd.Timestamp(K.t.iloc[-1], unit='ms')} · 11-10 대조 최대 차 {diff}")


def panels() -> dict:
    return {s: pd.read_parquet(OUT / "panel" / f"{s}USDT.parquet") for s in ["ETH"] + d4.COINS + ["SOXL"]}


def feats(allf: dict, target: str):
    """대상 코인을 "ETH" 자리에. SOXL 은 교차 19코인에 없으므로 ETH 를 빼고 19코인 그대로(X_btc·U_btc = BTC)."""
    f = dict(allf)
    if target == "SOXL":
        f["ETH"] = allf["SOXL"]
    elif target != "ETH":
        f["ETH"], f[target] = allf[target], allf["ETH"]
    F = d4.build_features(f)
    return F.ts.to_numpy(), F.y_4h.to_numpy(), F[d4.FEATS].to_numpy(np.float32)


def inside(ts, k):
    return (ts >= T[k][0]) & (ts < T[k][1])


# ── 양성 대조 ──────────────────────────────────────────────────────────────
def control():
    allf = panels()
    ts, y, Xe = feats(allf, "ETH")
    p = np.full(len(ts), np.nan)
    for m in pd.date_range("2025-04-01", "2026-09-01", freq="MS"):
        tr, te = R.windows(ts, m); tr &= np.isfinite(y)
        p[te] = R.hgb().fit(Xe[tr][::2], y[tr][::2]).predict(Xe[te])
        log(f"  ETH {m:%Y-%m} 학습 {int(tr.sum()):,}")
    s = pd.read_parquet(R.OOS_SEED)
    seed = pd.Series(s.pred.to_numpy(float), index=s.ts).reindex(pd.DatetimeIndex(ts)).to_numpy()
    m = (ts >= np.datetime64("2025-04-01")) & (ts < np.datetime64("2026-09-15")) & np.isfinite(seed) & np.isfinite(p) & np.isfinite(y)
    d = {"n": int(m.sum()), "corr": float(np.corrcoef(p[m], seed[m])[0, 1]), "ic_mine": R.spear(p[m], y[m]), "ic_seed": R.spear(seed[m], y[m]),
         "max_abs_diff_bp": float(np.abs(p[m] - seed[m]).max())}
    d["pass"] = bool(d["corr"] >= .99 and abs(d["ic_mine"] - d["ic_seed"]) < .003)
    log(f"양성 대조 4h {d}")
    BB.OUT = OUT / "box_control"
    b = BB.control()
    c = {"dir4h": d, "box": b, "ok": bool(d["pass"] and b["ok"])}
    (OUT / "control.json").write_text(json.dumps(c, ensure_ascii=False, indent=1, default=float))
    log(f"→ control.json ok={c['ok']}")


# ── 4h 방향 팔 ─────────────────────────────────────────────────────────────
def run_dir():
    allf = panels()
    ts, y, Xs = feats(allf, "SOXL")
    lag1 = np.vstack([np.full((1, Xs.shape[1]), np.nan, np.float32), Xs[:-1]])     # 한 봉 밀기 트립와이어용
    trS = (ts >= T["train"][0]) & (ts < T["train"][1] - PURGE) & np.isfinite(y)
    nan = np.isnan(Xs[trS]).mean(0)
    drop = [d4.FEATS[j] for j in np.flatnonzero(nan > NAN_DROP)]
    Xsj, lsj = Xs.copy(), lag1.copy()
    Xsj[:, nan > NAN_DROP] = np.nan; lsj[:, nan > NAN_DROP] = np.nan
    log(f"SOXL 피쳐 {len(ts):,}행 · 학습 {int(trS.sum()):,} · 결측>50% 뺌 {drop}")
    oos = ts >= T["val"][0]
    out = {a: np.full(len(ts), np.nan) for a in ("S", "E", "J", "S_lag", "E_lag", "J_lag")}
    S = R.hgb().fit(Xsj[trS][::2], y[trS][::2])
    out["S"][oos], out["S_lag"][oos] = S.predict(Xsj[oos]), S.predict(lsj[oos])
    tsE, yE, XE = feats(allf, "ETH")
    for m in MONTHS:
        tr, _ = R.windows(tsE, m); tr &= np.isfinite(yE)
        mdl = R.hgb().fit(XE[tr][::2], yE[tr][::2])
        te = (ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1)))
        out["E"][te], out["E_lag"][te] = mdl.predict(Xs[te]), mdl.predict(lag1[te])
        log(f"  E {m:%Y-%m} ETH 학습 {int(tr.sum()):,} → SOXL {int(te.sum()):,}")
    cut = pd.Timestamp(T["train"][1])
    Xp, yp, nrow = [], [], {}
    for c in COINS4:
        tc, yc, Xc = (tsE, yE, XE) if c == "ETH" else feats(allf, c)
        tr, _ = R.windows(tc, cut); tr &= np.isfinite(yc)
        Xp.append(Xc[tr][::4]); yp.append(yc[tr][::4] / np.nanstd(yc[tr])); nrow[c] = int(tr.sum())
    Xp.append(Xsj[trS][::4]); yp.append(y[trS][::4] / np.nanstd(y[trS])); nrow["SOXL"] = int(trS.sum())
    J = R.hgb().fit(np.vstack(Xp), np.concatenate(yp))
    out["J"][oos], out["J_lag"][oos] = J.predict(Xsj[oos]), J.predict(lsj[oos])
    log(f"  J 학습 {nrow}")
    np.savez_compressed(OUT / "dir_preds.npz", ts=ts, y=y, **out)
    (OUT / "dir_meta.json").write_text(json.dumps({"dropped_SJ": drop, "nan_share_train": dict(zip(d4.FEATS, nan.round(4).tolist())),
                                                   "train_rows": nrow}, ensure_ascii=False, indent=1))


# ── 박스 팔 ────────────────────────────────────────────────────────────────
def box_setup():
    X.SCALES = dict(BOX)
    ms.SCALES = {k: v for k, v in ms.SCALES.items() if k in BOX}
    X.imp.OUT = OUT / "box"; X.imp.OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(BB.XRP_OLD / "events.json", X.imp.OUT / "events.json")


def season_prof_train(K) -> np.ndarray:
    """X.season_prof 와 같은 식 -- 학습 구간(05-22~08-31)만."""
    t = K.t.to_numpy(); r2 = np.r_[0, np.diff(np.log(K.c.to_numpy(float)))] ** 2
    ts1 = pd.to_datetime(t, unit="ms"); how = (ts1.dayofweek * 24 + ts1.hour).to_numpy()
    tr = (ts1 >= pd.Timestamp(T["train"][0])) & (ts1 < pd.Timestamp(T["train"][1]))
    prof = pd.Series(r2[tr]).groupby(how[tr]).mean().reindex(range(168)).to_numpy()
    return prof / np.nanmean(r2[tr])


def run_box():
    box_setup()
    K = pd.read_parquet(OUT / "soxl_1m.parquet")
    D = ms.build(K[["t", "h", "l", "c"]])
    prof = season_prof_train(K)
    X.leak_check(D, K, prof)
    eth = joblib.load(X.ETH_ART)
    keep = D[["scale", "ts", "hold"] + X.F_BASE]
    A = pd.concat([keep, X.features(D, K, prof)], axis=1).replace([np.inf, -np.inf], np.nan)
    Ae = pd.concat([keep, X.features(D, K, np.asarray(eth["prof"], float))], axis=1).replace([np.inf, -np.inf], np.nan)
    l1 = {}
    pos = pd.Series(np.arange(len(K)), index=K.t)
    for nm, W in BOX.items():                              # L1: 라벨 독립 재계산
        yy = P.C.labels(*(K[k].to_numpy(float) for k in ("h", "l", "c")), W)
        g = A[A.scale == nm]
        i = pos.reindex(g.ts.astype("int64").to_numpy() // 10**6).to_numpy().astype(int)
        l1[nm] = float((yy[i] == g.hold.to_numpy()).mean())
        assert l1[nm] == 1.0, ("L1", nm, l1[nm])
    fs = sorted(glob.glob(f"{X.BV}/klines1m/ETHUSDT-1m-202[234]-*.parquet"))
    KE = pd.concat([pd.read_parquet(f)[["t", "h", "l", "c", "v", "tb"]] for f in fs]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    assert (np.diff(KE.t) == 60000).all()
    DE = ms.build(KE[["t", "h", "l", "c"]])
    AE = pd.concat([DE[["scale", "ts", "hold"] + X.F_BASE], X.features(DE, KE, np.asarray(eth["prof"], float))], axis=1).replace([np.inf, -np.inf], np.nan)
    del KE, DE
    rows, meta = [], {"L1": l1, "nan_rows_dropped": {}}
    for nm, W in BOX.items():
        ok = (A.scale == nm) & A[X.LEAN].notna().all(1)
        meta["nan_rows_dropped"][nm] = int(((A.scale == nm) & ~ok).sum())
        g, ge = A[ok], Ae[ok]
        lab_end = g.ts + pd.Timedelta(minutes=W + 1)
        tr = ((g.ts >= pd.Timestamp(T["train"][0])) & (lab_end <= pd.Timestamp(T["train"][1]))).to_numpy()
        oos = (g.ts >= pd.Timestamp(T["val"][0])).to_numpy()
        gt = g[tr]
        eth_tr = AE[(AE.scale == nm) & (AE.ts < "2025-01-01")].dropna(subset=X.LEAN)
        G = LogisticRegression(C=1.0, max_iter=1000).fit(gt[["log_ds"]], gt.hold)
        S = X.hgb().fit(gt[X.LEAN], gt.hold)
        J = X.hgb().fit(pd.concat([eth_tr[X.LEAN], gt[X.LEAN]]), np.r_[eth_tr.hold.to_numpy(), gt.hold.to_numpy()])
        ec = eth["scales"][nm]
        assert ec["cols"] == X.LEAN
        o = g[oos][["scale", "ts", "hold"]].copy()
        o["p_G"] = G.predict_proba(g[oos][["log_ds"]])[:, 1]
        o["p_S"] = S.predict_proba(g[oos][X.LEAN])[:, 1]
        o["p_E"] = ec["model"].predict_proba(ge[oos][ec["cols"]])[:, 1]
        o["p_J"] = J.predict_proba(g[oos][X.LEAN])[:, 1]
        rows.append(o)
        meta[nm] = {"train_n": int(tr.sum()), "train_base": float(gt.hold.mean()), "eth_rows_J": int(len(eth_tr)),
                    "month_hold": g.groupby(g.ts.dt.strftime("%Y-%m")).hold.agg(["mean", "size"]).round(4).to_dict("index")}
        log(f"  박스 {nm} 학습 {int(tr.sum()):,} (유지 {gt.hold.mean():.3f}) · J ETH {len(eth_tr):,} · 표본 밖 {int(oos.sum()):,}")
        # 기저율 흐름: 세션 × 구간
        sess = clock((g.ts.astype("int64").to_numpy() // 10**9) + 60, W).sess.to_numpy()
        per = np.select([tr, inside(g.ts.to_numpy(), "val"), inside(g.ts.to_numpy(), "test")], ["train", "val", "test"], "other")
        meta[nm]["sess_hold"] = pd.DataFrame({"s": [SESS[k] for k in sess], "per": per, "y": g.hold.to_numpy()}).query("per != 'other'") \
            .groupby(["s", "per"]).y.agg(["mean", "size"]).round(4).reset_index().to_dict("records")
    pd.concat(rows).to_parquet(OUT / "box_preds.parquet")
    (OUT / "box_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))


def run():
    c = json.loads((OUT / "control.json").read_text())
    assert c["ok"], "양성 대조 실패 → SOXL 계산 안 함"
    run_dir()
    run_box()


# ── 채점 ───────────────────────────────────────────────────────────────────
def month_q(p, ts, minp=MINP):
    """배포 thresholds() 와 같은 «달 시작 전 표본 밖 예측 20/80, 달 안 고정» -- 예열만 minp 로."""
    lo, hi = np.full(len(ts), np.nan), np.full(len(ts), np.nan)
    for m in MONTHS:
        past = p[(ts < np.datetime64(m)) & np.isfinite(p)]
        if len(past) >= minp:
            te = (ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1)))
            lo[te], hi[te] = np.quantile(past, [.2, .8])
    return lo, hi


def sess_of(ts, H):
    return clock(pd.DatetimeIndex(ts).asi8 // 10**9, H).sess.to_numpy()


def score_dir(rng) -> dict:
    z = np.load(OUT / "dir_preds.npz"); ts, y = z["ts"], z["y"]
    mv, mt = inside(ts, "val") & np.isfinite(y), inside(ts, "test") & np.isfinite(y)
    arms = {}
    for a in ("S", "E", "J"):
        p = z[a]; lo, hi = month_q(p, ts)
        hv, ht = R.hit_excess(p[mv], y[mv], lo[mv], hi[mv]), R.hit_excess(p[mt], y[mt], lo[mt], hi[mt])
        s = side(p[mt], lo[mt], hi[mt]); g = s != 0
        arms[a] = {"val_ic": R.spear(p[mv], y[mv]), "val_hit": {"hits": f"{hv[0]}/{hv[1]}", "excess": hv[4]},
                   "test_ic": R.spear(p[mt], y[mt]), "test_pearson": float(np.corrcoef(p[mt], y[mt])[0, 1]),
                   "test_hit": {"hits": f"{ht[0]}/{ht[1]}", "rate": ht[2], "base": ht[3], "excess": ht[4], "signal_share": ht[1] / int(mt.sum())},
                   "legs_bp": {"model": float((s[g] * y[mt][g]).mean()), "all_long": float(y[mt][g].mean()), "all_short": float(-y[mt][g].mean()),
                               "long_share": float((s[g] > 0).mean())},
                   "month_ic": {f"{m:%Y-%m}": round(R.spear(p[w], y[w]), 4) for m in MONTHS
                                for w in [(ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1))) & np.isfinite(y) & np.isfinite(p)] if w.sum() > 1000}}
    pick = max(arms, key=lambda a: arms[a]["val_ic"])
    p = z[pick]; lo, hi = month_q(p, ts)
    day = pd.DatetimeIndex(ts[mt]).normalize().asi8
    R.NB, R.BLK = NB, 1
    bic, bhx = R.boot(p[mt], y[mt], lo[mt], hi[mt], day, rng)
    R.BLK = 5
    bic5, bhx5 = R.boot(p[mt], y[mt], lo[mt], hi[mt], day, rng)
    idx = np.flatnonzero(inside(ts, "test")); ok = np.isfinite(y[idx]); nday = len(idx) // 288
    circ = np.array([R.spear(np.roll(p[idx], k * 288)[ok], y[idx][ok]) for k in range(1, nday)])
    rnd = np.array([R.spear(rng.standard_normal(ok.sum()), y[idx][ok]) for _ in range(200)])
    ic, hx = arms[pick]["test_ic"], arms[pick]["test_hit"]["excess"]
    v = {"pick": pick, "val_ic": {a: round(arms[a]["val_ic"], 4) for a in arms}, "test_ic": ic,
         "ic_p10": float(np.quantile(bic, .1)), "ic_p05_p95": [float(np.quantile(bic, .05)), float(np.quantile(bic, .95))],
         "hit_excess": hx, "hit_p10": float(np.quantile(bhx, .1)), "aux_5day_block": {"ic_p10": float(np.quantile(bic5, .1)), "hit_p10": float(np.quantile(bhx5, .1))},
         "circ_q90": float(np.quantile(circ, .9)), "circ_q95": float(np.quantile(circ, .95)), "circ_pct": float((circ < ic).mean()), "n_circ": len(circ),
         "rand_q95": float(np.quantile(rnd, .95)), "n_test": int(mt.sum()), "test_days": int(len(np.unique(day))), "n_val": int(mv.sum()),
         "base_up": {k: float((y[inside(ts, k) & np.isfinite(y)] > 0).mean()) for k in ("train", "val", "test")}}
    strict = v["ic_p10"] > 0 and ic > v["circ_q95"] and ic > v["rand_q95"] and v["hit_p10"] > 0
    cand = (not strict) and ic > 0 and ic > v["circ_q90"] and hx > 0
    v["verdict"] = "엄격 통과" if strict else ("통과 후보" if cand else "불통과")
    if ic > .08:   # 한 봉 밀기 트립와이어: 피쳐를 한 봉 묵혀도 IC 가 비슷해야 한다
        v["tripwire_lag1_ic"] = R.spear(z[f"{pick}_lag"][mt], y[mt])
    ss = sess_of(ts[mt] + np.timedelta64(5, "m"), 240)
    v["sessions"] = {}
    for k, nm in enumerate(SESS):
        w = ss == k
        if w.sum() > 200:
            h = R.hit_excess(p[mt][w], y[mt][w], lo[mt][w], hi[mt][w])
            v["sessions"][nm] = {"n": int(w.sum()), "ic": round(R.spear(p[mt][w], y[mt][w]), 4), "hits": f"{h[0]}/{h[1]}", "hit_excess": h[4]}
    log(f"4h {json.dumps({k: v[k] for k in ('pick', 'val_ic', 'verdict')}, ensure_ascii=False)} 시험 IC {ic:.4f} p10 {v['ic_p10']:.4f} "
        f"적중 초과 {hx:+.4f} p10 {v['hit_p10']:+.4f} 원형95 {v['circ_q95']:.4f}")
    return {"verdict": v, "arms": arms, "meta": json.loads((OUT / "dir_meta.json").read_text())}


def auc_boot(y, a, b, day, rng, B=1000):
    codes, uniq = pd.factorize(day)
    idx = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    out = np.empty(B)
    for i in range(B):
        ii = np.concatenate([idx[k] for k in rng.integers(0, len(idx), len(idx))])
        out[i] = roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b[ii])
    return float(np.quantile(out, .10)), [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def box_metrics(y, p, pG, day, rng, boot=True):
    ce, _ = X.reliability(y, p)
    d = {"n": int(len(y)), "days": int(len(np.unique(day))), "hold_rate": float(y.mean()), "mean_p": float(p.mean()),
         "auc": float(roc_auc_score(y, p)), "cal_maxgap_pp": ce * 100, "logloss": float(log_loss(y, np.clip(p, 1e-4, 1 - 1e-4))),
         "brier": float(np.mean((p - y) ** 2)), "dauc_vs_G": float(roc_auc_score(y, p) - roc_auc_score(y, pG))}
    if boot:
        d["dauc_p10"], d["dauc_ci95"] = auc_boot(y, p, pG, day, rng)
    return d


def score_box(rng) -> dict:
    B = pd.read_parquet(OUT / "box_preds.parquet"); meta = json.loads((OUT / "box_meta.json").read_text())
    rep = {"meta": meta}
    for nm, W in BOX.items():
        g = B[B.scale == nm]
        per = {k: g[inside(g.ts.to_numpy(), k)] for k in ("val", "test")}
        arms = {}
        for a in ("S", "E", "J"):
            arms[a] = {k: box_metrics(d.hold.to_numpy(), d[f"p_{a}"].to_numpy(), d.p_G.to_numpy(), d.ts.dt.floor("D").to_numpy(), rng, boot=False)
                       for k, d in per.items()}
        pick = min(arms, key=lambda a: arms[a]["val"]["logloss"])
        t = per["test"]; y = t.hold.to_numpy(); p = t[f"p_{pick}"].to_numpy(); day = t.ts.dt.floor("D").to_numpy()
        m = box_metrics(y, p, t.p_G.to_numpy(), day, rng)
        strict = m["auc"] >= .72 and m["cal_maxgap_pp"] <= 6 and m["dauc_p10"] > 0
        cand = (not strict) and m["auc"] >= .72 and m["cal_maxgap_pp"] <= 6 and m["dauc_vs_G"] > 0
        v = {"pick": pick, "val_logloss": {a: round(arms[a]["val"]["logloss"], 5) for a in arms}, "test": m,
             "G_test": box_metrics(y, t.p_G.to_numpy(), t.p_G.to_numpy(), day, rng, boot=False),
             "verdict": "엄격 통과" if strict else ("통과 후보" if cand else "불통과")}
        ss = sess_of(t.ts.to_numpy() + np.timedelta64(1, "m"), W)
        v["sessions"] = {}
        for k, sn in enumerate(SESS):
            w = ss == k
            if w.sum() > 200 and 0 < y[w].mean() < 1:
                ce, _ = X.reliability(y[w], p[w])
                v["sessions"][sn] = {"n": int(w.sum()), "hold_rate": round(float(y[w].mean()), 4), "mean_p": round(float(p[w].mean()), 4),
                                     "auc": round(float(roc_auc_score(y[w], p[w])), 4), "cal_maxgap_pp": round(ce * 100, 2)}
        if nm == "1h" and v["verdict"] != "불통과":   # 보조: 검증 9월로 맞춘 Platt 한 번 → 시험 ΔBrier(굴림 90일은 학습 구간과 겹쳐 안 함)
            va = per["val"]; lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))  # noqa: E731
            pl = LogisticRegression(C=1e6, max_iter=1000).fit(lg(va[f"p_{pick}"].to_numpy())[:, None], va.hold)
            pb = pl.predict_proba(lg(p)[:, None])[:, 1]
            dd = pd.Series((pb - y) ** 2 - (p - y) ** 2).groupby(day).mean()
            bs = [dd.sample(len(dd), replace=True, random_state=int(rng.integers(1 << 31))).mean() for _ in range(NB)]
            v["aux_platt_val"] = {"dbrier": float(np.mean((pb - y) ** 2) - np.mean((p - y) ** 2)), "upper90": float(np.quantile(bs, .9)),
                                  "cal_after_pp": X.reliability(y, pb)[0] * 100}
        rep[nm] = {"verdict": v, "arms": arms}
        log(f"박스 {nm} 고른 팔 {pick} {v['val_logloss']} → 시험 AUC {m['auc']:.4f} 보정 {m['cal_maxgap_pp']:.2f}pp "
            f"ΔAUC vs G {m['dauc_vs_G']:+.4f} p10 {m['dauc_p10']:+.4f} → {v['verdict']}")
    return rep


def score():
    rp = OUT / "report.json"
    assert not rp.exists(), "시험은 한 번만"
    rng = np.random.default_rng(20261117)
    rep = {"prereg": "docs/prereg/soxl_dir4h_box_20261117.md", "flags": FLAGS, "control": json.loads((OUT / "control.json").read_text()),
           "dir4h": score_dir(rng), "box": score_box(rng)}
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"→ {rp}")


def selftest():
    allf = panels()
    sub = {k: v[v.timestamp >= "2026-08-01"].reset_index(drop=True) for k, v in allf.items()}
    ts, y, Xs = feats(sub, "SOXL")
    sx = sub["SOXL"].set_index("timestamp").close; btc = sub["BTC"].set_index("timestamp").close
    i = 6000; t = pd.Timestamp(ts[i]); prev = t - pd.Timedelta("5min")
    assert abs(Xs[i, d4.FEATS.index("P_r5m")] - (np.log(sx[t]) - np.log(sx[prev])) * 1e4) < 1e-3          # 대상 = SOXL
    assert abs(Xs[i, d4.FEATS.index("X_btc_r5m")] - (np.log(btc[t]) - np.log(btc[prev])) * 1e4) < 1e-3     # 교차 = BTC
    assert abs(y[i] - (np.log(sx[t + pd.Timedelta(minutes=5 * d4.H)]) - np.log(sx[t])) * 1e4) < 1e-3       # 표적 τ → τ+48
    tr = (ts >= T["train"][0]) & (ts < T["train"][1] - PURGE)
    assert ts[tr].max() + PURGE + np.timedelta64(5, "m") <= T["val"][0]                                     # 학습 라벨 끝 ≤ 검증 시작
    p = np.arange(len(ts), dtype=float); lo, hi = month_q(p, ts)
    first = np.flatnonzero(np.isfinite(lo))[0]
    assert ts[first] == np.datetime64("2026-09-01") and hi[first] == np.quantile(p[ts < ts[first]], .8)    # 08-01~ 31일 → 9월부터 문턱
    s = clock(np.array([pd.Timestamp("2026-10-05 14:00", tz="UTC").timestamp(), pd.Timestamp("2026-10-10 14:00", tz="UTC").timestamp()], int), 60).sess
    assert list(s) == [0, 4]                                                                                  # 월 10:00 ET 정규장 · 토 = 주말
    K = pd.read_parquet(OUT / "soxl_1m.parquet").iloc[:20000]
    box_setup()
    D = ms.build(K[["t", "h", "l", "c"]])
    for nm, W in BOX.items():
        yy = P.C.labels(*(K[k].to_numpy(float) for k in ("h", "l", "c")), W)
        g = D[D.scale == nm]; i = ((g.ts.astype("int64") // 10**6 - K.t.iloc[0]) // 60000).to_numpy()
        assert (yy[i] == g.hold.to_numpy()).all()                                                             # 라벨 = 독립 재구현
    rng = np.random.default_rng(0); yb = rng.integers(0, 2, 3000); dd = np.repeat(np.arange(30), 100)
    assert auc_boot(yb, yb + rng.normal(0, .1, 3000), rng.random(3000), dd, rng, B=50)[0] > .3
    log("selftest ok")


if __name__ == "__main__":
    {"data": data, "selftest": selftest, "control": control, "run": run, "score": score}[sys.argv[1]]()
