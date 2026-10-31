#!/usr/bin/env python3
"""BTC·SOL 박스 유지 — 코인 전용 HGB(30분·1h·4h, XRP 10-20 관문) + 1h 굴림 Platt90(XRP 10-31 Q1). 사전등록 docs/prereg/btc_sol_box_hold_20261031.md.

라벨·특징·학습·관문 = scripts/research_xrp_box_hold_20261020.py 그대로 import(심볼만 바꿈) · Q1 = research_xrp_box_hold_platt90_qx4_20261031.q1 그대로.
양성 대조 먼저: XRP 를 같은 경로로 다시 돌려 10-20 특징·배포 모델·10-31 Q1 숫자와 맞춘다(실패면 BTC·SOL 채점 안 함).
    OMP_NUM_THREADS=3 nice python scripts/research_btc_sol_box_hold_20261031.py --selftest
    OMP_NUM_THREADS=3 nice python scripts/research_btc_sol_box_hold_20261031.py --run      # 한 번만(report.json 있으면 거절)
실주문 없음 · 바이낸스 REST 호출 없음(XRP 스크립트의 live_check 는 부르지 않는다).
"""
from __future__ import annotations

import gc
import glob
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import research_xrp_box_hold_20261020 as X                 # noqa: E402 -- ms·imp·features·evaluate·export·leak_check
import research_xrp_box_hold_platt90_qx4_20261031 as P     # noqa: E402 -- q1 · C(labels·rolling·metrics·day_boot)
import dashboard.box_hold as bh                            # noqa: E402

OUT = ROOT / "tmp/box_btc_sol_20261031"
XRP_OLD = X.OUT                                            # 메인 tmp/xrp_parity_20261020/box_hold (10-20 산출)
XRP_Q1 = ROOT / "docs/experiments/xrp_box_hold_platt90_qx4_20261031_report.json"
END = {"XRPUSDT": "2026-10-20", "BTCUSDT": "2026-10-28", "SOLUSDT": "2026-10-28"}   # 1분봉 끝 다음 분(10-28 일 파일 없음)
MAIN = "XRP HGB(배포 구성)"                                 # X.evaluate 의 주 모델 키(이름만 XRP)
MIN = pd.Timedelta("1min")
log = X.log


def load_1m(sym: str) -> pd.DataFrame:
    fs = sorted(glob.glob(f"{X.BV}/klines1m/{sym}-1m-*.parquet")) + sorted(glob.glob(f"{X.BV}/klines1m_daily/{sym}-1m-*.parquet"))
    K = pd.concat([pd.read_parquet(f)[["t", "h", "l", "c", "v", "tb"]] for f in fs]).drop_duplicates("t").sort_values("t")
    K = K[K.t < pd.Timestamp(END[sym]).value // 10 ** 6].reset_index(drop=True)
    assert K.t.iloc[0] == pd.Timestamp("2022-01-01").value // 10 ** 6 and (np.diff(K.t) == 60000).all(), "1분봉 빈칸"
    assert K.t.iloc[-1] == pd.Timestamp(END[sym]).value // 10 ** 6 - 60000
    return K


def build(sym: str):
    a = sym[:-4].lower()
    out = OUT / a
    out.mkdir(parents=True, exist_ok=True)
    X.SYM, X.OUT, X.ART = sym, out, out / f"box_hold_hgb_{a}_20261031.joblib"
    X.imp.OUT = out
    shutil.copy(XRP_OLD / "events.json", out / "events.json")   # 발표 일정 = 자산 무관(md5 fe960e7b)
    K = load_1m(sym)
    log(f"\n===== {sym} 1분봉 {len(K):,} · {pd.Timestamp(K.t.iloc[0], unit='ms')} ~ {pd.Timestamp(K.t.iloc[-1], unit='ms')}")
    D = X.ms.build(K[["t", "h", "l", "c"]])
    prof = X.season_prof(K)
    X.leak_check(D, K, prof)
    eth = joblib.load(X.ETH_ART)
    keep = D[["scale", "ts", "hold"] + X.F_BASE]
    A = pd.concat([keep, X.features(D, K, prof)], axis=1).replace([np.inf, -np.inf], np.nan)
    Ae = pd.concat([keep, X.features(D, K, np.asarray(eth["prof"], float))], axis=1).replace([np.inf, -np.inf], np.nan)
    del D; gc.collect()
    rep = X.evaluate(A, Ae, eth)
    del Ae; gc.collect()
    X.export(A, prof, {nm: rep[nm]["gate"] for nm in X.SCALES})
    art = joblib.load(X.ART)
    art["model"] = f"hgb-{a}-20261031"                          # X.export 는 이름이 hgb-xrp-20261020 고정
    joblib.dump(art, X.ART, compress=3)
    for nm in X.SCALES:   # 완화 «통과 후보»: CI 하한 대신 ΔAUC 점추정 > 0
        m = [rep[nm]["periods"][p]["models"][MAIN] for p in ("2025", "2026")]
        rep[nm]["gate"]["candidate"] = all(x["auc"] >= .72 and x["cal_maxgap_pp"] <= 6 and x["dauc_vs[G 기하]"] > 0 for x in m)
    return a, K, A, rep, art


def parity(a: str, A: pd.DataFrame, K: pd.DataFrame, art: dict) -> float:
    """X.parity 와 같은 40행 -- 단 서빙 경로 = payload(asset=a)(10-31 리팩토링 뒤 X._bh 의 bh.B 바꿔끼기는 payload 에 안 닿는다)."""
    bh.BUNDLES[a] = art                                         # 이 프로세스 안에서만
    bh.PLATT90.pop(a, None)                                     # 보정 파일 없음 = 원래 p
    evl = X._evl()
    S = A[A.ts >= "2025-01-01"].dropna(subset=X.LEAN)
    pick = pd.concat([S[S.M_major_in == 1].sample(10, random_state=1), S.sample(30, random_state=2)])
    pos = pd.Series(np.arange(len(K)), index=K.t)
    worst = 0.0
    for _, row in pick.iterrows():
        i = int(pos[int(row.ts.value // 10 ** 6)])
        k = K.iloc[i - bh.NEED + 1:i + 1]
        k1m = pd.DataFrame({"timestamp": pd.to_datetime(k.t, unit="ms", utc=True), "high": k.h, "low": k.l, "close": k.c,
                            "volume": k.v, "taker_buy_base": k.tb}).reset_index(drop=True)
        cf = art["scales"][row.scale]
        f = bh.features(k1m, evl, np.asarray(art["prof"], float))[row.scale]
        x, y = np.array([f[c] for c in cf["cols"]]), row[cf["cols"]].to_numpy(float)
        worst = max(worst, float(np.max(np.abs(x - y))))
        assert np.allclose(x, y, rtol=1e-6, atol=1e-8), (row.scale, row.ts)
        s = bh.payload(k1m, evl, asset=a)["scales"][row.scale]
        if cf["deployable"]:
            assert abs(s["p"] - cf["model"].predict_proba(row[cf["cols"]].to_frame().T.astype(float))[0, 1]) < 1e-4
        else:
            assert s["ok"] is False
    log(f"패리티 OK — 연구 행 {len(pick)}개 특징 최대 차 {worst:.2e} · payload(asset={a}) 확률 같음")
    return worst


def aux(A: pd.DataFrame, art: dict) -> dict:
    """보조: 사용자 분할 창 · 시드 1 재적합(판정에 안 씀)."""
    win = {"2025": ("2025-01-01", "2026-01-01"), "2026": ("2026-01-01", "2027-01-01"),
           "val_2026_01_05": ("2026-01-01", "2026-06-01"), "test_2026_06": ("2026-06-01", "2027-01-01")}
    r = {}
    for nm, cf in art["scales"].items():
        g = A[A.scale == nm].dropna(subset=X.LEAN)
        tr, te = g[g.ts < "2025-01-01"], g[g.ts >= "2025-01-01"]
        P_ = {"seed0": cf["model"].predict_proba(te[cf["cols"]])[:, 1],
              "seed1": X.hgb().set_params(random_state=1).fit(tr[cf["cols"]], tr.hold).predict_proba(te[cf["cols"]])[:, 1]}
        r[nm] = {}
        for k, (lo, hi) in win.items():
            mk = ((te.ts >= lo) & (te.ts < hi)).to_numpy()
            y = te.hold.to_numpy()[mk]
            r[nm][k] = {"n": int(mk.sum()), "days": int(te.ts[mk].dt.floor("D").nunique()), "hold_rate": float(y.mean())}
            for s, p in P_.items():
                r[nm][k][s] = {"auc": float(roc_auc_score(y, p[mk])), "cal_maxgap_pp": X.reliability(y, p[mk])[0] * 100}
        log(f"  보조 {nm}: " + " · ".join(f"{k} AUC {v['seed0']['auc']:.4f}/{v['seed1']['auc']:.4f} 보정 {v['seed0']['cal_maxgap_pp']:.1f}/{v['seed1']['cal_maxgap_pp']:.1f}"
                                         for k, v in r[nm].items()))
    return r


def q1(sym: str, A: pd.DataFrame, K: pd.DataFrame, bundle: dict) -> dict:
    """P.rows 와 같은 행(1h · 피쳐 유한 · 라벨 창 끝 ≤ END · 폭 > 0) + L1 라벨 독립 재계산 → P.q1 그대로."""
    E = pd.Timestamp(END[sym])
    cf = bundle["scales"]["1h"]
    R = A[A.scale == "1h"].dropna(subset=cf["cols"]).sort_values("ts").reset_index(drop=True)
    R = R[(R.ts + 61 * MIN <= E) & (np.exp(R.log_rng) > 0)].reset_index(drop=True)
    R["p"] = cf["model"].predict_proba(R[cf["cols"]])[:, 1]
    y = P.C.labels(*(K[k].to_numpy(float) for k in ("h", "l", "c")), 60)
    i = pd.Series(np.arange(len(K)), index=K.t).reindex(R.ts.astype("int64").to_numpy() // 10 ** 6).to_numpy()
    assert np.isfinite(i).all()
    l1 = float((y[i.astype(int)] == R.hold.to_numpy()).mean())
    assert l1 == 1.0, ("L1", l1)
    P.END = E
    P.J = {"J1": ("2025-02-01", "2026-01-01"), "J2": ("2026-01-01", E)}
    P.AUX1 = {"val_2026_01_05": ("2026-01-01", "2026-06-01"), "test_2026_06": ("2026-06-01", E)}
    r = P.q1(R, cf, 60)
    r["L1"] = {"rows": len(R), "match": l1}
    pl = r["pooled"]
    log(f"  Q1 {sym}: n {pl['n']:,}·{pl['days']}일 ΔBrier {pl['dbrier']['mean']:+.5f} 단측90 {pl['dbrier']['one_sided90']} "
        f"ECE {pl['A']['ece_pp']:.2f}→{pl['B']['ece_pp']:.2f} {r['verdict']} 통과 {r['pass']} 후보 {r['candidate']}")
    return r


def control() -> dict:
    """XRP 양성 대조 P0a·P0b·P0c (사전등록)."""
    a, K, A, rep, art = build("XRPUSDT")
    old = pd.read_parquet(XRP_OLD / "features.parquet")
    key = ["scale", "ts"]
    m = A.merge(old, on=key, suffixes=("", "_o"))
    assert len(m) == len(A) == len(old), (len(m), len(A), len(old))
    cols = ["hold"] + X.LEAN
    d = np.nanmax(np.abs(m[cols].to_numpy(float) - m[[c + "_o" for c in cols]].to_numpy(float)))
    same_nan = (m[cols].isna().to_numpy() == m[[c + "_o" for c in cols]].isna().to_numpy()).all()
    del old, m; gc.collect()
    old_rep = json.loads((XRP_OLD / "report.json").read_text())
    pred, auc = {}, {}
    for nm in X.SCALES:
        g = A[(A.scale == nm) & (A.ts >= "2025-01-01")].dropna(subset=X.LEAN)
        c_new, c_dep = art["scales"][nm], P.XB["scales"][nm]
        pred[nm] = float(np.max(np.abs(c_new["model"].predict_proba(g[c_new["cols"]])[:, 1] - c_dep["model"].predict_proba(g[c_dep["cols"]])[:, 1])))
        auc[nm] = max(abs(rep[nm]["periods"][p]["models"][MAIN]["auc"] - old_rep[nm]["periods"][p]["models"][MAIN]["auc"]) for p in ("2025", "2026"))
        assert rep[nm]["gate"]["pass"] == old_rep[nm]["gate"]["pass"]
    want = json.loads(XRP_Q1.read_text())["Q1_1h"]["pooled"]["dbrier"]["mean"]
    got = q1("XRPUSDT", A, K, P.XB)["pooled"]["dbrier"]["mean"]
    c = {"P0a_feature_maxdiff": float(d), "P0a_nan_same": bool(same_nan), "P0b_pred_maxdiff": pred, "P0b_auc_maxdiff": auc,
         "P0c_q1_dbrier": got, "P0c_want": want, "parity": parity(a, A, K, art)}
    c["ok"] = bool(d <= 1e-9 and same_nan and max(pred.values()) <= 1e-9 and max(auc.values()) <= 1e-12 and abs(got - want) <= 1e-12)
    log(f"양성 대조 {c}")
    return c


def selftest():
    X.selftest(); P.selftest()
    K = load_1m("BTCUSDT")
    assert len(K) == (pd.Timestamp(END["BTCUSDT"]) - pd.Timestamp("2022-01-01")) // MIN
    log("selftest OK — 1분봉 자르기(빈칸 0)")


def run():
    rp = OUT / "report.json"
    assert not rp.exists(), "판정은 한 번만"
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"prereg": "docs/prereg/btc_sol_box_hold_20261031.md", "control": control()}
    assert rep["control"]["ok"], "양성 대조 실패 → 채점 안 함"
    gc.collect()
    for sym in ("BTCUSDT", "SOLUSDT"):
        a, K, A, r, art = build(sym)
        r = {"Q0": r, "aux": aux(A, art), "parity": parity(a, A, K, art), "artifact": str(X.ART)}
        A.to_parquet(OUT / a / "features.parquet")              # 적합 스크립트 패리티 시험 입력
        if r["Q0"]["1h"]["gate"]["pass"]:
            r["Q1_1h"] = q1(sym, A, K, art)
        rep[a] = r
        rp.with_name(f"report_{a}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float))
        del K, A; gc.collect()
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"→ {rp}")
    for a in ("btc", "sol"):
        log(f"{a}: " + " · ".join(f"{nm} {rep[a]['Q0'][nm]['gate']}" for nm in X.SCALES)
            + (f" · Q1 통과 {rep[a]['Q1_1h']['pass']} 후보 {rep[a]['Q1_1h']['candidate']}" if "Q1_1h" in rep[a] else ""))


if __name__ == "__main__":
    run() if "--run" in sys.argv else selftest() if "--selftest" in sys.argv else print(__doc__)
