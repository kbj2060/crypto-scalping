#!/usr/bin/env python3
"""2단계 메이커 구조의 실제 수익 (2026-09-08).

  1단계 (앵커 봉 마감)  anchor47 이 되돌림 + 확신 상위 X% -> ±T 양쪽에 지정가를 건다
  2단계 (트리거 직후)   full69 가 되돌림 확인 -> 보유(메이커 진입) / 돌파로 뒤집히면 중도청산

체결되는 순간 자동으로 **발현 반대편** = 되돌림 포지션이므로 1단계 판정과 방향이 일치한다.
메이커 체결 가능성은 별도 실측(관통 ≥1틱 99.3%)에서 확인됐다.

비용: 메이커 진입 2.8bp + 테이커 청산 5.0bp = **왕복 7.8bp**(저장소 실측치).
      중도청산도 같은 구조(메이커 진입 + 테이커 청산)이고 손익만 다르다.
청산: 1분봉 first-touch, 같은 분 양쪽이면 비관적(SL 우선), 미터치면 12봉 뒤 종가.
"""
from __future__ import annotations
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "8")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B     # noqa: E402
import live_eth_breakout_features_20260908 as LF        # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT"); WDAYS = {"VAL": 122, "OOS": 90, "HOLDOUT_SPENT": 122}
H, TM, P = 12, 0.75, 0.0025
EMB = pd.Timedelta(hours=4); SEED, CHUNK = 20260908, 4000
COST_MAKER_RT, COST_TAKER_RT = 7.8, 10.0
S1_COVS = (0.5, 0.3, 0.2, 0.15)          # 1단계(주문 게이트) 커버리지
TPS, SLS = (20, 25, 30), (30, 40, 50)


def first_touch(hi1, lo1, start, up, dn, nmin):
    n = len(start); tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = start[a:b, None] + np.arange(nmin)[None, :]
        hu = hi1[idx] >= up[a:b, None]; hd = lo1[idx] <= dn[a:b, None]
        au = hu.any(axis=1); ad = hd.any(axis=1)
        tu[a:b] = np.where(au, hu.argmax(axis=1), -1); td[a:b] = np.where(ad, hd.argmax(axis=1), -1)
    return tu, td


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = pd.read_parquet(MY / "dataset_v2.parquet")
    d = d[(d.anchor == "first_fire") & (d.T_mult == TM)].sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    H5 = eth["high"].to_numpy(float); L5 = eth["low"].to_numpy(float)
    V5 = eth["volume"].to_numpy(float); TB = eth["taker_buy_base"].to_numpy(float)
    btc = B._load_kl(B.BTC_KL)
    bt5 = btc.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(ts5)).ffill().to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float)
    lo1 = m1["low"].to_numpy(float); cl1 = m1["close"].to_numpy(float)
    F, LV, atr5 = LF.bar_features(C5, H5, L5, V5, TB, bt5)
    mz = np.load(ROOT / "tmp/xsec_perp_screen_20260908/metrics_panel.npz", allow_pickle=True)
    xts = pd.DatetimeIndex(pd.to_datetime(mz["ts"]))
    ei = list(np.load(ROOT / "tmp/xsec_perp_screen_20260908/panel.npz", allow_pickle=True)["syms"]).index("ETHUSDT")
    XS = LF.metric_features({"retail": mz["count_long_short_ratio"][:, ei],
                             "ttc": mz["count_toptrader_long_short_ratio"][:, ei],
                             "ttp": mz["sum_toptrader_long_short_ratio"][:, ei],
                             "tkv": mz["sum_taker_long_short_vol_ratio"][:, ei]}, xts, ts5)

    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    ba = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float)
    entry = O5[np.minimum(ba + 1, len(O5) - 1)] * (1 + sgn * T)
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
    okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5)) & (bt >= 1) & (ba >= 900)
    tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0), entry * (1 + P), entry * (1 - P), H * 5)
    big = 1 << 30
    uo, do_ = tu >= 0, td >= 0
    au, ad = np.where(uo, tu, big), np.where(do_, td, big)
    cont = np.where(sgn > 0, uo & (au < ad), do_ & (ad < au))
    rev = np.where(sgn > 0, do_ & (ad < au), uo & (au < ad))
    x5 = np.minimum(bt + H, len(C5) - 1)
    clo = (C5[x5] - entry) / entry * 1e4 * sgn
    y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))

    # --- 1단계 모델(anchor47) 워크포워드 ---
    fk = sorted(F.keys()); xk = sorted(XS.keys())
    sigc = [f"sig_{s}" for s in ("sweep", "smt", "taker", "kal", "strz", "orth", "fib", "dem")]
    stat = d[sigc + ["n_signals", "side_bottom", "atr_at_anchor"]].to_numpy(np.float32)
    bidx = np.clip(ba, 0, len(C5) - 1)
    hr = pd.to_datetime(ts5[bidx])
    XA = np.hstack([np.stack([F[k][bidx] for k in fk] + [XS[k][bidx] for k in xk], axis=1),
                    np.stack([hr.hour.to_numpy(), hr.dayofweek.to_numpy(), T], axis=1), stat]).astype(np.float32)
    XA = np.nan_to_num(XA, nan=0.0, posinf=0.0, neginf=0.0)
    ts = d["timestamp"]; sp = d["split"].to_numpy()
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    trm = sp == "TRAIN"; cut = ts[trm].quantile(0.7)
    tr_fit = np.flatnonzero(trm & (ts <= cut).to_numpy()); tr_hold = trm & (ts > cut).to_numpy()
    print("1단계 anchor47 워크포워드 ...", flush=True)
    pa = np.full(len(y), np.nan)
    for i, mo in enumerate(uniq):
        if i < 5: continue
        te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
        if tr.sum() < 1500 or te.sum() < 30: continue
        itr = np.flatnonzero(tr)
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=SEED)
        pa[te] = c.fit(XA[itr], y[itr]).predict_proba(XA[te])[:, 1]
    c0 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                        l2_regularization=1.0, early_stopping=True,
                                        validation_fraction=0.15, random_state=SEED)
    conf_a = np.abs(c0.fit(XA[tr_fit], y[tr_fit]).predict_proba(XA[tr_hold])[:, 1] - 0.5)
    np.save(MY / "pretrigger_anchor_preds.npy", pa)

    pf = np.nanmean(np.load(MY / "c1c3_preds_T0.75.npz")["obs"], axis=0)   # 2단계 full69
    fin = np.isfinite(pa) & np.isfinite(pf) & okm
    print(f"두 모델 모두 예측 있는 사건 {int(fin.sum()):,}", flush=True)
    print(f"판정 일치율(돌파/되돌림): {((pa[fin] > .5) == (pf[fin] > .5)).mean()*100:.1f}%\n", flush=True)

    idx = np.flatnonzero(fin)
    span = np.arange(H * 5); J = s1[idx][:, None] + span[None, :]
    HI = hi1[np.clip(J, 0, len(hi1) - 1)]; LO = lo1[np.clip(J, 0, len(lo1) - 1)]
    ENT = entry[idx]; CLO = C5[x5][idx]
    side = -sgn[idx]                        # 지정가 체결 = 발현 반대편 = 되돌림 포지션
    ab_px = cl1[np.clip(s1[idx] + 1, 0, len(cl1) - 1)]      # 중도청산: 트리거 +1분 종가
    ab_bp = (ab_px - ENT) / ENT * 1e4 * side

    def sim(tp, sl):
        tp_px = ENT * (1 + side * tp / 1e4); sl_px = ENT * (1 - side * sl / 1e4)
        tph = np.where(side[:, None] > 0, HI >= tp_px[:, None], LO <= tp_px[:, None])
        slh = np.where(side[:, None] > 0, LO <= sl_px[:, None], HI >= sl_px[:, None])
        a = np.where(tph.any(1), tph.argmax(1), big); b = np.where(slh.any(1), slh.argmax(1), big)
        none = (a == big) & (b == big)
        return np.where(none, (CLO - ENT) / ENT * 1e4 * side, np.where(a < b, tp, -sl).astype(float))

    rows = []
    for cv in S1_COVS:
        thr = float(np.quantile(conf_a, 1 - cv))
        gate = (pa[idx] < 0.5) & (np.abs(pa[idx] - 0.5) >= thr)      # 1단계: 되돌림 + 확신
        conf2 = pf[idx] < 0.5                                        # 2단계: 되돌림 확인
        for tp in TPS:
            for sl in SLS:
                g = sim(tp, sl)
                for w in WINS:
                    mw = gate & (sp[idx] == w)
                    if mw.sum() < 30: continue
                    hold = mw & conf2; abort = mw & ~conf2
                    net = np.concatenate([g[hold] - COST_MAKER_RT, ab_bp[abort] - COST_MAKER_RT])
                    rows.append(dict(s1cov=int(cv * 100), tp=tp, sl=sl, win=w,
                                     n=int(mw.sum()), hold=int(hold.sum()), abort=int(abort.sum()),
                                     agree=float(hold.sum() / max(mw.sum(), 1)),
                                     net=float(net.mean()), hold_net=float((g[hold] - COST_MAKER_RT).mean()),
                                     abort_net=float((ab_bp[abort] - COST_MAKER_RT).mean()) if abort.sum() else np.nan,
                                     per_day=mw.sum() / WDAYS[w]))
    R = pd.DataFrame(rows); R.to_csv(MY / "two_stage_maker.csv", index=False)
    Q = R.pivot_table(index=["s1cov", "tp", "sl"], columns="win",
                      values=["net", "per_day", "agree", "hold_net", "abort_net"])
    Q.columns = [f"{a}_{b[:4]}" for a, b in Q.columns]; Q = Q.reset_index()
    Q["minnet"] = Q[[f"net_{w[:4]}" for w in WINS]].min(axis=1)
    Q = Q.sort_values("minnet", ascending=False)
    print("=" * 104)
    print(f"{'1단계커버':>9}{'TP':>4}{'SL':>4} | " + "".join(f"{w[:4]:>13}" for w in WINS)
          + f"{'최소':>8}{'건/일':>7}{'합의율':>8}")
    print("=" * 104)
    for _, r in Q.head(14).iterrows():
        print(f"{r['s1cov']:>8.0f}%{r['tp']:>4.0f}{r['sl']:>4.0f} | "
              + "".join(f"{r[f'net_{w[:4]}']:>+13.2f}" for w in WINS)
              + f"{r['minnet']:>+8.2f}{r['per_day_OOS']:>7.1f}{r['agree_OOS']*100:>7.0f}%")
    b = Q.iloc[0]
    print(f"\n최선 구성 분해(OOS): 보유 {b['hold_net_OOS']:+.2f}bp · 중도청산 {b['abort_net_OOS']:+.2f}bp "
          f"· 합의율 {b['agree_OOS']*100:.0f}%")
    print(json.dumps({"done": True, "pass": int((Q['minnet'] > 0).sum()), "total": len(Q)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
