#!/usr/bin/env python3
"""돌파/되돌림 피쳐 **v3 -- 체결·포지션·호가 (사용자 제안 3축)** (2026-09-08).

사용자 제안:
  1. **CVD/흡수**  -- 시장가 순매수(CVD)는 폭발했는데 가격이 안 뚫리면 지정가 벽이 흡수 중 -> 되돌림.
                     CVD 는 완만한데 가격이 뚫리면 반대편이 비어있다 -> 지속.
  2. **OI 변화**   -- 돌파+OI 급증 = 신규자금(지속). 돌파+OI 급감 = 청산연료 소진(되돌림).
  3. **OBI/스푸핑** -- 앞쪽 벽이 취소되며 뚫리면 지속. 뚫은 직후 반대 벽이 서면 유동성 사냥(되돌림).

## ⚠️인과성 (부록 AM 의 1분 미래참조 재발 방지)
라벨은 트리거 분 `s1` **부터** 배리어를 탐색한다. 따라서
  - 1분 테이프 경로 : `s0 .. s1-1`  (트리거 분 **제외**)
  - 5분 메트릭(OI)  : `fb = bt-1`   (트리거 봉 직전 완결봉)
  - 30초 호가       : `ts < ts1[s1]` 인 마지막 스냅샷
정규화 기준선(rolling)은 전부 `shift(1)` 로 자기 자신을 제외한다.

## ⚠️호가 데이터 결함 (이 세션 발견)
2025-10-11~15 · 2025-10-30~12-01 · 2026-09-04~ 는 한쪽 밴드가 붕괴한다.
`bd_ok=0` 스냅샷의 v3b_* 는 NaN 으로 둔다. **VAL 창의 30% 가 여기 해당**한다.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
TAPE = ROOT / "data/research/eth_tape_1m_20260906.parquet"
BD = ROOT / "data/research/eth_bookdepth_30s_20260908.parquet"
XMET = ROOT / "tmp/xsec_perp_screen_20260908/metrics_panel.npz"
XPAN = ROOT / "tmp/xsec_perp_screen_20260908/panel.npz"
DS2 = ROOT / "tmp/eth_breakout_reversal_20260908/dataset_v2.parquet"
OUT = ROOT / "tmp/eth_breakout_reversal_20260908"
NMOVE = 15
KEY = ["bar_idx", "anchor", "side_bottom", "T_mult"]


def rbase(x, w, log=True):
    """자기 자신을 제외한 rolling 기준선 (기하평균)."""
    s = pd.Series(np.log(np.maximum(x, 1e-12)) if log else x)
    m = s.rolling(w, min_periods=w // 4).mean().shift(1).to_numpy()
    return np.exp(m) if log else m


def main() -> int:
    print("[1/5] 로드 ...", flush=True)
    D = pd.read_parquet(DS2).reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL); ts5 = eth["timestamp"].to_numpy()
    O5 = eth["open"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); cl1 = m1["close"].to_numpy(float)

    # --- 테이프를 1분 격자에 정렬 ---
    tp = pd.read_parquet(TAPE); tp["ts"] = pd.to_datetime(tp["ts"])
    tp = tp.set_index("ts").reindex(pd.DatetimeIndex(ts1))
    TN = tp["notional"].to_numpy(float); TS = tp["signed_notional"].to_numpy(float)
    TV = tp["volume"].to_numpy(float)
    TK = tp["kyle_lambda"].to_numpy(float); TI = tp["impact_per_vol"].to_numpy(float)
    TXL = tp["xl_signed"].to_numpy(float); TLG = tp["lg_signed"].to_numpy(float)
    TLV = tp["lg_vol"].to_numpy(float); TSW = tp["switch_rate"].to_numpy(float)
    TAS = tp["avg_trade_size"].to_numpy(float); TIMB = tp["imbalance"].to_numpy(float)
    BN = rbase(TN, 240); BK = rbase(TK, 240); BI = rbase(TI, 240); BAS = rbase(TAS, 240)
    print(f"      테이프 정렬 {np.isfinite(TN).mean():.3f}", flush=True)

    # --- 메트릭(OI) 5분 격자 ---
    mz = np.load(XMET, allow_pickle=True); z = np.load(XPAN, allow_pickle=True)
    xts = pd.DatetimeIndex(pd.to_datetime(z["ts"])); ei = list(z["syms"]).index("ETHUSDT")
    oi = mz["sum_open_interest"][:, ei].astype(float)
    pos = pd.Index(xts).get_indexer(pd.DatetimeIndex(ts5))
    def m5(v):
        return np.where(pos >= 0, v[np.clip(pos, 0, len(v) - 1)], np.nan)
    OI5 = m5(oi)
    OID = {k: m5(np.concatenate([np.full(k, np.nan), oi[k:] / np.maximum(oi[:-k], 1e-9) - 1.0]))
           for k in (1, 3, 6, 12)}

    # --- 호가 30초 패널 ---
    bd = pd.read_parquet(BD); bdt = bd["ts"].to_numpy()
    COL = {"a02": "d0p2", "a1": "d1p0", "a2": "d2p0", "b02": "dm0p2", "b1": "dm1p0", "b2": "dm2p0"}
    BDv = {k: bd[c].to_numpy(float) for k, c in COL.items()}
    BDb = {k: rbase(v, 20) for k, v in BDv.items()}                      # 직전 10분 기준선
    BDm = {k: pd.Series(v).rolling(20, min_periods=5).max().shift(1).to_numpy()
           for k, v in BDv.items()}                                      # 직전 10분 최대 (벽 취소 탐지)
    BOK = bd["bd_ok"].to_numpy()
    print(f"      호가 {len(bd):,} bd_ok={BOK.mean():.4f}", flush=True)

    print("[2/5] 인덱스 재구성 ...", flush=True)
    bidx = D["bar_idx"].to_numpy(); tmin = D["trig_min"].to_numpy().astype(int)
    sgn = np.where(D["dir_up"].to_numpy() > 0, 1.0, -1.0)
    atr = np.maximum(D["atr_at_anchor"].to_numpy(float), 1e-9)
    ei0 = np.minimum(bidx + 1, len(O5) - 1); ref = O5[ei0]
    s0 = np.searchsorted(ts1, ts5[ei0])
    s0 = np.clip(s0, 0, len(ts1) - 1)
    s1 = np.clip(s0 + tmin, 0, len(ts1) - 1)
    bt = np.searchsorted(ts5, ts1[s1], side="right") - 1
    fb = bt - 1
    ok_fb = (fb == D["feat_bar"].to_numpy())
    print(f"      feat_bar 재구성 일치율 {ok_fb.mean():.5f}", flush=True)
    assert ok_fb.mean() > 0.999, "인덱스 규약 불일치 -- 중단"

    print("[3/5] v3c 체결/흡수 (경로 s0..s1-1) ...", flush=True)
    n = len(D); span = np.arange(NMOVE)[None, :]
    F = {}
    CH = 20000
    acc = {k: np.full(n, np.nan) for k in
           ("flow", "cvd", "cvdfrac", "align", "kyle", "impact", "xl", "lg", "sw", "asz", "div", "mv", "last_imb")}
    for a in range(0, n, CH):
        b = min(a + CH, n)
        j = s0[a:b, None] + span
        j = np.clip(j, 0, len(ts1) - 1)
        mask = (span <= (tmin[a:b, None] - 1)) & (s0[a:b, None] + span < len(ts1))
        sg = sgn[a:b, None]
        nn = np.where(mask, TN[j], np.nan); ss = np.where(mask, TS[j] * sg, np.nan)
        vv = np.where(mask, TV[j], np.nan)
        nsum = np.nansum(nn, axis=1); ssum = np.nansum(ss, axis=1); vsum = np.nansum(vv, axis=1)
        nb = np.maximum(mask.sum(1), 1); has = mask.sum(1) >= 1
        base = BN[np.clip(s0[a:b] - 1, 0, len(BN) - 1)]
        acc["flow"][a:b] = np.where(has, (nsum / nb) / np.maximum(base, 1e-9), np.nan)
        acc["cvd"][a:b] = np.where(has, (ssum / nb) / np.maximum(base, 1e-9), np.nan)
        acc["cvdfrac"][a:b] = np.where(has, ssum / np.maximum(nsum, 1e-9), np.nan)
        acc["align"][a:b] = np.where(has, np.nansum(np.where(mask, np.sign(TS[j]) == sg, np.nan), axis=1) / nb, np.nan)
        acc["kyle"][a:b] = np.where(has, np.nanmean(np.where(mask, TK[j], np.nan), axis=1)
                                    / np.maximum(BK[np.clip(s0[a:b] - 1, 0, len(BK) - 1)], 1e-12), np.nan)
        acc["impact"][a:b] = np.where(has, np.nanmean(np.where(mask, TI[j], np.nan), axis=1)
                                      / np.maximum(BI[np.clip(s0[a:b] - 1, 0, len(BI) - 1)], 1e-12), np.nan)
        acc["xl"][a:b] = np.where(has, np.nansum(np.where(mask, TXL[j] * sg, np.nan), axis=1)
                                  / np.maximum(vsum, 1e-9), np.nan)
        acc["lg"][a:b] = np.where(has, np.nansum(np.where(mask, TLG[j] * sg, np.nan), axis=1)
                                  / np.maximum(np.nansum(np.where(mask, TLV[j], np.nan), axis=1), 1e-9), np.nan)
        acc["sw"][a:b] = np.where(has, np.nanmean(np.where(mask, TSW[j], np.nan), axis=1), np.nan)
        acc["asz"][a:b] = np.where(has, np.nanmean(np.where(mask, TAS[j], np.nan), axis=1)
                                   / np.maximum(BAS[np.clip(s0[a:b] - 1, 0, len(BAS) - 1)], 1e-12), np.nan)
        # 실현 이동폭 (s1-1 종가 기준, ATR 단위) -- 흡수 분모
        last = np.clip(s0[a:b] + tmin[a:b] - 1, 0, len(cl1) - 1)
        mv = np.where(has, sgn[a:b] * (cl1[last] - ref[a:b]) / np.maximum(ref[a:b], 1e-9) / atr[a:b], np.nan)
        acc["mv"][a:b] = mv
        acc["last_imb"][a:b] = np.where(has, TIMB[last], np.nan)
        # 다이버전스: 후반 절반 CVD비중 - 전반 절반 (tmin>=2)
        half = np.maximum(tmin[a:b] // 2, 1)
        m1h = mask & (span < half[:, None]); m2h = mask & (span >= half[:, None])
        f1 = np.nansum(np.where(m1h, TS[j] * sg, np.nan), 1) / np.maximum(np.nansum(np.where(m1h, TN[j], np.nan), 1), 1e-9)
        f2 = np.nansum(np.where(m2h, TS[j] * sg, np.nan), 1) / np.maximum(np.nansum(np.where(m2h, TN[j], np.nan), 1), 1e-9)
        acc["div"][a:b] = np.where(tmin[a:b] >= 2, f2 - f1, np.nan)
    F["v3c_flow_r"] = acc["flow"]; F["v3c_cvd_r"] = acc["cvd"]; F["v3c_cvd_frac"] = acc["cvdfrac"]
    F["v3c_align"] = acc["align"]; F["v3c_kyle_r"] = acc["kyle"]; F["v3c_impact_r"] = acc["impact"]
    F["v3c_xl_frac"] = acc["xl"]; F["v3c_lg_imb"] = acc["lg"]; F["v3c_switch"] = acc["sw"]
    F["v3c_avgsz_r"] = acc["asz"]; F["v3c_cvd_div"] = acc["div"]; F["v3c_move_atr"] = acc["mv"]
    F["v3c_last_imb"] = acc["last_imb"]
    # ⭐흡수 계수: 같은 이동폭을 만드는 데 태운 순매수 연료. 클수록 흡수 -> 되돌림 가설.
    F["v3c_burn"] = acc["cvd"] / np.clip(np.abs(acc["mv"]), 0.05, None)
    F["v3c_burn_frac"] = acc["cvdfrac"] / np.clip(np.abs(acc["mv"]), 0.05, None)
    F["v3c_path_known"] = (tmin >= 1).astype(float)

    print("[4/5] v3o 미결제약정 (fb 까지) ...", flush=True)
    fbc = np.clip(fb, 0, len(OI5) - 1)
    for k in (1, 3, 6, 12):
        F[f"v3o_doi{k}"] = OID[k][fbc]
    F["v3o_doi_acc"] = OID[3][fbc] - OID[12][fbc]          # 최근 가속
    F["v3o_oi_lvl"] = OI5[fbc]

    print("[5/5] v3b 호가/벽 (ts < s1 마지막 스냅샷) ...", flush=True)
    jb = np.searchsorted(bdt, ts1[s1], side="left") - 1
    valid = (jb >= 20) & (jb < len(bdt))
    jb = np.clip(jb, 0, len(bdt) - 1)
    lag_min = (ts1[s1] - bdt[jb]) / np.timedelta64(1, "m")
    okb = valid & (BOK[jb] == 1) & (lag_min <= 3.0)
    up = sgn > 0
    def pick(dic, af, bf):
        return np.where(up, dic[af][jb], dic[bf][jb])
    ah1 = pick(BDv, "a1", "b1"); bh1 = pick(BDv, "b1", "a1")
    ah02 = pick(BDv, "a02", "b02"); bh02 = pick(BDv, "b02", "a02")
    ab1 = pick(BDb, "a1", "b1"); bb1 = pick(BDb, "b1", "a1")
    am1 = pick(BDm, "a1", "b1")
    nanb = lambda x: np.where(okb, x, np.nan)
    F["v3b_wall1"] = nanb(ah1 / np.maximum(ah1 + bh1, 1e-9))
    F["v3b_wall02"] = nanb(ah02 / np.maximum(ah02 + bh02, 1e-9))
    F["v3b_ahead_r"] = nanb(ah1 / np.maximum(ab1, 1e-9))
    F["v3b_behind_r"] = nanb(bh1 / np.maximum(bb1, 1e-9))
    F["v3b_ahead_drop"] = nanb(1.0 - ah1 / np.maximum(am1, 1e-9))       # ⭐벽 취소(스푸핑) 프록시
    F["v3b_tot_r"] = nanb((ah1 + bh1) / np.maximum(ab1 + bb1, 1e-9))
    F["v3b_near_conc"] = nanb(ah02 / np.maximum(ah1, 1e-9))             # 근접 집중도
    F["v3b_ok"] = okb.astype(float)

    V = pd.DataFrame({k: D[k].to_numpy() for k in KEY} | F)
    V = V.replace([np.inf, -np.inf], np.nan)
    M = D.merge(V.drop_duplicates(KEY), on=KEY, how="left", validate="one_to_one")
    M.to_parquet(OUT / "dataset_v3.parquet")
    cov = {g: float(M[[c for c in M.columns if c.startswith(g)]].notna().any(axis=1).mean())
           for g in ("v3c_", "v3o_", "v3b_")}
    print(json.dumps({"rows": len(M), "v3_feats": len(F), "coverage": cov,
                      "v3b_ok_by_split": M.groupby("split")["v3b_ok"].mean().round(4).to_dict()},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
