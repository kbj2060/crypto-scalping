#!/usr/bin/env python3
"""돌파/되돌림 **지연관측 팔** -- 돌파 '직후'를 보고 판단한다 (2026-09-08).

## 왜 필요한가
사용자가 설명한 메커니즘 중 둘은 **돌파 직후의 관측**이다:
  - *"가격이 극점을 강하게 뚫었을 때 OI 의 증감을 확인"*  (OI 급감 = 숏스퀴즈 연료 = 되돌림)
  - *"극점을 살짝 돌파한 직후, 반대 방향으로 거대한 호가벽이 생기며"*  (유동성 사냥 = 되돌림)
트리거 시점(s1) 의사결정에서는 이 정보가 **원리적으로 없다**. 그래서 판단을 D분 늦춘다.
부록 AL 의 "예측하지 않고 관측한 뒤 진입" 설계와 같은 계열이다.

## 설계
  - 의사결정 시각 : `sd = s1 + D` 분의 **종가**
  - 진입 기준가   : `cl1[sd]`
  - 라벨          : `sd+1` 분부터 ±P 첫 터치. 발현방향이 먼저면 돌파(y=1), 반대면 되돌림(y=0).
                    H봉 내 미터치는 종가 부호 -> 커버리지 100%.
  - ⚠️피쳐 창은 `sd` 까지, 라벨은 `sd+1` 부터. **한 봉도 공유하지 않는다**(부록 AM 규칙).
  - 대가: 진입이 D분 늦고 그만큼 움직임을 놓친다. 정확도로만 평가하므로 체결가정과 무관.
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
DS3 = ROOT / "tmp/eth_breakout_reversal_20260908/dataset_v3.parquet"
OUT = ROOT / "tmp/eth_breakout_reversal_20260908"
P = 0.005
H = 48
DELAYS = (3, 5, 10)
KEY = ["bar_idx", "anchor", "side_bottom", "T_mult"]
CHUNK = 4000


def first_touch(hi1, lo1, start, up, dn, nmin):
    n = len(start)
    tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = np.clip(start[a:b, None] + np.arange(nmin)[None, :], 0, len(hi1) - 1)
        hu = hi1[idx] >= up[a:b, None]; hd = lo1[idx] <= dn[a:b, None]
        au = hu.any(1); ad = hd.any(1)
        tu[a:b] = np.where(au, hu.argmax(1), -1); td[a:b] = np.where(ad, hd.argmax(1), -1)
    return tu, td


def rbase(x, w):
    s = pd.Series(np.log(np.maximum(x, 1e-12)))
    return np.exp(s.rolling(w, min_periods=w // 4).mean().shift(1).to_numpy())


def main() -> int:
    print("[1/4] 로드 ...", flush=True)
    D = pd.read_parquet(DS3).reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL); ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy()
    hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float); cl1 = m1["close"].to_numpy(float)

    tp = pd.read_parquet(TAPE); tp["ts"] = pd.to_datetime(tp["ts"])
    tp = tp.set_index("ts").reindex(pd.DatetimeIndex(ts1))
    TN = tp["notional"].to_numpy(float); TS = tp["signed_notional"].to_numpy(float)
    TK = tp["kyle_lambda"].to_numpy(float); TI = tp["impact_per_vol"].to_numpy(float)
    BN = rbase(TN, 240); BK = rbase(TK, 240); BI = rbase(TI, 240)

    mz = np.load(XMET, allow_pickle=True); z = np.load(XPAN, allow_pickle=True)
    xts = pd.DatetimeIndex(pd.to_datetime(z["ts"])); ei = list(z["syms"]).index("ETHUSDT")
    oi = mz["sum_open_interest"][:, ei].astype(float)
    pos = pd.Index(xts).get_indexer(pd.DatetimeIndex(ts5))
    OI5 = np.where(pos >= 0, oi[np.clip(pos, 0, len(oi) - 1)], np.nan)

    bd = pd.read_parquet(BD); bdt = bd["ts"].to_numpy()
    BDv = {k: bd[c].to_numpy(float) for k, c in
           {"a1": "d1p0", "b1": "dm1p0", "a02": "d0p2", "b02": "dm0p2"}.items()}
    BOK = bd["bd_ok"].to_numpy()

    print("[2/4] 인덱스 ...", flush=True)
    bidx = D["bar_idx"].to_numpy(); tmin = D["trig_min"].to_numpy().astype(int)
    sgn = np.where(D["dir_up"].to_numpy() > 0, 1.0, -1.0)
    atr = np.maximum(D["atr_at_anchor"].to_numpy(float), 1e-9)
    ei0 = np.minimum(bidx + 1, len(O5) - 1); ref = O5[ei0]
    s0 = np.clip(np.searchsorted(ts1, ts5[ei0]), 0, len(ts1) - 1)
    s1 = np.clip(s0 + tmin, 0, len(ts1) - 1)
    fb = np.searchsorted(ts5, ts1[s1], side="right") - 2   # 트리거 봉 직전 완결봉

    frames = []
    for Dm in DELAYS:
        print(f"[3/4] D={Dm}분 ...", flush=True)
        sd = s1 + Dm
        ok = sd + H * 5 + 2 < len(ts1)
        sdc = np.clip(sd, 0, len(ts1) - 1)
        e = cl1[sdc]
        tu, td = first_touch(hi1, lo1, np.clip(sdc + 1, 0, len(ts1) - 1),
                             e * (1 + P), e * (1 - P), H * 5)
        big = 1 << 30
        a = np.where(tu >= 0, tu, big); b = np.where(td >= 0, td, big)
        up_first = a < b
        resolved = (a < big) | (b < big)
        # 시간청산: H봉 뒤 종가 부호
        tex = cl1[np.clip(sdc + H * 5, 0, len(cl1) - 1)]
        tsign = np.sign(tex - e)
        moved_up = np.where(resolved, up_first, tsign > 0)
        y = (moved_up == (sgn > 0)).astype(int)     # 발현방향으로 갔으면 돌파

        # --- 사후 피쳐 (창 s1 .. sd, 라벨은 sd+1 부터라 겹치지 않음) ---
        W = Dm + 1
        span = np.arange(W)[None, :]
        n = len(D); f = {}
        cvd = np.full(n, np.nan); flow = np.full(n, np.nan)
        kyl = np.full(n, np.nan); imp = np.full(n, np.nan); mae = np.full(n, np.nan)
        for i in range(0, n, 20000):
            j2 = min(i + 20000, n)
            idx = np.clip(s1[i:j2, None] + span, 0, len(ts1) - 1)
            sg = sgn[i:j2, None]
            nn = TN[idx]; ss = TS[idx] * sg
            base = np.maximum(BN[np.clip(s1[i:j2] - 1, 0, len(BN) - 1)], 1e-9)
            flow[i:j2] = (np.nansum(nn, 1) / W) / base
            cvd[i:j2] = (np.nansum(ss, 1) / W) / base
            kyl[i:j2] = np.nanmean(TK[idx], 1) / np.maximum(BK[np.clip(s1[i:j2] - 1, 0, len(BK) - 1)], 1e-12)
            imp[i:j2] = np.nanmean(TI[idx], 1) / np.maximum(BI[np.clip(s1[i:j2] - 1, 0, len(BI) - 1)], 1e-12)
            tp0 = cl1[np.clip(s1[i:j2], 0, len(cl1) - 1)]
            adv = np.where(sgn[i:j2, None] > 0, (lo1[idx] - tp0[:, None]), (tp0[:, None] - hi1[idx]))
            mae[i:j2] = np.min(adv, 1) / np.maximum(tp0, 1e-9) / atr[i:j2]
        f["v4_cvd_post"] = cvd; f["v4_flow_post"] = flow
        f["v4_kyle_post"] = kyl; f["v4_impact_post"] = imp; f["v4_mae_post"] = mae
        f["v4_ret_post"] = sgn * (cl1[sdc] - cl1[np.clip(s1, 0, len(cl1) - 1)]) / np.maximum(cl1[np.clip(s1, 0, len(cl1) - 1)], 1e-9) / atr
        f["v4_move_tot"] = sgn * (cl1[sdc] - ref) / np.maximum(ref, 1e-9) / atr

        # ⭐OI: 트리거 직전 완결봉 -> 의사결정 시점 직전 완결봉
        fbp = np.searchsorted(ts5, ts1[sdc], side="right") - 2
        o_a = OI5[np.clip(fb, 0, len(OI5) - 1)]; o_b = OI5[np.clip(fbp, 0, len(OI5) - 1)]
        f["v4_doi_post"] = np.where(fbp > fb, o_b / np.maximum(o_a, 1e-9) - 1.0, np.nan)
        f["v4_doi_bars"] = (fbp - fb).astype(float)

        # ⭐호가: 트리거 직전 스냅샷 -> 의사결정 시점 스냅샷
        j_a = np.searchsorted(bdt, ts1[np.clip(s1, 0, len(ts1) - 1)], side="left") - 1
        j_b = np.searchsorted(bdt, ts1[sdc] + np.timedelta64(60, "s"), side="left") - 1
        vb = (j_a >= 0) & (j_b >= 0) & (j_b < len(bdt)) & (BOK[np.clip(j_a, 0, len(BOK) - 1)] == 1) \
             & (BOK[np.clip(j_b, 0, len(BOK) - 1)] == 1)
        j_a = np.clip(j_a, 0, len(bdt) - 1); j_b = np.clip(j_b, 0, len(bdt) - 1)
        up = sgn > 0
        ahead_a = np.where(up, BDv["a1"][j_a], BDv["b1"][j_a])
        ahead_b = np.where(up, BDv["a1"][j_b], BDv["b1"][j_b])
        behind_a = np.where(up, BDv["b1"][j_a], BDv["a1"][j_a])
        behind_b = np.where(up, BDv["b1"][j_b], BDv["a1"][j_b])
        nb_ = lambda x: np.where(vb, x, np.nan)
        f["v4_ahead_chg"] = nb_(ahead_b / np.maximum(ahead_a, 1e-9) - 1.0)
        f["v4_behind_build"] = nb_(behind_b / np.maximum(behind_a, 1e-9) - 1.0)   # ⭐반대 벽 형성
        f["v4_wall_post"] = nb_(ahead_b / np.maximum(ahead_b + behind_b, 1e-9))
        f["v4_bk_ok"] = vb.astype(float)

        r = pd.DataFrame({k: D[k].to_numpy() for k in KEY})
        r["timestamp"] = D["timestamp"].to_numpy(); r["split"] = D["split"].to_numpy()
        r["delay"] = Dm; r["y_d"] = y; r["resolved_d"] = resolved.astype(int)
        r["valid_d"] = ok.astype(int) if np.ndim(ok) else int(ok)
        r["valid_d"] = ((sd + H * 5 + 2) < len(ts1)).astype(int)
        for k, v in f.items():
            r[k] = v
        frames.append(r)
        print(f"      돌파율 {y[r.valid_d == 1].mean():.4f} · 배리어해소 {resolved.mean():.3f} "
              f"· OI관측가능 {np.isfinite(f['v4_doi_post']).mean():.3f} · 호가가능 {vb.mean():.3f}", flush=True)

    print("[4/4] 저장 ...", flush=True)
    R = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    R.to_parquet(OUT / "dataset_delayed.parquet")
    print(json.dumps({"rows": len(R), "delays": list(DELAYS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
