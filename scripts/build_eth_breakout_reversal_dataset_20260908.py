#!/usr/bin/env python3
"""**돌파/되돌림 분류 데이터셋** 빌더 (2026-09-08).

사용자: *"이 앵커 방향으로 돈을 버는게 아니야. 돌파하냐 되돌리냐만 정확도를 크게 올려줘."*

## 과제 정의
앵커 발동 후 가격이 먼저 ±T 만큼 **발현**하면(방향은 관측값), 그 지점부터
    y = 1  발현 방향으로 +P 배리어 먼저 터치 (**돌파**)
    y = 0  반대로 −P 배리어 먼저 터치 (**되돌림**)
    시간청산(H봉 내 미터치) -> H봉 뒤 종가 수익의 부호로 라벨 (**커버리지 100%**)
방향이 이미 알려져 있으므로 부록 Z·AA 가 닫은 **대칭 방향 예측 문제와 다르다**.
현재 무조건-되돌림 베이스라인 정확도 = **52.0~53.2%**(네 창, 부록 AL).

## 인과성 (게이트 L3/L4)
- 트리거는 5분봉 `bt` **안의 어느 분**에 발생한다. 그 순간 **완결된 마지막 봉은 `bt-1`**이다.
  ⇒ 모든 피쳐는 **`bt-1` 종가 기준**으로만 만든다(보수적). 봉 `bt` 자체는 절대 쓰지 않는다.
- 라벨은 트리거 **분**부터 시작한다.
- 횡단면 순위도 같은 `bt-1` 시각의 60종 값으로 계산한다.

## 피쳐군 (⭐표시는 이 문제에 처음 쓰임)
A 발현: T(ATR단위) · 앵커→트리거 경과분 · 발현속도 · 방향
B 가격: logret 3/6/12/48/144봉 · atr_pct · 고저폭 z · 거래량 z · 테이커매수비
C ⭐메트릭: OI 수준 z·변화(12/48) · 개미 롱숏비 수준/변화 · 상위트레이더 계정수/포지션 비 수준/변화 · 테이커비
D ⭐횡단면: 60종 중 ETH 의 순위 (12봉수익 · OI변화 · 개미비 · 상위트레이더비 · 변동성)
E BTC: 3/12/48봉 수익 · ETH−BTC 스프레드
F 세션: 시(UTC) · 요일
G 앵커: n_signals · 신호 8종 원핫 · side
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
SRC = ROOT / "tmp/eth_anchor_label_dataset_20260907/anchors_labels.parquet"
XPAN = ROOT / "tmp/xsec_perp_screen_20260908/panel.npz"
XMET = ROOT / "tmp/xsec_perp_screen_20260908/metrics_panel.npz"
OUT = ROOT / "tmp/eth_breakout_reversal_20260908"
ANCHORS = ("first_fire", "any2/Wc3", "any3/Wc3")
T_MULT = (0.5, 0.75, 1.0)
W_TRIG = 3
P = 0.005
H = 48
MAXMIN = (W_TRIG + H) * 5 + 10
CHUNK = 4000
SIGNALS = ("sweep", "smt", "taker", "kal", "strz", "orth", "fib", "dem")


def first_touch(hi1, lo1, start, up, dn, nmin):
    n = len(start)
    tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = start[a:b, None] + np.arange(nmin)[None, :]
        hu = hi1[idx] >= up[a:b, None]; hd = lo1[idx] <= dn[a:b, None]
        au = hu.any(1); ad = hd.any(1)
        tu[a:b] = np.where(au, hu.argmax(1), -1); td[a:b] = np.where(ad, hd.argmax(1), -1)
    return tu, td


def zs(x, w):
    s = pd.Series(x)
    return ((s - s.rolling(w, min_periods=w // 3).mean())
            / s.rolling(w, min_periods=w // 3).std().replace(0, np.nan)).to_numpy()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/5] 로드 ...", flush=True)
    D = pd.read_parquet(SRC).reset_index(drop=True)
    D = D[D["anchor"].isin(ANCHORS)].reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL); btc = B._load_kl(B.BTC_KL)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    ts5 = eth["timestamp"].to_numpy()
    O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    H5 = eth["high"].to_numpy(float); L5 = eth["low"].to_numpy(float)
    V5 = eth["volume"].to_numpy(float); TB = eth["taker_buy_base"].to_numpy(float)
    bt5 = btc.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(ts5)).ffill().to_numpy(float)
    CAP = len(eth) - MAXMIN // 5 - 2

    print("[2/5] ETH 봉 피쳐 (bt-1 기준용) ...", flush=True)
    lr = np.full(len(C5), np.nan); lr[1:] = np.log(C5[1:] / C5[:-1])
    F = {}
    for k in (3, 6, 12, 48, 144):
        F[f"ret{k}"] = np.concatenate([np.full(k, np.nan), C5[k:] / C5[:-k] - 1.0])
        F[f"btc_ret{k}"] = np.concatenate([np.full(k, np.nan), bt5[k:] / bt5[:-k] - 1.0])
    F["atr_pct"] = pd.Series((H5 - L5) / C5).rolling(192, min_periods=64).mean().to_numpy()
    F["rng_z"] = zs((H5 - L5) / C5, 288)
    F["vol_z"] = zs(V5, 288)
    F["taker_buy_ratio"] = np.where(V5 > 0, TB / np.maximum(V5, 1e-12), np.nan)
    F["taker_z"] = zs(F["taker_buy_ratio"], 288)
    F["rv48"] = pd.Series(lr).rolling(48, min_periods=16).std().to_numpy()
    F["rv288"] = pd.Series(lr).rolling(288, min_periods=96).std().to_numpy()
    F["rv_ratio"] = F["rv48"] / np.maximum(F["rv288"], 1e-12)
    for k in (3, 12, 48):
        F[f"eth_btc_sp{k}"] = F[f"ret{k}"] - F[f"btc_ret{k}"]

    print("[3/5] 메트릭·횡단면 (⭐신규 피쳐군) ...", flush=True)
    z = np.load(XPAN, allow_pickle=True); mz = np.load(XMET, allow_pickle=True)
    xts = pd.DatetimeIndex(pd.to_datetime(z["ts"])); xsyms = list(z["syms"])
    ei = xsyms.index("ETHUSDT")
    XC = z["C"]
    lg = lambda X: np.where(X > 0, np.log(np.maximum(X, 1e-9)), np.nan)
    MET = {"oi": mz["sum_open_interest"], "retail": lg(mz["count_long_short_ratio"]),
           "ttc": lg(mz["count_toptrader_long_short_ratio"]),
           "ttp": lg(mz["sum_toptrader_long_short_ratio"]),
           "tkv": lg(mz["sum_taker_long_short_vol_ratio"])}
    xret12 = np.full_like(XC, np.nan); xret12[12:] = XC[12:] / XC[:-12] - 1.0
    d_oi = np.full_like(MET["oi"], np.nan)
    d_oi[48:] = MET["oi"][48:] / np.maximum(MET["oi"][:-48], 1e-9) - 1.0
    xvol = pd.DataFrame(np.vstack([np.full((1, XC.shape[1]), np.nan),
                                   np.log(XC[1:] / XC[:-1])])).rolling(288, min_periods=96).std().to_numpy()

    def xrank(M):
        fin = np.isfinite(M); n = fin.sum(1)
        o = np.argsort(np.where(fin, M, np.inf), 1)
        rk = np.argsort(o, 1).astype(np.float32)
        r = np.where(fin, rk / np.maximum(n[:, None] - 1, 1), np.nan)
        return r[:, ei]
    XS = {"xr_ret12": xrank(xret12), "xr_doi": xrank(d_oi), "xr_retail": xrank(MET["retail"]),
          "xr_ttc": xrank(MET["ttc"]), "xr_ttp": xrank(MET["ttp"]), "xr_vol": xrank(xvol)}
    # ETH 자체 메트릭 (수준 z + 변화)
    for nm, M in MET.items():
        v = M[:, ei].astype(float)
        XS[f"m_{nm}_z"] = zs(v, 288)
        for k in (12, 48):
            XS[f"m_{nm}_d{k}"] = np.concatenate([np.full(k, np.nan), v[k:] - v[:-k]])
    # ETH 5분봉 격자에 맞춰 정렬
    pos = pd.Index(xts).get_indexer(pd.DatetimeIndex(ts5))
    for k in list(XS):
        src = XS[k]
        XS[k] = np.where(pos >= 0, src[np.clip(pos, 0, len(src) - 1)], np.nan)

    print("[4/5] 트리거·라벨 ...", flush=True)
    bidx = D["bar_idx"].to_numpy(); atr = D["atr_pct"].to_numpy()
    rows = []
    for Tm in T_MULT:
        ei0 = np.minimum(bidx + 1, len(O5) - 1)
        ref = O5[ei0]
        st = np.searchsorted(ts1, ts5[ei0])
        ok = (st < len(ts1) - MAXMIN) & (ts1[np.minimum(st, len(ts1) - 1)] == ts5[ei0]) & (bidx <= CAP)
        s0 = np.where(ok, st, 0)
        T = atr * Tm
        tu, td = first_touch(hi1, lo1, s0, ref * (1 + T), ref * (1 - T), W_TRIG * 5)
        big = 1 << 30
        a = np.where(tu >= 0, tu, big); b = np.where(td >= 0, td, big)
        fired = ok & ((a < big) | (b < big)) & (a != b)
        sgn = np.where(a < b, 1.0, -1.0)
        tmin = np.where(a < b, a, b)
        entry = ref * (1 + sgn * T)
        s1 = np.where(fired, s0 + np.where(fired, tmin, 0), 0)
        tu2, td2 = first_touch(hi1, lo1, s1, entry * (1 + P), entry * (1 - P), H * 5)
        # 트리거 시각 -> 5분봉 인덱스 bt, 피쳐봉 = bt-1
        trig_ts = ts1[np.minimum(s1, len(ts1) - 1)]
        bt = np.searchsorted(ts5, trig_ts, side="right") - 1
        fb = bt - 1                                        # ⭐완결된 마지막 봉
        x5 = np.minimum(bt + H, len(C5) - 1)
        clo = (C5[x5] - entry) / entry * 1e4 * sgn
        uo = tu2 >= 0; do_ = td2 >= 0
        au = np.where(uo, tu2, big); ad = np.where(do_, td2, big)
        up_first = uo & (au < ad); dn_first = do_ & (ad < au)
        cont = np.where(sgn > 0, up_first, dn_first)
        rev = np.where(sgn > 0, dn_first, up_first)
        y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))   # 커버리지 100%
        keep = fired & (fb >= 200) & (fb < len(C5))
        r = pd.DataFrame({
            "timestamp": pd.to_datetime(trig_ts), "anchor": D["anchor"].to_numpy(),
            "split": D["split"].to_numpy(), "T_mult": Tm, "y": y,
            "dir_up": (sgn > 0).astype(int), "trig_min": tmin.astype(float),
            "T_atr": T, "atr_at_anchor": atr, "n_signals": D["n_signals"].to_numpy(),
            "side_bottom": (D["side"].to_numpy() == "bottom").astype(int),
            "feat_bar": fb, "resolved": (cont | rev).astype(int), "bar_idx": bidx,
        })
        for s in SIGNALS:
            r[f"sig_{s}"] = D["signals"].astype(str).str.contains(s).astype(int).to_numpy()
        r = r[keep].reset_index(drop=True)
        fb2 = r["feat_bar"].to_numpy()
        for k, v in F.items(): r[f"f_{k}"] = v[fb2]
        for k, v in XS.items(): r[f"x_{k}"] = v[fb2]
        r["f_hour"] = pd.to_datetime(ts5[fb2]).hour
        r["f_dow"] = pd.to_datetime(ts5[fb2]).dayofweek
        r["f_speed"] = r["T_atr"] / np.maximum(r["trig_min"] + 1, 1)
        rows.append(r)
        print(f"      T={Tm}×ATR: 발동 {len(r):,} · 배리어해소 {r.resolved.mean():.1%} · "
              f"돌파율 {r.y.mean():.4f}", flush=True)
    A = pd.concat(rows, ignore_index=True)
    A.to_parquet(OUT / "dataset.parquet")
    print(f"\n[5/5] 저장 {A.shape} -> {OUT/'dataset.parquet'}")
    print("\n=== 무조건-되돌림 베이스라인 정확도 (1-돌파율) ===")
    print(A.pivot_table(index=["anchor", "T_mult"], columns="split",
                        values="y", aggfunc=lambda s: 1 - s.mean()).round(4).to_string())
    print(f"\n피쳐 {len([c for c in A.columns if c.startswith(('f_','x_','sig_'))])}개")
    print(json.dumps({"rows": len(A)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
