#!/usr/bin/env python3
"""Zeus — **더블 배리어에서도 역변동성 사이징이 이기는가** (2026-09-17, CPU 전용)

## 왜 다시 재나
09-16 전체판(120k 무작위진입·고유일 1,717)에서 역변동성 사이징이 1위였다
(SD −28.3% · 하위1% −32.0% · 50배 청산율 13.31→**7.06%** · 로그성장 +13.30 vs 고정 +12.93).
🔴**그런데 그 판은 「보유 4h 고정」이었다.** 그 설정에서는 고변동 = 4시간 동안 더 큰 움직임 =
더 큰 손익 분산이라 역변동성이 분산을 직접 줄인다.

**Zeus 는 더블 배리어(TP1.5%/SL1%, 시간청산 없음)로 간다. 그러면 건당 수익이 +150 또는 −100bp
로 «고정»되고, 건당 분산이 변동성과 무관해진다**(Var = p(1−p)·250²). 역변동성의 원래 근거가
사라진다. 남는 경로는 **시간**뿐이다 — 고변동에서 배리어에 빨리 닿아 회전이 빨라지고,
슬롯 하나 기준 단위시간당 노출이 커진다. 그걸 줄이는 효과가 남는지가 이 실험이다.

## 두 번째 질문 — 우리 후보는 조용한 봉에 몰린다
2026-09-17 측정: **부모 후보 봉의 ATR 중앙값이 전체 봉의 64.8%**(창96) / 72.9%(창192).
전체판은 무작위 진입이라 변동성 범위가 넓었다. **우리 후보 집합에서는 역변동성이 쓸 재료가 좁다.**
그래서 ATR 분포를 후보에 맞춘 진입(`volmatch`)을 나란히 돌린다 — 부모 없이도 잴 수 있다.

## 설계 (2×2)
    진입 {무작위, 변동성매칭} × 청산 {보유4h(=09-16 대조), 더블배리어}
09-16 규약을 그대로 따른다: 평균 명목 정규화 · 양측면 균형 · **①고정을 1번 팔로** ·
위험 분포(SD·하위1%·MAE99·50배 청산율)로 판정 · 실력 0 이라 평균은 ≈ −비용.
⭐기록된 규율: **두 판이 일치하는 항목만 행동 근거로 쓴다.**
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/home/kbj20/crypto-scalping")
OUT = ROOT / "tmp/zeus_sizing_double_barrier_20260917"
RNG = np.random.default_rng(20260917)
N_DRAWS = 120_000
WARMUP = 900
HOLD4H = 48
MAXBARS = 4032
COST_HOLD, COST_BARRIER = 5.88, 1.02      # 09-16 규약(peg) · Zeus 전제(USDC 메이커)
CAND_ATR_RATIO = 0.648                    # 부모 후보 ATR 중앙값 / 전체 봉 ATR 중앙값(창96)


def _mod(rel: str, name: str):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp); argv, sys.argv = sys.argv, ["x"]
    try: sp.loader.exec_module(m)
    finally: sys.argv = argv
    return m


def log(*a): print(*a, flush=True)


def tilt_weights(atr: np.ndarray, target_ratio: float) -> tuple[np.ndarray, float]:
    """w ∝ atr^(−λ) 로 기울여 **가중 중앙값이 target_ratio × 전체 중앙값**이 되게 λ 를 찾는다.

    잘라내지 않고 기울이므로 분포의 범위를 보존한다(절단하면 꼬리가 사라져 사이징이
    쓸 재료를 인위적으로 없앤다).
    """
    a = atr.copy()
    med = float(np.nanmedian(a))
    target = target_ratio * med

    def wmed(lam):
        w = np.power(np.maximum(a / med, 1e-9), -lam)
        o = np.argsort(a); cw = np.cumsum(w[o]); cw /= cw[-1]
        return float(a[o][np.searchsorted(cw, 0.5)])

    lo, hi = 0.0, 40.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if wmed(mid) > target: lo = mid
        else: hi = mid
    lam = (lo + hi) / 2
    w = np.power(np.maximum(a / med, 1e-9), -lam)
    return w / w.sum(), lam


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    K = _mod("scripts/research_omega461_side_skill_decomposition_20260917.py", "K")
    FD = _mod("scripts/research_sizing_horserace_fulldata_20260916.py", "FD")
    MQ = _mod("scripts/live_eth_mae_quantile_model_20260913.py", "MQ")
    SV = _mod("scripts/live_eth_sizing_vol_model_20260912.py", "SV")
    svm = MQ.svm

    kl = pd.read_csv(REPO / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
                     usecols=["timestamp", "open", "high", "low", "close", "volume",
                              "quote_volume", "trades"])
    kl["timestamp"] = pd.to_datetime(kl.timestamp)
    ts = kl.timestamp
    op, c, hi, lo = (kl.open.to_numpy(float), kl.close.to_numpy(float),
                     kl.high.to_numpy(float), kl.low.to_numpy(float))
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
    atr14 = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy() / np.maximum(c, 1e-12)
    atr96 = pd.Series(tr).rolling(96, min_periods=24).mean().to_numpy() / np.maximum(c, 1e-12)
    atr288 = pd.Series(tr).rolling(288, min_periods=144).mean().to_numpy() / np.maximum(c, 1e-12)
    n = len(kl); day = ts.dt.floor("D").astype("int64").to_numpy()
    elig = np.arange(WARMUP, n - MAXBARS - 2)
    ok = np.isfinite(atr14[elig]) & np.isfinite(atr96[elig]) & np.isfinite(atr288[elig])
    elig = elig[ok]
    log(f"ETH 5분봉 {n:,} · {ts.iloc[0]} ~ {ts.iloc[-1]} · 적격 진입봉 {len(elig):,}")

    pw, lam = tilt_weights(atr96[elig], CAND_ATR_RATIO)
    log(f"변동성매칭: w ∝ atr96^(−{lam:.2f}) · 목표 중앙비 {CAND_ATR_RATIO:.3f} "
        f"(부모 후보 실측) · 가중 중앙 ATR {np.nanmedian(atr96[elig]) * CAND_ATR_RATIO * 100:.4f}%")

    entries = {"무작위": RNG.choice(elig, N_DRAWS),
               "변동성매칭": RNG.choice(elig, N_DRAWS, p=pw)}
    base_all = svm.build_features(ts, c, kl.quote_volume.to_numpy(float),
                                  kl.trades.to_numpy(float), hi, lo).replace([np.inf, -np.inf], np.nan)
    art = MQ.load_model()
    try:
        sv = SV.load_model()
        Xv_all = SV.build_features(ts, c, kl.quote_volume.to_numpy(float),
                                   kl.trades.to_numpy(float), hi, lo)
    except Exception as ex:
        sv, Xv_all = None, None
        log(f"  ⚠️배포 vol 모델 없음: {type(ex).__name__} -- 그 팔은 제외")
    tl = pd.DataFrame([json.loads(l) for l in open(REPO / "data/live/account_round_trips.jsonl")])
    userq = (tl.entry_price * tl.max_qty).to_numpy(float)

    rows = []
    for ename, idx in entries.items():
        side = RNG.choice([1, -1], N_DRAWS)
        e = op[idx + 1]
        for xname in ("보유4h", "더블배리어"):
            if xname == "보유4h":
                ret = (c[idx + HOLD4H] - e) / e * 1e4 * side - COST_HOLD
                hold = np.full(N_DRAWS, float(HOLD4H))
                mae = np.empty(N_DRAWS)
                for k, (x, s) in enumerate(zip(idx, side)):
                    w = slice(x + 1, x + 1 + HOLD4H)
                    mae[k] = ((e[k] - lo[w].min()) if s > 0 else (hi[w].max() - e[k])) / e[k] * 1e4
                mae = np.maximum(mae, 0.0)
            else:
                # ⚠️_first_touch_open 은 side 를 «봉 배열»로 읽는다(side[entry_i]). 120k 를 뽑으면
                # 같은 봉이 중복 추출돼(기대 ~14.7k) 롱/숏이 덮어써진다. **측면별로 나눠 돌린다.**
                ret = np.empty(N_DRAWS); hold = np.empty(N_DRAWS); mae = np.empty(N_DRAWS)
                for sd in (1, -1):
                    m = side == sd
                    where = np.where(m)[0]
                    sarr = np.full(n, float(sd))
                    for st in range(0, len(where), 20_000):
                        k = where[st:st + 20_000]
                        r_, h_, _res, _rn, mae_ = K._first_touch_open(
                            idx[k], sarr, hi, lo, c, K.BASE_TP, K.BASE_SL, MAXBARS)
                        ret[k] = r_ * 1e4 - COST_BARRIER
                        hold[k] = h_.astype(float)
                        mae[k] = np.maximum(-mae_ * 1e4, 0.0)
                assert (hold >= 1).all(), "보유봉 0"
                res_sh = float(np.isin(np.round(ret + COST_BARRIER, 6),
                                       [K.BASE_TP * 1e4, -K.BASE_SL * 1e4]).mean())
                log(f"    배리어 해소 비율 {res_sh*100:.2f}% (나머지는 {MAXBARS}봉 미해소 종가청산)")

            r = base_all.iloc[idx].copy().reset_index(drop=True)
            r["log_h"] = np.log(np.maximum(hold, 1) * 5.0); r["side"] = side
            safe = MQ.safe_mae(art["models"], r[MQ.FEATURES], art["mult"])
            arms = {"①고정": np.ones(N_DRAWS),
                    "역ATR 1/atr14": 1.0 / np.maximum(atr14[idx], 1e-9),
                    "역ATR 1/atr288": 1.0 / np.maximum(atr288[idx], 1e-9),
                    "동일위험 1/safeMAE": 1.0 / np.maximum(safe, 1e-9),
                    "재량류(실계좌 분포)": RNG.choice(userq, N_DRAWS)}
            if sv is not None:
                pv = SV.predict_vol(sv["models"], Xv_all.iloc[idx][sv["features"]])
                arms["역변동성(배포모델)"] = 1.0 / np.maximum(pv, 1e-9)
            arms = {k: v for k, v in arms.items() if np.isfinite(v).all()}

            for aname, w in arms.items():
                s = FD.summarize(ret, w, day[idx], mae)
                wn = np.asarray(w, float); wn = wn / wn.mean()
                rows.append(dict(entry=ename, exit=xname, arm=aname, **s,
                                 mean_hold=float(hold.mean()),
                                 hold_w=float((hold * wn).mean()),
                                 per_day=288.0 / max(hold.mean(), 1e-9)))
            log(f"  [{ename} × {xname}] 평균 {ret.mean():+.2f}bp · 중앙보유 {np.median(hold):.0f}봉 "
                f"· 고유일 {len(np.unique(day[idx])):,}")

    D = pd.DataFrame(rows)
    D.round(4).to_csv(OUT / "race.csv", index=False)
    for (ename, xname), g in D.groupby(["entry", "exit"], sort=False):
        fx = g[g.arm == "①고정"].iloc[0]
        log(f"\n{'='*112}\n■ {ename} 진입 × {xname} 청산  (평균명목 정규화 · 작을수록 좋다)")
        log(f"{'규칙':<22}{'가중CV':>8}{'평균bp':>9}{'SD':>9}{'vs고정':>8}{'하위1%':>10}{'vs고정':>8}"
            f"{'50배청산율':>11}{'vs고정':>8}{'가중보유':>9}")
        for r_ in g.sort_values("sd").itertuples():
            log(f"{r_.arm:<22}{r_.cv:>8.2f}{r_.mean:>+9.2f}{r_.sd:>9.1f}"
                f"{(r_.sd/fx.sd-1)*100:>+7.1f}%{r_.p1:>10.1f}{(r_.p1/fx.p1-1)*100:>+7.1f}%"
                f"{r_.liq*100:>10.2f}%{(r_.liq/max(fx.liq,1e-12)-1)*100:>+7.1f}%{r_.hold_w:>9.0f}")
    log(f"\n저장: {OUT}/race.csv")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
