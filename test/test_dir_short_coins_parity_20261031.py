#!/usr/bin/env python3
"""BTC·SOL·XRP 15분·1시간 방향 서빙 파리티(2026-10-31): 서빙 train65()+load_art+predict 10월 행 = 연구 preds.npz 고른 팔(5시드 평균).

로컬 vision 패널(연구가 쓴 것)로 10월 모형(학습 경계 2026-10-01)을 학습해 data/live/{eth,btc,xrp}_dirshort65_model.joblib 에 쓰고(서버로 복사할 산출물),
코인별로 모아 연구 예측과 대조한다. 씨앗 data/live/{btc,sol,xrp}_dirshort_oos_seed.parquet 도 연구 예측과 같은지 본다(--seed 면 새로 쓴다). REST 없음.
    python test/test_dir_short_coins_parity_20261031.py [--seed] [--skip-train]
"""
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_dir4h_20261015 as d4  # noqa: E402
import live_eth_dir_short_20261028 as ds  # noqa: E402

WT = Path("/home/kbj20/crypto-scalping/.claude/worktrees")
PANEL = WT / "ema5-20-vwap-trading-rules-d6cb37/data/live/dirshort_panel"
RES = WT / "agent-a2a3b3123a31a1ea7/tmp"
# 코인 → {지평: 연구 예측 파일·키}. XRP 1시간(E 팔) = 코인 연구 XRP_control(ETH65 와 같은 스레드 1) · XRP 15분(X 팔) = XRP 연구(스레드 4)
REF = {"BTC": {3: ("coin_dir_short_20261031/BTC", "BTC_3"), 12: ("coin_dir_short_20261031/BTC", "ETH_12")},
       "SOL": {3: ("coin_dir_short_20261031/SOL", "ETH_3"), 12: ("coin_dir_short_20261031/SOL", "ETH_12")},
       "XRP": {3: ("xrp_dir_short_20261031", "XRP_3"), 12: ("coin_dir_short_20261031/XRP_control", "ETH_12")}}
SEED_REF = {"BTC": REF["BTC"], "SOL": REF["SOL"], "XRP": {3: ("xrp_dir_short_20261031", "XRP_3"), 12: ("xrp_dir_short_20261031", "ETH_12")}}   # 채점된 실행
MONTH, TEST = pd.Timestamp("2026-10-01"), (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-10-28"))
FR: dict = {}


def frames():
    if not FR:
        FR.update({s: pd.read_parquet(PANEL / f"{s}USDT.parquet") for s in ["ETH"] + d4.COINS})
    return FR


def fit(src):
    import joblib
    art = ds.train65(frames(), MONTH, src)
    joblib.dump(art, ds.ART65[src])
    return src


def ref(coin, h, table=REF):
    d, k = table[coin][h]; z = np.load(RES / d / "preds.npz")
    return pd.DatetimeIndex(z["t"]), z[k].mean(0)


if __name__ == "__main__":
    frames()
    if "--skip-train" not in sys.argv:
        with get_context("fork").Pool(2) as pool:   # ETH(지평 2개) ∥ BTC → XRP
            for s in pool.imap_unordered(fit, ["ETH", "BTC", "XRP"]):
                print("학습·저장", ds.ART65[s], flush=True)
    worst = 0.0
    for coin in ds.TABLE:
        art = ds.load_art(coin); f = d4.frames_for(frames(), coin)
        t, X, _, _ = ds.dataset(f, None)
        P = ds.predict(art, X[(t >= MONTH) & (t < MONTH + pd.offsets.MonthBegin(1))])
        for h, nm in ds.NAMES.items():
            tr, pr = ref(coin, h); m = (tr >= MONTH) & np.isfinite(pr)
            mine = pd.Series(P[h], index=t[(t >= MONTH) & (t < MONTH + pd.offsets.MonthBegin(1))]).reindex(tr[m]).to_numpy()
            diff = float(np.max(np.abs(mine - pr[m])))
            # 스레드 1 = 비트 단위 0 · 스레드 4(XRP 15분) = 히스토그램 합 순서 차 → ETH --parity 문턱 1e-4(예측 표준편차 ≈ 5)
            worst = max(worst, diff / (1e-9 if ds.NJ65[art["src"][nm]] == 1 else 1e-4))
            print(coin, nm, f"학습 {art['src'][nm]} · 10월 {m.sum():,}행 최대 차 {diff:.2e}", flush=True)
        # 분위 씨앗 = 연구 시험 창 고른 팔 5시드 평균
        seed = {}
        for h, nm in ds.NAMES.items():
            tr, pr = ref(coin, h, SEED_REF); seed[nm] = pd.Series(pr, index=tr)[(tr >= TEST[0]) & (tr < TEST[1])]
        S = pd.DataFrame({"ts": seed["15m"].index, "p15m": seed["15m"].to_numpy(), "p1h": seed["1h"].reindex(seed["15m"].index).to_numpy()}).dropna()
        if "--seed" in sys.argv:
            S.to_parquet(ds.OOS_SEEDS[coin], index=False); print("씨앗", ds.OOS_SEEDS[coin], len(S))
        old = pd.read_parquet(ds.OOS_SEEDS[coin])
        assert old.equals(S.reset_index(drop=True)), f"{coin} 씨앗 ≠ 연구 예측"
        pay = ds.payload_from_features(art, d4.build_features(f).tail(13))   # 화면 응답 = 같은 predict
        assert pay["asset"] == coin.lower() and pay["book"] == "n/a" and len(pay["hist"]) == 13 and set(pay["pred"]) == {"15m", "1h"}
        assert all(0 <= pay["pred"][nm]["pct"] <= 100 for nm in ds.NAMES.values()) and pay["src"] == art["src"]
    assert worst < 1, f"서빙 예측 ≠ 연구 예측 (문턱 대비 {worst:.2f}배)"
    print(f"dir-short coins parity ok — 최대 차/문턱 {worst:.2e}")
