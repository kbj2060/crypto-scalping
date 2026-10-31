#!/usr/bin/env python3
"""SOL·XRP 4h 방향 서빙 파리티(2026-10-31): 서빙 train()+payload_from_features 예측 = 연구 preds.npz(SOL_E·XRP_J) 2026-10 행.

서버와 같은 패널 모양(10월 모형 = 9-30 끝 480일 + 10월 봉)을 로컬 vision 패널에서 잘라 쓴다(REST 없음). 몇 분 걸린다.
    OMP_NUM_THREADS=3 python test/test_dir4h_coins_parity_20261031.py [--save]   # --save = data/live/{sol,xrp}_dir4h_model.joblib 로 저장
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_dir4h_20261015 as d4  # noqa: E402

PANEL = Path("/home/kbj20/crypto-scalping/.claude/worktrees/ema5-20-vwap-trading-rules-d6cb37/data/live/dirshort_panel")
PREDS = Path("/home/kbj20/crypto-scalping/.claude/worktrees/agent-ab3571256711ceb43/tmp/btc_sol_xrp_dir4h_20261031/preds.npz")
MONTH = pd.Timestamp("2026-10-01")

if __name__ == "__main__":
    import joblib
    z = np.load(PREDS)
    start = MONTH - pd.Timedelta("1D") - pd.Timedelta(days=480)   # vision_update(9-30, days=480) 이 남기는 첫 날
    frames = {s: (lambda f: f[f.timestamp >= start].reset_index(drop=True))(pd.read_parquet(PANEL / f"{s}USDT.parquet")) for s in ["ETH"] + d4.COINS}
    for coin, key in (("SOL", "SOL_E"), ("XRP", "XRP_J")):
        art = d4.train(frames, MONTH, coin=coin)
        F = d4.build_features(d4.frames_for(frames, coin))
        te = (F.ts >= MONTH).to_numpy()
        mine = d4.predict(art, F[te])
        ref = pd.Series(z[key], index=z["ts"]).reindex(F.ts[te]).to_numpy()
        diff = float(np.nanmax(np.abs(mine - ref)))
        past = z[key][(z["ts"] < np.datetime64(MONTH)) & np.isfinite(z[key])]
        qlo, qhi = np.quantile(past, [.2, .8])
        pay = d4.payload_from_features(art, F[te])      # 화면 응답 = 같은 예측 × bp_scale · 신호 = 모형 단위 문턱
        p_last = mine[-1]
        print(coin, f"10월 {te.sum():,}행 최대 차 {diff:.2e} · 문턱 {art['lo']:.4f}/{art['hi']:.4f} vs 연구 {qlo:.4f}/{qhi:.4f} · bp_scale {art['bp_scale']:.3f} · spread {art['spread_ratio']}")
        assert diff < 1e-9, f"{coin} 서빙 예측 ≠ 연구 예측 ({diff})"
        assert abs(art["lo"] - qlo) < 1e-4 * max(1, abs(qlo)) and abs(art["hi"] - qhi) < 1e-4 * max(1, abs(qhi)), "문턱 ≠ 연구 month_fixed_q"
        assert pay["asset"] == coin.lower() and pay["signal"] == (1 if p_last >= art["hi"] else -1 if p_last <= art["lo"] else 0)
        assert abs(pay["pred_bp"] - round(p_last * art["bp_scale"], 2)) < 1e-9 and pay["lo"] < 0 < pay["hi"]
        if "--save" in sys.argv:
            joblib.dump(art, d4.ARTIFACTS[coin])
            print("저장", d4.ARTIFACTS[coin])
    print("dir4h coins parity ok")
