"""4h 변동성 QX4(scripts/live_vol4h_qx4_20261025.py) «REST 5분봉 4코인 → 피쳐 → HGB·MLP 평균 → payload(+QX4I)» 정답지 (2026-11-21 · tests/qx4.rs).

원본 함수 그대로: frames_from_klines · predict_log(구성원별 _pred) · qx4_from_klines · atm_iv · qx4i_log + 서버 qx4_payload 의 QX4I 덧붙이기(:3064-3067 같은 식).
안전(소스 확인): 원본은 import 때 numpy·pandas 와 연구 모듈(research_vol4h_forecast_horserace → research_system_trend_ruleb · live_eth_sizing_vol_model ·
research_eth_dir4h_reverify → live_eth_dir4h; 전부 import 때 상수뿐 · REST 는 함수 안)만 부른다. 여기서는 refresh_rows·deribit_get(네트워크)을 부르지 않는다.
모델 = 개발기 사본 tmp/vol4h_qx4/qx4_2026-10-23.joblib(읽기만). 입력 = 합성 klines(REST 꼴 문자열) · 합성 Deribit 옵션 체결.

    (tools/server_env.sh 의 환경) python rtmodel/tools/export_qx4.py && python rtmodel/tools/gen_qx4_fixture.py   # → tests/fixtures/qx4_fixture.json
"""
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_vol4h_qx4_20261025 as Q  # noqa: E402

ART = Path("/home/kbj20/crypto-scalping/tmp/vol4h_qx4/qx4_2026-10-23.joblib")
OUT = ROOT / "rtmodel" / "tests" / "fixtures"
BAR = 300_000
PX = {"ETH": 3000.0, "XRP": 1.3, "BTC": 90000.0, "SOL": 150.0}


def bits(a):
    return np.ascontiguousarray(np.asarray(a, np.float64)).view(np.uint64).tolist()


def rows_for(rng, coin, end_ms, n, gaps=0, sig=2e-3):
    t = end_ms - BAR * np.arange(n, 0, -1)
    c = PX[coin] * np.exp(np.cumsum(rng.normal(0, sig, n) * np.exp(rng.normal(0, .3, n))))
    h = c * (1 + np.abs(rng.normal(0, sig, n))); l = c * (1 - np.abs(rng.normal(0, sig, n))); v = np.exp(rng.normal(8, 1, n)); nt = np.round(np.exp(rng.normal(6, .5, n)))
    keep = np.ones(n, bool)
    if gaps:
        keep[rng.choice(np.arange(100, n - 3), gaps, replace=False)] = False
    d = 4 if coin != "XRP" else 6
    return [[int(t[i]), f"{c[i]:.{d}f}", f"{h[i]:.{d}f}", f"{l[i]:.{d}f}", f"{c[i]:.{d}f}", f"{v[i]:.3f}", int(t[i]) + BAR - 1, "0", int(nt[i]), "0", "0", "0"]
            for i in range(n) if keep[i]]


def trades(rng, d_ms, coin, k=60):
    """[d−1h, d) 옵션 체결(만기·행사가·콜풋 섞음 · 블록·청산·범위 밖·만기 너무 멂·iv 없음 포함)."""
    out = []
    idx = {"ETH": 3000.0, "BTC": 90000.0}[coin]
    for j in range(k):
        ts = int(d_ms - rng.integers(-300_000, 3_600_000))
        exp = pd.Timestamp(d_ms, unit="ms").normalize() + pd.Timedelta(days=int(rng.integers(0, 14)))
        strike = int(idx * np.exp(rng.normal(0, .05)) // 50 * 50)
        cp = "C" if rng.random() < .5 else "P"
        t = {"timestamp": ts, "instrument_name": f"{coin}-{exp.strftime('%d%b%y').upper().lstrip('0')}-{strike}-{cp}", "iv": float(np.round(rng.uniform(30, 90), 2)),
             "index_price": idx * float(np.exp(rng.normal(0, .002))), "trade_id": f"t{j}"}
        if j % 17 == 0:
            t["block_trade_id"] = "b1"
        if j % 23 == 0:
            t["liquidation"] = "M"
        if j % 29 == 0:
            t["iv"] = None
        out.append(t)
    return out


def enc(o):
    if isinstance(o, dict):
        return {k: enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [enc(v) for v in o]
    if isinstance(o, (bool, np.bool_)) or o is None or isinstance(o, str):
        return o if not isinstance(o, np.bool_) else bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    return {"b": int(np.float64(o).view(np.uint64))}


def main():
    warnings.filterwarnings("ignore")
    art = Q.load_qx4(ART)
    rng = np.random.default_rng(20261121)
    end = int(pd.Timestamp("2026-11-20 14:00", tz="UTC").timestamp()) * 1000
    n0 = Q.NEED_BARS + 300
    cases = [("normal", {}, end + 60_000), ("gaps", {"gaps": 30}, end + 60_000), ("many_gaps", {"gaps": 40}, end + 60_000), ("stale", {}, end + 11 * 60_000),
             ("sol_lag", {"lag": "SOL"}, end + 60_000), ("forming", {"forming": True}, end + 60_000), ("short", {"n": Q.NEED_BARS - 2000}, end + 60_000),
             ("high_vol", {"sig": 8e-3}, end + 60_000)]
    fx = {"sets": [], "atm_iv": []}
    for name, kw, now in cases:
        rows = {s: rows_for(rng, s, end, kw.get("n", n0), kw.get("gaps", 0), kw.get("sig", 2e-3)) for s in Q.COINS}
        if kw.get("lag"):
            rows[kw["lag"]] = rows[kw["lag"]][:-1]
        if kw.get("forming"):
            for s in Q.COINS:
                r = rows[s][-1]; rows[s].append([r[0] + BAR, r[1], "99999", r[3], "99999", "1", now + 10**9, "0", 1, "0", "0", "0"])
        out = {}
        for coin in Q.COINS:
            res = Q.qx4_from_klines(rows, art, now, coin)
            if res.get("ok") and coin in Q.BETA_I:                         # 서버 qx4_payload 의 QX4I 덧붙이기(iv 는 합성 체결의 atm_iv)
                d = int((now - Q.SETTLE_MS) // Q.H1 * Q.H1)
                iv = Q.atm_iv(trades(rng, d, coin), d, coin)
                if iv is not None:
                    lq = Q.qx4i_log(res["log_rv"], iv, coin); s_ = math.exp(lq / 2)
                    res.update(iv_atm=iv, iv_hour_ms=d, sig4h_iv_pct=round(100 * s_, 3), sig4h_iv_usd=round(res["px"] * s_, 2), log_rv_iv=lq)
                out[coin + "_iv_input"] = iv
            out[coin] = res
        # 구성원별 반올림 전 예측(끝 40봉) -- 원인 보기용
        dbg = {}
        try:
            Fs, xs, filled = Q.frames_from_klines(rows, now)
            for coin in Q.COINS:
                F = Fs[coin].iloc[Q.WARM:]; good = F[np.isfinite(F[Q.R.SETS["M4"] + Q.R.MKT].to_numpy(float)).all(1)].tail(40)
                a = art["coins"][coin]
                dbg[coin] = {"ts": [int(t.value // 10**6) for t in good.index], "M3": bits(Q._pred(a["M3"], good)), "M6": bits(Q._pred(a["M6"], good)),
                             "PX3": bits(Q._pred(art["PX3"], good)), "PX6": bits(Q._pred(art["PX6"], good)),
                             "feats": {c: bits(good[c]) for c in Q.R.SETS["M4X"]}}
        except Exception as e:
            dbg = {"err": str(e)}
        fx["sets"].append({"name": name, "now_ms": now, "rows": rows, "out": enc(out), "dbg": dbg})
        print(name, {c: (o.get("reason") or (o.get("sig4h_pct"), o.get("sig4h_iv_pct"))) for c, o in out.items() if isinstance(o, dict)}, flush=True)
    for k in range(40):                                                     # atm_iv 단독(경계: 범위 밖 체결 · 델타 경계 · 없음)
        d = end - k * Q.H1; coin = ("ETH", "BTC")[k % 2]; tr = trades(rng, d, coin, 0 if k == 39 else 80)
        fx["atm_iv"].append({"d": d, "coin": coin, "trades": tr, "iv": Q.atm_iv(tr, d, coin)})
    p = OUT / "qx4_fixture.json"; p.write_text(json.dumps(fx)); print(p.name, f"{p.stat().st_size:,} 바이트 · 벌 {len(fx['sets'])}")


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
