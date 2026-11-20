"""박스 유지 «봉 열 → 피쳐 → 확률 → payload» 정답지 (2026-11-20 · R5 둘째 조각 · tests/box_hold.rs).

정답은 원본 dashboard/box_hold.py 를 그대로 불러 만든다. 안전: box_hold 는 import 때 joblib·json 파일 읽기뿐이고 네트워크·.env·거래소를
안 건드린다(소스 확인). 봉은 서버(server.py liq_klines_df)가 넘기는 꼴 그대로 합성한다 -- timestamp(UTC)·high·low·close·volume·taker_buy_base.
Platt90 파일은 개발기에 없어서(서버가 매달 씀) 대역 파일을 fixtures/platt/<변형>/ 에 쓰고 BH.PLATT90 의 경로만 바꾼다(코인 표는 그대로).
반올림 전 확률 = 같은 payload 를 BH.round 를 항등으로 가린 채 한 번 더(모듈 전역 이름이 내장 round 보다 먼저 찾아진다).

    python rtmodel/tools/gen_box_fixture.py      # → rtmodel/tests/fixtures/box_fixture.json (커밋 안 함 · 약 25MB)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dashboard"))
import box_hold as BH  # noqa: E402

OUT = ROOT / "rtmodel" / "tests" / "fixtures"
COINS = {"eth": (2500.0, 0.01), "btc": (90000.0, 0.1), "sol": (150.0, 0.01), "xrp": (1.3, 0.0001), "soxl": (30.0, 0.01)}   # 시작가·호가 단위
PLATT = ["none", "fresh", "stale", "wrong_model", "nan_a", "naive_created", "missing_key"]
VOLS = [None, "iv", "noiv", "stale", "notok"]


def b64(x) -> int:
    return int(np.array([x], np.float64).view(np.uint64)[0])


def enc(o):
    """payload → JSON: 실수는 {"b": f64 비트}(십진 왕복 없이 비트 그대로 대조)."""
    if isinstance(o, dict):
        return {k: enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [enc(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return {"b": b64(o)}
    return o


def bars(rng, n, px0, tick, start_ms, sig):
    c = px0 * np.exp(np.cumsum(rng.normal(0, sig, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, sig * 0.6, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, sig * 0.6, n)))
    q = lambda x: np.round(x / tick) * tick  # noqa: E731 -- 실제 시세처럼 호가 단위(같은 값이 자주 나온다)
    h, l, c = q(h), q(l), q(c)
    h, l = np.maximum(h, np.maximum(c, l)), np.minimum(l, np.minimum(c, h))
    v = np.exp(rng.normal(3, 1, n))
    tb = v * rng.uniform(0.2, 0.8, n)
    t = start_ms + 60_000 * np.arange(n, dtype=np.int64)
    return {"t": t, "h": h, "l": l, "c": c, "v": v, "tb": tb}


def events_for(rng, tc, dense):
    iso = lambda ms: pd.Timestamp(int(ms), unit="ms", tz="UTC").isoformat()  # noqa: E731
    ev = []
    if dense:   # 경계: tc 그 자리 · tc+W(각 척도) · 직전 60분 정확히 · 61분 · 기자회견·의사록(주요 아님) · 실업수당 · 산업생산(13) · 못 읽는 시각들
        for d in (0, 30, 60, 240, 1, 241, -60, -61):
            ev.append({"time_utc": iso(tc + d * 60_000), "category": "econ", "detail": f"CPI (FRED, release_id={[10, 50, 53, 54, 46, 9][abs(d) % 6]})"})
        ev += [{"time_utc": iso(tc + 20 * 60_000), "category": "fomc", "title_ko": "FOMC 기자회견"},
               {"time_utc": iso(tc + 25 * 60_000), "category": "fomc", "title_ko": "FOMC 의사록"},
               {"time_utc": iso(tc + 45 * 60_000), "category": "econ", "detail": "release_id=180"},
               {"time_utc": iso(tc + 50 * 60_000), "category": "econ", "detail": "release_id=13"},
               {"time_utc": iso(tc + 10 * 60_000), "category": "econ", "detail": "release_id= release_id=54"},
               {"time_utc": "bad", "category": "fomc", "title_ko": "FOMC 금리결정 발표"},
               {"category": "fomc", "title_ko": "FOMC 금리결정 발표"},
               {"time_utc": pd.Timestamp(int(tc + 200 * 60_000), unit="ms").isoformat(), "category": "fomc", "title_ko": "FOMC 금리결정 발표"},   # 시간대 없음 = UTC 로 읽음
               {"time_utc": pd.Timestamp(int(tc + 90 * 60_000), unit="ms", tz="UTC").tz_convert("America/New_York").isoformat(), "category": "econ", "detail": "release_id=180"},
               {"time_utc": pd.Timestamp(int(tc + 7 * 60_000 + 123456), unit="ms", tz="UTC").isoformat(), "category": "econ", "detail": "release_id=9"},
               {"time_utc": iso(tc - 30 * 60_000)[:10], "category": "econ", "detail": "release_id=46"}]
    else:
        for _ in range(rng.integers(0, 6)):
            d = int(rng.integers(-300, 600))
            if rng.random() < 0.3:
                ev.append({"time_utc": iso(tc + d * 60_000), "category": "fomc", "title_ko": "FOMC 금리결정 발표"})
            else:
                ev.append({"time_utc": iso(tc + d * 60_000), "category": "econ", "detail": f"x (FRED, release_id={int(rng.choice([10, 50, 180, 13, 9]))})"})
    return ev


def cases(asset, rng):
    px0, tick = COINS[asset]
    start = lambda: int(pd.Timestamp("2025-01-01", tz="UTC").value // 10**6) + int(rng.integers(0, 600 * 1440)) * 60_000  # noqa: E731
    sig = lambda: float(rng.uniform(3e-4, 2e-3))  # noqa: E731
    out = []
    for i in range(10):
        out.append(("normal", bars(rng, 1559, px0, tick, start(), sig())))
    out.append(("short_1498", bars(rng, 1498, px0, tick, start(), sig())))
    out.append(("exact_1499", bars(rng, 1499, px0, tick, start(), sig())))
    out.append(("long_2100", bars(rng, 2100, px0, tick, start(), sig())))
    b = bars(rng, 1559, px0, tick, start(), sig()); j = np.r_[np.zeros(1529), rng.normal(0, 0.01, 30)]   # 급등락
    for k in ("h", "l", "c"):
        b[k] = b[k] * np.exp(np.cumsum(j))
    out.append(("spike", b))
    b = bars(rng, 1559, px0, tick, start(), sig()); z = rng.random(1559) < 0.2; z[-5:] = True; b["v"][z] = 0; b["tb"][z] = 0
    out.append(("zero_vol", b))
    b = bars(rng, 1559, px0, tick, start(), sig()); b["v"][-240:] = 0; b["tb"][-240:] = 0
    out.append(("zero_vol_4h", b))
    b = bars(rng, 1559, px0, tick, start(), sig()); b["c"][-120:-60] = b["c"][-121]; b["h"][-120:-60] = b["l"][-120:-60] = b["c"][-121]
    out.append(("flat_prev_1h", b))   # box_hold selftest ④ -- 1h prng = 0
    b = bars(rng, 1559, px0, tick, start(), sig()); b["c"][-300:] = b["c"][-301]; b["h"][-300:] = b["l"][-300:] = b["c"][-301]
    out.append(("flat_300", b))       # 창폭 0 → 세 척도 다 ok False
    b = bars(rng, 1559, px0, tick, start(), sig()); b["c"][-1445:] = b["c"][-1446]; b["h"][-1445:] = b["l"][-1445:] = b["c"][-1446]
    out.append(("flat_24h", b))       # s24 = 0 → 파이썬 ZeroDivisionError
    b = bars(rng, 1559, px0, tick, start(), sig()); b["c"][700] = np.nan; b["h"][-20] = np.nan; b["v"][-100] = np.nan
    out.append(("nan_mix", b))
    b = bars(rng, 1559, px0, tick, start(), sig()); b["h"][300] = np.nan
    out.append(("nan_h_old", b))      # 4h R_squeeze 중앙값에만 닿는 NaN
    b = bars(rng, 1559, px0, tick, start(), sig()); b["tb"][-3] = np.nan
    out.append(("nan_tb", b))
    b = bars(rng, 1700, px0, tick, start(), sig()); keep = np.sort(rng.choice(1700, 1559, replace=False)); keep[-1] = 1699
    out.append(("gaps", {k: x[keep] for k, x in b.items()}))
    for nm, at in (("sun_2359", "2026-03-01 23:59"), ("mon_0000", "2026-03-02 00:00"), ("fri_2359", "2026-02-27 23:59"), ("sat_0000", "2026-02-28 00:00")):
        last = int(pd.Timestamp(at, tz="UTC").value // 10**6)
        out.append((nm, bars(rng, 1559, px0, tick, last - 1558 * 60_000, sig())))
    out.append(("events_dense", bars(rng, 1559, px0, tick, start(), sig())))
    for w in (30, 60, 240):   # 가장 가까운 다음 발표가 정확히 tc+W · 직전 발표가 정확히 60분 전(부등호 경계)
        out.append((f"ev_at_{w}", bars(rng, 1559, px0, tick, start(), sig())))
    out.append(("huge_px", bars(rng, 1559, px0 * 1e3, tick, start(), sig())))
    out.append(("tiny_px", bars(rng, 1559, px0 * 1e-4, tick * 1e-6, start(), sig())))
    for nm, last in (("px_10.00", 10.0), ("px_9.99", 9.99)):
        b = bars(rng, 1559, last, 0.01, start(), sig()); b["c"] = b["c"] * (last / b["c"][-1]); b["c"][-1] = last
        b["h"], b["l"] = np.maximum(b["h"], b["c"]), np.minimum(b["l"], b["c"])
        out.append((nm, b))
    return out


def platt_files(now_ms):
    now = pd.Timestamp(now_ms, unit="ms", tz="UTC")
    for v in PLATT[1:]:
        d = OUT / "platt" / v
        d.mkdir(parents=True, exist_ok=True)
        for a in ("eth", "xrp", "btc", "sol"):
            c = {"a": 1.0873, "b": -0.0412, "fit_start": "2026-09-01 00:00:00+00:00", "fit_end": "2026-10-31 00:00:00+00:00", "n_rows": 1,
                 "created_utc": (now - pd.Timedelta(days=3)).isoformat(), "model_file": BH.HGB[a], "asset": a}
            if v == "stale":
                c["created_utc"] = (now - pd.Timedelta(days=41)).isoformat()
            elif v == "wrong_model":
                c["model_file"] = "box_hold_hgb_old.joblib"
            elif v == "nan_a":
                c["a"] = float("nan")
            elif v == "naive_created":
                c["created_utc"] = (now - pd.Timedelta(days=3)).tz_localize(None).isoformat()
            elif v == "missing_key":
                del c["fit_end"]
            (d / f"box_hold_platt90_{a}_1h.json").write_text(json.dumps(c))


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rng = np.random.default_rng(20261120)
    now_ms = int(time.time() * 1000)
    platt_files(now_ms)
    orig = dict(BH.PLATT90)
    fx = {"now_ms": now_ms, "sets": []}
    tim = []
    for asset in COINS:
        bb = BH.BUNDLES[asset]
        prof = np.asarray(bb["prof"], float)
        for k, (name, b) in enumerate(cases(asset, rng)):
            n = len(b["t"])
            k1m = pd.DataFrame({"timestamp": pd.to_datetime(b["t"], unit="ms", utc=True), "high": b["h"], "low": b["l"], "close": b["c"],
                                "volume": b["v"], "taker_buy_base": b["tb"]})
            tc = int(b["t"][-1]) + 60_000
            ev = events_for(rng, tc, name == "events_dense" or k % 4 == 0)
            if name.startswith("ev_at_"):
                w = int(name[6:])
                iso = lambda ms: pd.Timestamp(int(ms), unit="ms", tz="UTC").isoformat()  # noqa: E731
                ev = [{"time_utc": iso(tc + w * 60_000), "category": "econ", "detail": "release_id=10"},
                      {"time_utc": iso(tc + w * 60_000), "category": "econ", "detail": "release_id=180"},
                      {"time_utc": iso(tc - 60 * 60_000), "category": "fomc", "title_ko": "FOMC 금리결정 발표"}]
            pv = PLATT[k % len(PLATT)]
            for a in orig:
                BH.PLATT90[a] = orig[a] if pv == "none" else OUT / "platt" / pv / f"box_hold_platt90_{a}_1h.json"
            vol, vk = None, VOLS[k % len(VOLS)] if asset == "eth" else None
            if vk:
                vol = {"ok": vk != "notok", "asof_ms": tc - (20 if vk == "stale" else 2) * 60_000,
                       "log_rv": float(np.log(rng.uniform(0.001, 0.02) ** 2)), "log_rv_iv": None if vk == "noiv" else float(np.log(rng.uniform(0.001, 0.02) ** 2))}
            feats = err = None
            if n >= BH.NEED:
                try:
                    F = BH.features(k1m.tail(BH.NEED + 60).reset_index(drop=True).astype({x: float for x in ("high", "low", "close", "volume", "taker_buy_base")}), ev, prof)
                    feats = {s: {c: b64(v) for c, v in f.items()} for s, f in F.items()}
                except ZeroDivisionError as e:
                    err = f"{type(e).__name__}: {e}"

            def serve():   # server.py api_box_hold produce(): payload → (ETH 만) with_qx4i · 예외 = {"ok": False, "reason": ...}
                try:
                    out = BH.payload(k1m.copy(), ev, asset)
                    return BH.with_qx4i(out, vol) if asset == "eth" else out
                except Exception as e:  # noqa: BLE001
                    return {"ok": False, "reason": f"{type(e).__name__}: {e}"[:200]}
            t0 = time.perf_counter_ns(); pay = serve(); tim.append(time.perf_counter_ns() - t0)
            BH.round = lambda x, n=None: x  # noqa: E731
            raw = serve()
            del BH.round
            fx["sets"].append({"asset": asset, "name": name, "bars": [[int(t), b64(h), b64(l), b64(c), b64(v), b64(tb)] for t, h, l, c, v, tb in
                                                                         zip(b["t"], b["h"], b["l"], b["c"], b["v"], b["tb"])],
                               "events": ev, "platt": pv, "vol": None if vol is None else {**vol, "log_rv": b64(vol["log_rv"]),
                                                                                         "log_rv_iv": None if vol["log_rv_iv"] is None else b64(vol["log_rv_iv"])},
                               "features": feats, "features_err": err, "payload": enc(pay), "payload_raw": enc(raw)})
    BH.PLATT90.update(orig)
    # 파이썬 round(x, nd) 대조표(반올림 규칙 -- 정확한 반값 포함)
    xs = list(rng.uniform(0, 1, 3000)) + list(rng.uniform(0, 1e5, 2000)) + [k / 2**m for m in (1, 2, 3, 4, 5, 8, 10) for k in range(0, 2**m * 3)]
    fx["round"] = [[b64(x), nd, b64(round(float(x), nd))] for x in xs for nd in (2, 3, 4, 6)]
    path = OUT / "box_fixture.json"
    path.write_text(json.dumps(fx))
    ok = sum(1 for s in fx["sets"] for v in (s["payload"].get("scales") or {}).values() if v.get("ok"))
    print(f"{path.name} {path.stat().st_size:,} 바이트 · 봉 열 {len(fx['sets'])} · ok 척도 {ok} · 피쳐 예외 {sum(s['features_err'] is not None for s in fx['sets'])}"
          f" · 파이썬 serve() 중앙 {np.median(tim) / 1e6:.2f}ms")


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
