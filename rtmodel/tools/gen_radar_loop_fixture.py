"""위험 레이더 cron 한 회차(scripts/live_risk_radar_20261115.py main) «REST 요청 열 → 캐시 → latest.json·forward.jsonl» 정답지 (2026-11-21 · tests/radar_bin.rs).

원본 함수 그대로: update_cache(1분봉 · 쪽마다 저장) · update_dvol(Deribit 시간봉) · grid · dvol_series · compute · compute_session.
REST(rest·dvol_rest)는 함수 인자로 가짜를 넘기고(합성 거래소 · 요청·응답을 적음), time.sleep 은 가짜 시계로 갈아 끼운다. 캐시 경로는 임시 디렉터리.
main() 은 부르지 않는다 -- 그 안의 «코인 순서 · 코인마다 try · DVOL 실패면 캐시로 · 지연 무효 · 파일 둘 쓰기»(:221-240)를 run() 에 같은 순서로 적었다.

    (tools/server_env.sh 의 환경) python rtmodel/tools/gen_radar_loop_fixture.py     # → tests/fixtures/radar_loop_fixture.json (커밋 안 함)
"""
import datetime as dt
import json
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_risk_radar_20261115 as W  # noqa: E402

OUT = ROOT / "rtmodel" / "tests" / "fixtures"
MIN, H = 60_000, 3_600_000
PX = {"ETHUSDT": 3000.0, "BTCUSDT": 90000.0, "SOLUSDT": 150.0, "XRPUSDT": 1.3, "SOXLUSDT": 30.0}


class Ex:
    def __init__(self, t0):
        self.t0, self.c = t0, {}

    def ser(self, sym):
        if sym not in self.c:
            rng = np.random.default_rng(int.from_bytes(sym.encode()[:8].ljust(8, b"_"), "little"))
            n = 50 * 1440
            self.c[sym] = PX.get(sym, 50.0) * np.exp(np.cumsum(rng.normal(0, 5e-4, n) * np.exp(rng.normal(0, .4, n)))), np.exp(rng.normal(6, 1, n))
        return self.c[sym]

    def klines(self, now, sym, start, lim, log):
        log.append(["fapi", "/fapi/v1/klines", [["symbol", sym], ["interval", "1m"], ["startTime", str(start)], ["limit", str(lim)]]])
        px, v = self.ser(sym); out = []
        t = (start + MIN - 1) // MIN * MIN
        while len(out) < lim and t <= now:
            i = (t - self.t0) // MIN; c = float(np.round(px[i], 4))
            out.append([t, f"{c}", f"{c * 1.0004:.4f}", f"{c * 0.9996:.4f}", f"{c}", f"{v[i]:.3f}", t + MIN - 1, "0", 1, "0", "0", "0"]); t += MIN
        log[-1].append(out)
        return out

    def dvol(self, now, coin, start, end, log, fail=False):
        log.append(["deribit", "/api/v2/public/get_volatility_index_data", [["currency", coin], ["start_timestamp", str(start)], ["end_timestamp", str(end)], ["resolution", "3600"]]])
        if fail:
            log[-1].append({"__status": 500})
            raise ConnectionError("가짜 500")
        out = [[t, 0, 0, 0, round(50 + 10 * np.sin(t / 3.6e7), 2)] for t in range((start + H - 1) // H * H, end, H)]   # 시작 이상 끝 미만 시간봉(형성 중 포함)
        log[-1].append({"result": {"data": out}})
        return out


def run(ex, now, cache, P, sess, log, dvol_fail=False):
    res = {}
    for coin, p in {**P["coins"], **sess}.items():
        try:
            df = W.update_cache(coin, now, lambda sym, s, lim: ex.klines(now, sym, s, lim, log), cache / f"{coin}_1m.parquet")
            if p.get("kind") == "session":
                out = W.compute_session(df, p)
            else:
                cl, hi, lo, m0, qv = W.grid(df)
                iv = None
                if "beta60_iv" in p:
                    try:
                        iv = W.dvol_series(W.update_dvol(coin, now, lambda c, s, e: ex.dvol(now, c, s, e, log, dvol_fail), cache / f"{coin}_dvol.parquet"))
                    except Exception:
                        try:
                            iv = W.dvol_series(pd.read_parquet(cache / f"{coin}_dvol.parquet"))
                        except Exception:
                            iv = None
                out, _ = W.compute(cl, hi, lo, m0, p, P, qv, iv)
            out["data_age_s"] = round((now - int(df.t.iloc[-1]) - MIN) / 1000, 1)
            if out["data_age_s"] > W.STALE_DATA_S:
                out.update(valid=False, error=f"데이터 지연 {out['data_age_s']:.0f}초")
        except Exception as e:
            out = dict(error=f"{type(e).__name__}: {str(e)[:200]}", valid=False)
        res[coin] = out
    payload = dict(updated=dt.datetime.fromtimestamp(now / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), params=W.PARAMS.name, coins=res)
    latest = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    lines = [json.dumps(dict(coin=c, **{k: o[k] for k in ("ts", "price", "sigma15_pct", "baseline7d_sigma15_pct", "sigma60_pct", "usual60_pct", "state60", "data_age_s")}, alert=o["alert"]["active"]),
                        ensure_ascii=False, allow_nan=False) for c, o in res.items() if "error" not in o]
    return latest, lines


def main():
    warnings.filterwarnings("ignore")
    W.time = types.SimpleNamespace(sleep=lambda s: None, time=None)   # 백필 쪽 사이 0.5초 쉼 -- 가짜(요청 열은 같다)
    P = json.loads(W.PARAMS.read_text()); sess = json.loads(W.PARAMS_SESS.read_text())["coins"]
    t0 = int(pd.Timestamp("2026-10-10", tz="UTC").timestamp()) * 1000
    ex = Ex(t0); cache = Path(tempfile.mkdtemp())
    now0 = int(pd.Timestamp("2026-11-20 14:03:07", tz="UTC").timestamp()) * 1000
    runs = []
    # 첫 회차 = 31일 백필 · 다음 = 1분·2분 뒤 꼬리 · 63분 뒤(DVOL 시간봉 새로 · 그 회차는 Deribit 500 → 캐시로) · 64분 뒤(DVOL 다시)
    for k, now in enumerate((now0, now0 + MIN, now0 + 2 * MIN, now0 + 63 * MIN, now0 + 64 * MIN)):
        log = []
        latest, lines = run(ex, now, cache, P, sess, log, dvol_fail=(k == 3))
        runs.append({"now_ms": now, "requests": log, "latest": latest, "log": lines})
        print(k, now, "요청", len(log), "·", {c: o.get("error") or o.get("state60") or o.get("session") for c, o in json.loads(latest)["coins"].items()}, flush=True)
    p = OUT / "radar_loop_fixture.json"; p.write_text(json.dumps({"params": W.PARAMS.name, "runs": runs})); print(p.name, f"{p.stat().st_size:,} 바이트")


if __name__ == "__main__":
    main()
