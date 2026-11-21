"""방향 전진 기록기(#8) · 개장 되돌림 · 4h 변동성 HGB(지평 사다리 대체 길) · 7일/4~13주 신호(dir_signals) 정답지 (2026-11-21 · tests/signals.rs).
원본을 가짜 시계·가짜 fetch 로 부른다 -- 네트워크 없음 · FRED 키는 가짜 문자열(환경 변수는 지운 채로 부른다).
  - scripts/live_eth_direction_forward_recorder_20261010.record(): 00:31·08:40·16:31 · 03:00(시각 밖 → 종료) · SOL 00:25 봉 빠짐(KeyError · 앞 코인은 이미 씀) · BTC 첫 30분 거래 0
  - dashboard/dir_signals.open_rev_from_klines · vol4h_from_klines(개발기 ETH 모델 data/live/eth_sizing_vol_model.joblib) · dir_signals_payload
    (tools/server_env.sh 의 환경) python rtmodel/tools/gen_signals_fixture.py      # → tests/fixtures/signals_fixture.json · svm_eth.json
"""
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_eth_direction_forward_recorder_20261010 as R  # noqa: E402
from dashboard import dir_signals as DS  # noqa: E402
from export_hgb import SKLEARN_OK  # noqa: E402
from export_svm import export  # noqa: E402
import sklearn  # noqa: E402

BAR, DAY = 300_000, 86_400_000
SVM = Path("/home/kbj20/crypto-scalping/data/live/eth_sizing_vol_model.joblib")
D0 = 1_795_000_000_000 // DAY * DAY            # 기준 날
T0 = D0 - 100 * DAY


class Ex:
    """합성 거래소: 심볼마다 5분봉(2021~ 아님 · T0 부터) -- 값은 시각의 결정적 함수."""
    def __init__(self):
        self.s = {}

    def ser(self, sym):
        if sym not in self.s:
            rng = np.random.default_rng(sum(map(ord, sym)))
            n = (D0 + 3 * DAY - T0) // BAR
            c = {"ETHUSDT": 3000.0, "BTCUSDT": 90000.0, "SOLUSDT": 150.0}[sym] * np.exp(np.cumsum(rng.normal(0, 2e-3, n)))
            v = np.exp(rng.normal(8, 1, n))
            self.s[sym] = (c, v, rng.uniform(0.3, 0.7, n), rng.integers(100, 9000, n))
        return self.s[sym]

    def m5(self, sym, t, now):
        c, v, tb, nt = self.ser(sym)
        i = (t - T0) // BAR
        x = c[i] * (1.0005 if t + BAR - 1 >= now else 1.0)
        vv = 0.0 if (sym == "BTCUSDT" and D0 + DAY <= t < D0 + DAY + 1_800_000) else v[i]   # 거래 0 인 첫 30분(imb = 0 길)
        return [t, f"{c[i - 1]:.2f}", f"{max(x, c[i - 1]) * 1.0007:.2f}", f"{min(x, c[i - 1]) * 0.9993:.2f}", f"{x:.2f}", f"{vv:.3f}", t + BAR - 1,
                f"{vv * x:.4f}", int(nt[i]), f"{vv * tb[i]:.3f}", f"{vv * tb[i] * x:.4f}", "0"]

    def klines(self, sym, interval, limit, now, drop=()):
        if interval == "5m":
            last = now // BAR * BAR
            return [self.m5(sym, t, now) for t in range(last - (limit - 1) * BAR, last + 1, BAR) if (sym, t) not in drop]
        last = now // DAY * DAY
        out = []
        for d in range(last - (limit - 1) * DAY, last + 1, DAY):
            k = [self.m5(sym, t, now) for t in range(d, min(d + DAY, now // BAR * BAR + BAR), BAR)]
            out.append([d, k[0][1], max(r[2] for r in k), min(r[3] for r in k), k[-1][4], "1", d + DAY - 1, "0", 1, "0", "0", "0"])
        return out


def main():
    if sklearn.__version__ != SKLEARN_OK:
        raise SystemExit("server_env.sh 의 환경으로")
    os.environ.pop("FRED_API_KEY", None)
    fx = ROOT / "rtmodel/tests/fixtures"
    ex = Ex()
    # ── 1. 전진 기록기
    tmp = Path(tempfile.mkdtemp())
    R.OUT = str(tmp / "eth_direction_forward.jsonl")
    rec = []
    for now, drop in ((D0 + 1_865_000, ()), (D0 + 8 * 3_600_000 + 2_400_000, ()), (D0 + 16 * 3_600_000 + 1_860_000, ()), (D0 + 3 * 3_600_000, ()),
                      (D0 + DAY + 1_900_000, ()), (D0 + 2 * DAY + 1_870_000, (("SOLUSDT", D0 + 2 * DAY + 1_500_000),)),
                      (D0 + 2 * DAY + 16 * 3_600_000 + 1_900_000, ("flat",))):   # 끝 = 일봉 종가 100 하나만 91일 전 200(votes 의 90일 창 경계)
        calls = []

        def kl(sym, interval, limit, now=now, calls=calls, drop=drop):
            r = ex.klines(sym, interval, limit, now, drop)
            if "flat" in drop and interval == "1d":
                r = [k[:4] + ["200" if j == 1 else "100"] + k[5:] for j, k in enumerate(r)]
            calls.append([sym, interval, limit, r])                              # 응답도 적는다
            return r
        R.klines = kl
        R.time = types.SimpleNamespace(time=lambda now=now: now / 1000, strftime=time.strftime, gmtime=time.gmtime)
        try:
            R.record()
            err = None
        except SystemExit as e:
            err = "exit"
        except KeyError:
            err = "KeyError"
        f = Path(R.OUT)
        rec.append({"now": now, "drop": [list(x) if isinstance(x, tuple) else x for x in drop], "err": err, "calls": calls, "file": f.read_text() if f.exists() else None})
        print("기록기", now, err, len(calls))
    # ── 2. 개장 되돌림
    orv = []
    for now in (D0 + 1_200_000, D0 + 1_860_000, D0 + 5 * 3_600_000, D0 + DAY - 60_000, D0 + 5 * 3_600_000 + 0.5):   # 끝 = 서버 시계(time.time()*1000 실수)
        daily, m5 = ex.klines("ETHUSDT", "1d", 93, int(now)), ex.klines("ETHUSDT", "5m", 600, int(now))
        orv.append({"now": now, "daily": daily, "m5": m5, "out": json.dumps(DS.open_rev_from_klines(daily, m5, now))})
    # ── 3. 4h 변동성 HGB(개발기 ETH 모델 · QX4 실패 때 대체 길)
    art = joblib.load(SVM)
    export(SVM, fx / "svm_eth.json")
    vol = []
    for k, now in enumerate((D0 + 7 * 3_600_000 + 30_000, D0 + 2 * DAY + 61_000)):
        rows = ex.klines("ETHUSDT", "5m", 10_000 if k == 0 else 3000, now)
        a = dict(art) if k == 0 else {**art, "expand_logit": [-1.2, 4.9]}
        vol.append({"now": now, "rows": rows, "expand_logit": a.get("expand_logit"), "out": json.dumps(DS.vol4h_from_klines(rows, a, now))})
    short = DS.vol4h_from_klines(ex.klines("ETHUSDT", "5m", 600, D0), art, D0)
    # ── 4. 7일·4~13주 신호(코인베이스·바이낸스 현물·FRED)
    rng = np.random.default_rng(5)
    days = pd.date_range("2015-01-01", pd.Timestamp(D0 + 2 * DAY, unit="ms"))
    fred = {sid: [{"date": str(d.date()), "value": "." if (sid == "RRPONTSYD" and i % 37 == 0) else f"{v:.3f}"}
                  for i, (d, v) in enumerate(zip(days, base * np.exp(np.cumsum(rng.normal(0, 3e-3, len(days)))))) if (d.dayofweek < 5 if sid != "WALCL" else d.dayofweek == 2)]
            for sid, base in (("DTWEXBGS", 120.0), ("WALCL", 7e6), ("RRPONTSYD", 500.0), ("WTREGEN", 7e5))}
    btc = 60000 * np.exp(np.cumsum(rng.normal(0, 0.02, 2000)))
    sig = []
    for now_s, key in (((D0 + 3_600_000) / 1000, "TESTKEY"), ((D0 + DAY + 7_200_000) / 1000, None)):
        urls = []

        def get(url, timeout=30.0, urls=urls, now_s=now_s):
            r = get1(url, now_s)
            urls.append([url, r])                                                # 응답도 적는다(Rust 시험이 그대로 돌려줌)
            return r

        def get1(url, now_s):
            if "coinbase" in url:
                q = dict(x.split("=") for x in url.split("?")[1].split("&"))
                st = int(pd.Timestamp(q["start"]).timestamp())
                en = int(pd.Timestamp(q["end"]).timestamp())
                return [[t, 1, 2, 3, float(f"{btc[t // 86400 % 2000] * 1.0004:.2f}"), 9] for t in range(en // 86400 * 86400, st - 1, -86400)]   # 최신 먼저
            if "binance" in url:
                t1 = int(now_s // 86400 * 86400)
                return [[t * 1000, "1", "2", "3", f"{btc[t // 86400 % 2000]:.2f}", "5", t * 1000 + DAY - 1] for t in range(t1 - 999 * 86400, t1 + 1, 86400)]
            sid = url.split("series_id=")[1].split("&")[0]
            return {"observations": fred[sid]}
        DS._get = get
        out = DS.dir_signals_payload(fred_key=key, now=now_s)
        sig.append({"now": now_s, "key": key, "urls": urls, "out": json.dumps(out)})
        print("신호", now_s, len(urls), out["cb"].get("v"), out["z91"].get("v", out["z91"].get("reason")))
    (fx / "signals_fixture.json").write_text(json.dumps({"rec": rec, "open_rev": orv, "vol": vol, "vol_short": json.dumps(short), "sig": sig}))
    print((fx / "signals_fixture.json").stat().st_size)


if __name__ == "__main__":
    main()
