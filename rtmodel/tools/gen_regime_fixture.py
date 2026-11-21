"""«띠 안·이탈(regime band)» 정답지 (2026-11-21 · tests/regime_band.rs). 원본 함수를 그대로 부른다 -- 네트워크 없음.
  - 모델: 개발기 사본 tmp/session_xrp_regime_20261102/regime_acc/regime_band_h48_model.joblib(서버와 md5 같음 f62774c6) → export_hgb.model
  - 합성 4코인 5분봉(REST 행 꼴 · 빈 봉·같은 값 구간 포함) + 가짜 fetch 로 scripts/live_regime_band_cron_20261102.main 을 매시 9회 + 오래된 자료 1회
    (기록·채점·요약 파일 글자 · 요청 열 · 코인별 level_at·피쳐 27열·p_trend 비트)
  - 격자: 마지막 회차 패널의 여러 결정 행에서 level_at · frame_for · predict_proba (빈 레벨·NaN 피쳐 칸 포함)
    (tools/server_env.sh 의 환경) python rtmodel/tools/gen_regime_fixture.py      # → tests/fixtures/regime_band.json · regime_fixture.json
"""
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_regime_band_cron_20261102 as C  # noqa: E402
from export_hgb import b1, bits  # noqa: E402
from export_regime import export  # noqa: E402
import sklearn  # noqa: E402

S, R = C.S, C.R
S.MODEL = Path("/home/kbj20/crypto-scalping/tmp/session_xrp_regime_20261102/regime_acc/regime_band_h48_model.joblib")   # 원본 record 가 bundle 없이 부른다
MODEL = S.MODEL
BAR = 300_000
N = 3700
P0 = {"ETH": (3000.0, "{:.2f}"), "BTC": (90000.0, "{:.1f}"), "SOL": (150.0, "{:.3f}"), "XRP": (2.2, "{:.4f}")}
GAPS = {"SOL": [(N - 150, 3)], "XRP": [(1000, 2)], "BTC": [(2500, 4)]}   # (위치, 빠진 봉 수)


def series(t0):
    rng = np.random.default_rng(7)
    out = {}
    for coin, (p0, fmt) in P0.items():
        sig = np.where((np.arange(N) // 400) % 2 == 0, 0.0015, 0.004)              # 조용한·출렁이는 구간 번갈아
        drift = np.where((np.arange(N) // 700) % 3 == 1, 0.0004, 0.0)              # 추세 구간
        lp = np.log(p0) + np.cumsum(rng.normal(0, 1, N) * sig + drift)
        c = np.exp(lp)
        o = np.r_[c[0], c[:-1]]
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 1, N)) * sig * 0.6)
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 1, N)) * sig * 0.6)
        qv = np.exp(rng.normal(16, 0.6, N))
        if coin == "XRP":                                                            # 같은 값 구간(가격 고정 · 거래 0)
            s = N - 400
            o[s:s + 50] = c[s:s + 50] = hi[s:s + 50] = lo[s:s + 50] = c[s - 1]
            qv[s:s + 50] = 0.0
        tb = qv * rng.uniform(0.3, 0.7, N)
        drop = {g + j for g, k in GAPS.get(coin, []) for j in range(k)}
        rows = []
        for i in range(N):
            if i in drop:
                continue
            t = t0 + i * BAR
            f = lambda x: fmt.format(x)  # noqa: E731
            rows.append([t, f(o[i]), f(hi[i]), f(lo[i]), f(c[i]), f"{qv[i] / c[i]:.3f}", t + BAR - 1, f"{qv[i]:.4f}", int(rng.integers(100, 5000)),
                         f"{tb[i] / c[i]:.3f}", f"{tb[i]:.4f}", "0"])
        out[coin] = rows
    return out


def bins_at(df, t):
    """level_at 이 compute_spliced_levels 안에서 만드는 두 벌 칸(넣은 순서 · 무게 비트) -- 레벨 가격(4자리)에 안 드러나는 무게 순서를 대조한다.
    입력 창은 level_at 줄을 그대로 옮겼다(원본은 칸을 밖으로 안 낸다)."""
    LM = S.LM
    d = pd.DataFrame({"timestamp": df.index.tz_localize("UTC"), "open": df.o.values, "high": df.h.values, "low": df.l.values, "close": df.c.values,
                      "volume": (df.qv / df.c).values, "qv": df.qv.values})
    g = df.assign(v=df.qv / df.c).resample("1h")
    h1 = pd.DataFrame({"high": g.h.max(), "low": g.l.min(), "close": g.c.last(), "volume": g.v.sum(), "n": g.c.count()})
    h1 = h1[h1.n == 12].drop(columns="n").reset_index().rename(columns={"open_time": "timestamp", "index": "timestamp"})
    h1["timestamp"] = pd.to_datetime(h1["timestamp"]).dt.tz_localize("UTC")
    h, m = LM._bar_arrays(h1), LM._bar_arrays(d)
    k = int(np.searchsorted(m[0], t))
    win = LM._window_at(h, m, t, S.LOOKBACK_H)
    cp = float(d.close.iloc[k - 1])
    cm = LM._prepare_common(win, cp, LM._bar_arrays(d.iloc[max(0, k - 2200):k]))
    if cm is None:
        return None
    w = win.reset_index(drop=True)
    two = (LM._bins_from_common(cm, (w.high.to_numpy(float) + w.low.to_numpy(float)) / 2.0), LM._bins_from_common(cm, None))
    return [None if b is None else [[int(a), b1(v)] for a, v in b.items()] for b in two]


def fb(x):
    return {"b": b1(x)}


def lv(t):
    p, sup, res = t
    return {"p": b1(p), "sup": bits(sup), "res": bits(res)}


def main():
    if sklearn.__version__ != "1.8.0":
        raise SystemExit("server_env.sh 의 환경으로")
    fx = ROOT / "rtmodel/tests/fixtures"
    bundle = joblib.load(MODEL)
    export(MODEL, fx / "regime_band.json")
    t0 = 1_794_000_000_000 // 3_600_000 * 3_600_000
    ser = series(t0)
    t_end = t0 + (N - 1) * BAR
    hours = [(t_end // 3_600_000) * 3_600_000 - (8 - k) * 3_600_000 for k in range(9)]
    hours = hours[:4] + hours[7:]                                                  # 3시간 건너뜀(크론 멈춤) → 첫 채점이 4시간보다 늦게(창 끝 t+235분이 드러남)
    nows = [h + 2 * 60_000 + 7_000 for h in hours] + [t_end + 80 * 60_000]          # 마지막 = 자료가 80분 묵음 → 실패
    vis = {hours[-1] + 65 * 60_000 + 7_000: hours[-1] - 5 * 60_000}                 # 자료가 늦게 오는 실행: 결정 시각 65분 뒤(70분 안 · 이미 적은 시각 → 새 줄 0)
    nows.insert(len(hours), *vis)
    tmp = Path(tempfile.mkdtemp())
    log, scored, summ = tmp / "f.jsonl", tmp / "s.jsonl", tmp / "sum.json"
    runs = []
    for now in nows:
        reqs = []

        def fetch(symbol, end_ms=None, limit=1500, now=now, reqs=reqs):
            reqs.append([symbol, end_ms])
            rows = [r for r in ser[symbol[:-4]] if r[0] <= min(end_ms if end_ms else now, vis.get(now, now))]
            return [list(r) for r in rows[-limit:]]
        run = {"now": now, "vis": vis.get(now, now)}
        try:
            run["out"] = json.dumps(C.main(now_ms=now, fetch=fetch, log=log, scored=scored, summary=summ), ensure_ascii=False, default=str)
        except Exception as e:  # noqa: BLE001 -- 원본 __main__ 이 FAIL 로 바꾸는 자리
            run["fail"] = type(e).__name__
        run["requests"] = list(reqs)                                                # main 의 것만(아래 다시 뜨기 전)
        run["files"] = {k: p.read_text() if p.exists() else None for k, p in (("log", log), ("scored", scored), ("summary", summ))}
        if "out" in run:                                                             # 같은 패널로 코인별 속을 다시 뜬다
            panels = {c: C.panel(f"{c}USDT", now, fetch) for c in R.COINS}
            panels = {c: p.loc[: p.index[p.index.minute == 55][-1]] for c, p in panels.items()}
            run["coins"] = {}
            for c in R.COINS:
                df = panels[c]
                t = int(df.index[-1].timestamp()) + 300
                o = S.predict_latest(panels, c, bundle)
                run["coins"][c] = {"n": len(df), "level": lv(S.level_at(df, t)), "bins": bins_at(df, t), "x": bits([o["features"][f] for f in bundle["features"]]), "p": b1(o["p_trend"])}
            last_panels = panels
        runs.append(run)
        print(now, run.get("out", run.get("fail")), len(reqs))
    grid = {}
    for c in R.COINS:
        df = last_panels[c]
        b = last_panels["BTC"].reindex(df.index)
        bc = b.c
        btc = pd.DataFrame({"er48": (bc - bc.shift(48)).abs() / bc.diff().abs().rolling(48).sum(),
                            "rv": np.log(np.sqrt((np.log(bc).diff() ** 2).rolling(48).sum()) / np.sqrt((np.log(bc).diff() ** 2).rolling(288).sum()))})
        i = np.r_[np.arange(287, len(df), 9), len(df) - 1]                          # i < 287 은 원본 sw(..)[i-287] 이 음수 색인으로 끝을 감는다 -- 뺌
        ts = df.index.values.astype("datetime64[s]").astype("int64")
        snaps = {int(ts[k]) + 300: S.level_at(df, int(ts[k]) + 300) for k in i}
        X, R1, S1, sig, p = S.acc.frame_for(df, i, btc, snaps)
        X = X.replace([np.inf, -np.inf], np.nan)[bundle["features"]]
        pr = bundle["model"].predict_proba(X)[:, 1]
        grid[c] = {"i": i.tolist(), "level": [lv(snaps[int(ts[k]) + 300]) for k in i], "bins": {str(k): bins_at(df, int(ts[k]) + 300) for k in i[::10]}, "x": [bits(r) for r in X.to_numpy()], "p": bits(pr)}
        print(c, len(i), "빈 저항", sum(not s[2] for s in snaps.values()), "NaN 칸", int(X.isna().sum().sum()))
    (fx / "regime_fixture.json").write_text(json.dumps({"series": ser, "runs": runs, "grid": grid}, ensure_ascii=False))
    print((fx / "regime_fixture.json").stat().st_size)


if __name__ == "__main__":
    main()
