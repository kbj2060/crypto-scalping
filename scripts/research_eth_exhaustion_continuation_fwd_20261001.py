#!/usr/bin/env python3
"""ETH «소진형 대량 델타» 4시간 지속 — 전진 사전등록 (docs/experiments/eth_exhaustion_continuation_fwd_prereg_20261001.md).

정의·라벨은 research_eth_exhaustion_delta_4h_prereg_20260930 의 함수를 그대로 쓴다(c = −r).
판정 대상 = 봉 t ≥ 2026-10-01 00:00 UTC 이고 라벨 완성. 점검 = 누적 125/250/375/500 건, O'Brien-Fleming 단측 α 0.025.
🔴눈가림: --status 는 사건 수만. --look 은 누적 수가 다음 점검 문턱에 도달했을 때만 결과를 낸다(그 전엔 거부).

사용:  --status   (사건 수 · 다음 점검까지 남은 수)
       --look     (문턱 도달 시 그 점검 1회 실행 → ..._looks.json 에 추가)
       --selfcheck
데이터: data.binance.vision 일별 5분 klines 를 받아 binance_data/klines/ETHUSDT/vision5m_daily/ 에 쌓는다(없는 날만 받음).
"""
from __future__ import annotations
import argparse, datetime as dt, glob, json, sys, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_exhaustion_delta_4h_prereg_20260930 as R                 # noqa: E402  같은 정의·같은 코드
import research_exhaustion_delta_4h_continuation_prereg_20260930 as C        # noqa: E402  vision CSV 읽기

MAIN = Path("/home/kbj20/crypto-scalping")
CACHE = MAIN / "binance_data/klines/ETHUSDT/vision5m_daily"
LOOKS_JSON = ROOT / "docs/experiments/eth_exhaustion_continuation_fwd_prereg_20261001_looks.json"
START = pd.Timestamp("2026-10-01 00:00:00")
WARM = dt.date(2026, 9, 1)
LOOKS = [(125, 4.048), (250, 2.862), (375, 2.337), (500, 2.024)]   # (누적 사건 수, 확정 z 문턱)
FUTILITY_FROM = 2                                                  # 2번째 점검부터 z < 0 이면 중단(비구속)
URL = "https://data.binance.vision/data/futures/um/daily/klines/ETHUSDT/5m/ETHUSDT-5m-{d}.zip"


def fetch_days(until: dt.date) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    d = WARM
    while d <= until:
        csv = CACHE / f"ETHUSDT-5m-{d}.csv"
        if not csv.exists():
            z = CACHE / f"ETHUSDT-5m-{d}.zip"
            try:
                urllib.request.urlretrieve(URL.format(d=d), z)
                with zipfile.ZipFile(z) as f:
                    f.extractall(CACHE)
                z.unlink()
            except Exception:                    # 아직 공개 안 된 날(보통 어제 이후) -- 거기서 멈춘다
                break
        d += dt.timedelta(days=1)


def load() -> pd.DataFrame:
    fetch_days(dt.date.today() - dt.timedelta(days=1))
    fs = sorted(glob.glob(str(CACHE / "ETHUSDT-5m-*.csv")))
    if not fs:
        raise SystemExit("데이터 없음")
    return C.finish(pd.concat([C.read_vision(f) for f in fs]), str(WARM), "2100-01-01")


def events(d: pd.DataFrame):
    ev, ct = R.events_a(d); eb, cb = R.events_b(d)
    ka, kb, ca, cbb = R.dedup(ev), R.dedup(eb), R.dedup(ct), R.dedup(cb)
    la, lb = R.labels(d, ka), R.labels(d, kb)
    ok = bool(np.array_equal(ka, kb) and np.array_equal(ca, cbb) and (len(la) == 0 or np.allclose(la.r, lb.r, atol=1e-9, rtol=0)))
    sel = lambda x: x[x.t >= START].assign(r=lambda z: -z.r).reset_index(drop=True)   # r 칸 = c(지속)
    return sel(la), sel(R.labels(d, ca)), ok


def done_looks() -> list:
    return json.loads(LOOKS_JSON.read_text()) if LOOKS_JSON.exists() else []


def status(E: pd.DataFrame, gaps: int) -> dict:
    k = len(done_looks())
    nxt = LOOKS[k][0] if k < len(LOOKS) else None
    return {"events": len(E), "days": int(E.day.nunique()) if len(E) else 0, "empty_bars": gaps,
            "looks_done": k, "next_look_at": nxt, "remaining": None if nxt is None else max(0, nxt - len(E)),
            "last_event_utc": str(E.t.max()) if len(E) else None}


def look(E: pd.DataFrame, Cn: pd.DataFrame, k: int) -> dict:
    n, zb = LOOKS[k]
    E = E.sort_values("t").head(n)                       # 점검 k 는 앞에서부터 정확히 n 건
    Cn = Cn[Cn.t <= E.t.max()]
    days = E.day.unique(); g = {x: v.r.to_numpy() for x, v in E.groupby("day")}
    rng = np.random.default_rng(20261001 + k)
    boots = [np.concatenate([g[p] for p in rng.choice(days, len(days), replace=True)]).mean() for _ in range(5000)]
    m, se = float(E.r.mean()), float(np.std(boots, ddof=1))
    z = m / se if se > 0 else float("nan")
    aux1 = float(E.r.mean() - Cn.r.mean()) if len(Cn) else float("nan")
    buy, sell = float(E[E.side == "buy"].r.mean()), float(E[E.side == "sell"].r.mean())
    confirm = bool(z >= zb and aux1 > 0 and buy > 0 and sell > 0)
    futile = bool(k + 1 >= FUTILITY_FROM and z < 0)
    c = E.r.to_numpy()
    return {"look": k + 1, "n": n, "through_utc": str(E.t.max()), "mean_c": m, "se": se, "z": z, "z_boundary": zb,
            "aux1_minus_control": aux1, "n_control": len(Cn), "buy": buy, "sell": sell,
            "decision": "확정" if confirm else ("무익 중단" if futile else ("기각(최종)" if k == len(LOOKS) - 1 else "계속")),
            "report_only": {"P_cont": float((c > 0).mean()), "median": float(np.median(c)),
                            "q90_cont": float(np.quantile(c, 0.9)), "q10_rev": float(-np.quantile(c, 0.1))},
            "run_at": dt.datetime.now(dt.timezone.utc).isoformat()}


def _selfcheck() -> None:
    R._selfcheck()
    rng = np.random.default_rng(1)
    t = pd.date_range("2026-10-01", periods=600, freq="4h")
    E = pd.DataFrame({"t": t, "day": t.floor("D"), "side": np.where(np.arange(600) % 2, "buy", "sell"), "r": rng.normal(40, 100, 600)})
    Cn = E.assign(r=rng.normal(0, 100, 600))
    r1 = look(E, Cn, 0)
    assert r1["n"] == 125 and r1["through_utc"] == str(E.t.iloc[124]), r1
    assert r1["decision"] in ("확정", "계속") and r1["z"] > 0
    Eneg = E.assign(r=rng.normal(-20, 100, 600))
    assert look(Eneg, Cn, 1)["decision"] == "무익 중단"
    assert look(Eneg, Cn, 0)["decision"] == "계속"            # 1번째 점검은 무익 중단 없음
    s = status(E.head(40), 0); assert s["remaining"] == 85 and "mean_c" not in s   # 눈가림: 결과 없음
    print("fwd selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true"); g.add_argument("--look", action="store_true"); g.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); sys.exit(0)
    d = load(); E, Cn, ok = events(d)
    if not ok:
        raise SystemExit("🔴 독립 재구성 불일치 -- 결과를 내지 않는다")
    gaps = int(d.close.isna().sum())
    st = status(E, gaps)
    if a.status:
        print(json.dumps(st, ensure_ascii=False, indent=1)); sys.exit(0)
    k = st["looks_done"]
    if k >= len(LOOKS):
        raise SystemExit("모든 점검이 끝났다")
    if st["remaining"] > 0:
        raise SystemExit(f"점검 {k + 1} 문턱({LOOKS[k][0]}건)에 {st['remaining']}건 모자라다 -- 결과를 보지 않는다")
    r = look(E, Cn, k); r["empty_bars"] = gaps
    LOOKS_JSON.write_text(json.dumps(done_looks() + [r], ensure_ascii=False, indent=1))
    print(json.dumps(r, ensure_ascii=False, indent=1))
