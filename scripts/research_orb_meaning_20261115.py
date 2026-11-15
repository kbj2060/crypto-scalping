#!/usr/bin/env python3
"""시가 범위 돌파(ORB) 뒤 이어짐 의미 검증 (2026-11-15). 사전등록 docs/prereg/orb_meaning_20261115.md (9fa6e420).

  selftest   손계산 봉으로 범위·첫 종가 돌파·경주 · 뉴욕 시가 서머타임 단언
  run        4코인 5분봉 실제 + 기하 위약 20벌 + 세션 아닌 정시 범위 대조 → tmp/orb_meaning_20261115/
경주·위약·판정은 research_smc_ict_meaning_20261115 그대로.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/orb_meaning_20261115"
LS = {"or30": 30, "or60": 60}
WIN, BAR = 72, 300_000
P3H = [h for h in range(24) if h not in (23, 0, 13, 14)]


def starts(t0: int, t1: int) -> dict[str, np.ndarray]:
    days = pd.date_range(pd.Timestamp(t0, unit="ms").normalize(), pd.Timestamp(t1, unit="ms"), freq="D")
    ny = (days + pd.Timedelta("9h30min")).tz_localize("America/New_York", nonexistent="shift_forward", ambiguous=True).tz_convert("UTC")
    ms = lambda x: np.asarray(x.asi8 // 10**6, dtype=np.int64)
    return {"UTC": ms(days.tz_localize("UTC")), "NY": ms(ny[ny.dayofweek < 5]),
            "P3": np.sort(np.concatenate([ms(days.tz_localize("UTC") + pd.Timedelta(hours=h)) for h in P3H]))}


def orb(B: dict, s0: int, L: int, i0: int):
    """(b, d, W) 또는 None. i0 = 시가 봉 인덱스."""
    h, l, c, t = B["h"], B["l"], B["c"], B["t"]; m = L // 5; n = len(c)
    if i0 < 0 or i0 + m >= n - 1 or t[i0 + m - 1] != s0 + (m - 1) * BAR:
        return None
    hi, lo = h[i0:i0 + m].max(), l[i0:i0 + m].min(); W = hi - lo
    if not W > 0:
        return None
    seg = c[i0 + m:min(i0 + m + WIN, n - 1)]
    k = np.flatnonzero((seg > hi) | (seg < lo))
    if not len(k):
        return None
    b = i0 + m + int(k[0])
    return b, (1 if c[b] > hi else -1), W


def events(B: dict, with_p3: bool) -> list[dict]:
    h, l, c, t = B["h"], B["l"], B["c"], B["t"]
    buf = np.zeros(len(c)); R = []
    S = starts(int(t[0]), int(t[-1]))
    for kind, ss in S.items():
        if kind == "P3" and not with_p3:
            continue
        ix = np.searchsorted(t, ss); ix[(ix >= len(t))] = -1
        ix = np.where((ix >= 0) & (t[np.clip(ix, 0, len(t) - 1)] == ss), ix, -1)
        for s0, i0 in zip(ss, ix):
            for tf, L in LS.items():
                o = orb(B, int(s0), L, int(i0))
                if o is None:
                    continue
                b, d, W = o
                for suf, k in (("", 1.0), ("_2w", 2.0)):
                    buf[b] = k * W
                    y = SM.first_hit_race(h, l, c, buf, b, d, WIN)
                    if y >= 0:
                        R.append(dict(cell=kind + suf, tf=tf, tk=int(t[b]) + BAR, y=float(y), w=float(W / c[b])))
                buf[b] = 0.0
    return R


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, aggs = [], []
    for ci, coin in enumerate(SM.COINS):
        k = SM.load_1m(f"{coin}USDT")
        for draw in range(-1, SM.NPLACEBO):
            kk = k if draw < 0 else SM.placebo(k, SM.SEED + 1000 * ci + draw)
            ev = pd.DataFrame(events(SM.bars(kk, 5), draw < 0)).assign(coin=coin, draw=draw)
            if draw < 0:
                rows.append(ev)
            elif len(ev):
                aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); PA.to_parquet(OUT / "placebo_agg.parquet")
    SM.TFS, SM.MAIN_TFS, SM.CELLS = {tf: 5 for tf in LS}, tuple(LS), ["UTC", "NY"]   # 주 판정 4칸 Holm (블록 = 1일)
    SM.P2OF = {"UTC": "P3", "NY": "P3"}                                           # judge 의 «P2» 자리 = 세션 아닌 정시 범위
    main = SM.judge(E, PA)
    SM.MAIN_TFS, SM.CELLS, SM.P2OF = (), ["UTC_2w", "NY_2w"], {"UTC_2w": "P3_2w", "NY_2w": "P3_2w"}
    aux = SM.judge(E, PA)
    terc = {}
    for tf in LS:                                                                 # 보조: W(가격 대비) 3분위별 이어짐
        r = E[(E.tf == tf) & (E.cell != "P3") & ~E.cell.str.endswith("_2w")]
        q = pd.qcut(r.w, 3, labels=["작음", "중간", "큼"])
        terc[tf] = {str(a): round(float(v), 4) for a, v in r.groupby(q, observed=True).y.mean().items()}
    json.dump(dict(main=main, aux=aux, w_tercile=terc), open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for k_, v in {**main, **aux}.items():
        print(k_, {x: v.get(x) for x in ("n", "real", "p1", "p2", "p2_n", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate", "reversed")})
    print("w_tercile", terc)


def selftest() -> None:
    S = starts(int(pd.Timestamp("2026-01-15").value // 10**6), int(pd.Timestamp("2026-07-16").value // 10**6))
    ny = pd.to_datetime(S["NY"], unit="ms", utc=True)
    assert pd.Timestamp("2026-01-15 14:30", tz="UTC") in ny and pd.Timestamp("2026-07-15 13:30", tz="UTC") in ny   # 겨울 14:30 · 여름 13:30
    assert pd.Timestamp("2026-01-17 14:30", tz="UTC") not in ny                                                     # 토요일
    assert all(pd.to_datetime(S["P3"], unit="ms", utc=True).hour.isin(P3H))
    n = 40; t = np.arange(n, dtype=np.int64) * BAR
    c = np.full(n, 100.0); h = c + 0.3; l = c - 0.3; h[2] = 101; l[3] = 99      # 첫 6봉 범위 99~101 (W 2)
    c[8] = 101.5; h[8] = 101.6                                                  # 봉 8 종가 위로 돌파
    h[12] = 103.6                                                               # 101.5 + 2 = 103.5 먼저
    B = dict(h=h, l=l, c=c, t=t)
    assert orb(B, 0, 30, 0) == (8, 1, 2.0)
    buf = np.zeros(n); buf[8] = 2.0
    assert SM.first_hit_race(h, l, c, buf, 8, 1, WIN) == 1
    l2 = l.copy(); l2[10] = 99.4                                                # 반대 W(99.5) 먼저
    assert SM.first_hit_race(h, l2, c, buf, 8, 1, WIN) == 0
    t2 = t.copy(); t2[3:] += BAR                                                # 범위 안 봉 빠짐 = 제외
    assert orb(dict(B, t=t2), 0, 30, 0) is None
    print("selftest ok")


def diag() -> None:
    """채점 뒤 진단(판정 안 씀): 미결을 빼고 «결판 난 경주»만의 이어짐 비율 · 결판 비율. 실제 가격만."""
    out = {}
    for coin in SM.COINS:
        B = SM.bars(SM.load_1m(f"{coin}USDT"), 5); h, l, c, t = B["h"], B["l"], B["c"], B["t"]
        S = starts(int(t[0]), int(t[-1]))
        for kind, ss in S.items():
            ix = np.searchsorted(t, ss); ok = (ix < len(t)) & (t[np.clip(ix, 0, len(t) - 1)] == ss)
            for s0, i0 in zip(ss[ok], ix[ok]):
                for tf, L in LS.items():
                    o = orb(B, int(s0), L, int(i0))
                    if o is None:
                        continue
                    b, d, W = o; ref = c[b]; res = 0
                    for j in range(b + 1, min(b + WIN, len(c) - 1) + 1):
                        bad = (l[j] <= ref - W) if d == 1 else (h[j] >= ref + W)
                        good = (h[j] >= ref + W) if d == 1 else (l[j] <= ref - W)
                        if bad: res = -1; break
                        if good: res = 1; break
                    a = out.setdefault(f"{tf}|{kind}", [0, 0, 0]); a[0] += res == 1; a[1] += res == -1; a[2] += 1
        print(coin, flush=True)
    r = {k: dict(n=v[2], decided=round((v[0] + v[1]) / v[2], 4), win_given_decided=round(v[0] / max(1, v[0] + v[1]), 4)) for k, v in out.items()}
    json.dump(r, open(OUT / "diag_decided.json", "w"), ensure_ascii=False, indent=1); print(r)


# ── 2라운드: 뉴욕 개장 · 결판 난 경주만 · 새 코인 5종 재현 (docs/prereg/orb_r2_ny_open_20261115.md, 79f20ae6) ──
NEW = ["ADA", "AVAX", "BNB", "DOGE", "LINK"]
OUT2 = OUT.parent / "orb_r2_ny_open_20261115"


def race3(h, l, c, b, d, W) -> int:
    """돌파 쪽 W 먼저 1 · 반대 먼저(같은 봉 동시 포함) 0 · 72봉 안 미결 −1."""
    ref = c[b]
    for j in range(b + 1, min(b + WIN, len(c) - 1) + 1):
        good = h[j] >= ref + W if d == 1 else l[j] <= ref - W
        bad = l[j] <= ref - W if d == 1 else h[j] >= ref + W
        if bad:
            return 0
        if good:
            return 1
    return -1


def load5(coin: str) -> pd.DataFrame:
    root = Path("/home/kbj20/crypto-scalping/data")
    d = pd.concat([pd.read_csv(root / f"{coin.lower()}_5m_{p}.csv", usecols=["ts", "high", "low", "close"]) for p in ("2022_2023", "2024_2026")])
    d = d.rename(columns={"ts": "t", "high": "h", "low": "l", "close": "c"}).astype({"t": "int64"})
    return d.drop_duplicates("t").sort_values("t").reset_index(drop=True)


def bars5(k: pd.DataFrame) -> dict:
    return dict(t=k.t.to_numpy(), h=k.h.to_numpy(float), l=k.l.to_numpy(float), c=k.c.to_numpy(float))


def events2(B: dict, with_p3: bool) -> list[dict]:
    h, l, c, t = B["h"], B["l"], B["c"], B["t"]; R = []
    S = starts(int(t[0]), int(t[-1]))
    wk = lambda ss: ss[pd.to_datetime(ss, unit="ms").dayofweek < 5]
    hr = lambda ss: pd.to_datetime(ss, unit="ms").hour
    sets = {"NY": S["NY"], "UTC": S["UTC"]}
    if with_p3:
        p3 = wk(S["P3"]); sets["P3"] = p3; sets["P3b"] = p3[(hr(p3) >= 15) & (hr(p3) <= 20)]
    for kind, ss in sets.items():
        ix = np.searchsorted(t, ss); ok = (ix < len(t)) & (t[np.clip(ix, 0, len(t) - 1)] == ss)
        for s0, i0 in zip(ss[ok], ix[ok]):
            for tf, L in LS.items():
                o = orb(B, int(s0), L, int(i0))
                if o is not None:
                    b, d, W = o
                    R.append(dict(cell=kind, tf=tf, tk=int(t[b]) + BAR, y=float(race3(h, l, c, b, d, W))))
    return R


def run2() -> None:
    OUT2.mkdir(parents=True, exist_ok=True)
    SM.TFS = {tf: 5 for tf in LS}; res = {}
    for grp, coins in (("new", NEW), ("orig", list(SM.COINS))):
        rows, aggs = [], []
        for ci, coin in enumerate(coins):
            k = load5(coin) if grp == "new" else SM.load_1m(f"{coin}USDT")
            mk = bars5 if grp == "new" else (lambda kk: SM.bars(kk, 5))
            for draw in range(-1, SM.NPLACEBO):
                kk = k if draw < 0 else SM.placebo(k, SM.SEED + 7000 + 1000 * ci + draw)
                ev = pd.DataFrame(events2(mk(kk), draw < 0)).assign(coin=coin, draw=draw)
                if draw < 0:
                    rows.append(ev)
                else:
                    ev = ev[ev.y >= 0]
                    aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
                print(grp, coin, draw, flush=True)
        E = pd.concat(rows, ignore_index=True); PA = pd.concat(aggs, ignore_index=True)
        E.to_parquet(OUT2 / f"events_{grp}.parquet")
        dec = {f"{tf}|{c_}": round(float((E[(E.tf == tf) & (E.cell == c_)].y >= 0).mean()), 4) for tf in LS for c_ in ("NY", "UTC", "P3", "P3b")}
        Ed = E[E.y >= 0]
        SM.COINS = coins
        SM.MAIN_TFS, SM.CELLS, SM.P2OF = tuple(LS), ["NY"], {"NY": "P3"}
        main = SM.judge(Ed, PA)
        for v in main.values():                                # 코인 ≥ 4/5 (새 코인) — judge 는 3 고정이라 다시 셈
            need = 4 if grp == "new" else 3
            ok = sum(x > 0 for x in v["coin_diff_pp"].values()) >= need and all(x > 0 for x in v["half_diff_pp"].values())
            v["pass"] = bool(v["holm"] and v["lo90_pp"] > 0 and v.get("lo90_2_pp", -1) > 0 and ok)
            v["candidate"] = bool(v["diff_pp"] > 0 and v.get("diff2_pp", -1) > 0 and ok)
        SM.MAIN_TFS, SM.P2OF = (), {"NY": "P3b"}
        afternoon = SM.judge(Ed, PA)
        SM.CELLS, SM.P2OF = ["UTC"], {"UTC": "P3"}
        utc = SM.judge(Ed, PA)
        res[grp] = dict(main=main, ny_vs_us_afternoon=afternoon, utc=utc, decided_share=dec)
    SM.COINS = ["ETH", "BTC", "SOL", "XRP"]
    json.dump(res, open(OUT2 / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for g, r in res.items():
        for part in ("main", "ny_vs_us_afternoon", "utc"):
            for k_, v in r[part].items():
                print(g, part, k_, {x: v.get(x) for x in ("n", "real", "p1", "p2", "p2_n", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate")})
        print(g, "decided", r["decided_share"])


def selftest2() -> None:
    h = np.array([10, 10.5, 12.1, 10.]); l = np.array([9, 9.5, 9.6, 9.]); c = np.array([10, 10, 10, 10.])
    assert race3(h, l, c, 0, 1, 2.0) == 1 and race3(h, l, c, 0, -1, 2.0) == 0          # 위 12 먼저 · 숏이면 위 = 반대
    assert race3(h, l, c, 0, 1, 5.0) == -1                                              # 미결 = 제외
    k = load5("DOGE"); B = bars5(k)
    assert np.all(np.diff(B["t"]) > 0) and B["t"][0] == 1640995200000 and len(B["c"]) > 400_000
    print("selftest2 ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run, "diag": diag, "run2": run2, "selftest2": selftest2}[sys.argv[1]]()
