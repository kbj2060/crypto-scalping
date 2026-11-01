#!/usr/bin/env python3
"""XRP 저거래량 지지/저항 돌파 → 반대 포지션(되돌림) 검정 (2026-11-02, 사용자 재량 매매 규칙의 기계화).

사전등록: docs/prereg/xrp_lowvol_sweep_fade_v2_20261102.md (v1 = research_xrp_lowvol_sweep_fade_20261102.py, SL 만 다름) (이 스크립트와 함께 계산 전 커밋). 파라미터는 전부 고정 -- 적합 없음.
  python scripts/research_xrp_lowvol_sweep_fade_20261102.py --selfcheck | (인자 없음 = 건수만) | --run
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_liquidation_map_20260824 as LM  # noqa: E402

PANEL = Path("/home/kbj20/crypto-scalping/tmp/meta_label_race_20261101/XRPUSDT_5m.parquet")
OUT = ROOT / "tmp" / "xrp_lowvol_sweep_fade_v2_20261102"
LEVELS = ROOT / "tmp" / "xrp_lowvol_sweep_fade_20261102" / "levels.pkl"   # v1 과 같은 시간별 청산맵 스냅샷(재계산 생략)
LOOKBACK_H, VOL_Q, VOL_N, ER_N, ATR_N, SL_ATR, MAX_HOLD = 24, 0.30, 6, 288, 48, 0.5, 288
QWIN = 2016                                  # 7일 -- 거래량 분위·횡보 중앙값의 후행 창
COST_IN, COST_TP, COST_SL = 4.0, 2.0, 4.0    # bp 한 방향: 진입 테이커 · 익절 메이커 · 손절/시간청산 테이커 (엄격)
WINDOWS = {"전체 24-01~": ("2024-01-08", "2100-01-01"), "학습 24-01~25-12": ("2024-01-08", "2026-01-01"), "검증 26-01~05": ("2026-01-01", "2026-06-01"),
           "시험 26-06~": ("2026-06-01", "2100-01-01")}
NBOOT, NPLA, SEED = 5000, 1000, 20261102


def load():
    p = pd.read_parquet(PANEL, columns=["o", "h", "l", "c", "qv"])
    d = pd.DataFrame({"timestamp": p.index.tz_localize("UTC"), "open": p.o.values, "high": p.h.values, "low": p.l.values,
                      "close": p.c.values, "volume": (p.qv / p.c).values, "qv": p.qv.values})
    g = p.assign(v=p.qv / p.c).resample("1h")
    h1 = pd.DataFrame({"high": g.h.max(), "low": g.l.min(), "close": g.c.last(), "volume": g.v.sum(), "n": g.c.count()})
    h1 = h1[h1.n == 12].drop(columns="n").reset_index().rename(columns={"open_time": "timestamp", "index": "timestamp"})
    h1["timestamp"] = pd.to_datetime(h1["timestamp"]).dt.tz_localize("UTC")
    return d, h1


def snapshots(d, h1):
    """시 정각 t 마다 «t 에 알 수 있던» 청산맵 레벨(대시보드와 같은 24h 창·compute_spliced_levels). 닫힌 5분봉만(open < t)."""
    f = LEVELS
    if f.exists():
        return pickle.load(open(f, "rb"))
    h, m = LM._bar_arrays(h1), LM._bar_arrays(d)
    out = {}
    for t in range(int(m[0][0]) // 3600 * 3600 + 8 * 86400, int(m[0][-1]) + 300, 3600):
        k = int(np.searchsorted(m[0], t))
        win = LM._window_at(h, m, t, LOOKBACK_H)
        r = LM.compute_spliced_levels(win, float(d.close.iloc[k - 1]), d.iloc[max(0, k - 2200):k])
        out[t] = (float(d.close.iloc[k - 1]), [x["price"] for x in r["support_levels"]], [x["price"] for x in r["resistance_levels"]])
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return out


def sim(hi, lo, cl, i0, side, tp, sl, entry):
    """i0 봉(진입 봉)부터 TP/SL/시간. 같은 봉에 둘 다 닿으면 SL 먼저. 반환 (청산가, 종류, 청산 봉 오프셋)."""
    n = min(MAX_HOLD, len(hi) - i0)
    h, l = hi[i0:i0 + n], lo[i0:i0 + n]
    hit_tp = np.flatnonzero(l <= tp if side < 0 else h >= tp)
    hit_sl = np.flatnonzero(h >= sl if side < 0 else l <= sl)
    a, b = (hit_tp[0] if len(hit_tp) else 1 << 30), (hit_sl[0] if len(hit_sl) else 1 << 30)
    if a == b == 1 << 30:
        return cl[i0 + n - 1], "time", n - 1
    return (sl, "sl", b) if b <= a else (tp, "tp", a)


def net_bp(side, entry, exit_px, kind, cost=True):
    g = side * (exit_px / entry - 1) * 1e4
    return g - (COST_IN + {"tp": COST_TP}.get(kind, COST_SL) if cost else 0.0)


def events(d, snaps, mode="main"):
    """mode: main(저거래량·횡보) | novol(횡보만) | noreg(저거래량만) | trend(main + 사용자 추세 합치 방향만)."""
    c, hi, lo, qv = (d[x].to_numpy() for x in ("close", "high", "low", "qv"))
    ts = d.timestamp.astype("int64").to_numpy() // 10**9
    s = pd.Series(c)
    r6 = pd.Series(qv).rolling(VOL_N).sum()
    low_vol = (r6 <= r6.shift(VOL_N).rolling(QWIN).quantile(VOL_Q)).to_numpy()
    er = ((s - s.shift(ER_N)).abs() / s.diff().abs().rolling(ER_N).sum())
    ranging = (er <= er.shift(1).rolling(QWIN).median()).to_numpy()
    tr = np.maximum(hi - lo, np.maximum(abs(hi - np.roll(c, 1)), abs(lo - np.roll(c, 1))))
    atr = pd.Series(tr).rolling(ATR_N).mean().to_numpy()
    e5, e20 = s.ewm(span=5, adjust=False).mean().to_numpy(), s.ewm(span=20, adjust=False).mean().to_numpy()
    vwma = (s * pd.Series(qv / c)).rolling(100).sum().to_numpy() / pd.Series(qv / c).rolling(100).sum().to_numpy()
    rows, busy = [], -1
    for i in range(QWIN + VOL_N + 10, len(c) - 2):
        if i <= busy:
            continue
        T = ts[i] + 300                                   # 봉 i 마감 시각 = 결정 시각
        sn = snaps.get(T // 3600 * 3600)
        if sn is None or (mode != "noreg" and not ranging[i]) or (mode != "novol" and not low_vol[i]):
            continue
        _, sup, res = sn
        up = [x for x in res if c[i - 1] <= x < c[i]]
        dn = [x for x in sup if c[i] < x <= c[i - 1]]
        if not (up or dn):
            continue
        side, lvl = (-1, max(up)) if up else (1, min(dn))   # 위 돌파 → 숏 · 아래 돌파 → 롱
        q = i - VOL_N                                     # 추세는 «밀기 시작 직전» 상태로 읽는다(밀기 자체가 EMA 를 뒤집으므로)
        if mode == "trend" and not ((side < 0 and c[q] < vwma[q] and e5[q] < e20[q]) or (side > 0 and c[q] > vwma[q] and e5[q] > e20[q])):
            continue
        entry = float(d.open.iloc[i + 1])
        tps = [x for x in (sup if side < 0 else res) if (x < entry if side < 0 else x > entry)]
        ext = hi[i - VOL_N + 1:i + 1].max() if side < 0 else lo[i - VOL_N + 1:i + 1].min()   # 밀기 구간 꼬리 끝
        sl = ext - side * SL_ATR * atr[i]
        if not tps or not (entry < sl if side < 0 else entry > sl):
            continue
        tp = tps[0]                                       # 이익 방향 가장 가까운 청산맵 레벨
        px, kind, off = sim(hi, lo, c, i + 1, side, tp, sl, entry)
        busy = i + 1 + off
        rows.append(dict(i=i, t=pd.Timestamp(T, unit="s"), side=side, entry=entry, tp_bp=abs(tp / entry - 1) * 1e4,
                         sl_bp=abs(sl / entry - 1) * 1e4, kind=kind, gross=net_bp(side, entry, px, kind, False), net=net_bp(side, entry, px, kind)))
    return pd.DataFrame(rows), ((ranging if mode == "novol" else low_vol & ranging) if mode != "noreg" else low_vol)


def boot_ci(df, col, rng):
    g = df.groupby(df.t.dt.date)[col].agg(["sum", "count"]).to_numpy()
    idx = rng.integers(0, len(g), (NBOOT, len(g)))
    m = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return np.percentile(m, [1.25, 98.75])   # 같은 데이터 2번째 시행 -- 97.5% 양측(Bonferroni 2)


def placebo(d, ev, pool, rng):
    """같은 조건(저거래량·횡보)이지만 돌파 없는 봉에서, 이벤트와 같은 방향·같은 TP/SL 거리(bp)로 진입한 평균 net 분포."""
    hi, lo, cl = d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy()
    op = d.open.to_numpy()
    out = np.empty(NPLA)
    for b in range(NPLA):
        pick = rng.choice(pool, len(ev))
        tot = 0.0
        for k, i in enumerate(pick):
            e, sd = op[i + 1], ev.side.iloc[k]
            tp, sl = e * (1 + sd * ev.tp_bp.iloc[k] / 1e4), e * (1 - sd * ev.sl_bp.iloc[k] / 1e4)
            px, kind, _ = sim(hi, lo, cl, i + 1, sd, tp, sl, e)
            tot += net_bp(sd, e, px, kind)
        out[b] = tot / len(ev)
    return out


def selfcheck():
    hi, lo, cl = np.array([1.0, 1.02, 1.0]), np.array([1.0, 0.97, 1.0]), np.array([1.0, 1.0, 1.0])
    assert sim(hi, lo, cl, 0, -1, 0.97, 1.02, 1.0)[:2] == (1.02, "sl")        # 같은 봉 양쪽 → SL 먼저
    assert sim(hi, lo, cl, 2, -1, 0.5, 2.0, 1.0)[1] == "time"
    assert abs(net_bp(-1, 1.0, 0.99, "tp") - (100 - 6)) < 1e-9
    d, h1 = load()
    snaps = snapshots(d, h1)
    ts = d.timestamp.astype("int64").to_numpy() // 10**9
    t = sorted(snaps)[500]
    k = int(np.searchsorted(ts, t))
    assert ts[k - 1] + 300 <= t and snaps[t][0] == d.close.iloc[k - 1]      # 스냅샷 가격 = t 이전 마지막 닫힌 봉 종가
    print("selfcheck ok", len(snaps), "snapshots")


def main():
    if "--selfcheck" in sys.argv:
        return selfcheck()
    d, h1 = load()
    snaps = snapshots(d, h1)
    res, rng = {}, np.random.default_rng(SEED)
    for mode in ("main", "novol", "noreg", "trend"):
        ev, cond = events(d, snaps, mode)
        for w, (a, b) in WINDOWS.items():
            e = ev[(ev.t >= a) & (ev.t < b)]
            r = dict(n=len(e), n_long=int((e.side > 0).sum()), n_short=int((e.side < 0).sum()))
            if len(e) and "--run" in sys.argv:       # 건수만 보는 단계에서는 성과를 계산해도 출력하지 않는다
                r.update(tp=int((e.kind == "tp").sum()), sl=int((e.kind == "sl").sum()), time=int((e.kind == "time").sum()),
                         mean_net=e.net.mean(), mean_gross=e.gross.mean(), win_net=float((e.net > 0).mean()),
                         mean_tp_bp=e.tp_bp.mean(), mean_sl_bp=e.sl_bp.mean())
            if "--run" in sys.argv and mode in ("main", "novol") and len(e) >= 30:
                r["ci_net"], r["ci_gross"] = boot_ci(e, "net", rng).tolist(), boot_ci(e, "gross", rng).tolist()
                sel = (d.timestamp >= pd.Timestamp(a, tz="UTC")) & (d.timestamp < pd.Timestamp(b, tz="UTC"))
                lv = np.flatnonzero(sel.to_numpy() & cond)           # 같은 조건(저거래량·횡보) 봉 -- 돌파 여부는 불문
                lv = lv[(lv > QWIN + VOL_N + 10) & (lv < len(d) - MAX_HOLD - 2)]
                pla = placebo(d, e, lv, rng)
                r["placebo_pct_net"] = float((pla < r["mean_net"]).mean() * 100)
                r["placebo_mean"] = float(pla.mean())
            res[f"{mode}|{w}"] = r
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / ("result.json" if "--run" in sys.argv else "count.json"), "w"), indent=1, default=float, ensure_ascii=False)
    for k, v in res.items():
        print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
