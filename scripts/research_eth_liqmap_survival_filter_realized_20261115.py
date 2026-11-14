"""청산맵 생존 필터 켬 vs 끔 -- 실측 청산 위치 일치 (2026-11-15, 사전등록 docs/prereg/liqmap_survival_filter_realized_20261115.md).

변형 A(켬·감쇠) B(끔·감쇠) C(켬·감쇠 없음) D(끔·감쇠 없음), W = 180일. 주 판정 = H 24h 짝 IC 차 B−A · D−C (일 블록 90% 양측).
데이터(서버에서 복사, 읽기 전용): tmp/liqmap_survival_20261115/{liq_tab_k1h_eth.parquet, bn_liq, bn_mark, okx_liq, bybit_liq.csv}.
실행: python scripts/research_eth_liqmap_survival_filter_realized_20261115.py [--selftest]
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_liqmap_venue_leverage_validation_20261008 as V  # noqa: E402

D = Path("/home/kbj20/crypto-scalping/tmp/liqmap_survival_20261115")
W, BIN, SPAN, MM = 180 * 24, 0.002, 250, V.LM.MAINTENANCE_MARGIN_RATE
TIERS = (10, 20, 25, 50, 75, 100)
VARS = {"A": (True, 240.0), "B": (False, 240.0), "C": (True, None), "D": (False, None)}   # (생존 필터, 감쇠 e-폴드 h)
O0, O1 = pd.Timestamp("2026-09-22", tz="UTC"), pd.Timestamp("2026-11-13", tz="UTC")       # 기준 시각 [O0, O1 − H]


def bars() -> pd.DataFrame:
    k = pd.read_parquet(D / "liq_tab_k1h_eth.parquet").sort_values("t").reset_index(drop=True)
    k["volume"] = k.qv / k.close
    k["timestamp"] = pd.to_datetime(k.t, unit="ms", utc=True)
    assert (k.t.diff().dropna() == 3_600_000).all(), "1시간봉 구멍"
    return k


def liqs() -> pd.DataFrame:
    rd = lambda n: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(D / n / "*/part.parquet")))], ignore_index=True)
    bn = rd("bn_liq")[["ts_ms", "side", "qty"]].assign(venue="binance")
    ok = rd("okx_liq"); ok = ok[ok.inst_id == "ETH-USDT-SWAP"]
    ok = ok.assign(qty=ok.sz_base.fillna(ok.sz_contracts * 0.1), side=ok.pos_side)[["ts_ms", "side", "qty"]].assign(venue="okx")   # ETH-USDT-SWAP 1계약 = 0.1 ETH
    by = pd.read_csv(D / "bybit_liq.csv"); by = by.assign(side=by.side.map({"Buy": "long", "Sell": "short"}))[["ts_ms", "side", "qty"]].assign(venue="bybit")
    d = pd.concat([bn, ok, by], ignore_index=True).dropna()
    d["ts_ms"] = d.ts_ms.astype("int64")
    m = rd("bn_mark")[["ts_ms", "mark"]].astype({"ts_ms": "int64"}).sort_values("ts_ms")
    d = pd.merge_asof(d.sort_values("ts_ms"), m, on="ts_ms", direction="backward", tolerance=5000).dropna(subset=["mark"])
    d["usd"] = d.qty * d.mark
    return d


def maps_at(k: pd.DataFrame, i: int, w: int = W) -> tuple[float, dict]:
    """봉 [i−w, i) → cp, {변형: {'long'|'short': 길이 2·SPAN+1 배열(칸 −SPAN..SPAN)}}."""
    d = k.iloc[i - w:i]
    hi, lo, cl, vol = d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy(), d.volume.to_numpy()
    n, cp, mid = len(d), float(cl[-1]), (hi + lo) / 2
    fmin = np.r_[np.minimum.accumulate(lo[::-1])[::-1][1:], np.inf]; fmax = np.r_[np.maximum.accumulate(hi[::-1])[::-1][1:], -np.inf]
    age = (n - 1 - np.arange(n)).astype(float)
    out = {}
    for name, (surv, hl) in VARS.items():
        rec = np.exp(-age / hl) if hl else np.ones(n)
        o = {}
        for side, ent in (("long", mid), ("short", cl)):
            arr = np.zeros(2 * SPAN + 1)
            for L in TIERS:
                e = ent * ((1 - 1 / L + MM) if side == "long" else (1 + 1 / L - MM))
                off = np.floor((e / cp - 1) / BIN).astype(int)
                ok = (off < 0) if side == "long" else (off >= 0)
                if surv:
                    ok &= (e < fmin) if side == "long" else (e > fmax)
                ok &= np.abs(off) <= SPAN
                arr += np.bincount(off[ok] + SPAN, weights=(vol * rec * ent / len(TIERS))[ok], minlength=2 * SPAN + 1)
            o[side] = arr
        out[name] = o
    return cp, out


def realized(k: pd.DataFrame, liq: pd.DataFrame, i: int, H: int, cp: float, venues) -> tuple[dict, dict]:
    t0 = int(k.t.iloc[i]); p = k.iloc[i:i + H]
    rng = {"long": np.arange(max(-SPAN, int(np.floor((p.low.min() / cp - 1) / BIN))), 0),
           "short": np.arange(0, min(SPAN, int(np.floor((p.high.max() / cp - 1) / BIN))) + 1)}
    x = liq[(liq.ts_ms >= t0) & (liq.ts_ms < t0 + H * 3_600_000) & liq.venue.isin(venues)]
    off = np.floor((x.mark.to_numpy() / cp - 1) / BIN).astype(int)
    r = {}
    for s, keep in (("long", off < 0), ("short", off >= 0)):
        m = (x.side.to_numpy() == s) & keep & (np.abs(off) <= SPAN)
        r[s] = np.bincount(off[m] + SPAN, weights=x.usd.to_numpy()[m], minlength=2 * SPAN + 1)
    return rng, r


def score(mp: dict, rng: dict, r: dict) -> dict:
    out = {}
    for s in ("long", "short"):
        b = rng[s] + SPAN
        x, y = mp[s][b], r[s][b]
        if len(b) < 4 or y.sum() <= 0 or np.ptp(x) == 0:
            continue
        out[s] = (float(spearmanr(x, y).statistic), float(y[x >= np.quantile(x, 0.8)].sum() / y.sum()))
    return out


def boot(v: pd.DataFrame, col: str, n=2000, seed=1) -> tuple[float, float, float]:
    g = v.groupby("day")[col].agg(["sum", "count"]); rs = np.random.default_rng(seed)
    idx = rs.integers(0, len(g), size=(n, len(g)))
    est = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return float(v[col].mean()), float(np.quantile(est, .05)), float(np.quantile(est, .95))


def evaluate(k, liq, maps, H, venues, nshuf=10, since=None) -> pd.DataFrame:
    origins = [i for i in maps if k.timestamp.iloc[i] + pd.Timedelta(hours=H) <= O1 and (since is None or k.timestamp.iloc[i] >= since)]
    rs, rows = np.random.default_rng(0), []
    for i in origins:
        cp, mv = maps[i]
        rng, r = realized(k, liq, i, H, cp, venues)
        far = [j for j in maps if abs(j - i) >= 48]
        sh = rs.choice(far, size=nshuf, replace=False) if nshuf else []
        for name in VARS:
            sc = score(mv[name], rng, r)
            null = [score(maps[j][1][name], rng, r) for j in sh]
            for s, (ic, cap) in sc.items():
                nl = [z[s][0] for z in null if s in z]
                rows.append({"i": i, "day": str(k.timestamp.iloc[i].date()), "side": s, "var": name, "ic": ic, "cap": cap,
                             "ic_null": float(np.mean(nl)) if nl else np.nan})
    return pd.DataFrame(rows)


def paired(ev: pd.DataFrame, a: str, b: str, col: str = "ic") -> dict:
    p = ev.pivot_table(index=["i", "day", "side"], columns="var", values=col).dropna(subset=[a, b]).reset_index()
    p["d"] = p[b] - p[a]
    m, lo, hi = boot(p, "d")
    return {"diff": round(m, 4), "ci90": [round(lo, 4), round(hi, 4)], "n": len(p), "days": p.day.nunique(),
            f"{a}": round(p[a].mean(), 4), f"{b}": round(p[b].mean(), 4)}


def selftest() -> None:
    k = bars()
    i = int(np.searchsorted(k.t, int(pd.Timestamp("2026-10-01", tz="UTC").value // 10**6)))
    cp, mv = maps_at(k, i, w=48)
    # 대조 1: 변형 A(켬·240h) = 10-08 검증 틀 V.liq_map(같은 규약)과 칸마다 같다
    cp2, ref = V.liq_map(k, i, 48, TIERS, "splice", k.volume.to_numpy())
    assert abs(cp - cp2) < 1e-9
    for (s, o), v in ref.items():
        if abs(o) <= SPAN:
            assert abs(mv["A"][s][o + SPAN] - v) <= 1e-6 * max(1, v), (s, o, mv["A"][s][o + SPAN], v)
    assert abs(sum(a.sum() for a in mv["A"].values()) - sum(v for (s, o), v in ref.items() if abs(o) <= SPAN)) < 1e-3
    # 대조 2: 필터 끔은 켬보다 칸 값이 항상 크거나 같고, 실제로 어딘가 더 크다(지운 몫이 있었다)
    for s in ("long", "short"):
        assert (mv["B"][s] >= mv["A"][s] - 1e-9).all() and (mv["D"][s] >= mv["C"][s] - 1e-9).all()
    assert any((mv["B"][s] > mv["A"][s] + 1e-6).any() for s in ("long", "short"))
    # 대조 3: 지도는 봉 i 이후를 안 본다 -- i 뒤 봉을 흔들어도 같다
    k2 = k.copy(); k2.loc[i:, ["high", "low", "close", "volume"]] *= 1.3
    _, mv2 = maps_at(k2, i, w=48)
    assert all(np.allclose(mv[n][s], mv2[n][s]) for n in VARS for s in ("long", "short"))
    print("selftest OK -- A = V.liq_map 칸 일치 · 끔 ≥ 켬 · 미래 봉 흔들어도 지도 불변")


def main() -> None:
    k, liq = bars(), liqs()
    print("실측", liq.groupby("venue").usd.agg(["count", "sum"]).round(0).to_dict(), liq.ts_ms.min(), liq.ts_ms.max(), flush=True)
    idx = np.where((k.timestamp >= O0) & (k.timestamp < O1))[0]
    assert idx[0] >= W, "180일 앞 봉 부족"
    maps = {int(i): maps_at(k, int(i)) for i in idx}
    out = {"n_origins": len(maps)}
    allv = ("binance", "okx", "bybit")
    for H in (24, 4, 168):
        ev = evaluate(k, liq, maps, H, allv, nshuf=10 if H != 168 else 0)
        ev.to_csv(D / f"ev_H{H}.csv", index=False)
        res = {"B-A": paired(ev, "A", "B"), "D-C": paired(ev, "C", "D"), "C-A": paired(ev, "A", "C"), "D-A": paired(ev, "A", "D"),
               "cap B-A": paired(ev, "A", "B", "cap"), "cap D-C": paired(ev, "C", "D", "cap")}
        if H != 168:
            for n in VARS:
                e = ev[ev["var"] == n].assign(x=lambda z: z.ic - z.ic_null)
                res[f"excess {n}"] = dict(zip(("mean", "lo", "hi"), [round(v, 4) for v in boot(e.dropna(subset=["x"]), "x")]))
        out[f"H{H}"] = res
        print(f"H{H}", json.dumps(res, ensure_ascii=False), flush=True)
    ev = evaluate(k, liq, maps, 24, ("binance",), nshuf=0)
    out["H24 binance"] = {"B-A": paired(ev, "A", "B"), "D-C": paired(ev, "C", "D")}
    ev = evaluate(k, liq, maps, 24, allv, nshuf=0, since=pd.Timestamp("2026-10-27", tz="UTC"))
    out["H24 since 10-27"] = {"B-A": paired(ev, "A", "B"), "D-C": paired(ev, "C", "D")}
    print(json.dumps({k_: out[k_] for k_ in ("H24 binance", "H24 since 10-27")}, ensure_ascii=False))
    (D / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
