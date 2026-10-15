"""«DOM 트레이딩 전략»(흡수 + 불균형 + 확인 + 핵심 레벨 + 1R 부분 익절·본전) 그림 그대로 기계 검정 (2026-10-16, 사용자 «실제로 유효한지 검증»).

결과 전 고정(사전등록) -- ETH 1초 전체 호가 재생(±10bp 깊이·추가·제거) + 1초 테이프, 2026-09-16 ~ 10-13 19:00 UTC.
- 핵심 레벨(매시 H, H 이전 데이터만): 직전 24h 볼륨 프로파일 POC·VAH·VAL(70% 가치 영역, $0.1 칸) · 전날(UTC) 고가·저가 · 직전 4h 고가·저가.
- 롱(숏 거울): 초 i 에서
  ① 레벨 L 에 도달: |mid − L| ≤ 5bp 이고 5분 전 mid > L(위에서 내려옴)
  ② 흡수: 직전 60초 테이커 매도 ≥ 직전 24h 90분위 · 60초 mid 변화 ≥ −2bp · 매수쪽 순보강(추가 − (제거−매도)) ≥ 0
  ③ 불균형: ±10bp 안 매수 깊이 비중 ≥ 65%
  ④ 확인: i 뒤 30초 안 첫 초 j 에서 10초 순매수 ≥ 직전 24h 80분위 & mid ≥ mid_i + 1bp → j 매도호가에 시장가 진입(4bp).
     (지정가판: j 매수호가에 걸고 60초 안 mid 가 그 아래로 관통하면 체결, 수수료 0)
- 청산: 손절 = min(L, 직전 60초 최저 mid) − 3bp(닿으면 매수호가 시장가 4bp) · 1R 에서 절반 지정가 익절(0bp) 후 손절 → 진입가(본전, 시장가) ·
  나머지 = 진입가 + 1.5R 이상 중 가장 가까운 레벨(없으면 3R) 지정가 · 4h 시간 상한(시장가). 한 번에 포지션 하나.
- 대조: C1 레벨 없이 ②③④ 만 · C2 레벨 도달 ①만(15분에 한 번, 즉시 시장가) · C3 무작위 초·무작위 방향 · 같은 청산 기하(R 8~20bp).
- 판정: 건당 순 bp 의 일 블록 CI 하한 > 0 이고 A > C1·C2·C3. 28일이라 검정력이 약하다(보고).
사용: python scripts/research_eth_dom_playbook_20261016.py [--selftest]
"""
import json, sys
from pathlib import Path

import duckdb
import numpy as np, pandas as pd

LK = Path.home() / "backups/crypto-scalping-server/data/lake"
OLD = Path("/home/kbj20/crypto-scalping/.claude/worktrees/realtime-ledger-analysis-bdd373/tmp/order_battle_20261007/panel_1s.parquet")
NEW = Path("tmp/strength_composite_1m_20261016/book_new_1s.parquet")
OUT = Path("tmp/dom_playbook_20261016")
T0, T1 = int(pd.Timestamp("2026-09-16", tz="UTC").timestamp()), int(pd.Timestamp("2026-10-13T19:00", tz="UTC").timestamp())
TAKER = 4.0


def load():
    old = pd.read_parquet(OLD); new = pd.read_parquet(NEW)
    bk = pd.concat([old.drop(columns=["B", "S"]).loc[old.index < new.index.min()], new]).sort_index()
    idx = pd.RangeIndex(T0 - 86400 - 3600, T1)
    bk = bk[~bk.index.duplicated(keep="last")].reindex(idx)
    rp = f"read_parquet('{LK}/binance/tape/coin=ETH/*/*.parquet', hive_partitioning=false)"
    tp = duckdb.sql(f"select ts_sec sec, sum(buy_qty) bq, sum(sell_qty) sq from {rp} where symbol='ethusdt' and ts_sec >= {idx[0]} and ts_sec < {T1} group by 1").df()
    tp = tp.set_index("sec").rename(columns={"bq": "B", "sq": "S"})   # duckdb 별칭은 대소문자 무시(s·S 충돌)
    vp = duckdb.sql(f"select ts_sec // 60 m, price_bin, sum(buy_qty + sell_qty) v from {rp} where symbol='ethusdt' and ts_sec >= {idx[0]} and ts_sec < {T1} group by 1, 2").df()
    bk = bk.join(tp).fillna({"B": 0.0, "S": 0.0})
    ok = bk.valid == 1
    bk["mid"] = np.where(ok, (bk.bb + bk.ba) / 2, np.nan)
    for c in ("bb", "ba", "bid10", "ask10", "add_b", "rem_b", "add_a", "rem_a"):
        bk.loc[~ok, c] = np.nan
    return bk, vp


def value_area(v):
    """칸별 거래량(가격 순) → (POC, VAL, VAH) 칸 인덱스, 70%."""
    poc = int(np.argmax(v)); lo = hi = poc; tot = v.sum(); acc = v[poc]
    while acc < 0.7 * tot and (lo > 0 or hi < len(v) - 1):
        up = v[hi + 1] if hi < len(v) - 1 else -1; dn = v[lo - 1] if lo > 0 else -1
        if up >= dn:
            hi += 1; acc += up
        else:
            lo -= 1; acc += dn
    return poc, lo, hi


def levels(bk, vp):
    """매시 H 의 레벨 배열(H 이전 데이터만)."""
    mid = bk.mid; out = {}
    vm = vp.m.to_numpy()
    for H in range((T0 // 3600) * 3600, T1, 3600):
        w = vp[(vm >= (H - 86400) // 60) & (vm < H // 60)]
        L = []
        if len(w):
            g = w.groupby("price_bin").v.sum().sort_index()
            poc, lo, hi = value_area(g.to_numpy())
            L += [g.index[poc] * 0.1, g.index[lo] * 0.1, g.index[hi] * 0.1]
        d0 = (H // 86400) * 86400
        for s in (mid.loc[d0 - 86400:d0 - 1], mid.loc[H - 14400:H - 1]):
            if s.notna().any():
                L += [float(s.max()), float(s.min())]
        out[H] = np.array(L)
    return out


def signals(bk, lv):
    idx = bk.index.to_numpy(); n = len(idx)
    mid = bk.mid.to_numpy(); B = bk.B.to_numpy(); S = bk.S.to_numpy()
    cs = lambda a: np.concatenate([[0.0], np.cumsum(np.nan_to_num(a))])
    win = lambda c, k: c[1:] - np.concatenate([np.zeros(k), c[1:-k]])       # 초 i 까지 k초 합
    S60, B60 = win(cs(S), 60), win(cs(B), 60)
    F10 = win(cs(B), 10) - win(cs(S), 10)
    dmid60 = (mid - np.concatenate([np.full(60, np.nan), mid[:-60]])) / mid * 1e4
    mid300 = np.concatenate([np.full(300, np.nan), mid[:-300]])
    ndb = np.nan_to_num(bk.add_b.to_numpy()) - np.clip(np.nan_to_num(bk.rem_b.to_numpy()) - S, 0, None)
    nda = np.nan_to_num(bk.add_a.to_numpy()) - np.clip(np.nan_to_num(bk.rem_a.to_numpy()) - B, 0, None)
    NDB, NDA = win(cs(ndb), 60), win(cs(nda), 60)
    imb = (bk.bid10 - bk.ask10).to_numpy() / (bk.bid10 + bk.ask10).to_numpy()
    lo60 = pd.Series(mid).rolling(60, min_periods=30).min().to_numpy(); hi60 = pd.Series(mid).rolling(60, min_periods=30).max().to_numpy()
    hour = (idx // 3600) * 3600
    qS90, qB90, qF80, qF20 = (np.full(n, np.nan) for _ in range(4))
    for H in np.unique(hour):
        if H < T0:
            continue
        a, b = np.searchsorted(idx, H - 86400), np.searchsorted(idx, H)
        sl = np.arange(a, b, 10); cur = slice(b, np.searchsorted(idx, H + 3600))
        qS90[cur] = np.nanquantile(S60[sl], .9); qB90[cur] = np.nanquantile(B60[sl], .9)
        qF80[cur] = np.nanquantile(F10[sl], .8); qF20[cur] = np.nanquantile(F10[sl], .2)
    near = np.zeros(n, np.int8); lvl = np.full(n, np.nan)               # +1 = 위에서 내려와 닿음(지지) · −1 = 아래서 올라와 닿음(저항)
    for H, L in lv.items():
        a, b = np.searchsorted(idx, H), np.searchsorted(idx, H + 3600)
        if not len(L) or a >= b:
            continue
        m = mid[a:b][:, None]
        with np.errstate(invalid="ignore"):
            d = np.abs(m - L[None]) / m * 1e4
        k = np.argmin(np.where(np.isfinite(d), d, 1e9), 1); dk = d[np.arange(b - a), k]; Lk = L[k]
        with np.errstate(invalid="ignore"):
            sup = (dk <= 5) & (mid300[a:b] > Lk); res = (dk <= 5) & (mid300[a:b] < Lk)
        near[a:b] = np.where(sup, 1, np.where(res, -1, 0)); lvl[a:b] = Lk
    ok = (idx >= T0) & np.isfinite(mid)
    with np.errstate(invalid="ignore"):
        dom_l = (S60 >= qS90) & (dmid60 >= -2) & (NDB >= 0) & (imb >= 0.3) & ok
        dom_s = (B60 >= qB90) & (dmid60 <= 2) & (NDA >= 0) & (imb <= -0.3) & ok
    return dict(idx=idx, mid=mid, bb=bk.bb.to_numpy(), ba=bk.ba.to_numpy(), F10=F10, qF80=qF80, qF20=qF20, near=near, lvl=lvl,
                lo60=lo60, hi60=hi60, dom_l=dom_l, dom_s=dom_s, ok=ok, hourlv=lv, hour=hour)


def confirm(D, i, side):
    """i 뒤 30초 안 확인 초 j(없으면 −1)."""
    for j in range(i + 1, min(i + 31, len(D["mid"]))):
        m = D["mid"][j]
        if not np.isfinite(m):
            continue
        if side == 1 and D["F10"][j] >= D["qF80"][j] and m >= D["mid"][i] * (1 + 1e-4):
            return j
        if side == -1 and D["F10"][j] <= D["qF20"][j] and m <= D["mid"][i] * (1 - 1e-4):
            return j
    return -1


def manage(D, j, side, sl, maker=False):
    """진입 초 j → (순 bp, 결과, 끝 초) 또는 None. 절반 1R 익절 → 본전 · 나머지 다음 레벨(≥1.5R) 또는 3R · 4h."""
    mid, bb, ba = D["mid"], D["bb"], D["ba"]
    if maker:
        px = bb[j] if side == 1 else ba[j]
        for k in range(j + 1, min(j + 61, len(mid))):
            if np.isfinite(mid[k]) and ((side == 1 and mid[k] < px) or (side == -1 and mid[k] > px)):
                break
        else:
            return None
        j, fee_in = k, 0.0
    else:
        px = ba[j] if side == 1 else bb[j]; fee_in = TAKER
    R = side * (px - sl)
    if not (R > 0):
        return None
    tp1 = px + side * R
    L = D["hourlv"].get(int(D["hour"][j]), np.array([]))
    cand = L[side * (L - px) >= 1.5 * R] if len(L) else L
    tp2 = cand[np.argmin(np.abs(cand - px))] if len(cand) else px + side * 3 * R
    half, stop, pnl = False, sl, 0.0
    end = min(j + 14400, len(mid) - 1)
    for k in range(j + 1, end + 1):
        m = mid[k]
        if not np.isfinite(m):
            continue
        if side * (m - stop) <= 0:                                          # 손절/본전 -- 시장가
            ex = bb[k] if side == 1 else ba[k]
            pnl += (0.5 if half else 1.0) * (side * (ex - px) / px * 1e4 - TAKER)
            return pnl - fee_in, ("본전" if half else "손절"), k
        if not half and side * (m - tp1) >= 0:
            pnl += 0.5 * (R / px * 1e4); half = True; stop = px
        if half and side * (m - tp2) >= 0:
            pnl += 0.5 * (side * (tp2 - px) / px * 1e4)
            return pnl - fee_in, "목표", k
    ex = bb[end] if side == 1 else ba[end]
    pnl += (0.5 if half else 1.0) * (side * (ex - px) / px * 1e4 - TAKER)
    return pnl - fee_in, "시간", end


def run_rule(D, kind, maker=False):
    n = len(D["mid"]); trades = []; i = int(np.argmax(D["ok"]))
    last_touch = {1: -10**9, -1: -10**9}
    while i < n - 60:
        side = 0
        if kind in ("A", "C1"):
            if D["dom_l"][i] and (kind == "C1" or D["near"][i] == 1):
                side = 1
            elif D["dom_s"][i] and (kind == "C1" or D["near"][i] == -1):
                side = -1
            if side:
                j = confirm(D, i, side)
                if j < 0:
                    i += 1; continue
        elif kind == "C2":
            s = int(D["near"][i])
            if s and i - last_touch[s] >= 900 and D["ok"][i]:
                side, j = s, i; last_touch[s] = i
        if not side:
            i += 1; continue
        ref = D["mid"][i] if kind == "C1" else D["lvl"][i]
        sl = min(ref, D["lo60"][i]) * (1 - 3e-4) if side == 1 else max(ref, D["hi60"][i]) * (1 + 3e-4)
        r = manage(D, j, side, sl, maker)
        if r is None:
            i = j + 1; continue
        trades.append((D["idx"][j], side, r[0], r[1])); i = r[2] + 1
    return trades


def run_random(D, nn, rng):
    ok = np.flatnonzero(D["ok"])[:-14400]; trades = []
    for _ in range(nn):
        j = int(rng.choice(ok)); side = int(rng.choice([-1, 1]))
        px = D["ba"][j] if side == 1 else D["bb"][j]
        if not np.isfinite(px):
            continue
        Rbp = float(rng.choice([8.0, 10, 12, 15, 20]))
        r = manage(D, j, side, px * (1 - side * Rbp / 1e4))
        if r is not None:
            trades.append((D["idx"][j], side, r[0], r[1]))
    return trades


def summarize(tr):
    if not tr:
        return {"건수": 0}
    df = pd.DataFrame(tr, columns=["t", "side", "bp", "how"]); df["day"] = df.t // 86400
    rng = np.random.default_rng(1); ud = df.day.unique(); g = {u: df.bp[df.day == u].to_numpy() for u in ud}
    bs = [np.mean(np.concatenate([g[u] for u in rng.choice(ud, len(ud))])) for _ in range(2000)]
    return {"건수": int(len(df)), "하루당": round(len(df) / ((T1 - T0) / 86400), 2), "승률": round(float((df.bp > 0).mean()), 3),
            "평균 순 bp": round(float(df.bp.mean()), 2), "CI": [round(float(np.quantile(bs, q)), 2) for q in (.025, .975)],
            "합 bp": round(float(df.bp.sum()), 1), "결과": df.how.value_counts().to_dict(), "롱/숏": [int((df.side == 1).sum()), int((df.side == -1).sum())]}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    bk, vp = load()
    print("초", len(bk), "유효", round(float(bk.mid.notna().mean()), 3), flush=True)
    lv = levels(bk, vp); D = signals(bk, lv)
    print("레벨 도달 초", int((D["near"] != 0).sum()), "흡수+불균형 초 롱/숏", int(D["dom_l"].sum()), int(D["dom_s"].sum()), flush=True)
    res = {"A 그림 전략(시장가 진입)": summarize(A := run_rule(D, "A")),
           "A' 지정가 진입": summarize(run_rule(D, "A", maker=True)),
           "C1 레벨 없이 호가·체결만": summarize(run_rule(D, "C1")),
           "C2 레벨 도달만(즉시 시장가)": summarize(run_rule(D, "C2")),
           "C3 무작위(같은 청산 기하)": summarize(run_random(D, max(300, 5 * len(A)), np.random.default_rng(20261016)))}
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False), flush=True)
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def selftest():
    assert value_area(np.array([1, 2, 10, 3, 1.])) == (2, 2, 3)
    # 롱: 진입 100(매도호가) · 손절 99 · 1R 101 → 절반 +100bp, 본전으로 내려와 매수호가 99.89 시장가
    mid = np.array([100, 100, 100.5, 101.2, 100.6, 99.9])
    D = {"mid": mid, "bb": mid - .01, "ba": mid.copy(), "hourlv": {}, "hour": np.zeros(6)}
    pnl, how, k = manage(D, 0, 1, 99.0)
    exp = 0.5 * 100 + 0.5 * ((99.89 - 100) / 100 * 1e4 - TAKER) - TAKER
    assert how == "본전" and abs(pnl - exp) < 1e-6, (pnl, how, exp)
    print("selftest ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else run()
