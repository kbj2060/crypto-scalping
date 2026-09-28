"""가격 행 단위 «흡수 라인» -- 버티는가, 방향을 주는가? (사전등록 2026-09-29, 결과 보기 전에 고정)

사용자 «풋프린트·호가·체결·수급으로 흡수 라인을 알 수 있나 -- 먼저 재 보고 통과하면 화면에».
09-08/09-20 의 흡수는 봉·60초 창 전체를 뭉친 값이었다(전부 0). 여기는 **가격 행 하나**를 본다.

입력(서버 읽기 전용 추출, tmp/absorb_20260929/export_probe.py):
  tape.parquet   trade_tape_1s  (ts_sec, price_bin=$0.1 칸, buy_qty, sell_qty)  09-16~
  bt_1s.parquet  bookTicker 초마다 마지막 bp,bq,ap,aq

사건(매도 쪽 = 위에서 매수를 받아낸 라인, 매수 쪽은 거울) -- 초 t 가 **끝난 뒤** 아는 것만(≤ t):
  창 W = [t-59, t].  H = 창 최고 체결 칸.  윗구역 = 칸 H-1..H ($0.2).
  버팀    : H 에 처음 닿은 초가 t-20 이전(20초 넘게 위로 못 뚫음) · 윗구역 매수 체결 초 ≥ 5 (반복해서 두드림)
            · 초 t 의 체결 중심이 윗구역 3칸 안(지금도 그 라인을 두드리는 중)
  흡수(A) : 윗구역 매수량 V ≥ DEV 의 «버팀» 초들 V 90분위
  빙산(B) : A 이면서 V ≥ 3 × (창 안에서 ap 가 그 칸일 때 보인 매도 잔량 중앙값) -- 보인 것보다 3배 넘게 먹혔다
  대조(C) : «버팀»은 같고 V ≤ DEV 50분위 -- 조용히 버틴 고점. ⭐흡수의 몫 = 사건 − C («20초 버틴 고점»이 공짜로 맞히는 몫을 뺀다)
  같은 쪽·같은 무리 사건은 300초 안에 다시 세지 않는다(라벨 창 안 겹침).

라벨(t+1 부터 -- 사건·라벨 경계 계약, 기준가 = 초 t 끝 mid):
  돌파  : t+1..t+300 에 ap_hi(매수 쪽은 bp_lo)가 라인(칸 H 상단 + 2bp, 매수 쪽은 칸 하단 − 2bp)을 넘는가
  방향  : s × (mid[t+300] − mid[t]) / mid[t] (bp), s = 매도 쪽 −1 · 매수 쪽 +1
          + 잔차판: DEV 전체 초로 적합한 OLS(이동60·이동300·60분 레인지 위치)를 빼고 s 를 곱한다
DEV = 앞 60% 기간(분위 임계는 여기서만) · HOLDOUT = 뒤 40%. 불확실성 = 시간(3600초) 블록 부트스트랩 2,000회.

통과(화면에 올림) -- HOLDOUT 에서:
  P1 라인 버팀 : 돌파율(사건 A∪B 또는 B) − 돌파율(C) ≤ −10pp, 95% CI 상단 < 0, DEV 같은 부호
  P2 방향      : 잔차 방향 평균 > 1.41bp(USDC 메이커 왕복) 이고 t ≥ 2, DEV 같은 부호
  P1 만 통과 = «라인(지지/저항)으로 표시» · P2 까지 = «방향 참고» 문구 허용. 둘 다 실패 = 화면에 안 올림.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "tmp/absorb_20260929/export"
W, HOLD, HITS, NEAR, H5, BREAK_BP, DEDUP = 60, 20, 5, 3, 300, 2.0, 300
RNG = np.random.default_rng(20260929)


def load():
    return pd.read_parquet(EXP / "tape.parquet"), pd.read_parquet(EXP / "bt_1s.parquet")


def build(t: pd.DataFrame, b: pd.DataFrame):
    """초 격자 배열 + 사건 후보 표(버팀 조건을 만족한 초만). side='ask'(위 라인) / 'bid'(아래 라인)."""
    t0, t1 = int(max(t.ts_sec.min(), b.index.min())), int(min(t.ts_sec.max(), b.index.max()))
    secs = np.arange(t0, t1 + 1)
    n = len(secs)
    bb = b.reindex(secs)
    mid = ((bb.bp + bb.ap) / 2).to_numpy()
    ap, aq, bp, bq = bb.ap.to_numpy(), bb.aq.to_numpy(), bb.bp.to_numpy(), bb.bq.to_numpy()
    ap_hi, bp_lo = bb.ap_hi.to_numpy(), bb.bp_lo.to_numpy()
    t = t[(t.ts_sec >= t0) & (t.ts_sec <= t1)].sort_values(["ts_sec", "price_bin"])
    ri = t.ts_sec.to_numpy() - t0
    rb, rbuy, rsell = t.price_bin.to_numpy().astype(float), t.buy_qty.to_numpy(), t.sell_qty.to_numpy()
    start = np.searchsorted(ri, np.arange(n)); end = np.searchsorted(ri, np.arange(n), side="right")
    has = end > start
    hi = np.full(n, np.nan); lo = np.full(n, np.nan)
    hi[has] = np.maximum.reduceat(rb, start[has]); lo[has] = np.minimum.reduceat(rb, start[has])
    w = rbuy + rsell
    cs = np.bincount(ri, weights=rb * w, minlength=n); cw = np.bincount(ri, weights=w, minlength=n)
    cpx = np.full(n, np.nan); cpx[cw > 0] = cs[cw > 0] / cw[cw > 0]   # 초 안 체결 중심 칸(초 안 순서는 없다)
    bt_ok = ~np.isnan(mid)
    ok_cum = np.concatenate([[0], np.cumsum(bt_ok)])
    roll_hi = pd.Series(hi).rolling(W, min_periods=1).max().to_numpy()
    roll_lo = pd.Series(lo).rolling(W, min_periods=1).min().to_numpy()

    rows = []
    for i in range(W, n - 2 * H5 - 1):   # 2×H5: 돌파 순간(≤ t+300) 뒤 300초까지 라벨이 필요하다(09-29 역방향 탐색)
        if ok_cum[i + 1] - ok_cum[i + 1 - W] < W * 0.9 or not bt_ok[i] or not bt_ok[i + H5] or np.isnan(cpx[i]):
            continue
        s, e = start[i - W + 1], end[i]
        if e <= s:
            continue
        wb, wbuy, wsell, wi = rb[s:e], rbuy[s:e], rsell[s:e], ri[s:e]
        for side in ("ask", "bid"):
            H = roll_hi[i] if side == "ask" else roll_lo[i]
            if np.isnan(H):
                continue
            if (cpx[i] < H - NEAR + 1) if side == "ask" else (cpx[i] > H + NEAR - 1):
                continue
            if i - wi[wb == H].min() < HOLD:
                continue
            zone = (wb >= H - 1) if side == "ask" else (wb <= H + 1)
            q = wbuy if side == "ask" else wsell
            if np.unique(wi[zone & (q > 0)]).size < HITS:
                continue
            V = q[zone].sum()
            lvl = H * 0.1
            sl = slice(i - W + 1, i + 1)
            if side == "ask":
                m = (ap[sl] >= lvl - 1e-9) & (ap[sl] < lvl + 0.1 - 1e-9)
                disp = np.nanmedian(aq[sl][m]) if m.any() else np.nan
                over = np.flatnonzero(ap_hi[i + 1:i + H5 + 1] > (lvl + 0.1) * (1 + BREAK_BP / 1e4))
                sg = -1.0
            else:
                m = (bp[sl] >= lvl - 1e-9) & (bp[sl] < lvl + 0.1 - 1e-9)
                disp = np.nanmedian(bq[sl][m]) if m.any() else np.nan
                over = np.flatnonzero(bp_lo[i + 1:i + H5 + 1] < lvl * (1 - BREAK_BP / 1e4))
                sg = 1.0
            brk = over.size > 0
            # 돌파 순간 tau = 라인을 처음 넘은 초(그 초가 끝나야 안다) -> 따라가기 라벨은 mid[tau] 에서 mid[tau+300] (돌파 방향 = −sg)
            tau = i + 1 + over[0] if brk else -1
            cont = (-sg) * (mid[tau + H5] - mid[tau]) / mid[tau] * 1e4 if brk and bt_ok[tau] and bt_ok[tau + H5] else np.nan
            rows.append((i, side, V, disp, bool(brk), sg * (mid[i + H5] - mid[i]) / mid[i] * 1e4, tau - i if brk else np.nan, cont))
    ev = pd.DataFrame(rows, columns=["i", "side", "V", "disp", "brk", "r5s", "ttb", "cont"])
    ev["ts"] = secs[ev.i] if len(ev) else []
    mids = pd.Series(mid)
    rmax = mids.rolling(3600, min_periods=600).max(); rmin = mids.rolling(3600, min_periods=600).min()
    X = pd.DataFrame({"d60": (mids / mids.shift(60) - 1) * 1e4, "d300": (mids / mids.shift(300) - 1) * 1e4,
                      "rpos": (mids - rmin) / (rmax - rmin), "fwd": (mids.shift(-H5) / mids - 1) * 1e4})
    return secs, X, ev


def dedup(ev: pd.DataFrame) -> pd.DataFrame:
    keep, last = [], {}
    for r in ev.sort_values("i").itertuples():
        if r.i - last.get(r.side, -10**9) >= DEDUP:
            keep.append(r.Index); last[r.side] = r.i
    return ev.loc[keep]


def _blocks(blk):
    u = np.unique(blk)
    return u, {b: np.flatnonzero(blk == b) for b in u}


def boot_mean(x, blk, n=2000):
    u, ix = _blocks(blk)
    return np.array([np.concatenate([x[ix[b]] for b in RNG.choice(u, len(u))]).mean() for _ in range(n)])


def boot_diff(a, ablk, c, cblk, n=2000):
    ua, ia = _blocks(ablk); uc, ic = _blocks(cblk)
    return np.array([np.concatenate([a[ia[b]] for b in RNG.choice(ua, len(ua))]).mean()
                     - np.concatenate([c[ic[b]] for b in RNG.choice(uc, len(uc))]).mean() for _ in range(n)])


def main():
    t, b = load()
    secs, X, ev = build(t, b)
    split = secs[0] + int((secs[-1] - secs[0]) * 0.6)
    ev["dev"] = ev.ts < split
    dev = ev[ev.dev]
    q90 = {s: dev[dev.side == s].V.quantile(0.9) for s in ("ask", "bid")}
    q50 = {s: dev[dev.side == s].V.quantile(0.5) for s in ("ask", "bid")}
    grp = np.where(ev.V >= ev.side.map(q90), "A", np.where(ev.V <= ev.side.map(q50), "C", "-"))
    ev["ice"] = ev.V >= 3 * ev.disp
    Xd = X[secs < split].dropna()
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Xd)), Xd[["d60", "d300", "rpos"]]], Xd.fwd.to_numpy(), rcond=None)
    xi = X.iloc[ev.i.to_numpy()][["d60", "d300", "rpos"]].to_numpy()
    pred = np.nan_to_num(np.c_[np.ones(len(xi)), xi] @ beta)
    ev["res"] = ev.r5s - np.where(ev.side == "ask", -1.0, 1.0) * pred
    groups = {"AB": ev[grp == "A"], "B": ev[(grp == "A") & ev.ice], "C": ev[grp == "C"]}
    groups = {k: dedup(v).assign(blk=lambda d: d.ts // 3600) for k, v in groups.items()}
    print(f"기간 {pd.to_datetime(secs[0], unit='s')} ~ {pd.to_datetime(secs[-1], unit='s')} UTC · 분할 {pd.to_datetime(split, unit='s')}")
    print(f"버팀 후보 초 {len(ev):,} · 임계 V90 위/아래 {q90['ask']:.1f}/{q90['bid']:.1f} ETH · V50 {q50['ask']:.1f}/{q50['bid']:.1f}")
    verdict = {}
    for part in ("DEV", "HOLDOUT"):
        print(f"\n== {part}")
        sub = {k: v[v.dev == (part == "DEV")] for k, v in groups.items()}
        for k, x in sub.items():
            if len(x) < 10:
                print(f"  {k}: n={len(x)} (부족)"); continue
            se = boot_mean(x.res.to_numpy(), x.blk.to_numpy()).std()
            print(f"  {k}: n={len(x)} (시간블록 {x.blk.nunique()}) · 위/아래 {(x.side=='ask').sum()}/{(x.side=='bid').sum()} · "
                  f"돌파 {x.brk.mean():.1%} · 방향 {x.r5s.mean():+.2f}bp · 잔차 {x.res.mean():+.2f} ± {se:.2f} (t {x.res.mean()/se:.2f})")
        c = sub["C"]
        for k in ("AB", "B"):
            a = sub[k]
            if len(a) < 10 or len(c) < 10:
                continue
            d = boot_diff(a.brk.to_numpy(float), a.blk.to_numpy(), c.brk.to_numpy(float), c.blk.to_numpy())
            dr = boot_diff(a.res.to_numpy(), a.blk.to_numpy(), c.res.to_numpy(), c.blk.to_numpy())
            lo_, hi_ = np.percentile(d, [2.5, 97.5])
            diff = a.brk.mean() - c.brk.mean()
            print(f"  {k}−C 돌파율 {diff*100:+.1f}pp [{lo_*100:+.1f}, {hi_*100:+.1f}] · 잔차 방향 차 {a.res.mean()-c.res.mean():+.2f} ± {dr.std():.2f}")
            se = boot_mean(a.res.to_numpy(), a.blk.to_numpy()).std()
            verdict[(part, k)] = dict(brk=diff, brk_hi=hi_, res=a.res.mean(), res_t=a.res.mean() / se)
    print("\n== 판정 (HOLDOUT, DEV 같은 부호)")
    for k in ("AB", "B"):
        h, d = verdict.get(("HOLDOUT", k)), verdict.get(("DEV", k))
        if not h or not d:
            print(f"  {k}: 표본 부족"); continue
        p1 = h["brk"] <= -0.10 and h["brk_hi"] < 0 and d["brk"] < 0
        p2 = h["res"] > 1.41 and h["res_t"] >= 2 and d["res"] > 0
        print(f"  {k}: P1 라인 버팀 {'통과' if p1 else '실패'} · P2 방향 {'통과' if p2 else '실패'}")


def _selfcheck():
    """합성: 위 라인 2000.0 을 100초 동안 두드리다 내려간다 -> 20초 버틴 뒤에야 매도 쪽 사건 · 돌파 없음 · 방향 양수."""
    n, base = 900, 1_700_000_000
    k = np.arange(n)
    rows = []
    for j in k:
        if 100 <= j < 200:
            rows += [(base + j, 20000, 2.0, 0.1), (base + j, 19999, 0.5, 0.5)]
        else:
            rows.append((base + j, 19990 if j < 100 else 19985, 0.2, 0.2))
    t = pd.DataFrame(rows, columns=["ts_sec", "price_bin", "buy_qty", "sell_qty"])
    px = np.where((k >= 100) & (k < 200), 1999.99, np.where(k >= 200, 1998.5, 1999.0))
    b = pd.DataFrame({"bp": px - 0.01, "bq": 5.0, "ap": px + 0.01, "aq": 1.0, "ap_hi": px + 0.01, "bp_lo": px - 0.01},
                     index=pd.Index(base + k, name="sec"))
    _, _, ev = build(t, b)
    a = ev[(ev.side == "ask") & ev.i.between(100, 199)]   # 평평한 앞 구간도 «버팀»으로 잡힌다(= 대조군 C 의 재료) -- 라인 구간만 본다
    assert len(a) and a.i.min() >= 100 + HOLD, a.head()   # 라인에 처음 닿고 20초 뒤부터
    assert not a.brk.any()                                 # 2000.0 을 안 넘었다
    assert (a.r5s > 0).all()                               # 매도 쪽 s=−1 × 하락 = 양수
    assert (a.V > ev[ev.i < 100].V.max()).all()            # 라인의 체결량이 평평한 구간보다 크다
    print("selfcheck ok", len(a))


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
