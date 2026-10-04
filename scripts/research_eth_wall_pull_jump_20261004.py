"""호가벽 «섰다가 체결 없이 사라짐(철회)» 뒤 가격이 튀는가 -- 사전등록 검정 (2026-10-04).

사용자: «벽이 생겼다가 갑자기 사라지면서 가격이 점핑하는데 이걸로 전략을 세워줘».
원천(서버, 읽기 전용): data/live/orderflow/depthdiff/<SYM>/<UTC시>.jsonl(.gz) (첫 줄 REST 스냅샷 + @depth@100ms)
                     data/live/orderflow/bookticker/<SYM>/<UTC시>.bt(.gz) (32B 행, 거래소 T ms)
                     data/lake/binance/tape/coin=<C>/date=*/ (1초 x 가격칸 테이커 매수/매도량)
실행:
  python scripts/research_eth_wall_pull_jump_20261004.py --selftest
  (서버) nice -n 19 python scripts/research_eth_wall_pull_jump_20261004.py --replay ETH   -> tmp/wall_pull_jump_20261004/
  python scripts/research_eth_wall_pull_jump_20261004.py --analyze ETH --phase dev        (DEV 먼저, 방향 고정)
  python scripts/research_eth_wall_pull_jump_20261004.py --analyze ETH --phase holdout    (한 번만)
  python scripts/research_eth_wall_pull_jump_20261004.py --analyze BTC --phase all        (다코인 재현)
시점 경계: known_ts = 감소를 담은 diff 의 거래소 T. 피쳐·분류는 known_ts 까지, 진입은 known_ts+L 이후 첫 bookTicker.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/wall_pull_jump_20261004"
T = lambda s: int(pd.Timestamp(s).timestamp()) * 1000  # noqa: E731  (ms)

# ── 사전등록 (결과 보기 전에 고정) ──────────────────────────────────────────────
CRITERIA = dict(
    # 벽 = 감소 시작(known-1s) 시점 수량이 같은 쪽 ±50bp 총 수량의 share 이상 (+ 추적 하한: 같은 쪽 ±50bp 비영 레벨 중앙의 20배).
    # 🔴처음 등록은 «중앙의 k=10배(5·20)»였다. 방향 통계(쪽별 수익)를 보기 전, 표본 1시간(09-28T12) 사건 수만 보고 바꿨다:
    #   k=10·T=3s·5~30bp 철회가 1초 중복제거 후 시간당 4,144건, k=50 도 2,702건(1.3초마다) · 중앙 명목 $14~24만
    #   -- ETH 는 먼지 레벨이 많아 중앙이 작고, «k배»는 벽이 아니라 마켓메이커 사다리 재배치다.
    #   share 분포(사건): 중앙 0.2% · p99 1.9% · p99.9 3.0%. share≥2%·T=3s 면 시간당 ~190건, 중앙 명목 $150만.
    wall_share=0.02, wall_share_sens=(0.01, 0.04),
    t_live_ms=3000, t_live_sens=(1000, 10000),  # 감소 시작(known-1s) 전에 연속으로 벽이었던 시간
    drop_frac=0.8, drop_window_ms=1000,         # 1초 안에 80% 이상 감소
    pull_fill_max=0.2, eaten_fill_min=0.8,      # 감소량 중 같은 가격칸 테이커 체결 몫
    dist_bp=(5, 30),                            # 감소 직전 mid 기준 벽 거리(주 정의)
    lat_ms=500, lat_sens=(100, 1000),           # 진입 지연 L (OBS=0 금지)
    horizons_s=(1, 5, 30, 60, 300), main_h=30,  # 주 지평 30초
    dedup_ms=1000,                              # 같은 쪽 사건 1초 안 연쇄는 첫 건만
    # 판정 통계 S = mean(r | 매도벽 철회) - mean(r | 매수벽 철회) (드리프트 상쇄). H_main: S>0, H_alt: S<0.
    # DEV: 1h 블록 부트스트랩 95% CI 가 0 을 배제하면 그 방향을 HOLDOUT 방향으로 고정(배제 못 하면 H_main 으로 검정).
    # HOLDOUT 통과 = 고정 방향으로 (a) 원 S, (b) 무작위 대조 잔차 S, (c) 직전 1·5·60초 이동 통제 S 셋 다 95% CI 0 배제.
    # 동행 판정: S 를 직전 1초(known-1s->known)와 지연 구간(known->known+L)에서도 잰다. 그 둘이 크고 L 이후가 0 이면 «동행».
    # 매매화(HOLDOUT 통과 시만): 테이커 왕복 8bp 차감, 일 클러스터 부트스트랩 95% CI 하한 > 0 이어야 통과.
    split_eth=("2026-09-15T00Z", "2026-09-26T10Z", "2026-10-04T00Z"),  # DEV 60% / HOLDOUT 40%(최신)
    multi_window=("2026-09-26T09Z", "2026-10-04T00Z"),
    taker_rt_bp=8.0, maker_rt_bp=1.41, boot=2000, seed=20261004,
)
KS = (20, 50, 100)   # 추적 하한 = KS[0]
TICKS = {"ETH": 100, "BTC": 10, "SOL": 100, "XRP": 10_000, "HYPE": 1_000}
# 체결 테이프 가격칸 = round(price / bucket)  (scripts/live_trade_tape_collector_20260916.py BUCKETS 와 같아야 한다)
BUCKETS = {"ETH": 0.1, "BTC": 1.0, "SOL": 0.01, "XRP": 0.0001, "HYPE": 0.001}
IDX_MAX = 10_000_000
LATS = (100, 500, 1000)


def _open(path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


# ── 북 재구성 + 벽 추적 ────────────────────────────────────────────────────────
def _best(arr, cur, low_side):
    w = max(int(cur * 0.02), 100)
    lo = max(cur - w, 0)
    nz = np.flatnonzero(arr[lo:cur + w + 1])
    if len(nz):
        return lo + (nz[-1] if low_side else nz[0])
    nz = np.flatnonzero(arr)
    return (nz[-1] if low_side else nz[0]) if len(nz) else -1


def replay(lines, tick: int, idx_max: int = IDX_MAX, ctrl_every_s: int = 5):
    """lines: 파일 줄 반복자. 반환 (events, controls, cover). 끊김(pu 불연속·_gap·잘린 줄)이면 그 시점에서 멈춘다
    (스냅샷이 시각 파일 첫 줄에만 있어 그 뒤 북은 복구 불가 -- ponytail: 다음 시각 파일에서 다시 시작)."""
    bid = np.zeros(idx_max); ask = np.zeros(idx_max)
    first = json.loads(next(lines))
    snap = first.get("_snapshot") if isinstance(first, dict) else None
    if not snap:
        return [], [], dict(status="no_snapshot")
    for p, q in snap["bids"]:
        i = int(round(float(p) * tick))
        if i < idx_max: bid[i] = float(q)
    for p, q in snap["asks"]:
        i = int(round(float(p) * tick))
        if i < idx_max: ask[i] = float(q)
    last_u, synced = snap["lastUpdateId"], False
    bb, ba = int(np.flatnonzero(bid)[-1]), int(np.flatnonzero(ask)[0])
    med = [np.nan, np.nan]
    dep = [np.nan, np.nan]      # 같은 쪽 ±50bp 총 수량
    tracked: dict = {}          # (side, idx) -> hist [(t, q, s_k for k in KS)]
    events, controls = [], []
    cur_sec = t0 = t1 = None
    status = "ok"
    for line in lines:
        try:
            m = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            status = "truncated"; break
        if m.get("_gap"):
            status = "gap"; break
        if m.get("e") != "depthUpdate":
            continue
        if not synced:
            if m["u"] < last_u:
                continue
            if not (m["U"] <= last_u <= m["u"]):
                status = "sync_fail"; break
            synced = True
        elif m["pu"] != last_u:
            status = "gap"; break
        last_u, t = m["u"], int(m["T"])
        t0 = t if t0 is None else t0; t1 = t
        sec = t // 1000
        if cur_sec is None or sec > cur_sec:
            cur_sec = sec
            mid_i = (bb + ba) / 2
            w = int(mid_i * 50e-4)
            for side, base, sl in ((0, max(bb - w, 0), bid[max(bb - w, 0):bb + 1]), (1, ba, ask[ba:ba + w + 1])):
                nz = sl[sl > 0]
                med[side] = md = float(np.median(nz)) if len(nz) else np.nan
                dep[side] = float(sl.sum())
                if np.isfinite(md):   # diff 로 안 건드려진 벽(스냅샷 벽 등)도 잡는다. 탄생 시각은 지금으로(나이 과소 = 보수적)
                    for j in np.flatnonzero(sl >= KS[0] * md):
                        tracked.setdefault((side, base + int(j)), [(t, float(sl[j])) + tuple(t if sl[j] >= k * md else None for k in KS)])
            far = int(mid_i * 60e-4)
            for key in [k for k, h in tracked.items()
                        if abs(k[1] - mid_i) > far or (h[-1][0] < t - 1500 and h[-1][2] is None)]:
                del tracked[key]
            if sec % ctrl_every_s == 0:
                for side in (0, 1):
                    # 대조 = 지금 서 있는(나이≥1초) 같은 쪽 가장 큰 레벨 -- 사라지지 않은 벽
                    cand = [(k, h[-1]) for k, h in tracked.items() if k[0] == side and h[-1][2] is not None
                            and t - h[-1][2] >= 1000 and h[-1][1] >= KS[0] * med[side]]
                    if cand:
                        (s_, i), e = max(cand, key=lambda c: c[1][1])
                        controls.append(_row(t, side, i, e, e[1], med[side], bb, ba, bb, ba, tick, t, dep[side]))
        pre_bb, pre_ba = bb, ba
        touched = []
        for side, key, arr in ((0, "b", bid), (1, "a", ask)):
            for p, q in m[key]:
                i = int(round(float(p) * tick))
                if i >= idx_max:
                    continue
                arr[i] = float(q)
                touched.append((side, i, float(q)))
        bb, ba = _best(bid, bb, True), _best(ask, ba, False)
        if bb < 0 or ba < 0:
            status = "empty_book"; break
        pre_mid = (pre_bb + pre_ba) / 2
        for side, i, q in touched:
            md = med[side]
            if not np.isfinite(md):
                continue
            h = tracked.get((side, i))
            if h is None:
                if q >= KS[0] * md and abs(i - pre_mid) <= pre_mid * 50e-4:
                    tracked[(side, i)] = [(t, q) + tuple(t if q >= k * md else None for k in KS)]
                continue
            prev = h[-1]
            s_new = tuple((ps if ps is not None else t) if q >= k * md else None for k, ps in zip(KS, prev[2:]))
            ref = next((e for e in reversed(h) if e[0] <= t - CRITERIA["drop_window_ms"]), None)
            if ref is not None and ref[2] is not None and q <= (1 - CRITERIA["drop_frac"]) * ref[1] \
                    and (ref[0] - ref[2]) >= 0 and (t - CRITERIA["drop_window_ms"]) - ref[2] >= 1000:
                events.append(_row(t, side, i, ref, q, md, pre_bb, pre_ba, bb, ba, tick, t - CRITERIA["drop_window_ms"], dep[side]))
                del tracked[(side, i)]
                continue
            h.append((t, q) + s_new)
            while len(h) > 2 and h[1][0] <= t - CRITERIA["drop_window_ms"]:
                h.pop(0)
    return events, controls, dict(status=status, t_first=t0, t_last=t1)


def _row(t, side, i, ref, q, md, pre_bb, pre_ba, bb, ba, tick, t_ref, dep):
    pre_mid = (pre_bb + pre_ba) / 2
    post_mid = (bb + ba) / 2
    dist = ((pre_mid - i) if side == 0 else (i - pre_mid)) / pre_mid * 1e4
    crossed = (post_mid < i) if side == 0 else (post_mid > i)
    ages = [(t_ref - s) if s is not None else np.nan for s in ref[2:]]
    return dict(t=t, side=side, px=i / tick, qref=ref[1], q=q, ratio=ref[1] / md, share=ref[1] / dep, dist_bp=dist,
                crossed=bool(crossed), at_top=bool(i == pre_bb if side == 0 else i == pre_ba),
                **{f"age{k}": a for k, a in zip(KS, ages)})


# ── 바깥 데이터: bookTicker·테이프 ─────────────────────────────────────────────
BT_DT = np.dtype([("ts", "<i8"), ("bp", "<f8"), ("bq", "<f4"), ("ap", "<f8"), ("aq", "<f4")])


def load_bt(paths):
    parts = []
    for p in paths:
        with _open(p) as f:
            b = f.read()
        parts.append(np.frombuffer(b, dtype=BT_DT, count=(len(b) - 32) // 32, offset=32))
    a = np.concatenate(parts) if parts else np.zeros(0, BT_DT)
    a = a[np.argsort(a["ts"], kind="stable")]
    return a["ts"], (a["bp"] + a["ap"]) / 2


def outcomes(df: pd.DataFrame, ts, mid) -> pd.DataFrame:
    """known_ts(df.t) 기준. 과거값은 «이하 마지막 행», 진입·청산은 «이상 첫 행»(2초 넘게 비면 NaN)."""
    if not len(df):
        return df
    t = df.t.to_numpy()

    def before(x):
        i = np.searchsorted(ts, x, "right") - 1
        ok = (i >= 0) & (x - ts[np.clip(i, 0, None)] <= 2000)
        return np.where(ok, mid[np.clip(i, 0, len(mid) - 1)], np.nan)

    def after(x):
        i = np.searchsorted(ts, x, "left")
        ok = (i < len(ts)) & (ts[np.clip(i, 0, len(ts) - 1)] - x <= 2000)
        return np.where(ok, mid[np.clip(i, 0, len(mid) - 1)], np.nan), np.where(ok, ts[np.clip(i, 0, len(ts) - 1)], -1)

    lr = lambda a, b: np.log(a / b) * 1e4  # noqa: E731
    m0 = before(t)
    df["mid0"] = m0
    for s in (1, 5, 60):
        df[f"pre{s}"] = lr(m0, before(t - s * 1000))
    sec = pd.Series(mid, index=ts // 1000).groupby(level=0).agg(["last", "max", "min"])
    r1 = np.log(sec["last"]).diff() * 1e4
    rv = r1.rolling(300, min_periods=150).std()
    df["rv300"] = rv.reindex(t // 1000 - 1).to_numpy()   # 직전 완결 초까지
    for L in LATS:
        me, te = after(t + L)
        assert np.all((te < 0) | (te >= t + L)), "진입이 known_ts+L 보다 앞섰다"
        df[f"conc{L}"] = lr(me, m0)
        for h in CRITERIA["horizons_s"]:
            mx, _ = after(t + L + h * 1000)
            df[f"r{L}_{h}"] = lr(mx, me)
        if L == CRITERIA["lat_ms"]:
            df["entry_ts"] = te
            for h in (60, 300):   # 1초 고가·저가로 MFE/MAE (청산 초까지)
                hi, lo = [], []
                for e, tt in zip(me, te):
                    if not np.isfinite(e):
                        hi.append(np.nan); lo.append(np.nan); continue
                    g = sec.loc[tt // 1000: tt // 1000 + h]
                    hi.append(lr(g["max"].max(), e)); lo.append(lr(g["min"].min(), e))
                df[f"up{h}"] = hi; df[f"dn{h}"] = lo
    return df


def fill_share(df, tape: pd.DataFrame, bucket: float):
    """감소량 중 [known-1s, known] 초들에서 그 가격칸의 반대쪽 테이커 체결 몫. 칸이 틱보다 넓으면(ETH 10틱) 과대 = 보수적."""
    if not len(df):
        return df
    if not len(tape):
        df["fill"] = np.nan; return df
    key = tape.set_index(["ts_sec", "price_bin"])[["buy_qty", "sell_qty"]]
    d = key.to_dict("index")
    out = []
    for t, side, px, qr, q in zip(df.t, df.side, df.px, df.qref, df.q):
        b = int(round(px / bucket))
        tot = sum(d.get((s, b), {}).get("sell_qty" if side == 0 else "buy_qty", 0.0)
                  for s in range((t - 1000) // 1000, t // 1000 + 1))
        out.append(min(1.0, tot / max(qr - q, 1e-12)))
    df["fill"] = out
    return df


def hour_job(args):
    coin, path = args
    sym = f"{coin}USDT"
    hstart = T(Path(path).name[:13] + ":00Z")
    with _open(path) as f:
        ev, ct, cov = replay(iter(f), TICKS[coin])
    cov.update(hour=Path(path).name[:13])
    if not cov.get("t_first"):
        return pd.DataFrame(), pd.DataFrame(), cov
    btd = ROOT / "data/live/orderflow/bookticker" / sym
    names = [pd.to_datetime(hstart + k * 3_600_000, unit="ms").strftime("%Y-%m-%dT%H") for k in (-1, 0, 1)]
    bts = [p for n in names for p in glob.glob(str(btd / f"{n}.bt*"))]
    ts, mid = load_bt(bts)
    day = pd.to_datetime(hstart, unit="ms").strftime("%Y-%m-%d")
    tf = glob.glob(str(ROOT / f"data/lake/binance/tape/coin={coin}/date={day}/*.parquet"))
    tape = pd.concat([pd.read_parquet(p, columns=["ts_sec", "price_bin", "buy_qty", "sell_qty"]) for p in tf]) if tf else pd.DataFrame()
    if len(tape):
        tape = tape[(tape.ts_sec >= hstart // 1000 - 2) & (tape.ts_sec < hstart // 1000 + 3602)]
        tape = tape.groupby(["ts_sec", "price_bin"], as_index=False).sum()
    E = pd.DataFrame(ev)
    if len(E):
        E = E[E.share >= 0.5 * min(CRITERIA["wall_share_sens"])].reset_index(drop=True)   # 저장량 절약(분석 하한의 절반까지)
    E = fill_share(outcomes(E, ts, mid), tape, BUCKETS[coin])
    C = outcomes(pd.DataFrame(ct), ts, mid)
    cov.update(n_ev=len(E), n_ctrl=len(C), tape=bool(len(tape)))
    return E, C, cov


def run_replay(coin: str):
    lo, hi = (T(CRITERIA["split_eth"][0]), T(CRITERIA["split_eth"][2])) if coin == "ETH" else \
        (T(CRITERIA["multi_window"][0]), T(CRITERIA["multi_window"][1]))
    files = [f for f in sorted(glob.glob(str(ROOT / f"data/live/orderflow/depthdiff/{coin}USDT/*.jsonl*")))
             if lo <= T(Path(f).name[:13] + ":00Z") < hi]
    OUT.mkdir(parents=True, exist_ok=True)
    Es, Cs, covs = [], [], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for E, C, cov in ex.map(hour_job, [(coin, f) for f in files]):
            Es.append(E); Cs.append(C); covs.append(cov)
            print(cov, flush=True)
    pd.concat(Es).to_parquet(OUT / f"events_{coin}.parquet")
    pd.concat(Cs).to_parquet(OUT / f"controls_{coin}.parquet")
    (OUT / f"cover_{coin}.json").write_text(json.dumps(covs, indent=0))


# ── 분석 ─────────────────────────────────────────────────────────────────────
def select(E, sh, tl, kind="pull", dist=None):
    dist = dist or CRITERIA["dist_bp"]
    m = (E.share >= sh) & (E[f"age{KS[0]}"] >= tl) & ~E.crossed
    if kind == "pull":
        m &= (E.fill < CRITERIA["pull_fill_max"]) & E.dist_bp.between(*dist)
    elif kind == "eaten":
        m &= E.fill >= CRITERIA["eaten_fill_min"]
    S = E[m].sort_values("t")
    keep, last = [], {0: -1e18, 1: -1e18}
    for t, s in zip(S.t, S.side):
        ok = t - last[s] > CRITERIA["dedup_ms"]
        keep.append(ok)
        if ok: last[s] = t
    return S[np.array(keep, bool)] if len(S) else S


def ols_boot(df, y, xcols, coef="ask", unit="blk"):
    """y ~ const + ask + xcols, 블록마다 X'X·X'y 를 모아 블록 부트스트랩. 반환 (추정, lo, hi, n, 블록수)."""
    d = df.dropna(subset=[y] + xcols)
    if len(d) < 20 or d.side.nunique() < 2:
        return (np.nan, np.nan, np.nan, len(d), 0)
    X = np.column_stack([np.ones(len(d)), (d.side == 1).astype(float)] + [d[c].to_numpy() for c in xcols])
    Y = d[y].to_numpy()
    g = d[unit].to_numpy()
    ub, inv = np.unique(g, return_inverse=True)
    p = X.shape[1]
    XX = np.zeros((len(ub), p, p)); XY = np.zeros((len(ub), p))
    np.add.at(XX, inv, X[:, :, None] * X[:, None, :]); np.add.at(XY, inv, X * Y[:, None])
    est = np.linalg.lstsq(XX.sum(0), XY.sum(0), rcond=None)[0][1]
    rng = np.random.default_rng(CRITERIA["seed"])
    bs = []
    for _ in range(CRITERIA["boot"]):
        w = np.bincount(rng.integers(len(ub), size=len(ub)), minlength=len(ub)).astype(float)
        A = np.tensordot(w, XX, 1); b = w @ XY
        try:
            bs.append(np.linalg.solve(A, b)[1])
        except np.linalg.LinAlgError:
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (est, lo, hi, len(d), len(ub))


def ctrl_residual(S, C, col):
    """대조(5초마다 같은 쪽 가장 큰 서 있는 벽) 중 같은 쪽·거리·변동성·직전60초 이동 칸의 평균을 뺀다."""
    C = C.dropna(subset=[col, "rv300", "pre60"])
    vq = C.rv300.quantile([1 / 3, 2 / 3]).to_numpy(); pq = C.pre60.quantile([1 / 3, 2 / 3]).to_numpy()
    cell = lambda X: list(zip(X.side, np.digitize(X.dist_bp, [10, 20]),  # noqa: E731
                             np.digitize(X.rv300, vq), np.digitize(X.pre60, pq)))
    base = pd.Series(C[col].to_numpy(), index=pd.MultiIndex.from_tuples(cell(C))).groupby(level=[0, 1, 2, 3]).mean()
    b = base.reindex(pd.MultiIndex.from_tuples(cell(S))).to_numpy() if len(S) else np.array([])
    return S[col].to_numpy() - b


def fmt(r):
    return f"{r[0]:+7.2f} [{r[1]:+6.2f},{r[2]:+6.2f}] n={r[3]} blk={r[4]}" if np.isfinite(r[0]) else f"   n/a  n={r[3]}"


def analyze(coin, phase):
    E = pd.read_parquet(OUT / f"events_{coin}.parquet"); C = pd.read_parquet(OUT / f"controls_{coin}.parquet")
    a, b, c = (T(x) for x in CRITERIA["split_eth"]) if coin == "ETH" else (T(CRITERIA["multi_window"][0]),) * 2 + (T(CRITERIA["multi_window"][1]),)
    lo, hi = {"dev": (a, b), "holdout": (b, c), "all": (a if coin == "ETH" else b, c)}[phase]
    E = E[(E.t >= lo) & (E.t < hi)].copy(); C = C[(C.t >= lo) & (C.t < hi)].copy()
    for X in (E, C):
        X["blk"] = X.t // 3_600_000; X["day"] = X.t // 86_400_000
    K, TL, L, H = CRITERIA["wall_share"], CRITERIA["t_live_ms"], CRITERIA["lat_ms"], CRITERIA["main_h"]
    res = dict(coin=coin, phase=phase, hours=int(E.blk.nunique()), days=(hi - lo) / 86_400_000,
               n_raw_events=len(E), n_ctrl=len(C))
    print(f"\n=== {coin} {phase}  {pd.to_datetime(lo, unit='ms')} ~ {pd.to_datetime(hi, unit='ms')}  사건(원) {len(E)}  대조 {len(C)}")
    P = select(E, K, TL, "pull"); Ea = select(E, K, TL, "eaten")
    print(f"주 정의 철회 {len(P)}건 (매수벽 {int((P.side == 0).sum())} · 매도벽 {int((P.side == 1).sum())}) · 소진 {len(Ea)} · 하루 {len(P) / res['days']:.1f}건")
    print(f"분류 분포(k={K},T={TL}, 교차 제외 전): fill<.2 {((E.fill < .2)).mean():.2%}  fill≥.8 {((E.fill >= .8)).mean():.2%}  교차 {E.crossed.mean():.2%}  fill NaN {E.fill.isna().mean():.2%}")
    tab = {}
    print(f"\nS = 매도벽철회 − 매수벽철회 (bp), L={L}ms. 양수 = H_main(지지/저항이 가짜)")
    print("구간               철회 원S                        소진 대조                       철회−무작위대조 잔차               직전이동 통제")
    rows = [("직전1초", "pre1"), ("지연0~L", f"conc{L}")] + [(f"+{h}s", f"r{L}_{h}") for h in CRITERIA["horizons_s"]]
    for name, col in rows:
        r_raw = ols_boot(P, col, []); r_eat = ols_boot(Ea, col, [])
        P2 = P.copy(); P2["res"] = ctrl_residual(P, C, col) if col.startswith("r") else np.nan
        r_res = ols_boot(P2, "res", []) if col.startswith("r") else (np.nan,) * 3 + (0, 0)
        r_ctl = ols_boot(P, col, ["pre1", "pre5", "pre60"]) if col.startswith("r") else (np.nan,) * 3 + (0, 0)
        tab[name] = dict(raw=r_raw, eaten=r_eat, resid=r_res, premove=r_ctl)
        print(f"{name:8s} {fmt(r_raw)} | {fmt(r_eat)} | {fmt(r_res)} | {fmt(r_ctl)}")
    res["table"] = {k: {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in tab.items()}
    print("\n민감도 (주 지평, 원 S)")
    sens = {}
    for k in (K,) + CRITERIA["wall_share_sens"]:
        for tl in (TL,) + CRITERIA["t_live_sens"]:
            for lat in (L,) + CRITERIA["lat_sens"]:
                r = ols_boot(select(E, k, tl, "pull"), f"r{lat}_{H}", [])
                sens[f"sh{k}_T{tl}_L{lat}"] = [float(x) for x in r]
                print(f"  share={k:.2f} T={tl:5d} L={lat:4d}: {fmt(r)}")
    res["sens"] = sens
    print("\n거리 칸별 (주 정의, +30s 원 S): ", {f"{d0}-{d1}": fmt(ols_boot(select(E, K, TL, 'pull', (d0, d1)), f'r{L}_{H}', []))
                                         for d0, d1 in ((0, 5), (5, 10), (10, 20), (20, 30))})
    r_main = tab[f"+{H}s"]
    sign = lambda r: 0 if not (np.isfinite(r[1]) and (r[1] > 0 or r[2] < 0)) else (1 if r[1] > 0 else -1)  # noqa: E731
    if phase == "dev":
        d = sign(r_main["raw"]) or 1
        res["frozen_direction"] = d
        print(f"\nDEV 판정: 원 S {fmt(r_main['raw'])} -> HOLDOUT 방향 고정 = {'H_main(+)' if d > 0 else 'H_alt(−)'}"
              f"{'' if sign(r_main['raw']) else ' (DEV 유의하지 않아 사전 지정 H_main)'}")
    else:
        dpath = OUT / f"result_{coin}_dev.json"
        d = json.loads(dpath.read_text())["frozen_direction"] if dpath.exists() else 1
        passed = all(sign(r_main[x]) == d for x in ("raw", "resid", "premove"))
        res["direction"] = d; res["pass"] = bool(passed)
        print(f"\n{phase.upper()} 판정 (방향 {'+' if d > 0 else '−'}): 원 {sign(r_main['raw'])} · 잔차 {sign(r_main['resid'])} · 직전이동 {sign(r_main['premove'])} -> {'통과' if passed else '불통과'}")
        if passed:
            res["trade"] = trade(P, d)
    (OUT / f"result_{coin}_{phase}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def trade(P, d):
    """철회 사건마다 테이커 진입(L 뒤), h 초 보유. 방향 d: 매도벽철회면 d 쪽, 매수벽철회면 −d 쪽."""
    out = {}
    L = CRITERIA["lat_ms"]
    sgn = np.where(P.side == 1, d, -d)
    print("\n매매화 (테이커 왕복 8bp · 메이커 1.41bp 참고), 일 클러스터 부트스트랩")
    for h in CRITERIA["horizons_s"]:
        g = pd.DataFrame({"day": P.day, "r": sgn * P[f"r{L}_{h}"]}).dropna()
        for name, cost in (("taker", CRITERIA["taker_rt_bp"]), ("maker", CRITERIA["maker_rt_bp"])):
            g["net"] = g.r - cost
            per = g.groupby("day").net.agg(["sum", "count"])
            rng = np.random.default_rng(CRITERIA["seed"])
            bs = [per.iloc[rng.integers(len(per), size=len(per))].pipe(lambda x: x["sum"].sum() / x["count"].sum())
                  for _ in range(CRITERIA["boot"])]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out[f"{name}_{h}"] = dict(net_per_trade=float(g.net.mean()), lo=float(lo), hi=float(hi), per_day=float(per["count"].mean()),
                                     net_per_day=float(per["sum"].mean()), days=len(per))
            print(f"  h={h:3d}s {name:5s}: 순 {g.net.mean():+6.2f}bp/건 [{lo:+6.2f},{hi:+6.2f}] · 하루 {per['count'].mean():.0f}건 · 하루 {per['sum'].mean():+.0f}bp")
    for h in (60, 300):
        adv = np.where(sgn > 0, -P[f"dn{h}"], P[f"up{h}"])
        adv = adv[np.isfinite(adv)]
        q = np.percentile(adv, [50, 90, 99]) if len(adv) else [np.nan] * 3
        out[f"mae{h}"] = dict(p50=float(q[0]), p90=float(q[1]), p99=float(q[2]), ge100=float((adv >= 100).mean()), ge250=float((adv >= 250).mean()))
        print(f"  MAE {h}s: p50 {q[0]:.1f} · p90 {q[1]:.1f} · p99 {q[2]:.1f}bp · ≥100bp {(adv >= 100).mean():.2%} · ≥250bp(20배 증거금 절반) {(adv >= 250).mean():.2%}")
    return out


# ── 자체점검 ─────────────────────────────────────────────────────────────────
def selftest():
    tick = 100
    snap = {"lastUpdateId": 10,
            "bids": [[f"{100 - j * 0.01:.2f}", "1"] for j in range(60)] + [["99.80", "100"]],   # 매수벽 20bp 아래
            "asks": [[f"{100.01 + j * 0.01:.2f}", "1"] for j in range(60)] + [["100.20", "100"]]}  # 매도벽 19bp 위
    lines = [json.dumps({"_snapshot": snap})]
    u = 10
    for n in range(80):                       # 8초, 100ms
        t = 1_000_000 + n * 100
        b, a = [["99.99", str(1 + n % 2)]], []
        if n == 50: b.append(["99.80", "0"])  # 5.0초에 매수벽 철회 (4초 이상 서 있었다)
        if n == 60: a.append(["100.20", "5"])  # 6.0초에 매도벽 95% 감소 (테이프로 소진 처리)
        lines.append(json.dumps({"e": "depthUpdate", "T": t, "U": u if n == 0 else u + 1, "u": u + 1, "pu": u, "b": b, "a": a}))
        u += 1
    ev, ct, cov = replay(iter(lines), tick, idx_max=20_000)
    assert cov["status"] == "ok", cov
    assert len(ev) == 2 and [e["side"] for e in ev] == [0, 1], ev
    assert ev[0]["t"] == 1_005_000 and abs(ev[0]["px"] - 99.80) < 1e-9 and ev[0]["age100"] >= 3000 and not ev[0]["crossed"]
    assert 15 < ev[0]["dist_bp"] < 25 and ev[0]["ratio"] >= 20
    assert len(ct) > 0 and all(c["age20"] >= 1000 for c in ct)
    # 끊김: pu 불연속이면 멈춘다
    bad = lines[:10] + [json.dumps({"e": "depthUpdate", "T": 2_000_000, "U": 999, "u": 1000, "pu": 998, "b": [], "a": []})] + lines[10:]
    assert replay(iter(bad), tick, idx_max=20_000)[2]["status"] == "gap"
    # 철회/소진 분류: 매도벽 95 감소 중 95 체결 -> 소진, 매수벽 체결 0 -> 철회
    E = pd.DataFrame(ev)
    tape = pd.DataFrame({"ts_sec": [1005, 1006], "price_bin": [1002, 1002], "buy_qty": [0.0, 95.0], "sell_qty": [0.0, 0.0]})
    E = fill_share(E, tape, 0.1)
    assert E.fill.iloc[0] == 0.0 and E.fill.iloc[1] >= 0.8, E.fill
    # 시점 경계: 진입은 known_ts+L 이상, 직전값은 known_ts 이하
    ts = np.arange(999_000, 1_400_000, 37); mid = 100 + (ts - 999_000) * 1e-6
    O = outcomes(E.copy(), ts, mid)
    assert (O.entry_ts >= O.t + CRITERIA["lat_ms"]).all()
    assert (O.conc500 > 0).all() and (O.pre1 > 0).all()
    # S 부트스트랩: 매도벽 r=+2, 매수벽 r=-2 -> S=+4
    D = pd.DataFrame({"side": [0, 1] * 50, "r": [-2.0, 2.0] * 50, "blk": np.repeat(np.arange(10), 10)})
    r = ols_boot(D, "r", [])
    assert abs(r[0] - 4) < 1e-9 and r[1] > 3.9, r
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--replay")
    ap.add_argument("--analyze")
    ap.add_argument("--phase", default="dev", choices=["dev", "holdout", "all"])
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.replay:
        run_replay(a.replay)
    elif a.analyze:
        analyze(a.analyze, a.phase)
    else:
        ap.print_help(); sys.exit(1)
