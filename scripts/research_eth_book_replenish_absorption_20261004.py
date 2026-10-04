"""«먹혀도 다시 채워지는 호가(재보충·빙산)» 흡수 -- 전체북 사전등록 검정 (2026-10-04).

사용자: «흡수 전략은 왜 계속 불합격인거지? 이렇게나 경우가 많은데». 지금까지의 흡수 검정은 1초 체결 + 최우선 호가 요약뿐이었다.
여기서는 전체북(depthdiff)으로 «같은 가격칸이 W초 동안 보이는 잔량의 R배 넘게 먹혔는데도 잔량이 남아 있는» 순간을 잡는다.
원천·재구성·bookTicker·부트스트랩은 scripts/research_eth_wall_pull_jump_20261004.py(wp) 를 그대로 쓴다.
실행:
  python scripts/research_eth_book_replenish_absorption_20261004.py --selftest
  (서버) nice -n 19 python scripts/research_eth_book_replenish_absorption_20261004.py --replay ETH  -> tmp/book_replenish_20261004/
  python scripts/research_eth_book_replenish_absorption_20261004.py --analyze ETH --phase dev       (방향 고정)
  python scripts/research_eth_book_replenish_absorption_20261004.py --analyze ETH --phase holdout   (한 번만)
  python scripts/research_eth_book_replenish_absorption_20261004.py --analyze BTC --phase all       (다코인 재현)
단위: 레벨 L = 체결 테이프 가격칸(round(price/bucket); ETH $0.1 = 10틱). 잔량도 같은 칸으로 합친다(테이프와 같은 해상도).
시점 경계: 창 = 테이프 초 s-W+1..s, 북은 각 초 끝 상태. known_ts = (s+1)*1000ms(그 초가 끝난 뒤). 진입 = known_ts+1000ms 이후 첫 bookTicker.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_eth_wall_pull_jump_20261004 as wp  # noqa: E402

ROOT = wp.ROOT
OUT = ROOT / "tmp/book_replenish_20261004"
T = wp.T

# ── 사전등록 (결과 보기 전에 고정, 2026-10-04) ───────────────────────────────────
CRITERIA = dict(
    W_main=30, W_sens=(10, 60),        # 창 길이(초)
    R_main=2, R_sens=(1, 3),           # 창 체결량 ≥ R × 창 시작 잔량 D0
    keep=0.5,                          # 재보충: 창 끝 잔량 D1 ≥ keep·D0
    deplete=0.2,                       # C1 소진: 같은 체결 조건인데 D1 < 0.2·D0
    zone_bp=30,                        # 창 시작 mid 에서 레벨 쪽으로 0~30bp (mid 가 든 칸 포함)
    vmin_q=0.75, quiet_q=0.50,         # 체결 하한/조용함 상한 = 구역 안 «V>0·안 뚫림» 칸의 V 분위 (ETH=DEV, 다른 코인=창 앞 60%의 표본 시각)
    # 안 뚫림(매도 레벨 b): 창 안 매 초 best bid 칸 ≤ b 이고 매수 체결 칸 ≤ b. 매수 레벨은 거울.
    # C2 조용함: 0 < V ≤ quiet 분위 · D1 ≥ keep·D0 · 안 뚫림.  C3: 층화(쪽 × 거리 3칸 × rv300 3분위 × 직전60초 «레벨 쪽» 이동 3분위).
    dedup_s=10,                        # 같은 (W, 분류, R, 쪽) 사건 10초 안 연쇄는 V 최대 하나(첫 초)
    lat_ms=1000,                       # 진입 지연 1초 (OBS=0 금지)
    horizons_s=(30, 60, 300, 900), main_h=300,
    # 가설: H_hold(재보충 레벨은 버틴다) -- 5분 안 돌파율 재보충 < C2 이고 S = r(매도 재보충) − r(매수 재보충) < 0.
    #       H_break(09-29 처럼 더 뚫린다) -- 돌파율 재보충 > C2 이고 S > 0.
    # 돌파 = 1초 bookTicker mid 가 레벨 칸 바깥 경계를 넘음(known_ts 초 ~ 진입+h).
    # DEV: 층화 돌파율 차(재보충 − C2, 5분) 1h 블록 부트스트랩 95% CI 가 0 을 배제하면 그 방향을 고정, 아니면 H_hold 로 검정.
    # HOLDOUT 통과 = 고정 방향으로 (a) 층화 돌파율 차(vs C2) CI 0 배제 (b) S(직전 1초·W·60초 이동 통제) CI 0 배제, 둘 다.
    #   H_hold 면 (c) 층화 돌파율 차(vs C1) < 0 도 보고(판정 보조).
    # 매매화(통과 시만): 레벨 반대쪽(H_hold)/레벨 쪽(H_break) 진입, 메이커 1.41·테이커 8bp, 일 클러스터 CI, 레벨 뒤 손절 변형.
    split_eth=("2026-09-15T16Z", "2026-09-26T16Z", "2026-10-04T00Z"),  # 테이프 시작(09-15 15:51) 이후 · DEV 60% / HOLDOUT 40%
    multi_window=("2026-09-26T09Z", "2026-10-04T00Z"),
    taker_rt_bp=8.0, maker_rt_bp=1.41, boot=2000, seed=20261004,
)
WS = (CRITERIA["W_main"],) + CRITERIA["W_sens"]
RS = (CRITERIA["R_main"],) + CRITERIA["R_sens"]


# ── 북 재구성: 초마다 mid 주변 칸별 잔량 ─────────────────────────────────────────
def book_seconds(lines, tick: int, bucket: float, idx_max: int = wp.IDX_MAX):
    """초 끝 상태를 칸 단위로. 반환 (dict of arrays, status). 끊기면 그 직전 완결 초까지만(wp.replay 와 같은 규약)."""
    import json as _j
    r = int(round(tick * bucket))                      # 칸당 틱 수
    bid = np.zeros(idx_max); ask = np.zeros(idx_max)
    first = _j.loads(next(lines))
    snap = first.get("_snapshot") if isinstance(first, dict) else None
    if not snap:
        return None, "no_snapshot"
    for side, arr in (("bids", bid), ("asks", ask)):
        for p, q in snap[side]:
            i = int(round(float(p) * tick))
            if i < idx_max: arr[i] = float(q)
    last_u, synced = snap["lastUpdateId"], False
    bb, ba = int(np.flatnonzero(bid)[-1]), int(np.flatnonzero(ask)[0])
    K = math.ceil((CRITERIA["zone_bp"] + 5) * 1e-4 * (bb / tick) / bucket)
    rows = []
    cur = None; bbmax = bamin = None

    def snap_sec(sec):
        mid_i = (bb + ba) / 2
        c = int(np.round(mid_i / r))
        lo_t, hi_t = max((c - K) * r - r, 0), (c + K) * r + r + 1
        bins = np.round(np.arange(lo_t, hi_t) / r).astype(np.int64) - (c - K)
        ok = (bins >= 0) & (bins <= 2 * K)
        rows.append((sec, c, np.bincount(bins[ok], bid[lo_t:hi_t][ok], 2 * K + 1).astype(np.float32),
                     np.bincount(bins[ok], ask[lo_t:hi_t][ok], 2 * K + 1).astype(np.float32),
                     int(np.round(bbmax / r)), int(np.round(bamin / r)), mid_i / tick))

    status = "ok"
    for line in lines:
        try:
            m = _j.loads(line)
        except (ValueError, UnicodeDecodeError):
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
        last_u, sec = m["u"], int(m["T"]) // 1000
        if cur is None:
            cur, bbmax, bamin = sec, bb, ba
        elif sec > cur:
            snap_sec(cur)
            for s in range(cur + 1, sec):           # 갱신 없는 초 = 상태 그대로
                bbmax, bamin = bb, ba; snap_sec(s)
            cur, bbmax, bamin = sec, bb, ba
        for key, arr in (("b", bid), ("a", ask)):
            for p, q in m[key]:
                i = int(round(float(p) * tick))
                if i < idx_max: arr[i] = float(q)
        bb, ba = wp._best(bid, bb, True), wp._best(ask, ba, False)
        if bb < 0 or ba < 0:
            status = "empty_book"; break
        bbmax, bamin = max(bbmax, bb), min(bamin, ba)
    if status == "ok" and cur is not None:
        snap_sec(cur)                                   # 파일 끝 = 시각 끝
    if not rows:
        return None, status
    sec, c, bv, av, bbx, ban, mid = map(np.array, zip(*rows))
    return dict(sec=sec, c=c, bv=np.stack(bv), av=np.stack(av), bbmax=bbx, bamin=ban, mid=mid, K=K), status


# ── 사건 검출 ────────────────────────────────────────────────────────────────
def detect(B: dict, tape: pd.DataFrame, bucket: float, vq: dict, probe: bool = False, rng=None):
    """B: book_seconds 결과(초 연속). vq: {W: (vmin, vquiet)}. 반환 사건 DataFrame (probe 면 {W: V 표본})."""
    sec, c, K = B["sec"], B["c"], B["K"]
    n = len(sec)
    base = int(c.min()) - K
    nb = int(c.max()) + K + 1 - base
    babs = base + np.arange(nb)
    cols = (c - K - base)[:, None] + np.arange(2 * K + 1)
    D = {}
    for side, key in ((0, "bv"), (1, "av")):
        A = np.full((n, nb), np.nan, np.float32)
        A[np.arange(n)[:, None], cols] = B[key]
        D[side] = A
    TV = {0: np.zeros((n, nb)), 1: np.zeros((n, nb))}   # 0: 매수 레벨을 때린 매도 체결, 1: 매도 레벨을 때린 매수 체결
    hi_buy = np.full(n, -np.inf); lo_sell = np.full(n, np.inf)
    if len(tape):
        t = tape[(tape.ts_sec >= sec[0]) & (tape.ts_sec <= sec[-1])]
        si = (t.ts_sec.to_numpy() - sec[0]).astype(int); bi = t.price_bin.to_numpy().astype(np.int64)
        b_, s_ = t.buy_qty.to_numpy(), t.sell_qty.to_numpy()
        np.maximum.at(hi_buy, si[b_ > 0], bi[b_ > 0]); np.minimum.at(lo_sell, si[s_ > 0], bi[s_ > 0])
        ok = (bi >= base) & (bi < base + nb)
        np.add.at(TV[1], (si[ok], bi[ok] - base), b_[ok]); np.add.at(TV[0], (si[ok], bi[ok] - base), s_[ok])
    mid = B["mid"]
    midbin = np.round(mid / bucket).astype(np.int64)
    up_lim = np.maximum(B["bbmax"], hi_buy)       # 매도 레벨 b 를 넘었나: 이 값 > b
    dn_lim = np.minimum(B["bamin"], lo_sell)      # 매수 레벨 b 를 넘었나: 이 값 < b
    out, samples = [], {}
    for W in WS:
        if n <= W:
            continue
        j = np.arange(W, n)                       # 창 = 초 j-W+1..j, D0 = 초 j-W 끝, D1 = 초 j 끝
        upW = pd.Series(up_lim).rolling(W).max().to_numpy()[j]
        dnW = pd.Series(dn_lim).rolling(W).min().to_numpy()[j]
        m0 = mid[j - W]
        for side in (0, 1):
            cs = np.vstack([np.zeros((1, nb)), np.cumsum(TV[side], 0)])
            V = cs[j + 1] - cs[j + 1 - W]
            D0 = D[side][j - W]; D1 = D[side][j]
            px = babs * bucket
            if side == 1:
                dist = (px[None, :] - m0[:, None]) / m0[:, None] * 1e4
                zone = (babs[None, :] >= midbin[j - W][:, None]) & (dist <= CRITERIA["zone_bp"])
                nocross = upW[:, None] <= babs[None, :]
            else:
                dist = (m0[:, None] - px[None, :]) / m0[:, None] * 1e4
                zone = (babs[None, :] <= midbin[j - W][:, None]) & (dist <= CRITERIA["zone_bp"])
                nocross = dnW[:, None] >= babs[None, :]
            base_ok = zone & nocross & (np.nan_to_num(D0) > 0) & np.isfinite(D1) & (V > 0)
            if probe:
                v = V[base_ok]
                samples.setdefault(W, []).append(rng.choice(v, min(len(v), 20000), replace=False) if len(v) else v)
                continue
            vmin, vquiet = vq[W]
            D1z = np.nan_to_num(D1)
            classes = [("quiet", 0, base_ok & (V <= vquiet) & (D1z >= CRITERIA["keep"] * D0))]
            for R in RS:
                hit = base_ok & (V >= R * D0) & (V >= vmin)
                classes += [("rep", R, hit & (D1z >= CRITERIA["keep"] * D0)), ("dep", R, hit & (D1z < CRITERIA["deplete"] * D0))]
            for cls, R, M in classes:
                on = M & ~np.vstack([np.zeros((1, nb), bool), M[:-1]])    # 처음 만족한 초(직전 초엔 아님)
                if W < n and M.shape[0]:
                    on[0] = False                                       # 첫 창은 직전 상태를 모른다
                rr, cc = np.nonzero(on)
                if not len(rr):
                    continue
                E = pd.DataFrame(dict(s=sec[j[rr]], side=side, bin=babs[cc], D0=D0[rr, cc], D1=D1[rr, cc], V=V[rr, cc],
                                      dist_bp=dist[rr, cc], W=W, cls=cls, R=R))
                E = E.sort_values(["s", "V"], ascending=[True, False])
                keep, last = [], -10 ** 12
                for s_ in E.s.to_numpy():
                    k = s_ - last >= CRITERIA["dedup_s"]
                    keep.append(k)
                    if k: last = s_
                out.append(E[np.array(keep)])
    if probe:
        return {W: np.concatenate(v) for W, v in samples.items()}
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ── 결과 (bookTicker 1초) ────────────────────────────────────────────────────
def attach_outcomes(E: pd.DataFrame, ts, mid, bucket: float) -> pd.DataFrame:
    if not len(E):
        return E
    E = E.copy()
    E["known"] = (E.s.to_numpy() + 1) * 1000
    t = E.known.to_numpy()
    lr = lambda a, b: np.log(a / b) * 1e4  # noqa: E731

    def before(x):
        i = np.searchsorted(ts, x, "right") - 1
        ok = (i >= 0) & (x - ts[np.clip(i, 0, None)] <= 2000)
        return np.where(ok, mid[np.clip(i, 0, len(mid) - 1)], np.nan)

    def after(x):
        i = np.clip(np.searchsorted(ts, x, "left"), 0, len(ts) - 1)
        ok = (ts[i] >= x) & (ts[i] - x <= 2000)
        return np.where(ok, mid[i], np.nan), np.where(ok, ts[i], -1)

    g = pd.Series(mid, index=ts // 1000).groupby(level=0).agg(["last", "max", "min"])
    s0 = int(g.index.min()); full = np.arange(s0, int(g.index.max()) + 1)
    g = g.reindex(full); g["last"] = g["last"].ffill(); g["max"] = g["max"].fillna(g["last"]); g["min"] = g["min"].fillna(g["last"])
    smax, smin = g["max"].to_numpy(), g["min"].to_numpy()
    r1 = np.log(g["last"]).diff() * 1e4
    rv = r1.rolling(300, min_periods=150).std().to_numpy()
    m0 = before(t)
    E["pre1"] = lr(m0, before(t - 1000)); E["pre60"] = lr(m0, before(t - 60_000))
    E["preW"] = lr(m0, before(t - E.W.to_numpy() * 1000))
    ki = t // 1000 - 1 - s0
    E["rv300"] = np.where((ki >= 0) & (ki < len(rv)), rv[np.clip(ki, 0, len(rv) - 1)], np.nan)
    me, te = after(t + CRITERIA["lat_ms"])
    assert np.all((te < 0) | (te >= t + CRITERIA["lat_ms"])), "진입이 known_ts+1초 보다 앞섰다"
    E["entry_ts"] = te; E["entry"] = me; E["conc"] = lr(me, m0)
    for h in CRITERIA["horizons_s"]:
        mx, tx = after(te + h * 1000)
        E[f"r{h}"] = np.where(te > 0, lr(mx, me), np.nan)
    edge = np.where(E.side == 1, (E.bin + 0.5) * bucket, (E.bin - 0.5) * bucket)
    E["edge"] = edge
    sgn_up = (E.side == 1).to_numpy()
    res = {f"brk{h}": [] for h in CRITERIA["horizons_s"]}
    res.update({k: [] for k in ("x_from_entry", "up300", "dn300", "up900", "dn900")})
    for k_sec, e_ts, e_px, ed, up in zip(t // 1000, te, me, edge, sgn_up):
        a = k_sec - s0
        if e_ts < 0 or not np.isfinite(e_px) or a < 0:
            for v in res.values(): v.append(np.nan)
            continue
        es = e_ts // 1000 - s0
        end = es + max(CRITERIA["horizons_s"])
        seg = smax[a:end + 1] if up else smin[a:end + 1]
        crossed = seg > ed if up else seg < ed
        first = int(np.argmax(crossed)) if crossed.any() else 10 ** 9
        for h in CRITERIA["horizons_s"]:
            hb = es + h - a
            res[f"brk{h}"].append(float(first <= hb) if a + hb < len(smax) else np.nan)
        res["x_from_entry"].append(first - (es - a) if first < 10 ** 9 else np.nan)   # 진입 초 기준 돌파까지 초(음수 = 진입 전)
        for h in (300, 900):
            res[f"up{h}"].append(lr(smax[es:es + h + 1].max(), e_px)); res[f"dn{h}"].append(lr(smin[es:es + h + 1].min(), e_px))
    for k, v in res.items():
        E[k] = v
    return E


def hour_job(args):
    coin, path, mode, vq = args
    sym = f"{coin}USDT"
    hstart = T(Path(path).name[:13] + ":00Z")
    bucket = wp.BUCKETS[coin]
    with wp._open(path) as f:
        B, status = book_seconds(iter(f), wp.TICKS[coin], bucket)
    cov = dict(hour=Path(path).name[:13], status=status, secs=0 if B is None else len(B["sec"]))
    if B is None:
        return None, cov
    day = pd.to_datetime(hstart, unit="ms").strftime("%Y-%m-%d")
    tf = glob.glob(str(ROOT / f"data/lake/binance/tape/coin={coin}/date={day}/*.parquet"))
    tape = pd.concat([pd.read_parquet(p, columns=["ts_sec", "price_bin", "buy_qty", "sell_qty"]) for p in tf]) if tf else pd.DataFrame()
    if len(tape):
        tape = tape[(tape.ts_sec >= hstart // 1000 - 2) & (tape.ts_sec < hstart // 1000 + 3602)]
        tape = tape.groupby(["ts_sec", "price_bin"], as_index=False).sum()
    cov["tape"] = bool(len(tape))
    if not len(tape):
        return None, cov                       # 테이프 없는 시각은 체결량을 모른다 -> 제외
    if mode == "probe":
        return detect(B, tape, bucket, {}, probe=True, rng=np.random.default_rng(hstart % 2 ** 32)), cov
    E = detect(B, tape, bucket, vq)
    btd = ROOT / "data/live/orderflow/bookticker" / sym
    names = [pd.to_datetime(hstart + k * 3_600_000, unit="ms").strftime("%Y-%m-%dT%H") for k in (-1, 0, 1)]
    ts, mid = wp.load_bt([p for nm in names for p in glob.glob(str(btd / f"{nm}.bt*"))])
    if len(E) and len(ts):
        E = attach_outcomes(E, ts, mid, bucket)
    cov["n_ev"] = len(E)
    return E, cov


def windows(coin):
    if coin == "ETH":
        a, b, c = (T(x) for x in CRITERIA["split_eth"])
    else:
        a, c = (T(x) for x in CRITERIA["multi_window"]); b = a + int(0.6 * (c - a)) // 3_600_000 * 3_600_000
    return a, b, c


def run_replay(coin: str):
    a, b, c = windows(coin)
    files = [f for f in sorted(glob.glob(str(ROOT / f"data/live/orderflow/depthdiff/{coin}USDT/*.jsonl*")))
             if a <= T(Path(f).name[:13] + ":00Z") < c]
    OUT.mkdir(parents=True, exist_ok=True)
    vpath = OUT / f"vmin_{coin}.json"
    if not vpath.exists():                                 # 체결 하한 = DEV(앞 60%) 4시각마다 표본의 분위 -- 결과 열어보기 전
        dev = [f for f in files if T(Path(f).name[:13] + ":00Z") < b][::4]
        S = {W: [] for W in WS}
        with ProcessPoolExecutor(max_workers=6) as ex:
            for smp, cov in ex.map(hour_job, [(coin, f, "probe", None) for f in dev]):
                print("probe", cov, flush=True)
                for W, v in (smp or {}).items(): S[W].append(v)
        vq = {W: [float(np.quantile(np.concatenate(S[W]), CRITERIA["vmin_q"])), float(np.quantile(np.concatenate(S[W]), CRITERIA["quiet_q"]))]
              for W in WS}
        vpath.write_text(json.dumps(dict(vq=vq, probe_hours=len(dev), n={W: int(sum(len(x) for x in S[W])) for W in WS}), indent=1))
    vq = {int(k): v for k, v in json.loads(vpath.read_text())["vq"].items()}
    print("vq", vq, flush=True)
    Es, covs = [], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for E, cov in ex.map(hour_job, [(coin, f, "replay", vq) for f in files]):
            covs.append(cov); print(cov, flush=True)
            if E is not None and len(E): Es.append(E)
    pd.concat(Es, ignore_index=True).to_parquet(OUT / f"events_{coin}.parquet")
    (OUT / f"cover_{coin}.json").write_text(json.dumps(covs, indent=0))


# ── 분석 ─────────────────────────────────────────────────────────────────────
def strat_diff(X, Y, y):
    """돌파율 차(X − Y), C3 층화: 쪽 × 거리 3칸 × rv300 3분위 × 직전60초 레벨쪽 이동 3분위 더미. wp.ols_boot 의 'side' 자리에 처치 표시."""
    d = pd.concat([X.assign(treat=1), Y.assign(treat=0)], ignore_index=True).dropna(subset=[y, "rv300", "pre60"])
    if not len(d):
        return (np.nan,) * 3 + (0, 0)
    tw = np.where(d.side == 1, d.pre60, -d.pre60)
    cell = (d.side.astype(str) + np.digitize(d.dist_bp, [5, 15]).astype(str)
            + pd.qcut(d.rv300, 3, labels=False, duplicates="drop").astype(str) + pd.qcut(tw, 3, labels=False, duplicates="drop").astype(str))
    dm = pd.get_dummies(cell, prefix="c", drop_first=True).astype(float)
    d = pd.concat([d[[y, "blk", "treat"]].rename(columns={"treat": "side"}), dm], axis=1)
    return wp.ols_boot(d, y, list(dm.columns))


def raw_diff(X, Y, y):
    d = pd.concat([X.assign(treat=1), Y.assign(treat=0)], ignore_index=True)[[y, "blk", "treat"]].rename(columns={"treat": "side"})
    return wp.ols_boot(d, y, [])


def pick(E, W, R):
    return (E[(E.W == W) & (E.cls == "rep") & (E.R == R)], E[(E.W == W) & (E.cls == "dep") & (E.R == R)],
            E[(E.W == W) & (E.cls == "quiet")])


def sign(r):
    return 0 if not (np.isfinite(r[1]) and (r[1] > 0 or r[2] < 0)) else (1 if r[1] > 0 else -1)


def pct(r):
    return f"{100 * r[0]:+6.1f}pp [{100 * r[1]:+5.1f},{100 * r[2]:+5.1f}]" if np.isfinite(r[0]) else "   n/a"


def analyze(coin, phase):
    E = pd.read_parquet(OUT / f"events_{coin}.parquet")
    a, b, c = windows(coin)
    lo, hi = {"dev": (a, b), "holdout": (b, c), "all": (a, c)}[phase]
    E = E[(E.known >= lo) & (E.known < hi)].copy()
    E["blk"] = E.known // 3_600_000; E["day"] = E.known // 86_400_000
    days = E.blk.nunique() / 24
    W, R, H = CRITERIA["W_main"], CRITERIA["R_main"], CRITERIA["main_h"]
    res = dict(coin=coin, phase=phase, hours=int(E.blk.nunique()), vq=json.loads((OUT / f"vmin_{coin}.json").read_text()))
    print(f"\n=== {coin} {phase}  {pd.to_datetime(lo, unit='ms')} ~ {pd.to_datetime(hi, unit='ms')}  유효 {E.blk.nunique()}시간({days:.1f}일)")
    print("vq(체결 하한, 조용함 상한):", res["vq"]["vq"])
    print("사건 수(하루당)  W×분류×R:")
    cnt = E.groupby(["W", "cls", "R"]).size()
    for k, v in cnt.items():
        print(f"  W={k[0]:2d} {k[1]:5s} R={k[2]}: {v:7d} ({v / days:7.1f}/일)  매도레벨 {int(((E.W == k[0]) & (E.cls == k[1]) & (E.R == k[2]) & (E.side == 1)).sum())}")
    res["counts"] = {f"W{k[0]}_{k[1]}_R{k[2]}": int(v) for k, v in cnt.items()}
    rep, c1, c2 = pick(E, W, R)
    print(f"\n주 정의 W={W} R={R}: 재보충 {len(rep)} · C1 소진 {len(c1)} · C2 조용 {len(c2)}")
    print("돌파율(평균)  지평:  재보충 / C1 / C2 | 재보충−C2 원 | 재보충−C2 층화 | 재보충−C1 원 | 재보충−C1 층화")
    tab = {}
    for h in CRITERIA["horizons_s"]:
        y = f"brk{h}"
        r = dict(rate=[float(X[y].mean()) for X in (rep, c1, c2)], d2=raw_diff(rep, c2, y), d2s=strat_diff(rep, c2, y),
                 d1=raw_diff(rep, c1, y), d1s=strat_diff(rep, c1, y))
        tab[f"brk{h}"] = r
        print(f"  {h:4d}s: {r['rate'][0]:.3f} / {r['rate'][1]:.3f} / {r['rate'][2]:.3f} | {pct(r['d2'])} | {pct(r['d2s'])} | {pct(r['d1'])} | {pct(r['d1s'])}")
    print("\nS = 매도레벨 − 매수레벨 (bp). H_hold 면 재보충 S<0.  재보충 원 | 재보충 직전이동 통제 | C1 원 | C2 원")
    for name, col in [("직전1초", "pre1"), (f"창 {W}s", "preW"), ("지연1초", "conc")] + [(f"+{h}s", f"r{h}") for h in CRITERIA["horizons_s"]]:
        ctl = col.startswith("r")
        r = dict(raw=wp.ols_boot(rep, col, []), ctl=wp.ols_boot(rep, col, ["pre1", "preW", "pre60"]) if ctl else (np.nan,) * 3 + (0, 0),
                 c1=wp.ols_boot(c1, col, []), c2=wp.ols_boot(c2, col, []))
        tab[name] = r
        print(f"  {name:8s} {wp.fmt(r['raw'])} | {wp.fmt(r['ctl'])} | {wp.fmt(r['c1'])} | {wp.fmt(r['c2'])}")
    res["table"] = {k: {kk: [float(x) for x in vv] if isinstance(vv, (tuple, list)) else vv for kk, vv in v.items()} for k, v in tab.items()}
    print(f"\n민감도 (지평 {H}s): W × R → 돌파율 차 재보충−C2 층화 · S 원")
    sens = {}
    for w in WS:
        for rr in RS:
            X, _, Q = pick(E, w, rr)
            a_, s_ = strat_diff(X, Q, f"brk{H}"), wp.ols_boot(X, f"r{H}", [])
            sens[f"W{w}_R{rr}"] = dict(d2s=[float(x) for x in a_], S=[float(x) for x in s_])
            print(f"  W={w:2d} R={rr}: n={len(X):6d}  {pct(a_)} · S {wp.fmt(s_)}")
    res["sens"] = sens
    main_brk, main_S = tab[f"brk{H}"]["d2s"], tab[f"+{H}s"]["ctl"]
    if phase == "dev":
        d = sign(main_brk) or -1          # −1 = H_hold (돌파율 낮음)
        res["frozen_direction"] = d
        print(f"\nDEV 판정: 층화 돌파율 차 {pct(main_brk)} -> HOLDOUT 방향 = {'H_hold' if d < 0 else 'H_break'}"
              f"{'' if sign(main_brk) else ' (DEV 유의하지 않아 사전 지정 H_hold)'}")
    else:
        dp = OUT / f"result_{coin if coin == 'ETH' else 'ETH'}_dev.json"
        d = json.loads(dp.read_text())["frozen_direction"] if dp.exists() else -1
        ok_a, ok_b = sign(main_brk) == d, sign(main_S) == d     # H_hold: 돌파율 차 <0, S<0 · H_break: 둘 다 >0
        res.update(direction=d, pass_a=bool(ok_a), pass_b=bool(ok_b), passed=bool(ok_a and ok_b),
                   aux_vs_C1=sign(tab[f"brk{H}"]["d1s"]))
        print(f"\n{phase.upper()} 판정 ({'H_hold' if d < 0 else 'H_break'}): (a) 돌파율 차 {sign(main_brk)} · (b) S 통제 {sign(main_S)}"
              f" -> {'통과' if ok_a and ok_b else '불통과'}  (보조: vs C1 {sign(tab[f'brk{H}']['d1s'])})")
        if ok_a and ok_b:
            res["trade"] = trade(rep, d)
    (OUT / f"result_{coin}_{phase}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def trade(P, d):
    """H_hold(d=−1): 레벨 반대쪽(매도 레벨 -> 숏). H_break(d=+1): 레벨 쪽. 손절 변형은 H_hold 만: mid 가 레벨 바깥 경계를 넘으면 경계가에 청산(미끄러짐 0 = 낙관)."""
    out = {}
    sgn = np.where(P.side == 1, 1, -1) * d       # 매도 레벨 & H_hold -> −1(숏)
    print("\n매매화 (일 클러스터 부트스트랩)")
    def show(name, r, day):
        for cn, cost in (("taker", CRITERIA["taker_rt_bp"]), ("maker", CRITERIA["maker_rt_bp"])):
            g = pd.DataFrame({"day": day, "net": r - cost}).dropna()
            per = g.groupby("day").net.agg(["sum", "count"])
            rng = np.random.default_rng(CRITERIA["seed"])
            bs = [per.iloc[rng.integers(len(per), size=len(per))].pipe(lambda x: x["sum"].sum() / x["count"].sum()) for _ in range(CRITERIA["boot"])]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out[f"{name}_{cn}"] = dict(net=float(g.net.mean()), lo=float(lo), hi=float(hi), per_day=float(per["count"].mean()), days=len(per))
            print(f"  {name:12s} {cn:5s}: 순 {g.net.mean():+6.2f}bp/건 [{lo:+6.2f},{hi:+6.2f}] · 하루 {per['count'].mean():.0f}건")
    for h in CRITERIA["horizons_s"]:
        show(f"보유{h}s", sgn * P[f"r{h}"].to_numpy(), P.day.to_numpy())
    if d < 0:
        for h in (300, 900):
            stop = np.where(P.x_from_entry.fillna(1e9) <= 0, 0.0, sgn * np.log(P.edge / P.entry) * 1e4)
            hit = P.x_from_entry.fillna(1e9) <= h
            show(f"손절+{h}s", np.where(hit, stop, sgn * P[f"r{h}"]), P.day.to_numpy())
    for h in (300, 900):
        adv = np.where(sgn > 0, -P[f"dn{h}"], P[f"up{h}"]); adv = adv[np.isfinite(adv)]
        q = np.percentile(adv, [50, 90, 99])
        out[f"mae{h}"] = dict(p50=float(q[0]), p90=float(q[1]), p99=float(q[2]), ge250=float((adv >= 250).mean()))
        print(f"  MAE {h}s: p50 {q[0]:.1f} · p90 {q[1]:.1f} · p99 {q[2]:.1f}bp · ≥250bp(20배 증거금 절반) {(adv >= 250).mean():.2%}")
    return out


# ── 자체점검 ─────────────────────────────────────────────────────────────────
def selftest():
    tick, bucket = 100, 0.1
    snap = {"lastUpdateId": 10, "bids": [["999.99", "1"], ["999.70", "50"]], "asks": [["1000.00", "1"], ["1000.20", "50"]]}

    def stream(n_sec, bid_q):
        lines, u = [json.dumps({"_snapshot": snap})], 10
        for n in range(n_sec * 10):
            t = 1_000_000_000 + n * 100
            b = [["999.99", str(1 + n % 2)]]
            if n % 10 == 5: b.append(["999.70", str(bid_q(n // 10))])
            lines.append(json.dumps({"e": "depthUpdate", "T": t, "U": u + 1 if n else u, "u": u + 1, "pu": u, "b": b,
                                     "a": [["1000.20", "50"]] if n % 10 == 5 else []}))   # 매도 1000.20 은 매 초 50 으로 다시 채워짐
            u += 1
        return lines

    secs = np.arange(1_000_000, 1_000_040)
    vq = {W: (100.0, 30.0) for W in WS}
    # 1) 매도 칸 10002 에 매수 5/s (재보충), 매수 칸 9997 에 매도 5/s + 30초 뒤 잔량 0 (소진)
    B, st = book_seconds(iter(stream(40, lambda s: 50 if s < 30 else 0)), tick, bucket, idx_max=200_000)
    assert st == "ok" and len(B["sec"]) == 40 and B["sec"][0] == 1_000_000, (st, B and B["sec"][:3])
    s15 = secs[15:]                                         # 체결은 15초째부터(첫 창은 직전 상태를 몰라 사건이 될 수 없다)
    tape = pd.DataFrame({"ts_sec": np.r_[s15, s15], "price_bin": [10002] * 25 + [9997] * 25,
                         "buy_qty": [5.0] * 25 + [0.0] * 25, "sell_qty": [0.0] * 25 + [5.0] * 25})
    E = detect(B, tape, bucket, vq)
    rep = E[(E.cls == "rep") & (E.W == 30) & (E.R == 2)]
    dep = E[(E.cls == "dep") & (E.W == 30) & (E.R == 2)]
    assert len(rep) == 1 and rep.side.iloc[0] == 1 and rep.bin.iloc[0] == 10002 and rep.D1.iloc[0] >= 25, rep
    assert len(dep) == 1 and dep.side.iloc[0] == 0 and dep.bin.iloc[0] == 9997, dep
    assert not len(E[(E.cls == "rep") & (E.side == 0)]) and not len(E[(E.cls == "dep") & (E.side == 1)])
    assert not len(E[(E.W == 10) & (E.cls == "rep")]), "W=10 창 체결 50 < 체결 하한 100"
    assert rep.s.iloc[0] == 1_000_034 and dep.s.iloc[0] == 1_000_034, (rep.s, dep.s)   # 15~34초 매수 5/s = 100 = 하한
    # 2) 뚫림: 매도 칸 10003 에 매수 체결이 생기면 10002 는 사건이 아니다
    tape2 = pd.concat([tape, pd.DataFrame({"ts_sec": [1_000_025], "price_bin": [10003], "buy_qty": [1.0], "sell_qty": [0.0]})])
    E2 = detect(B, tape2, bucket, vq)
    assert not len(E2[(E2.cls == "rep") & (E2.bin == 10002) & (E2.s >= 1_000_025) & (E2.s < 1_000_025 + 30) & (E2.W == 30)])
    # 3) 조용함: 매도 칸 10002 에 매수 0.5/s
    tape3 = pd.DataFrame({"ts_sec": secs[32:], "price_bin": 10002, "buy_qty": 0.5, "sell_qty": 0.0})
    E3 = detect(B, tape3, bucket, vq)
    assert len(E3[(E3.cls == "quiet") & (E3.side == 1) & (E3.W == 30)]) >= 1 and not len(E3[E3.cls == "rep"])
    # 4) 시점: known = (s+1)초, 진입 ≥ known+1초, 돌파 판정
    ts = np.arange(999_990_000, 1_001_200_000, 250); mid = 1000 + (ts - 999_990_000) * 1e-7   # 천천히 상승
    O = attach_outcomes(rep, ts, mid, bucket)
    assert (O.known == (O.s + 1) * 1000).all() and (O.entry_ts >= O.known + 1000).all(), O[["s", "known", "entry_ts"]]
    assert O.pre1.iloc[0] > 0 and O.r30.iloc[0] > 0 and O.brk900.iloc[0] in (0.0, 1.0)
    print("selftest OK", dict(rep_s=int(rep.s.iloc[0]), entry_lag_ms=int(O.entry_ts.iloc[0] - O.known.iloc[0])))


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
