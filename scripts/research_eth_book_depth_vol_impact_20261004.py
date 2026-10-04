"""호가 얇아짐 -> 다음 변동성(A) · 큰 시장가 버스트의 충격과 감쇠(C) -- 사전등록 검정 (2026-10-04).

사용자: «호가·지정가·시장가 체결로 예측할 수 있는 건 더 없나?» -> «모두 진행». 맥락: 10-02 23시~10-03 03시 KST 급락(2770->2651)
중 물타기 -- «미리 알았으면 스위칭했을 것».
원천(서버 04시 백업 사본, 읽기 전용 · BDV_DATA 로 바꿀 수 있다):
  live/orderflow/depthdiff/<SYM>/<UTC시>.jsonl(.gz)  첫 줄 REST 스냅샷(1000레벨) + @depth@100ms
  live/orderflow/bookticker/<SYM>/<UTC시>.bt(.gz)     32B 행
  lake/binance/tape/coin=<C>/date=*/                  1초 x 가격칸 테이커 매수/매도
  lake/binance/liquidations/coin=<C>/date=*/          청산(side=long -> 롱 강제청산 = 매도)
실행:
  python scripts/research_eth_book_depth_vol_impact_20261004.py --selftest
  python scripts/research_eth_book_depth_vol_impact_20261004.py --build ETH        -> tmp/book_depth_vol_impact_20261004/panel_ETH.parquet
  python scripts/research_eth_book_depth_vol_impact_20261004.py --a ETH            (A: DEV 적합 -> HOLDOUT 한 번)
  python scripts/research_eth_book_depth_vol_impact_20261004.py --case            (10-02 13~19Z 급락 그림)
  python scripts/research_eth_book_depth_vol_impact_20261004.py --c ETH            (C)
북: 시각 파일 사이에 pu 가 이어지면 북을 이어 붙인다(스냅샷 1000레벨 = ETH ±40bp · BTC ±15bp 뿐이라 파일마다 리셋하면 ±50bp 가
매 시각 비어 톱니가 생긴다). 끊기면 다음 파일 스냅샷으로 리셋하고 warm_s 동안의 행은 버린다.
시점 경계(A): 결정 s(분 경계) 피쳐는 초 s-1 끝 상태까지. 타깃은 s+60 부터(1분 간격) -- 같은 초를 공유하지 않는다.
시점 경계(C): 버스트 초 s, known=(s+1)초, 진입 = 초 s+2 의 첫 bookTicker mid(>= known+1초), 청산 = 초 s+2+h 의 첫 mid.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_eth_wall_pull_jump_20261004 as wp  # noqa: E402

ROOT = wp.ROOT
DATA = Path(os.getenv("BDV_DATA", str(Path.home() / "backups/crypto-scalping-server/data")))
OUT = ROOT / "tmp/book_depth_vol_impact_20261004"
T = wp.T

# ── 사전등록 (결과 보기 전에 고정, 2026-10-04) ───────────────────────────────────
CRITERIA = dict(
    # 창 끝 = 로컬 백업 lake 테이프의 마지막 완결일(10-02). DEV 앞 60% / HOLDOUT 뒤 40%(한 번만 본다).
    window_eth=("2026-09-15T16Z", "2026-09-26T02Z", "2026-10-03T00Z"),
    window_multi=("2026-09-26T09Z", "2026-09-30T08Z", "2026-10-03T00Z"),
    bands_bp=(10, 25, 50), chg_min=(1, 5, 15),
    z_days=14, z_min_days=2,           # 같은 UTC 시 z: 앞선 날만(최대 14일, 최소 2일)
    warm_s=3600,                       # 북 리셋 뒤 1시간 미만 행 제외(스냅샷 밖 레벨이 아직 안 채워짐)
    # A 통제군 M0 = log RV(직전 5분·1시간·1일) + log 직전 1분 봉 폭 + 체결건수 5분 같은시 z + VPIN(직전 60분, 1분 시간버킷) + UTC 시 더미
    # A 호가군   = log D(10·25·50) + 같은시 z(3) + ΔlogD 1·5·15분(9) + 직전 60초 평균 스프레드 + |log(매수25/매도25)|
    horizons_min=(5, 15, 30), main_h=15, big_bp=100, big_h=30, label_gap_s=60,
    # A 판정 (HOLDOUT, 1h 블록 부트스트랩 95%):
    #   A-RV(h): OOS ΔR²(M1−M0, DEV 계수) CI 하한 > 0  AND  HOLDOUT 재적합 M0+logD25 계수 CI 상한 < 0(얇을수록 큼).
    #   A-BIG  : OOS ΔAUC(로지스틱 M1−M0, DEV 적합) CI 하한 > 0.
    #   주 판정 = 15분 RV + BIG. 5·30분은 보조. «쓸모» 문턱(통계 통과와 별개): ΔR² ≥ 0.005 · ΔAUC ≥ 0.01.
    # A 경보 규칙(통과 시): zD25 ≤ DEV q10 -> HOLDOUT 발동률·정밀도(BIG)·재현율·발동 뒤 30분 최대 역행 vs 같은 발동률의 «직전 5분 RV 상위» 대조.
    burst_q=0.995, burst_dedup_s=10, ctrl_net_q=0.5, impact_bins=5, c_horizons_s=(30, 60, 300, 900), c_main_h=60,
    # C 판정: fade(반대 지정가, 메이커 1.41) · follow(테이커 8) 각각 HOLDOUT 순bp/건 1h 블록 CI 하한 > 0 이고 DEV 평균 > 0.
    #   보조: 버스트 − 대조(같은 |1초 이동| 칸, 테이커 순체결 ≤ DEV 중앙) 의 «이동 방향 부호» 수익 차. 청산 포함/제외 분리.
    liq_min_usd=10_000,
    taker_rt_bp=8.0, maker_rt_bp=1.41, boot=2000, seed=20261004,
)
BANDS = CRITERIA["bands_bp"]
SYM = lambda c: f"{c}USDT"  # noqa: E731


# ── 1) 전체북 이어 붙이기 → 초 끝 밴드 깊이 ─────────────────────────────────────
def new_state(idx_max=wp.IDX_MAX):
    return dict(bid=None, ask=None, last_u=None, bb=-1, ba=-1, reset_s=None, idx_max=idx_max)


def _load_snap(st, snap, tick):
    st["bid"] = np.zeros(st["idx_max"]); st["ask"] = np.zeros(st["idx_max"])
    for key, arr in (("bids", st["bid"]), ("asks", st["ask"])):
        for p, q in snap[key]:
            i = int(round(float(p) * tick))
            if i < st["idx_max"]: arr[i] = float(q)
    st["bb"], st["ba"] = int(np.flatnonzero(st["bid"])[-1]), int(np.flatnonzero(st["ask"])[0])
    st["last_u"] = snap["lastUpdateId"]


def _bands(bid, ask, bb, ba, k_list=BANDS):
    mid = (bb + ba) / 2
    out = []
    for k in k_list:
        out.append(bid[int(np.ceil(mid * (1 - k * 1e-4))):bb + 1].sum())
        out.append(ask[ba:int(mid * (1 + k * 1e-4)) + 1].sum())
    return out


def depth_file(lines, tick, st):
    """한 시각 파일. st 를 이어받아 갱신. 반환 (rows, cover). row = (sec, bb, ba, b10, a10, b25, a25, b50, a50, reset_s) -- 초 끝 상태(가격·수량)."""
    first = orjson.loads(next(lines))
    snap = first.get("_snapshot") if isinstance(first, dict) else None
    cov = dict(status="ok", mode=None, carry25=np.nan, carry_far=np.nan)
    rows, cur, synced = [], None, False
    for line in lines:
        try:
            m = orjson.loads(line)
        except orjson.JSONDecodeError:
            cov["status"] = "truncated"; break
        if m.get("_gap"):
            cov["status"] = "gap"; break
        if m.get("e") != "depthUpdate":
            continue
        if cov["mode"] is None:
            if st["last_u"] is not None and m["pu"] == st["last_u"]:
                cov["mode"], synced = "carry", True
                if snap:   # 이어 붙인 북 vs 새 스냅샷: ±25bp 와 스냅샷 끝 근처(±(끝-10bp)) 합 비율 -- 묵은 레벨 점검
                    s = new_state(st["idx_max"]); _load_snap(s, snap, tick)
                    a = _bands(st["bid"], st["ask"], s["bb"], s["ba"], (25,)); b = _bands(s["bid"], s["ask"], s["bb"], s["ba"], (25,))
                    cov["carry25"] = (a[0] + a[1]) / max(b[0] + b[1], 1e-12)
                    mid = (s["bb"] + s["ba"]) / 2
                    far = min((mid - float(snap["bids"][-1][0]) * tick) / mid, (float(snap["asks"][-1][0]) * tick - mid) / mid) * 1e4 - 10
                    if far > 0:
                        a = _bands(st["bid"], st["ask"], s["bb"], s["ba"], (far,)); b = _bands(s["bid"], s["ask"], s["bb"], s["ba"], (far,))
                        cov["carry_far"] = (a[0] + a[1]) / max(b[0] + b[1], 1e-12)
                    del s
            else:
                if not snap:
                    cov["status"] = "no_snapshot"; break
                _load_snap(st, snap, tick); cov["mode"] = "reset"
                st["reset_s"] = int(first.get("at_ms", m["T"])) // 1000
        if not synced:
            if m["u"] < st["last_u"]:
                continue
            if not (m["U"] <= st["last_u"] <= m["u"]):
                cov["status"] = "sync_fail"; break
            synced = True
        elif m["pu"] != st["last_u"]:
            cov["status"] = "gap"; break
        st["last_u"], sec = m["u"], int(m["T"]) // 1000
        if cur is None:
            cur = sec
        elif sec > cur:
            r = (st["bb"] / tick, st["ba"] / tick, *_bands(st["bid"], st["ask"], st["bb"], st["ba"]), st["reset_s"])
            rows.extend((s_,) + r for s_ in range(cur, sec))
            cur = sec
        for key, arr in (("b", st["bid"]), ("a", st["ask"])):
            for p, q in m[key]:
                i = int(round(float(p) * tick))
                if i < st["idx_max"]: arr[i] = float(q)
        st["bb"], st["ba"] = wp._best(st["bid"], st["bb"], True), wp._best(st["ask"], st["ba"], False)
        if st["bb"] < 0 or st["ba"] < 0:
            cov["status"] = "empty_book"; break
    if cov["status"] == "ok" and cur is not None:
        rows.append((cur, st["bb"] / tick, st["ba"] / tick, *_bands(st["bid"], st["ask"], st["bb"], st["ba"]), st["reset_s"]))
    if cov["status"] != "ok":
        st["last_u"] = None          # 다음 파일은 스냅샷으로 리셋
    return rows, cov


DCOLS = ["sec", "bb", "ba"] + [f"{s}{k}" for k in BANDS for s in ("b", "a")] + ["reset_s"]


def depth_chain(coin, lo, hi):
    files = [f for f in sorted(glob.glob(str(DATA / f"live/orderflow/depthdiff/{SYM(coin)}/*.jsonl*")))
             if lo <= T(Path(f).name[:13] + ":00Z") < hi]
    st, rows, covs = new_state(), [], []
    for f in files:
        with wp._open(f) as fh:
            r, c = depth_file(iter(fh), wp.TICKS[coin], st)
        rows += r; c.update(hour=Path(f).name[:13], n=len(r)); covs.append(c)
        print(coin, c, flush=True)
    D = pd.DataFrame(rows, columns=DCOLS).drop_duplicates("sec", keep="last").set_index("sec")
    return D, covs


def bt_hour(path):
    with wp._open(path) as f:
        b = f.read()
    a = np.frombuffer(b, dtype=wp.BT_DT, count=(len(b) - 32) // 32, offset=32)
    a = a[np.argsort(a["ts"], kind="stable")]
    mid = (a["bp"] + a["ap"]) / 2
    d = pd.DataFrame({"sec": a["ts"] // 1000, "mid": mid, "spr": (a["ap"] - a["bp"]) / mid * 1e4})
    g = d.groupby("sec")
    return pd.DataFrame({"mf": g.mid.first(), "ml": g.mid.last(), "mx": g.mid.max(), "mn": g.mid.min(),
                         "spr": g.spr.mean(), "spr_n": g.spr.count()})


def build(coin):
    lo, _, hi = (T(x) for x in CRITERIA["window_eth" if coin == "ETH" else "window_multi"])
    lo_b, hi_b = lo - 2 * 86_400_000, hi + 3_600_000          # 북은 lo 보다 2일 먼저(예열·z), 끝은 타깃용 1시간 더
    btf = [f for f in sorted(glob.glob(str(DATA / f"live/orderflow/bookticker/{SYM(coin)}/*.bt*")))
           if lo_b <= T(Path(f).name[:13] + ":00Z") < hi_b]
    with ProcessPoolExecutor(max_workers=6) as ex:
        fd = ex.submit(depth_chain, coin, lo_b, hi_b)
        bts = list(ex.map(bt_hour, btf))
        D, covs = fd.result()
    B = pd.concat(bts)
    B = B.groupby(level=0).agg(mf=("mf", "first"), ml=("ml", "last"), mx=("mx", "max"), mn=("mn", "min"), spr=("spr", "mean"))
    days = pd.date_range(pd.to_datetime(lo_b, unit="ms").normalize(), pd.to_datetime(hi_b, unit="ms"), freq="D").strftime("%Y-%m-%d")
    tp = []
    for d in days:
        for p in glob.glob(str(DATA / f"lake/binance/tape/coin={coin}/date={d}/*.parquet")):
            t = pd.read_parquet(p, columns=["ts_sec", "price_bin", "buy_qty", "sell_qty", "buy_n", "sell_n"])
            px = t.price_bin * wp.BUCKETS[coin]
            tp.append(pd.DataFrame({"sec": t.ts_sec, "buy": t.buy_qty * px, "sell": t.sell_qty * px, "n": t.buy_n + t.sell_n}))
    TP = pd.concat(tp).groupby("sec").sum() if tp else pd.DataFrame(columns=["buy", "sell", "n"])
    lq = [pd.read_parquet(p, columns=["ts_ms", "side", "usd"]) for p in glob.glob(str(DATA / f"lake/binance/liquidations/coin={coin}/date=*/*.parquet"))]
    if lq:
        L = pd.concat(lq); L["sec"] = L.ts_ms // 1000
        LQ = L.pivot_table(index="sec", columns="side", values="usd", aggfunc="sum").reindex(columns=["long", "short"]).fillna(0)
        LQ.columns = ["liq_long", "liq_short"]
    else:
        LQ = pd.DataFrame(columns=["liq_long", "liq_short"])
    P = B.join(D, how="left").join(TP, how="left").join(LQ, how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    P.to_parquet(OUT / f"panel_{coin}.parquet")
    meta = dict(tape_first=int(TP.index.min()) if len(TP) else None, tape_last=int(TP.index.max()) if len(TP) else None,
                liq_first=int(LQ.index.min()) if len(LQ) else None, liq_last=int(LQ.index.max()) if len(LQ) else None, cover=covs)
    (OUT / f"cover_{coin}.json").write_text(json.dumps(meta, indent=0, default=float))
    print("panel", coin, P.shape, flush=True)


# ── 공통: 1h 블록 부트스트랩(행 가중) ─────────────────────────────────────────────
def bboot(g, stat, B=None):
    """g: 행마다 블록 id. stat(w) -> float, w = 행 가중(블록 재표본 횟수). 반환 (추정, lo, hi, 블록수)."""
    ub, inv = np.unique(g, return_inverse=True)
    est = stat(np.ones(len(g)))
    rng = np.random.default_rng(CRITERIA["seed"])
    bs = []
    for _ in range(B or CRITERIA["boot"]):
        v = stat(np.bincount(rng.integers(len(ub), size=len(ub)), minlength=len(ub))[inv].astype(float))
        if np.isfinite(v): bs.append(v)
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return float(est), float(lo), float(hi), int(len(ub))


def wls(X, y, w):
    Xw = X * w[:, None]
    return np.linalg.lstsq(Xw.T @ X, Xw.T @ y, rcond=None)[0]


def load_panel(coin):
    """테이프 표는 체결 있는 초만 행이 있다 -> 빈 초(120초 미만 연속)는 체결 0. 더 긴 공백은 수집 중단으로 보고 NaN 유지."""
    P = pd.read_parquet(OUT / f"panel_{coin}.parquet")
    P = P.reindex(np.arange(int(P.index.min()), int(P.index.max()) + 1))
    na = P.n.isna()
    run = na.groupby((~na).cumsum()).transform("sum")
    lo, hi = P.n.first_valid_index(), P.n.last_valid_index()
    fill = na & (run < 120) & (P.index > lo) & (P.index < hi)
    P.loc[fill, ["buy", "sell", "n"]] = 0.0
    return P


# ── 2) A: 분 결정 프레임 ────────────────────────────────────────────────────────
def same_hour_z(df, col):
    """같은 UTC 시, 앞선 날(최대 z_days, 최소 z_min_days)의 평균·표준편차로 z."""
    x = df[col]
    g = pd.DataFrame({"day": df.day, "hour": df.hour, "x": x, "x2": x ** 2, "n": np.isfinite(x).astype(float)}).dropna(subset=["x"])
    S = g.pivot_table(index="day", columns="hour", values=["x", "x2", "n"], aggfunc="sum")
    S = S.reindex(np.arange(df.day.min(), df.day.max() + 1)).fillna(0)
    R = S.shift(1).rolling(CRITERIA["z_days"], min_periods=1).sum()
    nd = (S["n"] > 0).astype(float).shift(1).rolling(CRITERIA["z_days"], min_periods=1).sum()
    n = R["n"]; mu = R["x"] / n; sd = np.sqrt(R["x2"] / n - mu ** 2)
    mu, sd = mu.where(nd >= CRITERIA["z_min_days"]), sd.where(nd >= CRITERIA["z_min_days"])
    key = pd.MultiIndex.from_arrays([df.day, df.hour])
    m = mu.stack().reindex(key).to_numpy(); s = sd.stack().reindex(key).to_numpy()
    return (x.to_numpy() - m) / s


def minute_frame(P: pd.DataFrame) -> pd.DataFrame:
    s0, s1 = int(P.index.min()), int(P.index.max())
    P = P.reindex(np.arange(s0, s1 + 1))
    ml, mx, mn = P.ml.to_numpy(), P.mx.to_numpy(), P.mn.to_numpy()
    r1 = np.diff(np.log(ml), prepend=np.nan) * 1e4
    ok = np.isfinite(r1)
    C2 = np.r_[0, np.cumsum(np.where(ok, r1 ** 2, 0))]; CV = np.r_[0, np.cumsum(ok)]

    def rv(a, b):            # 초 위치 [a, b) -- 창이 패널 밖이거나 커버 90% 미만이면 NaN
        inside = (a >= 0) & (b <= len(ml))
        a = np.clip(a, 0, len(ml)); b = np.clip(b, 0, len(ml))
        cov = (CV[b] - CV[a]) / np.maximum(b - a, 1)
        return np.where(inside & (cov >= 0.9), np.sqrt(C2[b] - C2[a]), np.nan)

    first = (-(s0 % 60)) % 60
    pos = np.arange(first + 60, len(ml) - 1, 60)          # 결정 s = s0+pos (분 경계), 피쳐 = 위치 pos-1
    s = s0 + pos
    F = pd.DataFrame({"s": s, "day": s // 86400, "hour": (s // 3600) % 24})
    q = pos - 1
    mid = (P.bb.to_numpy() + P.ba.to_numpy()) / 2
    for k in BANDS:
        b, a = P[f"b{k}"].to_numpy() * mid, P[f"a{k}"].to_numpy() * mid
        lD = np.log(b + a)
        F[f"logD{k}"] = lD[q]
        for mm in CRITERIA["chg_min"]:
            F[f"dD{k}_{mm}"] = lD[q] - lD[np.clip(q - 60 * mm, 0, None)]
        if k == 25:
            F["asym25"] = np.abs(np.log(b[q] / a[q]))
    F["since_reset"] = s - 1 - P.reset_s.to_numpy()[q]
    F["spread60"] = pd.Series(P.spr.to_numpy()).rolling(60, min_periods=30).mean().to_numpy()[q]
    F["lrv5"], F["lrv60"], F["lrv1d"] = (np.log(rv(pos - w, pos)) for w in (300, 3600, 86400))
    F["lrange1"] = np.log(1 + (pd.Series(mx).rolling(60, min_periods=30).max() - pd.Series(mn).rolling(60, min_periods=30).min()).to_numpy()[q] / ml[q] * 1e4)
    n = P.n.to_numpy(); has_tape = np.isfinite(n)
    n0 = np.r_[0, np.cumsum(np.nan_to_num(n))]; nt = np.r_[0, np.cumsum(has_tape)]
    p3 = np.clip(pos - 300, 0, None)
    F["ltr5"] = np.where((pos >= 300) & (nt[pos] - nt[p3] >= 290), np.log1p(n0[pos] - n0[p3]), np.nan)
    bu, se = P.buy.to_numpy(), P.sell.to_numpy()
    mb = pd.DataFrame({"b": bu, "s": se}).groupby((np.arange(len(bu)) - first) // 60).sum(min_count=1)
    imb = (np.abs(mb.b - mb.s) / (mb.b + mb.s)).rolling(60, min_periods=50).mean()   # ponytail: 시간버킷 VPIN 근사(거래량 버킷 아님)
    bar = (pos - first) // 60                                                          # 결정 분 = 막대 bar 의 시작 -> 직전 60막대 = bar-60..bar-1
    F["vpin60"] = imb.reindex(bar - 1).to_numpy()
    for k in BANDS:
        F[f"zD{k}"] = same_hour_z(F, f"logD{k}")
    F["ztr5"] = same_hour_z(F, "ltr5")
    g = CRITERIA["label_gap_s"]
    for h in CRITERIA["horizons_min"]:
        F[f"y_rv{h}"] = np.log(rv(pos + g, pos + g + 60 * h))
    p0 = ml[np.clip(pos + g - 1, 0, len(ml) - 1)]
    H = CRITERIA["big_h"] * 60
    fmx = pd.Series(mx[::-1]).rolling(H, min_periods=int(0.9 * H)).max().to_numpy()[::-1]   # [i, i+H)
    fmn = pd.Series(mn[::-1]).rolling(H, min_periods=int(0.9 * H)).min().to_numpy()[::-1]
    i0 = np.clip(pos + g, 0, len(ml) - 1)
    up, dn = np.log(fmx[i0] / p0) * 1e4, np.log(fmn[i0] / p0) * 1e4
    F["exc_up"], F["exc_dn"] = up, dn
    F["y_big"] = np.where(np.isfinite(up) & np.isfinite(dn), (np.maximum(up, -dn) >= CRITERIA["big_bp"]).astype(float), np.nan)
    F["mid"] = ml[q]
    return F.replace([np.inf, -np.inf], np.nan)          # RV 0(시세 멈춤) -> log -inf 는 결측


M0 = ["lrv5", "lrv60", "lrv1d", "lrange1", "ztr5", "vpin60"]
BOOK = ([f"logD{k}" for k in BANDS] + [f"zD{k}" for k in BANDS] + [f"dD{k}_{m}" for k in BANDS for m in CRITERIA["chg_min"]]
        + ["spread60", "asym25"])


def design(F, cols):
    hd = pd.get_dummies(F.hour, prefix="h").reindex(columns=[f"h_{i}" for i in range(1, 24)], fill_value=0).astype(float)
    return np.column_stack([np.ones(len(F)), F[cols].to_numpy(float), hd.to_numpy()])


def r2_stat(X0, X1, y):
    def f(w):
        out = []
        for X in (X0, X1):
            b = wls(X, y, w); e = y - X @ b
            ybar = (w * y).sum() / w.sum()
            out.append(1 - (w * e ** 2).sum() / (w * (y - ybar) ** 2).sum())
        return out[1] - out[0]
    return f


def analyze_a(coin):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    P = load_panel(coin)
    F = minute_frame(P)
    a, b, c = (T(x) // 1000 for x in CRITERIA["window_eth" if coin == "ETH" else "window_multi"])
    F = F[(F.s >= a) & (F.s < c)].copy()
    F["blk"] = F.s // 3600
    feats = M0 + BOOK
    base = F[F.since_reset >= CRITERIA["warm_s"]].dropna(subset=feats)
    res = dict(coin=coin, n_minutes=len(F), n_usable=len(base), drop_warm=int((F.since_reset < CRITERIA["warm_s"]).sum()),
               window=[str(pd.to_datetime(x, unit="s")) for x in (a, b, c)])
    D_, Hd = base[base.s < b], base[base.s >= b]
    lines = [f"=== A {coin}  DEV {len(D_)}분 ({D_.blk.nunique()}h) · HOLDOUT {len(Hd)}분 ({Hd.blk.nunique()}h) · 예열 제외 {res['drop_warm']}"]
    res["dev_first"] = str(pd.to_datetime(D_.s.min(), unit="s")) if len(D_) else None
    tab = {}
    for h in CRITERIA["horizons_min"]:
        y = f"y_rv{h}"
        d, ho = D_.dropna(subset=[y]), Hd.dropna(subset=[y])
        X0d, X1d, X0h, X1h = design(d, M0), design(d, feats), design(ho, M0), design(ho, feats)
        yd, yh = d[y].to_numpy(), ho[y].to_numpy()
        r_dev = bboot(d.blk.to_numpy(), r2_stat(X0d, X1d, yd))
        b0, b1 = wls(X0d, yd, np.ones(len(yd))), wls(X1d, yd, np.ones(len(yd)))
        e0, e1 = yh - X0h @ b0, yh - X1h @ b1

        def oos(w, e0=e0, e1=e1, yh=yh):
            ybar = (w * yh).sum() / w.sum()
            return ((w * e0 ** 2).sum() - (w * e1 ** 2).sum()) / (w * (yh - ybar) ** 2).sum()
        r_oos = bboot(ho.blk.to_numpy(), oos)
        Xc_h = design(ho, M0 + ["logD25"]); jc = 1 + len(M0)
        r_coef_h = bboot(ho.blk.to_numpy(), lambda w: wls(Xc_h, yh, w)[jc])
        Xc_d = design(d, M0 + ["logD25"])
        r_coef_d = bboot(d.blk.to_numpy(), lambda w: wls(Xc_d, yd, w)[jc])
        r2_m0_oos = 1 - (e0 ** 2).sum() / ((yh - yh.mean()) ** 2).sum()
        passed = r_oos[1] > 0 and r_coef_h[2] < 0
        tab[f"rv{h}"] = dict(dev_dR2=r_dev, oos_dR2=r_oos, coef_logD25_dev=r_coef_d, coef_logD25_hold=r_coef_h,
                             oos_R2_M0=float(r2_m0_oos), passed=bool(passed), useful=bool(passed and r_oos[0] >= 0.005))
        lines.append(f"RV{h:2d}분: DEV ΔR² {fm(r_dev)} | HOLDOUT OOS ΔR² {fm(r_oos)} (M0 OOS R² {r2_m0_oos:.3f}) | "
                     f"logD25 계수 DEV {fm(r_coef_d)} · HOLD {fm(r_coef_h)} -> {'통과' if passed else '불통과'}")
    # BIG
    y = "y_big"
    d, ho = D_.dropna(subset=[y]), Hd.dropna(subset=[y])
    mu, sd = d[feats].mean(), d[feats].std().replace(0, 1)
    Z = lambda X, cols: np.column_stack([((X[cols] - mu[cols]) / sd[cols]).to_numpy(), design(X, [])[:, 1:]])  # noqa: E731
    probs = {}
    for name, cols in (("M0", M0), ("M1", feats)):
        lr = LogisticRegression(max_iter=5000).fit(Z(d, cols), d[y])
        probs[name] = (lr.predict_proba(Z(ho, cols))[:, 1], lr.predict_proba(Z(d, cols))[:, 1])
    yh = ho[y].to_numpy()

    def dauc(w, yh=yh):
        if len(np.unique(yh[w > 0])) < 2: return np.nan
        return roc_auc_score(yh, probs["M1"][0], sample_weight=w) - roc_auc_score(yh, probs["M0"][0], sample_weight=w)
    r_auc = bboot(ho.blk.to_numpy(), dauc, B=1000)
    auc = {k: float(roc_auc_score(yh, v[0])) for k, v in probs.items()} if len(np.unique(yh)) > 1 else {}
    tab["big"] = dict(dauc=r_auc, auc_hold=auc, base_dev=float(d[y].mean()), base_hold=float(yh.mean()),
                      n_event_blocks_hold=int(ho[ho[y] == 1].blk.nunique()), passed=bool(r_auc[1] > 0),
                      useful=bool(r_auc[1] > 0 and r_auc[0] >= 0.01))
    lines.append(f"BIG(30분 |이동|≥1%): 기저 DEV {d[y].mean():.3%} · HOLD {yh.mean():.3%} (사건 시각 {tab['big']['n_event_blocks_hold']}h) | "
                 f"AUC HOLD {auc} | ΔAUC {fm(r_auc)} -> {'통과' if r_auc[1] > 0 else '불통과'}")
    # 경보 규칙(통과 여부와 무관하게 기술 통계로 계산 -- 제안은 통과 시만)
    thr = float(d.zD25.quantile(0.10)); thr_rv = float(d.lrv5.quantile(0.90))
    alarm = {}
    for name, fire in (("zD25<=DEVq10", ho.zD25 <= thr), ("lrv5>=DEVq90(대조)", ho.lrv5 >= thr_rv),
                       ("둘다", (ho.zD25 <= thr) & (ho.lrv5 >= thr_rv)), ("M1확률 상위10%", pd.Series(probs["M1"][0] >= np.quantile(probs["M1"][1], 0.9), index=ho.index)),
                       ("M0확률 상위10%", pd.Series(probs["M0"][0] >= np.quantile(probs["M0"][1], 0.9), index=ho.index))):
        f = fire.to_numpy(bool)
        adv = np.maximum(ho.exc_up, -ho.exc_dn).to_numpy()
        prec = float(yh[f].mean()) if f.any() else np.nan
        rec = float(f[yh == 1].mean()) if (yh == 1).any() else np.nan
        dn = -ho.exc_dn.to_numpy()[f]
        alarm[name] = dict(rate=float(f.mean()), precision=prec, recall=rec,
                           adv_p50=float(np.nanpercentile(adv[f], 50)) if f.any() else np.nan,
                           adv_p90=float(np.nanpercentile(adv[f], 90)) if f.any() else np.nan,
                           dn_p90=float(np.nanpercentile(dn, 90)) if f.any() else np.nan)
        lines.append(f"  경보 {name:18s}: 발동률 {f.mean():.2%} · 정밀도 {prec:.2%} · 재현율 {rec:.2%} · 30분 최대이탈 p50 {alarm[name]['adv_p50']:.0f} / p90 {alarm[name]['adv_p90']:.0f}bp · 하방 p90 {alarm[name]['dn_p90']:.0f}bp")
    adv = np.maximum(ho.exc_up, -ho.exc_dn)
    lines.append(f"  (HOLDOUT 전체: 30분 최대이탈 p50 {np.nanpercentile(adv, 50):.0f} / p90 {np.nanpercentile(adv, 90):.0f}bp · 문턱 zD25 {thr:.2f} · lrv5 {thr_rv:.2f})")
    # 단변량 분위(탐지 신호 점검용): HOLDOUT 15분 RV 평균을 zD25 5분위로
    qz = pd.qcut(Hd.zD25, 5, labels=False, duplicates="drop")
    lines.append("  HOLDOUT zD25 5분위 -> 평균 log RV15: " + " ".join(
        f"q{int(k)}:{v:+.3f}" for k, v in Hd.groupby(qz).y_rv15.mean().items()))
    res.update(table=tab, alarm=alarm, alarm_thr=dict(zD25=thr, lrv5=thr_rv))
    print("\n".join(lines))
    (OUT / f"result_A_{coin}.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / f"result_A_{coin}.txt").write_text("\n".join(lines))
    F.to_parquet(OUT / f"minute_{coin}.parquet")
    return res


def fm(r):
    return f"{r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}]" if np.isfinite(r[0]) else "n/a"


# ── 3) 급락 사례 그림 ────────────────────────────────────────────────────────
def case_plot(coin="ETH", lo="2026-10-02T13Z", hi="2026-10-02T19Z"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = load_panel(coin)
    F = pd.read_parquet(OUT / f"minute_{coin}.parquet")
    a, b = T(lo) // 1000, T(hi) // 1000
    S = P.loc[a:b]
    t = pd.to_datetime(S.index, unit="s")
    mid = (S.bb + S.ba) / 2
    Fm = F[(F.s >= a) & (F.s <= b)]
    tm = pd.to_datetime(Fm.s, unit="s")
    r1m = S.ml.groupby(S.index // 60).last().pipe(lambda x: np.log(x).diff())
    legs = sorted(int(i) * 60 for i in r1m.nsmallest(2).index)
    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True, gridspec_kw=dict(hspace=0.08))
    ax[0].plot(t, S.ml, lw=0.8, color="#222"); ax[0].set_ylabel("mid (USDT)")
    for side, col, name in (("b", "#2a78d6", "bid (mid−25bp..best)"), ("a", "#eb6834", "ask (best..mid+25bp)")):
        ax[1].plot(t, (S[f"{side}25"] * mid / 1e6).rolling(10, min_periods=1).mean(), lw=0.7, color=col, label=name)
    ax[1].set_yscale("log"); ax[1].set_ylabel("depth ≤25bp ($M)"); ax[1].legend(loc="upper left", frameon=False)
    ax[2].plot(tm, Fm.zD25, color="#2a78d6", lw=1.2); ax[2].set_ylabel("z depth ±25bp")
    ax[2].axhline(0, color="#999", lw=0.5)
    ax[3].plot(t, S.spr.rolling(60, min_periods=10).mean(), color="#555", lw=0.8); ax[3].set_ylabel("spread bp (60s mean)")
    ax[4].plot(tm, np.exp(Fm.lrv5), color="#2a78d6", lw=1.2, label="RV past 5m")
    ax[4].plot(tm, np.exp(Fm.y_rv5), color="#eb6834", lw=1.2, label="RV next 5m (from s+1m)")
    ax[4].legend(loc="upper left", frameon=False); ax[4].set_ylabel("RV bp")
    for x in ax:
        x.grid(alpha=0.2)
        for leg in legs:   # 급락 다리(1분 하락 최대 두 곳)
            x.axvline(pd.to_datetime(leg, unit="s"), color="#999", lw=0.8, ls=":")
    ax[0].set_title(f"{coin} 2026-10-02 {lo[11:13]}–{hi[11:13]} UTC (KST +9): order book depth vs crash (dotted = two largest 1-min drops)")
    p = OUT / f"case_{coin}_20261002.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    print(p)
    # 수치 요약: 분 단위 표
    Fm = Fm.assign(t=tm.dt.strftime("%H:%M"), D25=np.exp(Fm.logD25) / 1e6, rv5=np.exp(Fm.lrv5))
    print(Fm[["t", "mid", "D25", "zD25", "dD25_5", "spread60", "rv5"]].iloc[::5].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


# ── 4) C: 시장가 버스트 충격·감쇠 ─────────────────────────────────────────────
def burst_frame(P: pd.DataFrame, thr: float | None = None):
    s0 = int(P.index.min())
    P = P.reindex(np.arange(s0, int(P.index.max()) + 1))
    net = (P.buy - P.sell).to_numpy()
    ml, mf = P.ml.to_numpy(), P.mf.to_numpy()
    n = len(net)
    mv = np.log(ml / np.r_[np.nan, ml[:-1]]) * 1e4                    # 1초 이동(초 s 끝 vs s-1 끝)
    lq_b = P.liq_short.to_numpy() if "liq_short" in P else np.full(n, np.nan)   # 숏 청산 = 강제 매수
    lq_s = P.liq_long.to_numpy() if "liq_long" in P else np.full(n, np.nan)

    def outcome(i, d):
        e = i + 2                                                      # 진입 = 초 s+2 의 첫 mid (>= known+1초)
        ok = e < n
        ent = np.where(ok, mf[np.clip(e, 0, n - 1)], np.nan)
        out = {"impact": d * mv[i], "conc": d * np.log(ent / ml[i]) * 1e4}
        for h in CRITERIA["c_horizons_s"]:
            x = np.clip(e + h, 0, n - 1)
            out[f"r{h}"] = np.where(e + h < n, d * np.log(mf[x] / ent) * 1e4, np.nan)
        return out
    return dict(s0=s0, net=net, mv=mv, lq_b=lq_b, lq_s=lq_s, outcome=outcome, n=n)


def analyze_c(coin):
    P = load_panel(coin)
    a, b, c = (T(x) // 1000 for x in CRITERIA["window_eth" if coin == "ETH" else "window_multi"])
    Bf = burst_frame(P)
    s0, net, mv = Bf["s0"], Bf["net"], Bf["mv"]
    sec = s0 + np.arange(Bf["n"])
    inwin = (sec >= a) & (sec < c) & np.isfinite(net)
    dev = inwin & (sec < b)
    thr = float(np.quantile(np.abs(net[dev]), CRITERIA["burst_q"]))
    small = float(np.quantile(np.abs(net[dev]), CRITERIA["ctrl_net_q"]))
    cand = np.flatnonzero(inwin & (np.abs(net) >= thr))
    keep, last = [], {1: -10 ** 12, -1: -10 ** 12}
    for i in cand:
        d = 1 if net[i] > 0 else -1
        if sec[i] - last[d] > CRITERIA["burst_dedup_s"]:   # 같은 쪽 남긴 사건 뒤 10초 안은 버림(wp.select 규약)
            keep.append(i); last[d] = sec[i]
    idx = np.array(keep)
    d = np.sign(net[idx])
    E = pd.DataFrame({"s": sec[idx], "d": d, "net": net[idx], **Bf["outcome"](idx, d)})
    lq = np.where(d > 0, np.nan_to_num(Bf["lq_b"][idx]) + np.nan_to_num(Bf["lq_b"][idx - 1]),
                  np.nan_to_num(Bf["lq_s"][idx]) + np.nan_to_num(Bf["lq_s"][idx - 1]))
    P_l = P[["liq_long", "liq_short"]].dropna(how="all")
    liq_lo = int(P_l.index.min()) if len(P_l) else 10 ** 12
    E["liq"] = np.where(E.s >= liq_lo, np.where(lq >= CRITERIA["liq_min_usd"], "with_liq", "no_liq"), "unknown")
    E["blk"] = E.s // 3600; E["phase"] = np.where(E.s < b, "DEV", "HOLD")
    # 대조: 테이커 순체결 작은 초, |1초 이동| 칸 맞춤
    ci = np.flatnonzero(inwin & (np.abs(net) <= small) & np.isfinite(mv) & (mv != 0))
    dc = np.sign(mv[ci])
    Cc = pd.DataFrame({"s": sec[ci], **Bf["outcome"](ci, dc)})
    Cc["blk"] = Cc.s // 3600; Cc["phase"] = np.where(Cc.s < b, "DEV", "HOLD")
    edges = np.quantile(E.impact[(E.phase == "DEV") & (E.impact > 0)], np.linspace(0, 1, CRITERIA["impact_bins"] + 1))
    lines = [f"=== C {coin}: 문턱 |순체결| ≥ ${thr:,.0f}/초 (DEV q99.5) · 대조 |순체결| ≤ ${small:,.0f} · 버스트 DEV {int((E.phase == 'DEV').sum())} · HOLD {int((E.phase == 'HOLD').sum())}",
             f"충격 칸 경계(bp, DEV 버스트 중 이동>0): {np.round(edges, 2).tolist()} · 충격>0 비율 {float((E.impact > 0).mean()):.1%}"]
    res = dict(coin=coin, thr=thr, small=small, n=E.groupby("phase").size().to_dict(), edges=edges.tolist(), table={})
    for ph in ("DEV", "HOLD"):
        X = E[E.phase == ph]
        lines.append(f"\n[{ph}] 버스트 {len(X)} (하루 {len(X) / max(X.blk.nunique() / 24, 1e-9):.0f}건) · 충격 중앙 {X.impact.median():+.2f}bp · 지연 1초 이동 {X.conc.mean():+.2f}bp")
        for h in CRITERIA["c_horizons_s"]:
            y = X[f"r{h}"].to_numpy(); g = X.blk.to_numpy(); m = np.isfinite(y)
            fade = bboot(g[m], lambda w, y=-y[m]: (w * (y - CRITERIA["maker_rt_bp"])).sum() / w.sum())
            fol = bboot(g[m], lambda w, y=y[m]: (w * (y - CRITERIA["taker_rt_bp"])).sum() / w.sum())
            pos = X.impact > 1
            ratio = float(np.nanmedian(-X[f"r{h}"][pos] / X.impact[pos])) if pos.any() else np.nan
            # 버스트 − 대조(이동>0 버스트만, 칸 맞춤)
            Xp = X[X.impact > 0]; Cp = Cc[(Cc.phase == ph) & (np.abs(Cc.impact) >= edges[0]) & (np.abs(Cc.impact) <= edges[-1])]
            diff = strat_diff(Xp, Cp, f"r{h}", edges)
            res["table"][f"{ph}_{h}"] = dict(fade_maker=fade, follow_taker=fol, raw=float(np.nanmean(y)), revert_ratio_med=ratio, burst_minus_ctrl=diff)
            lines.append(f"  +{h:4d}s: 원 {np.nanmean(y):+6.2f}bp(+ = 지속) · 되돌림비 중앙 {ratio:+.2f} | fade(메이커) {fm2(fade)} | follow(테이커) {fm2(fol)} | 버스트−대조 {fm2(diff)}")
        for lk in ("with_liq", "no_liq"):
            Xl = X[X.liq == lk]
            if len(Xl) >= 20:
                y = Xl[f"r{CRITERIA['c_main_h']}"].to_numpy(); m = np.isfinite(y)
                r = bboot(Xl.blk.to_numpy()[m], lambda w, y=y[m]: (w * y).sum() / w.sum())
                lines.append(f"  청산 {lk:8s} n={len(Xl):4d}: +{CRITERIA['c_main_h']}s 원(+ = 지속) {fm2(r)} · 충격 중앙 {Xl.impact.median():+.2f}")
                res["table"][f"{ph}_liq_{lk}"] = r
    H = CRITERIA["c_main_h"]
    verdict = {}
    for side in ("fade_maker", "follow_taker"):
        dv, ho = res["table"][f"DEV_{H}"][side], res["table"][f"HOLD_{H}"][side]
        verdict[side] = bool(dv[0] > 0 and ho[1] > 0)
    res["verdict"] = verdict
    lines.append(f"\n판정(주 지평 {H}s): fade {'통과' if verdict['fade_maker'] else '불통과'} · follow {'통과' if verdict['follow_taker'] else '불통과'}")
    print("\n".join(lines))
    (OUT / f"result_C_{coin}.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / f"result_C_{coin}.txt").write_text("\n".join(lines))
    E.to_parquet(OUT / f"bursts_{coin}.parquet")
    return res


def strat_diff(X, Cc, y, edges):
    """칸(|충격|)별 (버스트 평균 − 대조 평균)을 버스트 수로 가중. 1h 블록 부트스트랩(두 집합 같은 블록 재표본)."""
    K = len(edges) - 1
    bx = np.clip(np.digitize(np.abs(X.impact), edges[1:-1]), 0, K - 1); bc = np.clip(np.digitize(np.abs(Cc.impact), edges[1:-1]), 0, K - 1)
    vx, vc = X[y].to_numpy(), Cc[y].to_numpy()
    mx_, mc_ = np.isfinite(vx), np.isfinite(vc)
    blocks = np.r_[X.blk.to_numpy()[mx_], Cc.blk.to_numpy()[mc_]]
    isx = np.r_[np.ones(mx_.sum(), bool), np.zeros(mc_.sum(), bool)]
    bins = np.r_[bx[mx_], bc[mc_]]; vals = np.r_[vx[mx_], vc[mc_]]

    def f(w):
        tot, wt = 0.0, 0.0
        for k in range(K):
            a_ = isx & (bins == k); c_ = ~isx & (bins == k)
            wa, wc = w[a_].sum(), w[c_].sum()
            if wa == 0 or wc == 0: continue
            tot += wa * ((w[a_] * vals[a_]).sum() / wa - (w[c_] * vals[c_]).sum() / wc); wt += wa
        return tot / wt if wt else np.nan
    return bboot(blocks, f, B=500)


def fm2(r):
    return f"{r[0]:+6.2f} [{r[1]:+6.2f},{r[2]:+6.2f}]" if np.isfinite(r[0]) else "n/a"


# ── 자체점검 ─────────────────────────────────────────────────────────────────
def selftest():
    tick = 100
    snap = {"lastUpdateId": 10, "bids": [[f"{100 - j * 0.01:.2f}", "1"] for j in range(30)],   # 100.00 .. 99.71 (±29bp)
            "asks": [[f"{100.01 + j * 0.01:.2f}", "1"] for j in range(30)]}

    def file(u0, diffs, with_snap=True):
        L = [json.dumps({"_snapshot": dict(snap, lastUpdateId=u0), "at_ms": 1_000_000})] if with_snap else []
        u = u0
        for n, (b, a) in enumerate(diffs):
            L.append(json.dumps({"e": "depthUpdate", "T": 1_000_000 + n * 500, "U": u if n == 0 else u + 1, "u": u + 1, "pu": u, "b": b, "a": a}))
            u += 1
        return L, u
    st = new_state(20_000)
    f1, u = file(10, [([["99.55", "7"]], []), ([], []), ([], [["100.40", "3"]])])   # 스냅샷 밖(45bp·40bp) 레벨을 diff 가 채움
    r1, c1 = depth_file(iter(f1), tick, st)
    assert c1["mode"] == "reset" and c1["status"] == "ok" and r1[0][0] == 1000, (c1, r1[:1])
    last = r1[-1]                                   # 초 1001 끝: ±10bp = 11레벨·10레벨, ±50bp 는 +7(매수 45bp)·+3(매도 40bp)
    assert last[3] == 10 and last[4] == 10 and last[7] == 30 + 7 and last[8] == 30 + 3, last
    # 다음 파일: pu 가 이어지면 스냅샷(먼 레벨 없음)을 무시하고 이어 붙인다 -> 45bp 레벨 유지
    f2 = [json.dumps({"_snapshot": dict(snap, lastUpdateId=u + 5), "at_ms": 2_000_000})] + \
         [json.dumps({"e": "depthUpdate", "T": 2_000_000, "U": u + 1, "u": u + 1, "pu": u, "b": [], "a": []})]
    r2, c2 = depth_file(iter(f2), tick, st)
    assert c2["mode"] == "carry" and r2[-1][7] == 37 and abs(c2["carry25"] - 1) < 1e-9, (c2, r2)
    # 끊긴 파일 -> 리셋(먼 레벨 사라짐, reset_s 갱신)
    f3, _ = file(500, [([], [])])
    r3, c3 = depth_file(iter(f3), tick, st)
    assert c3["mode"] == "reset" and r3[-1][7] == 30 and r3[-1][-1] == 1000, (c3, r3)
    # 분 프레임: 피쳐는 s-1 까지, 타깃은 s+60 부터 -- 미래 구간만 흔들면 피쳐는 그대로
    n = 3 * 86400
    sec = np.arange(1_700_000_000 - 1_700_000_000 % 86400, 1_700_000_000 - 1_700_000_000 % 86400 + n)
    rng = np.random.default_rng(0)
    ml = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    P = pd.DataFrame({"mf": ml, "ml": ml, "mx": ml, "mn": ml, "spr": 1.0, "bb": ml - .005, "ba": ml + .005,
                      **{f"{s}{k}": 100.0 + rng.random(n) for k in BANDS for s in "ba"}, "reset_s": sec[0] - 7200,
                      "buy": 1.0, "sell": 0.5, "n": 3.0}, index=sec)
    F1 = minute_frame(P)
    P2 = P.copy(); cut = sec[0] + 2 * 86400 + 600
    P2.loc[cut:, ["ml", "mx", "mn", "mf"]] *= 1.05; P2.loc[cut:, "b25"] = 1.0
    F2 = minute_frame(P2)
    row1, row2 = F1[F1.s == cut].iloc[0], F2[F2.s == cut].iloc[0]
    feats = [c for c in F1.columns if not c.startswith(("y_", "exc_"))]
    assert all(np.allclose(row1[c], row2[c], equal_nan=True) for c in feats), "피쳐가 결정 시각 이후를 봤다"
    r_prev = F2[F2.s == cut - 60].iloc[0]           # 이 행의 타깃 창은 [cut, ...) -> 점프(5%)를 본다
    assert r_prev.y_big == 1 and row2.y_big == 0, (r_prev.y_big, row2.y_big)
    assert np.isnan(F1.zD25.iloc[0]) and np.isfinite(F1.zD25.iloc[-1])     # 앞선 날 2일 미만이면 z 없음
    # bboot: 평균 추정
    g = np.repeat(np.arange(20), 5); x = np.arange(100.0)
    r = bboot(g, lambda w: (w * x).sum() / w.sum(), B=200)
    assert abs(r[0] - 49.5) < 1e-9 and r[1] < 49.5 < r[2]
    # 버스트 진입 시점: 초 s+2 첫 mid
    Bf = burst_frame(P.iloc[:100])
    o = Bf["outcome"](np.array([10]), np.array([1.0]))
    assert np.isclose(o["conc"][0], np.log(P.mf.iloc[12] / P.ml.iloc[10]) * 1e4)
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build"); ap.add_argument("--a"); ap.add_argument("--c")
    ap.add_argument("--case", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.build:
        build(a.build)
    elif a.a:
        analyze_a(a.a)
    elif a.c:
        analyze_c(a.c)
    elif a.case:
        case_plot()
    else:
        ap.print_help(); sys.exit(1)
