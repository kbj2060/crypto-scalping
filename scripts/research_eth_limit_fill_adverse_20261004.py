"""지금 최우선가에 지정가를 걸면 (a) N초 안에 체결되나 (b) 체결되면 역선택인가 (c) 미체결이면 얼마를 잃나
-> 지정가/시장가 선택 규칙이 «항상 지정가(T초 뒤 미체결이면 시장가)»보다 건당 실행비용을 줄이는가. 사전등록 (2026-10-04).

사용자: «호가와 지정가와 시장가 체결로 예측할 수 있는 건 더 없나?» -> B(실행 품질). 신호와 무관한 일반 실행 결정.
원천(로컬 사본, 읽기 전용 -- 바이낸스 호출 없음):
  bookTicker   : data/live/orderflow/bookticker/<SYM>/<UTC시>.bt(.gz)  (없으면 ~/backups/crypto-scalping-server/... 사본)
  ETH 체결     : data.binance.vision 일별 aggTrades zip (RL 세션이 받아 둔 사본, LFA_AGG_DIR) -- 틱 단위 «뚫림»·큐 판정
  ETH 깊이     : RL 세션 1초 패널 panel_ext.parquet 의 dd_bid/ask10·50 (DepthBook = research_rt5_1s_panel_build_20260920.dd_file 과 같은 계산)
  BTC 체결     : lake tape(1초 x 가격칸) -- 가격칸째 뚫림(보수)만. BTC 깊이 = dd_file(TICK=10) 재생 -> OUT/dd_BTC.parquet
  섀도우 대조  : 서버 maker_fill_shadow(_ethusdc).duckdb 의 static 다리(OUT/static_legs*.parquet, 서버에서 사본으로 추출)
실행:
  python scripts/research_eth_limit_fill_adverse_20261004.py --selftest
  python scripts/research_eth_limit_fill_adverse_20261004.py --build ETH        # -> OUT/orders_ETH.parquet
  python scripts/research_eth_limit_fill_adverse_20261004.py --build-dd BTC     # BTC 깊이(시간 걸림)
  python scripts/research_eth_limit_fill_adverse_20261004.py --build BTC
  python scripts/research_eth_limit_fill_adverse_20261004.py --calib            # 시뮬 체결 vs 섀도우 static 다리(USDT·USDC)
  python scripts/research_eth_limit_fill_adverse_20261004.py --analyze ETH      # DEV 학습 -> HOLDOUT 한 번
시점 경계: 결정 초 s 의 피쳐는 s_end=(s+1)*1000ms 미만(호가는 s_end 이하) 원천만. 주문 도착 T0 = s_end + lat_ms.
           체결·결과는 T0 «초과» 체결/호가만. 기준가 m0 = T0 의 mid (지정가·시장가 공통).
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
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import data_store as ds  # noqa: E402
from scripts.research_eth_wall_pull_jump_20261004 import BUCKETS, TICKS  # noqa: E402  (테이프 가격칸·호가 틱 규약)

OUT = ROOT / "tmp/limit_fill_adverse_20261004"
_BK = Path.home() / "backups/crypto-scalping-server/data/live/orderflow"
# 🔴워크트리 data/live·data/lake 는 다른 세션이 받아 둔 «일부» 사본일 수 있다 -> 서버 전체 백업을 먼저 쓴다
ORDERFLOW = Path(os.getenv("LFA_ORDERFLOW", _BK if _BK.exists() else ds.ORDERFLOW))
LAKE = Path(os.getenv("LFA_LAKE", ds._BACKUP_LAKE if ds._BACKUP_LAKE.exists() else ds.LAKE))
RL = Path(os.getenv("LFA_RL_DIR", Path.home() / "crypto-scalping/.claude/worktrees/reinforcement-learning-trading-42c07e"
                                    "/data/research/rl_1s_agent_20261002"))
AGG = RL / "aggtrades"
T = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp())  # noqa: E731  (초)

# ── 사전등록 (결과 보기 전에 고정) ──────────────────────────────────────────────
CRITERIA = dict(
    grid_s=10,                 # 10초마다 매수·매도 각 1건 가상 주문(최우선가 지정가)
    lat_ms=450,                # 결정(초 마감) -> 주문 도착. RL 집행 표와 같은 값(정산 350 + 계산 55 + 왕복 40)
    unit={"ETH": 1.0, "BTC": 0.025},   # 주문 수량(큐 판정용). BTC 는 가격칸 규칙만 써서 무관
    Ts=(10, 30, 60), main_T=30,        # 체결 대기 T초, 미체결이면 T초 뒤 시장가
    marks=(10, 60),                    # 역선택 = 체결 시각부터 h초 뒤 mid 의 불리한 이동
    maker_fee_bp=0.0, taker_fee_bp=4.0,  # USDC 계정 실요율(maker_fill_shadow_worker.py ETHUSDC 0/4)
    rules={"ETH": ("through", "queue", "bin"), "BTC": ("bin",)},
    # through = 내 가격보다 «나쁜» 가격의 테이커 체결이 나오면 체결(틱 단위, 보수 -- 주 규칙)
    # queue   = through ∪ 내 가격 정확히의 테이커 체결 누적 ≥ 도착 시 앞 잔량 + 내 수량 ∪ 반대 호가가 내 가격까지 옴(섀도우 워커 규칙)
    # bin     = lake 테이프 가격칸째 뚫림(칸 = BUCKETS), T0 이후 «온전한» 초만, 체결 시각 = 그 초 끝(가장 보수)
    main_rule={"ETH": "through", "BTC": "bin"},
    split={"ETH": ("2026-09-15", "2026-09-25T05", "2026-10-02"),     # DEV 앞 60% / HOLDOUT 뒤 40%(최신)
           "BTC": ("2026-09-26T10", "2026-09-30T09", "2026-10-03")},
    hgb=dict(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=200, random_state=20261004),
    # 결정 ③ = 예측 지정가 비용(HGB 회귀, DEV 학습) < 결정 시각에 아는 시장가 비용(반스프레드 + 테이커 4bp) 이면 지정가, 아니면 시장가.
    # 판정(주: ETH·through·T=30·HGB): HOLDOUT 건당 비용 ③−② 평균 < 0 이고 1시간 블록 부트스트랩 95% CI 상한 < 0 이면 통과.
    # 보조(판정 아님): T=10·60, queue·bin 규칙, 선형(Ridge) 회귀, BTC 재현. 결과 지표: AUC·보정(10분위·Brier)·역선택 IC.
    boot=2000, seed=20261004,
)
FEATS = ["side", "spread_bp", "lq_own", "lq_opp", "qi_own", "imb10", "imb50", "ldep10",
         "flow1", "flow5", "flow30", "lvol30", "ln30", "ret1", "ret5", "ret30", "ret300", "rv300", "hour_sin", "hour_cos"]
TMAX = max(CRITERIA["Ts"])


# ── 원천 적재 ─────────────────────────────────────────────────────────────────
def load_bt(coin: str, lo_s: int, hi_s: int) -> dict:
    sym = f"{coin}USDT"
    h0 = pd.Timestamp(lo_s - 3600, unit="s").strftime("%Y-%m-%dT%H")
    h1 = pd.Timestamp(hi_s + 3600, unit="s").strftime("%Y-%m-%dT%H")
    fs = sorted(p for p in (ORDERFLOW / "bookticker" / sym).iterdir() if h0 <= p.name[:13] <= h1)
    a = np.concatenate([ds.read_bt(p) for p in fs]) if fs else np.zeros(0, ds.BT_DTYPE)
    a = a[(a["bid_px"] > 0) & (a["ask_px"] > a["bid_px"])]
    a = a[np.argsort(a["ts_ms"], kind="stable")]
    tk = TICKS[coin]
    return dict(ts=a["ts_ms"].astype(np.int64), bid=np.round(a["bid_px"] * tk).astype(np.int64),
                ask=np.round(a["ask_px"] * tk).astype(np.int64), bq=a["bid_qty"].astype("f8"), aq=a["ask_qty"].astype("f8"),
                mid=(a["bid_px"] + a["ask_px"]) / 2)


def load_agg(day: str, coin: str = "ETH") -> pd.DataFrame:
    z = AGG / f"{coin}USDT-aggTrades-{day}.zip"
    if not z.exists():
        return pd.DataFrame(columns=["ts", "tick", "qty", "sell"])
    t = pd.read_csv(z, usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
    return pd.DataFrame({"ts": t.transact_time.to_numpy(np.int64), "tick": np.round(t.price.to_numpy() * TICKS[coin]).astype(np.int64),
                         "qty": t.quantity.to_numpy(), "sell": t.is_buyer_maker.to_numpy(bool)})


def load_tape(coin: str, d0: str, d1: str) -> pd.DataFrame:
    t = ds.read("binance", "tape", coin, d0, d1, columns="ts_sec, price_bin, buy_qty, sell_qty, buy_n, sell_n", lake=LAKE)
    return t.groupby(["ts_sec", "price_bin"], as_index=False).sum().sort_values(["ts_sec", "price_bin"], kind="stable")


# ── 1초 배열 + 피쳐 (결정 초 s: s_end 미만 원천만) ─────────────────────────────
def sec_arrays(bt: dict, base: int, n: int, flows: pd.DataFrame) -> dict:
    """flows: 열 sec·buy·sell·cnt (초 s = [s*1000, s*1000+999] ms). 초별 마지막 mid = s_end 미만 마지막 호가."""
    sec = base + np.arange(n)
    i = np.searchsorted(bt["ts"], (sec + 1) * 1000, "left") - 1
    ok = (i >= 0) & (((sec + 1) * 1000 - bt["ts"][np.clip(i, 0, None)]) <= 2000)
    M = np.where(ok, bt["mid"][np.clip(i, 0, None)], np.nan)
    B, S, N = (np.zeros(n) for _ in range(3))
    f = flows[(flows.sec >= base) & (flows.sec < base + n)]
    k = (f.sec - base).to_numpy()
    np.add.at(B, k, f.buy.to_numpy()); np.add.at(S, k, f.sell.to_numpy()); np.add.at(N, k, f.cnt.to_numpy())
    return dict(base=base, M=M, B=B, S=S, N=N, cB=np.concatenate([[0], np.cumsum(B)]),
                cS=np.concatenate([[0], np.cumsum(S)]), cN=np.concatenate([[0], np.cumsum(N)]))


def features(sa: dict, bt: dict, dd: pd.DataFrame, s: np.ndarray, side: np.ndarray) -> pd.DataFrame:
    """s(초)·side(+1 매수/−1 매도) -> 쪽 기준(own = 내가 서는 쪽) 피쳐. 전부 s_end=(s+1)*1000 미만/이하."""
    s_end = (s + 1) * 1000
    i = np.searchsorted(bt["ts"], s_end, "right") - 1          # 호가: s_end 이하 마지막
    ok = (i >= 0) & (s_end - bt["ts"][np.clip(i, 0, None)] <= 2000)
    i = np.clip(i, 0, None)
    assert np.all(bt["ts"][i][ok] <= s_end[ok]), "피쳐 호가가 s_end 를 넘었다"
    buy = side > 0
    q_own = np.where(buy, bt["bq"][i], bt["aq"][i]); q_opp = np.where(buy, bt["aq"][i], bt["bq"][i])
    f = pd.DataFrame({"side": side.astype(float)})
    f["spread_bp"] = np.where(ok, (bt["ask"][i] - bt["bid"][i]) / (bt["ask"][i] + bt["bid"][i]) * 2e4, np.nan)
    f["lq_own"], f["lq_opp"] = np.log1p(q_own), np.log1p(q_opp)
    f["qi_own"] = (q_own - q_opp) / (q_own + q_opp)
    d = dd.reindex(s)
    for b in (10, 50):
        f[f"imb{b}"] = side * ((d[f"dd_bid{b}"] - d[f"dd_ask{b}"]) / (d[f"dd_bid{b}"] + d[f"dd_ask{b}"])).to_numpy()
    f["ldep10"] = np.log1p((d.dd_bid10 + d.dd_ask10).to_numpy())
    f.loc[(d.dd_valid != 1).to_numpy(), ["imb10", "imb50", "ldep10"]] = np.nan
    j = s - sa["base"] + 1                                        # 누적합 끝(초 s 포함)
    win = lambda c, k: sa[c][j] - sa[c][j - k]  # noqa: E731
    for k in (1, 5, 30):
        b_, s_ = win("cB", k), win("cS", k)
        f[f"flow{k}"] = side * (b_ - s_) / (b_ + s_ + 1e-9)
    f["lvol30"] = np.log1p(win("cB", 30) + win("cS", 30)); f["ln30"] = np.log1p(win("cN", 30))
    lm = np.log(sa["M"])
    for k in (1, 5, 30, 300):
        f[f"ret{k}"] = side * (lm[j - 1] - lm[j - 1 - k]) * 1e4
    r1 = np.diff(lm) * 1e4
    cs, cs2, cn = (np.concatenate([[0], np.nancumsum(x)]) for x in (r1, r1 ** 2, np.isfinite(r1).astype(float)))
    a, b = j - 1 - 299, j - 1                                    # r1[k] = lm[k+1]-lm[k] -> 초 s-299..s 의 수익 299개
    n_ = cn[b] - cn[a]
    f["rv300"] = np.where(n_ >= 150, np.sqrt(np.maximum((cs2[b] - cs2[a]) / n_ - ((cs[b] - cs[a]) / n_) ** 2, 0)), np.nan)
    h = (s % 86400) / 86400 * 2 * np.pi
    f["hour_sin"], f["hour_cos"] = np.sin(h), np.cos(h)
    return f


# ── 체결·결과 (T0 초과 원천만) ────────────────────────────────────────────────
def first_fill(rule: str, side: int, p: int, ahead: float, T0: int, bt: dict, tr: dict | None, tp: dict | None,
               unit: float, bucket_ratio: float) -> float:
    """내 지정가(틱 정수 p)가 (T0, T0+TMAX초] 안에 처음 체결된 시각(ms), 없으면 nan."""
    hi = T0 + TMAX * 1000
    out = np.inf
    if rule in ("through", "queue"):
        a, b = np.searchsorted(tr["ts"], T0, "right"), np.searchsorted(tr["ts"], hi, "right")
        ts, tk, q, sl = tr["ts"][a:b], tr["tick"][a:b], tr["qty"][a:b], tr["sell"][a:b]
        agg = sl if side > 0 else ~sl                               # 내 지정가를 칠 수 있는 테이커 쪽
        thr = agg & ((tk < p) if side > 0 else (tk > p))
        k = np.flatnonzero(thr)
        if len(k):
            out = ts[k[0]]
        if rule == "queue":
            cum = np.cumsum(np.where(agg & (tk == p), q, 0.0))
            k = np.flatnonzero(cum >= ahead + unit)
            if len(k):
                out = min(out, ts[k[0]])
            a, b = np.searchsorted(bt["ts"], T0, "right"), np.searchsorted(bt["ts"], hi, "right")
            cross = (bt["ask"][a:b] <= p) if side > 0 else (bt["bid"][a:b] >= p)
            k = np.flatnonzero(cross)
            if len(k):
                out = min(out, bt["ts"][a + k[0]])
    elif rule == "bin":
        s0 = -(-T0 // 1000)                                          # T0 이후 온전한 첫 초
        a, b = np.searchsorted(tp["sec"], s0, "left"), np.searchsorted(tp["sec"], hi // 1000, "right")
        pb = p / bucket_ratio                                        # 내 가격의 칸 좌표(칸 = round(px/bucket))
        cond = (tp["sell"][a:b] > 0) & (tp["bin"][a:b] < np.round(pb)) if side > 0 else \
               (tp["buy"][a:b] > 0) & (tp["bin"][a:b] > np.round(pb))
        k = np.flatnonzero(cond)
        if len(k):
            t = tp["sec"][a + k[0]] * 1000 + 999
            out = t if t <= hi else np.inf
    assert not np.isfinite(out) or out > T0, "체결이 T0 이전 원천을 썼다"
    return float(out) if np.isfinite(out) else np.nan


def at(bt: dict, t: np.ndarray, key: str) -> np.ndarray:
    """t 이하 마지막 호가 값(2초 넘게 비면 nan)."""
    i = np.searchsorted(bt["ts"], t, "right") - 1
    ok = (i >= 0) & (t - bt["ts"][np.clip(i, 0, None)] <= 2000)
    return np.where(ok, bt[key][np.clip(i, 0, None)].astype(float), np.nan)


def outcomes(coin: str, s: np.ndarray, side: np.ndarray, bt: dict, tr, tp, rules) -> pd.DataFrame:
    tk, fm, ft = TICKS[coin], CRITERIA["maker_fee_bp"], CRITERIA["taker_fee_bp"]
    T0 = (s + 1) * 1000 + CRITERIA["lat_ms"]
    buy = side > 0
    bid, ask, m0 = at(bt, T0, "bid"), at(bt, T0, "ask"), at(bt, T0, "mid")
    bq, aq = at(bt, T0, "bq"), at(bt, T0, "aq")
    p = np.where(buy, bid, ask); ahead = np.where(buy, bq, aq)
    o = pd.DataFrame({"T0": T0, "p": p / tk, "m0": m0, "ahead": ahead})
    pm0 = np.where(buy, ask, bid) / tk
    o["cost_M"] = side * (pm0 / m0 - 1) * 1e4 + ft
    o["fillc"] = side * (p / tk / m0 - 1) * 1e4 + fm
    for Tn in CRITERIA["Ts"]:
        pT = np.where(buy, at(bt, T0 + Tn * 1000, "ask"), at(bt, T0 + Tn * 1000, "bid")) / tk
        o[f"fbc{Tn}"] = side * (pT / m0 - 1) * 1e4 + ft               # 미체결 -> T초 뒤 시장가(수수료 포함)
        o[f"opp{Tn}"] = side * (pT / pm0 - 1) * 1e4                   # 기회비용: 지금 시장가 대비
    br = BUCKETS[coin] * tk
    for r in rules:
        tf = np.array([first_fill(r, int(sd), int(pp), aa, int(t0), bt, tr, tp, CRITERIA["unit"][coin], br)
                       if np.isfinite(pp) else np.nan for sd, pp, aa, t0 in zip(side, p, ahead, T0)])
        o[f"lat_{r}"] = tf - T0
        ok = np.isfinite(tf)
        tfi = np.where(ok, tf, 0).astype(np.int64)
        for h in CRITERIA["marks"]:
            o[f"adv{h}_{r}"] = np.where(ok, side * (at(bt, tfi + h * 1000, "mid") / (p / tk) - 1) * 1e4, np.nan)
    return o


# ── 하루 단위 빌드 ───────────────────────────────────────────────────────────
def build_day(args):
    coin, day, dd_path = args
    lo, hi = T(day), T(day) + 86400
    bt = load_bt(coin, lo - 400, hi + 200)
    if not len(bt["ts"]):
        return pd.DataFrame()
    nxt, prv = (pd.Timestamp(day) + pd.Timedelta(days=k) for k in (1, -1))
    tape = load_tape(coin, prv.strftime("%Y-%m-%d"), (nxt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    tape = tape[(tape.ts_sec >= lo - 400) & (tape.ts_sec < hi + 200)]
    tp = dict(sec=tape.ts_sec.to_numpy(), bin=tape.price_bin.to_numpy(), buy=tape.buy_qty.to_numpy(), sell=tape.sell_qty.to_numpy())
    tr = None
    if coin == "ETH":
        a = pd.concat([x for x in (load_agg(d.strftime("%Y-%m-%d")) for d in (prv, pd.Timestamp(day), nxt)) if len(x)])
        a = a[(a.ts >= (lo - 400) * 1000) & (a.ts < (hi + 200) * 1000)].sort_values("ts", kind="stable")
        tr = {c: a[c].to_numpy() for c in a.columns}
        fl = a.assign(sec=a.ts // 1000, buy=np.where(a.sell, 0.0, a.qty), sell=np.where(a.sell, a.qty, 0.0), cnt=1.0) \
              .groupby("sec", as_index=False)[["buy", "sell", "cnt"]].sum()
    else:
        fl = tape.assign(sec=tape.ts_sec, buy=tape.buy_qty, sell=tape.sell_qty, cnt=tape.buy_n + tape.sell_n) \
                 .groupby("sec", as_index=False)[["buy", "sell", "cnt"]].sum()
    dd = pd.read_parquet(dd_path, columns=["dd_valid", "dd_bid10", "dd_ask10", "dd_bid50", "dd_ask50"])
    dd = dd[(dd.index >= lo - 400) & (dd.index < hi)]
    dd = dd[~dd.index.duplicated(keep="last")]
    base = lo - 400
    sa = sec_arrays(bt, base, 86400 + 600, fl)
    g = np.arange(lo, hi, CRITERIA["grid_s"])
    s = np.repeat(g, 2); side = np.tile([1, -1], len(g))
    F = features(sa, bt, dd, s, side)
    O = outcomes(coin, s, side, bt, tr, tp, CRITERIA["rules"][coin])
    # 체결 원천 커버리지: 그 시각 테이프(lake)·aggTrades 가 비면 «미체결»로 오판 -> 표시해서 뺀다
    # (결과 창 T0+120초가 다음 시각으로 넘어가는 끝 2분은 다음 시각 커버도 요구 -- ponytail: 시각 단위 근사)
    hrs = lambda u: np.isin(s // 3600, u) & np.isin((s + 180) // 3600, u)  # noqa: E731
    cov_tape = hrs(np.unique(tape.ts_sec // 3600)) if len(tape) else np.zeros(len(s), bool)
    cov_agg = hrs(np.unique(tr["ts"] // 3_600_000)) if tr is not None else np.ones(len(s), bool)
    D = pd.concat([pd.DataFrame({"s": s, "coin": coin, "cov_tape": cov_tape, "cov_agg": cov_agg}), F.drop(columns="side"), O], axis=1)
    D["side"] = side
    print(f"{coin} {day}: {len(D)} 주문 · 커버 테이프 {cov_tape.mean():.3f} · aggTrades {cov_agg.mean():.3f}", flush=True)
    return D


def build(coin: str):
    lo, _, hi = CRITERIA["split"][coin]
    days = pd.date_range(pd.Timestamp(lo).floor("D"), pd.Timestamp(hi) - pd.Timedelta(seconds=1), freq="D").strftime("%Y-%m-%d")
    dd_path = RL / "panel_ext.parquet" if coin == "ETH" else OUT / f"dd_{coin}.parquet"
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as ex:
        D = pd.concat(list(ex.map(build_day, [(coin, d, dd_path) for d in days])), ignore_index=True)
    D = D[(D.s >= T(lo)) & (D.s < T(hi))]
    D.to_parquet(OUT / f"orders_{coin}.parquet")


def _dd_job(args):
    coin, path = args
    from scripts import research_rt5_1s_panel_build_20260920 as pb
    pb.TICK = TICKS[coin]                                            # 모듈 전역(dd_file·_snapshot_row 가 호출 시 읽는다)
    return pb.dd_file(str(path))


def build_dd(coin: str):
    lo, _, hi = CRITERIA["split"][coin]
    h0, h1 = pd.Timestamp(lo).strftime("%Y-%m-%dT%H"), pd.Timestamp(hi).strftime("%Y-%m-%dT%H")
    fs = sorted(p for p in (ORDERFLOW / "depthdiff" / f"{coin}USDT").iterdir() if h0 <= p.name[:13] <= h1)
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as ex:
        dd = pd.concat([d for d in ex.map(_dd_job, [(coin, f) for f in fs]) if len(d)])
    dd = dd[~dd.index.duplicated(keep="last")].sort_index()
    dd[["dd_valid", "dd_bid10", "dd_ask10", "dd_bid50", "dd_ask50"]].to_parquet(OUT / f"dd_{coin}.parquet")
    print(f"dd_{coin}: {len(dd)} 초 · 유효 {dd.dd_valid.mean():.3f}")


# ── 섀도우 static 다리 대조 (시뮬 체결 규칙의 자 검증) ─────────────────────────
def calib():
    tk = TICKS["ETH"]
    rows = []
    for name in ("static_legs", "static_legs_usdc"):
        L = pd.read_parquet(OUT / f"{name}.parquet")
        L = L[(L.arrival_ex_ts >= T("2026-09-15") * 1000) & (L.arrival_ex_ts < T("2026-10-02") * 1000) & (L.fill_mode != "aborted_stale")]
        for day, g in L.groupby(L.arrival_ex_ts // 86_400_000):
            d = pd.Timestamp(day * 86400, unit="s")
            lo = int(day * 86400)
            bt = load_bt("ETH", lo - 400, lo + 86400 + 200)
            a = pd.concat([load_agg(x.strftime("%Y-%m-%d")) for x in (d, d + pd.Timedelta(days=1))]).sort_values("ts", kind="stable")
            tr = {c: a[c].to_numpy() for c in a.columns}
            for r in g.itertuples():
                side = 1 if r.side == "buy" else -1
                T0 = int(r.arrival_ex_ts) + 200                          # 섀도우 LATENCY_MS
                # 섀도우 다리의 실제 가격(USDC 는 USDC 호가)이 아니라 같은 시각 USDT 최우선가로 시뮬 -- 데이터=USDT, 주문=USDC 규약 대조
                i = np.searchsorted(bt["ts"], int(r.arrival_ex_ts), "right") - 1
                p = int(bt["bid"][i] if side > 0 else bt["ask"][i]); ah = float(bt["bq"][i] if side > 0 else bt["aq"][i])
                sim = {k: first_fill(k, side, p, ah, T0, bt, tr, None, 1.0, 1.0) - r.arrival_ex_ts for k in ("through", "queue")}
                # 섀도우가 체결로 본 순간: aggTrades 에 ±30ms 체결이 있었나 · 보이는 호가가 내 가격을 지나갔나(진단)
                tf = int(r.arrival_ex_ts + (r.fill_t_ms if r.filled else 0))
                near = np.searchsorted(tr["ts"], tf + 30, "right") - np.searchsorted(tr["ts"], tf - 30, "left") > 0
                j = np.searchsorted(bt["ts"], tf, "right") - 1
                thr = (bt["bid"][j] < p) if side > 0 else (bt["ask"][j] > p)
                rows.append(dict(src=name, mode=r.fill_mode, filled=bool(r.filled), lat=r.fill_t_ms if r.filled else np.nan,
                                 agg_near=bool(near), book_thr=bool(thr), **sim))
    C = pd.DataFrame(rows)
    out = {}
    for src, g in C.groupby("src"):
        out[src] = {f"≤{t}s": dict(shadow=float((g.lat <= t * 1000).mean()), sim_through=float((g.through <= t * 1000).mean()),
                                   sim_queue=float((g.queue <= t * 1000).mean())) for t in (5,) + CRITERIA["Ts"]}
        out[src]["n"] = len(g)
        out[src]["by_mode"] = g.groupby("mode").agg(n=("lat", "size"), agg_near=("agg_near", "mean"), book_thr=("book_thr", "mean"),
                                                    sim_q_le_shadow_1s=("queue", lambda x: float((x <= g.loc[x.index, "lat"] + 1000).mean()))
                                                    ).round(3).to_dict("index")
        print(src, json.dumps(out[src], ensure_ascii=False))
    (OUT / "calib.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


# ── 분석 ─────────────────────────────────────────────────────────────────────
def boot_mean(x, blk, B=None, seed=None):
    """1시간 블록 부트스트랩 평균의 95% CI."""
    B = B or CRITERIA["boot"]
    d = pd.DataFrame({"x": x, "b": blk}).dropna()
    g = d.groupby("b").x.agg(["sum", "count"])
    rng = np.random.default_rng(seed or CRITERIA["seed"])
    k = rng.integers(len(g), size=(B, len(g)))
    bs = g["sum"].to_numpy()[k].sum(1) / g["count"].to_numpy()[k].sum(1)
    return float(d.x.mean()), *np.percentile(bs, [2.5, 97.5]).tolist()


def boot_auc(y, p, blk, B=500):
    from sklearn.metrics import roc_auc_score
    d = pd.DataFrame({"y": y, "p": p, "b": blk}).dropna()
    ub = d.b.unique(); grp = {b: i for b, i in d.groupby("b").indices.items()}
    rng = np.random.default_rng(CRITERIA["seed"])
    bs = []
    for _ in range(B):
        idx = np.concatenate([grp[b] for b in rng.choice(ub, len(ub))])
        yy = d.y.to_numpy()[idx]
        if 0 < yy.mean() < 1:
            bs.append(roc_auc_score(yy, d.p.to_numpy()[idx]))
    return float(roc_auc_score(d.y, d.p)), *np.percentile(bs, [2.5, 97.5]).tolist()


def spearman_boot(y, p, blk, B=500):
    d = pd.DataFrame({"y": y, "p": p, "b": blk}).dropna()
    d["ry"], d["rp"] = d.y.rank(), d.p.rank()
    ub = d.b.unique(); grp = d.groupby("b").indices
    rng = np.random.default_rng(CRITERIA["seed"])
    bs = [np.corrcoef(*d[["ry", "rp"]].to_numpy()[np.concatenate([grp[b] for b in rng.choice(ub, len(ub))])].T)[0, 1] for _ in range(B)]
    return float(np.corrcoef(d.ry, d.rp)[0, 1]), *np.percentile(bs, [2.5, 97.5]).tolist()


def analyze(coin: str):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    D = pd.read_parquet(OUT / f"orders_{coin}.parquet")
    lo, mid, hi = (T(x) for x in CRITERIA["split"][coin])
    n0 = len(D)
    D = D[D.cov_agg & (D.cov_tape if coin != "ETH" else True) & D.m0.notna() & D.cost_M.notna() & D[[f"fbc{t}" for t in CRITERIA["Ts"]]].notna().all(1)
          & D.ret300.notna() & D.rv300.notna()].copy()
    D["blk"] = D.s // 3600
    dev, hol = D[D.s < mid], D[D.s >= mid]
    res = dict(coin=coin, n_raw=n0, n_dev=len(dev), n_hol=len(hol), hours_dev=int(dev.blk.nunique()), hours_hol=int(hol.blk.nunique()),
               window=[CRITERIA["split"][coin][0], CRITERIA["split"][coin][2]])
    print(f"\n=== {coin}: 주문 {n0} -> 유효 {len(D)} (DEV {len(dev)} / HOLDOUT {len(hol)}, {res['hours_hol']}h)")
    DEV, HOL = dev, hol
    sub = lambda r: (DEV[DEV.cov_tape], HOL[HOL.cov_tape]) if r == "bin" else (DEV, HOL)  # noqa: E731  (가격칸 규칙은 lake 테이프 있는 시각만)
    X = lambda d: d[FEATS].to_numpy(float)  # noqa: E731
    hgb_c = lambda: HistGradientBoostingClassifier(**CRITERIA["hgb"])  # noqa: E731
    hgb_r = lambda: HistGradientBoostingRegressor(**CRITERIA["hgb"])   # noqa: E731
    from sklearn.impute import SimpleImputer
    lin_c = lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))  # noqa: E731
    lin_r = lambda: make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0))  # noqa: E731

    # (a) 체결확률
    print("\n(a) T초 안 체결 -- 기저율(DEV/HOLD) · HOLDOUT AUC [블록 CI] · Brier · 10분위 보정(예측→실측)")
    res["fill"] = {}
    for r in CRITERIA["rules"][coin]:
        dev, hol = sub(r)
        for Tn in CRITERIA["Ts"]:
            y = lambda d: (d[f"lat_{r}"] <= Tn * 1000).astype(float)  # noqa: E731
            row = dict(base_dev=float(y(dev).mean()), base_hol=float(y(hol).mean()))
            for nm, mk in (("hgb", hgb_c), ("logit", lin_c)):
                m = mk().fit(X(dev), y(dev))
                ph = m.predict_proba(X(hol))[:, 1]
                yh = y(hol).to_numpy()
                q = pd.qcut(ph, 10, labels=False, duplicates="drop")
                cal = pd.DataFrame({"p": ph, "y": yh, "q": q}).groupby("q").mean()
                row[nm] = dict(auc=boot_auc(yh, ph, hol.blk.to_numpy()), brier=float(np.mean((ph - yh) ** 2)),
                               brier_base=float(np.mean((y(dev).mean() - yh) ** 2)),
                               ece=float(np.average(np.abs(cal.p - cal.y))), calib=[[float(a), float(b)] for a, b in zip(cal.p, cal.y)])
            res["fill"][f"{r}_T{Tn}"] = row
            h = row["hgb"]
            print(f"  {r:7s} T={Tn:2d}s 기저 {row['base_dev']:.3f}/{row['base_hol']:.3f} · HGB AUC {h['auc'][0]:.3f} [{h['auc'][1]:.3f},{h['auc'][2]:.3f}]"
                  f" · 로짓 {row['logit']['auc'][0]:.3f} · Brier {h['brier']:.4f} (기저 {h['brier_base']:.4f}) · ECE {h['ece']:.3f}")
            if r == CRITERIA["main_rule"][coin] and Tn == CRITERIA["main_T"]:
                print("    10분위 (예측, 실측):", " ".join(f"({a:.2f},{b:.2f})" for a, b in h["calib"]))

    # (b) 역선택: 체결(60초 안) 다리의 체결 뒤 h초 mid 이동(+ = 유리)
    print("\n(b) 역선택 -- 체결 뒤 h초 mid 이동(bp, 내 쪽 + 유리) · 예측 IC [블록 CI] · P(불리) AUC · 예측 5분위 실측")
    res["adverse"] = {}
    for r in CRITERIA["rules"][coin]:
        dev, hol = sub(r)
        for h in CRITERIA["marks"]:
            c = f"adv{h}_{r}"
            fd, fh = dev[dev[c].notna()], hol[hol[c].notna()]
            m = hgb_r().fit(X(fd), fd[c]); ph = m.predict(X(fh))
            mc = hgb_c().fit(X(fd), (fd[c] < 0).astype(float)); pc = mc.predict_proba(X(fh))[:, 1]
            qq = pd.qcut(ph, 5, labels=False, duplicates="drop")
            row = dict(mean_dev=float(fd[c].mean()), mean_hol=boot_mean(fh[c].to_numpy(), fh.blk.to_numpy()),
                       p_adverse_hol=float((fh[c] < 0).mean()), ic=spearman_boot(fh[c].to_numpy(), ph, fh.blk.to_numpy()),
                       auc_adverse=boot_auc((fh[c] < 0).astype(float).to_numpy(), pc, fh.blk.to_numpy()),
                       quint=pd.Series(fh[c].to_numpy()).groupby(qq).mean().round(3).tolist())
            res["adverse"][c] = row
            print(f"  {r:7s} +{h:2d}s 평균 DEV {row['mean_dev']:+.2f} · HOLD {row['mean_hol'][0]:+.2f} [{row['mean_hol'][1]:+.2f},{row['mean_hol'][2]:+.2f}]"
                  f" · P(불리) {row['p_adverse_hol']:.2f} · IC {row['ic'][0]:+.3f} [{row['ic'][1]:+.3f},{row['ic'][2]:+.3f}]"
                  f" · AUC {row['auc_adverse'][0]:.3f} · 5분위 {row['quint']}")

    # (c) 미체결 기회비용
    print("\n(c) 미체결 기회비용(지금 시장가 대비 T초 뒤 시장가, bp, + = 비싸짐)")
    res["opp"] = {}
    for r in CRITERIA["rules"][coin]:
        dev, hol = sub(r)
        for Tn in CRITERIA["Ts"]:
            u = hol[~(hol[f"lat_{r}"] <= Tn * 1000)]
            res["opp"][f"{r}_T{Tn}"] = dict(share=float(len(u) / len(hol)), mean=boot_mean(u[f"opp{Tn}"].to_numpy(), u.blk.to_numpy()))
            v = res["opp"][f"{r}_T{Tn}"]
            print(f"  {r:7s} T={Tn:2d}s 미체결 {v['share']:.3f} · 기회비용 {v['mean'][0]:+.2f} [{v['mean'][1]:+.2f},{v['mean'][2]:+.2f}]")

    # 결정 규칙 3종
    print("\n결정 규칙 -- HOLDOUT 건당 실행비용(bp, 기준 = 도착 mid, 수수료·역선택·미체결 포함) [1h 블록 CI]")
    res["decision"] = {}
    for r in CRITERIA["rules"][coin]:
        dev, hol = sub(r)
        for Tn in CRITERIA["Ts"]:
            cL = lambda d: np.where(d[f"lat_{r}"] <= Tn * 1000, d.fillc, d[f"fbc{Tn}"])  # noqa: E731
            known_M = lambda d: d.spread_bp / 2 + CRITERIA["taker_fee_bp"]               # noqa: E731
            out = {}
            for nm, mk in (("hgb", hgb_r), ("ridge", lin_r)):
                m = mk().fit(X(dev), cL(dev))
                pick_L = m.predict(X(hol)) < known_M(hol).to_numpy()
                c3 = np.where(pick_L, cL(hol), hol.cost_M)
                out[nm] = dict(share_limit=float(pick_L.mean()), cost=boot_mean(c3, hol.blk.to_numpy()),
                               d32=boot_mean(c3 - cL(hol), hol.blk.to_numpy()))
            c1, c2 = hol.cost_M.to_numpy(), cL(hol)
            row = dict(always_market=boot_mean(c1, hol.blk.to_numpy()), always_limit=boot_mean(c2, hol.blk.to_numpy()),
                       d21=boot_mean(c2 - c1, hol.blk.to_numpy()), oracle=boot_mean(np.minimum(c1, c2), hol.blk.to_numpy()),
                       dev_always_limit=float(cL(dev).mean()), **out)
            res["decision"][f"{r}_T{Tn}"] = row
            main = r == CRITERIA["main_rule"][coin] and Tn == CRITERIA["main_T"]
            f = lambda x: f"{x[0]:+6.3f} [{x[1]:+6.3f},{x[2]:+6.3f}]"  # noqa: E731
            print(f"  {r:7s} T={Tn:2d}s {'(주)' if main else '    '} ①시장가 {f(row['always_market'])} · ②지정가 {f(row['always_limit'])}"
                  f" · ③HGB {f(row['hgb']['cost'])} (지정가 {row['hgb']['share_limit']:.1%}) · ③−② {f(row['hgb']['d32'])}"
                  f" · Ridge ③−② {f(row['ridge']['d32'])} (지정가 {row['ridge']['share_limit']:.1%}) · 상한(오라클) {f(row['oracle'])}")
            if main:
                # 해석용: ③ 이 시장가를 거의 안 고르는 이유 -- 예측 비용·직전 30초 추격(ret30) 10분위별 실현 지정가 비용 vs 시장가
                m = hgb_r().fit(X(dev), cL(dev)); ph = m.predict(X(hol))
                for nm, key in (("예측 지정가 비용", ph), ("직전30초 이동(내 쪽 + = 추격)", hol.ret30.to_numpy())):
                    q = pd.qcut(key, 10, labels=False, duplicates="drop")
                    t = pd.DataFrame({"k": key, "L": cL(hol), "M": hol.cost_M.to_numpy(), "f": (hol[f"lat_{r}"] <= Tn * 1000).to_numpy()}).groupby(q).mean()
                    row[f"dec_{'pred' if nm.startswith('예측') else 'ret30'}"] = t.round(3).values.tolist()
                    print(f"    {nm} 10분위 (키, 실현 지정가, 시장가, 체결률):", " ".join(f"({a:+.2f},{l:.2f},{mm:.2f},{f:.2f})" for a, l, mm, f in t.values))
                d = row["hgb"]["d32"]
                res["pass"] = bool(d[0] < 0 and d[2] < 0)
                print(f"  ==> 판정(주): ③−② {f(d)} -> {'통과' if res['pass'] else '불통과'}")
    # 피쳐 중요도(주 체결확률 모델, HOLDOUT 순열) -- 해석용
    from sklearn.inspection import permutation_importance
    r, Tn = CRITERIA["main_rule"][coin], CRITERIA["main_T"]
    dev, hol = sub(r)
    m = hgb_c().fit(X(dev), (dev[f"lat_{r}"] <= Tn * 1000).astype(float))
    sub = hol.sample(min(len(hol), 40000), random_state=0)
    pi = permutation_importance(m, X(sub), (sub[f"lat_{r}"] <= Tn * 1000).astype(float), scoring="roc_auc", n_repeats=3, random_state=0)
    res["perm_auc"] = dict(sorted(zip(FEATS, pi.importances_mean.round(4).tolist()), key=lambda x: -x[1]))
    print("\n체결확률(주) 순열 중요도(AUC 하락):", res["perm_auc"])
    (OUT / f"result_{coin}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    return res


# ── 자체점검 ─────────────────────────────────────────────────────────────────
def selftest():
    tk = 100
    # 호가: 100ms 마다 bid 100.00 / ask 100.01, 잔량 5/7. t=20s 부터 ask 가 100.00 로 내려옴(매수 지정가 quote cross)
    ts = np.arange(0, 40_000, 100, dtype=np.int64)
    bid = np.full(len(ts), 10000); ask = np.where(ts >= 20_000, 10000, 10001); bid = np.where(ts >= 20_000, 9999, bid)
    bt = dict(ts=ts, bid=bid, ask=ask, bq=np.full(len(ts), 5.0), aq=np.full(len(ts), 7.0), mid=(bid + ask) / 2 / tk)
    # 체결: 매도 테이커 at 100.00 2개씩(t=3,5,7s) -> 누적 6 ≥ 앞 5 + 1 => 큐 체결 7s. 뚫림 매도 at 99.99 t=12s. T0 이하 뚫림(t=1.0s)은 무시돼야.
    tr = dict(ts=np.array([1000, 3000, 5000, 7000, 12_000], np.int64), tick=np.array([9999, 10000, 10000, 10000, 9999], np.int64),
              qty=np.array([9.0, 2.0, 2.0, 2.0, 1.0]), sell=np.array([True] * 5))
    T0 = 1000
    assert first_fill("through", 1, 10000, 5.0, T0, bt, tr, None, 1.0, 10.0) == 12_000
    assert first_fill("queue", 1, 10000, 5.0, T0, bt, tr, None, 1.0, 10.0) == 7_000
    assert first_fill("queue", 1, 10000, 50.0, T0, bt, tr, None, 1.0, 10.0) == 12_000           # 앞 잔량 크면 뚫림 시각
    tr2 = dict(tr, tick=np.where(tr["tick"] == 9999, 10000, tr["tick"]))                          # 뚫림 없음
    assert first_fill("queue", 1, 10000, 50.0, T0, bt, tr2, None, 1.0, 10.0) == 20_000          # quote cross
    assert np.isnan(first_fill("through", 1, 10000, 50.0, T0, bt, tr2, None, 1.0, 10.0))
    assert np.isnan(first_fill("through", -1, 10001, 7.0, T0, bt, tr, None, 1.0, 10.0))          # 매도 지정가는 매도 테이커로 안 참
    # 가격칸(칸 = 10틱 = $0.1): 매수 100.00 -> 칸 1000. 칸 999 매도는 뚫림, 칸 1000 은 아님. T0=1000 -> 첫 온전한 초 = 1
    tp = dict(sec=np.array([0, 1, 2, 4]), bin=np.array([999, 1000, 1000, 999]), buy=np.zeros(4), sell=np.ones(4))
    assert first_fill("bin", 1, 10000, 0, 1000, bt, None, tp, 1.0, 10.0) == 4999
    assert first_fill("bin", 1, 10000, 0, 1001, bt, None, tp, 1.0, 10.0) == 4999
    tpe = dict(sec=np.array([0, 1]), bin=np.array([999, 1000]), buy=np.zeros(2), sell=np.ones(2))
    assert np.isnan(first_fill("bin", 1, 10000, 0, 500, bt, None, tpe, 1.0, 10.0))               # T0 가 걸친 초 0 의 뚫림은 못 쓴다
    # 시점 경계: 피쳐는 s_end 이후 원천을 바꿔도 같고, 결과는 T0 이하 원천을 바꿔도 같다
    rng = np.random.default_rng(0)
    n = 4000
    ts = np.arange(0, n * 1000, 250, dtype=np.int64)
    b0 = 10000 + np.cumsum(rng.integers(-1, 2, len(ts)))
    bt = dict(ts=ts, bid=b0, ask=b0 + 1, bq=rng.uniform(1, 9, len(ts)), aq=rng.uniform(1, 9, len(ts)), mid=(2 * b0 + 1) / 2 / tk)
    fl = pd.DataFrame({"sec": np.arange(n), "buy": rng.uniform(0, 5, n), "sell": rng.uniform(0, 5, n), "cnt": rng.integers(1, 9, n)})
    dd = pd.DataFrame({"dd_valid": 1, "dd_bid10": rng.uniform(1, 9, n), "dd_ask10": rng.uniform(1, 9, n),
                       "dd_bid50": rng.uniform(1, 9, n), "dd_ask50": rng.uniform(1, 9, n)}, index=np.arange(n))
    s = np.array([1000, 1000, 2000]); side = np.array([1, -1, 1])
    F1 = features(sec_arrays(bt, 0, n, fl), bt, dd, s, side)
    bt2 = {k: v.copy() for k, v in bt.items()}
    cut = np.searchsorted(ts, (s.min() + 1) * 1000, "right")                                     # s_end 초과 호가를 바꿈
    bt2["bid"][cut:] += 50; bt2["ask"][cut:] += 50; bt2["mid"][cut:] += 0.5; bt2["bq"][cut:] *= 3
    fl2 = fl.copy(); fl2.loc[fl2.sec > s.min(), ["buy", "sell"]] *= 7
    dd2 = dd.copy(); dd2.loc[dd2.index > s.min(), "dd_bid10"] *= 5
    F2 = features(sec_arrays(bt2, 0, n, fl2), bt2, dd2, s, side)
    pd.testing.assert_frame_equal(F1.iloc[:2], F2.iloc[:2])                                      # s=1000 행은 그대로
    assert not F1.iloc[2].equals(F2.iloc[2])                                                      # s=2000 행은 바뀐다(양성 대조)
    T0 = (1000 + 1) * 1000 + CRITERIA["lat_ms"]
    trr = dict(ts=np.array([T0 - 5, T0, T0 + 5]), tick=np.array([0, 0, 0]), qty=np.ones(3), sell=np.array([True] * 3))
    assert first_fill("through", 1, 10, 0, T0, bt, trr, None, 1.0, 10.0) == T0 + 5                 # T0 이하 체결은 못 쓴다
    O = outcomes("ETH", np.array([1000]), np.array([1]), bt, trr, None, ["through"])
    assert O.lat_through.iloc[0] == 5.0, O
    # 비용 회계: 매수 시장가 = 반스프레드 + 4, 지정가 체결 = −반스프레드
    hs = (bt["ask"][0] - bt["bid"][0]) / 2 / tk
    O = outcomes("ETH", np.array([1000]), np.array([1]), bt, trr, None, ["through"])
    m0 = O.m0.iloc[0]
    assert abs(O.cost_M.iloc[0] - (hs / m0 * 1e4 + 4)) < 1e-6 and abs(O.fillc.iloc[0] + hs / m0 * 1e4) < 1e-6
    # 블록 부트스트랩: 상수 −1 이면 CI = [−1, −1]
    r = boot_mean(np.full(100, -1.0), np.repeat(np.arange(10), 10), B=200)
    assert r == (-1.0, -1.0, -1.0), r
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build"); ap.add_argument("--build-dd"); ap.add_argument("--analyze")
    ap.add_argument("--calib", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.build:
        build(a.build)
    elif a.build_dd:
        build_dd(a.build_dd)
    elif a.calib:
        calib()
    elif a.analyze:
        analyze(a.analyze)
    else:
        ap.print_help(); sys.exit(1)
