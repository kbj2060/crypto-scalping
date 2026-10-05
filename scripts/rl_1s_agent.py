"""ETH 매초 강화학습 매매 에이전트 (2026-10-02, 사용자 지시 -- 모의 매매 전용, 실주문 없음).

참고: huggingface.co/JonusNattapong/Reinforcement-Learning-for-Gold-Trading-Model
  = Stable-Baselines3 PPO · MlpPolicy · VecNormalize · lr 3e-4 · batch 256 · γ .99 · λ .95 · clip .2 · ent .01 · 1M 스텝.
  그대로 따르되 바꾼 것: 15분봉 → 1초 결정(γ .999 ≈ 17분 지평) · 연속 크기 → 목표 포지션 {숏, 없음, 롱}(사용자 선택)
  · 오프라인 1회 학습 → 사전학습 후 실시간 계속 학습(live).

결정: 초 s 가 닫히고 LAT_MS 뒤(T)에 목표 포지션을 고른다. 다르면 그 순간 최우선 호가에 지정가(매수=bid, 매도=ask).
체결(큐 모델, 2026-08-22 메이커 시뮬이 쓴 보수적 규칙 = OOS 실측 3.47~3.59bp/leg 가 예측 밴드 안):
  걸 때 그 가격의 보이는 잔량이 전부 내 앞이다. 다음 결정까지 (T, T+1초] 의 틱 체결 중
  ① 내 가격을 뚫은 체결이 있거나 ② 내 가격 정확히의 체결량이 «앞 잔량 + 내 수량»을 넘으면 체결.
  앞 사람의 취소는 무시(큐가 안 줄어든다 = 보수적). 같은 가격이 계속 최우선이면 주문을 유지해 줄 자리를 잇고,
  최우선이 바뀌면 새 최우선으로 다시 건다(peg, 줄 맨 뒤).
보상(bp): 보유분 시가평가(결정 시각 mid) + 체결분 (다음 결정 mid / 체결가 − 1) − 체결 수량 × FEE_BP.

학습·실전 피쳐 동일성: 두 경로 모두 같은 «1초 패널» 열을 만들고 같은 features() 를 부른다.
  hot 저장소가 늦게 쓰는 원천(체결 5초·맥락 10초 flush)은 학습에서도 같은 만큼 늦춰서 본다(TAPE_LAG·CTX_LAG).

  python scripts/rl_1s_agent.py selftest
  python scripts/rl_1s_agent.py build  --start 2026-09-20 --end 2026-09-30
  python scripts/rl_1s_agent.py build-exec --start 2026-09-20 --end 2026-09-30   # aggtrades/ 에 일별 zip 필요
  python scripts/rl_1s_agent.py calib
  python scripts/rl_1s_agent.py train  --seed 123
  python scripts/rl_1s_agent.py eval   --seed 123
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import deque
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import data_store as ds  # noqa: E402

SYMBOL = __import__("os").getenv("RL_SYMBOL", "ETHUSDT").upper()   # 2026-10-04 SOL·XRP 판(사용자 «ETH 에 있는 모의 판을 다른 코인도») -- 프로세스 하나 = 코인 하나
COIN = SYMBOL[:-4]
TAG = "" if SYMBOL == "ETHUSDT" else f"_{COIN.lower()}"   # 산출 파일 접미사 -- ETH 는 옛 이름 그대로(paper/ · panel.parquet)
# 코인마다 다른 값만 여기. px = 가격 정수화 배율(호가 1틱 = 1) · unit = 주문 1단위(ETH 1개와 같은 명목 ~$2,700) ·
#   whale/retail = 고정 경계(대시보드·테이프 수집기 SIZE_BANDS_USD 와 같은 값 -- ETH 거래량 비중 10/43/47% 에 맞춘 것) ·
#   wall_q·pullback·liq = «학습 7일» 분위(ETH 09-20~26 · SOL·XRP 09-27~10-03, calib-coin 으로 잰다) · poi·er = 4.7년 분위(R2 검정).
COINCFG = {
    "ETHUSDT": dict(px=100, unit=1.0, idx_max=100_000 * 100, whale=100_000.0, retail=10_000.0,
                    wall_q={70: 0.1002, 80: 0.1294, 90: 0.1807}, pullback=7.22, liq=(12.732, 13.21), poi=-26.68, er=14.95),
    # wall_q·pullback·liq = calib-coin 09-27~10-04(ETH 같은 식 재현: 0.1006/0.1299/0.1811 · 12.74/13.211 · 7.2) ·
    #   poi·er = 4.7년 같은 정의(docs/experiments/trend4_poi1h_solxrp_20261004.md, ETH −26.68·14.95 재현)
    "SOLUSDT": dict(px=100, unit=22.0, idx_max=2_000 * 100, whale=55_000.0, retail=7_500.0,
                    wall_q={70: 0.0766, 80: 0.0957, 90: 0.1242}, pullback=9.16, liq=(11.347, 10.995), poi=-42.96, er=21.75),
    "XRPUSDT": dict(px=10_000, unit=1_800.0, idx_max=20 * 10_000, whale=23_000.0, retail=3_400.0,
                    wall_q={70: 0.0894, 80: 0.1098, 90: 0.1399}, pullback=9.43, liq=(11.296, 9.866), poi=-32.54, er=16.54),
}
CFG = COINCFG[SYMBOL]
PX = CFG["px"]
FEE_BP = 0.0              # USDC 메이커 0% (maker_fill_shadow_worker.py:49-53 계정 실요율). 데이터는 USDT, 주문은 USDC
TAKER_FEE_BP = 4.0        # USDC 테이커 (maker_fill_shadow_worker.py:54). 1 ETH 는 최우선 잔량(중앙 89 ETH)에 비해 작아 미끄러짐 0
TAKER_FEE_BP = 4.0        # USDC 테이커 (maker_fill_shadow_worker.py:54). 1 ETH 는 최우선 잔량(중앙 89 ETH)에 비해 작아 미끄러짐 0
UNIT = CFG["unit"]        # 주문 1단위 = 1 ETH (큐 소진 비교용) -- 다른 코인은 같은 명목
BAR = 300                 # 진입 결정 주기(초) = 5분봉
BAR = 300                 # 진입 결정 주기(초) = 5분봉
LAT_MS = 450              # 초 마감 -> 주문 도착: SETTLE_S 350 + 관측 계산 ~55 + 왕복 ~40
TAPE_LAG = 8              # hot 체결 테이프 flush 5초 + 완결 대기 -> 학습도 8초 늦은 체결만 본다
CTX_LAG = 12              # hot OI·마크·청산 flush 10초
BANDS = (5, 10, 25, 50)
WARMUP = 4200             # features() 의 가장 긴 창(3600) + 5분봉 + 지연
OUT = ROOT / "data/research/rl_1s_agent_20261002"
PANEL = __import__("os").getenv("RL_PANEL", f"panel{TAG}.parquet")   # 학습·평가 패널(기본 = 모의 매매 엔진과 같은 것). panel_zh = 제우스·호메로스 열 추가판
_BACKUP_OF = Path.home() / "backups/crypto-scalping-server/data/live/orderflow"
ORDERFLOW = ds.ORDERFLOW if ds.ORDERFLOW.exists() else _BACKUP_OF
SEED_OF = Path(__import__("os").getenv("RL_SEED_ORDERFLOW", str(ORDERFLOW)))   # 2026-10-05 라이브 시작 이력(bookTicker·depth) 폴더 -- 없으면 웹소켓으로 WARMUP 만큼 모은다
SPLIT = {"train": ("2026-09-20", "2026-09-27"), "val": ("2026-09-27", "2026-09-28"), "test": ("2026-09-28", "2026-09-30")}


def _ts(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp())


# ── 1초 패널: 원천마다 초 단위로 ────────────────────────────────────────────
def bt_agg(a: np.ndarray) -> pd.DataFrame:
    """bookTicker 행(ts_ms·bid_px·bid_qty·ask_px·ask_qty) -> 초별. 라이브는 WS 메시지로 같은 배열을 만든다."""
    d = pd.DataFrame({"sec": a["ts_ms"] // 1000, "bid": a["bid_px"], "ask": a["ask_px"],
                      "bq": a["bid_qty"].astype("f8"), "aq": a["ask_qty"].astype("f8")})
    d = d[(d.bid > 0) & (d.ask > d.bid)]
    mid = (d.bid + d.ask) / 2
    d["mid"] = mid
    d["qi"] = (d.bq - d.aq) / (d.bq + d.aq)
    d["micro"] = ((d.bid * d.aq + d.ask * d.bq) / (d.bq + d.aq) - mid) / mid * 1e4
    return d.groupby("sec").agg(bid=("bid", "last"), ask=("ask", "last"), mid=("mid", "last"),
                                qi_last=("qi", "last"), qi_mean=("qi", "mean"), micro_bp=("micro", "mean"),
                                mid_hi=("mid", "max"), mid_lo=("mid", "min"))


class DepthBook:
    """@depth@100ms 리플레이(research_rt5_1s_panel_build_20260920.dd_file 과 같은 계산 -- selftest 가 대조).
    feed() 가 초 경계를 넘을 때 직전 초의 행을 돌려준다(그 초 마지막 메시지 이후 상태)."""
    TICK, IDX_MAX = PX, CFG["idx_max"]

    def __init__(self, snap: dict):
        self.bid = np.zeros(self.IDX_MAX)
        self.ask = np.zeros(self.IDX_MAX)
        for p, q in snap["bids"]:
            self.bid[int(round(float(p) * self.TICK))] = float(q)
        for p, q in snap["asks"]:
            self.ask[int(round(float(p) * self.TICK))] = float(q)
        self.last_u = snap["lastUpdateId"]
        self.best = [int(np.flatnonzero(self.bid)[-1]), int(np.flatnonzero(self.ask)[0])]
        self.synced, self.valid, self.cur = False, 1, None
        self._acc()

    def _acc(self):
        self.acc = dict(add_b=0.0, rem_b=0.0, add_a=0.0, rem_a=0.0)

    def feed(self, m: dict) -> list[dict]:
        if m.get("e") != "depthUpdate":
            return []
        if not self.synced:
            if m["u"] < self.last_u:
                return []
            if not (m["U"] <= self.last_u <= m["u"]):
                self.valid = 0
            self.synced = True
        elif m["pu"] != self.last_u:
            self.valid = 0
        self.last_u = m["u"]
        sec, out = m["E"] // 1000, []
        if self.cur is None:
            self.cur = sec
        while sec > self.cur:
            out.append(self.row())
            self._acc()
            self.cur += 1
        for side, arr, ka, kr in (("b", self.bid, "add_b", "rem_b"), ("a", self.ask, "add_a", "rem_a")):
            for p, q in m[side]:
                i = int(round(float(p) * self.TICK))
                if i >= self.IDX_MAX:
                    continue
                q = float(q)
                dq = q - arr[i]
                if dq > 0:
                    self.acc[ka] += dq
                else:
                    self.acc[kr] -= dq
                arr[i] = q
        return out

    def row(self) -> dict:
        bid, ask, best, T = self.bid, self.ask, self.best, self.TICK
        w = max(int(best[0] * 0.02), 100)
        nb = np.flatnonzero(bid[max(best[0] - w, 0):best[0] + w + 1])
        bb = (max(best[0] - w, 0) + nb[-1]) if len(nb) else (np.flatnonzero(bid)[-1] if bid.any() else -1)
        na = np.flatnonzero(ask[max(best[1] - w, 0):best[1] + w + 1])
        ba = (max(best[1] - w, 0) + na[0]) if len(na) else (np.flatnonzero(ask)[0] if ask.any() else -1)
        if bb < 0 or ba < 0:
            return dict(sec=self.cur, dd_valid=0)
        best[0], best[1] = int(bb), int(ba)
        mid = (bb + ba) / 2 / T
        r = dict(sec=self.cur, dd_valid=self.valid, **{f"dd_{k}": v for k, v in self.acc.items()})
        for bp in BANDS:
            w = int(mid * bp / 1e4 * T)
            r[f"dd_bid{bp}"] = bid[max(bb - w, 0):bb + 1].sum()
            r[f"dd_ask{bp}"] = ask[ba:ba + w + 1].sum()
        w = int(mid * 50 / 1e4 * T)
        sb, sa = bid[max(bb - w, 0):bb + 1], ask[ba:ba + w + 1]
        r["dd_wall_b"], r["dd_wall_b_bp"] = sb.max(), (len(sb) - 1 - sb.argmax()) / T / mid * 1e4
        r["dd_wall_a"], r["dd_wall_a_bp"] = sa.max(), sa.argmax() / T / mid * 1e4
        return r


def dd_hour(path: Path) -> pd.DataFrame:
    rows, book = [], None
    with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as f:
        for i, line in enumerate(f):
            try:
                m = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):   # 크래시·쓰는 중 잘린 줄
                if book:
                    book.valid = 0
                continue
            if i == 0:
                snap = m.get("_snapshot") if isinstance(m, dict) else None
                if not snap:
                    return pd.DataFrame()
                book = DepthBook(snap)
            elif book:
                rows += book.feed(m)
    if book and book.cur is not None:
        rows.append(book.row())
    return pd.DataFrame(rows).set_index("sec") if rows else pd.DataFrame()


def bt_hour(path: Path) -> pd.DataFrame:
    return bt_agg(ds.read_bt(path))


def tape_agg(t: pd.DataFrame) -> pd.DataFrame:
    """테이프 행(lake·hot 같은 열) -> 초별. 체결 없는 초는 행이 없다(호출 쪽이 0 으로)."""
    g = t.groupby("ts_sec").agg(tr_buy=("buy_qty", "sum"), tr_sell=("sell_qty", "sum"),
                                tr_wb=("whale_buy_qty", "sum"), tr_ws=("whale_sell_qty", "sum"),
                                tr_rb=("retail_buy_qty", "sum"), tr_rs=("retail_sell_qty", "sum"),
                                tr_n=("buy_n", "sum"), tr_n2=("sell_n", "sum"))
    g["tr_n"] += g.pop("tr_n2")
    return g


def ctx_agg(oi: pd.DataFrame, mark: pd.DataFrame, liq: pd.DataFrame) -> pd.DataFrame:
    o = oi.assign(sec=oi.ts_ms // 1000).groupby("sec").open_interest.last().rename("oi")
    m = mark.assign(sec=mark.ts_ms // 1000).groupby("sec")[["mark", "index_px", "funding_rate"]].last()
    lq = liq.assign(sec=liq.ts_ms // 1000, lu=np.where(liq.side == "long", liq.usd, 0.0),
                    su=np.where(liq.side == "short", liq.usd, 0.0)).groupby("sec")[["lu", "su"]].sum()
    return pd.concat([o, m, lq.rename(columns={"lu": "liq_long", "su": "liq_short"})], axis=1)


def assemble(lo: int, hi: int, bt, dd, tape, ctx) -> pd.DataFrame:
    p = pd.DataFrame(index=pd.RangeIndex(lo, hi, name="sec"))
    p = p.join(bt).join(dd).join(tape).join(ctx)
    zero = ["tr_buy", "tr_sell", "tr_wb", "tr_ws", "tr_rb", "tr_rs", "tr_n", "liq_long", "liq_short"]
    p[zero] = p[zero].fillna(0.0)
    p[["oi", "mark", "index_px", "funding_rate"]] = p[["oi", "mark", "index_px", "funding_rate"]].ffill(limit=60)
    return p


def build(start: str, end: str) -> pd.DataFrame:
    lo, hi = _ts(start), _ts(end)
    hstart, hend = pd.Timestamp(lo, unit="s").strftime("%Y-%m-%dT%H"), pd.Timestamp(hi, unit="s").strftime("%Y-%m-%dT%H")
    files = lambda s: sorted(p for p in (ORDERFLOW / s / SYMBOL).iterdir() if hstart <= p.name[:13] < hend)
    with ProcessPoolExecutor(max_workers=8) as ex:
        bt = pd.concat(list(ex.map(bt_hour, files("bookticker"))))
        dd = pd.concat([d for d in ex.map(dd_hour, files("depthdiff")) if len(d)])
    bt, dd = bt[~bt.index.duplicated(keep="last")], dd[~dd.index.duplicated(keep="last")]
    tape = tape_agg(ds.read("binance", "tape", COIN, start, end))
    try:
        mark = ds.read("binance", "mark_1s", COIN, start, end)
    except Exception as exc:  # noqa: BLE001 -- ponytail: SOL·XRP 마크는 2026-10-04 부터 수집(lake 에 없음). 규칙 판은 마크를 안 쓴다(basis·funding 피쳐만 0)
        print(f"[build] 마크 원천 없음 -> 빈 열: {type(exc).__name__}", flush=True)
        mark = pd.DataFrame(columns=["ts_ms", "mark", "index_px", "funding_rate"], dtype=float)
    ctx = ctx_agg(ds.read("binance", "oi_1s", COIN, start, end), mark, ds.read("binance", "liquidations", COIN, start, end))
    return assemble(lo, hi, bt, dd, tape, ctx)


# ── 집행 표: 결정 시각 T=(s+1)초+LAT_MS 의 최우선 호가·잔량 + (T, T+1초] 틱 체결 요약 ──────────
EXEC_COLS = ["P", "A", "bq", "aq", "midT", "sv", "smin", "bv", "bmax"]   # 가격은 센트 정수(float), 수량은 ETH


def trades_day(day: str) -> pd.DataFrame:
    """data.binance.vision 일별 aggTrades(무료 아카이브 -- API 가중치·IP 밴과 무관). sell = 테이커 매도(is_buyer_maker)."""
    t = pd.read_csv(OUT / "aggtrades" / f"{SYMBOL}-aggTrades-{day}.zip",
                    usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
    return pd.DataFrame({"ts": t.transact_time.to_numpy(np.int64), "c": np.round(t.price.to_numpy() * PX),
                         "qty": t.quantity.to_numpy(), "sell": t.is_buyer_maker.to_numpy(bool)})


def window_agg(tr: pd.DataFrame, pc: float, ac: float) -> tuple[float, float, float, float]:
    """한 창의 틱 체결 -> (내 매수가 pc 정확히의 테이커 매도량, 최저 매도가, 내 매도가 ac 정확히의 테이커 매수량, 최고 매수가).
    라이브가 창마다 부른다 -- 오프라인 exec_hour 의 벡터 계산과 selftest 가 대조한다."""
    sell = tr.sell.to_numpy(bool)                     # 🔴빈 창이면 object 열이라 tr[tr.sell] 이 «열 고르기»로 읽힌다(10-02 라이브 첫 실행 사고)
    s, b = tr[sell], tr[~sell]
    return (float(s.qty[s.c == pc].sum()), float(s.c.min()) if len(s) else np.inf,
            float(b.qty[b.c == ac].sum()), float(b.c.max()) if len(b) else -np.inf)


def exec_hour(path: Path) -> pd.DataFrame:
    a = ds.read_bt(path)
    a = a[(a["bid_px"] > 0) & (a["ask_px"] > a["bid_px"])]
    h0 = _ts(path.name[:13].replace("T", " ") + ":00")
    s = np.arange(h0, h0 + 3600)
    T = (s + 1) * 1000 + LAT_MS
    i = np.searchsorted(a["ts_ms"], T, side="right") - 1
    ok = (i >= 0) & (T <= a["ts_ms"][-1])                  # 시각 파일 끝을 넘는 T(시간당 ~1초)는 버린다
    i, s, T = i[ok], s[ok], T[ok]
    if not len(s):
        return pd.DataFrame(columns=EXEC_COLS)
    e = pd.DataFrame({"P": np.round(a["bid_px"][i] * PX), "A": np.round(a["ask_px"][i] * PX),
                      "bq": a["bid_qty"][i].astype("f8"), "aq": a["ask_qty"][i].astype("f8"),
                      "midT": (a["bid_px"][i] + a["ask_px"][i]) / 2}, index=pd.Index(s, name="sec"))
    days = sorted({pd.Timestamp(x, unit="s").strftime("%Y-%m-%d") for x in (h0, h0 + 3605)})
    tr = pd.concat([trades_day(d) for d in days if (OUT / "aggtrades" / f"{SYMBOL}-aggTrades-{d}.zip").exists()])
    tr = tr[(tr.ts > T.min()) & (tr.ts <= T.max() + 1000)]
    tr["sec"] = (tr.ts - LAT_MS - 1) // 1000 - 1           # (T_s, T_s+1000] -> s
    tr = tr.join(e[["P", "A"]], on="sec", how="inner")
    sl, by = tr[tr.sell], tr[~tr.sell]
    e["sv"] = sl[sl.c == sl.P].groupby("sec").qty.sum()
    e["smin"] = sl.groupby("sec").c.min()
    e["bv"] = by[by.c == by.A].groupby("sec").qty.sum()
    e["bmax"] = by.groupby("sec").c.max()
    return e.fillna({"sv": 0.0, "bv": 0.0, "smin": np.inf, "bmax": -np.inf})


WHALE_USD, RETAIL_USD = CFG["whale"], CFG["retail"]


# 움직이는 경계(10-03 tmp/whale_moving_tiers_4y.py 의 A): 고정 $100k 의 고래 거래대금 비중이 2022 19% → 2025~26 39% 로 떠서,
#   날마다 직전 30일(그날 제외) 거래대금 중 리테일 10.1%·고래 46.7% 가 되는 금액을 경계로(칸 log10 20칸/10배 경계로 반올림).
TIER_SHARES, TIER_WIN, TIER_PER_DEC = (0.101, 0.467), 30, 20
_CUTS: dict[str, tuple[float, float]] = {}


def day_gross(day: str) -> np.ndarray | None:
    """그날 aggTrade 줄 금액 칸별 총 거래대금(180칸). zip 옆에 .gross.npy 로 캐시, zip 이 없으면 None."""
    z = OUT / "aggtrades" / f"{SYMBOL}-aggTrades-{day}.zip"
    c = z.with_suffix(".gross.npy")
    if c.exists():
        return np.load(c)
    if not z.exists():
        return None
    t = pd.read_csv(z, usecols=["price", "quantity"])
    usd = (t.price * t.quantity).to_numpy()
    b = np.clip(np.floor(np.log10(np.maximum(usd, 1.0)) * TIER_PER_DEC + 1e-9).astype(int), 0, 179)
    g = np.bincount(b, weights=usd, minlength=180)
    np.save(c, g)
    return g


def moving_cuts(day: str) -> tuple[float, float] | None:
    """(리테일 상한, 고래 하한) USD -- 직전 30일 중 아카이브가 있는 날이 20일 미만이면 None."""
    if day not in _CUTS:
        d0 = pd.Timestamp(day)
        g = [x for x in (day_gross(f"{d0 - pd.Timedelta(days=k):%Y-%m-%d}") for k in range(1, TIER_WIN + 1)) if x is not None]
        if len(g) < 20:
            return None
        c = np.concatenate([[0.0], np.cumsum(np.sum(g, axis=0))])
        c /= c[-1]                                                   # c[k] = 칸 k 미만 비중
        sr, sw = TIER_SHARES
        _CUTS[day] = (10 ** (np.abs(c - sr).argmin() / TIER_PER_DEC), 10 ** (np.abs(c - (1 - sw)).argmin() / TIER_PER_DEC))
    return _CUTS[day]


def agg_minutes(price, qty, ts_ms, buyer_maker, cut=None) -> pd.DataFrame:
    """aggTrade 줄들 -> 1분별 고래(줄 금액 ≥$100k)·리테일(<$10k) 테이커 순매수(USD). 아카이브·라이브 WS 공용.
    cut(날짜) -> (리테일 상한, 고래 하한) 을 주면 움직이는 경계판 w2·r2 도(경계 없는 날은 NaN)."""
    usd, sgn = price * qty, np.where(buyer_maker, -1.0, 1.0)
    d = {"m": (ts_ms // 60_000) * 60, "w": np.where(usd >= WHALE_USD, usd, 0.0) * sgn, "r": np.where(usd < RETAIL_USD, usd, 0.0) * sgn}
    if cut:
        day = ts_ms // 86_400_000
        rc, wc = np.full(len(usd), np.nan), np.full(len(usd), np.nan)
        for k in np.unique(day):
            c = cut(f"{pd.Timestamp(int(k) * 86400, unit='s'):%Y-%m-%d}")
            if c:
                rc[day == k], wc[day == k] = c
        ok = np.isfinite(wc)
        d["w2"] = np.where(ok, np.where(usd >= wc, usd, 0.0) * sgn, np.nan)
        d["r2"] = np.where(ok, np.where(usd < rc, usd, 0.0) * sgn, np.nan)
    return pd.DataFrame(d).groupby("m").sum(min_count=1)


def whale_z(g: pd.DataFrame) -> pd.DataFrame:
    """1분별 w·r(빈 분은 0, 모르는 분은 NaN) -> 직전 60분 합의 자기 과거 30일 z. w2·r2(움직이는 경계)가 있으면 zw2·zr2 도."""
    g = g.copy()
    for c in [c for c in ("w", "r", "w2", "r2") if c in g]:
        n60 = g[c].rolling(60).sum()
        g[f"z{c}"] = (n60 - n60.rolling(43200, min_periods=20000).mean()) / n60.rolling(43200, min_periods=20000).std()
    return g


def whale_raw(days=None) -> pd.DataFrame:
    """aggtrades/ 아카이브 -> 1분별 w·r, 연속 구간을 0 으로 채움."""
    rows = []
    for f in sorted((OUT / "aggtrades").glob(f"{SYMBOL}-aggTrades-*.zip")):
        day_gross(f.name[-14:-4])                       # 오름차순이라 뒷날 경계가 쓸 앞날 캐시가 먼저 생긴다
        if days is None or f.name[-14:-4] in days:
            t = pd.read_csv(f, usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
            rows.append(agg_minutes(t.price.to_numpy(), t.quantity.to_numpy(), t.transact_time.to_numpy(np.int64),
                                    t.is_buyer_maker.to_numpy(bool), cut=moving_cuts))
    g = pd.concat(rows).groupby(level=0).sum(min_count=1)
    return g.reindex(pd.RangeIndex(g.index.min(), g.index.max() + 60, 60), fill_value=0.0)


def whale_minutes() -> pd.DataFrame:
    return whale_z(whale_raw())


def ensure_aggtrades(days: int = 62) -> None:                 # 30일 z × 그 첫날의 직전 30일 경계(움직이는 경계판)
    """data.binance.vision 일별 aggTrades(무료 아카이브 -- API 가 아니라 가중치·IP 밴과 무관)를 어제까지 받아 둔다."""
    import urllib.request
    d = OUT / "aggtrades"
    d.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz="UTC").normalize()
    for k in range(1, days + 1):
        name = f"{SYMBOL}-aggTrades-{today - pd.Timedelta(days=k):%Y-%m-%d}.zip"
        if not (d / name).exists():
            try:
                urllib.request.urlretrieve(f"https://data.binance.vision/data/futures/um/daily/aggTrades/{SYMBOL}/{name}",
                                           d / (name + ".part"))
                (d / (name + ".part")).rename(d / name)
            except Exception as exc:                  # 어제 파일은 몇 시간 늦게 올라온다 -- 없으면 라이브 분으로 메운다
                print(f"[aggtrades] {name} 없음: {exc}", flush=True)


def add_whale(p: pd.DataFrame) -> pd.DataFrame:
    """초 s 행에는 s 가 끝난 시점의 «마지막 완결 1분»까지만(1분 = [m, m+60))."""
    g = whale_minutes()
    m = (p.index.to_numpy() + 1) // 60 * 60 - 60
    return p.assign(wz=g.zw.reindex(m).to_numpy(), rz=g.zr.reindex(m).to_numpy())


def add_zeus_homer(p: pd.DataFrame) -> pd.DataFrame:
    """제우스 v4 157열·호메로스 Tier0 21열(tmp/zeus_homer_frames.py, 5분봉 timestamp = 봉 시작)을 초 행에 붙인다.
    초 s 가 끝난 시점의 마지막 봉 경계 B 에서: 호메로스 = 방금 닫힌 봉(B−300 시작) · 제우스 = 한 봉 더 늦춤(B−600 시작,
    OI 스탬프 누수 규약 -- rerun_failed_hypotheses_180feat_20260927 의 «제우스 열 한 봉 늦춤 기본»)."""
    B = (p.index.to_numpy() + 1) // BAR * BAR
    for name, lag, pre in (("zeus_5m", 2 * BAR, "zs_"), ("homer_5m", BAR, "hm_")):
        df = pd.read_parquet(OUT / f"{name}.parquet")
        # 가격 수준 그대로인 5열은 뺀다 -- 학습 7일로는 «이 가격대면 산다»를 외운다(제우스 모델은 자체 스케일러로 썼다)
        v = df.drop(columns=["timestamp"] + [c for c in ("open", "high", "low", "close", "close_btc") if c in df]).astype("float32")
        v.index = ((pd.to_datetime(df.timestamp) - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy()   # 단위(ns/us) 무관
        assert (v.index % BAR == 0).all()
        sub = v.reindex(B - lag)
        sub.columns, sub.index = [pre + c for c in sub.columns], p.index
        p = p.join(sub)
    return p


def build_exec(start: str, end: str) -> pd.DataFrame:
    lo, hi = _ts(start), _ts(end)
    hs, he = pd.Timestamp(lo, unit="s").strftime("%Y-%m-%dT%H"), pd.Timestamp(hi, unit="s").strftime("%Y-%m-%dT%H")
    files = sorted(p for p in (ORDERFLOW / "bookticker" / SYMBOL).iterdir() if hs <= p.name[:13] < he)
    with ProcessPoolExecutor(max_workers=6) as ex:
        e = pd.concat(list(ex.map(exec_hour, files)))
    return e[~e.index.duplicated(keep="last")]


# ── 피쳐: 학습·실전 공용. 행 t 는 t 까지(지연 원천은 t-LAG 까지)만 본다 ───────────────
def features(p: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=p.index)
    lm = np.log(p.mid.ffill(limit=5))
    for w in (1, 5, 30, 300):
        f[f"ret{w}"] = (lm - lm.shift(w)) * 1e4
    f["rv60"] = (lm.diff() * 1e4).rolling(60, min_periods=30).std()
    f["qi"], f["qi_mean"], f["micro_bp"] = p.qi_last, p.qi_mean, p.micro_bp
    f["spread_bp"] = (p.ask - p.bid) / p.mid * 1e4
    for b in BANDS:                                   # 호가 깊이 불균형
        f[f"imb{b}"] = (p[f"dd_bid{b}"] - p[f"dd_ask{b}"]) / (p[f"dd_bid{b}"] + p[f"dd_ask{b}"])
    ofi = (p.dd_add_b - p.dd_rem_b) - (p.dd_add_a - p.dd_rem_a)
    dep = p.dd_bid25 + p.dd_ask25
    f["ofi1"], f["ofi10"] = ofi / dep, ofi.rolling(10, min_periods=5).sum() / dep
    f["wall_ratio"] = np.log(p.dd_wall_b / p.dd_wall_a)
    f["wall_b_bp"], f["wall_a_bp"] = p.dd_wall_b_bp, p.dd_wall_a_bp
    t = p[["tr_buy", "tr_sell", "tr_wb", "tr_ws", "tr_rb", "tr_rs", "tr_n"]].shift(TAPE_LAG)   # 체결·CVD·풋프린트
    vol, delta = t.tr_buy + t.tr_sell, t.tr_buy - t.tr_sell
    for w in (10, 60, 300):
        f[f"cvd{w}"] = delta.rolling(w).sum() / (vol.rolling(w).sum() + 1e-9)
    v60 = vol.rolling(60).sum()
    f["vol60_rel"] = np.log((v60 + 1e-6) / (vol.rolling(3600, min_periods=600).mean() * 60 + 1e-6))
    f["whale60"] = (t.tr_wb - t.tr_ws).rolling(60).sum() / (v60 + 1e-9)
    f["retail60"] = (t.tr_rb - t.tr_rs).rolling(60).sum() / (v60 + 1e-9)
    f["trades60"] = np.log1p(t.tr_n.rolling(60).sum())
    bar = pd.Series(p.index // 300, index=p.index)    # 풋프린트 5분봉: 봉 안 누적 델타 · 봉 고저 안 위치
    f["bar_delta"] = delta.groupby(bar).cumsum() / (vol.groupby(bar).cumsum() + 1e-9)
    hi5, lo5 = p.mid_hi.groupby(bar).cummax(), p.mid_lo.groupby(bar).cummin()
    f["bar_pos"] = (p.mid - lo5) / (hi5 - lo5).replace(0, np.nan)
    hi1h, lo1h = p.mid_hi.rolling(3600, min_periods=600).max(), p.mid_lo.rolling(3600, min_periods=600).min()
    f["dist_hi1h"], f["dist_lo1h"] = (hi1h / p.mid - 1) * 1e4, (p.mid / lo1h - 1) * 1e4   # 지지/저항(1시간 고저)
    c = p[["oi", "mark", "index_px", "funding_rate", "liq_long", "liq_short"]].shift(CTX_LAG)   # 수급·시장 맥락
    for w in (10, 60, 300):
        f[f"doi{w}"] = (c.oi / c.oi.shift(w) - 1) * 1e4
    d300 = c.oi.diff(300)
    f["oi_z"] = d300 / d300.rolling(3600, min_periods=600).std()
    f["basis_bp"] = (c.mark / c.index_px - 1) * 1e4
    f["funding_bp"] = c.funding_rate * 1e4
    f["liq_long60"], f["liq_short60"] = np.log1p(c.liq_long.rolling(60).sum()), np.log1p(c.liq_short.rolling(60).sum())
    h = (p.index % 86400) / 86400 * 2 * np.pi
    f["hour_sin"], f["hour_cos"] = np.sin(h), np.cos(h)
    # 고래↔리테일 맞대결(whale_mid_retail_follow_20260925, aggTrade «한 줄» 정의). 🔴라이브 피드에는 아직 없다(wz·rz) --
    #   LiveFeed 에 @aggTrade(/market/ws/) + 30일 이력 부트스트랩을 붙이기 전엔 여기서 KeyError 로 멈춘다(조용히 0 금지).
    wz, rz = p.wz, p.rz
    f["whale_z"], f["retail_z"] = wz, rz
    f["whale_vs_retail"] = np.where((wz * rz < 0) & (wz.abs() >= 0.5) & (rz.abs() >= 0.5), np.sign(wz), 0.0)
    ext = [c for c in p.columns if c.startswith(("zs_", "hm_"))]   # 제우스·호메로스 5분봉 열(add_zeus_homer) -- 있을 때만 뒤에 붙인다
    if ext:
        f = pd.concat([f, p[ext].astype("float32")], axis=1)
    return f.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


# ── 체결·보상: 학습·실전 공용 순수 함수 ───────────────────────────────────────
def queue_step(order, q: int, P, A, bq, aq, sv, smin, bv, bmax):
    """한 창의 지정가 큐. order = (측면, 가격센트, 앞 잔량) 또는 None, q = 목표 − 포지션.
    돌려줌 (남은 order, 체결가 센트 또는 None). 같은 측면·같은 가격이면 줄 자리를 잇고, 아니면 줄 맨 뒤에 새로 선다.
    sv/bv 는 «이번 창 최우선가 P/A 정확히»의 체결량 -- 주문은 늘 P/A 에 있으므로 그 가격이 곧 내 가격이다."""
    if q == 0:
        return None, None
    side = 1 if q > 0 else -1
    px = P if side > 0 else A
    ahead = order[2] if order and order[0] == side and order[1] == px else (bq if side > 0 else aq)
    through, at = (smin < px, sv) if side > 0 else (bmax > px, bv)
    ahead -= at
    if through or ahead + abs(q) * UNIT <= 0:
        return None, px
    return (side, px, ahead), None


def step_reward(pos: int, q: int, px: float, mid0: float, mid1: float, fill: bool) -> tuple[float, int]:
    r = pos * (mid1 / mid0 - 1) * 1e4
    if fill:
        r += q * (mid1 / px - 1) * 1e4 - abs(q) * FEE_BP
        pos += q
    return r, pos


def is_bar(s: int) -> bool:
    """초 s 가 닫히면 5분봉도 닫힌다 = 진입 결정 시각."""
    return (s + 1) % BAR == 0


class Trader:
    """포지션·주문 상태기계 -- 학습(SecEnv)·실시간(LiveEnv) 공용. e = 결정 시각 호가(P·A 센트, bq·aq, midT)."""
    N_STATE = 7

    def __init__(self):
        self.pos, self.tgt, self.entry, self.order, self.since = 0, 0, 0.0, None, 0
        self.k = 1.0                                # 손익 배수(크기 조절 판만 바꾼다, 1 단위 체결 모델은 그대로)
        self.k_in = 1.0                             # 지금 포지션이 진입 체결 때 받은 배수 -- 원장 bp_k·state unr (10-05 원장 bp 에 k 가 빠져 있었다)
        self.pnl = dict(hold=0.0, maker=0.0, taker=0.0)
        self.n = dict(maker_fills=0, taker_exits=0, secs=0, secs_long=0, secs_short=0)
        self.now, self.t_entry, self.closed = None, None, []   # 원장: 호출 쪽이 now(초)를 넣어 주면 끝난 거래를 closed 에 쌓는다

    def _book(self, prev: int, px: float, kind: str) -> None:
        """포지션이 prev -> self.pos 로 바뀐 체결 하나를 원장에 반영(청산 다리 기록 + 새 진입 시각)."""
        if prev:
            fee = abs(prev) * TAKER_FEE_BP if kind == "taker" else 0.0
            bp = prev * (px / self.entry - 1) * 1e4 - fee   # bp = 1 단위(기존 의미 그대로) · bp_k = 배수 반영(손익 합과 같은 단위)
            self.closed.append(dict(side="long" if prev > 0 else "short", t_in=self.t_entry, px_in=self.entry, t_out=self.now,
                                    px_out=px, exit=kind, bp=round(bp, 3), k=self.k_in, bp_k=round(bp * self.k_in, 3)))
        if self.pos and np.sign(self.pos) != np.sign(prev):
            self.t_entry, self.k_in = self.now, self.k

    def needs_decision(self, s: int) -> bool:
        return is_bar(s) or self.pos != 0

    def decide(self, a: int, bar: bool, e: dict) -> float:
        """봉 경계: a = 목표 숏/없음/롱(지정가). 포지션 보유 중 매초: a = 그대로/지정가 청산/시장가 청산.
        self.idle = 이번 결정이 «포지션 없는데 또 없음»이었나 -- 학습 보상의 관망 벌점용(실손익에는 안 들어간다)."""
        self.idle = bar and self.pos == 0 and int(a) == 1
        if bar:
            self.tgt = int(a) - 1
            return 0.0
        if self.pos == 0:
            return 0.0
        if a == 0:
            pass                                    # 그대로 = 봉 경계에서 정한 목표(청산·전환 지정가 포함)를 계속 쫓는다.
            # 🔴10-02 전엔 «보유 = 청산 지정가 취소»였다 -- 규칙이 봉 경계에서 정한 청산·전환을 다음 초 «보유»가 취소해,
            #   규칙 평가가 «청산 지정가가 1초 안에 안 차면 계속 보유»로 변질돼 있었다.
        elif a == 1:
            self.tgt = 0
        else:                                       # 시장가: 결정 순간 반대편 최우선에 즉시
            px = (e["P"] if self.pos > 0 else e["A"]) / PX
            r = (self.pos * (px / e["midT"] - 1) * 1e4 - abs(self.pos) * TAKER_FEE_BP) * self.k
            self.pnl["taker"] += r
            self.n["taker_exits"] += 1
            prev, self.pos, self.tgt, self.order = self.pos, 0, 0, None
            self._book(prev, px, "taker")
            return r
        return 0.0

    def mark(self, mid0: float, mid1: float) -> None:
        """체결 판정 없이 시가평가만 -- paper 가 tick 사이에 빈 구간(봉 경계 관측 계산·호가 공백)을 잇는다."""
        self.pnl["hold"] += self.pos * (mid1 / mid0 - 1) * 1e4 * self.k

    def tick(self, e: dict, w: tuple, mid1: float) -> float:
        """결정 시각부터 다음 초 결정 시각까지: 지정가 대기열 + 시가평가."""
        q = self.tgt - self.pos
        self.order, fc = queue_step(self.order, q, e["P"], e["A"], e["bq"], e["aq"], *w)
        fill, px, prev = fc is not None, (fc or 0) / PX, self.pos
        h = prev * (mid1 / e["midT"] - 1) * 1e4 * self.k
        r, self.pos = step_reward(prev, q, px, e["midT"], mid1, fill)
        r *= self.k
        self.pnl["hold"] += h
        self.pnl["maker"] += r - h
        if fill:
            self.n["maker_fills"] += abs(q)
            self._book(prev, px, "maker")
            if self.pos and np.sign(self.pos) != np.sign(prev):
                self.entry, self.since = px, 0
        self.since += 1
        self.n["secs"] += 1
        self.n["secs_long"] += self.pos > 0
        self.n["secs_short"] += self.pos < 0
        return r

    def state(self, mid: float, s: int) -> list:
        unr = self.pos * (mid / self.entry - 1) * 1e4 if self.pos else 0.0
        return [self.pos, np.clip(unr, -100, 100) / 10, float(is_bar(s)), ((s + 1) % BAR) / BAR,
                np.log1p(self.since if self.pos else 0) / 8, float(self.pos != 0 and self.tgt != self.pos),
                float(self.pos == 0 and self.tgt != 0)]


# ── 환경 ─────────────────────────────────────────────────────────────────────
def arrays(p: pd.DataFrame, e: pd.DataFrame) -> dict:
    """p = 1초 패널, e = 집행 표(같은 sec 인덱스). mid 는 결정 시각 mid(midT) -- 보상 시가평가 기준."""
    X = features(p).to_numpy()
    e = e.reindex(p.index)
    # 집행 표는 bookTicker 시각 파일마다 만들어 매시 끝 ~1초가 빈다 -> 그 1초를 «데이터 구멍»으로 보고 포지션을 중간가에
    #   공짜로 정리하고 있었다(10-02 원장: 272건 중 77건). 1초 구멍은 직전 호가로 잇고 그 초 체결은 없는 것으로(보수적).
    gap1 = e.P.isna() & e.P.shift(1).notna() & e.P.shift(-1).notna()
    e.loc[gap1, ["P", "A", "bq", "aq", "midT"]] = e[["P", "A", "bq", "aq", "midT"]].shift(1)[gap1]
    e.loc[gap1, ["sv", "smin", "bv", "bmax"]] = [0.0, np.inf, 0.0, -np.inf]
    ok = p.mid.notna() & (p.dd_valid == 1) & e.P.notna()
    ok = ok & ok.shift(-1, fill_value=False)
    ok.iloc[:WARMUP] = False
    out = {c: e[c].to_numpy(float) for c in EXEC_COLS}
    out.update(X=X, mid=out.pop("midT"), mid1=e.midT.shift(-1).to_numpy(float), ok=ok.to_numpy(), s0=int(p.index[0]))
    return out


def make_env_cls():
    import gymnasium as gym

    class SecEnv(gym.Env):
        """스텝 = 결정이 있는 초(봉 경계 또는 포지션 보유 중 매초). obs = 피쳐 + Trader.state.
        보상 = 이 결정부터 다음 결정까지 초들의 손익 합(bp). seq=True 는 순차 1초 전진 평가(구멍은 다음 봉 경계로 건너뜀)."""

        def __init__(self, A: dict, ep_len: int = 3600, seq: bool = False, seed: int = 0, idle_penalty: float = 0.0):
            self.A, self.ep_len, self.seq, s0 = A, ep_len, seq, A["s0"]
            self.idle_penalty = idle_penalty
            self.s0 = s0
            self.rng = np.random.default_rng(seed)
            sec = s0 + np.arange(len(A["ok"]))
            self.starts = np.flatnonzero(A["ok"] & ((sec + 1) % BAR == 0))
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (A["X"].shape[1] + Trader.N_STATE,), np.float32)
            self.action_space = gym.spaces.Discrete(3)
            self.t, self.tr = 0, Trader()

        def _e(self, t):
            A = self.A
            return dict(P=A["P"][t], A=A["A"][t], bq=A["bq"][t], aq=A["aq"][t], midT=A["mid"][t])

        def _w(self, t):
            A = self.A
            return A["sv"][t], A["smin"][t], A["bv"][t], A["bmax"][t]

        def _obs(self):
            return np.append(self.A["X"][self.t], self.tr.state(self.A["mid"][self.t], self.s0 + self.t)).astype(np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.t = int(self.starts[0] if self.seq else self.rng.choice(self.starts))
            self.end = len(self.A["mid"]) - 1 if self.seq else self.t + self.ep_len
            self.tr = Trader()
            return self._obs(), {}

        def step(self, a):
            A, tr = self.A, self.tr
            r = tr.decide(int(a), is_bar(self.s0 + self.t), self._e(self.t)) - self.idle_penalty * tr.idle
            while True:
                t = self.t
                r += tr.tick(self._e(t), self._w(t), A["mid1"][t])
                self.t += 1
                stop = self.t >= self.end or not A["ok"][self.t]
                if stop and self.seq:                  # 구멍: 포지션은 그 자리에서 정리된 것으로(0) 다음 봉 경계부터
                    nxt = self.starts[self.starts >= self.t]
                    if len(nxt):
                        pnl, n = tr.pnl, tr.n
                        self.t, self.tr, stop = int(nxt[0]), Trader(), False
                        self.tr.pnl, self.tr.n = pnl, n
                        break
                if stop or tr.needs_decision(self.s0 + self.t):
                    break
            return self._obs(), float(r), bool(stop), False, {}

    return SecEnv


# ── 학습·평가 ────────────────────────────────────────────────────────────────
def load_split(name: str) -> dict:
    p = pd.read_parquet(OUT / PANEL)
    lo, hi = _ts(SPLIT[name][0]) - WARMUP, _ts(SPLIT[name][1])
    return arrays(p.loc[lo:hi - 1], pd.read_parquet(OUT / "exec.parquet"))


def run_policy(A: dict, act) -> dict:
    """순차 1초씩 전진(fresh-forward). act(obs, t, bar) -> 행동. 손익은 보유·지정가 체결·시장가 청산으로 나눠 센다."""
    env = make_env_cls()(A, seq=True)
    obs, _ = env.reset()
    total, done = 0.0, False
    while not done:
        obs, r, done, _, _ = env.step(act(obs, env.t, is_bar(env.s0 + env.t)))
        total += r
    tr = env.tr
    days = tr.n["secs"] / 86400
    return dict(pnl_bp_day=total / days, **{f"{k}_bp_day": v / days for k, v in tr.pnl.items()},
                maker_fills_day=tr.n["maker_fills"] / days, taker_exits_day=tr.n["taker_exits"] / days,
                long=tr.n["secs_long"] / tr.n["secs"], short=tr.n["secs_short"] / tr.n["secs"], days=days)


def model_dir(seed: int, idle: float = 0.0) -> Path:
    """RL_TAG 환경변수 = 피쳐셋·실험 묶음 이름(다른 묶음의 모델을 덮어쓰지 않게)."""
    import os
    tag = os.getenv("RL_TAG", "")
    return OUT / (f"seed{seed}" + (f"_idle{idle:g}" if idle else "") + (f"_{tag}" if tag else ""))


def train(seed: int, steps: int, idle: float = 0.0) -> Path:
    """idle = 관망 벌점(bp, 학습 보상만). 지정가 왕복 비용(~2bp) 근처면 «실력 없는 무작위 거래 ≈ 관망»이 되어
    관망 수렴(10-02 5시드)을 막는다. 벌점이 거래 손실보다 작으면 무력, 너무 크면 한쪽 쏠림(09-14) -- 평가는 실손익만."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    Env, A = make_env_cls(), load_split("train")
    venv = VecNormalize(DummyVecEnv([lambda i=i: Env(A, seed=seed * 100 + i, idle_penalty=idle) for i in range(8)]),
                        norm_obs=True, norm_reward=False, clip_obs=10.0)

    class EntDecay(BaseCallback):       # 엔트로피 0.01 -> 0.001 감쇠: «탐색해 보니 손해라 관망»과 «탐색이 먼저 죽음»을 가른다
        def _on_step(self):
            self.model.ent_coef = 0.01 + (0.001 - 0.01) * min(1.0, self.num_timesteps / steps)
            return True

    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=2048, batch_size=256, gamma=0.999, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, seed=seed, verbose=0, device="cpu")
    model.learn(total_timesteps=steps, callback=EntDecay())
    d = model_dir(seed, idle)
    d.mkdir(parents=True, exist_ok=True)
    model.save(d / "ppo.zip")
    venv.save(d / "vecnormalize.pkl")
    return d


def wall_thr() -> float:
    """깊은 벽 규칙 문턱 = 학습구간 |imb50| 80분위(0.1294)."""
    return float(np.quantile(np.abs(load_split("train")["X"][:, features_cols().index("imb50")]), 0.8))


def calib_coin(start: str, end: str) -> dict:
    """COINCFG 의 «학습 7일» 값을 이 코인 패널로 잰다(ETH 상수와 같은 정의): 벽 = 전 초 |imb50| 70/80/90분위 ·
    청산 = 전 초 «직전 60초 청산액»(log1p USD) 99.5분위 · 되돌림 = 전 초 |5분 이동| 중앙값(bp). ETH 패널 09-20~27 로 원 상수 재현을 먼저 본다."""
    p = pd.read_parquet(OUT / PANEL)
    f = features(p).loc[_ts(start):_ts(end) - 1]
    ok = (p.dd_valid.reindex(f.index) == 1) & p.mid.reindex(f.index).notna()
    f = f[ok]
    a = np.abs(f.imb50.to_numpy())
    return dict(symbol=SYMBOL, rows=int(len(f)), wall_q={q: round(float(np.quantile(a, q / 100)), 4) for q in (70, 80, 90)},
                liq=(round(float(np.quantile(f.liq_long60, 0.995)), 3), round(float(np.quantile(f.liq_short60, 0.995)), 3)),
                pullback=round(float(np.median(np.abs(f.ret300))), 2))


def teacher_act(A: dict, thr: float):
    """교사 규칙 = 깊은 벽 또는 고래 맞대결, 둘이 반대면 관망(10-02 규칙 평가 +77bp/일 [+16,+142], 9규칙 중 하나).
    봉 경계: 방향(없으면 관망) · 봉 중간: 보유. 피쳐만 보므로 포지션 상태와 무관."""
    cols = features_cols()
    iw, iv = cols.index("imb50"), cols.index("whale_vs_retail")

    def act(o, t, bar):
        if not bar:
            return 0
        x = A["X"][t, iw]
        return int(np.sign((1 if x > thr else -1 if x < -thr else 0) + A["X"][t, iv])) + 1
    return act


def evaluate(seed: int, splits=("val", "test"), idle: float = 0.0) -> dict:
    return evaluate_dir(model_dir(seed, idle), seed, splits)


def evaluate_dir(d: Path, seed: int = 0, splits=("val", "test")) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    model = PPO.load(d / "ppo.zip", device="cpu")
    imb, thr = features_cols().index("imb50"), wall_thr()
    out = {}
    for s in splits:
        A = load_split(s)
        vn = VecNormalize.load(d / "vecnormalize.pkl", DummyVecEnv([lambda: make_env_cls()(A)]))
        vn.training = False
        rng = np.random.default_rng(seed)
        tch, agree = teacher_act(A, thr), []

        def ppo(o, t, bar):
            a = int(model.predict(vn.normalize_obs(o), deterministic=True)[0])
            if bar:
                agree.append(a == tch(o, t, bar))
            return a
        res = {"ppo": run_policy(A, ppo)}
        res["ppo"]["bar_agree_teacher"] = float(np.mean(agree))      # 봉 경계 결정이 교사와 같은 비율
        res.update({"ppo_no_sec_exit": run_policy(A, lambda o, t, bar: ppo(o, t, bar) if bar else 0),   # 매초 청산을 끈 같은 정책
                    "teacher": run_policy(A, tch),
                    "flat": run_policy(A, lambda o, t, bar: 1),
                    "deep_wall_rule": run_policy(A, lambda o, t, bar: (2 if A["X"][t, imb] > thr else 0 if A["X"][t, imb] < -thr
                                                                       else 1) if bar else 0)})
        pl, ps = res["ppo"]["long"], res["ppo"]["short"]  # 같은 노출 비율의 무작위 봉 경계 목표, 봉 끝까지 보유
        res["random_bar"] = run_policy(A, lambda o, t, bar: int(rng.choice(3, p=[ps, 1 - pl - ps, pl])) if bar else 0)
        out[s] = res
    (d / "eval.json").write_text(json.dumps(out, indent=1))
    return out


def train_bcrl(seed: int, ckpts=(0, 100_000, 300_000, 1_000_000), lr: float = 1e-5) -> Path:
    """교사 규칙 행동 모방(BC)으로 정책·가치를 맞춘 뒤 낮은 학습률 PPO 로 미세조정. ckpts 마다 저장 -> «진화가 개선하나» 곡선.
    BC: 교사를 학습구간에 순차로 돌려 (관측, 행동, 이후 할인 손익)을 모은다. 봉 경계 결정은 1% 남짓이라 가중치로 균형."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    torch.manual_seed(seed)
    A = load_split("train")
    Env, tch = make_env_cls(), teacher_act(A, wall_thr())
    env = Env(A, seq=True)
    o, _ = env.reset()
    O, act, bar, rew, done = [], [], [], [], False
    while not done:
        b = is_bar(env.s0 + env.t)
        a = tch(o, env.t, b)
        O.append(o), act.append(a), bar.append(b)
        o, r, done, *_ = env.step(a)
        rew.append(r)
    O, act, bar = np.array(O, np.float32), np.array(act), np.array(bar)
    G, g = np.zeros(len(rew)), 0.0
    for i in range(len(rew) - 1, -1, -1):                       # 이후 할인 손익(γ .999, PPO 와 같은 값) -> 가치망 목표
        g = rew[i] + 0.999 * g
        G[i] = g
    w = np.where(bar, (~bar).sum() / max(bar.sum(), 1), 1.0)
    venv = VecNormalize(DummyVecEnv([lambda i=i: Env(A, seed=seed * 100 + i) for i in range(8)]),
                        norm_obs=True, norm_reward=False, clip_obs=10.0)
    venv.obs_rms.mean, venv.obs_rms.var, venv.obs_rms.count = O.mean(0).astype(np.float64), O.var(0).astype(np.float64), len(O)
    model = PPO("MlpPolicy", venv, learning_rate=lr, n_steps=2048, batch_size=256, gamma=0.999, gae_lambda=0.95,
                clip_range=0.1, ent_coef=0.001, seed=seed, verbose=0, device="cpu")
    pol = model.policy
    On = torch.as_tensor(np.clip((O - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var + 1e-8), -10, 10), dtype=torch.float32)
    At, Wt, Gt = torch.as_tensor(act), torch.as_tensor(w, dtype=torch.float32), torch.as_tensor(G, dtype=torch.float32)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    for ep in range(30):
        for idx in np.array_split(rng.permutation(len(O)), max(len(O) // 4096, 1)):
            dist = pol.get_distribution(On[idx])
            loss = -(Wt[idx] * dist.log_prob(At[idx])).mean() / Wt[idx].mean() \
                + 0.5 * ((pol.predict_values(On[idx]).squeeze(-1) - Gt[idx]) ** 2).mean() / 100.0
            opt.zero_grad(), loss.backward(), opt.step()
    with torch.no_grad():
        pred = pol.get_distribution(On).distribution.probs.argmax(-1).numpy()
    print(f"BC 학습구간 일치: 봉 경계 {np.mean(pred[bar] == act[bar]):.3f} · 봉 중간 {np.mean(pred[~bar] == act[~bar]):.3f}", flush=True)
    root = model_dir(seed).with_name(model_dir(seed).name + "_bcrl")

    def save(n):
        d = root / f"ckpt{n}"
        d.mkdir(parents=True, exist_ok=True)
        model.save(d / "ppo.zip")
        venv.save(d / "vecnormalize.pkl")

    class Ckpt(BaseCallback):
        def _on_step(self):
            for n in ckpts:
                if n and self.num_timesteps - self.model.n_envs < n <= self.num_timesteps:
                    save(n)
            return True
    save(0)
    model.learn(total_timesteps=max(ckpts), callback=Ckpt())
    return root


def features_cols() -> list[str]:
    p = pd.read_parquet(OUT / PANEL).iloc[:WARMUP + 10]
    return list(features(p).columns)


# ── 실시간: 모의 매매하며 계속 학습 (서버) ───────────────────────────────────
LIVE = OUT / "live"
HOT = ROOT / "data" / "hot"
WS = f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@"     # bookTicker·depth@100ms·trade 셋 다 /ws/ (binance_ws_stream_path_table)
WS_MARKET = f"wss://fstream.binance.com/market/ws/{SYMBOL.lower()}@"   # @aggTrade 는 /market/ws/ 에서만 온다(같은 표)
SETTLE_S = 0.35                                 # 초 s 는 s+1+SETTLE_S 에 닫는다(depth 행은 다음 초 첫 메시지에 나온다)


class ReplayFeed:
    """저장 패널을 실시간 피드처럼 -- LiveEnv 를 오프라인에서 돌려 SecEnv 와 대조하고 학습 루프를 시험한다."""

    def __init__(self, p: pd.DataFrame, e: pd.DataFrame):
        self.p, self.X, self.e = p, features(p).to_numpy(), e.reindex(p.index)
        self.s0 = int(p.index[0]) + WARMUP

    def start(self):
        pass

    def wait_ready(self) -> int:
        return self.s0

    def wait_next(self, s: int) -> int:
        if s + 1 > self.p.index[-1]:
            raise StopIteration
        return s + 1

    def obs_row(self, s: int):
        return self.X[s - self.p.index[0]]

    def exec_now(self, s: int) -> dict:
        return dict(self.e.loc[s, ["P", "A", "bq", "aq", "midT"]], T=s)

    def window(self, T0, T1, pc, ac):
        assert T1 == T0 + 1
        return tuple(self.e.loc[T0, ["sv", "smin", "bv", "bmax"]])


class LiveFeed:
    """WS(bookTicker·depth·trade) + hot SQLite(테이프·OI·마크·청산) -> 오프라인과 같은 열의 1초 패널.
    시작할 때 직전 시각 파일 2개로 bookTicker·depth 이력을 채운다(features 의 1시간 창)."""

    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.bt_raw: dict[int, list] = {}
        self.bt: dict[int, dict] = {}
        self.dd: dict[int, dict] = {}
        self.trades: list[tuple] = []                  # (ts, 가격센트, 수량, 테이커매도) -- 최근 몇 초만
        self.last_bt = None
        self.book, self.pending, self.served = None, [], []
        self.tape = pd.DataFrame()
        self.ctx_raw = {}
        self.agg_rows: list[tuple] = []                # @aggTrade 줄 (가격, 수량, ms, 매수자메이커) -- 1분 집계 전
        self.wlive = pd.DataFrame(columns=["w", "r", "w2", "r2"], dtype=float)
        self.wbase = None
        self.cnt = dict(bt=0, dd=0, trade=0, agg=0)   # 스트림별 받은 메시지 수 -- 경로가 틀리면 «연결 정상·이벤트 0»으로 조용히 실패한다

    def start(self):
        import asyncio
        import threading
        ensure_aggtrades()                             # 고래 z 의 30일 이력 = 아카이브(어제까지) + 이 프로세스가 받은 라이브 분
        self.wbase = whale_raw()
        now = pd.Timestamp.now(tz="UTC")
        hours = {(now - pd.Timedelta(hours=k)).strftime("%Y-%m-%dT%H") for k in (0, 1)}
        for stream, fn, dst in (("bookticker", bt_hour, self.bt), ("depthdiff", dd_hour, self.dd)):
            seed = SEED_OF / stream / SYMBOL                   # SOL·XRP = 서버 엔진 전용 수집 폴더(아카이브는 Pi 가 1시간 늦게 복제)
            for f in sorted(p for p in (seed.iterdir() if seed.exists() else []) if p.name[:13] in hours):
                d = fn(f)
                dst.update({int(k): v for k, v in d.to_dict("index").items()} if len(d) else {})
        threading.Thread(target=lambda: asyncio.run(self._ws()), daemon=True).start()

    async def _ws(self):
        import asyncio
        await asyncio.gather(self._run("bookTicker", self._on_bt), self._run("depth@100ms", self._on_dd),
                             self._run("trade", self._on_tr), self._run("aggTrade", self._on_agg, WS_MARKET))

    async def _run(self, stream, cb, base=None):
        import asyncio
        import websockets
        while True:
            try:
                async with websockets.connect((base or WS) + stream, ping_interval=20, ping_timeout=10) as ws:
                    if stream.startswith("depth"):
                        self.book, self.pending = None, []
                        asyncio.get_running_loop().run_in_executor(None, self._snapshot)
                    async for raw in ws:
                        cb(json.loads(raw))
            except Exception as exc:   # 끊김은 다시 붙는다 -- 그 사이 초는 bookTicker 가 비어 ok=False(행동 안 함)
                print(f"[live] ws {stream} {type(exc).__name__}: {exc} -- 3초 뒤 재연결", flush=True)
                await asyncio.sleep(3)

    def _snapshot(self):
        import requests
        import scripts.binance_ban_guard  # noqa: F401  -- 차단 중이면 요청을 안 보낸다
        snap = requests.get(f"https://fapi.binance.com/fapi/v1/depth?symbol={SYMBOL}&limit=1000", timeout=10).json()
        with self.lock:
            book = DepthBook(snap)
            for m in self.pending:
                self._dd_rows(book.feed(m))
            self.book, self.pending = book, []

    def _dd_rows(self, rows):
        for r in rows:
            self.dd[int(r.pop("sec"))] = r

    def _on_dd(self, m):
        self.cnt["dd"] += 1
        with self.lock:
            if self.book is None:
                self.pending.append(m)
                return
            self._dd_rows(self.book.feed(m))
            if not self.book.valid:                    # pu 사슬 끊김 -> 새 스냅샷
                self.book, self.pending = None, []
                import threading
                threading.Thread(target=self._snapshot, daemon=True).start()

    def _on_bt(self, d):
        ts = int(d.get("T") or d.get("E"))             # 수집기와 같은 시각
        row = (ts, float(d["b"]), float(d["B"]), float(d["a"]), float(d["A"]))
        self.cnt["bt"] += 1
        with self.lock:
            self.bt_raw.setdefault(ts // 1000, []).append(row)
            self.last_bt = row

    def _on_tr(self, t):
        if t.get("e") != "trade":
            return
        px, qty = float(t["p"]), float(t["q"])
        if not (px > 0 and qty > 0):                   # 수집기와 같은 거름(가격 0 메시지)
            return
        self.cnt["trade"] += 1
        with self.lock:
            self.trades.append((int(t["T"]), round(px * PX), qty, bool(t["m"])))

    def _on_agg(self, d):
        if d.get("e") == "aggTrade":
            self.cnt["agg"] += 1
            with self.lock:
                self.agg_rows.append((float(d["p"]), float(d["q"]), int(d["T"]), bool(d["m"])))

    def whale_frame(self) -> pd.DataFrame:
        """아카이브 분(0 채움) + 라이브 분(받기 시작한 분부터 0 채움), 그 사이 모르는 분은 NaN -> whale_z.
        아카이브와 겹치는 분은 아카이브가 이긴다(라이브 첫 분은 받다 만 분이다)."""
        with self.lock:
            rows, self.agg_rows = self.agg_rows, []
        if rows:
            a = np.array(rows, dtype=float)
            # ponytail: 오늘 경계는 시작 때 받은 아카이브로 정한다(실행 중엔 새 zip 을 안 받음) -- 직전 30일 중 ≥20일이면 유효, 길게 돌면 재시작
            new = agg_minutes(a[:, 0], a[:, 1], a[:, 2].astype(np.int64), a[:, 3].astype(bool), cut=moving_cuts)
            self.wlive = pd.concat([self.wlive, new]).groupby(level=0).sum(min_count=1)      # 걸친 분은 합친다
        base = self.wbase
        hi = max(base.index.max(), self.wlive.index.max() if len(self.wlive) else 0)
        g = base.reindex(pd.RangeIndex(base.index.min(), hi + 60, 60))
        lv = self.wlive[self.wlive.index > base.index.max()]
        if len(lv):
            lv = lv.reindex(pd.RangeIndex(lv.index.min(), lv.index.max() + 60, 60), fill_value=0.0)
            g.loc[lv.index, list(lv.columns)] = lv.to_numpy()
        return whale_z(g.iloc[-(43200 + 240):])

    @staticmethod
    def latest() -> int:
        import time
        return int(time.time() - SETTLE_S) - 1

    def _close(self, s: int):
        with self.lock:
            for sec in [k for k in self.bt_raw if k <= s]:
                a = np.array(self.bt_raw.pop(sec), dtype=[("ts_ms", "<i8"), ("bid_px", "<f8"), ("bid_qty", "<f4"),
                                                         ("ask_px", "<f8"), ("ask_qty", "<f4")])
                g = bt_agg(a)
                if len(g):
                    self.bt[sec] = g.iloc[0].to_dict()
            b = self.book                               # depth 행은 다음 초 첫 메시지에 나온다 -- 아직이면 지금 상태로
            if b is not None and s not in self.dd and b.cur is not None and b.cur <= s:
                r = b.row()
                r.pop("sec")
                if b.cur < s:                           # 그 초에 메시지 없음 = 상태 그대로, 추가·취소 0 (dd_hour 와 같다)
                    r.update(dd_add_b=0.0, dd_rem_b=0.0, dd_add_a=0.0, dd_rem_a=0.0)
                self.dd[s] = r
            lo = s - WARMUP - 600
            for d in (self.bt, self.dd):
                for k in [k for k in d if k < lo]:
                    del d[k]
            self.trades = [x for x in self.trades if x[0] >= (s - 10) * 1000]

    def _hot(self, s: int):
        lo = s - WARMUP - 60 if self.tape.empty else s - 30     # 늦게 들어온 초까지 다시 읽는다
        rows = ds.read_rows(HOT / "binance_tape.sqlite",
                            "select ts_sec, price_bin, buy_qty, sell_qty, buy_n, sell_n, whale_buy_qty, whale_sell_qty, "
                            "retail_buy_qty, retail_sell_qty from trade_tape_1s where symbol = ? and ts_sec >= ?", (SYMBOL.lower(), lo))
        new = pd.DataFrame(rows, columns=["ts_sec", "price_bin", "buy_qty", "sell_qty", "buy_n", "sell_n", "whale_buy_qty",
                                          "whale_sell_qty", "retail_buy_qty", "retail_sell_qty"])
        self.tape = pd.concat([self.tape[self.tape.ts_sec < lo] if len(self.tape) else self.tape, new])
        self.tape = self.tape[self.tape.ts_sec >= s - WARMUP - 60]
        ms = (s - WARMUP - 60) * 1000
        q = lambda tbl, cols: pd.DataFrame(ds.read_rows(HOT / "binance_ctx.sqlite",
                                                        f"select {cols} from {tbl} where lower(symbol) = ? and ts_ms >= ?",
                                                        (SYMBOL.lower(), ms)), columns=cols.split(", "))
        # ponytail: 맥락 표는 초당 몇 행이라 매초 창 전체를 다시 읽는다(수 ms). 느려지면 테이프처럼 증분으로.
        self.ctx_raw = dict(oi=q("oi_1s", "ts_ms, open_interest"), mark=q("mark_price_1s", "ts_ms, mark, index_px, funding_rate"),
                            liq=q("liquidations", "ts_ms, side, usd"))

    def wait_ready(self) -> int:
        import time
        while True:
            s = self.latest()
            self._close(s)
            if sum(1 for k in self.bt if k > s - WARMUP) > WARMUP * 0.9 and s in self.bt and s in self.dd:
                return s
            print(f"[live] 준비 중: bookTicker {len(self.bt)} 초 · depth {len(self.dd)} 초", flush=True)
            time.sleep(5)

    def wait_next(self, s: int) -> int:
        import time
        while self.latest() <= s:
            time.sleep(0.02)
        s1 = self.latest()
        self._close(s1)
        return s1

    def obs_row(self, s: int):
        self._hot(s)
        lo = s - WARMUP + 1
        with self.lock:
            bt = pd.DataFrame.from_dict({k: v for k, v in self.bt.items() if k >= lo}, orient="index")
            dd = pd.DataFrame.from_dict({k: v for k, v in self.dd.items() if k >= lo}, orient="index")
        p = assemble(lo, s + 1, bt, dd, tape_agg(self.tape), ctx_agg(**self.ctx_raw))
        z, m = self.whale_frame(), (p.index.to_numpy() + 1) // 60 * 60 - 60       # add_whale 과 같은 «마지막 완결 1분»
        p = p.assign(wz=z.zw.reindex(m).to_numpy(), rz=z.zr.reindex(m).to_numpy(),
                     wz2=z.zw2.reindex(m).to_numpy(), rz2=z.zr2.reindex(m).to_numpy())   # 움직이는 경계판(피쳐 아님, paper 가 served 로 읽는다)
        self.served.append(p.iloc[-1].rename(s))         # 실제로 먹인 패널 행 -- 나중에 오프라인 재구성과 대조
        return features(p).to_numpy()[-1]

    def exec_now(self, s: int) -> dict:
        """지금(= 주문을 내는 순간) 최우선 호가. 오프라인은 T=(s+1)초+LAT_MS 의 bookTicker 를 쓴다."""
        import time
        with self.lock:
            b = self.last_bt
        T = int(time.time() * 1000)
        if b is None or T - b[0] > 2000:               # 호가가 2초 넘게 안 왔다 -> 주문 안 냄
            return dict(P=np.nan, A=np.nan, bq=np.nan, aq=np.nan, midT=np.nan, T=T)
        return dict(P=round(b[1] * PX), A=round(b[3] * PX), bq=b[2], aq=b[4], midT=(b[1] + b[3]) / 2, T=T)

    def window(self, T0, T1, pc, ac):
        # ponytail: T1 직전 체결이 아직 안 도착했으면 이 창에서 빠진다(수십 ms·보수적 방향). 필요하면 T1+지연까지 기다렸다 센다.
        with self.lock:
            w = [x for x in self.trades if T0 < x[0] <= T1]
        return window_agg(pd.DataFrame(w, columns=["ts", "c", "qty", "sell"]), pc, ac)


def make_live_env_cls():
    import gymnasium as gym

    class LiveEnv(gym.Env):
        """SecEnv 와 같은 Trader(5분봉 진입 · 보유 중 매초 청산)를 실시간 초로. 결정 없는 초는 안에서 흘려보낸다.
        매초 호가를 찍어 대기열을 진행하고, 결정이 필요한 초에만 관측(55ms)을 만든다. 에피소드는 잘라만 준다(포지션 유지)."""

        def __init__(self, feed, n_feat: int, ledger: Path | None = None, idle_penalty: float = 0.0):
            self.feed, self.ledger, self.idle_penalty = feed, ledger, idle_penalty
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_feat + Trader.N_STATE,), np.float32)
            self.action_space = gym.spaces.Discrete(3)
            self.tr, self.k, self.s = Trader(), 0, None
            self.rewards: list[float] = []

        def _obs(self):
            return np.append(self.x, self.tr.state(self.ex["midT"], self.s)).astype(np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            if self.s is None:                         # 첫 결정은 다음 5분봉 경계
                self.s = self.feed.wait_ready()
                while not is_bar(self.s):
                    self.s = self.feed.wait_next(self.s)
                self.ex = self.feed.exec_now(self.s)
                self.x = self.feed.obs_row(self.s)
            return self._obs(), {}

        def step(self, a):
            tr, quoted = self.tr, lambda e: e["P"] == e["P"] and e["midT"] == e["midT"]
            bar = is_bar(self.s)
            r = tr.decide(int(a), bar, self.ex) if quoted(self.ex) else 0.0     # 호가 없음 -> 주문·청산 안 함
            log = [dict(s=self.s, a=int(a), bar=bar)]
            while True:
                e0, s0 = self.ex, self.s
                self.s = self.feed.wait_next(s0)
                self.ex = self.feed.exec_now(self.s)
                if quoted(e0) and quoted(self.ex):
                    w = self.feed.window(e0["T"], self.ex["T"], e0["P"], e0["A"])
                    r += tr.tick(e0, w, self.ex["midT"])
                log.append(dict(s=self.s, pos=tr.pos, tgt=tr.tgt, P=e0["P"], A=e0["A"],
                                ahead=tr.order[2] if tr.order else None, skip=self.s - s0 - 1))
                if tr.needs_decision(self.s):
                    break
            self.x = self.feed.obs_row(self.s)
            self.rewards.append(r)
            if self.ledger:
                log[-1]["r"] = round(r, 4)
                with open(self.ledger.with_name(f"ledger_{pd.Timestamp(self.s, unit='s'):%Y%m%d}.jsonl"), "a") as f:
                    f.write("".join(json.dumps(x, default=float) + "\n" for x in log))
            self.k += 1
            return self._obs(), float(r - self.idle_penalty * tr.idle), False, self.k % 3600 == 0, {}   # rewards·원장 = 실손익

    return LiveEnv


def live(seed: int, replay: pd.DataFrame | None = None, steps: int = 10**12, out: Path = LIVE, idle: float = 0.0):
    """사전학습 정책을 이어받아 모의 매매하며 PPO 를 계속 갱신한다. 체크포인트가 있으면 거기서 잇는다(재시작에도 진화가 이어진다).
    🔴실주문 코드 없음 -- 주문은 ledger_*.jsonl 에 적히기만 한다."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    out.mkdir(parents=True, exist_ok=True)
    src = out if (out / "ppo.zip").exists() else model_dir(seed, idle)
    feed = ReplayFeed(replay, pd.read_parquet(OUT / "exec.parquet")) if replay is not None else LiveFeed()
    feed.start()
    n_feat = len(features_cols()) if replay is None else feed.X.shape[1]
    env = make_live_env_cls()(feed, n_feat, out / "ledger.jsonl", idle_penalty=idle)
    venv = VecNormalize.load(src / "vecnormalize.pkl", DummyVecEnv([lambda: env]))
    venv.training = True                               # 정규화 통계도 계속 따라간다
    model = PPO.load(src / "ppo.zip", env=venv, device="cpu", custom_objects={"ent_coef": 0.001})

    class Save(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 3600 == 0:
                model.save(out / "ppo.zip")
                venv.save(out / "vecnormalize.pkl")
                if isinstance(feed, LiveFeed) and feed.served:
                    pd.DataFrame(feed.served).to_parquet(out / f"served_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M}.parquet")
                    feed.served = []
                print(f"[live] {self.num_timesteps} 스텝 · 최근 1시간 {sum(env.rewards[-3600:]):+.1f}bp · pos {env.tr.pos}", flush=True)
            return True

    try:
        model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=Save())
    except StopIteration:                              # ReplayFeed 끝
        pass
    return env


# ── 모의 매매: 규칙 9판 병렬 + 매일 «활성 판» 고르기 (= 작은 진화) ──────────────────────
PAPER = OUT / f"paper{TAG}"
WALL_Q = CFG["wall_q"]                          # 학습 7일(ETH 09-20~26) |imb50| 분위 -- 서버가 데이터에 의존하지 않게 고정
TEACHER = "w80_z0.5"                            # 10-02 규칙 평가 +75bp/일 [+13,+145] (9규칙 중 하나)
# 10-02 원장 진단(표본 안 발견 -> 실시간 판으로만 전진 검증): 추격 진입 −1.1 vs 역행 뒤 +8.3bp/건 · 진 거래 22%가 한때 +10bp ·
#   청산 대기 9~104초 청산 체결 −2.75. 교사에 하나씩 얹은 판 3개.
VARIANTS = {"nochase": "진입 직전 5분이 이미 신호 방향이면 새로 들어가지 않음", "tp10": "미실현 +10bp 에 지정가 익절, 다음 봉까지 쉼",
            "xc60": "봉 경계 뒤 60초 안에 청산·전환이 안 차면 쫓지 않고 다음 봉에"}
ARMS = [f"w{q}_" + (f"z{z:g}" if z else "off") for q in WALL_Q for z in (None, 0.5, 1.0)] + [f"{TEACHER}_{v}" for v in VARIANTS] \
    + [f"{TEACHER}_nochase_{v}" for v in ("w60", "liq", "holdloss")] + [f"{TEACHER}_pullback"] \
    + [f"{TEACHER}_mv"] \
    + [f"{TEACHER}_a24", f"{TEACHER}_a24_mh15", f"{TEACHER}_sticky60", "w100_z0.5_w60"] \
    + ["trend4", "trend4_vs", "poi1h", f"{TEACHER}_a24_mh15_vt", "w100_z0.5_w60_vt", f"{TEACHER}_a24_mh15_er", "w100_z0.5_w60_er"]
# 10-03 추가 4판 + 움직이는 경계판 + 회전·고래 단독 4판 · 10-04 «통과했는데 모의 판에 없던 것» 7판(사용자 지시):
#   trend4    = 1~4주 추세(7·14·21·28일 종가 부호 평균, 매일 00시 UTC 결정) -- 현물 9년 샤프 0.99·선물 4.7년 0.62(eth_tsmom_1_4w_voltarget)
#   trend4_vs = 같은 방향 × 변동성 사이징 크기(연 50% ÷ 20일 실현변동성, ±2배) -- 대시보드 추세 칸과 같은 식, 손익 배수로 반영
#   poi1h     = 5분마다 직전 1시간 수익 ≤ 하위 25% 이면 ΔOI 1시간 < 0 → 롱 · > 0 → 숏, 마지막 발동 뒤 1시간 보유(price_oi_quadrant 1h +3.5bp 차 원 연구 정의).
#               🔴2026-10-05 정시 결정에서 5분 판정으로(사용자 «원 연구 정의로 진행 · 마이너스 포함돼도 문제없어») -- 2025~ 는 원 정의도 CI 0 포함
#               (OI↓−OI↑ +3.37[−0.32,+7.53] · 이 판정의 매매 −8.25bp/일[−33,+17], 2022~24 +13.0) · docs/experiments/trend4_poi1h_solxrp_20261004.md §3
#   _vt       = 24시간 SMA(5분봉 288개) 반대쪽 목표는 0(역추세 금지, czz SMA288 veto +12.1bp)
#   _er       = 같은 거래를 동일위험 크기로: 직전 4시간 5분 수익 표준편차 대비 4.7년 중앙값(배 0.5~2). 🔴배포 위험모델(MAE 분위)이 아니라
#               실현변동성 대용 -- 그 모델은 사이징 워커와 함께 09-27 에 꺼져 있다(dashboard_vol_model_removed).
TREND_L, TREND_VOL_N, TREND_TARGET, TREND_MAX = (7, 14, 21, 28), 20, 0.50, 2.0
POI_DOWN_BP = CFG["poi"]                        # ETH -26.68 = 4.7년(2022-01~2026-09) 5분 간격 1시간 수익 25분위 -- 원 연구 «Δ가격 하위 25%»
ER_REF_BP, ER_CLIP = CFG["er"], (0.5, 2.0)          # 4.7년 «직전 48개 5분 수익 표준편차» 중앙값(bp)
TREND = {"day": None, "sig": 0.0, "size": 0.0}  # 엔진이 매일 일봉으로 채운다(refresh_trend). 비면 추세 판은 관망
SWITCH_MIN_DAYS, SWITCH_T = 7, 2.5              # 교사에서 갈아타려면: 최근 ≤14일 «판 − 교사» 일 손익 차가 n≥7·t>2.5 (16판 -- 우연 1등 막으려 2.0 에서 올림)
# 10-02 저장 10일 시험: «3일 뒤부터 최근 평균 1등»은 +34bp/일 < 고정 교사 +75 (09-23 에 w70_off 로 갈아타 −176) = 잡음 추종.


PORTFOLIOS = {"p1": {f"{TEACHER}_a24_mh15": 0.5, "w100_z0.5_w60": 0.5}}   # 10-03 반반: 벽(적응+최소보유)·고래 단독은 일 손익 상관 −0.04


def dd_init(daily: list[float]) -> list[float]:
    """일별 손익 -> [고점, 최대 낙폭] (고점은 0 부터 -- 첫날부터 잃으면 그게 낙폭)."""
    peak, mdd, c = 0.0, 0.0, 0.0
    for v in daily:
        c += v
        peak = max(peak, c)
        mdd = min(mdd, c - peak)
    return [peak, mdd]


def dd_step(pm: list[float], cum: float) -> list[float]:
    peak = max(pm[0], cum)
    return [peak, min(pm[1], cum - peak)]


def trend4_signal(closes: list[float]) -> tuple[float, float]:
    """완결 일봉 종가(오래된 것부터) -> (신호 = 4기간 부호 평균 ∈ [−1,1], 크기 = 신호 × 50% ÷ 20일 연율 변동성, ±2배)."""
    c = np.asarray(closes, float)
    if len(c) < max(TREND_L) + 1 or len(c) < TREND_VOL_N + 1:
        return 0.0, 0.0
    sig = float(np.mean([np.sign(c[-1] - c[-1 - L]) for L in TREND_L]))
    vol = float(np.std(np.diff(np.log(c[-TREND_VOL_N - 1:])), ddof=1) * np.sqrt(365))
    return sig, float(np.clip(sig * TREND_TARGET / vol, -TREND_MAX, TREND_MAX)) if vol > 0 else 0.0


def fetch_closes(interval: str, limit: int, now_ms: int) -> list[float]:
    """바이낸스 선물 klines 완결 봉 종가(오래된 것부터). 서버(엔진·대시보드와 같은 IP)에서 하루 한두 번만 부른다."""
    import requests
    k = requests.get("https://fapi.binance.com/fapi/v1/klines",
                     params=dict(symbol=SYMBOL, interval=interval, limit=limit), timeout=10).json()
    return [float(r[4]) for r in k if int(r[6]) < now_ms]


def refresh_trend(day: int) -> bool:
    """TREND 를 «day 시작 직전까지 완결된 일봉»으로 채운다. 새 일봉이 아직 안 붙었으면(경계 ~5초 지연) False."""
    try:
        c = fetch_closes("1d", 40, day * 86_400_000)
    except Exception as exc:  # noqa: BLE001
        print(f"[paper] 일봉 가져오기 실패(추세 판은 관망): {type(exc).__name__}: {exc}", flush=True)
        return False
    sig, size = trend4_signal(c)
    TREND.update(day=day, sig=sig, size=size)
    print(f"[paper] 추세 판 갱신 {pd.Timestamp(day * 86400, unit='s').date()} 신호 {sig:+.2f} 크기 {size:+.2f}", flush=True)
    return True


def arm_params(arm: str) -> tuple[float, float | None]:
    w, z = arm.split("_")[:2]
    thr = np.inf if w == "w100" else WALL_Q[int(w[1:])]          # w100 = 벽 끔(고래 단독). WALL_Q 에 넣으면 ARMS 가 3판을 자동으로 만든다
    return thr, (None if z == "off" else float(z[1:]))


PULLBACK_BP = CFG["pullback"]                   # ETH 7.22 · pullback 판: 직전 5분이 신호 반대로 «학습 7일 |5분 이동| 중앙값» 이상 밀렸을 때만 새 진입
LIQ_BURST = dict(zip(("liq_long60", "liq_short60"), CFG["liq"]))   # 학습 7일 «직전 60초 청산액» q99.5 (log1p USD ≈ $34만/$55만)
# 09-25 연구(liq_hunt_entry_with_liq_burst_exit): 청산 동반 급등은 안 되돌고 같은 크기 비청산 급등은 되돈다 -> «청산 실린 급등 역매매 금지».
#   롱 청산 버스트 = 청산 실린 급락 -> 새 매수 금지, 숏 청산 버스트 = 청산 실린 급등 -> 새 매도 금지.


class ArmPolicy:
    """판 하나의 봉 경계 결정 + 그 판만의 상태. 실시간 엔진(paper)·백테스트(paper_backtest·4년) 공용.
    판 이름 = w{문턱}_z{고래}[_플래그...]: nochase(추격 거르기) · pullback(직전 5분이 신호 반대일 때만 새 진입) ·
    w60(고래는 정시에만 정하고 60분 유지 -- 4년 근거) · liq(청산 버스트 반대 진입 금지) · holdloss(신호가 꺼졌는데 손실 중이면 한 봉 더) ·
    mv(고래·리테일 z 를 움직이는 경계판 whale_z_mv·retail_z_mv 로) · a24(문턱 = 직전 24시간 봉마다 |imb50| 80분위, 12시간 미만이면 고정) ·
    mh15(새 진입 결정 뒤 15분은 청산·전환 무시) · stickyN(신호가 꺼져도 마지막 확인 뒤 N분까지 보유, 반대 신호면 전환) ·
    tpN(w60 고래판과 함께면 익절 뒤 다음 정시까지 재진입 금지 -- 익절 자체는 sec_action).
    10-03 tmp/turnover_test.py: 고정 문턱이 불균형 큰 날 더 자주 켜져 체결 53 -> 83건/일 -- 적응 문턱·회전 줄이기로 비용을 줄인다."""

    def __init__(self, arm: str):
        self.arm, self.f = arm, set(arm.split("_")[2:])
        self.whale_hold, self.held = 0, False
        self.hist, self.entry_s, self.last_ok = deque(maxlen=288), None, -10**9   # ponytail: 재시작 뒤 12시간은 a24 가 고정 문턱
        self.sticky = next((int(x[6:]) for x in self.f if x.startswith("sticky") and x[6:].isdigit()), 0)
        self.tp = any(x[:2] == "tp" and x[2:].isdigit() for x in self.f)
        self.tp_block = -1
        self.k = 1.0                                               # 크기 배수(엔진이 Trader.k 로 옮긴다)
        self.logp = deque(maxlen=288)                              # 5분봉 로그가격×1e4 (vt: 24h SMA) -- seed_logp 로 미리 채운다
        self.r5, self.o5 = deque(maxlen=48), deque(maxlen=12)       # 봉마다 ret300(er 4h 변동성 · poi 1h 수익) · doi300(poi 1h ΔOI)
        self.hour_t, self.poi_left = 0, 0                          # poi1h: 목표 · 마지막 발동 뒤 남은 봉(12 = 1시간)

    def seed_logp(self, closes: list[float]) -> None:
        self.logp.extend(np.log(np.asarray(closes, float)) * 1e4)

    def _special(self, x: np.ndarray, idx: dict, s: int, pos: int) -> int:
        """벽·고래 계열이 아닌 판(trend4·poi1h)."""
        if self.arm.startswith("trend4"):
            if TREND["day"] is None:
                return 0
            t = int(np.sign(TREND["sig"]))
            if t and t != pos:                                     # 크기는 새 진입 때만 바꾼다(보유 중 배수 고정)
                self.k = abs(TREND["size"]) if self.arm == "trend4_vs" else 1.0
            return t
        r1h, o1h = (sum(list(self.r5)[-12:]), sum(self.o5)) if len(self.r5) >= 12 else (np.inf, 0.0)
        if r1h <= POI_DOWN_BP and np.isfinite(o1h) and o1h != 0:   # poi1h: 5분마다 판정(원 연구 정의) -- 발동하면 그 방향, 1시간 시계를 다시 건다
            #   🔴OI 결측(NaN)은 발동 아님 -- «NaN < 0» 이 거짓이라 숏으로 읽혔다(대조 시험이 잡음, 실시간은 피쳐가 0 으로 채워 안 일어남)
            self.hour_t, self.poi_left = (1 if o1h < 0 else -1), 12
        elif self.poi_left > 0:
            self.poi_left -= 1
        else:
            self.hour_t = 0
        return self.hour_t

    def bar(self, x: np.ndarray, idx: dict, s: int, pos: int, unr: float) -> int:
        r5 = float(x[idx["ret300"]])
        self.r5.append(r5); self.o5.append(float(x[idx["doi300"]]) if "doi300" in idx else 0.0)
        self.logp.append((self.logp[-1] if self.logp else 0.0) + r5)
        if self.arm.startswith(("trend4", "poi1h")):
            return self._special(x, idx, s, pos)
        t = self._bar(x, idx, s, pos, unr)
        if "vt" in self.f and len(self.logp) >= 288:              # 역추세 금지: 24h SMA 반대쪽 목표는 0
            if t * (self.logp[-1] - np.mean(self.logp)) < 0:
                t = 0
        if "er" in self.f and t and t != pos and len(self.r5) >= 24:
            self.k = float(np.clip(ER_REF_BP / max(np.std(self.r5, ddof=1), 1e-9), *ER_CLIP))
        return t

    def _bar(self, x: np.ndarray, idx: dict, s: int, pos: int, unr: float) -> int:
        thr, zt = arm_params(self.arm)
        v = x[idx["imb50"]]
        if "a24" in self.f:
            self.hist.append(abs(v))
            if len(self.hist) >= 144:
                thr = float(np.quantile(self.hist, 0.8))
        wd = 1 if v > thr else -1 if v < -thr else 0
        wz, rz = x[idx["whale_z"]], x[idx["retail_z"]]
        if "mv" in self.f:
            wz, rz = x[idx["whale_z_mv"]], x[idx["retail_z_mv"]]
        hd = int(np.sign(wz)) if zt is not None and wz * rz < 0 and abs(wz) >= zt and abs(rz) >= zt else 0
        if "w60" in self.f:
            if (s + 1) % 3600 == 0:
                self.whale_hold = hd
            hd = self.whale_hold
        t = int(np.sign(wd + hd))
        if t != 0 and pos != t:                                    # 새 진입·전환에만 거르기
            r5 = t * x[idx["ret300"]]
            liq = x[idx["liq_long60"]] > LIQ_BURST["liq_long60"] if t > 0 else x[idx["liq_short60"]] > LIQ_BURST["liq_short60"]
            if ("nochase" in self.f and r5 > 0) or ("pullback" in self.f and r5 > -PULLBACK_BP) or ("liq" in self.f and liq):
                t = 0
        if pos and t == pos:
            self.last_ok = s
        if "mh15" in self.f and pos and t != pos and self.entry_s is not None and s - self.entry_s < 900:
            t = pos                                                # 최소 보유(진입 결정 시각부터 -- peg 체결 중앙 3초)
        if self.sticky and pos and t == 0 and s - self.last_ok < self.sticky * 60:
            t = pos
        if not pos and s < self.tp_block:                          # 고래 익절 뒤 다음 정시까지 재진입 금지(on_tp)
            t = 0
        if t and t != pos:
            self.entry_s = self.last_ok = s
        if "holdloss" in self.f and pos:
            if t == 0 and unr < 0 and not self.held:
                t, self.held = pos, True
            elif t == pos:
                self.held = False
        if pos == 0:
            self.held = False
        return t

    def on_tp(self, s: int) -> None:
        """엔진이 sec_action 익절을 실행했을 때 부른다. w60 고래판은 다음 정시 결정까지 다시 들어가지 않는다
        (진입·익절이 한 봉 안에서 끝나면 정책은 포지션을 못 보므로 포지션 변화로는 감지할 수 없다)."""
        if self.tp and "w60" in self.f:
            self.tp_block = ((s + 1) // 3600 + 1) * 3600 - 1


def sec_action(arm: str, pos: int, tgt: int, unr_bp: float, sec_in_bar: int) -> str | None:
    """봉 중간 매초. tp10: 보유 중 미실현 ≥ +10bp -> 지정가 익절('exit'). xc60: 봉 경계 뒤 60초 지나도 청산·전환이
    안 찼으면 주문을 거두고 다음 봉에('cancel')."""
    tp = next((int(x[2:]) for x in arm.split("_")[2:] if x[:2] == "tp" and x[2:].isdigit()), 0)   # tp10·tp20 … 미실현 ≥ N bp 지정가 익절
    if tp and pos and tgt == pos and unr_bp >= tp:
        return "exit"
    if arm.endswith("_xc60") and pos and tgt != pos and sec_in_bar >= 60:
        return "cancel"
    return None


def rule_target(x: np.ndarray, idx: dict, arm: str) -> int:
    """깊은 벽(|imb50|>문턱) + 고래 맞대결(z 문턱, 끄면 0) -> 합의/한쪽만이면 그 방향, 반대면 0. 반환 −1/0/+1."""
    thr, zt = arm_params(arm)
    v = x[idx["imb50"]]
    wd = 1 if v > thr else -1 if v < -thr else 0
    wz, rz = x[idx["whale_z"]], x[idx["retail_z"]]
    hd = int(np.sign(wz)) if zt is not None and wz * rz < 0 and abs(wz) >= zt and abs(rz) >= zt else 0
    return int(np.sign(wd + hd))


def pick_arm(hist: dict[str, list[float]]) -> str:
    """지난 날들의 일 손익만으로(그날 데이터 X) 활성 판을 고른다. 기본은 교사, 교사를 같은 날 짝 비교로
    분명히(n≥7·t>2) 이긴 판이 있을 때만 그중 평균 차가 가장 큰 판으로 갈아탄다."""
    base = np.array(hist.get(TEACHER, [])[-14:])
    best, best_d = TEACHER, 0.0
    for a, v in hist.items():
        v = np.array(v[-14:])
        n = min(len(v), len(base))                 # 중간에 추가된 판은 기록이 짧다 -- 같은 최근 n일끼리(10-03 전엔 길이가 달라 broadcast 오류)
        d = v[len(v) - n:] - base[len(base) - n:]
        if a == TEACHER or len(d) < SWITCH_MIN_DAYS or d.std(ddof=1) == 0:
            continue
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        if t > SWITCH_T and d.mean() > best_d:
            best, best_d = a, d.mean()
    return best


def paper_backtest(days=None) -> dict:
    """저장 10일에 9판을 날짜별로 돌리고 같은 선택 규칙을 인과적으로 적용 -> 고정 교사 대비 선택의 값."""
    p, e = pd.read_parquet(OUT / PANEL), pd.read_parquet(OUT / f"exec{TAG}.parquet")
    cols = features_cols()
    idx = {c: cols.index(c) for c in ("imb50", "whale_z", "retail_z", "ret300", "doi300", "liq_long60", "liq_short60")}
    days = days or [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-09-20", "2026-09-29")]
    table = {a: [] for a in ARMS}
    for day in days:
        s0 = _ts(day)
        A = arrays(p.loc[s0 - WARMUP:s0 + 86400 - 1], e)
        for a in ARMS:
            if a.endswith(("_xc60", "_mv")) or a.startswith("trend4"):   # 주문 거두기·움직이는 경계 z·일봉 추세는 저장 패널에 없다 -> 실시간 판에서만
                table[a].append(np.nan)
                continue

            pol = ArmPolicy(a)

            def act(o, t, bar, a=a, pol=pol):           # o[-7] = 포지션, o[-6] = 미실현/10, o[-2] = 청산 대기 (Trader.state)
                if bar:
                    return pol.bar(A["X"][t], idx, A["s0"] + t, int(o[-7]), o[-6] * 10) + 1
                return 1 if sec_action(a, int(o[-7]), int(o[-7]) if o[-2] == 0 else 0, o[-6] * 10, 0) == "exit" else 0
            table[a].append(run_policy(A, act)["pnl_bp_day"])
    active = [pick_arm({a: v[:i] for a, v in table.items()}) for i in range(len(days))]
    track = [table[a][i] for i, a in enumerate(active)]
    return dict(days=days, table=table, active=active, selected=track, teacher=table[TEACHER])


def paper(out: Path = PAPER, feed=None) -> None:
    """서버 상주: LiveFeed 하나로 9판을 동시에 모의 매매(체결은 큐 모델, 실주문 없음). 매 UTC 날짜 바뀔 때 pick_arm.
    산출: state.json(1분마다) · daily.jsonl(판별 일 손익·활성 판·기록 시간) · served_*.parquet(실제 먹인 패널 행, 매시).
    재시작 이어받기(10-05 검증 결함): state.json 의 resume(1분마다 + SIGTERM 때)에서 같은 UTC 날이면 오늘 손익을 잇고, 날이 바뀌었으면
    끊긴 날을 저장값으로 daily 에 쓴다(restart_cut). 열린 포지션은 이어받고 대기 주문은 버린다 -- 체결 모델이 큐 자리를 잃으므로
    새로 서는 것과 같고, 재시작 시각에 가짜 청산을 원장에 쓰면 판이 하지 않은 거래가 된다. 장중 고점·낙폭도 잇는다.
    feed = selftest 의 가짜 피드(기본 LiveFeed)."""
    import signal
    import time
    out.mkdir(parents=True, exist_ok=True)
    cols = features_cols()
    idx = {c: cols.index(c) for c in ("imb50", "whale_z", "retail_z", "ret300", "doi300", "liq_long60", "liq_short60")}
    idx.update(whale_z_mv=len(cols), retail_z_mv=len(cols) + 1)    # 피쳐 뒤에 붙인다(obs_row 가 served 에 남긴 wz2·rz2)
    feed = feed or LiveFeed()
    feed.start()
    s = feed.wait_ready()
    while not is_bar(s):
        s = feed.wait_next(s)
    ex = feed.exec_now(s)
    st0, rs = {}, {}
    try:
        st0 = json.loads((out / "state.json").read_text())
        rs = st0.get("resume") or {}               # 10-05 전 state 에는 없다 -> 새로 시작
    except Exception:  # noqa: BLE001  -- 없음·깨짐 = 새로 시작
        pass
    if rs and rs["day"] < s // 86400:              # 날이 바뀐 뒤 재시작 -> 끊긴 날을 저장값으로(그날 줄이 이미 있으면 안 쓴다)
        dstr = str(pd.Timestamp(rs["day"] * 86400, unit="s").date())
        lines = (out / "daily.jsonl").read_text().splitlines() if (out / "daily.jsonl").exists() else []
        if not lines or json.loads(lines[-1])["day"] != dstr:
            with open(out / "daily.jsonl", "a") as f:
                f.write(json.dumps(dict(day=dstr, active=st0.get("active"), pnl={a: v["today"] for a, v in rs["arms"].items()},
                                        fills={a: v.get("fills") for a, v in (st0.get("arms") or {}).items()},
                                        hours=round(rs["rec_s"] / 3600, 2), restart_cut=True)) + "\n")
    hist = {a: [] for a in ARMS}
    if (out / "daily.jsonl").exists():
        for line in (out / "daily.jsonl").read_text().splitlines():
            for a, v in json.loads(line)["pnl"].items():
                hist.setdefault(a, []).append(v)
    active = pick_arm(hist)
    traders = {a: Trader() for a in ARMS}
    policies = {a: ArmPolicy(a) for a in ARMS}
    try:                                           # vt 판의 24h SMA 를 재시작 직후부터 쓰게 5분봉 288개로 미리 채운다(실패하면 24h 예열)
        c5 = fetch_closes("5m", 300, int(time.time() * 1000))[-288:]
        for a in ARMS:
            if a.endswith("_vt"):
                policies[a].seed_logp(c5)
    except Exception as exc:  # noqa: BLE001
        print(f"[paper] 5분봉 미리 채우기 실패(vt 판은 24h 예열): {type(exc).__name__}: {exc}", flush=True)
    # ponytail: 정지가 길어도(수 시간+) 그대로 잇는다 -- 정지 동안 움직임은 재시작 첫 호가에 한 번에 시가평가된다. 피쳐 이력(a24·er·vt 창)은 다시 예열.
    for a, v in (rs.get("arms") or {}).items():
        if a in traders:
            for k, x in (v.get("pol") or {}).items():   # 보유 시계(고래 60분·poi 1시간·최소 보유 등)
                setattr(policies[a], k, x)
            traders[a].k = policies[a].k = v["k"]
            if v["pos"]:
                tr = traders[a]
                tr.pos = tr.tgt = v["pos"]
                tr.entry, tr.t_entry, tr.k_in = v["entry"], v["t_entry"], v["k_in"]
    phist = {k: [sum(wt * hist[a][len(hist[a]) - n + i] for a, wt in ws.items()) for i in range(n)]
             for k, ws in PORTFOLIOS.items() for n in [min(len(hist.get(a, [])) for a in ws)]}
    dd = {**{a: dd_init(hist[a]) for a in ARMS}, **{k: dd_init(v) for k, v in phist.items()}}
    dd.update({k: v for k, v in (rs.get("dd") or {}).items() if k in dd})   # 장중 고점·낙폭(일 종가로 다시 세면 장중 낙폭이 사라진다)
    sig = {}                                       # 대시보드 신호(마지막 봉 마감 값) -- 연구와 같은 계산을 엔진이 한 번만 한다
    quoted = lambda e: e["P"] == e["P"] and e["midT"] == e["midT"]
    same = rs.get("day") == s // 86400
    day, day0 = s // 86400, {a: -rs["arms"][a]["today"] if same and a in rs["arms"] else 0.0 for a in ARMS}
    rec_s, gap, mk, stop = (rs["rec_s"] if same else 0), 0, [rs.get("mid")], []   # 오늘 기록 초 · 시가평가만 한 초 · 마지막 시가평가 mid

    def remark(e):
        """tick 사이 빈 구간을 시가평가로 잇는다(체결 판정은 안 함 = 보수적). 전엔 ① 봉 경계에서 관측 계산 뒤 호가를 다시 받아 다음 tick 이
        거기서 시작해 계산 동안의 움직임이 빠졌고 ② 호가가 2초 넘게 묵은 초(exec_now NaN)는 tick 을 통째로 건너뛰었다.
        10-05 검증: 10-04 10:00 정시 봉에서 원장 대비 +1.9bp 어긋남(그 시각 WS 재연결 없음 -> ① 로 추정)."""
        if quoted(e):
            if mk[0] is not None:
                for tr in traders.values():
                    tr.mark(mk[0], e["midT"])
            mk[0] = e["midT"]
    last_state = last_served = time.time()
    last_beat = last_state - 540                   # 첫 하트비트는 1분 뒤(스트림 확인)
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))   # 재시작(handoff stop·예약 스크립트) -> 상태를 쓰고 끝낸다
    print(f"[paper] 시작 {pd.Timestamp(s, unit='s')} · 활성 {active} · 이어받기 "
          f"{'같은 날' if same else '날 바뀜' if rs else '없음'} 포지션 {sum(1 for t in traders.values() if t.pos)}판", flush=True)
    while True:
        if is_bar(s):
            if TREND["day"] != (s + 1) // 86400:                   # 하루 한 번(실패하면 다음 봉에 다시)
                refresh_trend((s + 1) // 86400)
            x = feed.obs_row(s)
            x = np.append(x, np.nan_to_num(feed.served[-1][["wz2", "rz2"]].to_numpy(float)))   # features 와 같은 규약: NaN -> 0
            ex = feed.exec_now(s)                      # 관측 계산 뒤 = 주문이 나가는 순간의 호가
            remark(ex)
            if quoted(ex):
                tg = {}
                for a, tr in traders.items():
                    tr.now = s + 1
                    unr = tr.pos * (ex["midT"] / tr.entry - 1) * 1e4 if tr.pos else 0.0
                    tg[a] = policies[a].bar(x, idx, s, tr.pos, unr)
                    tr.k = policies[a].k
                    tr.decide(tg[a] + 1, True, ex)
                pa = policies.get(f"{TEACHER}_a24_mh15")
                sig = dict(bar_s=s, zw=float(x[idx["whale_z"]]), zr=float(x[idx["retail_z"]]), imb50=float(x[idx["imb50"]]),
                           r5=float(x[idx["ret300"]]), whale_hold=int(policies["w100_z0.5_w60"].whale_hold) if "w100_z0.5_w60" in policies else 0,
                           thr=float(np.quantile(pa.hist, 0.8)) if pa is not None and len(pa.hist) >= 144 else WALL_Q[80])
                with open(out / "decisions.jsonl", "a") as f:   # 봉마다 판단 근거 + 판별 목표(원장 진단용)
                    f.write(json.dumps(dict(s=s, x={c: round(float(x[i]), 4) for c, i in idx.items()}, tgt=tg)) + "\n")
        elif quoted(ex):                               # 봉 중간: 익절·청산 쫓기 제한 판만
            remark(ex)
            for a, tr in traders.items():
                unr = tr.pos * (ex["midT"] / tr.entry - 1) * 1e4 if tr.pos else 0.0
                act = sec_action(a, tr.pos, tr.tgt, unr, (s + 1) % BAR)
                if act == "exit":
                    tr.now = s + 1
                    tr.decide(1, False, ex)
                    policies[a].on_tp(s)
                elif act == "cancel":
                    tr.tgt = tr.pos
        s1 = feed.wait_next(s)
        ex1 = feed.exec_now(s1)
        if quoted(ex) and quoted(ex1):
            w = feed.window(ex["T"], ex1["T"], ex["P"], ex["A"])
            for tr in traders.values():
                tr.now = s1 + 1
                tr.tick(ex, w, ex1["midT"])
            mk[0] = ex1["midT"]
        else:
            gap += s1 - s                              # 체결 판정 없이 지나간 초(시가평가는 회복 첫 초에 remark)
        rec_s += s1 - s
        with open(out / "trades.jsonl", "a") as f:     # 끝난 거래 원장(판별)
            for a, tr in traders.items():
                for c in tr.closed:
                    f.write(json.dumps(dict(arm=a, **c), default=float) + "\n")
                tr.closed = []
        s, ex = s1, ex1
        tot = {a: sum(tr.pnl.values()) for a, tr in traders.items()}
        if s // 86400 != day:                          # UTC 날짜 바뀜 -> 판별 일 손익 기록 + 다음 날 활성 판
            rec = {a: tot[a] - day0[a] for a in ARMS}
            with open(out / "daily.jsonl", "a") as f:
                f.write(json.dumps(dict(day=str(pd.Timestamp(day * 86400, unit="s").date()), active=active, pnl=rec,
                                        fills={a: traders[a].n["maker_fills"] for a in ARMS}, hours=round(rec_s / 3600, 2))) + "\n")
            for a in ARMS:
                hist[a].append(rec[a])
            for k, ws in PORTFOLIOS.items():
                phist[k].append(sum(wt * rec[a] for a, wt in ws.items()))
            active, day, day0, rec_s = pick_arm(hist), s // 86400, dict(tot), 0
            print(f"[paper] {rec[TEACHER]:+.1f}bp(교사) · 다음 활성 {active}", flush=True)
        if time.time() - last_state >= 60 or stop:
            st = dict(ts=s, active=active, gap_s=gap, arms={a: dict(pos=tr.pos, tgt=tr.tgt, pnl_today=round(tot[a] - day0[a], 2),
                                                        pnl_total=round(tot[a], 2), fills=tr.n["maker_fills"])
                                                 for a, tr in traders.items()})
            try:                                       # 화면용 덧붙임 -- 실패해도 엔진·기본 상태는 계속(로그만)
                mid = ex["midT"] if quoted(ex) else None
                for a, tr in traders.items():
                    cum = sum(hist[a]) + tot[a] - day0[a]
                    dd[a] = dd_step(dd[a], cum)
                    st["arms"][a].update(entry=round(tr.entry, 2) if tr.pos else None, cum=round(cum, 1), mdd=round(dd[a][1], 1),
                                         days=len(hist[a]) + 1, k=tr.k_in if tr.pos else None,   # unr = 배수 반영(원장 bp_k 와 같은 단위)
                                         unr=round(tr.pos * (mid / tr.entry - 1) * 1e4 * tr.k_in, 2) if tr.pos and mid else None)
                st["port"] = {}
                for k, ws in PORTFOLIOS.items():
                    cum = sum(wt * st["arms"][a]["cum"] for a, wt in ws.items())
                    dd[k] = dd_step(dd[k], cum)
                    st["port"][k] = dict(weights=ws, cum=round(cum, 1), mdd=round(dd[k][1], 1), days=len(phist[k]) + 1,
                                         pnl_today=round(sum(wt * st["arms"][a]["pnl_today"] for a, wt in ws.items()), 2),
                                         pos=sum(wt * traders[a].pos for a, wt in ws.items()), tgt=sum(wt * traders[a].tgt for a, wt in ws.items()))
                st["signals"] = sig
            except Exception as exc:  # noqa: BLE001
                print(f"[paper] 화면용 상태 계산 실패(기본 상태는 씀): {type(exc).__name__}: {exc}", flush=True)
            st["resume"] = dict(day=day, rec_s=rec_s, mid=mk[0], dd=dd, arms={      # 재시작 이어받기(paper 시작부)
                a: dict(today=tot[a] - day0[a], pos=tr.pos, entry=tr.entry, t_entry=tr.t_entry, k=policies[a].k, k_in=tr.k_in,
                        pol={k: getattr(policies[a], k) for k in ("whale_hold", "hour_t", "poi_left", "entry_s", "last_ok", "held", "tp_block")})
                for a, tr in traders.items()})
            (out / "state.tmp").write_text(json.dumps(st))
            (out / "state.tmp").replace(out / "state.json")
            last_state = time.time()
            if stop:
                print(f"[paper] 종료 신호 -- 상태 저장 {pd.Timestamp(s, unit='s')}", flush=True)
                return
        if time.time() - last_beat >= 600:             # 10분마다 스트림별 메시지 수 -- 0 이면 그 스트림이 죽은 것
            print(f"[paper] {pd.Timestamp(s, unit='s')} 메시지 {feed.cnt} · 교사 오늘 {tot[TEACHER] - day0[TEACHER]:+.1f}bp "
                  f"pos {traders[TEACHER].pos}", flush=True)
            feed.cnt = dict.fromkeys(feed.cnt, 0)
            last_beat = time.time()
        if time.time() - last_served >= 3600 and feed.served:
            pd.DataFrame(feed.served).to_parquet(out / f"served_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M}.parquet")
            feed.served, last_served = [], time.time()


def calib(n: int = 5000, timeout: int = 120, seed: int = 0) -> dict:
    """자 검증: 이 큐 모델로 «peg 다리 하나»(최우선에 걸고 바뀌면 따라가며 체결까지)를 모사해
    이 저장소 섀도우 실측(ETHUSDT peg 7,982다리, 09-13: 중앙 체결 15.2→1.8초 · 수수료 뺀 비용 ~1bp · 저변동 미체결 5%)과 견준다."""
    e = pd.read_parquet(OUT / "exec.parquet")
    e = e.reindex(pd.RangeIndex(e.index.min(), e.index.max() + 1))
    E = {c: e[c].to_numpy(float) for c in EXEC_COLS}
    lr = np.log(e.midT).diff() * 1e4
    vol = (lr.rolling(1800, min_periods=900).std() * np.sqrt(60)).to_numpy()   # bp/√분, 직전 30분
    rng = np.random.default_rng(seed)
    rows = []
    for i in rng.choice(np.flatnonzero(np.isfinite(E["P"][:-timeout]) & np.isfinite(vol[:-timeout])), n, replace=False):
        side, order, mid0 = int(rng.choice([-1, 1])), None, E["midT"][i]
        for k in range(timeout):
            j = i + k
            if not np.isfinite(E["P"][j]):
                break
            order, fc = queue_step(order, side, *(E[c][j] for c in ("P", "A", "bq", "aq", "sv", "smin", "bv", "bmax")))
            if fc is not None:
                rows.append((vol[i], k + 1, side * (fc / PX / mid0 - 1) * 1e4))
                break
        else:
            rows.append((vol[i], np.nan, np.nan))
    d = pd.DataFrame(rows, columns=["vol", "secs", "cost_bp"])
    d["q"] = pd.qcut(d.vol, 5, labels=False)
    by = d.groupby("q").agg(vol=("vol", "median"), med_secs=("secs", "median"), cost_bp=("cost_bp", "mean"),
                            unfilled=("secs", lambda x: x.isna().mean()))
    return dict(legs=len(d), fill_rate=float(d.secs.notna().mean()), med_secs=float(d.secs.median()),
                cost_bp=float(d.cost_bp.mean()), by_vol=by.round(3).to_dict("index"))


# ── 자체점검 ─────────────────────────────────────────────────────────────────
def selftest() -> None:
    inf = np.inf
    o, f = queue_step(None, 1, 268730, 268731, 5.0, 9.0, 3.0, 268730, 0, -inf)   # 앞 5 중 3 소진 -> 대기, 앞 2
    assert f is None and o == (1, 268730, 2.0)
    o, f = queue_step(o, 1, 268730, 268731, 7.0, 9.0, 2.5, 268730, 0, -inf)      # 줄 자리 유지: 앞 2 - 2.5 = -0.5 > -1(내 수량)
    assert f is None and o[2] == -0.5
    o2, f = queue_step(o, 1, 268730, 268731, 7.0, 9.0, 0.5, 268730, 0, -inf)     # 누적 3.0 >= 2+1 -> 체결
    assert f == 268730 and o2 is None
    o, f = queue_step(o, 1, 268731, 268732, 7.0, 9.0, 0.0, 268731, 0, -inf)      # 최우선이 올랐다 -> 새 줄(앞 7)
    assert f is None and o == (1, 268731, 7.0)
    assert queue_step(o, 1, 268731, 268732, 7.0, 9.0, 0.0, 268730, 0, -inf)[1] == 268731   # 내 가격 아래 체결 = 뚫림
    assert queue_step(None, -2, 268730, 268731, 5.0, 1.0, 0, inf, 2.5, 268731)[1] is None  # 앞 1 + 내 2 > 2.5
    assert queue_step(None, -2, 268730, 268731, 5.0, 1.0, 0, inf, 3.0, 268731)[1] == 268731
    assert queue_step(o, 0, 268731, 268732, 7.0, 9.0, 0, -inf, 0, inf) == (None, None)     # 목표 도달 -> 취소
    r, pos = step_reward(0, 1, 100.0, 100.01, 100.02, True)
    assert pos == 1 and abs(r - (2.0 - FEE_BP)) < 1e-9
    tr = Trader()                                     # 봉 경계 롱 -> 지정가 체결 -> 봉 중간 시장가 청산
    assert not tr.needs_decision(0) and tr.needs_decision(299)
    e = dict(P=10000.0, A=10001.0, bq=1.0, aq=1.0, midT=100.005)
    assert tr.decide(2, True, e) == 0.0 and tr.tgt == 1 and tr.decide(2, False, e) == 0.0   # 포지션 없을 때 봉 중간 행동은 무시
    r = tr.tick(e, (2.5, inf, 0.0, -inf), 100.02)    # 앞 1 + 내 1 <= 2.5 -> 100.00 체결
    assert tr.pos == 1 and abs(r - (100.02 / 100.00 - 1) * 1e4) < 1e-9 and tr.needs_decision(5)
    r = tr.decide(2, False, dict(P=10002.0, A=10003.0, bq=1.0, aq=1.0, midT=100.025))
    assert tr.pos == 0 and abs(r - ((100.02 / 100.025 - 1) * 1e4 - TAKER_FEE_BP)) < 1e-9 and tr.n["taker_exits"] == 1
    assert abs(sum(tr.pnl.values()) - (100.02 / 100.00 - 1) * 1e4 - r) < 1e-9          # 손익 분해 합 = 보상 합
    assert len(tr.closed) == 1 and tr.closed[0]["exit"] == "taker" and tr.closed[0]["side"] == "long" \
        and abs(tr.closed[0]["bp"] - ((100.02 / 100.00 - 1) * 1e4 - TAKER_FEE_BP)) < 1e-3     # 원장 = 진입가→청산가 − 수수료
    ix4 = {"imb50": 0, "whale_z": 1, "retail_z": 2, "ret300": 3}
    ix6 = {"imb50": 0, "whale_z": 1, "retail_z": 2, "ret300": 3, "liq_long60": 4, "liq_short60": 5}
    X = lambda imb=0.0, wz=0.0, rz=0.0, r5=0.0, ll=0.0, ls=0.0: np.array([imb, wz, rz, r5, ll, ls])
    assert ArmPolicy("w80_z0.5_nochase").bar(X(0.2, r5=5), ix6, 0, 0, 0) == 0             # 이미 오른 뒤 롱 = 추격 -> 관망
    assert ArmPolicy("w80_z0.5_nochase").bar(X(0.2, r5=5), ix6, 0, 1, 0) == 1             # 보유 중이면 유지
    assert ArmPolicy("w80_z0.5_nochase").bar(X(0.2, r5=-5), ix6, 0, 0, 0) == 1 and ArmPolicy("w80_z0.5").bar(X(0.2, r5=5), ix6, 0, 0, 0) == 1
    assert ArmPolicy("w80_z0.5_pullback").bar(X(0.2, r5=-1), ix6, 0, 0, 0) == 0 and ArmPolicy("w80_z0.5_pullback").bar(X(0.2, r5=-8), ix6, 0, 0, 0) == 1
    assert ArmPolicy("w80_z0.5_nochase_liq").bar(X(0.2, r5=-1, ll=13.0), ix6, 0, 0, 0) == 0            # 롱 청산 버스트 -> 매수 금지
    assert ArmPolicy("w80_z0.5_nochase_liq").bar(X(-0.2, r5=1, ll=13.0), ix6, 0, 0, 0) == -1          # 매도는 허용
    p60 = ArmPolicy("w80_z0.5_nochase_w60")                                              # 고래: 정시에 정하고 60분 유지
    assert p60.bar(X(0, 0.8, -0.6, r5=-1), ix6, 3599, 0, 0) == 1 and p60.bar(X(0, 0, 0, r5=-1), ix6, 3899, 1, 0) == 1
    assert p60.bar(X(0, 0, 0), ix6, 7199, 1, 0) == 0                                      # 다음 정시에 고래 없음 -> 해제
    ph = ArmPolicy("w80_z0.5_nochase_holdloss")
    assert ph.bar(X(0), ix6, 0, 1, -5.0) == 1 and ph.bar(X(0), ix6, 300, 1, -5.0) == 0     # 손실 중 신호 꺼짐 -> 한 봉만 더
    assert ArmPolicy("w80_z0.5_nochase_holdloss").bar(X(0), ix6, 0, 1, 5.0) == 0          # 이익 중이면 바로 청산
    pa = ArmPolicy(f"{TEACHER}_a24")                                                        # 적응 문턱: 늘 |imb| .05 면 문턱 .05
    for k in range(200):
        pa.bar(X(0.05 if k % 2 else -0.05), ix6, 300 * k, 0, 0)
    assert pa.bar(X(0.06), ix6, 300 * 200, 0, 0) == 1 and ArmPolicy(TEACHER).bar(X(0.06), ix6, 0, 0, 0) == 0
    assert ArmPolicy(f"{TEACHER}_a24").bar(X(0.06), ix6, 0, 0, 0) == 0                     # 12시간 전엔 고정 문턱
    pm = ArmPolicy(f"{TEACHER}_a24_mh15")
    assert pm.bar(X(0.2), ix6, 0, 0, 0) == 1 and pm.bar(X(0), ix6, 600, 1, 0) == 1 and pm.bar(X(0), ix6, 900, 1, 0) == 0   # 15분 최소 보유
    ps = ArmPolicy(f"{TEACHER}_sticky60")
    assert ps.bar(X(0.2), ix6, 0, 0, 0) == 1 and ps.bar(X(0.2), ix6, 300, 1, 0) == 1 and ps.bar(X(0), ix6, 3000, 1, 0) == 1
    assert ps.bar(X(0), ix6, 3900, 1, 0) == 0 and ArmPolicy(f"{TEACHER}_sticky60").bar(X(-0.2), ix6, 600, 1, 0) == -1   # 60분 뒤 청산 · 반대면 전환
    assert dd_init([10, -30, 5, 40]) == [25.0, -30.0] and dd_init([-5, -5]) == [0.0, -10.0]   # 고점 0 에서 시작
    assert dd_step([25.0, -30.0], -10.0) == [25.0, -35.0] and dd_step([25.0, -30.0], 40.0) == [40.0, -30.0]
    pw = ArmPolicy("w100_z0.5_w60")                                                          # 고래 단독: 벽은 무시, 정시 고래 60분
    assert pw.bar(X(0.9), ix6, 299, 0, 0) == 0 and pw.bar(X(0.9, 0.8, -0.6), ix6, 3599, 0, 0) == 1 and pw.bar(X(-0.9), ix6, 3899, 1, 0) == 1
    p30 = ArmPolicy(f"{TEACHER}_a24_mh15_sticky30")                                        # 보유 연장 30분(최소 보유 15분 뒤)
    assert p30.bar(X(0.2), ix6, 0, 0, 0) == 1 and p30.bar(X(0), ix6, 1200, 1, 0) == 1 and p30.bar(X(0), ix6, 1800, 1, 0) == 0
    pt = ArmPolicy("w100_z0.5_w60_tp20")                                                     # 고래 익절 뒤 다음 정시까지 재진입 금지
    assert pt.bar(X(0, 0.8, -0.6), ix6, 3599, 0, 0) == 1
    pt.on_tp(3700)
    assert pt.bar(X(0), ix6, 3899, 0, 0) == 0 and pt.bar(X(0), ix6, 4199, 0, 0) == 0
    assert pt.bar(X(0, 0.8, -0.6), ix6, 7199, 0, 0) == 1                                    # 다음 정시: 맞대결이면 다시 진입
    assert sec_action("w100_z0.5_w60_tp20", 1, 1, 20.0, 30) == "exit" and sec_action("w100_z0.5_w60_tp20", 1, 1, 19.9, 30) is None
    assert sec_action("w80_z0.5_tp10", 1, 1, 10.0, 30) == "exit" and sec_action("w80_z0.5_tp10", 1, 0, 12.0, 30) is None
    assert sec_action("w80_z0.5_xc60", -1, 0, 0.0, 61) == "cancel" and sec_action("w80_z0.5_xc60", -1, 0, 0.0, 30) is None
    assert sec_action("w80_z0.5", 1, 1, 50.0, 200) is None
    tr.decide(1, True, e)                             # 포지션 없음 + 봉 경계 «없음» = 관망
    assert tr.idle and not (tr.decide(2, True, e) or tr.idle) and not (tr.decide(1, False, e) or tr.idle)
    r, pos = step_reward(1, -2, 100.04, 100.02, 100.00, True)    # 롱 1 -> 숏 1 (2개 매도)
    assert pos == -1 and abs(r - ((100.00 / 100.02 - 1) * 1e4 - 2 * (100.00 / 100.04 - 1) * 1e4 - 2 * FEE_BP)) < 1e-9
    # DepthBook == 기존 패널 빌더 dd_file (같은 시각 파일)
    from scripts.research_rt5_1s_panel_build_20260920 import dd_file
    f = sorted((ORDERFLOW / "depthdiff" / SYMBOL).glob("2026-09-25T03.jsonl*"))[0]
    a, b = dd_hour(f), dd_file(str(f))
    cols = [c for c in a.columns if c in b.columns]
    assert len(a) == len(b) > 3000 and np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True), cols
    # 규칙: 깊은 벽 + 고래 맞대결, 반대면 0
    ix = {"imb50": 0, "whale_z": 1, "retail_z": 2}
    assert rule_target(np.array([0.2, 0, 0]), ix, "w80_off") == 1 and rule_target(np.array([0.11, 0, 0]), ix, "w80_off") == 0
    assert rule_target(np.array([0.11, 0, 0]), ix, "w70_off") == 1
    assert rule_target(np.array([0.0, 0.8, -0.6]), ix, "w80_z0.5") == 1 and rule_target(np.array([0.0, 0.8, -0.6]), ix, "w80_z1") == 0
    assert rule_target(np.array([-0.2, 0.8, -0.6]), ix, "w80_z0.5") == 0 and rule_target(np.array([0.0, 0.8, -0.6]), ix, "w80_off") == 0
    noise = np.random.default_rng(3).normal(0, 50, 10)
    hist = {a: list(noise) for a in ARMS}
    assert pick_arm(hist) == TEACHER                                                        # 다 같으면 교사
    hist["w90_z1"] = list(noise + 30 + np.random.default_rng(4).normal(0, 5, 10))           # 짝 차이 +30±5 -> t≫2
    assert pick_arm(hist) == "w90_z1" and pick_arm({a: v[:5] for a, v in hist.items()}) == TEACHER   # 5일은 부족
    hist["w90_z1"] = list(noise + np.random.default_rng(5).normal(30, 100, 10))             # 평균 +30 이라도 잡음 크면 유지
    assert pick_arm(hist) == TEACHER
    hist = {a: list(noise) for a in ARMS}                                                   # 중간에 추가된 판 = 기록이 짧다
    hist[f"{TEACHER}_mv"] = list(noise[-8:] + 30 + np.random.default_rng(6).normal(0, 5, 8))
    assert pick_arm(hist) == f"{TEACHER}_mv"
    hist[f"{TEACHER}_mv"] = list(noise[-2:])                                                # 2일 vs 10일 -> 오류 없이 교사
    assert pick_arm(hist) == TEACHER
    ix8 = dict(ix6, whale_z_mv=6, retail_z_mv=7)                                            # mv 판은 움직이는 경계 z 만 본다
    x8 = np.array([0, 0, 0, -1, 0, 0, 0.8, -0.6])
    assert ArmPolicy(f"{TEACHER}_mv").bar(x8, ix8, 0, 0, 0) == 1 and ArmPolicy(TEACHER).bar(x8, ix8, 0, 0, 0) == 0
    one = lambda cut: agg_minutes(np.array([100.0, 100.0]), np.array([2000.0, 1.0]), np.array([0, 1000]), np.array([False, True]), cut=cut)
    g1 = one(lambda d: None)
    assert g1.w.iloc[0] == 200000.0 and g1.r.iloc[0] == -100.0 and g1.w2.isna().all()    # 경계 없는 날 = NaN(0 아님)
    g2 = one(lambda d: (50.0, 150000.0))
    assert g2.w2.iloc[0] == 200000.0 and g2.r2.iloc[0] == 0.0                              # $100 줄은 리테일 상한 $50 이상
    # 라이브 고래 z(@aggTrade 줄 -> whale_frame) == 아카이브 whale_minutes: 마지막 날 앞 3시간을 WS 메시지처럼 흘려 넣는다
    zips = sorted((OUT / "aggtrades").glob(f"{SYMBOL}-aggTrades-*.zip"))
    if len(zips) >= 25:
        last = zips[-1].name[-14:-4]
        lf = LiveFeed()
        lf.wbase = whale_raw([z.name[-14:-4] for z in zips[:-1]])
        t = pd.read_csv(zips[-1], usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
        t = t[t.transact_time < (_ts(last) + 3 * 3600) * 1000]
        for r in t.itertuples(index=False):
            lf._on_agg({"e": "aggTrade", "p": str(r.price), "q": str(r.quantity), "T": r.transact_time, "m": r.is_buyer_maker})
        live, ref = lf.whale_frame(), whale_minutes()
        mins = pd.RangeIndex(_ts(last), _ts(last) + 3 * 3600, 60)
        c8 = ["w", "r", "zw", "zr", "w2", "r2", "zw2", "zr2"]
        assert np.allclose(live.loc[mins, c8].to_numpy(float), ref.loc[mins, c8].to_numpy(float),
                           rtol=1e-9, atol=1e-6, equal_nan=True) and live.loc[mins, ["zw", "zw2"]].notna().all().all()
    # 집행 표(벡터) == 라이브의 창별 window_agg (같은 시각 파일·같은 틱)
    bt = sorted((ORDERFLOW / "bookticker" / SYMBOL).glob("2026-09-25T03.bt*"))[0]
    e = exec_hour(bt)
    tr = trades_day("2026-09-25")
    for s in np.random.default_rng(1).choice(e.index, 300, replace=False):
        T = (s + 1) * 1000 + LAT_MS
        got = window_agg(tr[(tr.ts > T) & (tr.ts <= T + 1000)], e.P[s], e.A[s])
        assert np.allclose(got, e.loc[s, ["sv", "smin", "bv", "bmax"]].to_numpy(float)), (s, got)
    assert (e.sv > 0).mean() > 0.05 and np.isfinite(e.smin).mean() > 0.5      # 대조군: 실제로 값이 있는 창이 충분
    assert window_agg(pd.DataFrame([], columns=["ts", "c", "qty", "sell"]), 1.0, 2.0) == (0.0, np.inf, 0.0, -np.inf)   # 체결 없는 창(라이브)
    # 피쳐 인과성: 미래 행을 바꿔도 과거 피쳐가 안 바뀐다
    if (OUT / "panel.parquet").exists():
        p = pd.read_parquet(OUT / "panel.parquet").iloc[:6000]
        f0 = features(p).iloc[:5000]
        p2 = p.copy()
        p2.iloc[5000:] = p2.iloc[5000:].to_numpy()[::-1]
        assert np.array_equal(f0.to_numpy(), features(p2).iloc[:5000].to_numpy())
        # LiveEnv(ReplayFeed) == SecEnv: 같은 행동열이면 보상이 1e-9 안에서 같다(실시간 경로가 학습 환경을 재현)
        p = pd.read_parquet(OUT / "panel.parquet")
        i0 = 200_000 + (10 - (p.index[200_000 + WARMUP] % 3600)) % 3600      # 시각 경계 직후 3000초(집행 표는 매시 끝 ~1초가 빈다)
        p = p.iloc[i0:i0 + WARMUP + 3000]
        ex = pd.read_parquet(OUT / "exec.parquet")
        A = arrays(p, ex)
        assert A["ok"][WARMUP:-1].all()
        rng = np.random.default_rng(0)
        bar_a, sec_a = rng.choice(3, 5000, p=[.4, .2, .4]), rng.choice(3, 5000, p=[.985, .01, .005])
        act = lambda o, i: bar_a[i] if o[-5] == 1.0 else sec_a[i]          # o[-5] = 봉 경계 표시(Trader.state 3번째)
        se = make_env_cls()(A, seq=True)
        o, _ = se.reset()
        s1, done = [], False
        while not done:
            o, r, done, *_ = se.step(act(o, len(s1)))
            s1.append((o, r))
        le = make_live_env_cls()(ReplayFeed(p, ex), A["X"].shape[1])
        o, _ = le.reset()
        s2 = []
        try:
            while len(s2) < len(s1) - 1:
                o, r, *_ = le.step(act(o, len(s2)))
                s2.append((o, r))
        except StopIteration:
            pass
        n = len(s2)
        assert n > 100 and n >= len(s1) - 2, (n, len(s1))
        assert np.allclose([r for _, r in s1[:n]], [r for _, r in s2], atol=1e-9)
        assert np.allclose(np.array([o for o, _ in s1[:n]]), np.array([o for o, _ in s2]), atol=1e-6)
        assert se.tr.n["maker_fills"] > 5 and se.tr.n["taker_exits"] >= 1, se.tr.n        # 대조군: 두 체결 경로가 실제로 돌았다
        sec_b = rng.choice(3, 5000, p=[.9, .08, .02])        # 관망 벌점은 학습 보상만 바꾸고 실손익은 그대로(자주 청산해 관망이 생기게)
        act2 = lambda o, i: bar_a[i] if o[-5] == 1.0 else sec_b[i]
        runs = []
        for pen in (0.0, 2.0):
            sp = make_env_cls()(A, seq=True, idle_penalty=pen)
            o, _ = sp.reset()
            rs, idles, done = [], 0, False
            while not done:
                o, r, done, *_ = sp.step(act2(o, len(rs)))
                rs.append(r)
                idles += sp.tr.idle
            runs.append((sum(rs), idles, sum(sp.tr.pnl.values())))
        (r0, i0_, p0), (r2, i2, p2) = runs
        assert i0_ == i2 > 0 and abs(p0 - p2) < 1e-9 and abs(r2 - (r0 - 2.0 * i2)) < 1e-6, runs
    # 10-04 새 판: 추세 신호·크기 / 가격×OI 1시간 / 역추세 금지 / 동일위험 배수 / Trader 손익 배수
    up = list(np.exp(np.linspace(0, 0.3, 40) + 0.01 * (-1.0) ** np.arange(40)) * 2000)   # 변동성 0 이면 크기 0 -- 잡음을 섞는다
    sig, size = trend4_signal(up)
    assert sig == 1.0 and 0 < size <= TREND_MAX and trend4_signal(up[:10]) == (0.0, 0.0)
    dn = up[::-1]; dn[-1] = dn[-8] * 1.01                  # 7일 전보다만 위 -> 부호 [+,−,−,−] 평균 −0.5
    assert trend4_signal(dn)[0] == -0.5
    ix7 = {"imb50": 0, "whale_z": 1, "retail_z": 2, "ret300": 3, "doi300": 4, "liq_long60": 5, "liq_short60": 6}
    X7 = lambda r5=0.0, o5=0.0, imb=0.0: np.array([imb, 0.0, 0.0, r5, o5, 0.0, 0.0])
    TREND.update(day=1, sig=0.5, size=-1.3)
    pv = ArmPolicy("trend4_vs")
    assert pv.bar(X7(), ix7, 299, 0, 0) == 1 and abs(pv.k - 1.3) < 1e-12 and ArmPolicy("trend4").bar(X7(), ix7, 299, 0, 0) == 1
    TREND.update(day=None)
    assert ArmPolicy("trend4").bar(X7(), ix7, 299, 0, 0) == 0              # 일봉 없으면 관망
    po = ArmPolicy("poi1h")
    for i in range(11):
        po.bar(X7(r5=-3.0, o5=+1.0), ix7, 300 * i + 899, 0, 0)            # 정시가 아닌 봉에서 시작 -- 5분 판정
    assert po.bar(X7(r5=-3.0, o5=+1.0), ix7, 300 * 11 + 899, 0, 0) == -1  # 1h −36bp(≤ −26.68) · OI↑ -> 숏(정시 아님)
    tt = [po.bar(X7(), ix7, 300 * k + 899, -1, 0) for k in range(12, 40)]
    assert tt[0] == -1 and tt[-1] == 0                                     # 조건이 꺼져도 마지막 발동 뒤 1시간은 유지, 그 뒤 관망
    assert sum(1 for v in tt if v == -1) == 3 + 12                         # 발동이 3봉 더 이어진 뒤(−33·−30·−27) 12봉 = 1시간
    po2 = ArmPolicy("poi1h")
    for i in range(12):
        t = po2.bar(X7(r5=-3.0, o5=-1.0), ix7, 300 * i + 299, 0, 0)
    assert t == 1                                                           # OI↓ -> 롱(롱 이탈 끝물 되돌림)
    pt = ArmPolicy("w80_off_vt")
    pt.seed_logp([2000.0 * (1 + 0.001 * i) for i in range(288)])           # 상승 추세 -> 지금가 > SMA
    assert pt.bar(X7(imb=-0.5), ix7, 299, 0, 0) == 0 and pt.bar(X7(imb=0.5), ix7, 599, 0, 0) == 1   # 숏 금지 · 롱 허용
    pe = ArmPolicy("w80_off_er")
    for i in range(30):
        pe.bar(X7(r5=(-1) ** i * 30.0), ix7, 300 * i + 299, 0, 0)          # 4h 변동성 ~30bp(기준 14.95 의 2배)
    assert pe.bar(X7(imb=0.5, r5=30.0), ix7, 9299, 0, 0) == 1 and abs(pe.k - 0.5) < 0.02
    e0 = dict(P=10000.0, A=10001.0, bq=1.0, aq=1.0, midT=100.005)       # 위 Trader 시험과 같은 호가(e 는 그 사이 집행 표로 바뀐다)
    t1, t0 = Trader(), Trader(); t1.k = 1.0                 # 기존 판(k=1)은 손익이 배수 도입 전과 같다(대시보드 두 판)
    for tt in (t1, t0):
        tt.decide(2, True, e0); tt.tick(e0, (2.5, inf, 0.0, -inf), 100.02); tt.decide(2, False, dict(P=10002.0, A=10003.0, bq=1.0, aq=1.0, midT=100.025))
    assert t1.pnl == t0.pnl and abs(sum(t1.pnl.values()) - ((100.02 / 100.00 - 1) * 1e4 + (100.02 / 100.025 - 1) * 1e4 - TAKER_FEE_BP)) < 1e-9
    tk = Trader(); tk.k = 2.0
    tk.decide(2, True, e0)
    r = tk.tick(e0, (2.5, inf, 0.0, -inf), 100.02)
    assert abs(r - 2 * (100.02 / 100.00 - 1) * 1e4) < 1e-9 and abs(sum(tk.pnl.values()) - r) < 1e-9
    assert all(a in ARMS for a in ("trend4", "trend4_vs", "poi1h", "w100_z0.5_w60_vt", f"{TEACHER}_a24_mh15_er"))
    # 10-05 기록 결함: 원장 bp_k = 진입 배수 반영(bp 는 1 단위 그대로) · 시가평가만 잇기
    assert t1.closed[0]["k"] == 1.0 and t1.closed[0]["bp_k"] == t1.closed[0]["bp"]
    tk = Trader(); tk.k = 2.0
    tk.decide(2, True, e0); tk.tick(e0, (2.5, inf, 0.0, -inf), 100.02)
    tk.k = 0.5                                              # 보유 중 배수가 바뀌어도(전환 결정) 원장은 진입 체결 때 배수
    tk.decide(2, False, dict(P=10002.0, A=10003.0, bq=1.0, aq=1.0, midT=100.025))
    c = tk.closed[0]
    assert c["k"] == 2.0 and abs(c["bp_k"] - 2 * c["bp"]) < 2e-3 and abs(c["bp"] - ((100.02 / 100.00 - 1) * 1e4 - TAKER_FEE_BP)) < 1e-3
    tk = Trader(); tk.pos, tk.k = -1, 1.5
    tk.mark(100.0, 100.01)
    assert abs(tk.pnl["hold"] + 1.5) < 1e-9
    if (OUT / PANEL).exists():                              # paper() 를 가짜 피드로: 재시작 이어받기 · 장중 낙폭 · 배수 판 · 빈 구간 시가평가
        import os
        import signal
        import tempfile
        ncol, i50 = len(features_cols()), features_cols().index("imb50")

        class Fake:                                         # 초 s 의 mid = m(s) · 봉 경계 관측 계산 동안 1초가 흐른다 · gaps 초는 호가 묵음
            def __init__(self, s0, s1, m, gaps=()):
                self.s0, self.s1, self.m, self.gaps, self.obs = s0, s1, m, set(gaps), False
                self.served, self.cnt = [pd.Series(dict(wz2=0.0, rz2=0.0))], {}

            def start(self):
                pass

            def wait_ready(self):
                return self.s0

            def wait_next(self, s):
                if s + 1 >= self.s1:
                    os.kill(os.getpid(), signal.SIGTERM)    # 재시작 = 종료 신호 -> 상태 저장 후 끝
                return s + 1

            def obs_row(self, s):
                self.obs = True
                x = np.zeros(ncol)
                x[i50] = 0.5                                # 깊은 매수벽 -> 교사 롱
                return x

            def exec_now(self, s):
                if s in self.gaps:
                    return dict(P=np.nan, A=np.nan, bq=np.nan, aq=np.nan, midT=np.nan, T=s)
                m, self.obs = self.m(s + 1 if self.obs else s), False
                return dict(P=round((m - 0.01) * PX), A=round((m + 0.01) * PX), bq=1.0, aq=1.0, midT=m, T=s)

            def window(self, T0, T1, pc, ac):
                return (0.0, pc - 1, 0.0, ac + 1)           # 양쪽 다 뚫림 -> 지정가 즉시 체결

        def boom(*a, **k):
            raise RuntimeError("selftest: REST 금지")
        g, D = globals(), 20_000
        s0, fetch0, trend0 = D * 86400 + 36000 - 1, g["fetch_closes"], dict(TREND)
        g["fetch_closes"] = boom
        TREND.update(day=D, sig=1.0, size=1.5)              # trend4_vs = 롱 × 1.5
        try:
            with tempfile.TemporaryDirectory() as td:
                out = Path(td)
                rd = lambda: json.loads((out / "state.json").read_text())
                # A: 진입(1999.99) -> 봉 경계 관측 중 +1 -> 호가 묵음 5초 사이 −2 -> 1999 에서 종료 신호
                paper(out, Fake(s0, s0 + 1000, lambda s: 2000.0 + (s > s0 + 300) - 2 * (s >= s0 + 402), gaps=range(s0 + 400, s0 + 405)))
                st = rd()
                a, v = st["arms"][TEACHER], st["arms"]["trend4_vs"]
                want = (1999 / 1999.99 - 1) * 1e4               # 빈 구간 둘(+5bp·−10bp)을 다 시가평가해야 진입가→지금 mid 와 같다
                assert a["pos"] == 1 and abs(a["pnl_today"] - want) < 0.05 and st["gap_s"] == 6, (a, want, st["gap_s"])
                assert abs(v["pnl_today"] - 1.5 * want) < 0.08 and v["k"] == 1.5 and abs(v["unr"] - 1.5 * a["unr"]) < 0.02
                assert abs(a["mdd"] - want) < 0.1
                # B: 같은 날 재시작(정지 중 2002 로) -> 오늘 손익·포지션·배수·장중 낙폭을 잇는다, 가짜 청산 없음
                paper(out, Fake(s0 + 1500, s0 + 1800, lambda s: 2002.0))
                st = rd()
                a, v = st["arms"][TEACHER], st["arms"]["trend4_vs"]
                want = (2002 / 1999.99 - 1) * 1e4
                assert a["pos"] == 1 and a["fills"] == 0 and abs(a["pnl_today"] - want) < 0.05, (a, want)
                assert abs(v["pnl_today"] - 1.5 * want) < 0.08 and abs(a["mdd"] - (1999 / 1999.99 - 1) * 1e4) < 0.1
                assert not (out / "daily.jsonl").exists() and not [x for x in (out / "trades.jsonl").read_text().splitlines()
                                                                   if json.loads(x)["arm"] == TEACHER]
                # C: 다음 날 재시작 -> 끊긴 날 = 저장값(restart_cut·기록 시간) · 새 날은 정지 중 움직임부터
                paper(out, Fake((D + 1) * 86400 + 299, (D + 1) * 86400 + 400, lambda s: 2003.0))
                d = [json.loads(x) for x in (out / "daily.jsonl").read_text().splitlines()]
                assert len(d) == 1 and d[0]["restart_cut"] and abs(d[0]["pnl"][TEACHER] - want) < 0.05 and abs(d[0]["hours"] - 1300 / 3600) < 0.006, d
                st = rd()
                a = st["arms"][TEACHER]
                assert a["pos"] == 1 and a["days"] == 2 and abs(a["pnl_today"] - (2003 / 2002 - 1) * 1e4) < 0.05
                assert abs(a["cum"] - (2003 / 1999.99 - 1) * 1e4) < 0.1, a
        finally:
            g["fetch_closes"] = fetch0
            TREND.clear(); TREND.update(trend0)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "build", "build-exec", "build-whale", "build-zh", "calib", "train", "eval", "live",
                                    "paper", "paper-backtest", "calib-coin"])
    ap.add_argument("--replay-day", help="live 를 저장 패널의 이 날짜로 시험(예: 2026-09-29)")
    ap.add_argument("--start", default="2026-09-20")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--idle", type=float, default=0.0, help="관망 벌점 bp(학습 보상만, 봉 경계에서 «없음→없음»마다)")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "build":
        OUT.mkdir(parents=True, exist_ok=True)
        p = build(a.start, a.end)
        p.to_parquet(OUT / f"panel{TAG}.parquet")
        print(len(p), "rows", p.notna().mean().round(3).to_string())
    elif a.cmd == "build-zh":                         # panel_zh.parquet = panel + 제우스·호메로스 (모의 매매 엔진의 panel 은 그대로)
        p = add_zeus_homer(pd.read_parquet(OUT / "panel.parquet"))
        p.to_parquet(OUT / "panel_zh.parquet")
        z = [c for c in p.columns if c.startswith(("zs_", "hm_"))]
        print(len(z), "열 추가 · 결측률", round(float(p[z].iloc[WARMUP:].isna().mean().mean()), 4))
    elif a.cmd == "build-whale":                      # panel.parquet 에 wz·rz 열을 붙인다(aggtrades/ 에 30일 이전부터 필요)
        p = add_whale(pd.read_parquet(OUT / f"panel{TAG}.parquet").drop(columns=["wz", "rz"], errors="ignore"))
        p.to_parquet(OUT / f"panel{TAG}.parquet")
        print(p[["wz", "rz"]].describe().round(3).to_string())
    elif a.cmd == "build-exec":
        e = build_exec(a.start, a.end)
        e.to_parquet(OUT / f"exec{TAG}.parquet")
        print(len(e), "rows", e.describe().T.round(3).to_string())
    elif a.cmd == "paper":
        paper()
    elif a.cmd == "paper-backtest":
        r = paper_backtest()
        for k, v in r["table"].items():
            print(f"{k:10s} {np.mean(v):+7.1f} 양수 {sum(x > 0 for x in v)}/{len(v)} | " + " ".join(f"{x:+5.0f}" for x in v))
        print("활성 판   ", " ".join(r["active"]))
        print(f"선택 트랙  {np.mean(r['selected']):+7.1f} | " + " ".join(f"{x:+5.0f}" for x in r["selected"]))
        print(f"고정 교사  {np.mean(r['teacher']):+7.1f}")
    elif a.cmd == "calib-coin":                       # 예: RL_SYMBOL=SOLUSDT ... calib-coin --start 2026-09-27 --end 2026-10-04
        print(json.dumps(calib_coin(a.start, a.end)))
    elif a.cmd == "calib":
        print(json.dumps(calib(), indent=1))
    elif a.cmd == "train":
        print(train(a.seed, a.steps, a.idle))
    elif a.cmd == "live":
        rp = None
        if a.replay_day:
            lo = _ts(a.replay_day)
            rp = pd.read_parquet(OUT / "panel.parquet").loc[lo - WARMUP:lo + 86400 - 1]
        env = live(a.seed, rp, a.steps, LIVE if rp is None else OUT / f"live_replay_{a.replay_day}", a.idle)
        print(f"steps {env.k} · pnl {sum(env.rewards):+.1f}bp")
    else:
        print(json.dumps(evaluate(a.seed, idle=a.idle), indent=1))


if __name__ == "__main__":
    main()
