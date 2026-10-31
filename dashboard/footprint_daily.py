"""일봉 풋프린트 (2026-10-08, 사용자 «일봉 풋프린트를 최대치로 · 확대/축소 · 델타·거래대금·OI·청산 · 30일 이상»).

원천(ETH 만, 날짜 = UTC 자정 = KST 09:00, 바이낸스 일봉과 같은 경계):
  ① 소급본 data/footprint_daily/ETHUSDT_{days,cells}.parquet — aggTrades 아카이브(2020-01~)로 만든 것
     (scripts/build_eth_footprint_daily_20261008.py). OI 는 2022-01 부터, 청산은 없음.
  ② 그 뒤 날 — 체결 테이프(trade_tape_1s): lake 봉인일 → hot(봉인 전 날·오늘). OI = oi_1s 그날 마지막 값.
  ③ 청산(바이낸스 강제주문, long = 롱 포지션 청산) — lake + hot liquidations. 수집 시작(2026-09-20) 전은 «모름»(None),
     시작이 걸친 날은 반쪽(part=2).
가격 칸 = $1(테이프 0.1 칸을 내림으로 묶음). 화면 행 크기는 ?row= 달러로 서버가 묶는다.
2026-10-20 XRP: for_asset("xrp") = 같은 코드의 코인별 사본(칸 CELL 0.0005 ≈ 3.5bp @ $1.4 -- ETH $1 ≈ 4bp @ $2.5k 와 같은 상대 폭 ·
  테이프 칸 0.0001 의 5배라 정수 나눗셈으로 묶음). 저장 bin = floor(가격 / CELL) 정수 번호 · 응답 가격·?row= 는 달러.
닫힌 날은 한 번 계산해 캐시(봉인일은 안 바뀐다), 오늘만 TODAY_TTL 로 다시 센다.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST_DIR = ROOT / "data" / "footprint_daily"
# 코인 -> (선물 심볼, 테이프 칸 = 수집기 BUCKETS, 일봉 칸 CELL, 응답 가격 자릿수)
ASSETS = {"eth": ("ETHUSDT", 0.1, 1.0, 4), "xrp": ("XRPUSDT", 0.0001, 0.0005, 6),
          "btc": ("BTCUSDT", 1.0, 30.0, 2)}   # 2026-10-31 BTC $30 ≈ 3.9bp @ $77.5k(ETH $1 ≈ 4bp 와 같은 상대 폭) · 테이프 칸 1.0 의 30배
_ASSET = globals().get("_ASSET", "eth")      # for_asset 가 사본에 미리 심는다
SYMBOL, TAPE_BUCKET, CELL, PRICE_DP = ASSETS[_ASSET]
COIN, TAPE_PER_CELL = SYMBOL[:-4], round(CELL / TAPE_BUCKET)
HOT_SYM = SYMBOL.lower()     # 🔴hot 의 trade_tape_1s·oi_1s 는 소문자(ethusdt), liquidations 만 대문자(10-08 배포 직후 실측 -- 대문자로 물으면 빈 결과)
MAX_CELL_DAYS = 400          # 한 요청에 칸을 줄 최대 일수(확대 상태에선 화면이 ~100일을 넘지 않는다)
TODAY_TTL = 30.0
CTX_DAYS = 45                # OI·청산을 hot/lake 에서 읽는 깊이(그 전 OI 는 소급본, 청산은 수집 시작 이후뿐)

_lock = threading.Lock()
_hist: dict[str, Any] = {"mtime": None, "days": None, "cells": {}}
_recent: dict[str, Any] = {"days": {}, "cells": {}}      # 날짜 문자열 -> 닫힌 날 결과(불변)
_today: dict[str, Any] = {"at": 0.0, "day": None, "summ": None, "cells": None}
_ctx_lake: dict[str, Any] = {"at": 0.0, "key": None, "val": None}     # lake(봉인일) OI·청산 -- 1시간 캐시
CTX_LAKE_TTL = 3600.0


def for_asset(asset: str):
    """코인별 모듈. 캐시(_hist·_recent·_today·_h4·_heat …)가 모듈 전역이라 코인마다 따로 싣는다 -- eth = 이 모듈.
    ponytail: 모듈 사본 = 상태를 DailyFootprint 인스턴스로 옮기는 리팩터 대신. 코인이 3개를 넘거나 상태가 더 늘면 그쪽으로."""
    if asset == _ASSET:
        return sys.modules[__name__]
    spec = importlib.util.spec_from_file_location(f"{__name__}__{asset}", __file__)
    m = importlib.util.module_from_spec(spec)
    m._ASSET = asset
    spec.loader.exec_module(m)
    return m


def _ds(d) -> str:
    return d.isoformat() if isinstance(d, date) else str(d)[:10]


def _ms_day(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).date().isoformat()


def load_hist(hist_dir: Path = HIST_DIR) -> tuple[pd.DataFrame | None, dict]:
    """소급본(없으면 None). 파일이 바뀌면 다시 읽는다."""
    fd, fc = hist_dir / f"{SYMBOL}_days.parquet", hist_dir / f"{SYMBOL}_cells.parquet"
    if not (fd.exists() and fc.exists()):
        return None, {}
    mt = (fd.stat().st_mtime, fc.stat().st_mtime)
    with _lock:
        if _hist["mtime"] != mt:
            days = pd.read_parquet(fd)
            days["day"] = days.day.map(_ds)
            cells = pd.read_parquet(fc)
            cells["day"] = cells.day.map(_ds)
            by = {d: (g.bin.to_numpy(np.int64), g.buy.to_numpy(float), g.sell.to_numpy(float))
                  for d, g in cells.groupby("day", sort=False)}
            _hist.update(mtime=mt, days=days, cells=by)
        return _hist["days"], _hist["cells"]


def day_from_tape(sec: np.ndarray, pbin: np.ndarray, buy: np.ndarray, sell: np.ndarray) -> tuple[dict, tuple]:
    """한 날의 테이프 행(초, 테이프 칸, 매수, 매도) → (일 요약, CELL 칸). 시가·종가 = 첫·끝 초의 VWAP, 고저 = 체결 칸 극값."""
    px = pbin.astype(float) * TAPE_BUCKET
    q = buy + sell
    o = np.argsort(sec, kind="stable")
    sec, px, q, buy, sell = sec[o], px[o], q[o], buy[o], sell[o]
    first, last, has = sec == sec[0], sec == sec[-1], q > 0
    vw = lambda m: float(np.average(px[m], weights=q[m])) if q[m].sum() > 0 else float(px[m][0])  # noqa: E731
    summ = {"open": vw(first), "close": vw(last), "high": float(px[has].max()), "low": float(px[has].min()),
            "buy_qty": float(buy.sum()), "sell_qty": float(sell.sum()),
            "buy_quote": float((buy * px).sum()), "sell_quote": float((sell * px).sum())}
    b = np.floor_divide(pbin.astype(np.int64)[o], TAPE_PER_CELL)   # 정수 나눗셈 -- 부동소수 경계 새는 것 없음(ETH = 옛 floor(px+1e-9) 와 같다)
    s = pd.DataFrame({"b": b, "buy": buy, "sell": sell}).groupby("b", sort=True).sum()
    return summ, (s.index.to_numpy(np.int64), s.buy.to_numpy(float), s.sell.to_numpy(float))


def tape_rows_by_day(read_lake: Callable, read_hot: Callable, start: date, today: date, since: int = 0) -> dict[str, pd.DataFrame]:
    """[start, today] 날짜별 테이프 행. lake 에 있는 날은 lake, 없는 날(봉인 전)은 hot(since 초부터 -- 4시간 칸 다시 세기가 그날 0시부터 읽지 않게)."""
    out: dict[str, pd.DataFrame] = {}
    lk = read_lake("tape", start, today + timedelta(days=1), "ts_sec, price_bin, buy_qty, sell_qty")
    if lk is not None and len(lk):
        lk = lk.assign(day=pd.to_datetime(lk.ts_sec, unit="s").dt.strftime("%Y-%m-%d"))
        out.update(dict(tuple(lk.groupby("day"))))
    missing = [_ds(d) for d in pd.date_range(start, today).date if _ds(d) not in out]
    if missing:
        t0 = max(since, int(datetime.fromisoformat(missing[0]).replace(tzinfo=timezone.utc).timestamp()))
        hot = read_hot("SELECT ts_sec, price_bin, buy_qty, sell_qty FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ?",
                       [HOT_SYM, t0])
        if hot:
            h = pd.DataFrame(hot, columns=["ts_sec", "price_bin", "buy_qty", "sell_qty"])
            h["day"] = pd.to_datetime(h.ts_sec, unit="s").dt.strftime("%Y-%m-%d")
            out.update({d: g for d, g in h.groupby("day") if d in set(missing)})
    return out


def _ctx_from_lake(read_lake: Callable, start: date, today: date) -> tuple[dict, dict, int | None]:
    """lake 의 OI(그날 마지막 값)·청산(일 합). 1초 OI 가 하루 8.6만 행이라 1시간 캐시한다(봉인일은 안 바뀐다)."""
    key, now = (start, today), time.time()
    if _ctx_lake["key"] == key and now - _ctx_lake["at"] < CTX_LAKE_TTL:
        return _ctx_lake["val"]
    oi: dict[str, float] = {}
    liq: dict[str, list[float]] = {}
    o = read_lake("oi_1s", start, today + timedelta(days=1), "ts_ms, open_interest")
    if o is not None and len(o):
        o = o.sort_values("ts_ms").assign(day=lambda x: pd.to_datetime(x.ts_ms, unit="ms").dt.strftime("%Y-%m-%d"))
        oi.update(o.groupby("day").open_interest.last().astype(float).to_dict())
    lq = read_lake("liquidations", None, None, "ts_ms, side, usd")
    liq_from = int(lq.ts_ms.min()) if lq is not None and len(lq) else None
    if lq is not None and len(lq):
        lq = lq.assign(day=pd.to_datetime(lq.ts_ms, unit="ms").dt.strftime("%Y-%m-%d"))
        for (d, sd), v in lq.groupby(["day", "side"]).usd.sum().items():
            liq.setdefault(d, [0.0, 0.0])[0 if sd == "long" else 1] += float(v)
    _ctx_lake.update(at=now, key=key, val=(oi, liq, liq_from))
    return oi, liq, liq_from


def oi_liq_by_day(read_lake: Callable, read_ctx: Callable, start: date, today: date) -> tuple[dict, dict, int | None]:
    """(날짜 -> 그날 마지막 OI(ETH), 날짜 -> [롱청산$, 숏청산$], 청산 수집 시작 ms). lake 봉인일은 lake, 그 뒤는 hot."""
    loi, lliq, liq_from = _ctx_from_lake(read_lake, start, today)
    oi, liq = dict(loi), {d: list(v) for d, v in lliq.items()}
    lake_oi, lake_liq = set(oi), set(liq)
    first_hot = max([start] + [date.fromisoformat(d) + timedelta(days=1) for d in lake_oi])   # 봉인 안 된 날만 hot 에서
    since = int(datetime.combine(first_hot, datetime.min.time(), timezone.utc).timestamp() * 1000)
    for ts, v in read_ctx("SELECT ts_ms, open_interest FROM oi_1s WHERE symbol = ? AND ts_ms >= ? ORDER BY ts_ms",
                          [HOT_SYM, since]) or []:
        if (d := _ms_day(ts)) not in lake_oi:
            oi[d] = float(v)                      # 시간순이라 마지막 값이 남는다
    for ts, sd, usd in read_ctx("SELECT ts_ms, side, usd FROM liquidations WHERE symbol = ? AND ts_ms >= ?",
                                [SYMBOL, since]) or []:
        if (d := _ms_day(ts)) not in lake_liq:
            liq.setdefault(d, [0.0, 0.0])[0 if sd == "long" else 1] += float(usd or 0.0)
            liq_from = int(ts) if liq_from is None else min(liq_from, int(ts))
    return oi, liq, liq_from


COLS = ("d", "o", "h", "l", "c", "bq", "sq", "bv", "sv", "oi", "ll", "ls", "src", "part")


def build_days(hist_days: pd.DataFrame | None, recent: dict[str, dict], oi: dict, liq: dict,
               liq_from: int | None, today: str) -> dict[str, list]:
    """일 요약 열 묶음(열 이름 → 배열). 소급본이 있는 날은 소급본(aggTrades)이 이긴다."""
    rows: dict[str, dict] = {}
    if hist_days is not None:
        for r in hist_days.itertuples(index=False):
            rows[r.day] = {"open": r.open, "high": r.high, "low": r.low, "close": r.close, "buy_qty": r.buy_qty,
                           "sell_qty": r.sell_qty, "buy_quote": r.buy_quote, "sell_quote": r.sell_quote,
                           "oi": None if pd.isna(r.oi) else float(r.oi), "src": "a"}
    for d, s in recent.items():
        rows.setdefault(d, {**s, "oi": None, "src": "t"})
    liq_day0 = _ms_day(liq_from) if liq_from else None
    cols: dict[str, list] = {k: [] for k in COLS}
    for d in sorted(rows):
        r = rows[d]
        if d in oi and (r["oi"] is None or d == today):
            r["oi"] = float(oi[d])
        lq = (liq.get(d) or [0.0, 0.0]) if liq_day0 and d >= liq_day0 else None   # 수집 중 0 건 = 진짜 0
        vals = (d, r["open"], r["high"], r["low"], r["close"], r["buy_qty"], r["sell_qty"], r["buy_quote"], r["sell_quote"],
                r["oi"], lq[0] if lq else None, lq[1] if lq else None, r["src"])
        for k, v in zip(cols, vals):
            cols[k].append(round(float(v), 4) if isinstance(v, (float, np.floating, int, np.integer)) and k != "d" else v)
        cols["part"].append(1 if d == today else (2 if d == liq_day0 else 0))   # 1 = 형성 중 · 2 = 청산 수집 첫날(반쪽)
    return cols


def rebin(bins: np.ndarray, buy: np.ndarray, sell: np.ndarray, row: int) -> list:
    """CELL 칸 → row 칸(CELL 단위 정수). [첫 칸 하한($), [매수…], [매도…]] (빈 칸 0 채움)."""
    if not len(bins):
        return [0, [], []]
    k = np.floor_divide(bins, row)
    lo, n = int(k.min()), int(k.max() - k.min() + 1)
    bb, ss = np.zeros(n), np.zeros(n)
    np.add.at(bb, k - lo, buy); np.add.at(ss, k - lo, sell)
    return [round(lo * row * CELL, PRICE_DP), np.round(bb, 3).tolist(), np.round(ss, 3).tolist()]


# ── 4시간봉 풋프린트 (2026-10-20 사용자 «4h 버튼 = 1주일 데이터를 4시간봉 풋프린트로» → «1d 에서 데이터 양만 4h 로») ───────
#   같은 테이프(lake 봉인일 → hot)를 UTC 00·04·08·12·16·20시(KST 09·13·17·21·01·05) 경계로 묶는다 · 지난 7일 + 형성 중 1개.
#   열 = 일봉과 같다: 델타·거래대금(테이프) · OI(칸 마지막 값) · 청산 롱/숏 $(청산 수집기) · 청산 밀도 배경 = h4_heat_payload(칸 마감 시점).
#   닫힌 칸은 경계 뒤 2분이 지나면 캐시(늦게 들어오는 1초 테이프 몫) · 형성 중 칸은 TODAY_TTL 마다 다시 센다.
H4_SEC, H4_DAYS, H4_SETTLE = 14400, 7, 120
H4_BACKFILL, H4_GAP_TTL = 47 * 3600, 600.0   # 테이프 수집기 backfill_loop 소급 한계(BINANCE_BACKFILL_LOOKBACK_S) · «없음» 칸 다시 보기 간격
_h4: dict[str, Any] = {"closed": {}, "live": {}, "at": 0.0, "gap_at": 0.0}
_h4_heat: dict[str, tuple] = {}
_h4_lock = threading.Lock()


def h4_key(ts_sec: float) -> str:
    return datetime.fromtimestamp(int(ts_sec) // H4_SEC * H4_SEC, timezone.utc).strftime("%Y-%m-%dT%H")


def _h4_sec(key: str) -> int:
    return int(datetime.strptime(key, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc).timestamp())


def _h4_keys(ts_sec: np.ndarray) -> np.ndarray:
    return (np.asarray(ts_sec, dtype=np.int64) // H4_SEC) * H4_SEC


def h4_split(frames: dict[str, pd.DataFrame]) -> dict[str, tuple[dict, tuple]]:
    """날짜별 테이프 행 → 4시간 칸별 (요약, $1 칸)."""
    if not frames:
        return {}
    df = pd.concat(frames.values(), ignore_index=True)
    return {h4_key(t): day_from_tape(g.ts_sec.to_numpy(), g.price_bin.to_numpy(), g.buy_qty.to_numpy(float), g.sell_qty.to_numpy(float))
            for t, g in df.groupby(_h4_keys(df.ts_sec), sort=True)}


def _after_lake(df: pd.DataFrame | None, col: str, per_sec: int, floor: int) -> int:
    """lake 에 있는 마지막 날 다음 날 0시(같은 단위) -- 그 전은 lake, 그 뒤는 hot(겹쳐 두 번 세지 않게). lake 가 비면 floor."""
    if df is None or not len(df):
        return floor
    last = datetime.fromtimestamp(int(df[col].max()) // per_sec, timezone.utc).date() + timedelta(days=1)
    return max(floor, int(datetime.combine(last, datetime.min.time(), timezone.utc).timestamp()) * per_sec)


def h4_ctx(read_lake: Callable, read_ctx: Callable, t0: int, now: float) -> dict[str, dict]:
    """[t0, now] 4시간 칸별 {oi: 칸 마지막 OI(코인), ll·ls: 롱·숏 청산 $}. lake 봉인일 → 그 다음 날부터 hot."""
    d0, d1 = datetime.fromtimestamp(t0, timezone.utc).date(), datetime.fromtimestamp(now, timezone.utc).date() + timedelta(days=1)
    out: dict[str, dict] = {}
    put = lambda key, name, v: out.setdefault(key, {}).__setitem__(name, v)   # noqa: E731
    # OI -- 칸 마지막 값
    lk = read_lake("oi_1s", d0, d1, "ts_ms, open_interest")
    lk = lk[lk.ts_ms >= t0 * 1000] if lk is not None and len(lk) else None
    hot = read_ctx("SELECT ts_ms, open_interest FROM oi_1s WHERE symbol = ? AND ts_ms >= ? ORDER BY ts_ms",
                   [HOT_SYM, _after_lake(lk, "ts_ms", 1000, t0 * 1000)]) or []
    oi = pd.concat([x for x in (lk, pd.DataFrame(hot, columns=["ts_ms", "open_interest"])) if x is not None and len(x)], ignore_index=True) \
        if (lk is not None and len(lk)) or hot else None
    if oi is not None:
        oi = oi.sort_values("ts_ms")
        for k, v in oi.groupby(_h4_keys(oi.ts_ms.to_numpy(np.int64) // 1000)).open_interest.last().items():
            put(h4_key(k), "oi", float(v))
    # 청산 -- 롱(롱 포지션 청산)·숏 $ 합
    lq = read_lake("liquidations", d0, d1, "ts_ms, side, usd")
    lq = lq[lq.ts_ms >= t0 * 1000] if lq is not None and len(lq) else None
    hq = read_ctx("SELECT ts_ms, side, usd FROM liquidations WHERE symbol = ? AND ts_ms >= ?",
                  [SYMBOL, _after_lake(lq, "ts_ms", 1000, t0 * 1000)]) or []
    rows = [x for x in (lq, pd.DataFrame(hq, columns=["ts_ms", "side", "usd"])) if x is not None and len(x)]
    if rows:
        q = pd.concat(rows, ignore_index=True)
        for (k, sd), v in q.groupby([_h4_keys(q.ts_ms.to_numpy(np.int64) // 1000), q.side]).usd.sum().items():
            put(h4_key(k), "ll" if sd == "long" else "ls", float(v))
    return out


def h4_cols(buckets: dict[str, tuple[dict, tuple]], live_key: str) -> dict[str, list]:
    """일봉과 같은 열 묶음(build_days 그대로 -- 리뷰: 열 조립 두 벌). 청산 0 = 진짜 0 · OI 없음 = None.
    liq_from=1(1970) = 4h 창(지난 7일)은 전부 청산 수집 시작(09-20) 뒤 -- «모름»·반쪽 칸이 없다."""
    s = {k: v[0] for k, v in buckets.items()}
    return build_days(None, s, {k: r["oi"] for k, r in s.items() if r.get("oi") is not None},
                      {k: [r.get("ll", 0.0), r.get("ls", 0.0)] for k, r in s.items()}, 1, live_key)


class DailyFootprint:
    """서버가 하나 둔다. read_lake(stream, start, end, cols) · read_hot(sql, params) · read_ctx(sql, params) 를 주입한다."""

    def __init__(self, read_lake: Callable, read_hot: Callable, read_ctx: Callable, hist_dir: Path = HIST_DIR):
        self.read_lake, self.read_hot, self.read_ctx, self.hist_dir = read_lake, read_hot, read_ctx, hist_dir

    def _recent(self, now: float) -> tuple[dict, dict]:
        """소급본 끝 다음 날 ~ 오늘의 (일 요약, $1 칸). 닫힌 날은 캐시, 오늘은 TODAY_TTL."""
        hist_days, _ = load_hist(self.hist_dir)
        today = datetime.fromtimestamp(now, timezone.utc).date()
        start = date.fromisoformat(hist_days.day.max()) + timedelta(days=1) if hist_days is not None else today - timedelta(days=30)
        closed = [d for d in pd.date_range(start, today - timedelta(days=1)).date if _ds(d) not in _recent["days"]]
        stale = not (_today["day"] == _ds(today) and now - _today["at"] < TODAY_TTL)
        if closed or stale:
            got = tape_rows_by_day(self.read_lake, self.read_hot, closed[0] if closed else today, today)
            for d, g in got.items():
                s, c = day_from_tape(g.ts_sec.to_numpy(), g.price_bin.to_numpy(),
                                     g.buy_qty.to_numpy(float), g.sell_qty.to_numpy(float))
                if d == _ds(today):
                    _today.update(at=now, day=d, summ=s, cells=c)
                elif d < _ds(today):
                    _recent["days"][d], _recent["cells"][d] = s, c
        days, cells = dict(_recent["days"]), dict(_recent["cells"])
        if _today["day"] == _ds(today) and _today["summ"]:
            days[_today["day"]], cells[_today["day"]] = _today["summ"], _today["cells"]
        return days, cells

    def days_payload(self, now: float) -> dict:
        hist_days, _ = load_hist(self.hist_dir)
        recent, _ = self._recent(now)
        today = datetime.fromtimestamp(now, timezone.utc).date()
        start = today - timedelta(days=CTX_DAYS)
        if hist_days is not None:                   # 소급본이 오래돼도 그 뒤 날의 OI 가 비지 않게
            start = min(start, date.fromisoformat(hist_days.day.max()) + timedelta(days=1))
        oi, liq, liq_from = oi_liq_by_day(self.read_lake, self.read_ctx, start, today)
        cols = build_days(hist_days, recent, oi, liq, liq_from, _ds(today))
        return {"symbol": SYMBOL, "tz": "UTC", "cols": cols, "liq_from_ms": liq_from,
                "hist_until": None if hist_days is None else hist_days.day.max(), "at": now}

    def cells_payload(self, now: float, frm: str, to: str, row: int) -> dict:
        _, hist_cells = load_hist(self.hist_dir)
        _, recent_cells = self._recent(now)
        d0, d1 = date.fromisoformat(frm), date.fromisoformat(to)
        if d1 < d0 or (d1 - d0).days > 4000:
            raise ValueError("range")
        have = [_ds(d) for d in pd.date_range(d0, d1).date if _ds(d) in hist_cells or _ds(d) in recent_cells]
        if len(have) > MAX_CELL_DAYS:              # 제한은 «있는 날 수»로 -- 원천 사이 빈 날이 달력 폭을 부풀려도 된다
            raise ValueError("range")
        out = {}
        for k in have:
            c = hist_cells.get(k)
            out[k] = rebin(*(c if c is not None else recent_cells[k]), row)
        return {"row": round(row * CELL, PRICE_DP), "cells": out}

    def _h4(self, now: float) -> tuple[dict[str, tuple[dict, tuple]], str]:
        """지난 H4_DAYS 일의 4시간 칸 + 형성 중 칸. 반환 (키 → (요약, $1 칸), 형성 중 키)."""
        cur = int(now) // H4_SEC * H4_SEC
        first = cur - H4_DAYS * 86400
        want = [h4_key(t) for t in range(first, cur, H4_SEC)]
        settled = lambda k: now >= _h4_sec(k) + H4_SEC + H4_SETTLE   # noqa: E731 -- 닫히고 2분 지남 = 불변
        with _h4_lock:
            closed = _h4["closed"]
            if now - _h4["gap_at"] >= H4_GAP_TTL:
                # 리뷰: «없음»으로 굳힌 칸도 수집기가 아직 메울 수 있는 동안(칸 끝 + 47h)은 10분마다 다시 본다(메워진 칸이 재시작 전까지 안 보였다)
                # ponytail: 콜드 스타트에 hot DB 가 잠깐 없으면 47h 넘은 hot 전용 칸은 재시작 전까지 «없음» -- 생기면 gap 다시 보기를 7일로
                for k in [k for k, v in closed.items() if v is None and now < _h4_sec(k) + H4_SEC + H4_BACKFILL]:
                    del closed[k]
                _h4["gap_at"] = now
            missing = [k for k in want if k not in closed and k not in _h4["live"]]   # 막 닫힌(아직 2분 전) 칸은 TTL 로 다시 센다
            if missing or now - _h4["at"] >= TODAY_TTL:
                # 리뷰: 안 굳은 가장 앞 칸(빠진 칸 · 지난번 live -- 한가한 틈이 자정을 넘으면 live 의 옛 칸이 missing 에 안 잡혀
                #   다시 세기에서 빠지고 화면 since 병합이 지웠다)부터만 다시 센다. 그 앞 칸은 전부 closed 라 읽을 필요가 없다
                #   (전에는 그날 0시부터 -- 형성 칸 하나에 최대 20시간치 1초 테이프).
                t0 = max(first, min(_h4_sec(k) for k in [*missing, *_h4["live"], h4_key(cur)]))
                got = h4_split(tape_rows_by_day(self.read_lake, self.read_hot, datetime.fromtimestamp(t0, timezone.utc).date(),
                                                datetime.fromtimestamp(now, timezone.utc).date(), since=t0))
                ctx = h4_ctx(self.read_lake, self.read_ctx, t0, now)
                live = {}
                for k, v in got.items():
                    if _h4_sec(k) < t0:
                        continue                        # lake 날은 하루 통째로 온다 -- t0 앞 칸은 이미 closed
                    v[0].update(ctx.get(k, {}))
                    if settled(k):
                        closed.setdefault(k, v)         # 이미 굳힌 칸은 덮어쓰지 않는다
                    else:
                        live[k] = v                     # 형성 중 · 막 닫힌 칸
                for k in missing:
                    if k not in got and settled(k):
                        closed.setdefault(k, None)      # 리뷰: 테이프에 아예 없는 칸(수집기 정지) = «없음»으로 굳힌다 -- 안 그러면 7일 내내 매 호출 다시 읽었다
                for k in [k for k in closed if k < h4_key(first)]:
                    del closed[k]
                _h4.update(live=live, at=now)
            out = {**{k: closed[k] for k in want if closed.get(k) is not None}, **_h4["live"]}
        return out, h4_key(cur)

    def h4_payload(self, now: float) -> dict:
        b, live_key = self._h4(now)
        return {"symbol": SYMBOL, "tz": "UTC", "period": "4h", "cols": h4_cols(b, live_key), "at": now}

    def h4_cells_payload(self, now: float, frm: str, to: str, row: int) -> dict:
        b, _ = self._h4(now)
        return {"row": round(row * CELL, PRICE_DP), "cells": {k: rebin(*v[1], row) for k, v in b.items() if frm <= k <= to}}


HEAT_LOOKBACK_H = 168     # 일봉 청산 히트맵 입력 창(7일 1시간봉) -- 라이브 5분 화면은 24h, 일봉 한 칸이 24h 라 7일로


def heat_for_day(k1h: pd.DataFrame, d: date) -> list[tuple[float, float, float]]:
    """그날 UTC 마감(다음 날 0시) 시점 -- heat_at 의 일봉판."""
    return heat_at(k1h, pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1))


def heat_at(k1h: pd.DataFrame, end: pd.Timestamp) -> list[tuple[float, float, float]]:
    """그날 UTC 마감 시점의 추정 청산 밀도 [(칸 가격, w 0~1, s)] -- 라이브 청산맵과 같은 계산(compute_spliced_levels 의 heatmap_bins:
    현재가 아래 = (고+저)/2 진입 · 위 = 종가 진입, w = 쪽마다 최대값 대비). s = 그 칸이 지도 전체(두 쪽 합)에서 차지하는 몫
    -- 화면이 «s × 2 × 그날 OI 달러»로 금액을 낸다(청산맵 tier_profile 의 2×OI 규모와 같은 뜻).
    k1h: timestamp(봉 시작, UTC)·high·low·close·volume. 마감 뒤 봉은 안 본다(인과). 20봉 미만이면 []."""
    from scripts import live_liquidation_map_20260824 as lm
    win = k1h[(k1h.timestamp < end) & (k1h.timestamp >= end - pd.Timedelta(hours=HEAT_LOOKBACK_H))].reset_index(drop=True)
    if len(win) < 20 or win.timestamp.iloc[0] > end - pd.Timedelta(hours=HEAT_LOOKBACK_H - 2):   # 500봉 밖으로 밀린 날 = 짧은 창으로 조용히 그리지 않는다
        return []
    cp = float(win.close.iloc[-1])
    common = lm._prepare_common(win, cp)
    if common is None:
        return []
    bm = lm._bins_from_common(common, (win.high.to_numpy(float) + win.low.to_numpy(float)) / 2.0)
    bc = lm._bins_from_common(common, None)
    if bm is None or bc is None:
        return []
    bw = common["bin_width"]
    sides = [{b: w for b, w in bm.items() if b * bw < cp}, {b: w for b, w in bc.items() if b * bw >= cp}]
    tot = sum(sum(x.values()) for x in sides)
    out = []
    for side in sides:
        mx = max(side.values(), default=0.0)
        if mx > 0:
            out += [(round(b * bw, PRICE_DP), round(w / mx, 4), w / tot) for b, w in side.items() if w > 0]
    return sorted(out)


def rebin_heat(price: np.ndarray, w: np.ndarray, row: int, s: np.ndarray | None = None) -> list:
    """히트맵 칸 → row 칸(CELL 단위). [첫 칸 하한($), [칸 안 최대 w…], [칸 안 몫 합 s…]] (s 가 없으면 둘째 목록까지)."""
    if not len(price):
        return [0, []] if s is None else [0, [], []]
    k = np.floor(np.asarray(price, float) / (row * CELL) + 1e-9).astype(np.int64)
    lo, n = int(k.min()), int(k.max() - k.min() + 1)
    out = np.zeros(n)
    np.maximum.at(out, k - lo, w)
    lo_px = round(lo * row * CELL, PRICE_DP)
    if s is None:
        return [lo_px, np.round(out, 3).tolist()]
    sm = np.zeros(n)
    np.add.at(sm, k - lo, s)
    return [lo_px, np.round(out, 3).tolist(), [float(f"{v:.4g}") for v in sm]]


_heat: dict[str, Any] = {"mtime": None, "by": {}, "until": None, "recent": {}}


def load_heat(hist_dir: Path = HIST_DIR) -> tuple[dict, str | None]:
    """히트맵 소급본(scripts/build_eth_liq_heat_daily_20261008.py) -> (날짜 -> (price[], w[], s[] | None), 마지막 날)."""
    f = hist_dir / f"{SYMBOL}_liqheat.parquet"
    if not f.exists():
        return {}, None
    with _lock:
        if _heat["mtime"] != f.stat().st_mtime:
            h = pd.read_parquet(f)
            h["day"] = h.day.map(_ds)
            _heat.update(mtime=f.stat().st_mtime, until=h.day.max(),
                         by={d: (g.price.to_numpy(float), g.w.to_numpy(float), g.s.to_numpy(float) if "s" in g else None)
                             for d, g in h.groupby("day", sort=False)})
        return _heat["by"], _heat["until"]


def klines_1h(raw: list | None, now: float) -> pd.DataFrame | None:
    """바이낸스 /fapi/v1/klines(1h) 원시 행 -> 마감된 봉만(timestamp·high·low·close·volume)."""
    if not raw:
        return None
    k = pd.DataFrame({"timestamp": pd.to_datetime([int(r[0]) for r in raw], unit="ms", utc=True),
                      "high": [float(r[2]) for r in raw], "low": [float(r[3]) for r in raw],
                      "close": [float(r[4]) for r in raw], "volume": [float(r[5]) for r in raw],
                      "close_ms": [int(r[6]) for r in raw]})
    return k[k.close_ms < now * 1000].drop(columns="close_ms").reset_index(drop=True)


def heat_payload(now: float, frm: str, to: str, row: int, raw_1h: list | None = None, hist_dir: Path = HIST_DIR) -> dict:
    """보이는 날들의 청산 히트맵(row 달러 칸, 칸 안 최대 w). 소급본 뒤 날(오늘 포함)은 raw_1h 로 계산 -- 닫힌 날은 캐시.
    ponytail: raw_1h 는 최근 500봉(~20일)이라 소급본이 그보다 오래되면 사이 날은 «모름» -- 빌더를 다시 돌리면 된다."""
    by, until = load_heat(hist_dir)
    d0, d1 = date.fromisoformat(frm), date.fromisoformat(to)
    if d1 < d0 or (d1 - d0).days > 4000:
        raise ValueError("range")
    today = datetime.fromtimestamp(now, timezone.utc).date()
    k = None
    out = {}
    for d in pd.date_range(d0, min(d1, today)).date:
        key = _ds(d)
        c = by.get(key)
        if c is None and (until is None or key > until):
            c = _heat["recent"].get(key) if d < today else None
            if c is None:
                k = k if k is not None else klines_1h(raw_1h, now)
                if k is not None and len(k):
                    pw = heat_for_day(k, d)
                    c = (np.array([x[0] for x in pw]), np.array([x[1] for x in pw]), np.array([x[2] for x in pw]))
                    if d < today and k.timestamp.iloc[-1] >= pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=23):   # 그날 마지막 봉까지 있을 때만 굳힌다
                        _heat["recent"][key] = c
        if c is not None and len(c[0]):
            out[key] = rebin_heat(c[0], c[1], row, c[2])
    return {"row": round(row * CELL, PRICE_DP), "heat": out, "lookback_h": HEAT_LOOKBACK_H, "until": until}


def h4_heat_payload(now: float, frm: str, to: str, row: int, raw_1h: list | None = None) -> dict:
    """4시간 칸마다 그 칸 마감(형성 중이면 마지막 닫힌 1시간봉) 시점의 추정 청산 밀도 -- 일봉 히트맵과 같은 계산(7일 1시간봉).
    닫힌 칸은 캐시. 범위는 60칸(10일)까지."""
    t0, t1 = _h4_sec(frm), _h4_sec(to)
    if t1 < t0 or t1 - t0 > 60 * H4_SEC:
        raise ValueError("range")
    k = klines_1h(raw_1h, now)
    out: dict[str, list] = {}
    if k is None or not len(k):
        return {"row": round(row * CELL, PRICE_DP), "heat": out, "lookback_h": HEAT_LOOKBACK_H}
    last_end = k.timestamp.iloc[-1] + pd.Timedelta(hours=1)
    for t in range(t0, t1 + 1, H4_SEC):
        key, bend = h4_key(t), pd.Timestamp(t + H4_SEC, unit="s", tz="UTC")
        closed = bend <= last_end
        c = _h4_heat.get(key) if closed else None
        if c is None:
            pw = heat_at(k, min(bend, last_end))
            c = (np.array([x[0] for x in pw]), np.array([x[1] for x in pw]), np.array([x[2] for x in pw]))
            if closed:
                _h4_heat[key] = c
        if len(c[0]):
            out[key] = rebin_heat(c[0], c[1], row, c[2])
    for old in [x for x in list(_h4_heat) if x < h4_key(now - (H4_DAYS + 2) * 86400)]:
        _h4_heat.pop(old, None)             # 리뷰: to_thread 요청이 겹쳐 돈다 -- list() 로 떠서 지우고, 이미 지워졌으면 그냥 간다
    return {"row": round(row * CELL, PRICE_DP), "heat": out, "lookback_h": HEAT_LOOKBACK_H}


# ── 5분봉 1주일 (2026-10-20 사용자 «5m, 30m, 1h 모두 데이터를 일주일 치를 가져와서 보여줘») ──────────────────────
#   라이브 풋프린트 창(서버 링 12시간)의 과거분 -- server.api_footprint_week 가 테이프(coin_flow.okx_tape_footprint)를 읽어 여기서 묶는다.
def week_bars(bn: dict, ox: dict | None, d0: int, d1: int, bar_s: int = 300) -> dict[int, dict[int, list[float]]]:
    """바이낸스·OKX 테이프 봉 → 라이브 /api/footprint 처럼 두 거래소 칸(6열) 합. 두 테이프가 다 온전한(1분봉 대조 통과) 봉만 --
    하나라도 비면 그 봉은 «모름»(라이브의 OKX 합류 규칙처럼 한 창에 1·2거래소 봉을 섞지 않는다). ox=None(OKX 테이프 없음) = 바이낸스만."""
    full = lambda g, b: all(m in g["ok_min"] for m in range(b, b + bar_s, 60))   # noqa: E731
    out: dict[int, dict[int, list[float]]] = {}
    for b in range(d0, d1, bar_s):
        if b not in bn["bars"] or not full(bn, b) or (ox is not None and not full(ox, b)):
            continue
        cells = {k: list(v) for k, v in bn["bars"][b].items()}
        for k, v in (ox["bars"].get(b, {}) if ox is not None else {}).items():
            t = cells.setdefault(k, [0.0] * 6)
            for j in range(6):
                t[j] += v[j]
        out[b] = cells
    return out


def oi_deltas(oc: dict[int, tuple[float, float]], bar_s: int = 300) -> dict[int, tuple[float, float, int]]:
    """{봉: (첫 OI, 끝 OI)} → {봉: (Δ, 끝, 공백)}. Δ = 직전 봉 끝 → 이 봉 끝 · 앞 봉이 비었으면 봉 안에서만(공백 1)
    -- server.oi_5m_buckets(바이낸스)·api_oi_5m(OKX)와 같은 규칙."""
    out: dict[int, tuple[float, float, int]] = {}
    prev: tuple[int, float] | None = None
    for b in sorted(oc):
        o, c = oc[b]
        gap = prev is None or b - prev[0] > bar_s
        out[b] = ((c - o) if gap else (c - prev[1]), c, int(gap))
        prev = (b, c)
    return out


def parse_row(v: str | None) -> int:
    """화면이 고르는 행 크기($) → CELL 칸 수. 신뢰경계 — 1~500 칸만(ETH $1~500 · XRP 0.0005~0.25 · BTC $30~15,000)."""
    try:
        r = round(float(v) / CELL) if v else 5
    except (ValueError, OverflowError):        # 숫자 아님 · nan · inf
        return 5
    return max(1, min(500, r))


def selftest() -> None:
    # 테이프 한 날: 테이프 칸 → CELL 칸 · 첫/끝 초 VWAP · 공격자 합 (ETH $2000.0 · 2000.9 · 2001.5 · 1999.5 · XRP 2.0000 · 2.0009 · 2.0015 · 1.9995)
    u = TAPE_BUCKET
    sec = np.array([10, 10, 20, 30]); pb = np.array([20000, 20009, 20015, 19995])
    bq = np.array([1.0, 0.0, 2.0, 0.0]); sq = np.array([0.0, 1.0, 0.0, 3.0])
    s, (b, bu, se) = day_from_tape(sec, pb, bq, sq)
    assert abs(s["open"] - 20004.5 * u) < 1e-9 and abs(s["close"] - 19995 * u) < 1e-9 and s["high"] == 20015 * u and s["low"] == 19995 * u
    want = {"ETH": ([1999, 2000, 2001], [0, 1, 2], [3, 1, 0]), "XRP": ([3999, 4000, 4001, 4003], [0, 1, 0, 2], [3, 0, 1, 0]),
            "BTC": ([666, 667], [1, 2], [4, 0])}[COIN]
    assert (list(b), list(bu), list(se)) == want, (list(b), list(bu), list(se))
    assert abs(s["buy_quote"] - (20000 + 2 * 20015) * u) < 1e-6
    _, (b2, _, _) = day_from_tape(np.array([1]), np.array([20010]), np.array([1.0]), np.array([0.0]))
    assert list(b2) == [{"ETH": 2001, "XRP": 4002, "BTC": 667}[COIN]]   # 칸 하한 정확히(ETH 2001.0 · XRP 2.0010) 은 그 칸
    lo, bb, ss = rebin(np.array([1999, 2000, 2004, 2006]), np.array([1, 2, 3, 4.0]), np.array([0, 1, 0, 1.0]), 5)
    assert lo == round(1995 * CELL, PRICE_DP) and bb == [1, 5, 4] and ss == [0, 1, 1], (lo, bb, ss)
    assert rebin_heat(np.array([1999.5 * CELL, 2004.9 * CELL]), np.array([0.2, 1.0]), 5)[:2] == [round(1995 * CELL, PRICE_DP), [0.2, 1.0]]
    hist = pd.DataFrame([{"day": "2026-09-19", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "buy_qty": 1.0,
                          "sell_qty": 1.0, "buy_quote": 1.0, "sell_quote": 1.0, "oi": 10.0}])
    one = {"open": 1.0, "high": 2.0, "low": 1.0, "close": 2.0, "buy_qty": 1.0, "sell_qty": 1.0, "buy_quote": 1.0, "sell_quote": 1.0}
    lf = int(datetime(2026, 9, 20, 13, tzinfo=timezone.utc).timestamp() * 1000)
    c = build_days(hist, {"2026-09-20": one, "2026-09-21": one}, {"2026-09-21": 20.0}, {"2026-09-20": [5.0, 1.0]}, lf, "2026-09-21")
    assert c["d"] == ["2026-09-19", "2026-09-20", "2026-09-21"] and c["ll"] == [None, 5.0, 0.0] and c["part"] == [0, 2, 1]
    assert c["oi"] == [10.0, None, 20.0] and c["src"] == ["a", "t", "t"]
    assert parse_row("abc") == 5 and parse_row("0") == 1 and parse_row(str(9999 * CELL)) == 500
    assert parse_row(str(25 * CELL)) == 25 and parse_row("nan") == 5 and parse_row("inf") == 5 and parse_row(None) == 5
    # 4시간 칸: UTC 03:59:59 와 04:00:00 은 다른 칸
    t = int(datetime(2026, 10, 19, 3, 59, 59, tzinfo=timezone.utc).timestamp())
    fr = {"2026-10-19": pd.DataFrame({"ts_sec": [t, t + 1, t + 2], "price_bin": [20000, 20010, 20020],
                                      "buy_qty": [1.0, 2.0, 0.0], "sell_qty": [0.0, 0.0, 3.0]})}
    sp = h4_split(fr)
    assert list(sp) == ["2026-10-19T00", "2026-10-19T04"], list(sp)
    assert sp["2026-10-19T00"][0]["buy_qty"] == 1.0 and sp["2026-10-19T04"][0]["sell_qty"] == 3.0
    assert _h4_sec("2026-10-19T04") == t + 1 and h4_key(t) == "2026-10-19T00"
    hc = h4_cols(sp, "2026-10-19T04")
    assert hc["d"] == ["2026-10-19T00", "2026-10-19T04"] and hc["part"] == [0, 1] and hc["oi"] == [None, None] and hc["ll"] == [0.0, 0.0]
    # 맥락(OI 마지막 값 · 청산 롱/숏) · lake/hot 이음(겹치는 날은 lake 만)
    ms = lambda h, m=0: int(datetime(2026, 10, 19, h, m, tzinfo=timezone.utc).timestamp() * 1000)   # noqa: E731
    lake_oi = pd.DataFrame({"ts_ms": [ms(1), ms(3)], "open_interest": [10.0, 11.0]})
    def lake(stream, a, b, cols):
        return {"oi_1s": lake_oi, "liquidations": pd.DataFrame({"ts_ms": [ms(5)], "side": ["long"], "usd": [7.0]})}[stream]
    def ctx(sql, params):
        assert params[1] == ms(0) + 86400000, params     # lake 마지막 날(10-19) 다음 날부터 hot
        return [(ms(0) + 86400000 + 1, 99.0)] if "oi_1s" in sql else [(ms(0) + 86400000 + 5, "short", 4.0)]
    t00 = ms(0) // 1000
    cx = h4_ctx(lake, ctx, t00, t00 + 86400 + 3600)
    assert cx["2026-10-19T00"]["oi"] == 11.0 and cx["2026-10-19T04"]["ll"] == 7.0, cx
    # 다시 세기(TTL)에도 같은 날 앞 칸의 OI 가 남는다(10-20 버그: 두 번째 호출에서 None)
    tape = pd.DataFrame({"ts_sec": [t00 + 3600 * h for h in (1, 5, 9)], "price_bin": [20000] * 3, "buy_qty": [1.0] * 3, "sell_qty": [0.0] * 3})
    oi3 = pd.DataFrame({"ts_ms": [(t00 + 3600 * h) * 1000 for h in (1, 5, 9)], "open_interest": [1.0, 2.0, 3.0]})
    lk2 = lambda stream, a, b, cols: {"tape": tape, "oi_1s": oi3, "liquidations": None}[stream]   # noqa: E731
    f2 = DailyFootprint(lk2, lambda q, p: [], lambda q, p: [])
    _h4.update(closed={}, live={}, at=0.0, gap_at=0.0)
    n0 = t00 + 10 * 3600                                  # 08시 칸 형성 중
    one = f2.h4_payload(n0)["cols"]
    two = f2.h4_payload(n0 + TODAY_TTL + 1)["cols"]       # TTL 지나 다시 셈
    assert one["oi"] == [1.0, 2.0, 3.0] and two["oi"] == [1.0, 2.0, 3.0], (one["oi"], two["oi"])
    # 리뷰: 형성 칸(10-19 T20) 뒤 한가한 틈이 자정을 넘어도 그 칸이 굳어 남는다 · 다시 세기는 안 굳은 가장 앞 칸부터(그날 0시·7일 전 아님)
    #   · 테이프에 아예 없는 칸은 «없음»으로 굳어 다음 호출이 7일을 다시 읽지 않는다
    rows, since = [(t00 + 86340, 20000, 1.0, 0.0)], []
    def hot(q, p):
        since.append(p[1])
        return [r for r in rows if r[0] >= p[1]]
    f3 = DailyFootprint(lambda *a: None, hot, lambda q, p: [])
    _h4.update(closed={}, live={}, at=0.0, gap_at=t00 + 86400 + 4 * 3600 + 180)   # 이 검사 동안은 «없음» 다시 보기 안 함(아래에서 따로)
    assert f3.h4_payload(t00 + 86370)["cols"]["d"] == ["2026-10-19T20"] and since == [t00 + 86370 // H4_SEC * H4_SEC - H4_DAYS * 86400]
    rows.append((t00 + 86400 + 4 * 3600 + 60, 20010, 1.0, 0.0))
    since.clear()
    hc3 = f3.h4_payload(t00 + 86400 + 4 * 3600 + 180)["cols"]
    assert hc3["d"] == ["2026-10-19T20", "2026-10-20T04"] and hc3["part"] == [0, 1] and since == [t00 + 20 * 3600], (hc3["d"], since)
    assert _h4["closed"]["2026-10-20T00"] is None and _h4["closed"]["2026-10-19T20"] is not None
    rows.append((t00 + 86400 + 60, 20005, 1.0, 0.0))     # 수집기가 나중에 메운 칸 -- 10분 뒤 다시 보면 나타난다
    assert f3.h4_payload(t00 + 86400 + 4 * 3600 + 180 + TODAY_TTL)["cols"]["d"] == ["2026-10-19T20", "2026-10-20T04"]
    assert f3.h4_payload(t00 + 86400 + 4 * 3600 + 180 + H4_GAP_TTL)["cols"]["d"] == ["2026-10-19T20", "2026-10-20T00", "2026-10-20T04"]
    _h4.update(closed={}, live={}, at=0.0, gap_at=0.0)
    assert cx["2026-10-20T00"]["oi"] == 99.0 and cx["2026-10-20T00"]["ls"] == 4.0, cx
    # 5분봉 1주일: 두 테이프 다 온전한 봉만 칸 합 · OKX 테이프가 없으면 바이낸스만 · OI Δ 규칙
    okm = lambda b0, n: set(range(b0, b0 + 60 * n, 60))   # noqa: E731
    bn = {"bars": {0: {10: [1.0, 2, 0, 0, 1, 0]}, 300: {10: [5.0, 0, 0, 0, 0, 0]}, 600: {11: [1.0, 1, 0, 0, 0, 0]}}, "ok_min": okm(0, 15)}
    ox = {"bars": {0: {10: [1.0, 1, 1, 0, 0, 0], 12: [0, 3.0, 0, 0, 0, 0]}}, "ok_min": okm(0, 10)}   # 600 봉은 OKX 가 다 못 봄
    wb = week_bars(bn, ox, 0, 900)
    assert sorted(wb) == [0, 300] and wb[0] == {10: [2.0, 3, 1, 0, 1, 0], 12: [0, 3.0, 0, 0, 0, 0]} and bn["bars"][0][10][0] == 1.0, wb
    assert sorted(week_bars(bn, None, 0, 900)) == [0, 300, 600]
    assert sorted(week_bars({**bn, "ok_min": okm(0, 14)}, None, 0, 900)) == [0, 300]          # 바이낸스 마지막 1분 빔 = 그 봉 모름
    od = oi_deltas({0: (10.0, 12.0), 300: (12.5, 13.0), 900: (20.0, 19.0)})
    assert od == {0: (2.0, 12.0, 1), 300: (1.0, 13.0, 0), 900: (-1.0, 19.0, 1)}, od
    print(f"{COIN} selftest OK -- 테이프 일 요약·$1 칸·행 묶기·청산 모름/반쪽/0·OI 병합·행 입력 검증·4시간 칸 경계·OI/청산 4시간 묶기·lake→hot 이음"
          "·5분봉 1주일(두 거래소 온전한 봉만 합·OI Δ)")


if __name__ == "__main__":
    for a in ASSETS:
        for_asset(a).selftest()
