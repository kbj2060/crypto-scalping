"""일봉 풋프린트 (2026-10-08, 사용자 «일봉 풋프린트를 최대치로 · 확대/축소 · 델타·거래대금·OI·청산 · 30일 이상»).

원천(ETH 만, 날짜 = UTC 자정 = KST 09:00, 바이낸스 일봉과 같은 경계):
  ① 소급본 data/footprint_daily/ETHUSDT_{days,cells}.parquet — aggTrades 아카이브(2020-01~)로 만든 것
     (scripts/build_eth_footprint_daily_20261008.py). OI 는 2022-01 부터, 청산은 없음.
  ② 그 뒤 날 — 체결 테이프(trade_tape_1s): lake 봉인일 → hot(봉인 전 날·오늘). OI = oi_1s 그날 마지막 값.
  ③ 청산(바이낸스 강제주문, long = 롱 포지션 청산) — lake + hot liquidations. 수집 시작(2026-09-20) 전은 «모름»(None),
     시작이 걸친 날은 반쪽(part=2).
가격 칸 = $1(테이프 0.1 칸을 내림으로 묶음). 화면 행 크기는 ?row= 달러로 서버가 묶는다.
닫힌 날은 한 번 계산해 캐시(봉인일은 안 바뀐다), 오늘만 TODAY_TTL 로 다시 센다.
"""
from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST_DIR = ROOT / "data" / "footprint_daily"
SYMBOL, COIN, TAPE_BUCKET = "ETHUSDT", "ETH", 0.1
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
    """한 날의 테이프 행(초, 0.1 칸, 매수, 매도) → (일 요약, $1 칸). 시가·종가 = 첫·끝 초의 VWAP, 고저 = 체결 칸 극값."""
    px = pbin.astype(float) * TAPE_BUCKET
    q = buy + sell
    o = np.argsort(sec, kind="stable")
    sec, px, q, buy, sell = sec[o], px[o], q[o], buy[o], sell[o]
    first, last, has = sec == sec[0], sec == sec[-1], q > 0
    vw = lambda m: float(np.average(px[m], weights=q[m])) if q[m].sum() > 0 else float(px[m][0])  # noqa: E731
    summ = {"open": vw(first), "close": vw(last), "high": float(px[has].max()), "low": float(px[has].min()),
            "buy_qty": float(buy.sum()), "sell_qty": float(sell.sum()),
            "buy_quote": float((buy * px).sum()), "sell_quote": float((sell * px).sum())}
    b = np.floor(px + 1e-9).astype(np.int64)          # 0.1×k 가 부동소수로 k/10 아래로 새는 것 막음
    s = pd.DataFrame({"b": b, "buy": buy, "sell": sell}).groupby("b", sort=True).sum()
    return summ, (s.index.to_numpy(np.int64), s.buy.to_numpy(float), s.sell.to_numpy(float))


def tape_rows_by_day(read_lake: Callable, read_hot: Callable, start: date, today: date) -> dict[str, pd.DataFrame]:
    """[start, today] 날짜별 테이프 행. lake 에 있는 날은 lake, 없는 날(봉인 전)은 hot."""
    out: dict[str, pd.DataFrame] = {}
    lk = read_lake("tape", start, today + timedelta(days=1), "ts_sec, price_bin, buy_qty, sell_qty")
    if lk is not None and len(lk):
        lk = lk.assign(day=pd.to_datetime(lk.ts_sec, unit="s").dt.strftime("%Y-%m-%d"))
        out.update(dict(tuple(lk.groupby("day"))))
    missing = [_ds(d) for d in pd.date_range(start, today).date if _ds(d) not in out]
    if missing:
        t0 = int(datetime.fromisoformat(missing[0]).replace(tzinfo=timezone.utc).timestamp())
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
    cols: dict[str, list] = {k: [] for k in ("d", "o", "h", "l", "c", "bq", "sq", "bv", "sv", "oi", "ll", "ls", "src", "part")}
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
    """$1 칸 → row 달러 칸. [첫 칸 하한, [매수…], [매도…]] (빈 칸 0 채움)."""
    if not len(bins):
        return [0, [], []]
    k = np.floor_divide(bins, row)
    lo, n = int(k.min()), int(k.max() - k.min() + 1)
    bb, ss = np.zeros(n), np.zeros(n)
    np.add.at(bb, k - lo, buy); np.add.at(ss, k - lo, sell)
    return [lo * row, np.round(bb, 3).tolist(), np.round(ss, 3).tolist()]


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

    def days_payload(self, now: float, since: str | None = None) -> dict:
        hist_days, _ = load_hist(self.hist_dir)
        recent, _ = self._recent(now)
        today = datetime.fromtimestamp(now, timezone.utc).date()
        start = today - timedelta(days=CTX_DAYS)
        if hist_days is not None:                   # 소급본이 오래돼도 그 뒤 날의 OI 가 비지 않게
            start = min(start, date.fromisoformat(hist_days.day.max()) + timedelta(days=1))
        oi, liq, liq_from = oi_liq_by_day(self.read_lake, self.read_ctx, start, today)
        cols = build_days(hist_days, recent, oi, liq, liq_from, _ds(today))
        if since:
            keep = [i for i, d in enumerate(cols["d"]) if d >= since]
            cols = {k: [v[i] for i in keep] for k, v in cols.items()}
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
        return {"row": row, "cells": out}


HEAT_LOOKBACK_H = 168     # 일봉 청산 히트맵 입력 창(7일 1시간봉) -- 라이브 5분 화면은 24h, 일봉 한 칸이 24h 라 7일로


def heat_for_day(k1h: pd.DataFrame, d: date) -> list[tuple[float, float]]:
    """그날 UTC 마감 시점의 추정 청산 밀도 [(칸 가격, w 0~1)] -- 라이브 청산맵과 같은 함수(compute_spliced_levels).
    k1h: timestamp(봉 시작, UTC)·high·low·close·volume. 마감 뒤 봉은 안 본다(인과). 20봉 미만이면 []."""
    from scripts.live_liquidation_map_20260824 import compute_spliced_levels
    end = pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1)
    win = k1h[(k1h.timestamp < end) & (k1h.timestamp >= end - pd.Timedelta(hours=HEAT_LOOKBACK_H))]
    if len(win) < 20:
        return []
    res = compute_spliced_levels(win.reset_index(drop=True), float(win.close.iloc[-1]))
    return [(float(b["price"]), float(b["weight_pct"])) for b in res.get("heatmap_bins") or [] if b["weight_pct"] > 0]


def rebin_heat(price: np.ndarray, w: np.ndarray, row: int) -> list:
    """히트맵 칸 → row 달러 칸, 칸 안 최대값(0~1 척도 유지). [첫 칸 하한, [w…]]."""
    if not len(price):
        return [0, []]
    k = np.floor_divide(price, row).astype(np.int64)
    lo, n = int(k.min()), int(k.max() - k.min() + 1)
    out = np.zeros(n)
    np.maximum.at(out, k - lo, w)
    return [lo * row, np.round(out, 3).tolist()]


_heat: dict[str, Any] = {"mtime": None, "by": {}, "until": None, "recent": {}}


def load_heat(hist_dir: Path = HIST_DIR) -> tuple[dict, str | None]:
    """히트맵 소급본(scripts/build_eth_liq_heat_daily_20261008.py) -> (날짜 -> (price[], w[]), 마지막 날)."""
    f = hist_dir / f"{SYMBOL}_liqheat.parquet"
    if not f.exists():
        return {}, None
    with _lock:
        if _heat["mtime"] != f.stat().st_mtime:
            h = pd.read_parquet(f)
            h["day"] = h.day.map(_ds)
            _heat.update(mtime=f.stat().st_mtime, until=h.day.max(),
                         by={d: (g.price.to_numpy(float), g.w.to_numpy(float)) for d, g in h.groupby("day", sort=False)})
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
                    c = (np.array([p for p, _ in pw]), np.array([w for _, w in pw]))
                    if d < today:
                        _heat["recent"][key] = c
        if c is not None and len(c[0]):
            out[key] = rebin_heat(c[0], c[1], row)
    return {"row": row, "heat": out, "lookback_h": HEAT_LOOKBACK_H, "until": until}


def parse_row(v: str | None) -> int:
    """화면이 고르는 행 크기($). 신뢰경계 — 정수 1~500 만."""
    try:
        r = int(v or 5)
    except ValueError:
        return 5
    return max(1, min(500, r))


def selftest() -> None:
    # 테이프 한 날: 0.1 칸 → $1 칸 · 첫/끝 초 VWAP · 공격자 합
    sec = np.array([10, 10, 20, 30]); pb = np.array([20000, 20009, 20015, 19995])   # $2000.0 · 2000.9 · 2001.5 · 1999.5
    bq = np.array([1.0, 0.0, 2.0, 0.0]); sq = np.array([0.0, 1.0, 0.0, 3.0])
    s, (b, bu, se) = day_from_tape(sec, pb, bq, sq)
    assert abs(s["open"] - 2000.45) < 1e-9 and abs(s["close"] - 1999.5) < 1e-9 and s["high"] == 2001.5 and s["low"] == 1999.5
    assert list(b) == [1999, 2000, 2001] and list(bu) == [0, 1, 2] and list(se) == [3, 1, 0]
    assert abs(s["buy_quote"] - (2000.0 + 2 * 2001.5)) < 1e-6
    _, (b2, _, _) = day_from_tape(np.array([1]), np.array([20010]), np.array([1.0]), np.array([0.0]))
    assert list(b2) == [2001]                          # 2001.0 은 2001 칸
    lo, bb, ss = rebin(np.array([1999, 2000, 2004, 2006]), np.array([1, 2, 3, 4.0]), np.array([0, 1, 0, 1.0]), 5)
    assert lo == 1995 and bb == [1, 5, 4] and ss == [0, 1, 1], (lo, bb, ss)
    hist = pd.DataFrame([{"day": "2026-09-19", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "buy_qty": 1.0,
                          "sell_qty": 1.0, "buy_quote": 1.0, "sell_quote": 1.0, "oi": 10.0}])
    one = {"open": 1.0, "high": 2.0, "low": 1.0, "close": 2.0, "buy_qty": 1.0, "sell_qty": 1.0, "buy_quote": 1.0, "sell_quote": 1.0}
    lf = int(datetime(2026, 9, 20, 13, tzinfo=timezone.utc).timestamp() * 1000)
    c = build_days(hist, {"2026-09-20": one, "2026-09-21": one}, {"2026-09-21": 20.0}, {"2026-09-20": [5.0, 1.0]}, lf, "2026-09-21")
    assert c["d"] == ["2026-09-19", "2026-09-20", "2026-09-21"] and c["ll"] == [None, 5.0, 0.0] and c["part"] == [0, 2, 1]
    assert c["oi"] == [10.0, None, 20.0] and c["src"] == ["a", "t", "t"]
    assert parse_row("abc") == 5 and parse_row("0") == 1 and parse_row("9999") == 500
    print("selftest OK -- 테이프 일 요약·$1 칸·행 묶기·청산 모름/반쪽/0·OI 병합·행 입력 검증")


if __name__ == "__main__":
    selftest()
