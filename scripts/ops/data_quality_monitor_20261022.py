"""수집기 데이터 품질 감시 (2026-10-22, 사용자 «빈 구간·거래소 어긋남·전진 검정이 쓰는 데이터 커버리지 자동 감시»).

ops_watchdog(30초) = «지금 살아 있나»(마지막 행 나이). 이 감시(매시) = 그 사이에 안 보이던 세 가지:
  ① 빈 구간: 지난 24시간 분 단위로 채워졌나 · 체결 테이프는 수집기 자체 대조(verify_1m: 테이프 합 vs 거래소 1분봉) 불일치·미검사 분
  ② 거래소 어긋남: 같은 분 마크가 바이낸스 대비 bp 차 · 거래량 몫(OKX·Bybit / 바이낸스)이 지난 날들보다 크게 변했나 · 청산 0건 거래소
  ③ 전진 검정 커버리지: 판정 달력(dashboard/verdict_calendar.json) 항목이 읽는 입력이 시작일부터 날마다 찼나 — 구멍 난 날은 판정 전에 알아야 한다
읽기 전용: hot SQLite 는 WAL 이라 그대로 읽는다(seal_to_lake 와 같은 방식). 🔴라이브 DuckDB 는 열지 않는다(읽기 전용이어도 쓰는 쪽이
  lock 으로 죽는다, seal_to_lake 도크스트링) — 데리빗 체인은 lake 봉인본(끝난 날)만 센다.
출력: data/live/ops_watchdog/data_quality.json(최근 판정) · data_quality_daily.jsonl(끝난 UTC 날마다 한 줄 = 전진 커버리지 장부, 한 번 쓰면 안 고친다)
  · 상태가 바뀐 항목만 텔레그램(ops_watchdog.send_telegram) · 아침 감사관이 data_quality.json 을 읽는다.
cron(서버 KST): 17 * * * * cd ~/crypto-scalping && nice -n 10 <python> scripts/ops/data_quality_monitor_20261022.py >> logs/data_quality.log 2>&1
  --dry-run 보내지 않고 출력만 · --selftest 합성 DB 로 판정 함수 점검
"""
from __future__ import annotations

import glob
import json
import sqlite3
import statistics
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts")]
import ops_watchdog as W  # noqa: E402  (.env · send_telegram)

HOT, LAKE = ROOT / "data" / "hot", ROOT / "data" / "lake"
BT = ROOT / "data" / "live" / "orderflow" / "bookticker"
OUT = ROOT / "data" / "live" / "ops_watchdog"
REPORT, DAILY, STATE = OUT / "data_quality.json", OUT / "data_quality_daily.jsonl", OUT / "data_quality_state.json"
COINS = ("ETH", "BTC", "SOL", "XRP")
TOL = 1e-4                                   # 수집기 VERIFY_TOLERANCE 와 같다(테이프 합 vs 1분봉 상대오차)
LEDGER_FROM = "2026-10-01"                   # 장부 첫 날 -- 판정 달력에서 수집기에 기대는 가장 이른 시작(gex_rev)

# 체결 테이프: 거래소 → (DB, 코인 → 심볼). 심볼 표기는 수집기마다 다르다.
TAPES = {"binance": ("binance_tape.sqlite", lambda c: f"{c.lower()}usdt"), "okx": ("okx_tape.sqlite", lambda c: f"{c}-USDT-SWAP"),
         "bybit": ("bybit_tape.sqlite", lambda c: f"{c}USDT"), "binance_spot": ("binance_spot_tape.sqlite", lambda c: f"{c.lower()}usdt")}
# 맥락(1초·수초 간격): 이름 → (DB, 표, 시각 열(ms), 키 열, 코인 → 키)
CTX = {"mark:binance": ("binance_ctx.sqlite", "mark_price_1s", "ts_ms", "symbol", lambda c: f"{c.lower()}usdt"),
       "oi:binance": ("binance_ctx.sqlite", "oi_1s", "ts_ms", "symbol", lambda c: f"{c.lower()}usdt"),
       "mark:okx": ("okx_ctx.sqlite", "okx_mark", "ts_ms", "inst", lambda c: f"{c}-USDT-SWAP"),
       "mark:bybit": ("bybit_ctx.sqlite", "bybit_mark", "ts_ms", "inst", lambda c: f"{c}USDT"),
       "oi:bybit": ("bybit_ctx.sqlite", "bybit_oi", "ts_ms", "inst", lambda c: f"{c}USDT"),
       "ctx:hl": ("hl_ctx.sqlite", "hl_asset_ctx", "recv_ms", "coin", lambda c: c)}
LIQS = {"binance": ("binance_ctx.sqlite", "liquidations", "SELECT count(*) FROM liquidations WHERE symbol=? AND ts_ms>=? AND ts_ms<?", lambda c: f"{c}USDT"),
        "okx": ("okx_ctx.sqlite", "okx_liquidations", "SELECT count(*) FROM okx_liquidations WHERE inst_id=? AND ts_ms>=? AND ts_ms<?", lambda c: f"{c}-USDT-SWAP"),
        "bybit": ("bybit_liq.sqlite", "bybit_liquidations", "SELECT count(*) FROM bybit_liquidations WHERE symbol=? AND ts_ms>=? AND ts_ms<?", lambda c: f"{c}USDT")}
# lake 봉인본(hot 보관 기간 밖의 날): 입력 → (lake 하위 경로, coin= 분할 값 또는 None)
LAKE_OF = {"tape:binance:ETH": ("binance/tape", "ETH"), "liq:binance": ("binance/liquidations", None), "liq:okx": ("okx/liquidations", None),
           "liq:bybit": ("bybit/liquidations", None), "mark:binance:ETH": ("binance/mark_1s", None), "deribit_chain": ("deribit/chain", None)}
# 판정 달력 항목 → 시작일·입력. 바이낸스 vision(공개 아카이브)만 읽는 항목은 수집기와 무관해서 뺀다(EXTERNAL).
FORWARD = [
    {"ids": ["ob_mid", "f0b", "h4"], "start": "2026-10-04", "inputs": ["tape:binance:ETH", "bt:ETH"]},
    {"ids": ["h3"], "start": "2026-10-04", "inputs": ["bt:ETH", "bt:BTC", "bt:SOL", "bt:XRP", "bt:HYPE"]},
    {"ids": ["liqmap_fwd"], "start": "2026-10-08", "inputs": ["liqmap_score", "liq:binance", "liq:okx", "liq:bybit", "mark:binance:ETH"]},
    {"ids": ["gex_rev", "opt_info", "maxpain_oos"], "start": "2026-10-01", "inputs": ["deribit_chain"]},
    {"ids": ["dir4h_p2", "dir3asset"], "start": "2026-10-12", "inputs": ["dir_rec"]},   # 기록기 첫 줄 10-12(판정 자체는 klines 재계산)
]
EXTERNAL = ["exh1", "pp_volsize1", "oi_adv_exit1", "trend4_1", "whale_duel1",   # data.binance.vision 만 읽는다
            "soxl_wknd1", "soxl_wknd2"]                                       # 기록기가 fapi REST 를 직접 읽는다(수집기 무관)


def utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def day_bounds(day: str) -> tuple[int, int]:
    t0 = int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    return t0, t0 + 86400


def q(db: str, sql: str, args=()) -> list:
    with sqlite3.connect(f"file:{HOT / db}?mode=ro", uri=True, timeout=20) as con:
        return con.execute(sql, args).fetchall()


# ── 원시 측정(구간 [t0, t1) 초) ──────────────────────────────────────────────────────────────
def minute_stats(minutes: list[int], t0: int, t1: int) -> dict:
    """분 번호(초/60) → 채움 비율 · 가장 긴 빈 구간(분). 구간 양 끝 바깥도 빈 것으로 센다."""
    n = (t1 - t0) // 60
    if not n:
        return {"cov": None, "gap_min": None}
    edges = [t0 // 60 - 1] + sorted(set(minutes)) + [t1 // 60]
    return {"cov": round(len(set(minutes)) / n, 4), "gap_min": max(b - a - 1 for a, b in zip(edges, edges[1:]))}


def tape_stats(db: str, sym: str, t0: int, t1: int) -> dict:
    mins = [r[0] for r in q(db, "SELECT DISTINCT ts_sec/60 FROM trade_tape_1s WHERE symbol=? AND ts_sec>=? AND ts_sec<?", (sym, t0, t1))]
    out = minute_stats(mins, t0, t1)
    if not mins:
        return {**out, "qty": 0.0, "bad": 0, "unver": 0}
    qty = tape_qty(db, sym, t0, t1)
    old = t1 - 3600                                                     # 수집기는 끝난 지 2분 넘은 분을 5분마다 검사 -- 1시간이면 다 돌았어야 한다
    bad = q(db, "SELECT count(*) FROM verify_1m WHERE symbol=? AND ts_min>=? AND ts_min<? AND abs(rel_err)>?", (sym, t0, t1, TOL))[0][0]
    ver_old = q(db, "SELECT count(*) FROM verify_1m WHERE symbol=? AND ts_min>=? AND ts_min<?", (sym, t0, old))[0][0]
    return {**out, "qty": round(qty, 3), "bad": int(bad), "unver": max(0, sum(m * 60 < old for m in mins) - ver_old)}


def tape_qty(db: str, sym: str, t0: int, t1: int) -> float:
    return q(db, "SELECT sum(buy_qty+sell_qty) FROM trade_tape_1s WHERE symbol=? AND ts_sec>=? AND ts_sec<?", (sym, t0, t1))[0][0] or 0.0


def ctx_stats(db: str, table: str, tcol: str, kcol: str, key: str, t0: int, t1: int) -> dict:
    mins = [r[0] for r in q(db, f"SELECT DISTINCT {tcol}/60000 FROM {table} WHERE {kcol}=? AND {tcol}>=? AND {tcol}<?", (key, t0 * 1000, t1 * 1000))]
    return minute_stats(mins, t0, t1)


def minute_marks(db, table, tcol, kcol, key, t0, t1, col) -> dict[int, float]:
    return dict(q(db, f"SELECT {tcol}/60000, avg({col}) FROM {table} WHERE {kcol}=? AND {tcol}>=? AND {tcol}<? GROUP BY 1",
                  (key, t0 * 1000, t1 * 1000)))


def price_gap(base: dict[int, float], other: dict[int, float]) -> dict:
    """같은 분 평균 마크의 차(bp) — 중앙 |차| · 99분위 |차| · 맞춘 분 수."""
    d = sorted(abs(other[m] / base[m] - 1) * 1e4 for m in base.keys() & other.keys() if base[m] > 0)
    if not d:
        return {"n": 0, "med_bp": None, "p99_bp": None}
    return {"n": len(d), "med_bp": round(statistics.median(d), 2), "p99_bp": round(d[min(len(d) - 1, int(0.99 * len(d)))], 2)}


def lake_rows(sub: str, coin: str | None, day: str) -> int | None:
    """lake 봉인본 그 날 행 수(파케이 메타데이터만) — 파일이 없으면 None."""
    import pyarrow.parquet as pq
    fs = [f for f in glob.glob(str(LAKE / sub / "**" / f"date={day}" / "*.parquet"), recursive=True)
          if coin is None or f"coin={coin}" in f or "coin=" not in f]
    return sum(pq.ParquetFile(f).metadata.num_rows for f in fs) if fs else None


def hot_floor(db: str, table: str, tcol: str, scale: int) -> int | None:
    r = q(db, f"SELECT min({tcol}) FROM {table}")[0][0]
    return None if r is None else int(r) // scale


# ── 끝난 날 장부(전진 커버리지) ──────────────────────────────────────────────────────────────
_LM: dict = {}


def _lake_medians() -> dict:
    """lake 입력별 최근 14개 날 행 수 중앙값 -- 그 절반 미만이면 덜 찬 날."""
    if not _LM:
        for k, (sub, coin) in LAKE_OF.items():
            days = sorted({Path(f).parent.name[5:] for f in glob.glob(str(LAKE / sub / "**" / "date=*" / "*.parquet"), recursive=True)})[-14:]
            rows = [r for d in days if (r := lake_rows(sub, coin, d))]
            _LM[k] = statistics.median(rows) if rows else None
    return _LM


def day_entry(day: str) -> dict:
    """끝난 UTC 날 하나 → 입력별 {ok, 근거}. hot 에 그 날 전체가 남아 있으면 분 단위, 아니면 lake 행 수."""
    t0, t1 = day_bounds(day)
    e: dict = {}
    tf = hot_floor("binance_tape.sqlite", "trade_tape_1s", "ts_sec", 1)
    if tf is not None and tf <= t0:
        s = tape_stats("binance_tape.sqlite", "ethusdt", t0, t1)
        e["tape:binance:ETH"] = {"src": "hot", **s, "ok": s["cov"] >= 0.99 and s["bad"] <= 3}
    mf = hot_floor("binance_ctx.sqlite", "mark_price_1s", "ts_ms", 1000)
    if mf is not None and mf <= t0:
        s = ctx_stats(*CTX["mark:binance"][:4], "ethusdt", t0, t1)
        e["mark:binance:ETH"] = {"src": "hot", **s, "ok": s["cov"] >= 0.99}
    for v, (db, table, sql, sym) in LIQS.items():
        floor = hot_floor(db, table, "ts_ms", 1000)
        if floor is not None and floor <= t0:
            n = q(db, sql, (sym("ETH"), t0 * 1000, t1 * 1000))[0][0]
            e[f"liq:{v}"] = {"src": "hot", "n": n, "ok": n > 0}
    lake_med = _lake_medians()
    for k, (sub, coin) in LAKE_OF.items():
        if k not in e:
            n = lake_rows(sub, coin, day)
            if k.startswith("liq:"):                                    # 청산은 사건 -- 조용한 날은 적다(10-10 바이낸스 711건). 있기만 하면 된다
                ok = bool(n)
            elif k == "mark:binance:ETH":                               # 분할에 코인이 섞여 있다(10-03 까지 ETH 만 86,400행 → 뒤엔 3코인) -- 코인 하나 하루치 이상
                ok = bool(n) and n >= 0.98 * 86400
            else:
                ok = bool(n) and (not lake_med.get(k) or n >= 0.5 * lake_med[k])
            e[k] = {"src": "lake", "rows": n, "ok": ok}
    for c in ("ETH", "BTC", "SOL", "XRP", "HYPE"):
        n = len(glob.glob(str(BT / f"{c}USDT" / f"{day}T*.bt*")))
        e[f"bt:{c}"] = {"src": "file", "hours": n, "ok": n >= 23}
    p = ROOT / "data" / "live" / "liqmap_score.jsonl"
    n = sum(1 for ln in p.open() if f'"origin": "{day}T' in ln) if p.exists() else 0
    e["liqmap_score"] = {"src": "file", "hours": n, "ok": n >= 22}
    p = ROOT / "data" / "live" / "eth_direction_forward.jsonl"
    syms = {json.loads(ln)["sym"] for ln in p.open() if f'"date": "{day}"' in ln} if p.exists() else set()
    e["dir_rec"] = {"src": "file", "syms": sorted(syms), "ok": len(syms) >= 3}
    return e


def update_ledger(today: str) -> dict[str, dict]:
    have = {}
    if DAILY.exists():
        for ln in DAILY.open():
            r = json.loads(ln); have[r["day"]] = r["inputs"]
    d = datetime.strptime(LEDGER_FROM, "%Y-%m-%d")
    while (day := d.strftime("%Y-%m-%d")) < today:
        if day not in have:
            have[day] = day_entry(day)
            with DAILY.open("a") as f:
                f.write(json.dumps({"day": day, "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "inputs": have[day]}, ensure_ascii=False) + "\n")
        d += timedelta(days=1)
    return have


def forward_coverage(ledger: dict[str, dict], today: str) -> list[dict]:
    out = []
    for f in FORWARD:
        days = [d for d in sorted(ledger) if f["start"] <= d < today]
        holes = {d: [k for k in f["inputs"] if not ledger[d].get(k, {}).get("ok")] for d in days}
        holes = {d: v for d, v in holes.items() if v}
        st = "OK" if not holes else ("CRIT" if len(holes) > max(1, len(days) // 10) else "WARN")
        first = next(iter(holes), None)
        out.append({"key": "forward:" + "+".join(f["ids"]), "status": st, "days": len(days), "hole_days": holes,
                    "summary": f"{len(days)}일 중 구멍 {len(holes)}일" + (f" (예: {first} {holes[first]})" if first else "")})
    return out


# ── 등급(자체점검 대상) ──────────────────────────────────────────────────────────────────────
def grade_tape(s: dict) -> str:
    n = max(1, round((s["cov"] or 0) * 1440))
    if s["cov"] is None or s["cov"] < 0.95 or s["bad"] / n > 0.02:
        return "CRIT"
    if s["cov"] < 0.995 or s["bad"] / n > 0.002 or s["unver"] / n > 0.01:
        return "WARN"
    return "OK"


def grade_ctx(s: dict, base: float | None) -> str:
    b = 0.99 if base is None else min(base, 0.99)
    if s["cov"] is None or s["cov"] < b - 0.10 or (s["gap_min"] or 0) >= 30:
        return "CRIT"
    return "WARN" if s["cov"] < b - 0.02 else "OK"


def grade_price(g: dict) -> str:
    if not g["n"]:
        return "WARN"
    return "CRIT" if g["med_bp"] > 10 else ("WARN" if g["med_bp"] > 3 or g["p99_bp"] > 25 else "OK")


def grade_share(now: float | None, hist: list[float]) -> str:
    if now is None or len(hist) < 3:
        return "OK"
    m = statistics.median(hist)
    r = now / m if m > 0 else None
    return "OK" if r is None or 0.5 <= r <= 2.0 else ("CRIT" if r < 0.25 or r > 4 else "WARN")


# ── 지난 24시간 판정 ─────────────────────────────────────────────────────────────────────────
def run(now: float | None = None) -> dict:
    now = time.time() if now is None else now
    t1 = int(now) // 60 * 60 - 600; t0 = t1 - 86400                 # 끝난 지 10분 넘은 분까지의 24시간
    checks: list[dict] = []
    add = lambda key, status, summary, **d: checks.append({"key": key, "status": status, "summary": summary, **d})  # noqa: E731
    hist = [(t0 - 86400 * k, t1 - 86400 * k) for k in range(1, 7)]     # 지난 6개 «24시간»(같은 시각 기준)
    qty = {}
    for v, (db, sym) in TAPES.items():
        for c in COINS:
            try:
                s = tape_stats(db, sym(c), t0, t1)
            except sqlite3.Error as exc:
                add(f"tape:{v}:{c}", "WARN", f"읽기 실패 {exc}"); continue
            if s["cov"] == 0 and v != "binance":                         # 그 거래소가 원래 안 받는 코인
                continue
            qty[(v, c)] = s["qty"]
            add(f"tape:{v}:{c}", grade_tape(s), f"채움 {s['cov']:.1%} · 최장 공백 {s['gap_min']}분 · 1분봉 불일치 {s['bad']} · 미검사 {s['unver']}", **s)
    for name, (db, table, tcol, kcol, key) in CTX.items():
        for c in COINS:
            try:
                s = ctx_stats(db, table, tcol, kcol, key(c), t0, t1)
                base = [x for a, b in hist[:3] if (x := ctx_stats(db, table, tcol, kcol, key(c), a, b)["cov"])]
                if s["cov"] == 0 and not base:                          # 원래 안 받는 코인(바이낸스 맥락 BTC 등) -- 받다가 끊기면 base 가 있어 CRIT
                    continue
            except sqlite3.Error as exc:
                add(f"{name}:{c}", "WARN", f"읽기 실패 {exc}"); continue
            b = statistics.median(base) if base else None
            add(f"{name}:{c}", grade_ctx(s, b), f"채움 {s['cov']:.1%}" + (f"(평소 {b:.1%})" if b else "") + f" · 최장 공백 {s['gap_min']}분", base=b, **s)
    for c in COINS:                                                     # 거래소 어긋남 -- 같은 분 마크가
        try:
            bn = minute_marks(*CTX["mark:binance"][:4], CTX["mark:binance"][4](c), t0, t1, "mark")
            if not bn:                                                  # 바이낸스 마크가 없는 코인은 기준이 없다(끊긴 건 위 mark: 항목이 잡는다)
                continue
            for name in ("mark:okx", "mark:bybit", "ctx:hl"):
                db, table, tcol, kcol, key = CTX[name]
                o = minute_marks(db, table, tcol, kcol, key(c), t0, t1, "mark_px")
                if o:
                    g = price_gap(bn, o)
                    add(f"price:{name.split(':')[1]}:{c}", grade_price(g), f"바이낸스 대비 마크 차 중앙 {g['med_bp']}bp · 99분위 {g['p99_bp']}bp ({g['n']}분)", **g)
        except sqlite3.Error as exc:
            add(f"price:{c}", "WARN", f"읽기 실패 {exc}")
    for c in COINS:                                                     # 거래량 몫(거래소 / 바이낸스) -- 지난 6일과 비교
        for v in ("okx", "bybit"):
            if (v, c) not in qty or not qty.get(("binance", c)):
                continue
            share, past = qty[(v, c)] / qty[("binance", c)], []
            for a, b in hist:
                try:
                    bq, vq = tape_qty(TAPES["binance"][0], TAPES["binance"][1](c), a, b), tape_qty(TAPES[v][0], TAPES[v][1](c), a, b)
                    if bq and vq:
                        past.append(vq / bq)
                except sqlite3.Error:
                    pass
            add(f"share:{v}:{c}", grade_share(share, past), f"거래량 몫 {share:.3f}" + (f" (지난 날 중앙 {statistics.median(past):.3f})" if past else ""),
                share=round(share, 4), hist=[round(h, 4) for h in past])
    try:                                                                # 청산 -- 바이낸스에 30건 이상인데 0건인 거래소
        n = {v: q(db, sql, (sym("ETH"), t0 * 1000, t1 * 1000))[0][0] for v, (db, _, sql, sym) in LIQS.items()}
        add("liq:ETH", "WARN" if any(k == 0 for k in n.values()) and n["binance"] >= 30 else "OK",
            "24h 청산 " + " · ".join(f"{v} {k}" for v, k in n.items()), counts=n)
    except sqlite3.Error as exc:
        add("liq:ETH", "WARN", f"읽기 실패 {exc}")
    today = utc_day(now)
    checks += forward_coverage(update_ledger(today), today)
    rank = {"OK": 0, "WARN": 1, "CRIT": 2}
    return {"generated_at_kst": W.iso_now(),
            "window_utc": [datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M") for t in (t0, t1)],
            "status": max((c["status"] for c in checks), key=rank.get, default="OK"),
            "not_ok": [{k: c[k] for k in ("key", "status", "summary")} for c in checks if c["status"] != "OK"],
            "checks": checks, "external_forward": EXTERNAL}


def transitions(prev: dict[str, str], checks: list[dict]) -> list[str]:
    """상태가 바뀐 항목만 문장으로(새로 나빠짐 · 회복). 처음 보는 OK 는 조용히."""
    msgs = []
    for c in checks:
        p = prev.get(c["key"], "OK")
        if c["status"] != p:
            msgs.append(f"🟢 회복 {c['key']} ({p} → OK)" if c["status"] == "OK" else
                        f"{'🔴' if c['status'] == 'CRIT' else '🟠'} {c['key']}: {c['summary']}")
    return msgs


def main() -> int:
    global OUT, REPORT, DAILY, STATE
    a = sys.argv[1:]
    if "--selftest" in a:
        return selftest()
    if "--dry-run" in a:                                                # 보내지 않고 · 운영 장부·상태도 안 건드린다(첫 실행 확인용)
        OUT = ROOT / "tmp" / f"data_quality_dryrun_{int(time.time())}"
        REPORT, DAILY, STATE = OUT / "data_quality.json", OUT / "data_quality_daily.jsonl", OUT / "data_quality_state.json"
    OUT.mkdir(parents=True, exist_ok=True)
    rep = run()
    tmp = REPORT.with_suffix(".tmp"); tmp.write_text(json.dumps(rep, ensure_ascii=False, indent=1)); tmp.replace(REPORT)
    prev = json.loads(STATE.read_text()) if STATE.exists() else {}
    msgs = transitions(prev, rep["checks"])
    print(f"[{rep['generated_at_kst']}] 데이터 품질 {rep['status']} · 이상 {len(rep['not_ok'])}건 · 바뀜 {len(msgs)}건", flush=True)
    for m in msgs:
        print("  " + m, flush=True)
    if "--dry-run" in a:
        return 0
    if msgs:
        body = "🧪 <b>데이터 품질</b>\n" + "\n".join(W.html.escape(m) for m in msgs[:25]) + ("\n…" if len(msgs) > 25 else "")
        if W.send_telegram(body) is False:
            return 1                                                    # 보내지 못했으면 상태를 안 바꾼다 -- 다음 시간에 다시
    STATE.write_text(json.dumps({c["key"]: c["status"] for c in rep["checks"]}, ensure_ascii=False))
    return 0


def selftest() -> int:
    global HOT
    assert minute_stats([10, 11, 12], 600, 900) == {"cov": 0.6, "gap_min": 2}             # 10~14분 중 13·14 빔
    assert minute_stats([], 600, 900) == {"cov": 0.0, "gap_min": 5}
    assert price_gap({1: 100.0, 2: 100.0}, {1: 100.1, 2: 100.0, 3: 5.0}) == {"n": 2, "med_bp": 5.0, "p99_bp": 10.0}
    assert grade_tape({"cov": 1.0, "bad": 0, "unver": 0}) == "OK" and grade_tape({"cov": 0.9, "bad": 0, "unver": 0}) == "CRIT"
    assert grade_tape({"cov": 1.0, "bad": 10, "unver": 0}) == "WARN" and grade_tape({"cov": 1.0, "bad": 0, "unver": 30}) == "WARN"
    assert grade_ctx({"cov": 0.40, "gap_min": 3}, 0.41) == "OK" and grade_ctx({"cov": 0.30, "gap_min": 3}, 0.41) == "CRIT"
    assert grade_ctx({"cov": 0.995, "gap_min": 45}, 1.0) == "CRIT"                         # 채움은 높아도 45분 공백
    assert grade_price({"n": 100, "med_bp": 0.8, "p99_bp": 6}) == "OK" and grade_price({"n": 100, "med_bp": 40, "p99_bp": 60}) == "CRIT"
    assert grade_share(0.30, [0.31, 0.29, 0.33]) == "OK" and grade_share(0.05, [0.31, 0.29, 0.33]) == "CRIT"
    assert transitions({"a": "OK", "b": "WARN"}, [{"key": "a", "status": "CRIT", "summary": "x"}, {"key": "b", "status": "OK", "summary": ""},
                                                 {"key": "c", "status": "OK", "summary": ""}]) == ["🔴 a: x", "🟢 회복 b (WARN → OK)"]
    led = {"2026-10-04": {"bt:ETH": {"ok": True}, "tape:binance:ETH": {"ok": False}}, "2026-10-05": {"bt:ETH": {"ok": True}, "tape:binance:ETH": {"ok": True}}}
    fc = forward_coverage(led, "2026-10-06")[0]
    assert fc["status"] == "WARN" and fc["hole_days"] == {"2026-10-04": ["tape:binance:ETH"]}, fc
    with tempfile.TemporaryDirectory() as d:                           # 합성 hot DB: 하루 중 10분 빔 + 1분봉 불일치 3분 → WARN
        HOT = Path(d)
        day0 = int(datetime(2026, 10, 20, tzinfo=timezone.utc).timestamp())
        with sqlite3.connect(HOT / "t.sqlite") as con:
            con.execute("CREATE TABLE trade_tape_1s(symbol, ts_sec, price_bin, buy_qty, sell_qty)")
            con.execute("CREATE TABLE verify_1m(symbol, ts_min, tape_qty, kline_qty, rel_err, checked_at)")
            mins = [m for m in range(1440) if not 600 <= m < 610]
            con.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?)", [("ethusdt", day0 + m * 60 + 5, 1, 1.0, 1.0) for m in mins])
            con.executemany("INSERT INTO verify_1m VALUES (?,?,0,0,?,0)", [("ethusdt", day0 + m * 60, 0.05 if m < 3 else 0.0) for m in mins])
        s = tape_stats("t.sqlite", "ethusdt", day0, day0 + 86400)
        assert s["cov"] == round(1430 / 1440, 4) and s["gap_min"] == 10 and s["bad"] == 3 and s["unver"] == 0 and s["qty"] == 2860.0, s
        assert grade_tape(s) == "WARN"
    print("selftest OK -- 분 채움·최장 공백 · 마크 차 · 등급 넷 · 상태 전환 문장 · 전진 구멍 날 · 합성 테이프(10분 공백·불일치 3분)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
