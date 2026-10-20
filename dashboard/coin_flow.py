"""코인별 흐름 엔진 -- 풋프린트·1초 수급·OI·OKX·현물 수집기 공장(make_coin_flow)과 그것이 쓰는 상수·읽기 함수.

2026-10-19 server.py 에서 내용 그대로 옮겼다(함수·클래스 본문 바이트 동일). server.py 가 이름을 전부 다시 내보낸다.
🔴이 모듈은 dashboard.server 를 import 하면 안 된다 -- 서버는 `python dashboard/server.py`(__main__)로 돌아서
  import 하면 server.py 가 한 번 더 실행된다(load_dotenv·스레드·FLOW_SPECS).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
import json
import os
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Any

import duckdb
from aiohttp import ClientSession, ClientTimeout, TCPConnector, WSMsgType, web

from trading_bot_modules.duckdb_access import duckdb_path_lock
from scripts.live_trade_tape_collector_20260916 import (
    BUCKETS as TAPE_BUCKETS,
    TakerOrderAggregator,
    HOT_TAPE_DB,
    size_bands,
)
from scripts.data_store import read_rows
from scripts.live_binance_ctx_collector_20261001 import HOT_DB as BINANCE_CTX_DB

# server.py 와 같은 값(두 파일 다 dashboard/ 아래).
REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO_ROOT / "data" / "live"

# 볼륨 풋프린트 (2026-09-15) -- ETH 만. 자세한 근거는 collect_footprint() 주석.
FOOTPRINT_SYMBOL = "ETHUSDT"
FOOTPRINT_BAR_SECONDS = 300      # 차트 캔들과 같은 5분봉
FOOTPRINT_BARS = 12              # 1시간
FOOTPRINT_BUCKET = 0.5           # 가격 버킷(달러). 화면 행 크기는 이것의 배수로 클라가 묶는다
# 백필 페이스. 요청당 가중치 20 이므로 2.5초 = ~480/분. 한도는 IP 당 2400/분인데 그 IP 를
# **트레이딩 봇과 대시보드의 다른 폴링이 같이 쓴다** -- 1.2초(~1000/분)로 돌렸더니 2026-09-15
# 실제로 429 가 났다. 차트 하나 빨리 채우자고 주문 경로를 위협할 이유가 없다(백필은 최신 봉부터
# 채우므로 느려도 «지금 보는 자리»는 1분 안에 찬다).
FOOTPRINT_CATCHUP_SECONDS = 2.5
# 받은 테이프를 디스크에 남긴다. 안 남기면 재시작마다 1시간을 REST 로 다시 사와야 하는데,
# 대시보드는 배포마다 재시작된다(2026-09-15 하루에도 여러 번). 남기면 재시작 뒤 메울 구간이
# «꺼져 있던 시간»으로 줄어든다 -- 보통 몇 초다.
FOOTPRINT_SNAPSHOT_PATH = LIVE_DIR / "footprint_eth.json"
FOOTPRINT_SNAPSHOT_SECONDS = 30.0
# 화면 토글이 고를 수 있는 **가장 긴 창**(4h). 저장·복원은 여기에 맞춘다 -- 12봉만 저장하면
# 재시작(배포마다 하루 여러 번) 직후 4h 를 골라도 1h 밖에 안 보인다. 실측 51KB/48봉이고
# 30초마다 덮어써도 하루 150MB 쓰기다(288봉이면 1GB 라 링 전체는 여전히 안 쓴다).
#
# ⭐**복원 경로는 이 스냅샷 하나다**(2026-09-19 사용자 결정). 체결 테이프 duckdb 에서 읽어
#   오는 길도 만들어 봤다 -- 셀 단위 오차 0.0 ETH 로 정확하긴 했는데, 실측 **250ms vs
#   0.8ms** 로 300배 느리고 20.7MB 파일을 여는 대가였다. 같은 4시간이 이 json 안에 이미
#   있으므로 두 벌을 둘 이유가 없다. duckdb 는 수집기가 계속 쌓는 연구용 아카이브로 남는다.
# 🔴백필(REST)도 **넓히지 않는다**. 1시간이 ~200요청인데 4시간이면 예산 400을 넘고, 그 IP 를
#   트레이딩 봇이 같이 쓴다. 스냅샷이 못 덮는 구간은 WS 가 돌면서 자연히 찬다.
# 🔴이 상수의 뜻은 «화면이 고를 수 있는 가장 긴 창»이다(아래 저장 주석). 2026-09-23
#   12h 토글이 생겨 정의상 144 가 됐다 -- 48 로 두면 재시작 뒤 12h 창이 4h 만 보인다
#   (실측: 오늘 배포 직후 144 봉 요청에 50 봉만 나왔다). 스냅샷 85KB -> ~255KB.
FOOTPRINT_MAX_WINDOW_BARS = 144

# ── 리테일/고래 수급 (2026-09-19) ───────────────────────────────────────────
# 셀은 [매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도] 6칸이다.
# **중형($10k~$100k)은 칸이 없다** -- 매수 - 고래매수 - 리테일매수로 정확히 나온다.
#
# ⭐경계는 체결 테이프 수집기에서 **가져다 쓴다**. 화면과 저장이 같은 숫자를 써야 나중에
#   「그날 화면의 고래」와 「DB 의 고래」가 같은 것이 된다. 여기 상수를 새로 만들면 두 개가
#   조용히 갈라진다 -- 이 저장소가 레짐 분류기에서 한 번 겪은 일이다(2026-09-17).
# ⭐왜 분위가 아니라 달러 고정인지, 왜 경계가 둘인지는 그 파일의 상수 주석에 실측과 함께 있다.
FOOTPRINT_KEEP_BARS = 288
# 최근 5분을 **1초 해상도**로 보는 화면용 링(2026-09-19 사용자 지시). 5분봉 리본을 걷어내고
# 이걸로 갈음했다 -- 「바로바로」가 뜻하는 건 5분봉이 아니라 초 단위였다.
# 가격빈을 버리고 초마다 한 칸만 남긴다(가격축은 수급 프로파일이 따로 본다). 360초를 두는 건
# 화면이 300초를 그리는데 경계에서 모자라지 않게 하려는 여유다.
# 🔴2026-09-20: 660 으로 올렸다가 다시 내렸다. 660 은 «창 왼쪽이 이전 봉에 걸친다»는
# 시안 G 전제였는데, 채택된 시안 H 는 **현재 5분봉 하나**만 그린다(최악 now-300).
# 두 봉치(600)에 여유를 더해 두면 봉이 바뀌는 순간과 폴링 지각을 모두 덮는다.
SUPPLY_1S_SECONDS = 660
# 같은 5분 창의 **미결제약정(OI)**. 「신규 계약이 몇 개 생겼나」는 OI 의 증분이다.
# 🔴바이낸스는 OI 를 매 초 갱신하지 않는다 -- /fapi/v1/openInterest 를 1초 간격 20회 때려보니
#   서로 다른 스냅샷은 6개, 간격 0.6~6.7초(중앙 ~3.5초)였다(2026-09-19 실측). 그래서 이 링은
#   **값이 바뀐 초에만** 점을 남긴다. 매초 같은 값을 복붙하면 없는 해상도를 있는 척하게 된다.
# 화면 링(6분)은 재기동하면 비고, 5분 누적 패널은 몇 시간을 봐야 한다 -- 그래서 남긴다.
# 🔴체결 테이프 duckdb(TAPE_DB_PATH)에 끼워 넣지 않는다: 저쪽 writer 는 별도 프로세스가
#   **연결을 붙들고** 있어 외부에서는 read_only 조차 거부된다(2026-09-19 실측). 여기서는
#   대시보드가 유일한 writer 이고, 매 flush 마다 연결-작업-닫기라 연구/감시 쪽 읽기를 막지 않는다.
OI_1S_DB_PATH = BINANCE_CTX_DB      # 4d: 수집기 hot(SQLite). 옛 data/live/oi_1s.duckdb 는 10-01 에 얼었다
OI_1S_TABLE = "oi_1s"
OI_5M_BAR_SECONDS = 300

# ⑦⑧ OKX 실시간(2026-09-23) -- 바이낸스 레인 **바로 아래**에 같은 그림을 그려 눈으로 대조한다.
#    합산은 하지 않는다: 합산 크기는 MM 헤지 이중계상으로 |합산|/|바이낸스| 중앙 2.1배로 부푼다
#    (3일 백필 실측). 나란히 두면 ctVal 같은 체계적 버그가 «한 레인만 10배»로 즉시 보인다.
OKX_INST = os.getenv("DASHBOARD_OKX_INST", "ETH-USDT-SWAP").upper()
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
# 🔴계약 크기(1계약 = ctVal 기초자산). 안 곱하면 이 레인이 10배로 그려진다. 표는 수집기와 **하나**
#   (import) -- 종목을 env 로 바꿔도 맞는 값을 쓰고, 모르는 종목이면 OKX 레인을 켜지 않는다.
#   시작할 때 거래소 REST 와 대조해 갈라졌으면 역시 켜지 않는다(collect_okx_flow).
from scripts.live_okx_trade_tape_collector_20260923 import (  # noqa: E402
    BUCKETS as OKX_TAPE_BUCKETS, CT_VALS as OKX_CT_VALS, HTTP_HEADERS as OKX_HTTP_HEADERS,
    INSTRUMENTS_URL as OKX_INSTRUMENTS_URL, OKX_HOT_CTX_DB, OKX_HOT_TAPE_DB)
OKX_TAPE_DB_PATH = OKX_HOT_TAPE_DB     # 2026-10-01 hot SQLite(5종목 한 파일, 락 없음) -- 쿼리가 종목으로 거른다
OKX_CTX_DB_PATH = OKX_HOT_CTX_DB      # 2026-10-01 4b hot SQLite(종목 = inst 열, 락 없음)
# OKX 청산은 ETH 맥락 수집기 하나가 전 종목(instType=SWAP)을 받아 여기에 쌓는다 -- 코인별 맥락 DB 에는 0건(09-27 실측).
OKX_LIQ_DB_PATH = OKX_CTX_DB_PATH


def _read_only_rows(path: Path, sql: str, params: list) -> list[tuple]:
    """수집기 저장소를 잠깐 열어 읽는다 -- .sqlite(hot)면 락 없이, .duckdb 면 read_only + 짧은 재시도(data_store.read_rows)."""
    return read_rows(path, sql, params)


def okx_tape_footprint(tape_db: Path, ctx_db: Path, inst: str, lo_sec: int, t0_ms: int,
                       bar_seconds: int, bucket: float, tape_bucket: float | None = None,
                       liq_db: Path | None = None) -> dict:
    """OKX 체결 테이프(수집기 duckdb)에서 풋프린트 봉을 되살린다 (2026-09-24, 사용자 A안).

    왜: 대시보드가 재기동하면 OKX 풋프린트 과거분이 사라지고(REST history-trades 백필은 10.8초치
      뿐), 그래서 «OKX 가 온전히 덮은 봉부터만» 내보냈다(C안) -- 재기동마다 화면이 몇 분치로
      줄었다. 그런데 OKX 테이프 수집기가 **같은 WS `trades` 를 1초×$0.1 칸으로 빠짐없이** 쌓고
      있고(1분봉 대조 + 자동 복구), 칸의 뜻도 라이브와 같다(메시지 = 주문 단위로 고래/리테일).
    돌려주는 것 -- 읽기만 한다(메모리 반영은 이벤트 루프가 한다):
      bars:   {봉: {가격칸: [매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도]}}  (ts < t0 초만)
      ok_min: 온전한 분 집합 -- 1분봉과 맞은 분(자동 복구분 포함) 또는 «아직 검사 전인 최근 15분 중
              기록된 공백과 안 겹치는 분»(검사가 2~5분 늦게 돈다).
      oi:     {봉: (첫 OI, 끝 OI)} -- okx_context 의 WS OI(ETH). ts < t0 만.
      ctx_ok: OI·청산을 다 읽었나(못 읽으면 비우고 간다 -- 1주일 풋프린트가 그 날을 굳히지 않게, 리뷰 2026-10-20).
    🔴가격칸: 테이프 칸(0.1)을 풋프린트 칸(0.5)으로 올린다. 0.5 가 0.1 의 정수배라 경계가 겹친다
      (정수/5 는 .5 가 안 나와 반올림 동률이 없다)."""
    t0_sec = t0_ms // 1000
    q = """SELECT ts_sec - ts_sec % ? AS bar, price_bin,
                  sum(buy_qty), sum(sell_qty),
                  sum(coalesce(whale_buy_qty, 0)), sum(coalesce(whale_sell_qty, 0)),
                  sum(coalesce(retail_buy_qty, 0)), sum(coalesce(retail_sell_qty, 0))
           FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? GROUP BY 1, 2"""
    # 2026-09-26 바이낸스 체결 테이프(trade_tape_<coin>.duckdb)도 같은 표 모양이라 이 함수를 쓴다 -- 칸 폭만 준다.
    tape_bucket = tape_bucket or OKX_TAPE_BUCKETS[inst]
    bars: dict[int, dict[int, list[float]]] = {}
    # 🔴2026-10-20 칸 키는 라이브와 **같은 식**(int(round(가격 / 칸)), 파이썬 반올림)으로 파이썬에서 -- SQL round(테이프칸×0.01/0.02)는
    #   SOL 홀수 센트가 .5 동률이라 SQLite(0 에서 먼 쪽)와 라이브 float 가 갈랐다(서버 사본 140봉 대조: SOL 0봉 일치 → 140봉 중 테이프 공백 외 전부).
    #   round(·, 10) = 테이프 칸 가격의 십진값 그대로(라이브의 float(체결가)와 같은 수) -- ETH·XRP 는 원래도 동률이 없어 결과가 같다.
    key: dict[int, int] = {}
    # 🔴SQL 은 DuckDB(OKX 테이프)·SQLite(바이낸스 hot) 둘 다에서 같게 -- `//` 는 SQLite 에 없다(data_store.read_rows).
    for bar, pb, *v in _read_only_rows(tape_db, q, [bar_seconds, inst, lo_sec, t0_sec]):
        k = key.get(pb)
        if k is None:
            k = key[pb] = int(round(round(int(pb) * tape_bucket, 10) / bucket))
        t = bars.setdefault(int(bar), {}).setdefault(k, [0.0] * 6)
        for j in range(6):
            t[j] += float(v[j] or 0.0)
    ver = _read_only_rows(tape_db, "SELECT ts_min, rel_err FROM verify_1m WHERE symbol = ? AND ts_min >= ?",
                          [inst, lo_sec])
    gaps = _read_only_rows(tape_db, "SELECT from_ms, to_ms FROM gaps WHERE symbol = ? AND to_ms > ?",
                           [inst, lo_sec * 1000])
    good = {int(m) for m, r in ver if abs(r) <= 1e-4}
    seen = {int(m) for m, _ in ver}
    now = int(time.time())
    ok_min = set(good)
    for m in range(lo_sec, t0_sec + 60, 60):
        if m not in seen and m >= now - 900 and not any(a < (m + 60) * 1000 and b > m * 1000
                                                         for a, b in gaps):
            ok_min.add(m)
    oi: dict[int, tuple[float, float]] = {}
    ctx_ok = True
    try:
        # 봉별 첫·끝 OI 는 파이썬에서 -- arg_min/arg_max·`//` 는 SQLite(hot, 4b)에 없다. 12h 창 ~수천 행.
        for ts, v in _read_only_rows(ctx_db, "SELECT ts_ms, oi_base FROM okx_oi WHERE inst = ? AND ts_ms >= ? "
                                              "AND ts_ms < ? ORDER BY ts_ms", [inst, lo_sec * 1000, t0_ms]):
            bar = int(ts) // 1000 // bar_seconds * bar_seconds
            oi[bar] = (oi[bar][0], float(v)) if bar in oi else (float(v), float(v))
    except (duckdb.Error, sqlite3.Error):
        ctx_ok = False            # OI 가 없어도 풋프린트는 되살린다(사분면 OI 만 바이낸스로 남는다)
    # 청산도 같은 구간을 되살린다 -- 봉에 «바이낸스+OKX» 가 붙는데 OKX 청산이 0 이면 합산 표시가
    #   거짓이 된다(피어 지적, 2026-09-24). 모양은 라이브 okx_liq_events 원소와 같다.
    liq: list[dict] = []
    try:
        for ts, pos, qty, px in _read_only_rows(liq_db or ctx_db, """
                SELECT ts_ms, pos_side, sz_base, bk_px FROM okx_liquidations
                WHERE inst_id = ? AND ts_ms >= ? AND ts_ms < ? AND sz_base > 0 ORDER BY ts_ms""",
                [inst, lo_sec * 1000, t0_ms]):
            liq.append({"ts_ms": int(ts), "side": "long" if pos == "long" else "short",
                        "qty": float(qty), "price": float(px or 0.0),
                        "usd": float(qty) * float(px or 0.0), "symbol": inst})
    except (duckdb.Error, sqlite3.Error):
        ctx_ok = False
    tape_max = _read_only_rows(tape_db, "SELECT max(ts_sec) FROM trade_tape_1s WHERE symbol = ?", [inst])
    return {"bars": bars, "ok_min": ok_min, "oi": oi, "liq": liq, "tape_max": int(tape_max[0][0] or 0), "ctx_ok": ctx_ok}
def tape_seconds(tape_db: Path, symbol: str, lo_sec: int, hi_sec: int,
                 tape_bucket: float) -> tuple[dict[int, list[float]], int]:
    """체결 테이프 -> 1초 수급 칸 {초: [리테일매수, 리테일매도, 고래매수, 고래매도, 총매수, 총매도, 가격]} (2026-09-27 재시작 복원).
    고래/리테일 경계는 테이프 수집기와 같은 상수(주문 단위)라 라이브 칸과 뜻이 같다. [lo, hi) 초, 테이프 끝 초도 준다.
    ponytail: 가격은 그 초의 «마지막 체결가»가 아니라 칸 가중평균(테이프에 순서가 없다) -- 오차 < 칸 1개(ETH $0.1)."""
    rows = _read_only_rows(tape_db, """
        SELECT ts_sec, sum(coalesce(retail_buy_qty, 0)), sum(coalesce(retail_sell_qty, 0)),
               sum(coalesce(whale_buy_qty, 0)), sum(coalesce(whale_sell_qty, 0)), sum(buy_qty), sum(sell_qty),
               sum(price_bin * (buy_qty + sell_qty)) / nullif(sum(buy_qty + sell_qty), 0) * ?
        FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? GROUP BY 1""",
        [tape_bucket, symbol, lo_sec, hi_sec])
    top = _read_only_rows(tape_db, "SELECT max(ts_sec) FROM trade_tape_1s WHERE symbol = ?", [symbol])
    return {int(r[0]): [float(x or 0.0) for x in r[1:]] for r in rows}, int(top[0][0] or 0)


OKX_CT_VAL = OKX_CT_VALS.get(OKX_INST)
# ⭐OI 는 **WS `open-interest`** 로 받는다(2026-09-23 정정). REST 0.25초 폴링이 «2.7배 더 본» 변화는
#   60%가 A->B->A 되돌림(응답 노드 불일치) 잡음이었고, 동시 75초에 REST 고유 값 10개 = WS 10개였다.
#   거래소 OI 갱신은 5~10초이고 WS 가 전부 준다(`ts` 도 진짜 거래소 시각). REST 는 수집기와 같은 IP
#   에서 한도 80% 를 먹다 429 를 냈다.
OKX_RECV_TIMEOUT = 25.0
# ⑨ 바이낸스 **현물**(2026-09-23). 점유는 4.5%로 작지만 3일 백필에서 OKX 위에 얹었을 때
#   1분 +1.6pp / 5분 +3.0pp 로 **3일 전부 양수**였고 비용이 스트림 한 줄이다.
#   🔴`@aggTrade` 는 OKX `trades` 와 같이 **이미 테이커 주문 단위**다(체결 수는 l-f+1).
#   🔴`m`(매수자가 메이커)은 선물과 **같은 규약**이다 -- OKX 만 `side` 가 직접이라 반대다.
#   🔴수량은 ETH 직접이다(계약 환산 없음).
SPOT_WS_URL = "wss://stream.binance.com:9443/ws/ethusdt@aggTrade"
SPOT_RECV_TIMEOUT = 60.0        # 12 msg/s 지만 한산한 초가 있어 여유를 둔다


# ── 코인별 흐름 엔진 (2026-09-26 사용자 지시: SOL·XRP 도 ETH 와 같은 구성) ──────────────────
# 풋프린트·1초 수급·OI·OKX·현물 수집을 **코인 하나를 받아 만드는** 공장이다. 본문은 make_app 에
#   있던 ETH 코드를 옮긴 것이다(로그 표식·OI 저장 심볼 인자만 바뀜) -- 맨 위에서 같은 이름의 지역 변수
#   (FOOTPRINT_SYMBOL 등)가 모듈 상수를 가려 코인별 값을 쓰게 한다. ETH 는 전과 같은 값·파일·로그다.
# 🔴데이터는 전부 **USDT 시장**이다(선물 <COIN>USDT · 현물 <coin>usdt · OKX <COIN>-USDT-SWAP). 주문만 USDC
#   (MANUAL_EXEC_SYMBOL) -- 사용자 규칙. 여기에 USDC 심볼을 넣지 않는다.
# 🔴가격칸(bucket)은 ETH 의 상대 폭(0.5/2,680 ≈ 1.9bp)에 맞추고 거래소 틱의 배수로 잡았다. OKX 테이프 칸
#   (OKX_TAPE_BUCKETS: SOL 0.01 · XRP 0.0001)의 정수배여야 복원(okx_tape_footprint)이 칸 경계를 안 찢는다.
#   price_dp 는 칸 가격을 응답에 실을 때 반올림 자릿수 -- 2 로 고정하면 XRP 0.0003 칸들이 한 값으로 뭉친다.
@dataclass(frozen=True)
class FlowSpec:
    asset: str
    symbol: str              # 바이낸스 선물(USDT)
    bucket: float
    price_dp: int
    okx_inst: str            # OKX 무기한(USDT)
    spot_url: str            # 바이낸스 현물(USDT) @aggTrade
    snapshot_path: Path
    okx_tape_db: Path
    okx_ctx_db: Path


def _flow_spec(asset: str, bucket: float, price_dp: int) -> FlowSpec:
    eth = asset == "eth"
    return FlowSpec(
        asset=asset, symbol=f"{asset.upper()}USDT", bucket=bucket, price_dp=price_dp,
        okx_inst=OKX_INST if eth else f"{asset.upper()}-USDT-SWAP",
        spot_url=f"wss://stream.binance.com:9443/ws/{asset}usdt@aggTrade",
        snapshot_path=LIVE_DIR / f"footprint_{asset}.json",
        okx_tape_db=OKX_HOT_TAPE_DB,
        # SOL·XRP 의 OKX 맥락(OI·청산)은 Pi 가 수집하고 3분마다 .from_pi 로 복제한다(Pi crontab). 이 파일이 없으면
        #   복원이 체결만 되살리고, OI 5분봉은 커버리지 가드가 OKX 없는 봉을 버려 재시작 이후분만 남는다.
        okx_ctx_db=OKX_HOT_CTX_DB)      # 4b: SOL·XRP 도 서버 수집(Pi 복제 .from_pi 끝)


FLOW_SPECS = {"eth": _flow_spec("eth", FOOTPRINT_BUCKET, 2),
              "sol": _flow_spec("sol", 0.02, 2),
              "xrp": _flow_spec("xrp", 0.0003, 4)}
assert FLOW_SPECS["eth"].snapshot_path == FOOTPRINT_SNAPSHOT_PATH and FLOW_SPECS["eth"].spot_url == SPOT_WS_URL


def make_coin_flow(spec: FlowSpec, fetch_binance_json: Any, http_session: dict) -> SimpleNamespace:
    FOOTPRINT_SYMBOL = spec.symbol                  # noqa: N806 -- 아래 옮긴 본문이 이 이름들을 그대로 쓴다
    FOOTPRINT_BUCKET = spec.bucket                  # noqa: N806
    FOOTPRINT_SNAPSHOT_PATH = spec.snapshot_path    # noqa: N806
    OKX_INST = spec.okx_inst                        # noqa: N806
    OKX_CT_VAL = OKX_CT_VALS.get(spec.okx_inst)     # noqa: N806
    SPOT_WS_URL = spec.spot_url                     # noqa: N806
    OKX_TAPE_DB_PATH = spec.okx_tape_db             # noqa: N806
    OKX_CTX_DB_PATH = spec.okx_ctx_db               # noqa: N806
    RETAIL_MAX_USD, WHALE_MIN_USD = size_bands(spec.symbol)   # noqa: N806 -- 2026-09-27 코인별 고래 경계(테이프 수집기와 같은 표)
    TAG = "" if spec.asset == "eth" else f"[{spec.asset}] "   # noqa: N806 -- 로그 표식(ETH 는 전과 같게 비움)
    # 🔴2026-09-26 IP 밴 사고: aggTrades 백필은 요청당 가중치 20 이고 IP 한도(2,400/분)를 봇·대시보드·수집기가 같이
    #   쓴다. 코인마다 REST 백필을 돌리면 콜드스타트에 한도를 넘긴다(실측: 418 밴 27분). ETH 외 코인은 서버에 이미
    #   쌓이는 **코인별 체결 테이프**(같은 되묶기·같은 고래/리테일 칸, 1분봉 대조 통과)에서 되살린다 -- REST 0.
    REST_BACKFILL = spec.asset == "eth"                                      # noqa: N806
    TAPE_DB = HOT_TAPE_DB     # 2026-10-01 hot SQLite(5코인 한 파일, 락 없음) -- 쿼리가 심볼로 거른다  # noqa: N806

    # ── 볼륨 풋프린트 체결 테이프 (2026-09-15) ────────────────────────────────────
    # 봉 하나를 가격레벨로 쪼개 **공격적 매수/매도** 체결량을 따로 센다. klines 에는 이 정보가
    # 없다(봉당 taker_buy 합계 하나뿐) -- 그래서 체결 테이프를 직접 누적한다.
    # `m` = "매수자가 메이커" 이므로 m=True 면 **공격자는 매도자**다(check_footprint_tape 로 검증).
    #
    # 실시간은 WS, 과거는 REST 다. 처음엔 REST 폴링만으로 만들었다가 갈아엎었다 -- 2026-09-15
    # 22:35 봉이 275,040 ETH(평소의 8배)로 터지자 1000건/요청 페이스가 **25분** 뒤처졌다.
    # 풋프린트가 가장 필요한 순간이 바로 그때인데 그때 밀린다. WS 는 가중치가 0이라 급증에도
    # 안 밀리고, REST 는 «연결 이전 구간» 백필에만 쓴다(이건 늦어도 라벨로 알리면 그만이다).
    # ⚠️@aggTrade 는 2026-09-02 부터 바이낸스가 배달을 멈췄다(구독은 에러 없이 되고 메시지만
    #   안 온다 -- 2026-09-15 재확인: 10초에 0건). @trade 를 쓴다(개별 체결, 같은 p/q/T/m).
    FOOTPRINT_AGG_URL = "https://fapi.binance.com/fapi/v1/aggTrades"
    FOOTPRINT_WS_URL = f"wss://fstream.binance.com/ws/{FOOTPRINT_SYMBOL.lower()}@trade"
    footprint_state: dict[str, Any] = {"bars": {}, "ready": False, "updated": 0.0,
                                   "last_ms": 0, "saved_at": 0.0,
                                   # aggTrades 로 메운 봉. 출처 표시일 뿐 «덜 정확하다»는 뜻이
                                   # 아니다 -- 크기 구간은 주문 단위라 aggTrades 쪽이 오히려
                                   # 정확하다(footprint_add 도크스트링의 2026-09-19 정정).
                                   "agg_bars": set(),
                                   # 초 -> [리테일매수, 리테일매도, 고래매수, 고래매도,
                                   #        총매수, 총매도, 마지막가격]. 아래 supply_1s_cell 참고.
                                   "sec": {}}

    def supply_1s_cell(ts_ms: int) -> list[float]:
        """그 초의 칸. 오래된 초는 새 초가 생길 때만 버린다(체결마다 돌 일이 아니다)."""
        sec = int(ts_ms) // 1000
        by_sec = footprint_state["sec"]
        cell = by_sec.get(sec)
        if cell is None:
            cell = by_sec[sec] = [0.0] * 7
            cutoff = sec - SUPPLY_1S_SECONDS
            for old in [s for s in by_sec if s < cutoff]:
                del by_sec[old]
        return cell

    def footprint_bar_start(ts_ms: float) -> int:
        return (int(ts_ms) // 1000) // FOOTPRINT_BAR_SECONDS * FOOTPRINT_BAR_SECONDS

    def footprint_add(price: float, qty: float, ts_ms: int, sell: bool,
                      agg: bool = False, order: bool = False) -> None:
        """order=False 는 **개별 체결**(셀 총량만), True 는 **테이커 주문**(크기 구간만).

        백필의 aggTrades 한 줄은 그 자체가 주문이라 둘 다다. 실시간 `@trade` 는 총량을 바로
        넣고, 주문이 닫힐 때 구간을 따로 넣는다 -- 한 번에 못 하는 이유는 주문이 닫히기
        전에는 그 주문이 어느 통에 속하는지 알 수 없기 때문이다(쓸어담는 중에는 계속 자란다).

        🔴여기서 한 번 틀렸다(2026-09-19). 처음엔 개별 체결(`@trade`)로 갈랐는데, 큰 주문이
          호가를 쓸면 그게 작은 체결 수십 건으로 쪼개져 **고래가 사라진다**(같은 11.3초 구간
          실측: aggTrade 기준 37.4% vs @trade 기준 9.6%, 총 명목은 동일). 그다음엔 반대로
          백필 덩어리를 개별 수로 나눠봤는데 그건 **과교정**이라 고래가 0.0% 가 됐다 --
          덩어리는 이미 «주문 하나»라 나눌 것이 아니었다.
          답은 둘 다 **주문 단위로 맞추는 것**이다: 백필은 덩어리를 그대로 쓰고, 실시간은
          같은 규칙(가격·방향·ms)으로 되묶는다. 기존 `nif_whale` 도 @aggTrade 기준이라
          이래야 이 저장소에서 「고래」가 한 뜻이 된다."""
        bars = footprint_state["bars"]
        bar = footprint_bar_start(ts_ms)
        cells = bars.get(bar)
        if cells is None:
            cutoff = (footprint_bar_start(time.time() * 1000)
                      - (FOOTPRINT_KEEP_BARS - 1) * FOOTPRINT_BAR_SECONDS)
            if bar < cutoff:
                return              # 창을 벗어난 봉 -- 넣어봐야 바로 아래에서 지워진다
            cells = bars[bar] = {}
            for old_bar in [b for b in bars if b < cutoff]:
                del bars[old_bar]   # 새 봉이 생길 때만 정리한다 -- 체결마다 돌 일이 아니다(266/s)
                footprint_state["agg_bars"].discard(old_bar)
        cell = cells.setdefault(int(round(price / FOOTPRINT_BUCKET)), [0.0] * 6)
        sec_cell = supply_1s_cell(ts_ms)
        side = 1 if sell else 0
        if not order:
            cell[side] += qty
            sec_cell[4 + side] += qty
            sec_cell[6] = price          # 그 초의 마지막 체결가 = 1초 가격선
        if order or agg:
            notional = price * qty
            if notional >= WHALE_MIN_USD:
                cell[2 + side] += qty
                sec_cell[2 + side] += qty
            elif notional < RETAIL_MAX_USD:
                cell[4 + side] += qty
                sec_cell[side] += qty
        if agg:
            footprint_state["agg_bars"].add(bar)
        # 🔴`now` 는 아래 스냅샷 저장 조건이 쓴다. 2026-09-19 리팩터에서 이 줄을 지웠다가
        #   **ready 가 되는 순간 모든 체결이 NameError** 로 터졌다(단락평가 때문에 ready
        #   이전에는 조용했다). 결과: WS 루프가 크래시->재연결->백필을 무한 반복해 실시간
        #   누적이 아예 안 됐고 REST 를 계속 때렸다. 한 줄이 지워진 걸 테스트가 못 잡은 건
        #   이 경로에 «ready 이후 체결» 을 태우는 시험이 없었기 때문이다.
        now = time.time()
        footprint_state["updated"] = now
        # ⚠️ready 일 때만 저장한다. 백필이 **진행 중인 봉**을 저장하면, 다음 판이 그걸 «이미 있는
        # 봉»으로 보고 건너뛰어 반쪽짜리로 굳는다(2026-09-15 시험에서 한 봉이 -83.7% 로 남았다).
        # 저장된 스냅샷의 계약은 «last_ms 까지 공백이 없다» 이고, 그 보증이 곧 ready 다.
        if footprint_state["ready"] and now - footprint_state["saved_at"] >= FOOTPRINT_SNAPSHOT_SECONDS:
            footprint_save()   # 30초마다 -- 죽어도 잃는 건 30초어치이고 그건 REST 로 메운다

    def footprint_save() -> None:
        """봉 상태를 통째로 덮어쓴다(수십 KB). 증분 append 를 안 쓰는 이유는 진행 중인 봉이
        계속 자라기 때문 -- 어차피 마지막 상태만 쓸모 있다. tmp -> replace 로 원자적으로."""
        try:
            tmp = FOOTPRINT_SNAPSHOT_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "bar_seconds": FOOTPRINT_BAR_SECONDS, "bucket": FOOTPRINT_BUCKET,
                "symbol": FOOTPRINT_SYMBOL, "last_ms": footprint_state["last_ms"],
                # 셀 칸수 표식. 2칸 시절 스냅샷을 4칸 코드가 읽으면 IndexError 가 아니라
                # **조용히 고래 0** 이 된다 -- 그래서 버전을 적고 다르면 통째로 버린다.
                "cells_v": 3,
                # 봉 링은 24시간이지만 저장은 **화면이 고를 수 있는 가장 긴 창**만 한다
                # (위 FOOTPRINT_MAX_WINDOW_BARS). 그 밖은 재시작 뒤 다시 찬다 -- 화면이
                # 실제 봉 수를 적으므로 짧아진 걸 숨기지 않는다.
                "bars": {str(bar): {str(k): v for k, v in cells.items()}
                         for bar, cells in sorted(footprint_state["bars"].items())[-FOOTPRINT_MAX_WINDOW_BARS:]},
            }))
            tmp.replace(FOOTPRINT_SNAPSHOT_PATH)
            footprint_state["saved_at"] = time.time()
        except OSError as exc:
            footprint_state["saved_at"] = time.time()   # 매 체결마다 재시도하지 않게
            print(f"{TAG}footprint snapshot save failed: {exc}", flush=True)

    def footprint_load() -> None:
        """재시작 직후 1회. 창 밖 봉은 버리고, last_ms 를 복원해 백필이 «꺼져 있던 구간»만
        메우게 한다. 봉 길이나 버킷이 바뀌었으면 통째로 무시한다 -- 섞이면 조용히 틀린다."""
        try:
            saved = json.loads(FOOTPRINT_SNAPSHOT_PATH.read_text())
        except (OSError, ValueError):
            return
        if (saved.get("bar_seconds") != FOOTPRINT_BAR_SECONDS
                or saved.get("bucket") != FOOTPRINT_BUCKET
                or saved.get("symbol") != FOOTPRINT_SYMBOL
                or saved.get("cells_v") != 3):
            print(f"{TAG}footprint snapshot: 설정이 달라 무시한다", flush=True)
            return
        cutoff = (footprint_bar_start(time.time() * 1000)
                  - (FOOTPRINT_MAX_WINDOW_BARS - 1) * FOOTPRINT_BAR_SECONDS)
        bars = {int(bar): {int(k): [float(x) for x in v] for k, v in cells.items()}
                for bar, cells in (saved.get("bars") or {}).items() if int(bar) >= cutoff}
        if not bars:
            return
        footprint_state["bars"] = bars
        footprint_state["last_ms"] = int(saved.get("last_ms") or 0)
        print(f"{TAG}footprint snapshot: {len(bars)}봉 복원", flush=True)

    async def footprint_backfill(gap_from_ms: int, until_ms: int) -> None:
        """WS 가 못 준 구간을 aggTrades 로 메운다. 최신 봉부터, 봉마다 «받을 창»을 따로 잡는다.

        창 계산이 이 함수의 전부다:
          - state 에 **없는** 봉  -> 봉 전체(단 until_ms 까지). 처음 뜰 때·WS 가 오래 끊겼을 때.
          - state에 **있는** 봉   -> [gap_from_ms, until_ms] 와 겹치는 부분만. gap_from_ms 는
            끊기기 직전 마지막 WS 체결 시각이라, 이미 센 체결을 다시 세지 않는다.
        이 구분이 없으면 둘 중 하나가 깨진다 -- 「있는 봉은 건너뛴다」로 두면 **서버가 뜬 봉의
        앞부분**이 통째로 빈다(2026-09-15 실측 43% 누락), 무조건 다시 받으면 이중계상이다.

        왜 최신 봉부터인가: 오래된 쪽부터 한 줄로 걸었더니 거래량이 8배로 터진 봉(275,040 ETH
        ≈ 요청 130회)에 막혀 **정작 사용자가 보는 최근 봉이 계속 비어 있었다**.

        요청당 1000건·가중치 20 이라 FOOTPRINT_CATCHUP_SECONDS 로 페이스를 걸고, 총 요청 수에
        상한을 둔다(폭주 구간에서 무한정 긁지 않도록)."""
        # 🔴2026-09-23 400 -> 1200. 창을 1시간(12봉)에서 12시간(144봉)으로 넓혔으므로 예산도
        #   같이 늘린다 -- 안 늘리면 깊은 꼬리가 영영 안 찬다. 페이스가 2.5초라 최악 50분이고
        #   가중치는 1200*20/50분 = 480/분(한도 2400/분)이라 여유가 있다. 동시 실행 가드가
        #   있어(`backfill is None or backfill.done()`) 재연결이 잦아도 겹치지 않는다.
        # ponytail: 그래도 콜드스타트 12시간을 한 번에 다 못 채울 수 있다(봉당 요청 수가
        #   거래량에 비례). 최신 봉부터 돌므로 «보이는 쪽»이 먼저 차고, 남은 꼬리는 다음
        #   재연결과 라이브 누적이 메운다. 더 필요하면 duckdb trade_tape 에서 파생이 정답.
        budget = 1200         # 평소 1시간 백필은 ~200회면 끝난다
        MISS_LIMIT = 10       # 빈 응답(429·5xx) 연속 허용치 -- 한 번에 포기하면 조용히 죽는다
        now_bar = footprint_bar_start(until_ms)
        # 🔴2026-09-23 여기가 FOOTPRINT_BARS(=12, 1시간)였다. 차트에 1h 창 하나뿐이던 시절의
        #   상수인데 지금 창은 1h/2h/4h/12h(12/24/48/144봉)다. 그래서 재시작 뒤 **최근 1시간만**
        #   보장되고 나머지는 라이브로 쌓이길 기다렸다 -- 실측 링 64봉(5.3h)/144. 창 밖 봉은
        #   화면에서 셀 없는 맨 캔들로 남는다(사용자 신고 «맨 왼쪽 캔들이 그냥 캔들이야»).
        #   스냅샷이 저장하는 깊이와 같은 상수로 맞춘다 -- 둘이 어긋나면 복원해도 구멍이 남는다.
        #   따뜻한 재시작에서는 대부분 `lo >= hi` 로 건너뛰어 요청이 0 이다. 비싼 건 콜드스타트뿐.
        window_floor = (now_bar - (FOOTPRINT_MAX_WINDOW_BARS - 1) * FOOTPRINT_BAR_SECONDS)
        try:
            for idx, bar in enumerate(
                    [now_bar - i * FOOTPRINT_BAR_SECONDS
                     for i in range(FOOTPRINT_MAX_WINDOW_BARS)]):
                # 🔴최근 1시간이 끝나면 곧바로 ready 다. 창을 12배로 넓히면서 이걸 안 하면
                #   콜드스타트에서 budget 이 깊은 꼬리에 소진돼 **ready 가 영영 False** 로 남고,
                #   화면은 계속 「수집 중」이며 full=not ready 라 매 폴링이 전량(12.5KB)이 된다.
                if idx == FOOTPRINT_BARS:
                    footprint_state["ready"] = True
                if bar < window_floor:
                    continue
                lo = (bar * 1000 if bar not in footprint_state["bars"]
                      else max(bar * 1000, gap_from_ms))
                hi = min((bar + FOOTPRINT_BAR_SECONDS) * 1000, until_ms)
                if lo >= hi:
                    continue                  # 이 봉은 이미 채워져 있다(겹치는 공백이 없다)
                next_id, misses = None, 0
                while budget > 0:
                    budget -= 1
                    rows = None
                    if next_id is None:
                        seed = await fetch_binance_json(FOOTPRINT_AGG_URL, {
                            "symbol": FOOTPRINT_SYMBOL, "startTime": lo,
                            "endTime": lo + 2000, "limit": 1})
                        if seed:
                            next_id, misses = int(seed[0]["a"]), 0
                            continue
                    else:
                        rows = await fetch_binance_json(FOOTPRINT_AGG_URL, {
                            "symbol": FOOTPRINT_SYMBOL, "fromId": next_id, "limit": 1000})
                    if not rows:
                        misses += 1
                        if misses > MISS_LIMIT:
                            print(f"{TAG}footprint backfill: {bar} 봉 빈 응답 {MISS_LIMIT}회, 포기",
                                  flush=True)
                            break
                        await asyncio.sleep(5.0)
                        continue
                    misses = 0
                    for row in rows:
                        ts_ms = int(row["T"])
                        if lo <= ts_ms < hi:   # 창 밖은 그 창의 차례에 받는다(또는 이미 있다)
                            # aggTrades 한 줄이 곧 주문 하나다 -- 나누지 않는다.
                            footprint_add(float(row["p"]), float(row["q"]), ts_ms,
                                          bool(row["m"]), agg=True)
                    next_id = int(rows[-1]["a"]) + 1
                    if len(rows) < 1000 or int(rows[-1]["T"]) >= hi:
                        break                  # 이 봉 끝
                    await asyncio.sleep(FOOTPRINT_CATCHUP_SECONDS)
                if budget <= 0:
                    print(f"{TAG}footprint backfill: 요청 상한 소진 -- 남은 봉은 다음 재연결에", flush=True)
                    return
            footprint_state["ready"] = True    # 공백이 없다 = 화면의 「수집 중」을 내린다
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- 실패하면 ready 가 False 로 남아 화면이
            # 계속 「수집 중」이라고 말한다. 다음 재연결이 다시 시도한다.
            print(f"{TAG}footprint backfill failed: {exc}", flush=True)

    async def footprint_tape_restore(gap_from_ms: int, until_ms: int) -> None:
        """REST 백필 대신 체결 테이프로 (gap_from_ms, until_ms) 를 메운다(ETH 외 코인).

        초 단위라 경계 두 초는 뺀다 -- 끊기기 직전 초(WS 가 이미 일부 셈)와 붙은 첫 초(WS 가 나머지를 센다).
        ponytail: 경계 두 초의 일부 체결은 빠진다(5분봉의 <1%). 초를 ms 로 쪼개 겹치게 하려면 테이프가 ms 를 들어야 한다.
        테이프는 5초마다 쓰므로 붙은 직후엔 아직 그 앞 몇 초가 없다 -- 따라잡을 때까지 기다린다(최대 60초)."""
        now_bar = footprint_bar_start(until_ms)
        floor_sec = now_bar - (FOOTPRINT_MAX_WINDOW_BARS - 1) * FOOTPRINT_BAR_SECONDS
        lo_sec = max(floor_sec, gap_from_ms // 1000 + 1) if gap_from_ms else floor_sec
        hi_sec = until_ms // 1000                     # 이 초는 WS 가 센다(배타)
        got = None
        try:
            for _ in range(12):
                got = await asyncio.to_thread(okx_tape_footprint, TAPE_DB, LIVE_DIR / "_no_ctx.duckdb",
                                              spec.symbol.lower(), lo_sec, hi_sec * 1000,
                                              FOOTPRINT_BAR_SECONDS, FOOTPRINT_BUCKET,
                                              TAPE_BUCKETS[spec.symbol.lower()])
                if got["tape_max"] >= hi_sec - 1:
                    break
                await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- 테이프가 없으면 라이브만으로 산다(그 앞 봉은 빈 캔들)
            print(f"{TAG}footprint 테이프 복원 실패: {exc!r}", flush=True)
        if got:
            cut = now_bar - (FOOTPRINT_KEEP_BARS - 1) * FOOTPRINT_BAR_SECONDS
            bars = footprint_state["bars"]
            for bar, cells in got["bars"].items():
                if bar < cut:
                    continue
                dst = bars.setdefault(bar, {})
                for k, v in cells.items():
                    t = dst.setdefault(k, [0.0] * 6)
                    for j in range(6):
                        t[j] += v[j]
            print(f"{TAG}footprint 테이프 복원: {len(got['bars'])}봉 (테이프 끝 {hi_sec - got['tape_max']}초 전)",
                  flush=True)
        footprint_state["ready"] = True

    # 진행 중인 테이커 주문(같은 가격·방향·ms). 수집기와 **같은 클래스**를 쓴다 -- 되묶기
    # 규칙이 두 벌이 되면 화면과 DB 의 「고래」가 또 갈라진다.
    fp_orders = TakerOrderAggregator()

    async def collect_footprint(app: web.Application) -> None:
        backfill: asyncio.Task | None = None
        footprint_load()   # 지난 판이 남긴 봉들 -- 이게 있으면 아래 백필은 공백만 메운다
        # 공용 세션(binance_session)은 total=10초라 WS 에 못 쓴다 -- aiohttp 버전에 따라 그
        # 타임아웃이 WS 에도 걸려 10초마다 끊긴다(끊길 때마다 백필이 다시 뜬다). 전용 세션을
        # 쓰되 total=None 을 **명시**한다: aiohttp 기본값은 5분이라 그냥 두면 5분마다 끊긴다.
        ws_session = ClientSession(timeout=ClientTimeout(total=None),
                                   connector=TCPConnector(limit=2))
        try:
            while True:
                try:
                    async with ws_session.ws_connect(FOOTPRINT_WS_URL, heartbeat=30) as ws:
                        # WS 는 연결 이후만 준다 -- 그 앞의 빈 봉은 REST 가 메운다. 메시지를
                        # 읽으면서 **동시에** 채운다(백필을 기다리면 그동안 오는 체결이 aiohttp
                        # 큐에 수만 건 쌓인다).
                        first_ms: int | None = None
                        async for msg in ws:
                            if msg.type is not WSMsgType.TEXT:
                                break
                            trade = json.loads(msg.data)
                            if trade.get("e") != "trade":
                                continue
                            price, qty = float(trade["p"]), float(trade["q"])
                            if not (price > 0 and qty > 0):
                                # 바이낸스가 {"p":"0","q":"0","X":"NA","st":1} 를 섞어 보낸다
                                # (2026-09-16 실측 0.3%). 수량이 0이라 합계는 안 틀려서
                                # klines 대조로는 안 잡힌다 -- 대신 **가격 0 레벨**이 생긴다.
                                continue
                            ts_ms = int(trade["T"])
                            if first_ms is None:
                                # 백필의 경계를 **로컬 시계가 아니라 첫 체결의 거래소 시각**으로
                                # 잡는다. time.time() 으로 잡았더니 시계 차이만큼 REST 와 WS 가
                                # 겹쳐 그 봉만 +0.54% 더 세어졌다(2026-09-15 klines 대조).
                                first_ms = ts_ms
                                # 돌고 있는 백필이 있으면 새로 띄우지 않는다(REST 가중치가 두 배).
                                # 대가: 첫 백필 도중 WS 가 끊기면 그 공백은 다음 재연결 때 메워진다.
                                if backfill is None or backfill.done():
                                    fp_orders.reset()   # 끊김 전 묶음은 버린다
                                    footprint_state["ready"] = False
                                    backfill = asyncio.create_task(
                                        (footprint_backfill if REST_BACKFILL else footprint_tape_restore)(
                                            footprint_state["last_ms"], first_ms))
                            sell = bool(trade["m"])
                            footprint_add(price, qty, ts_ms, sell)      # 총량
                            tid = trade.get("t")
                            done = fp_orders.add(price, qty, ts_ms, sell,
                                                 None if tid is None else int(tid))
                            if done is not None:                        # 닫힌 주문의 크기 구간
                                footprint_add(done[0], done[1], done[2], done[3], order=True)
                            footprint_state["last_ms"] = ts_ms   # 다음 재연결이 메울 공백의 시작
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- 한 번의 끊김/에러가 수집기를 영구히
                    # 죽이면 재시작 전까지 풋프린트가 통째로 빈다(publish_dashboard_events 와
                    # 같은 이유). 끊기면 3초 뒤 다시 붙는다.
                    print(f"{TAG}footprint collector cycle failed (will reconnect): {exc}", flush=True)
                await asyncio.sleep(3.0)
        finally:
            if backfill is not None:
                backfill.cancel()
            await ws_session.close()

    # 초 -> 그 초의 미결제약정(ETH). 값이 **바뀐** 초만 들어간다(SUPPLY_1S_SECONDS 아래 OI 주석 참고).
    oi_1s: dict[int, float] = {}

    async def collect_oi_1s(app: web.Application) -> None:
        """🔴2026-10-01 저장 재설계 4d: REST 폴링은 수집기(scripts/live_binance_ctx_collector_20261001.py) **하나만** 한다 --
        대시보드와 둘이 0.25초씩 폴링하면 IP 한도(2,400/분)를 넘는다. 여기서는 그 수집기가 쓰는 hot(WAL)을 0.5초마다
        꼬리만 읽어 화면 링을 채운다. 원천 갱신이 3~7초라 0.5초 읽기로 충분하다(수집기 flush 10초만큼 늦을 수 있다).
        ponytail: 수집기 flush(10초)가 화면 OI 지연의 하한이다 -- 더 빨라야 하면 수집기 flush 를 1초로."""
        sym = FOOTPRINT_SYMBOL.lower()
        last_ms = int(time.time() * 1000) - SUPPLY_1S_SECONDS * 1000
        while True:
            try:
                # 30초 겹쳐 읽는다 -- 늦게 도착한 stamp(약 3.5%)가 flush 경계를 넘어 last_ms 보다 앞에 써지면 놓친다(10-01 리뷰)
                rows = await asyncio.to_thread(read_rows, BINANCE_CTX_DB, "SELECT ts_ms, open_interest FROM oi_1s "
                                               "WHERE symbol = ? AND ts_ms > ? ORDER BY ts_ms", [sym, last_ms - 30_000])
                for ts_ms, value in rows:
                    oi_1s[int(ts_ms) // 1000] = float(value)
                    last_ms = max(last_ms, int(ts_ms))
                if rows:
                    cut = int(time.time()) - SUPPLY_1S_SECONDS
                    for old in [x for x in oi_1s if x < cut]:
                        del oi_1s[old]
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- 수집기가 아직 없거나 파일이 없으면 다음에
                print(f"{TAG}oi-1s hot read failed (will retry): {exc!r}", flush=True)
                await asyncio.sleep(5.0)
            await asyncio.sleep(0.5)

    # ── OKX 실시간 ─────────────────────────────────────────────────────────
    # 바이낸스와 **같은 모양**으로 든다. 그래야 클라 렌더러가 데이터 출처만 바꿔 끼우면
    # 누적·스택·청산점 수식이 그대로 돈다(복사본을 만들면 한쪽만 고쳐진다).
    #   okx_sec[s] = [리테일매수, 리테일매도, 고래매수, 고래매도, 총매수, 총매도, 가격]
    okx_sec: dict[int, list[float]] = {}
    okx_liq_events: deque = deque(maxlen=5000)
    okx_oi_1s: dict[int, float] = {}
    okx_state: dict[str, Any] = {"connected": False, "trades": 0, "errors": 0,
                                 "last_trade_ms": 0, "last_oi_ms": 0, "last_liq_ms": 0}
    # 풋프린트용 OKX 봉. `footprint_state["bars"]` 와 **같은 격자·같은 6칸**이다
    # ([매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도]).
    # 🔴바이낸스 bars 에 **섞지 않는다** -- 거기엔 kline 완전성 검사·REST 백필·스냅샷이
    #   전부 «바이낸스 aggTrades» 를 전제로 걸려 있다. 합치는 건 API 응답을 만들 때만.
    okx_bars: dict[int, dict[int, list[float]]] = {}
    # 🔴OKX 는 **풋프린트 백필이 불가능하다**(history-trades 가 100행=10.8초, 2026-09-23 실측).
    #   그래서 서버가 뜨기 전 봉은 바이낸스만 있고, 그냥 더하면 재기동마다 **봉 높이가
    #   경계에서 튄다**(흡수·벽을 읽는 그림에서 없던 단차). 답은 «OKX 가 **완전히 덮은**
    #   첫 봉 이후만 내보내기»다 -- 창이 짧아지는 대신 단차가 없다(사용자 선택 C).
    #   ⚠️«OKX 가 있는 봉만» 으로 짜면 OKX 가 잠깐 끊길 때 풋프린트가 **통째로 사라진다**.
    #     시작 봉만 기억하는 이 방식은 순간 끊김에 안 죽는다(재연결 3초 = 봉의 1%).
    # ⭐2026-09-24 A안: 기동 뒤 okx_footprint_restore 가 OKX 테이프에서 과거 봉을 되살려
    #   first_bar 를 창 끝까지 당긴다(아래 okx_tape_footprint). 그 전까지는 위 C안 그대로다.
    okx_fp = {"first_bar": 0, "t0_ms": 0, "resync_until": 0.0}
    # 5분봉별 OKX OI [봉 첫 값, 봉 끝 값](ETH, oiCcy). 2026-09-24 사용자 «사분면·누적 OI 도 OKX
    #   합산». okx_oi_1s 는 11분 링이라 창(1~4h)의 봉별 Δ 를 못 낸다. 재기동 전 봉은 없다 --
    #   풋프린트가 어차피 OKX 가 덮은 봉부터만 나가므로(okx_fp) 레인에 그려지는 봉과 겹친다.
    okx_oi_5m: dict[int, list[float]] = {}

    def okx_sec_cell(ts_ms: int) -> list[float]:
        """그 초의 칸. 오래된 초는 새 초가 생길 때만 버린다(체결마다 돌 일이 아니다)."""
        sec = int(ts_ms) // 1000
        cell = okx_sec.get(sec)
        if cell is None:
            cell = okx_sec[sec] = [0.0] * 7
            cutoff = sec - SUPPLY_1S_SECONDS
            for old in [x for x in okx_sec if x < cutoff]:
                del okx_sec[old]
        return cell

    async def collect_okx_flow(app: web.Application) -> None:
        """⑦ OKX 체결 + 청산. 주문 없음, 바이낸스 weight 안 씀.

        🔴세 가지를 바이낸스와 다르게 다뤄야 한다(전부 2026-09-23 실측으로 확정):
          1. `sz` 는 **계약 수**다 -> OKX_CT_VAL 을 곱해야 ETH 다.
          2. `side` 는 **테이커 측면 그 자체**다 -> 바이낸스 `m`(매수자가 메이커)처럼 뒤집으면
             부호가 통째로 반대가 된다.
          3. `trades` 채널은 **이미 주문 단위**다(`trades-all` 4,810건 vs `trades` 1,780건,
             sz 합은 1.0000x 동일) -> 되묶기를 이식하면 이중 집계다. 크기 구간을 바로 가른다.
        ⚠️청산 `bkPx` 는 **파산가격**이라 바이낸스 `ap`(평균체결가)와 같은 것이 아니다.
          USD 환산에 쓰긴 하지만 두 거래소 청산«가»를 같은 축에 놓으면 안 된다."""
        ws_session = ClientSession(timeout=ClientTimeout(total=None), connector=TCPConnector(limit=2))
        args = [{"channel": "trades", "instId": OKX_INST},
                {"channel": "open-interest", "instId": OKX_INST},
                {"channel": "liquidation-orders", "instType": "SWAP"}]
        # 🔴계약 크기가 틀리면 레인이 조용히 10배가 된다 -- 모르거나 거래소와 다르면 **켜지 않는다**
        #   (화면에선 «합에서 빠짐: OKX» 로 보인다). REST 가 안 될 때만 표 값으로 간다.
        if OKX_CT_VAL is None:
            print(f"{TAG}okx ws: {OKX_INST} ctVal 모름 -- OKX 레인 끔", flush=True)
            await ws_session.close()
            return
        try:
            async with ws_session.get(OKX_INSTRUMENTS_URL, headers=OKX_HTTP_HEADERS,
                                      params={"instType": "SWAP", "instId": OKX_INST},
                                      timeout=ClientTimeout(total=10)) as r:
                got = float((await r.json())["data"][0]["ctVal"])
            if abs(got - OKX_CT_VAL) > 1e-12:
                print(f"{TAG}okx ws: 🔴ctVal 갈라짐 표 {OKX_CT_VAL} vs 거래소 {got} -- OKX 레인 끔", flush=True)
                await ws_session.close()
                return
        except Exception as exc:  # noqa: BLE001
            print(f"{TAG}okx ws: ctVal 대조 실패({exc!r}) -- 표 값 {OKX_CT_VAL} 로 계속", flush=True)
        try:
            while True:
                try:
                    async with ws_session.ws_connect(OKX_WS_URL, heartbeat=20) as ws:
                        await ws.send_json({"op": "subscribe", "args": args})
                        okx_state.update(connected=True)
                        print(f"{TAG}okx ws: connected", flush=True)
                        while True:
                            msg = await ws.receive(timeout=OKX_RECV_TIMEOUT)
                            if msg.type is not WSMsgType.TEXT:
                                print(f"{TAG}okx ws: non-text {msg.type!r} -> reconnect", flush=True)
                                break
                            if msg.data == "pong":
                                continue
                            d = json.loads(msg.data)
                            if d.get("event") == "error":
                                print(f"{TAG}okx ws: 구독 거부 {d.get('msg')}", flush=True)
                                break
                            if d.get("event"):
                                continue
                            ch = (d.get("arg") or {}).get("channel")
                            if ch == "trades":
                                for t in d.get("data") or []:
                                    price = float(t.get("px") or 0.0)
                                    qty = float(t.get("sz") or 0.0) * OKX_CT_VAL
                                    side = str(t.get("side") or "")
                                    if not (price > 0 and qty > 0 and side in ("buy", "sell")):
                                        continue
                                    ts_ms = int(t.get("ts") or time.time() * 1000)
                                    i = 1 if side == "sell" else 0
                                    cell = okx_sec_cell(ts_ms)
                                    cell[4 + i] += qty
                                    cell[6] = price
                                    notional = price * qty
                                    if notional >= WHALE_MIN_USD:
                                        cell[2 + i] += qty
                                    elif notional < RETAIL_MAX_USD:
                                        cell[i] += qty
                                    okx_state["trades"] += 1
                                    okx_state["last_trade_ms"] = ts_ms
                                    # 풋프린트 봉(바이낸스와 같은 격자)
                                    bar = footprint_bar_start(ts_ms)
                                    if not okx_fp["first_bar"]:
                                        okx_fp["first_bar"] = bar
                                        okx_fp["t0_ms"] = ts_ms      # 이 앞은 테이프가 맡는다
                                    cells = okx_bars.get(bar)
                                    if cells is None:
                                        cells = okx_bars[bar] = {}
                                        cut = (footprint_bar_start(time.time() * 1000)
                                               - (FOOTPRINT_KEEP_BARS - 1) * FOOTPRINT_BAR_SECONDS)
                                        for ob in [b for b in okx_bars if b < cut]:
                                            del okx_bars[ob]
                                    fc = cells.setdefault(int(round(price / FOOTPRINT_BUCKET)),
                                                          [0.0] * 6)
                                    fc[i] += qty
                                    if notional >= WHALE_MIN_USD:
                                        fc[2 + i] += qty
                                    elif notional < RETAIL_MAX_USD:
                                        fc[4 + i] += qty
                            elif ch == "open-interest":
                                for r in d.get("data") or []:
                                    oi = float(r.get("oiCcy") or 0.0)      # oiCcy = 기초자산(ETH)
                                    ts_ms = int(r.get("ts") or 0)
                                    if oi > 0 and ts_ms > 0:
                                        sec = ts_ms // 1000
                                        okx_oi_1s[sec] = oi
                                        okx_state["last_oi_ms"] = ts_ms
                                        ob = sec // OI_5M_BAR_SECONDS * OI_5M_BAR_SECONDS
                                        if ob in okx_oi_5m:
                                            okx_oi_5m[ob][1] = oi
                                        else:
                                            okx_oi_5m[ob] = [oi, oi]
                                            for old in [x for x in okx_oi_5m
                                                        if x < ob - FOOTPRINT_KEEP_BARS * OI_5M_BAR_SECONDS]:
                                                del okx_oi_5m[old]
                                        for old in [x for x in okx_oi_1s if x < sec - SUPPLY_1S_SECONDS]:
                                            del okx_oi_1s[old]
                            elif ch == "liquidation-orders":
                                for r in d.get("data") or []:
                                    if str(r.get("instId") or "") != OKX_INST:
                                        continue            # instType:SWAP 은 전 종목이 온다
                                    for det in r.get("details") or []:
                                        qty = float(det.get("sz") or 0.0) * OKX_CT_VAL
                                        px = float(det.get("bkPx") or 0.0)
                                        if qty <= 0:
                                            continue
                                        ts_ms = int(det.get("ts") or time.time() * 1000)
                                        okx_liq_events.append({
                                            "ts_ms": ts_ms,
                                            # posSide 가 **청산된 포지션**의 방향이다(바이낸스는
                                            # 주문 방향에서 뒤집어 얻는다 -- 결과는 같은 뜻).
                                            "side": "long" if det.get("posSide") == "long" else "short",
                                            "qty": qty, "price": px, "usd": qty * px,
                                            "symbol": OKX_INST})
                                        okx_state["last_liq_ms"] = ts_ms
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- 끊기면 3초 뒤 다시. 봇에 영향 없음
                    okx_state.update(errors=okx_state["errors"] + 1)
                    print(f"{TAG}okx ws: {exc!r}", flush=True)
                    await asyncio.sleep(3.0)
                finally:
                    okx_state["connected"] = False
        finally:
            await ws_session.close()

    # ── 바이낸스 현물 (2026-09-23) ─────────────────────────────────────────
    # OKX 와 **같은 7칸 모양**. 합산 패널이 셋을 그대로 더한다.
    # 🔴현물엔 OI·펀딩·마크·강제청산이 **존재하지 않는다** -- 포지션 개념이 없다.
    spot_sec: dict[int, list[float]] = {}
    spot_state: dict[str, Any] = {"connected": False, "trades": 0, "errors": 0, "last_trade_ms": 0}

    def spot_sec_cell(ts_ms: int) -> list[float]:
        sec = int(ts_ms) // 1000
        cell = spot_sec.get(sec)
        if cell is None:
            cell = spot_sec[sec] = [0.0] * 7
            cutoff = sec - SUPPLY_1S_SECONDS
            for old in [x for x in spot_sec if x < cutoff]:
                del spot_sec[old]
        return cell

    async def collect_spot_flow(app: web.Application) -> None:
        """⑨ 바이낸스 현물 @aggTrade. 주문 없음. 선물과 **다른 시장**이라 weight 도 별개다.

        ⚠️풋프린트에는 안 쓴다 -- 현물은 선물보다 +4.69bp 높아 $0.1 빈 기준 **12.9빈**
          어긋난다(2026-09-23 실측). 가격축에 쌓으면 가짜 이중 봉우리가 된다. 수급(CVD)은
          «수량의 합»이라 가격이 무관해서 문제가 없다."""
        # 현물 체결 저장은 2026-10-01 저장 재설계 4d 에 테이프 수집기로 옮겼다(TAPE_MARKET=spot -> data/hot/binance_spot_tape.sqlite,
        #   공백 기록·1분봉 대조·REST 복구 포함). 여기는 화면 흐름만. 옛 data/live/trade_tape_spot*.duckdb 는 09-29~10-01 이력으로 얼었다.
        ws_session = ClientSession(timeout=ClientTimeout(total=None), connector=TCPConnector(limit=2))
        try:
            while True:
                try:
                    async with ws_session.ws_connect(SPOT_WS_URL, heartbeat=30) as ws:
                        spot_state.update(connected=True)
                        print(f"{TAG}spot ws: connected", flush=True)
                        while True:
                            msg = await ws.receive(timeout=SPOT_RECV_TIMEOUT)
                            if msg.type is not WSMsgType.TEXT:
                                break
                            t = json.loads(msg.data)
                            if t.get("e") != "aggTrade":
                                continue
                            price, qty = float(t.get("p") or 0.0), float(t.get("q") or 0.0)
                            if not (price > 0 and qty > 0):
                                continue
                            ts_ms = int(t.get("T") or time.time() * 1000)
                            i = 1 if t.get("m") else 0          # 매수자가 메이커 -> 공격자는 매도
                            cell = spot_sec_cell(ts_ms)
                            cell[4 + i] += qty
                            cell[6] = price
                            # 이 7칸에는 체결 수 자리가 없다(총량·구간·가격뿐) -- `f`/`l` 은
                            # 안 쓴다. 필요해지면 그때 칸을 늘린다.
                            notional = price * qty
                            if notional >= WHALE_MIN_USD:
                                cell[2 + i] += qty
                            elif notional < RETAIL_MAX_USD:
                                cell[i] += qty
                            spot_state["trades"] += 1
                            spot_state["last_trade_ms"] = ts_ms
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    spot_state.update(errors=spot_state["errors"] + 1)
                    print(f"{TAG}spot ws: {exc!r}", flush=True)
                    await asyncio.sleep(3.0)
                finally:
                    spot_state["connected"] = False
        finally:
            await ws_session.close()

    async def okx_footprint_restore(app: web.Application) -> None:
        """A안(2026-09-24): 기동 뒤 OKX 풋프린트 과거분을 OKX 체결 테이프에서 되살린다.

        진행 중인 봉은 **T0(라이브 첫 체결) 앞 몫만** 테이프에서 더하고, 그 앞 봉들은 테이프로
        통째로 채운다 -- 온전한 봉이 끊김 없이 이어지는 데까지만(중간에 온전하지 않은 봉이 있으면
        거기서 멈춘다: 그 앞을 내보내면 C안이 막으려던 단차가 창 안에 생긴다). 사분면 OI 도 같은 봉까지.
        테이프 자동 복구가 공백을 메우는 데 몇 분 걸리므로 5분마다 35분까지 다시 본다.
        🔴DB 읽기는 스레드, 메모리 반영은 **이 루프에서** -- 라이브 체결이 같은 dict 를 만진다."""
        while not okx_fp["t0_ms"]:
            await asyncio.sleep(1.0)
        t0 = okx_fp["t0_ms"]
        start_bar = footprint_bar_start(t0)
        lo = start_bar - (FOOTPRINT_MAX_WINDOW_BARS - 1) * FOOTPRINT_BAR_SECONDS
        t0_min = (t0 // 1000) // 60 * 60
        start_filled, restored, liq_upto = False, set(), t0   # liq_upto: 청산을 이미 채운 구간의 앞끝
        for attempt in range(8):
            await asyncio.sleep(15.0 if attempt == 0 else 300.0)
            try:
                got = await asyncio.to_thread(okx_tape_footprint, OKX_TAPE_DB_PATH, OKX_CTX_DB_PATH,
                                              OKX_INST, lo, t0, FOOTPRINT_BAR_SECONDS, FOOTPRINT_BUCKET,
                                              liq_db=OKX_LIQ_DB_PATH)
            except Exception as exc:  # noqa: BLE001 -- 테이프가 없으면 C안 그대로 산다
                print(f"{TAG}okx footprint 복원 실패({attempt + 1}회): {exc!r}", flush=True)
                continue
            if got["tape_max"] < t0 // 1000:
                continue                          # 테이프가 아직 T0 를 지나 쓰지 않았다
            ok = got["ok_min"]
            covered = lambda a, b: all(m in ok for m in range(a, b, 60))  # noqa: E731
            if not start_filled:
                if not covered(start_bar, t0_min + 60):
                    continue
                cells = okx_bars.setdefault(start_bar, {})
                for k, v in got["bars"].get(start_bar, {}).items():
                    t = cells.setdefault(k, [0.0] * 6)
                    for j in range(6):
                        t[j] += v[j]
                start_filled = True
            earliest, b = start_bar, start_bar - FOOTPRINT_BAR_SECONDS
            while b >= lo and covered(b, b + FOOTPRINT_BAR_SECONDS):
                if b not in restored:             # 라이브는 이 봉을 모른다 -- 교체
                    okx_bars[b] = got["bars"].get(b, {})
                    restored.add(b)
                earliest, b = b, b - FOOTPRINT_BAR_SECONDS
            for ob, (o, c) in got["oi"].items():
                if earliest <= ob < start_bar:
                    okx_oi_5m[ob] = [o, c]
                elif ob == start_bar:
                    okx_oi_5m.setdefault(ob, [o, c])[0] = o
            # 청산: 이번에 새로 덮은 구간 [earliest, liq_upto) 만 앞에 넣는다(시각 순 유지).
            #   🔴deque(maxlen) 에 appendleft 하다 차면 **가장 최신**이 밀려난다 -- 남은 자리만큼만.
            add = [e for e in got["liq"] if earliest * 1000 <= e["ts_ms"] < liq_upto]
            room = (okx_liq_events.maxlen or 10**9) - len(okx_liq_events)
            for e in reversed(add[-room:] if room > 0 else []):
                okx_liq_events.appendleft(e)
            liq_upto = earliest * 1000
            okx_fp["first_bar"] = earliest - FOOTPRINT_BAR_SECONDS
            okx_fp["resync_until"] = time.time() + 15.0   # 열린 화면도 전량 한 번 받게
            print(f"{TAG}okx footprint: 테이프에서 {len(restored)}봉 + 진행 봉 앞부분 복원 "
                  f"(창 {(start_bar - earliest) // FOOTPRINT_BAR_SECONDS + 1}"
                  f"/{FOOTPRINT_MAX_WINDOW_BARS}봉)", flush=True)
            if earliest <= lo:
                return

    async def supply_1s_restore(app: web.Application) -> None:
        """재시작 직후 1초 수급 11분을 되살린다(2026-09-27 사용자 지시). 라이브가 첫 초(t0)를 잡은 뒤 그 **앞만**
        채우고 라이브가 가진 초는 안 건드린다. 체결 = 체결 테이프(바이낸스·OKX), OI = oi_1s DB·OKX 맥락 DB.
        청산은 이미 되살린다(liq_events_load·okx_footprint_restore). 🔴현물은 테이프가 없어 못 한다.
        ponytail: 복원은 새로 연 화면(커서 0)만 받는다 -- 열려 있던 화면은 제 커서 뒤라 재시작 공백이 그대로다."""
        async def fill(dst: dict, load: Any, name: str) -> None:
            for _ in range(60):                     # 라이브 첫 초를 기다린다
                if dst:
                    break
                await asyncio.sleep(1.0)
            if not dst:
                return
            t0 = min(dst)
            got, err = None, None
            for _ in range(12):                     # 테이프는 5초마다 쓴다 -- t0 직전까지 따라잡을 때까지
                try:
                    got, top = await asyncio.to_thread(load, t0 - SUPPLY_1S_SECONDS, t0)
                    if top >= t0 - 1:
                        break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- 외부 쓰기 연결이 잠금을 쥐면 다음 회차에 다시(09-27 실측)
                    err = exc
                await asyncio.sleep(5.0)
            if got is None:                         # 원천이 없으면 라이브만으로 산다
                print(f"{TAG}1초 수급 복원 실패({name}): {err!r}", flush=True)
                return
            n = sum(1 for k, v in got.items() if dst.setdefault(k, v) is v)
            print(f"{TAG}1초 수급 복원({name}): {n}초", flush=True)

        def oi_rows(db: Path, sql: str, key: str, lo: int, hi: int) -> tuple[dict[int, float], int]:
            with duckdb_path_lock(db):
                rows = _read_only_rows(db, sql, [key, lo * 1000, hi * 1000])
            out = {int(ms) // 1000: float(v) for ms, v in rows}   # 시각 순이라 같은 초는 마지막 값
            return out, hi - 1                                    # DB 는 기다려도 안 는다(10초 flush) -- 한 번만
        await asyncio.gather(
            fill(footprint_state["sec"], lambda lo, hi: tape_seconds(
                TAPE_DB, spec.symbol.lower(), lo, hi, TAPE_BUCKETS[spec.symbol.lower()]), "바이낸스 체결"),
            fill(okx_sec, lambda lo, hi: tape_seconds(
                OKX_TAPE_DB_PATH, OKX_INST, lo, hi, OKX_TAPE_BUCKETS[OKX_INST]), "OKX 체결"),
            fill(oi_1s, lambda lo, hi: oi_rows(
                OI_1S_DB_PATH, f"SELECT ts_ms, open_interest FROM {OI_1S_TABLE} WHERE symbol = ? "
                "AND ts_ms >= ? AND ts_ms < ? ORDER BY ts_ms", spec.symbol.lower(), lo, hi), "바이낸스 OI"),
            fill(okx_oi_1s, lambda lo, hi: oi_rows(
                OKX_CTX_DB_PATH, "SELECT ts_ms, oi_base FROM okx_oi WHERE inst = ? "
                "AND ts_ms >= ? AND ts_ms < ? ORDER BY ts_ms", OKX_INST, lo, hi), "OKX OI"))

    tasks: list[asyncio.Task] = []

    async def start(app: web.Application) -> None:
        for fn in (collect_footprint, collect_oi_1s, collect_okx_flow, okx_footprint_restore, collect_spot_flow, supply_1s_restore):
            tasks.append(asyncio.create_task(fn(app)))

    async def stop(app: web.Application) -> None:
        # 내려가기 직전에 한 번 더 남긴다 -- 다음 판이 메울 공백이 «재시작에 걸린 시간»으로 준다.
        if footprint_state["ready"]:
            footprint_save()
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    return SimpleNamespace(
        spec=spec, footprint_state=footprint_state, oi_1s=oi_1s,
        okx_sec=okx_sec, okx_liq_events=okx_liq_events, okx_oi_1s=okx_oi_1s, okx_state=okx_state,
        okx_bars=okx_bars, okx_fp=okx_fp, okx_oi_5m=okx_oi_5m, spot_sec=spot_sec, spot_state=spot_state,
        footprint_tape_restore=footprint_tape_restore, start=start, stop=stop,
        retail_max=RETAIL_MAX_USD, whale_min=WHALE_MIN_USD)
