#!/usr/bin/env python3
"""**체결 테이프 수집기** — 1초 × 가격빈으로 모은 공격적 매수/매도. (2026-09-16)

왜 필요한가: 이 저장소는 호가는 연속으로 쌓지만(orderflow/ 래스터·bookTicker·depthDiff)
**체결은 어디에도 연속으로 안 쌓인다**. 2026-09-15 실측 — `l2_anomaly_trades` 는 이상치 구간만
이라 최근 1시간 12봉 중 9봉이 **0%**, 나머지도 3~17% 였고, `maker_fill_shadow` 는 시뮬레이션
원장이며 `microstructure.duckdb` 는 1분 집계(가격 없음)다. 방향 예측 재료로 쓰려면 새로 받아야
한다. (풋프린트 대시보드는 자기 WS 로 자기 창만 들고 있다 — duckdb 는 프로세스 하나만 열 수
있어서 공유가 불가능하다. 그래서 둘은 서로를 모른다.)

**왜 원시 틱이 아니라 1초 집계인가 — 실측하고 정했다(2026-09-16, ETHUSDT):**
  원시 틱        11.3M 행/일   (체결 크기 중앙값 **0.010 ETH** — 대부분 먼지 체결이다)
  1초 × $0.01    1.94M 행/일   (초당 22.5빈)
  1초 × $0.1      326k 행/일   (초당 3.8빈)  ← 채택, ~4GB/년
$0.1 은 $2,400 자산에서 0.4bp 다. 이 저장소의 왕복 비용(5.88bp)보다 15배 가늘어서 어떤 결정도
이 격자 때문에 바뀌지 않는다. 잃는 건 «한 초 안의 체결 순서»뿐이고, 크기 분포는 빈마다
`n`(건수)과 `max`(그 초의 최대 체결)로 남긴다 — 고래 프린트(상위1% 13.6 ETH)는 보존된다.
원시 틱이 정말 필요해지면 `data.binance.vision` 일별 zip 으로 소급 재구성한다(호가와 달리
체결은 공개돼 있다 — docs/dashboard_orderflow_footprint_heatmap_design_20260914.md §1).

**자기 검증이 내장돼 있다(선택이 아니다).** 체결이 빠지면 델타가 **조용히** 틀어진다 — 화면도
쿼리도 아무 말을 안 한다. 그래서 5분마다 직전 분들의 `sum(buy_qty+sell_qty)` 를 같은 분의
kline `volume` 과 대조해 `verify_1m` 에 남긴다. 연구 쿼리는 이 표를 조인해 나쁜 분을 빼면 된다.
WS 가 끊긴 구간은 `gaps` 에 기록한다 — 메우지 않고 **기록만** 한다(메우면 그게 진짜인지
아닌지 알 수 없어진다. 메울 때는 벌크 zip 으로 그 날을 통째로 덮어쓴다).

트레이딩 봇과 완전 분리: 자기 WS · 자기 duckdb · 주문 없음 · 죽어도 봇에 영향 없다.
(래스터/bookTicker 수집기 도크스트링의 규약을 그대로 따른다.)
⚠️@aggTrade 는 2026-09-02 부터 바이낸스가 배달을 멈췄다(구독은 조용히 성공하고 메시지가 0건).
  `@trade`(개별 체결)를 쓴다. `m`=매수자가 메이커 → **공격자는 매도자**.

사용:
  python scripts/live_trade_tape_collector_20260916.py                 # ethusdt
  TAPE_SYMBOL=btcusdt python scripts/live_trade_tape_collector_20260916.py
  python scripts/live_trade_tape_collector_20260916.py --selftest      # 네트워크·DB 없이 로직 점검
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "live" / "trade_tape.duckdb"
# 가격빈. 자산마다 틱과 가격대가 달라 한 값을 못 쓴다 -- 대략 «가격의 0.4bp» 로 맞췄다.
BUCKETS = {"ethusdt": 0.1, "btcusdt": 1.0, "solusdt": 0.01, "xrpusdt": 0.0001, "hypeusdt": 0.001}
WS_URL = "wss://fstream.binance.com/ws/{symbol}@trade"
KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
FLUSH_SECONDS = 5.0      # 완결된 초만 쓴다 -- 한 초는 정확히 한 번 기록된다
VERIFY_SECONDS = 300.0
VERIFY_TOLERANCE = 1e-4  # 이보다 어긋나면 로그로 떠든다(행은 어차피 남긴다)
SCHEMA_VERSION = 2
# ── 체결 크기 구간 (2026-09-19) ─────────────────────────────────────────────
# 「리테일이 사고 고래가 판다」를 나중에 물어보려면 **크기별로 갈라서** 쌓아야 한다.
# 총량만 쌓으면 소급 복원이 불가능하다(일별 zip 을 다시 받는 것 말고는 길이 없다).
#
# ⭐경계는 **달러 고정**이고 **분위가 아니다**. 분위(p99)로 쌓으면 「이 행의 기준이 얼마였나」를
#   행마다 따로 적어야 하고, 안 적으면 몇 달 뒤 그 숫자를 해석할 방법이 사라진다. 달러 고정은
#   ETH 가격이 올라도 뜻이 안 변한다(고정 «수량» 10 ETH 를 박는 것과는 정반대다).
#
# ⭐두 경계인 이유 — 2026-09-19 ETHUSDT 39,712건(5분) 실측:
#     ≥$10k    건수  4.10%   물량 73.1%
#     ≥$25k    건수  1.53%   물량 48.4%
#     ≥$100k   건수  0.17%   물량 14.6%   (분당 13.4건 = 5분봉당 67건)
#   경계 하나로는 못 가른다. $10k 를 고래라 하면 **물량의 3/4이 고래**라 리테일이 사라지고,
#   $100k 위만 고래라 하고 나머지를 리테일이라 부르면 **물량의 58.5%($10k~$100k)를 리테일로
#   거짓말**하게 된다. 그 중간은 개인도 고래도 아니다 -- 자기 칸을 줘야 한다.
#
# ⭐고래 경계 $100k 는 **이 저장소에 이미 있던 값과 같다** -- `microstructure_scanner.py` 의
#   `MS_WHALE_USD_TH` 기본값(= `nif_whale`/`nif_retail` 칩과 그 2026-08-25 IC 검증이 선 경계)이
#   정확히 100000 이다. 실측으로 따로 고른 값이 같은 자리에 떨어졌다. 그래서 새 화면의 「고래」와
#   기존 「수급 흐름」 칩의 「큰손」은 **같은 것을 가리킨다**.
# 🔴다만 **리테일은 다르다**: `nif_retail` 은 「고래가 아닌 전부」(<$100k)라 위 실측대로 그 안의
#   대부분이 실은 중형이다. 여기서 쓰는 리테일(<$10k)과 같은 말이 아니다 -- 두 화면을 나란히
#   볼 때 이 차이를 잊으면 안 된다. 그래서 화면은 경계를 늘 숫자로 적는다.
#
# 경계를 **하드코딩**한 이유: 쌓인 행의 뜻이 영원히 고정돼야 하기 때문이다. 환경변수로 읽으면
# 누가 값을 바꾼 날 이후의 행이 조용히 다른 뜻이 된다(meta 표는 마지막 값만 남는다).
# 대신 아래 `warn_if_whale_threshold_diverged()` 가 «갈라진 순간»을 시끄럽게 만든다.
RETAIL_MAX_USD = 10_000.0     # 미만 = 리테일
WHALE_MIN_USD = 100_000.0     # 이상 = 고래 (그 사이는 중형)


class TakerOrderAggregator:
    """`@trade` 개별 체결을 **테이커 주문** 단위로 되묶는다(바이낸스 aggTrade 재구성).

    🔴왜 필요한가: 큰 주문 하나가 호가 30개를 쓸어담으면 `@trade` 는 그걸 **작은 체결 30건**
      으로 보고한다. 개별 체결로 「고래」를 세면 같은 시장에서 고래 물량이 통째로 사라진다 --
      2026-09-19 같은 11.3초 구간 실측: **aggTrade 기준 37.4% vs @trade 기준 9.6%**,
      그런데 총 명목은 $1,086,005 vs $1,085,736 으로 **같다**. 같은 체결을 다르게 셀 뿐이다.
    ⭐기존 `nif_whale`(microstructure_scanner)도 `@aggTrade` 를 쓴다. 거기 맞춰야 이 저장소
      에서 「고래」가 한 뜻이 된다. (@aggTrade 스트림 자체는 2026-09-02 부터 배달이 멈춰서
      재구성 말고는 길이 없다 -- 이 파일 맨 위 도크스트링 참고.)

    묶는 규칙: **같은 가격 · 같은 방향 · 연속된 체결ID · 직전 체결과 ORDER_GAP_MS 이내 ·
    같은 초**.
    🔴처음엔 «같은 밀리초»로 묶었다가 **고래 물량을 10% 놓쳤다**. 한 주문의 체결이 ms 를
      걸치면 거기서 잘렸기 때문이다. 2026-09-19 두 구간(각 78초) 실측, 바이낸스 aggTrades 를
      정답으로 둔 대조:
          규칙            주문수차      고래물량차   고래비중차
          ms 정확(구)     +29~34%      -9.8~-10.5%  -5.4~-8.3pp
          ms 20           +9.7%        +0.8~+1.3%   -1.1~+0.4pp
          연속ID 단독     -4~-6.8%     +3.7~+4.2%   +0.4~+2.1pp
          연속ID + ms100  -0.7~+0.5%   +1.0~+3.2%   +0.1~+0.5pp   ← 채택
    ⚠️완벽하지 않다. 남는 오차는 «과대» 쪽이다 -- 같은 값을 연달아 친 서로 다른 테이커가
      하나로 합쳐진다. 2026-09-19 분 단위 실측(3분 연속): **+0.00 / +0.02 / +0.00%**.
      같은 날 다른 표본(수집기 DB 5분)에서는 네 분이 -0.00~+0.60% 인데 **한 분만 +11.12%**
      였다. 그 한 분은 재현되지 않았다 -- 바이낸스 주문 목록에 이 규칙을 그대로 먹여도
      한 건도 합쳐지지 않았고(1263->1263), 오프라인 재현에서도 그 크기가 안 나온다.
      🔴**원인 미상으로 남긴다.** 분 단위로 이 값을 쓰는 연구는 그 꼬리를 감안해야 한다
      (총량은 kline 과 소수점까지 일치하므로 «어느 체결이 빠졌나»의 문제는 아니다).
    ⚠️문턱을 20ms 로 좁히면 더 나빠진다 -- 같은 실측에서 분 단위 -0.84 / **-7.96** / -0.12%.
      주문 수도 20ms 는 +9.7%, 100ms 는 ±1% 다.
    ⚠️총량·건수·최대체결은 **개별 체결 그대로** 센다(기존 컬럼 뜻을 바꾸지 않는다).
      되묶기는 **크기 구간 분류에만** 쓴다.
    🔴«같은 초»가 규칙에 있는 이유: 총량은 체결이 일어난 초에 쌓이는데 구간은 주문을 통째로
      한 초에 넣으므로, 주문이 초 경계를 걸치면 그 초에서 **구간 합 > 총량**이 된다(실제로
      251초 수집에서 12행이 이 상태였다). 초에서 자르면 정확히 사라진다.
      비용은 잰 뒤에 받아들였다 -- 2026-09-19 실측(90초·체결 10,010·주문 2,020):
      경계를 걸치는 주문 **0.50%**(물량 0.33%), **고래 물량 차이 0.000%**(고래 주문은 한
      건도 안 걸렸다), 주문 수 +0.50%. 쓸어담기는 대개 20ms 안에 끝나서 초를 잘 안 넘는다.

    마지막 묶음은 다음 체결이 와야 닫힌다. ETH 는 초당 100건 넘게 체결되므로 그 지연은
    밀리초 수준이고, 묶음은 자기 ts_ms 를 들고 있어 늦게 닫혀도 제 초에 들어간다.
    스트림이 끊기면 `take()` 로 직접 꺼낸다 -- 안 꺼내면 그 주문이 조용히 사라진다."""

    ORDER_GAP_MS = 100

    def __init__(self) -> None:
        self.key: tuple | None = None
        self.qty = 0.0
        self.last_ms = 0
        self.last_tid = -1

    def add(self, price: float, qty: float, ts_ms: int, sell: bool,
            tid: int | None = None) -> tuple | None:
        """묶음이 닫히면 (price, qty, ts_ms, sell) 을 돌려준다. 아니면 None.

        tid(체결ID)가 없으면 연속성 조건만 빠지고 시간·가격·방향 조건은 그대로다."""
        same = (self.key is not None
                and self.key == (price, sell)
                and ts_ms - self.last_ms <= self.ORDER_GAP_MS
                and ts_ms // 1000 == self.last_ms // 1000
                and (tid is None or self.last_tid < 0 or tid == self.last_tid + 1))
        done = None
        if not same:
            done = self.take()
            self.key = (price, sell)
            self.qty = 0.0
            self.first_ms = ts_ms
        self.qty += qty
        self.last_ms = ts_ms
        self.last_tid = -1 if tid is None else tid
        return done

    def take(self) -> tuple | None:
        if self.key is None or self.qty <= 0:
            self.key = None
            return None
        price, sell = self.key
        # 주문의 시각은 **첫 체결**로 잡는다 -- 마지막으로 잡으면 100ms 뒤의 초로 넘어가
        # 드물게 이웃 초에 실린다.
        out = (price, self.qty, self.first_ms, sell)
        self.key, self.qty, self.last_tid = None, 0.0, -1
        return out

    def reset(self) -> None:
        """스트림이 끊겼다 -- 열려 있던 묶음을 **버린다**. 끊김 뒤 첫 체결에 이어 붙이면
        그 사이가 통째로 빠진 주문을 «한 주문»이라 부르게 된다(그 구간은 gaps 에 적힌다)."""
        self.key, self.qty, self.last_tid = None, 0.0, -1


def warn_if_whale_threshold_diverged() -> None:
    """`MS_WHALE_USD_TH` 가 우리 경계와 달라지면 말한다.

    두 정의가 갈라지는 건 «언젠가» 가 아니라 «누가 환경변수를 바꾼 그 순간» 이다. 그때
    아무 말도 안 하면, 몇 주 뒤 두 화면이 다른 숫자를 보일 때 원인을 찾느라 하루를 쓴다."""
    raw = os.getenv("MS_WHALE_USD_TH")
    if raw is None:
        return
    try:
        other = float(raw)
    except ValueError:
        log(f"⚠️MS_WHALE_USD_TH={raw!r} 를 숫자로 못 읽는다 -- 경계 대조를 건너뛴다")
        return
    if other != WHALE_MIN_USD:
        log(f"🔴고래 경계가 갈라졌다: 이 표는 ${WHALE_MIN_USD:,.0f}, "
            f"microstructure(nif_whale)는 ${other:,.0f}. "
            "두 화면의 「고래」가 다른 것을 뜻하게 된다 -- 한쪽을 맞추거나 이름을 갈라야 한다.")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


class TapeBuffer:
    """(초, 가격빈) -> 12칸.

      0~1   매수량 · 매도량              (총량 -- 아래 세 구간의 합이다)
      2~3   매수건수 · 매도건수
      4~5   매수최대 · 매도최대          (그 초의 가장 큰 체결 하나)
      6~7   리테일 매수량 · 매도량       (< RETAIL_MAX_USD)
      8~9   고래 매수량 · 매도량         (>= WHALE_MIN_USD)
      10~11 고래 매수건수 · 매도건수     ← **주문** 건수다
      12~13 리테일 매수건수 · 매도건수   ← 같은 단위
      14~15 전체 매수주문수 · 매도주문수 ← 같은 단위. 이게 있어야 뺄셈이 닫힌다

    ⚠️**건수에 단위가 둘 있다.** 2~3 은 **개별 체결** 수(2026-09-16 이래 뜻 그대로)이고,
      10~15 는 **테이커 주문** 수다. 2~3 과 10~15 를 서로 빼면 안 된다.
    ⭐대신 주문 단위 «안에서는» 모든 뺄셈이 성립한다:
        중형 물량   = 총량     - 리테일물량 - 고래물량
        중형 주문수 = 전체주문 - 리테일주문 - 고래주문
      (2026-09-19 세 번에 걸쳐 닫았다: 처음엔 고래 건수만 뒀고 -- 그러면 리테일 건수를 구할
       길이 없었다. 리테일을 붙였더니 이번엔 중형을 구할 길이 없었다. 전체를 붙여야 닫힌다.)
    ⭐덤: «체결수 / 주문수» 가 그 초에 주문 하나가 평균 몇 호가를 쓸었는지를 말해 준다.
    DB 도 네트워크도 모른다 -- 그래서 --selftest 가 이 클래스만 찔러 볼 수 있다."""

    WIDTH = 16

    def __init__(self, bucket: float) -> None:
        self.bucket = bucket
        self.rows: dict[tuple[int, int], list[float]] = {}
        self.max_sec = 0
        self.closed_before = 0   # 이 초보다 앞은 이미 꺼내 갔다
        self.dropped_late = 0

    def _cell(self, ts_ms: int, price: float) -> list[float] | None:
        """🔴이미 DB 로 나간 초는 **되살리지 않는다**. 되묶기가 도입되면서 «늦게 닫히는 주문»이
        생겼는데, 스트림이 몇 초 멎은 뒤 그 주문이 닫히면 이미 기록된 초에 행이 새로 생겨
        같은 (ts_sec, price_bin) 이 **두 줄**이 된다(하나는 총량만, 하나는 구간만).
        그 경우는 버린다 -- 잃는 건 주문 하나이고, 얻는 건 표의 유일성이다."""
        sec = ts_ms // 1000
        if sec < self.closed_before:
            self.dropped_late += 1
            return None
        self.max_sec = max(self.max_sec, sec)
        key = (sec, round(price / self.bucket))
        cell = self.rows.get(key)
        if cell is None:
            cell = self.rows[key] = [0.0] * self.WIDTH
        return cell

    def add(self, ts_ms: int, price: float, qty: float, sell: bool) -> None:
        """개별 체결 하나. 총량 · 건수 · 최대체결만 센다(2026-09-16 이래 뜻이 그대로다)."""
        cell = self._cell(ts_ms, price)
        if cell is None:
            return
        i = 1 if sell else 0
        cell[i] += qty
        cell[2 + i] += 1
        cell[4 + i] = max(cell[4 + i], qty)

    def add_order(self, ts_ms: int, price: float, qty: float, sell: bool) -> None:
        """되묶은 **테이커 주문** 하나. 크기 구간은 여기서만 갈린다 -- 이유는
        TakerOrderAggregator 도크스트링."""
        cell = self._cell(ts_ms, price)
        if cell is None:
            return
        i = 1 if sell else 0
        cell[14 + i] += 1          # 구간과 무관하게 «주문 하나»
        notional = price * qty
        if notional < RETAIL_MAX_USD:
            cell[6 + i] += qty
            cell[12 + i] += 1
        elif notional >= WHALE_MIN_USD:
            cell[8 + i] += qty
            cell[10 + i] += 1

    def take_closed(self) -> list[tuple]:
        """진행 중인 초(max_sec)를 빼고 꺼낸다. 그 초는 아직 체결이 더 올 수 있다."""
        done = [(sec, b, c) for (sec, b), c in self.rows.items() if sec < self.max_sec]
        for sec, b, _ in done:
            del self.rows[(sec, b)]
        self.closed_before = max(self.closed_before, self.max_sec)
        return sorted(
            (sec, b, c[0], c[1], int(c[2]), int(c[3]), c[4], c[5],
             c[6], c[7], c[8], c[9], int(c[10]), int(c[11]),
             int(c[12]), int(c[13]), int(c[14]), int(c[15])) for sec, b, c in done)


class TapeStore:
    """duckdb 는 **프로세스 하나만** 파일을 연다. 그래서 연결을 붙들지 않고 **쓸 때만** 열었다
    닫는다 -- 붙들면 ops_watchdog 의 신선도 검사(읽기 전용 연결)가 매 사이클 BLOCKED 가 되어
    거짓 CRITICAL 을 쏜다(2026-09-16 서버에서 실제로 확인했다. maker_fill_shadow.duckdb 가
    같은 이유로 읽히지 않는다). 저장소의 다른 writer 들(microstructure_scanner)도 같은 규약이다.

    반대 방향도 막아야 한다: 검사기가 읽는 동안 우리 쓰기가 잠깐 막힐 수 있다. 그때 행을
    버리면 조용한 유실이므로, 못 쓴 행은 보류했다가 다음 주기에 다시 쓴다."""

    PENDING_CAP = 50_000   # ~3분치. 이보다 밀리면 락이 풀릴 가망이 없다고 보고 버리며 **말한다**

    def __init__(self, db_path: Path, symbol: str, bucket: float) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.symbol = symbol
        self.pending: list[tuple] = []
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS trade_tape_1s(
                  symbol VARCHAR, ts_sec BIGINT, price_bin INTEGER,
                  buy_qty DOUBLE, sell_qty DOUBLE, buy_n INTEGER, sell_n INTEGER,
                  buy_max DOUBLE, sell_max DOUBLE)""")
            # 크기 구간(2026-09-19 추가). 이미 있는 DB 에는 ALTER 로 붙인다 -- **기존 행은
            # NULL 로 남는다**. 0 이 아니라 NULL 인 것이 중요하다: 「그 구간엔 고래가 없었다」와
            # 「그때는 안 갈랐다」는 다른 말이고, 연구 쿼리가 그 둘을 구별할 수 있어야 한다.
            for col, typ in (("retail_buy_qty", "DOUBLE"), ("retail_sell_qty", "DOUBLE"),
                             ("whale_buy_qty", "DOUBLE"), ("whale_sell_qty", "DOUBLE"),
                             ("whale_buy_n", "INTEGER"), ("whale_sell_n", "INTEGER"),
                             ("retail_buy_n", "INTEGER"), ("retail_sell_n", "INTEGER"),
                             ("order_buy_n", "INTEGER"), ("order_sell_n", "INTEGER")):
                try:
                    con.execute(f"ALTER TABLE trade_tape_1s ADD COLUMN {col} {typ}")
                except Exception:  # noqa: BLE001 -- 이미 있으면 그게 정상이다
                    pass
            # 끊긴 구간. 연구 쿼리는 이 표를 봐야 «0» 과 «모름» 을 구분할 수 있다.
            con.execute("""
                CREATE TABLE IF NOT EXISTS gaps(
                  symbol VARCHAR, from_ms BIGINT, to_ms BIGINT, reason VARCHAR)""")
            # 분별 완전성. kline volume 과 맞는지 -- 체결 유실은 이 표에서만 드러난다.
            con.execute("""
                CREATE TABLE IF NOT EXISTS verify_1m(
                  symbol VARCHAR, ts_min BIGINT, tape_qty DOUBLE, kline_qty DOUBLE,
                  rel_err DOUBLE, checked_at TIMESTAMP)""")
            con.execute("CREATE TABLE IF NOT EXISTS meta(key VARCHAR, value VARCHAR)")
            # 경계는 **데이터와 함께** 남는다. 코드가 바뀌어도 이 표를 보면 그때 기준을 안다.
            for key, value in (("schema_version", str(SCHEMA_VERSION)),
                               (f"bucket:{symbol}", repr(bucket)),
                               ("retail_max_usd", repr(RETAIL_MAX_USD)),
                               ("whale_min_usd", repr(WHALE_MIN_USD))):
                con.execute("DELETE FROM meta WHERE key = ?", [key])
                con.execute("INSERT INTO meta VALUES (?, ?)", [key, value])

    def _connect(self):
        import duckdb

        return duckdb.connect(str(self.db_path))

    def last_ts_ms(self) -> int:
        with self._connect() as con:
            row = con.execute("SELECT max(ts_sec) FROM trade_tape_1s WHERE symbol = ?",
                              [self.symbol]).fetchone()
        return int(row[0] + 1) * 1000 if row and row[0] else 0

    def write(self, rows: list[tuple]) -> None:
        self.pending.extend(rows)
        if not self.pending:
            return
        try:
            with self._connect() as con:
                con.executemany(
                    "INSERT INTO trade_tape_1s VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                [(self.symbol, *r) for r in self.pending])
            self.pending.clear()
        except Exception as exc:  # noqa: BLE001 -- 락 충돌(읽는 쪽이 잡고 있음)이 대부분이다
            if len(self.pending) > self.PENDING_CAP:
                dropped = len(self.pending) - self.PENDING_CAP // 2
                self.pending = self.pending[dropped:]
                log(f"⚠️쓰기가 계속 막혀 {dropped}행 버림 -- 그 구간은 벌크 zip 으로 메워야 한다")
            else:
                log(f"쓰기 보류 {len(self.pending)}행, 다음 주기 재시도: {type(exc).__name__}")

    def record_gap(self, from_ms: int, to_ms: int, reason: str) -> None:
        if to_ms - from_ms < 1500:   # 재연결 한 번에 1초 미만이면 기록할 값이 없다
            return
        try:
            with self._connect() as con:
                con.execute("INSERT INTO gaps VALUES (?,?,?,?)",
                            [self.symbol, from_ms, to_ms, reason])
        except Exception as exc:  # noqa: BLE001
            log(f"gap 기록 실패(수집은 계속): {type(exc).__name__}")
            return
        log(f"gap {(to_ms - from_ms) / 1000:.1f}s 기록 ({reason})")

    def verify(self, kvol: dict[int, float], minutes: list[int]) -> list[tuple[int, float]]:
        """분별 테이프 합계를 kline 과 대조해 기록하고, (분, 상대오차) 목록을 돌려준다.
        연결 한 번 안에서 읽기·쓰기를 끝낸다 -- 검사마다 파일을 여러 번 여는 건 낭비다."""
        out: list[tuple[int, float]] = []
        with self._connect() as con:
            for ts_min in minutes:
                if ts_min not in kvol:
                    continue
                tape_qty = float(con.execute("""
                    SELECT coalesce(sum(buy_qty + sell_qty), 0) FROM trade_tape_1s
                    WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? + 60""",
                    [self.symbol, ts_min, ts_min]).fetchone()[0])
                kline_qty = kvol[ts_min]
                rel = (tape_qty - kline_qty) / kline_qty if kline_qty else 0.0
                con.execute("INSERT INTO verify_1m VALUES (?,?,?,?,?,now())",
                            [self.symbol, ts_min, tape_qty, kline_qty, rel])
                # 밀리초 환산을 SQL 안에서 하면 안 된다 -- duckdb 가 바인드 파라미터를 INT32 로
                # 보고 `1789485840 * 1000` 에서 오버플로를 낸다(2026-09-16 시험에서 터졌다).
                gapped = con.execute(
                    "SELECT count(*) FROM gaps WHERE symbol = ? AND from_ms < ? AND to_ms > ?",
                    [self.symbol, (ts_min + 60) * 1000, ts_min * 1000]).fetchone()[0]
                out.append((ts_min, rel, bool(gapped)))
        return out

    def unverified_minutes(self, limit: int = 5) -> list[int]:
        with self._connect() as con:
            rows = con.execute("""
                SELECT DISTINCT ts_sec // 60 * 60 AS m FROM trade_tape_1s
                WHERE symbol = ? AND ts_sec < ? - 60
                  AND NOT EXISTS (SELECT 1 FROM verify_1m v
                                  WHERE v.symbol = trade_tape_1s.symbol AND v.ts_min = m)
                ORDER BY m DESC LIMIT ?""", [self.symbol, int(time.time()), limit]).fetchall()
        return [int(r[0]) for r in rows]


async def verify_recent(store: TapeStore, session) -> None:
    """직전 분들을 kline volume 과 대조해 verify_1m 에 남긴다. 실패해도 수집은 계속한다."""
    minutes = store.unverified_minutes()
    if not minutes:
        return
    params = {"symbol": store.symbol.upper(), "interval": "1m",
              "startTime": min(minutes) * 1000, "limit": len(minutes) + 2}
    async with session.get(KLINES_URL, params=params) as response:
        if response.status != 200:
            return
        klines = await response.json()
    kvol = {int(k[0]) // 1000: float(k[5]) for k in klines}
    results = store.verify(kvol, minutes)
    ok = [r for r in results if abs(r[1]) <= VERIFY_TOLERANCE]
    # 성공도 한 줄 남긴다 -- 아무 말이 없으면 «검사가 도는지» 자체를 알 수 없다(그 상태로
    # 조용히 꺼져 있던 게 2026-09-16 의 INT32 오버플로였다).
    if ok:
        log(f"완전성 OK {len(ok)}분 (최근 {time.strftime('%H:%M', time.localtime(max(r[0] for r in ok)))})")
    for ts_min, rel, gapped in results:
        if abs(rel) <= VERIFY_TOLERANCE:
            continue
        stamp = time.strftime('%H:%M', time.localtime(ts_min))
        if gapped:
            log(f"완전성 {stamp} rel_err {rel:+.4%} -- 기록된 공백과 겹친다(예상된 부족)")
        else:
            log(f"⚠️완전성 {stamp} rel_err {rel:+.4%}"
                " -- 체결 유실. 그 구간은 벌크 zip 으로 덮어써야 한다")


async def collect(symbol: str, db_path: Path) -> None:
    from aiohttp import ClientSession, ClientTimeout, WSMsgType

    warn_if_whale_threshold_diverged()
    bucket = BUCKETS.get(symbol, 0.01)
    store = TapeStore(db_path, symbol, bucket)
    buffer = TapeBuffer(bucket)
    orders = TakerOrderAggregator()
    last_ms = store.last_ts_ms()      # 지난 판이 남긴 끝 -- 재시작 공백을 gaps 에 적으려고
    log(f"{symbol} 수집 시작 (빈 {bucket}, db {db_path})")
    # total=None 을 **명시**한다: aiohttp 기본 5분이라 그냥 두면 5분마다 끊긴다.
    async with ClientSession(timeout=ClientTimeout(total=None)) as session:
        flushed_at = verified_at = time.monotonic()
        while True:
            try:
                async with session.ws_connect(WS_URL.format(symbol=symbol), heartbeat=30) as ws:
                    first = True
                    async for msg in ws:
                        if msg.type is not WSMsgType.TEXT:
                            break
                        trade = json.loads(msg.data)
                        if trade.get("e") != "trade":
                            continue
                        price, qty = float(trade["p"]), float(trade["q"])
                        if not (price > 0 and qty > 0):
                            # 바이낸스가 {"p":"0","q":"0","X":"NA","st":1} 를 섞어 보낸다
                            # (2026-09-16 실측 12,312건 중 37건=0.3%). 합계는 안 틀리지만
                            # **가격 0 자리에 빈이 생긴다** -- 받는 쪽에서 막는다.
                            continue
                        ts_ms = int(trade["T"])
                        if first:
                            first = False
                            # 처음 켠 판이면 그 «분의 시작부터 여기까지»가 안 받은 구간이다.
                            # 적어두지 않으면 아래 완전성 검사가 그걸 유실로 오해한다.
                            orders.reset()   # 끊김 전에 열려 있던 묶음은 버린다
                            store.record_gap(last_ms or (ts_ms // 60_000) * 60_000, ts_ms,
                                             "ws_reconnect" if last_ms else "startup")
                            log("스트림 연결됨")
                        sell = bool(trade["m"])
                        buffer.add(ts_ms, price, qty, sell)
                        tid = trade.get("t")
                        order = orders.add(price, qty, ts_ms, sell,
                                           None if tid is None else int(tid))
                        if order is not None:
                            buffer.add_order(order[2], order[0], order[1], order[3])
                        last_ms = ts_ms
                        now = time.monotonic()
                        if now - flushed_at >= FLUSH_SECONDS:
                            flushed_at = now
                            store.write(buffer.take_closed())
                        if now - verified_at >= VERIFY_SECONDS:
                            verified_at = now
                            await verify_recent(store, session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- 한 번의 끊김이 수집기를 죽이면 그 뒤가
                # 통째로 빈다. 빈 구간은 소급 불가가 아니지만(zip) 알아채는 게 늦어진다.
                log(f"연결 실패, 3초 뒤 재시도: {type(exc).__name__} {exc}")
            store.write(buffer.take_closed())
            await asyncio.sleep(3.0)


def selftest() -> None:
    """네트워크·DB 없이 집계 규칙만 점검한다."""
    buf = TapeBuffer(0.1)
    buf.add(1_000_000, 2440.04, 1.5, sell=False)   # -> 빈 24400 (반올림)
    buf.add(1_000_400, 2440.02, 0.5, sell=True)    # 같은 초·같은 빈, 반대쪽
    buf.add(1_000_900, 2440.44, 3.0, sell=False)   # 같은 초, 다른 빈(24404)
    buf.add(1_001_000, 2440.04, 9.0, sell=False)   # 다음 초 -> 진행 중이라 안 나온다
    rows = buf.take_closed()
    assert [r[:2] for r in rows] == [(1000, 24400), (1000, 24404)], rows
    (sec, _bin, buy, sell, buy_n, sell_n, buy_max, sell_max,
     r_buy, r_sell, w_buy, w_sell, w_buy_n, w_sell_n, r_buy_n, r_sell_n,
     o_buy_n, o_sell_n) = rows[0]
    assert (buy, sell, buy_n, sell_n, buy_max, sell_max) == (1.5, 0.5, 1, 1, 1.5, 0.5), rows[0]
    # add() 는 총량/건수/최대만 센다 -- 크기 구간은 add_order() 몫이라 여기선 전부 0이다.
    assert (r_buy, r_sell, w_buy, w_sell) == (0.0, 0.0, 0.0, 0.0), rows[0]
    assert (w_buy_n, w_sell_n, r_buy_n, r_sell_n, o_buy_n, o_sell_n) == (0,) * 6, rows[0]

    # 크기 구간: 경계 **양쪽**을 찌른다. 부등호를 뒤집는 실수는 테스트가 아니면 안 보인다.
    sizes = TapeBuffer(0.1)
    sizes.add_order(2_000_000, 2500.0, 3.9, sell=False)    # $9,750  -> 리테일(< $10k)
    sizes.add_order(2_000_100, 2500.0, 4.0, sell=False)    # $10,000 -> 중형(경계는 리테일이 아니다)
    sizes.add_order(2_000_200, 2500.0, 39.9, sell=False)   # $99,750 -> 중형
    sizes.add_order(2_000_300, 2500.0, 40.0, sell=True)    # $100,000 -> 고래(경계 포함)
    for ts, q, s in ((2_000_000, 3.9, False), (2_000_100, 4.0, False),
                     (2_000_200, 39.9, False), (2_000_300, 40.0, True)):
        sizes.add(ts, 2500.0, q, sell=s)                   # 총량은 개별 체결에서 온다
    sizes.add(2_001_000, 2500.0, 1.0, sell=False)          # 다음 초 -- 위 초를 닫는다
    row = sizes.take_closed()[0]
    total_buy, total_sell = row[2], row[3]
    assert (total_buy, total_sell) == (47.8, 40.0), row
    assert (row[8], row[9]) == (3.9, 0.0), ("리테일 물량", row)
    assert (row[14], row[15]) == (1, 0), ("리테일 건수", row)
    assert (row[10], row[11], row[12], row[13]) == (0.0, 40.0, 0, 1), ("고래", row)
    assert (row[16], row[17]) == (3, 1), ("전체 주문수", row)
    # 중형은 칸이 없다 -- **물량도 건수도** 뺄셈으로 정확히 나와야 한다.
    assert round(total_buy - row[8] - row[10], 6) == 43.9, ("중형 매수 물량", row)
    assert round(total_sell - row[9] - row[11], 6) == 0.0, ("중형 매도 물량", row)
    assert row[16] - row[14] - row[12] == 2, ("중형 매수 주문수", row)
    assert row[17] - row[15] - row[13] == 0, ("중형 매도 주문수", row)

    # 되묶기: 같은 가격·방향 + 연속 체결ID + 100ms 이내가 한 주문이다.
    agg = TakerOrderAggregator()
    assert agg.add(2500.0, 10.0, 1_000, False, tid=1) is None       # 첫 묶음
    assert agg.add(2500.0, 12.0, 1_003, False, tid=2) is None       # ms 가 달라도 이어 붙는다
    done = agg.add(2500.5, 1.0, 1_004, False, tid=3)                # 가격이 달라지면 닫힌다
    assert done == (2500.0, 22.0, 1_000, False), done               # 시각은 **첫** 체결
    done = agg.add(2500.5, 1.0, 1_005, True, tid=4)                 # 방향이 달라져도 닫힌다
    assert done == (2500.5, 1.0, 1_004, False), done
    done = agg.add(2500.5, 1.0, 1_006, True, tid=99)                # 체결ID 가 끊기면 닫힌다
    assert done == (2500.5, 1.0, 1_005, True), done
    done = agg.add(2500.5, 1.0, 1_200, True, tid=100)               # 100ms 를 넘으면 닫힌다
    assert done == (2500.5, 1.0, 1_006, True), done
    done = agg.add(2500.5, 1.0, 1_999, True, tid=101)               # 같은 초, 100ms 초과 -> 닫힘
    assert done == (2500.5, 1.0, 1_200, True), done
    done = agg.add(2500.5, 1.0, 2_001, True, tid=102)               # 초가 넘어가면 닫힌다
    assert done == (2500.5, 1.0, 1_999, True), ("초 경계에서 안 잘렸다", done)
    assert agg.take() == (2500.5, 1.0, 2_001, True), "마지막 묶음은 take() 로 꺼낸다"
    assert agg.take() is None, "두 번 꺼내면 안 된다"
    agg.add(2500.0, 5.0, 2_000, False, tid=1)
    agg.reset()
    assert agg.take() is None, "reset 하면 열려 있던 묶음이 버려진다"

    # 닫힌 초는 되살아나지 않는다 -- 늦게 닫힌 주문이 중복 행을 만들면 안 된다.
    late = TapeBuffer(0.1)
    late.add(5_000_000, 2500.0, 1.0, sell=False)
    late.add(5_002_000, 2500.0, 1.0, sell=False)     # max_sec 를 5002 로
    assert len(late.take_closed()) == 1, "5000 초가 나갔다"
    late.add_order(5_000_000, 2500.0, 40.0, sell=False)   # 이미 나간 초 -> 버려야 한다
    assert late.take_closed() == [], "닫힌 초가 되살아났다"
    assert late.dropped_late == 1, late.dropped_late
    assert buf.rows and buf.max_sec == 1001, "진행 중인 초는 남아 있어야 한다"
    assert buf.take_closed() == [], "같은 초를 두 번 쓰면 안 된다"
    buf.add(1_002_000, 2440.04, 2.0, sell=True)    # 1001 초가 완결됨
    assert [r[:2] for r in buf.take_closed()] == [(1001, 24400)]
    print("selftest OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default=os.getenv("TAPE_SYMBOL", "ethusdt").lower())
    parser.add_argument("--db", type=Path, default=Path(os.getenv("TAPE_DB_PATH", DEFAULT_DB)))
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    try:
        asyncio.run(collect(args.symbol, args.db))
    except KeyboardInterrupt:
        log("종료")


if __name__ == "__main__":
    main()
