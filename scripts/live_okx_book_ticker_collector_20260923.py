#!/usr/bin/env python3
"""**OKX 최우선 호가(bbo-tbt) 수집기** — 바이낸스 bookTicker 의 OKX 짝. (2026-09-23)

왜: 이 저장소의 micro-price·OFI 는 전부 바이낸스 `@bookTicker` 하나에서 나온다. 그런데 OKX 는
ETH 테이커 플로우의 **34.8%**(2026-09-17~19 3일 |순델타| 점유)이고, 같은 판정에서 「수급 +인데
하락」의 부호를 뒤집는 기여가 OKX 하나로 **+23.5pp(1분) / +41.4pp(5분)** 였다. 체결만 받고
호가를 안 받으면 그 거래소의 큐 불균형(QI)·micro-price 를 영원히 못 만든다 -- 호가는 **소급
재구성이 불가능**하다(체결과 달리 일별 덤프가 없다). 체결 테이프와 같은 날 띄우는 이유다.

⭐**왜 `books`(400레벨)가 아니라 `bbo-tbt` 인가**: 바이낸스 수집기가 2026-09-14 에 겪은 것과
  같은 이유다 -- 문헌(Cont·Kukanov·Stoikov 2013 · Stoikov 2018)이 요구하는 재료는 **최우선
  호가의 가격과 수량 그 자체**이고, 깊은 북에서 복원하면 격자 잔차가 인공물을 만든다.
  `books` 는 2.3GB/일(실측)로 이 수집기의 **66배**인데, 흡수(호가벽) 축은 이미 갖고 있는
  바이낸스 `depthDiff` 로 먼저 증명할 일이라 지금은 안 받는다. 받게 되면 별도 수집기다.

🔴**`sz` 는 ETH 가 아니라 계약 수다**(ETH-USDT-SWAP `ctVal`=0.1). 2026-09-23 실측으로 확정했다 --
  ±0.1% 밴드 명목이 계약 그대로면 **$200.0M**, ctVal 을 곱하면 **$20.0M** 인데 같은 시각
  바이낸스가 **$19.8M** 이다. 안 곱하면 이 거래소가 10배 깊어 보인다.

⭐**파일 포맷은 바이낸스와 완전히 같다**(magic `BTKR` · 32B 헤더 · 32B 행). 수량을 ctVal 로
  환산해 **ETH 로 적으므로** 단위까지 같고, 기존 리더가 그대로 읽는다
  (`research_eth_microprice_second_horizon_20260914.py::read_bt` · `research_rt5_1s_panel_build_20260920.py`).
  `HourFile`/`_gzip_done` 은 복사하지 않고 **바이낸스 수집기에서 import 한다** -- 포맷이 갈라지면
  리더가 조용히 한쪽만 맞게 된다.
⚠️거래소 구분은 **디렉터리**다(`orderflow/okx_bookticker/<INST>/`). 헤더의 `flags` 에 표식을
  넣을 수도 있었지만 리더가 그 칸을 읽지 않아(둘 다 `_f` 로 버린다) 지금은 소비자가 없다.
  파일이 경로에서 떨어져 나갈 일이 생기면 그때 ver 를 올려 넣는 게 맞다.

⭐순서는 `seqId` 로 검증한다(바이낸스 `u` 와 같은 자리). 역행/중복은 버리고 카운터만 올린다.
⭐행이 **불규칙 시각**이다 -- 최우선이 바뀔 때만 온다. 초 격자로 맞추는 건 분석 쪽 일이다.

용량: 실측 51 msg/s = 141MB/일(원본). 시각이 바뀔 때 gzip 하므로 ~35MB/일. 바이낸스
(729 msg/s · 0.8GB/일)의 **1/23** 이다.

봇과 완전 분리: 자기 WS · 자기 디렉터리 · 주문 없음. 바이낸스 REST weight 를 쓰지 않는다.

사용:
  python scripts/live_okx_book_ticker_collector_20260923.py                   # ETH-USDT-SWAP
  OKX_BT_INST=BTC-USDT-SWAP python scripts/live_okx_book_ticker_collector_20260923.py
  python scripts/live_okx_book_ticker_collector_20260923.py --selftest        # 네트워크 없이
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 🔴복사가 아니라 import 다. 포맷(magic·헤더·행)과 시각 회전·압축 규약이 **같은 것이어야**
#   기존 리더가 두 거래소 파일을 구별 없이 읽는다.
_bn = importlib.import_module("scripts.live_book_ticker_collector_20260914")
HourFile = _bn.HourFile
MAGIC, VER, HDR, ROW = _bn.MAGIC, _bn.VER, _bn.HDR, _bn.ROW

# 체결 테이프 수집기와 **같은 표**를 본다 -- ctVal 이 두 곳에서 갈라지면 물량과 호가가
# 서로 다른 단위가 된다.
_tape = importlib.import_module("scripts.live_okx_trade_tape_collector_20260923")
CT_VALS = _tape.CT_VALS
assert_ct_val = _tape.assert_ct_val
HTTP_HEADERS = _tape.HTTP_HEADERS

INST = os.getenv("OKX_BT_INST", "ETH-USDT-SWAP").upper()
BT_ROOT = Path(os.getenv("OKX_BT_ROOT",
                         str(ROOT / "data" / "live" / "orderflow" / "okx_bookticker")))
WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
RECV_TIMEOUT = 25.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("okx_bookticker")


def parse_bbo(row: dict, ct_val: float) -> tuple[int, float, float, float, float, int] | None:
    """`bbo-tbt` 한 행 → (ts_ms, bid_px, bid_qty_eth, ask_px, ask_qty_eth, seq).

    🔴수량은 계약 수라 ctVal 을 곱한다. 한쪽이 비어 있거나 교차한 북은 버린다."""
    bids, asks = row.get("bids") or [], row.get("asks") or []
    if not bids or not asks:
        return None
    try:
        bid_px, bid_sz = float(bids[0][0]), float(bids[0][1])
        ask_px, ask_sz = float(asks[0][0]), float(asks[0][1])
        ts_ms = int(row["ts"])
        seq = int(row.get("seqId") or 0)
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not (bid_px > 0 and ask_px > bid_px and bid_sz > 0 and ask_sz > 0 and ts_ms > 0):
        return None            # 교차·0 수량은 스냅샷 경계의 쓰레기다 -- 적으면 IC 가 오염된다
    return ts_ms, bid_px, bid_sz * ct_val, ask_px, ask_sz * ct_val, seq


async def run(inst: str, root: Path) -> None:
    from aiohttp import ClientSession, ClientTimeout, WSMsgType

    ct_val = CT_VALS.get(inst)
    if ct_val is None:
        raise SystemExit(f"🔴{inst} 의 ctVal 을 모른다 -- CT_VALS 에 실측값을 적고 다시 돌릴 것")
    out = HourFile(root, inst)
    last_seq = -1
    n = dropped = 0
    t0 = time.time()
    log.info("수집 시작 %s → %s (ctVal %s)", inst, root, ct_val)
    async with ClientSession(timeout=ClientTimeout(total=None)) as session:
        await assert_ct_val(session, inst, ct_val)
        while True:
            try:
                async with session.ws_connect(WS_URL, heartbeat=20) as ws:
                    await ws.send_json({"op": "subscribe",
                                        "args": [{"channel": "bbo-tbt", "instId": inst}]})
                    log.info("연결 %s", WS_URL)
                    # 🔴OKX 는 점검 때 seqId 를 **리셋**한다(문서). 옛 last_seq 를 들고 가면 그 뒤
                    #   전부가 «역행»으로 버려지고, 로그도 n 이 안 늘어 조용하다. 연결마다 새로 잰다.
                    last_seq = -1
                    while True:
                        msg = await ws.receive(timeout=RECV_TIMEOUT)
                        if msg.type is not WSMsgType.TEXT:
                            break
                        if msg.data == "pong":
                            continue
                        try:
                            payload = json.loads(msg.data)
                        except ValueError:
                            continue
                        if payload.get("event") == "error":
                            log.error("구독 거부: %s (code %s)",
                                      payload.get("msg"), payload.get("code"))
                            break
                        if payload.get("event"):
                            continue
                        for row in payload.get("data") or []:
                            parsed = parse_bbo(row, ct_val)
                            if parsed is None:
                                dropped += 1
                                continue
                            ts, bid, bq, ask, aq, seq = parsed
                            if seq and seq <= last_seq:   # 역행/중복 -- 순서는 seqId 가 보장
                                dropped += 1
                                continue
                            last_seq = seq
                            out.write(ts, bid, bq, ask, aq, seq)
                            n += 1
                            if n % 20000 == 0:
                                el = time.time() - t0
                                log.info("%s 누적 %d행 · %.1f행/초 · 버림 %d · 스프레드 %.4f",
                                         inst, n, n / max(el, 1e-9), dropped, ask - bid)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                log.warning("25초 무음 -- 재연결")
                out.close()
                await asyncio.sleep(1)
            except Exception as exc:  # noqa: BLE001 -- 빈 구간은 영원히 못 채운다. 다시 붙는다.
                log.warning("WS 끊김 %s: %s — 3초 뒤 재연결", type(exc).__name__, exc)
                out.close()
                await asyncio.sleep(3)


def selftest() -> None:
    """네트워크 없이 파싱과 **바이트 왕복**을 점검한다.

    ⭐포맷은 읽는 쪽이 따로 있으므로 「쓴 바이트를 기존 리더 구조체로 되읽어」 확인한다 --
      그러지 않으면 단위나 자리 실수가 몇 주 뒤 연구 스크립트에서 터진다."""
    import gzip
    import struct
    import tempfile

    # ── 파싱: 계약→ETH · 교차/빈쪽 거르기 ─────────────────────────────────
    row = {"bids": [["2757.99", "3870.61", "0", "58"]],
           "asks": [["2758.00", "7603.87", "0", "59"]],
           "ts": "1790085497409", "seqId": 339515261591}
    p = parse_bbo(row, 0.1)
    assert p is not None
    ts, bid, bq, ask, aq, seq = p
    assert (ts, bid, ask, seq) == (1790085497409, 2757.99, 2758.00, 339515261591), p
    assert abs(bq - 387.061) < 1e-9 and abs(aq - 760.387) < 1e-9, ("계약 → ETH", bq, aq)
    assert parse_bbo({**row, "bids": []}, 0.1) is None, "한쪽이 비면 버린다"
    assert parse_bbo({**row, "asks": [["2757.98", "1", "0", "1"]]}, 0.1) is None, "교차는 버린다"
    assert parse_bbo({**row, "bids": [["2757.99", "0", "0", "0"]]}, 0.1) is None, "0 수량은 버린다"
    assert parse_bbo({"ts": "1", "seqId": 1}, 0.1) is None

    # ── 바이트 왕복: 기존 리더(read_bt)의 구조체로 되읽는다 ────────────────
    BT_HDR = struct.Struct("<4sHHqQII")          # research_eth_microprice_second_horizon_20260914
    BT_ROW = struct.Struct("<qdfdf")
    assert (BT_HDR.size, BT_ROW.size) == (HDR.size, ROW.size) == (32, 32)
    with tempfile.TemporaryDirectory() as td:
        hf = HourFile(Path(td), "ETH-USDT-SWAP")
        hf.write(ts, bid, bq, ask, aq, seq)
        hf.close()                                # 재연결 -- 같은 시각에 다시 써져야 한다
        hf.write(ts + 7, bid, bq, ask, aq, seq + 1)
        hf.close()                                # compress=False -- 바로 읽는다
        files = sorted(Path(td).glob("ETH-USDT-SWAP/*.bt"))
        assert len(files) == 1, files
        raw = files[0].read_bytes()
        magic, ver, rowsz, hour_ms, first_u, flags, _ = BT_HDR.unpack_from(raw, 0)
        assert magic == b"BTKR" and rowsz == 32, (magic, rowsz)
        assert ver == VER and first_u == seq and flags == 0
        assert hour_ms % 3_600_000 == 0 and hour_ms <= ts < hour_ms + 3_600_000, hour_ms
        assert (len(raw) - 32) % 32 == 0 and (len(raw) - 32) // 32 == 2, len(raw)
        r0 = BT_ROW.unpack_from(raw, 32)
        assert r0[0] == ts and r0[1] == bid and r0[3] == ask, r0
        # 🔴수량은 f32 다 -- 되읽은 값이 f32 정밀도 안에서 같아야 한다(단위 실수를 여기서 잡는다)
        assert abs(r0[2] - bq) < bq * 1e-6 and abs(r0[4] - aq) < aq * 1e-6, r0
        # 압축 경로도 리더가 읽는 모양인지(리더는 .bt.gz 를 그대로 받는다)
        gz = files[0].with_suffix(".bt.gz")
        gz.write_bytes(gzip.compress(raw))
        assert BT_HDR.unpack_from(gzip.decompress(gz.read_bytes()), 0)[0] == b"BTKR"
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inst", default=INST)
    ap.add_argument("--root", default=str(BT_ROOT))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    try:
        asyncio.run(run(a.inst.upper(), Path(a.root)))
    except KeyboardInterrupt:
        log.info("종료")


if __name__ == "__main__":
    main()
