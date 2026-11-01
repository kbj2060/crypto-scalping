#!/usr/bin/env python3
"""**CBOE 지연 옵션 체인 수집기** -- 미국 주식·ETF 옵션 체인 스냅샷을 원본 그대로 저장한다. (2026-11-01)

왜: Binance TradFi 선물(SNDK 등)의 옵션 맥락은 Binance·Deribit 에 없다. CBOE 사이트의 지연 시세 JSON 이 종목 한 번 호출로
  체인 전체(계약별 호가·IV·그릭스·OI·거래량·마지막 체결)를 준다(2026-11-01 SNDK 11,816계약·23만기·5.2MB 실측).
무엇이 안 되는가: 체결 한 건 한 건(블록 포함)은 없다 -- 계약별 «마지막 체결»과 하루 거래량 집계뿐이다. 체결 전체는 OPRA 유료 공급자가 필요하다.
🔴공식 개발자 API 가 아니라 사이트가 쓰는 주소다. 자동 수집 약관 허용 여부는 확인하지 못했다 -- 막히거나 바뀌면 이 수집기는 조용히 멈춘다(로그만 남는다).
요약: 저장한 직후 cboe_option_summary_20261101.write_state -> data/live/cboe_option_state.json (Option 카드).
저장: data/live/options_equity/cboe/<SYM>/<UTC날짜>/<HHMMSS>.json.gz (거래소 응답 그대로 · 해석하지 않는다 · SNDK 한 번 ~0.7MB).
장 휴장 판단: 응답의 `seqno` 가 직전과 같으면 저장하지 않는다 -- 주말·휴장·장외에 같은 마감 스냅샷이 쌓이지 않는다(달력 불필요).
주문 없음 · 바이낸스 호출 없음. cron 이 10분마다 부른다:
  python scripts/live_cboe_option_chain_collector_20261101.py [SNDK ...] | --selftest
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "live" / "options_equity" / "cboe"
URL = "https://cdn-api.cboe.com/api/global/delayed_quotes/options/{sym}.json"
UA = {"User-Agent": "Mozilla/5.0 (compatible; crypto-scalping-collector/1.0)"}


def fetch(sym: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(URL.format(sym=sym), headers=UA), timeout=30) as r:
        return r.read()


def collect(sym: str, out: Path = OUT, get=fetch, now: float | None = None) -> str:
    """저장했으면 경로, 직전과 같은 스냅샷이면 '' 를 돌려준다. 실패는 예외로 올린다(cron 로그에 남는다)."""
    if not re.fullmatch(r"[A-Z0-9.]{1,12}", sym):      # 신뢰경계: URL 과 파일 경로에 그대로 들어간다
        raise ValueError(f"종목 표기가 이상하다: {sym!r}")
    raw = get(sym)
    seq = json.loads(raw)["data"]["seqno"]
    d = out / sym
    state = d / "last_seqno"
    if state.exists() and state.read_text().strip() == str(seq):
        return ""
    t = time.gmtime(now if now is not None else time.time())
    path = d / time.strftime("%Y%m%d", t) / f"{time.strftime('%H%M%S', t)}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")     # 수동 실행이 cron 과 겹쳐도 임시 파일을 같이 쓰지 않는다
    tmp.write_bytes(gzip.compress(raw, 6))
    tmp.replace(path)
    state.write_text(str(seq))            # 파일을 먼저 두고 표식을 나중에 -- 중간에 죽으면 다음 판이 다시 저장한다
    return str(path)


def main(argv: list[str]) -> int:
    rc = 0
    for sym in [a.upper() for a in argv] or ["SNDK"]:
        try:
            got = collect(sym)
            if got:      # 저장한 스냅샷을 대시보드 Option 카드용 요약으로(원본 저장이 먼저 -- 요약이 실패해도 원본은 남는다)
                import cboe_option_summary_20261101 as summ
                summ.write_state(Path(got))
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {sym} {'저장 ' + got if got else '변화 없음(휴장·장외)'}", flush=True)
        except Exception as exc:  # noqa: BLE001 -- 한 종목 실패가 다른 종목을 막지 않는다
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {sym} 실패 {type(exc).__name__}: {exc}", flush=True)
            rc = 1
    return rc


def _selftest() -> None:
    body = lambda seq: json.dumps({"data": {"seqno": seq, "options": [{"option": "X"}]}}).encode()  # noqa: E731
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        p1 = collect("SNDK", out, lambda s: body(1), now=0)
        assert p1.endswith("19700101/000000.json.gz") and json.loads(gzip.decompress(Path(p1).read_bytes()))["data"]["seqno"] == 1
        assert collect("SNDK", out, lambda s: body(1), now=600) == "", "같은 seqno 는 저장하지 않는다"
        assert collect("SNDK", out, lambda s: body(2), now=600).endswith("001000.json.gz")
        assert collect("SNDK", out, lambda s: body(3), now=601).endswith("001001.json.gz"), "같은 분의 두 스냅샷이 서로 덮어쓰면 안 된다"
        try:
            collect("../x", out, lambda s: body(9), now=0)
            raise AssertionError("이상한 종목 표기는 거부해야 한다")
        except ValueError:
            pass
        try:
            collect("SNDK", out, lambda s: (_ for _ in ()).throw(OSError("down")), now=1200)
            raise AssertionError("실패는 예외로 올라와야 한다")
        except OSError:
            pass
        assert not list(out.rglob("*.part")), "임시 파일이 남으면 안 된다"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main(sys.argv[1:]))
