"""soxl_weekend_forward 정답지: 원본 live_soxl_weekend_forward_recorder_20261022.record() 를 가짜 urlopen·가짜 시계로 세 번.

  1회: 장부에 한 주말이 이미 있음(건너뜀) · 서머타임 끝나는 주말(10-31) · 5분봉이 멀리 떨어진 시각(lag) · 펀딩 0건(sum = 정수 0)
  2회: 한 주 뒤 새 주말 하나 · 3회: 그다음 주말 중간에 HTTP 500(원본은 예외로 끝 -- 그 앞 줄만 남는다)
출력 = tests/fixtures/soxl_weekend_{input,golden}.json. 바깥 접속 없음.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import types
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_soxl_weekend_forward_recorder_20261022 as S  # noqa: E402

FIX = ROOT / "rtcollect/tests/fixtures"


def h(*a) -> int:
    return int(hashlib.sha256(repr(a).encode()).hexdigest()[:8], 16)


def respond(url: str, fail_after: int | None, n: list) -> bytes:
    n[0] += 1
    if fail_after is not None and n[0] > fail_after:
        raise urllib.error.HTTPError(url, 500, "Internal Server Error", {}, None)
    q = dict(x.split("=") for x in url.split("?")[1].split("&"))
    if "/klines?" in url:
        end = int(q["endTime"])
        lag = 7 * 60_000 if h(q["symbol"], end) % 4 == 0 else 0
        o = end - lag
        c = f"{(40 if q['symbol'] == 'SOXLUSDT' else 3000) * (1 + (h(url) % 2000 - 1000) / 1e4):.4f}"
        return json.dumps([[o, "1", "2", "0.5", c, "10", o + 299_999, "1", 1, "1", "1", "0"]]).encode()
    st = int(q["startTime"])
    k = 0 if st == 1_793_419_200_001 else h(st) % 4 or 2   # 10-31 주말 = 펀딩 0건(원본 sum([]) = 정수 0)
    return json.dumps([{"symbol": "SOXLUSDT", "fundingTime": st + i * 28_800_000, "fundingRate": f"{(h(st, i) % 200 - 100) / 1e6:.8f}", "markPrice": "40"} for i in range(k)]).encode()


def main() -> None:
    reqs: list = []
    bodies: dict = {}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.jsonl"
        S.OUT = str(out)
        S.read_rows.__defaults__ = (str(out),)   # 원본 기본 인자는 정의 때 OUT 을 붙잡는다
        pre = json.dumps({"weekend": "2026-10-24", "x": 1}) + "\n\n"
        out.write_text(pre)
        runs, gold = [], []
        for t_now, fail in ((datetime(2026, 11, 10, 21, 30, tzinfo=timezone.utc), None),
                          (datetime(2026, 11, 17, 21, 30, tzinfo=timezone.utc), None),
                          (datetime(2026, 11, 24, 21, 30, tzinfo=timezone.utc), 4)):
            n = [0]
            reqs.clear()

            def urlopen(u, timeout=None, fail=fail, n=n):
                reqs.append([u, timeout])
                b = respond(u, fail, n)
                bodies[u] = b.decode()
                return io.BytesIO(b)

            class FakeDT(datetime):
                @classmethod
                def now(cls, tz=None, t=t_now):
                    return t
            S.urllib.request.urlopen, S.datetime = urlopen, FakeDT
            S.time = types.SimpleNamespace(time=lambda t=t_now: t.timestamp())
            before = out.read_text()
            try:
                S.record()
                err = None
            except urllib.error.HTTPError as e:
                err = repr(e)
            runs.append({"now_s": t_now.timestamp(), "fail_after": fail})
            gold.append({"added": out.read_text()[len(before):], "reqs": list(reqs), "error": err})
    (FIX / "soxl_weekend_input.json").write_text(json.dumps({"pre": pre, "runs": runs}))
    (FIX / "soxl_weekend_golden.json").write_text(json.dumps({"runs": gold, "bodies": sorted(bodies.items())}, ensure_ascii=False))
    print(f"정답지: 새 줄 {[g['added'].count(chr(10)) for g in gold]} · 요청 {[len(g['reqs']) for g in gold]} · 오류 {[g['error'] for g in gold]}")


if __name__ == "__main__":
    main()
